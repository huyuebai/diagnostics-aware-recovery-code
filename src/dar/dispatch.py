from __future__ import annotations

import dataclasses
from typing import Any

from dar.vendorpath import add_vendor_to_syspath

add_vendor_to_syspath()
from evaluation.core.sandbox import ExecutionEngine

from dar.diagnoser.base import DiagnosisInput
from dar.diagnoser.fixed import FixedLabelDiagnoser
from dar.diagnoser.oracle import OracleDiagnoser
from dar.labels import Label, correct_label, true_type_from_mode
from dar.np_trigger import np_trigger, tool_call_succeeded
from dar.router import RecoveryAction, TemplateRouter, template_fingerprint
from dar.router.evidence import evidence_from_round


def _assert_trace_independent(diagnoser: Any) -> None:
    if getattr(diagnoser, "TRACE_INDEPENDENT", False) is not True:
        raise RuntimeError(
            "")


class DispatchConfigError(RuntimeError):
    pass


class DispatchEngine(ExecutionEngine):
    _tier: str = "none"
    _router: Any = None
    np_trigger_provenance: Any = None

    @classmethod
    def configure(cls, *, tier: str = "none", router: Any = None) -> None:
        cls._tier = tier
        cls._router = router

    @classmethod
    def reset_config(cls) -> None:
        cls._tier = "none"
        cls._router = None

    def _router_or_default(self) -> Any:
        return self._router if self._router is not None else TemplateRouter()

    def _make_diagnoser(self) -> Any:
        tier = self._tier
        if tier == "oracle":
            return OracleDiagnoser(true_type_from_mode(self.mode))
        if tier.startswith("fixed:"):
            value = tier.split(":", 1)[1]
            try:
                return FixedLabelDiagnoser(Label(value))
            except ValueError as e:
                raise DispatchConfigError("") from e
        raise DispatchConfigError(
            ""
        )

    def _setup_np_trigger(self, diagnoser: Any) -> frozenset[str]:
        self.np_trigger_provenance = None
        if self.mode != "P0":
            return frozenset()
        _assert_trace_independent(diagnoser)
        bootstrap = DiagnosisInput.from_messages(
            self.logger.messages, task_description=self.task_description, trigger_round=0
        )
        label = diagnoser.diagnose(bootstrap).label
        spec = np_trigger(self.complexity.lower(), self.task_id, label)
        if spec is None:
            return frozenset()
        self.np_trigger_provenance = spec
        return frozenset(spec["victim_tools"])

    def run(self, max_rounds: int = 15):
        setter = getattr(self.agent, "set_crn_task_id", None)
        if setter is not None:
            setter(self.task_id)
        if self._tier == "none":
            return super().run(max_rounds)
        return self._run_with_dispatch(max_rounds)

    def _run_with_dispatch(self, max_rounds: int = 15):
        self._dispatched = False
        diagnoser = self._make_diagnoser()
        router = self._router_or_default()

        if hasattr(self.agent, "perturbation_mode"):
            self.agent.perturbation_mode = self.mode
        self.agent.initialize(self.task_description, self.tool_definitions)

        current_message = self.task_description
        round_num = 0
        self.logger.log_user_message(self.task_description)
        np_victims = self._setup_np_trigger(diagnoser)

        while round_num < max_rounds:
            round_num += 1
            action = self.agent.step(user_message=current_message)
            current_message = None
            action_dict = {
                "type": action.type,
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "content": action.content,
                "thought": action.thought,
            }

            if action.type == "final_answer":
                self.logger.log_round(round_num=round_num, agent_action=action_dict,
                                      tool_result=None, perturbation_status="n/a")
                break

            if action.type == "tool_call":
                perturbed_call: tuple[str, Any, Any] | None = None
                np_touch: tuple[str, Any, Any] | None = None
                if action.tool_calls and len(action.tool_calls) > 1:
                    for idx, tc in enumerate(action.tool_calls):
                        tool_result, status = self._intercept_tool_call(tc.tool_name, tc.arguments)
                        tc_action_dict = {
                            "type": action.type,
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "content": action.content if idx == 0 else None,
                            "thought": action.thought if idx == 0 else None,
                        }
                        self.logger.log_round(round_num=round_num, agent_action=tc_action_dict,
                                              tool_result=tool_result, perturbation_status=status)
                        self.agent.receive_tool_result(tc.tool_name, tool_result, tool_call_index=idx)
                        if status == "perturbed" and perturbed_call is None:
                            perturbed_call = (tc.tool_name, tc.arguments, tool_result)
                        elif (status == "clean" and np_touch is None
                              and tc.tool_name in np_victims
                              and tool_call_succeeded(tool_result)):
                            np_touch = (tc.tool_name, tc.arguments, tool_result)
                else:
                    tool_result, status = self._intercept_tool_call(action.tool_name, action.arguments)
                    self.logger.log_round(round_num=round_num, agent_action=action_dict,
                                          tool_result=tool_result, perturbation_status=status)
                    self.agent.receive_tool_result(action.tool_name, tool_result)
                    if status == "perturbed":
                        perturbed_call = (action.tool_name, action.arguments, tool_result)
                    elif (status == "clean" and action.tool_name in np_victims
                          and tool_call_succeeded(tool_result)):
                        np_touch = (action.tool_name, action.arguments, tool_result)

                if not self._dispatched:
                    if perturbed_call is not None:
                        injected = self._do_dispatch(round_num, perturbed_call, diagnoser, router)
                    elif np_touch is not None:
                        injected = self._do_dispatch(round_num, np_touch, diagnoser, router,
                                                     np_provenance=self.np_trigger_provenance)
                    else:
                        injected = None
                    if injected is not None:
                        current_message = injected

        token_usage = self.agent.get_token_usage().to_dict()
        return self.logger, token_usage

    def _do_dispatch(self, round_num: int, trigger_call: tuple, diagnoser: Any, router: Any,
                     np_provenance: dict[str, Any] | None = None) -> str | None:
        self._dispatched = True
        tool_name, args, result = trigger_call

        obs = DiagnosisInput.from_messages(
            self.logger.messages, task_description=self.task_description, trigger_round=round_num
        )
        diag = diagnoser.diagnose(obs)
        label = diag.label

        ev = evidence_from_round(
            tool_name=tool_name,
            args=args or {},
            tool_result=result,
            task=self.task_json,
            step_index=round_num,
            available_tools=[
                {"name": d["name"]}
                for d in self.tool_definitions
                if isinstance(d, dict) and d.get("name")
            ],
            oracle_label=correct_label(true_type_from_mode(self.mode)),
        )
        decision = router.route(label, ev)

        meta: dict[str, Any] = {
            "tier": self._tier,
            "diagnosed_label": label.value,
            "action": decision.action.value,
            "trigger_round": round_num,
            "template_fingerprint": template_fingerprint(),
            "fields_rendered": decision.fields_rendered,
            "diagnoser_cost": _cost_dict(diag.cost),
        }
        if np_provenance is not None:
            meta["np_pseudo_trigger"] = True
            meta["np_trigger"] = np_provenance

        if decision.action is RecoveryAction.NO_OP or not decision.recovery_context:
            meta["injected"] = False
            self._log_dispatch("", meta)
            return None

        meta["injected"] = True
        meta["injected_message_index"] = len(self.logger.messages)
        self._log_dispatch(decision.recovery_context, meta)
        return decision.recovery_context

    def _log_dispatch(self, recovery_context: str, meta: dict[str, Any]) -> None:
        self.logger.messages.append(
            {"role": "user", "content": recovery_context, "metadata": {"dar_dispatch": meta}}
        )


def _cost_dict(cost: Any) -> dict[str, Any] | None:
    if dataclasses.is_dataclass(cost):
        return dataclasses.asdict(cost)
    return None


def wire_dispatch_engine(run_eval_module: Any, tier: str, router: Any = None) -> None:
    DispatchEngine.configure(tier=tier, router=router)
    run_eval_module.ExecutionEngine = DispatchEngine


def configure_agent_from_operation_point(agent_cls: Any, op_config: dict[str, Any], *,
                                         force_prompt: str | None = "fault_aware") -> None:
    agent = op_config.get("agent") or {}
    wb = op_config.get("writeback") or {}
    agent_cls.configure(
        strip_history_thinking=agent.get("strip_history_thinking"),
        writeback_reasoning=wb.get("reasoning_content_to_trace"),
        force_prompt=force_prompt,
        sampling=agent.get("sampling"),
        seed_mode=agent.get("seed_mode"),
    )
