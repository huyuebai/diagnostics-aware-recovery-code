from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from dar.ga import _match_path, _successful_calls
from dar.taskmodel import is_action, iter_paths

VERDICT_VOCAB = frozenset({"EQUIVALENT", "NOT_EQUIVALENT", "CANNOT_DETERMINE"})

JUDGE_PROMPT_TEMPLATE = """You are grading whether a value recorded by an automated \
workflow is an acceptable realization of a reference value, for the purpose of \
fulfilling a user's request.

USER REQUEST:
{user_query}

TOOL: {tool_name}
TOOL DESCRIPTION: {tool_description}
PARAMETER: {arg_name}

REFERENCE VALUE (what a correct execution would record):
{reference_value}

OBSERVED VALUE (what this execution recorded):
{observed_value}

Question: for the purpose of fulfilling the user's request, does the OBSERVED VALUE \
convey the same information and produce the same outcome as the REFERENCE VALUE?

Rules:
- Formatting differences are EQUIVALENT: casing, units spelled out vs abbreviated, \
date/time format variants, reordering of list items with identical content, \
extra whitespace or punctuation.
- Paraphrase is EQUIVALENT only if every substantive element of the REFERENCE VALUE \
is preserved: same entities, same quantities, same requested content.
- Missing, contradictory, or unrelated substantive content is NOT_EQUIVALENT. \
An OBSERVED VALUE of "(absent)" is NOT_EQUIVALENT unless the REFERENCE VALUE is \
itself empty.
- If you cannot decide from the information given, answer CANNOT_DETERMINE. \
Do not guess.

Answer with exactly one token: EQUIVALENT or NOT_EQUIVALENT or CANNOT_DETERMINE."""


SUBSTITUTION_PROMPT_TEMPLATE = """You are grading whether an automated workflow \
accomplished a required step of a user's request through alternative means.

USER REQUEST:
{user_query}

REQUIRED STEP (what a reference execution would perform):
tool: {ref_tool}
arguments: {ref_args}

ACTIONS THIS EXECUTION ACTUALLY PERFORMED (all of its state-changing calls):
{observed_actions}

Question: do the performed actions, taken together, accomplish the same outcome as \
the REQUIRED STEP for the purpose of fulfilling the user's request?

Rules:
- A different tool that produces the same outcome with the same substantive inputs \
is EQUIVALENT (e.g., an alternative provider of the same service).
- If no performed action accomplishes what the REQUIRED STEP accomplishes — or the \
substantive inputs differ (different item, different quantity, different recipient) \
— answer NOT_EQUIVALENT.
- If you cannot decide from the information given, answer CANNOT_DETERMINE. \
Do not guess.

Answer with exactly one token: EQUIVALENT or NOT_EQUIVALENT or CANNOT_DETERMINE."""


def judge_prompt_fingerprint() -> str:
    return hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def substitution_prompt_fingerprint() -> str:
    return hashlib.sha256(SUBSTITUTION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SemanticItem:
    path_id: str
    idx: int
    step: int
    tool: str
    arg: str
    kind: str
    reference: Any
    observed: Any

    def render_prompt(self, user_query: str, tool_description: str = "") -> str:
        def _fmt(v: Any) -> str:
            if v is None:
                return "(absent)"
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if self.kind == "action_substitution":
            ref = self.reference or {}
            return SUBSTITUTION_PROMPT_TEMPLATE.format(
                user_query=user_query,
                ref_tool=ref.get("tool", self.tool),
                ref_args=json.dumps(ref.get("arguments") or {}, ensure_ascii=False),
                observed_actions=json.dumps(self.observed or [], ensure_ascii=False),
            )
        return JUDGE_PROMPT_TEMPLATE.format(
            user_query=user_query,
            tool_name=self.tool,
            tool_description=tool_description or "(none)",
            arg_name=self.arg,
            reference_value=_fmt(self.reference),
            observed_value=_fmt(self.observed),
        )


def semantic_items(messages: list[dict[str, Any]],
                   task_p0: dict[str, Any]) -> dict[str, list[SemanticItem]]:
    calls = _successful_calls(messages)
    run_actions = [{"tool": c["tool"], "arguments": c["args"]}
                   for c in calls if is_action(c["tool"])]
    called_tools = {c["tool"] for c in calls}
    out: dict[str, list[SemanticItem]] = {}
    for pid, steps in iter_paths(task_p0):
        pv = _match_path(pid, steps, calls)
        if pv.n_actions == 0:
            continue
        if pv.missing_actions and not run_actions:
            continue
        items: list[SemanticItem] = []
        for i, s in enumerate(steps):
            tool = s.get("tool_name")
            if is_action(tool) and tool not in called_tools:
                items.append(SemanticItem(pid, len(items), i, tool, "(action)",
                                          "action_substitution",
                                          {"tool": tool,
                                           "arguments": s.get("arguments") or {}},
                                          run_actions))
        for rec in pv.scalar_mismatches:
            items.append(SemanticItem(pid, len(items), rec["step"], rec["tool"],
                                      rec["arg"], "scalar_variant",
                                      rec["gt"], rec.get("run")))
        for rec in pv.free_text_divergent:
            items.append(SemanticItem(pid, len(items), rec["step"], rec["tool"],
                                      rec["arg"], "free_text",
                                      rec["gt"], rec.get("run")))
        out[pid] = items
    return out


def apply_semantic_verdicts(items_by_path: dict[str, list[SemanticItem]],
                            verdicts: dict[tuple[str, int], str]) -> dict[str, Any]:
    path_results: dict[str, str] = {}
    for pid, items in items_by_path.items():
        if not items:
            path_results[pid] = "undecided"
            continue
        got = []
        for it in items:
            v = verdicts.get((it.path_id, it.idx))
            if v is None:
                raise ValueError("")
            if v not in VERDICT_VOCAB:
                raise ValueError("")
            got.append(v)
        if all(v == "EQUIVALENT" for v in got):
            path_results[pid] = "pass"
        elif any(v == "NOT_EQUIVALENT" for v in got):
            path_results[pid] = "fail"
        else:
            path_results[pid] = "undecided"

    if any(r == "pass" for r in path_results.values()):
        verdict = "achieved"
    elif path_results and all(r == "fail" for r in path_results.values()):
        verdict = "not_achieved"
    else:
        verdict = "undetermined"
    return {"verdict": verdict, "layer": "B", "path_results": path_results}


def ga_layered_verdict(messages: list[dict[str, Any]], task_p0: dict[str, Any],
                       judge_fn: Callable[[SemanticItem], str] | None = None) -> dict[str, Any]:
    from dar.ga import ga_verdict
    a = ga_verdict(messages, task_p0)
    if (a["verdict"] != "undetermined"
            or a.get("reason_code") not in ("arg_divergence",
                                            "action_substitution_candidate")):
        return {**a, "layer": "A"}
    if judge_fn is None:
        return {**a, "layer": "A"}
    items_by_path = semantic_items(messages, task_p0)
    verdicts = {(it.path_id, it.idx): judge_fn(it)
                for its in items_by_path.values() for it in its}
    b = apply_semantic_verdicts(items_by_path, verdicts)
    return {**b,
            "reason_code": "semantic",
            "lexical": {k: a[k] for k in ("closest_path", "scalar_mismatches",
                                          "free_text_divergent", "missing_actions",
                                          "run_actions") if k in a},
            "n_items": sum(len(v) for v in items_by_path.values()),
            "judge_prompt_fingerprint": judge_prompt_fingerprint(),
            "substitution_prompt_fingerprint": substitution_prompt_fingerprint()}


def ga_ordered_verdict(messages: list[dict[str, Any]], task_p0: dict[str, Any],
                       task_mode: dict[str, Any],
                       judge_fn: Callable[[SemanticItem], str] | None = None) -> dict[str, Any]:
    from dar.ga import corrupt_distinctive_tokens, corrupt_echo_in_actions
    toks = corrupt_distinctive_tokens(task_mode, task_p0)
    if corrupt_echo_in_actions(messages, toks):
        return {"verdict": "not_achieved", "reason": "corrupt_echo",
                "reason_code": "corrupt_echo", "layer": "echo"}
    return ga_layered_verdict(messages, task_p0, judge_fn)


def main_estimator(verdict: str) -> bool:
    if verdict not in {"achieved", "not_achieved", "undetermined"}:
        raise ValueError("")
    return verdict == "achieved"
