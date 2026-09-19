from __future__ import annotations

from typing import Any

from dar.diagnoser.base import DiagnoserCostRecord, DiagnosisInput, DiagnosisOutput
from dar.labels import Label
from dar.np_trigger import tool_call_succeeded

PERSISTENT_RUN_LENGTH = 2


def _is_error(content: Any) -> bool:
    return not tool_call_succeeded(content)


class RuleDiagnoser:
    name = "rule"

    def reset(self) -> None:
        pass

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        n_err = 0
        max_run: dict[str, int] = {}
        run: dict[str, int] = {}
        for m in obs.trace_prefix:
            if m.get("role") != "tool":
                continue
            tool = m.get("name") or "?"
            if _is_error(m.get("content")):
                n_err += 1
                run[tool] = run.get(tool, 0) + 1
                max_run[tool] = max(max_run.get(tool, 0), run[tool])
            else:
                run[tool] = 0

        if max_run and max(max_run.values()) >= PERSISTENT_RUN_LENGTH:
            label = Label.PERSISTENT
        elif n_err > 0:
            label = Label.TRANSIENT
        else:
            label = Label.NP

        return DiagnosisOutput(
            label=label,
            cost=DiagnoserCostRecord(calls=0, basis="rule=0( python )"),
            detail={"source": "rule", "n_error_results": n_err,
                    "max_consecutive_same_tool_errors": max(max_run.values()) if max_run else 0},
        )
