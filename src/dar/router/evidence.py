from __future__ import annotations

from typing import Any

from dar.labels import Label
from dar.router.actions import RecoveryEvidence
from dar.taskmodel import has_alternative_path


def _extract_error(tool_result: Any) -> str | None:
    if not isinstance(tool_result, dict):
        return None
    err = tool_result.get("error")
    if not err:
        return None
    if isinstance(err, dict):
        code = err.get("code")
        msg = err.get("message") or err.get("msg")
        if code is not None and msg is not None:
            return f"{code}: {msg}"
        s = str(msg if msg is not None else err).strip()
        return s or None
    s = str(err).strip()
    return s or None


def evidence_from_round(
    *,
    tool_name: str,
    args: dict[str, Any],
    tool_result: Any,
    task: dict[str, Any],
    step_index: int,
    available_tools: list[dict[str, Any]] | None = None,
    oracle_label: Label | None = None,
) -> RecoveryEvidence:
    return RecoveryEvidence(
        last_action={"tool_name": tool_name, "args": dict(args)},
        last_error=_extract_error(tool_result),
        step_index=step_index,
        has_alternative_path=has_alternative_path(task),
        available_tools=available_tools,
        oracle_label=oracle_label,
    )
