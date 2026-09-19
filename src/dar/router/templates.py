from __future__ import annotations

import hashlib
import json
from string import Formatter
from typing import Any

from dar.router.actions import RecoveryAction, RecoveryEvidence

ORACLE_TOKENS: frozenset[str] = frozenset(
    {"perturbation_status", "perturbation_mode", "is_perturbed"}
)

_FRAMING: dict[RecoveryAction, str] = {
    RecoveryAction.NO_OP: "",
    RecoveryAction.RETRY_SAME_TOOL: (
        "Diagnosis: the last call hit a transient failure — the same call may succeed if repeated.\n"
        "Tool called: {tool}\n"
        "Arguments sent: {args}\n"
        "Error reported: {error}"
    ),
    RecoveryAction.SWITCH_PATH: (
        "Diagnosis: the last call hit a persistent failure — repeating it will fail the same way.\n"
        "Tool called: {tool}\n"
        "Arguments sent: {args}\n"
        "Error reported: {error}\n"
        "Other tools available in this environment: {available_tools}"
    ),
    RecoveryAction.GRACEFUL_ABORT: (
        "Diagnosis: the last call hit a persistent failure and there is no alternative route for this subgoal.\n"
        "Tool called: {tool}\n"
        "Arguments sent: {args}\n"
        "Error reported: {error}"
    ),
}

_INSTRUCTION: dict[RecoveryAction, str] = {
    RecoveryAction.NO_OP: "",
    RecoveryAction.RETRY_SAME_TOOL: (
        "Recovery: retry the SAME call unchanged. Do not rewrite the arguments and do not switch tools."
    ),
    RecoveryAction.SWITCH_PATH: (
        "Recovery: this subgoal can be reached another way. Proceed via an alternative that does not "
        "depend on the failed tool, using the tools listed above."
    ),
    RecoveryAction.GRACEFUL_ABORT: (
        "Recovery: this step cannot be completed. State that it could not be done and finish, rather than "
        "issuing further calls that will fail the same way."
    ),
}

_FACT_LINE: dict[str, str] = {
    "available_tools": "Other tools available in this environment: {available_tools}",
}

_INSTRUCTION_REQUIRES: dict[RecoveryAction, tuple[str, ...]] = {
    RecoveryAction.NO_OP: (),
    RecoveryAction.RETRY_SAME_TOOL: (),
    RecoveryAction.SWITCH_PATH: ("available_tools",),
    RecoveryAction.GRACEFUL_ABORT: (),
}


def compose_template(framing: RecoveryAction, instruction: RecoveryAction) -> str:
    head, tail = _FRAMING[framing], _INSTRUCTION[instruction]
    if not head and not tail:
        return ""
    if not head or not tail:
        raise ValueError(
            "")
    lines = [head]
    for field in _INSTRUCTION_REQUIRES[instruction]:
        if "{" + field + "}" not in head:
            lines.append(_FACT_LINE[field])
    lines.append(tail)
    return "\n".join(lines)


_TEMPLATES: dict[RecoveryAction, str] = {
    a: compose_template(a, a) for a in RecoveryAction
}

_GENERIC_FRAMING: dict[RecoveryAction, str] = {
    RecoveryAction.NO_OP: "",
    RecoveryAction.RETRY_SAME_TOOL: "The previous call hit a transient failure",
    RecoveryAction.SWITCH_PATH: "The previous call hit a persistent failure",
    RecoveryAction.GRACEFUL_ABORT: "The previous call hit a persistent failure with no alternative",
}

_GENERIC_INSTRUCTION: dict[RecoveryAction, str] = {
    RecoveryAction.NO_OP: "",
    RecoveryAction.RETRY_SAME_TOOL: "retry the same call.",
    RecoveryAction.SWITCH_PATH: "reach this subgoal another way.",
    RecoveryAction.GRACEFUL_ABORT: "report it and finish.",
}


def compose_generic(framing: RecoveryAction, instruction: RecoveryAction) -> str:
    head, tail = _GENERIC_FRAMING[framing], _GENERIC_INSTRUCTION[instruction]
    if not head and not tail:
        return ""
    if not head or not tail:
        raise ValueError(
            "")
    return f"{head}; {tail}"


_GENERIC_CONTEXT: dict[RecoveryAction, str] = {
    a: compose_generic(a, a) for a in RecoveryAction
}

_DEFAULT_PLACEHOLDERS: dict[str, str] = {
    "tool": "no tool call is on record for this step",
    "args": "no arguments are on record for this step",
    "error": "no error was reported for this call",
    "available_tools": "the tool list is not available here; use the schemas in your observation",
}


def _template_fields(template: str) -> list[str]:
    seen: list[str] = []
    for _, name, _, _ in Formatter().parse(template):
        if name and name not in seen:
            seen.append(name)
    return seen


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def extract_facts(evidence: RecoveryEvidence) -> dict[str, str | None]:
    tool: str | None = None
    args: str | None = None
    if evidence.last_action is not None:
        missing = {"tool_name", "args"} - set(evidence.last_action)
        if missing:
            raise ValueError(
                ""
            )
        tool = str(evidence.last_action["tool_name"])
        args = _dumps(evidence.last_action["args"])

    available: str | None = None
    if evidence.available_tools:
        victim = str(evidence.last_action.get("tool_name")) if evidence.last_action else None
        names = [
            str(s["name"])
            for s in evidence.available_tools
            if s.get("name") and str(s["name"]) != victim
        ]
        if names:
            available = ", ".join(names)

    return {
        "tool": tool,
        "args": args,
        "error": evidence.last_error,
        "available_tools": available,
    }


def render_split_context(
    framing: RecoveryAction, instruction: RecoveryAction, evidence: RecoveryEvidence
) -> tuple[str, dict[str, str | None]]:
    template = compose_template(framing, instruction)
    if not template:
        return "", {}
    facts = extract_facts(evidence)
    rendered = {name: facts[name] for name in _template_fields(template)}
    filled = {
        name: (value if value is not None else _DEFAULT_PLACEHOLDERS[name])
        for name, value in rendered.items()
    }
    return template.format(**filled), rendered


def render_context(
    action: RecoveryAction, evidence: RecoveryEvidence
) -> tuple[str, dict[str, str | None]]:
    return render_split_context(action, action, evidence)


def generic_split_context(framing: RecoveryAction, instruction: RecoveryAction) -> str:
    return compose_generic(framing, instruction)


def generic_context(action: RecoveryAction) -> str:
    return _GENERIC_CONTEXT[action]


def template_fingerprint() -> str:
    payload = _dumps(
        {
            "templates": {k.value: v for k, v in _TEMPLATES.items()},
            "generic": {k.value: v for k, v in _GENERIC_CONTEXT.items()},
            "placeholders": _DEFAULT_PLACEHOLDERS,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
