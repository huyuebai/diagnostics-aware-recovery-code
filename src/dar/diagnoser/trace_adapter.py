from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from dar.accounting import DISPATCH_META_KEY
from dar.diagnoser.base import GROUND_TRUTH_KEYS, DiagnosisInput
from dar.np_trigger import tool_call_succeeded

INFERENCE_TOP_KEYS = frozenset({"task_id", "mode", "tokens", "messages"})

FORBIDDEN_TOP_KEYS = frozenset({"mode", "tokens"})


def load_inference(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("")
    got = set(obj)
    if got != INFERENCE_TOP_KEYS:
        raise ValueError(
            "")
    if not isinstance(obj["messages"], list):
        raise ValueError("")
    return obj


def _is_tool_msg(m: dict[str, Any]) -> bool:
    return m.get("role") == "tool"


def trigger_index_perturbed(messages: Sequence[dict[str, Any]]) -> int | None:
    for i, m in enumerate(messages):
        if _is_tool_msg(m) and (m.get("metadata") or {}).get("perturbation_status") == "perturbed":
            return i
    return None


def trigger_index_np(messages: Sequence[dict[str, Any]],
                     victim_tools: Iterable[str]) -> int | None:
    victims = set(victim_tools)
    if not victims:
        raise ValueError("")
    for i, m in enumerate(messages):
        if _is_tool_msg(m) and m.get("name") in victims and tool_call_succeeded(m.get("content")):
            return i
    return None


def prefix_at(messages: Sequence[dict[str, Any]], idx: int) -> list[dict[str, Any]]:
    if not 0 <= idx < len(messages):
        raise IndexError("")
    return list(messages[:idx + 1])


def has_dispatch_message(messages: Iterable[dict[str, Any]]) -> bool:
    return any(DISPATCH_META_KEY in (m.get("metadata") or {}) for m in messages)


def to_diagnosis_input(prefix: Sequence[dict[str, Any]], *,
                       task_description: str = "",
                       tool_schemas: Sequence[dict[str, Any]] = (),
                       trigger_round: int = -1) -> DiagnosisInput:
    obs = DiagnosisInput.from_messages(list(prefix),
                                       task_description=task_description,
                                       tool_schemas=list(tool_schemas),
                                       trigger_round=trigger_round)
    assert_input_is_clean(obs)
    return obs


def _walk(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def assert_input_is_clean(obs: DiagnosisInput) -> None:
    leaked = sorted({k for field in (obs.trace_prefix, obs.tool_schemas)
                     for node in _walk(field) if isinstance(node, dict)
                     for k in node if k in GROUND_TRUTH_KEYS})
    text_leak = sorted(k for k in GROUND_TRUTH_KEYS if k in (obs.task_description or ""))
    if leaked or text_leak:
        raise AssertionError(
            "")
    if has_dispatch_message(obs.trace_prefix):
        raise AssertionError(
            "")
    for k in FORBIDDEN_TOP_KEYS:
        if hasattr(obs, k):
            raise AssertionError("")


def build_detection_oracle_input(inference: dict[str, Any], *,
                                 victim_tools: Iterable[str] | None = None,
                                 task_description: str = "",
                                 tool_schemas: Sequence[dict[str, Any]] = (),
                                 trigger_index_fn: Callable[..., int | None] | None = None,
                                 ) -> tuple[DiagnosisInput | None, dict[str, Any]]:
    messages = inference["messages"]
    if trigger_index_fn is not None:
        idx = trigger_index_fn(messages)
    elif victim_tools is None:
        idx = trigger_index_perturbed(messages)
    else:
        idx = trigger_index_np(messages, victim_tools)
    if idx is None:
        return None, {"reason": "armed_not_touched" if victim_tools is not None else "no_trigger",
                      "trigger_index": None, "n_messages": len(messages)}
    prefix = prefix_at(messages, idx)
    if has_dispatch_message(prefix):
        return None, {"reason": "dispatch_in_prefix", "trigger_index": idx,
                      "n_prefix": len(prefix), "n_messages": len(messages)}
    obs = to_diagnosis_input(prefix, task_description=task_description,
                             tool_schemas=tool_schemas, trigger_round=idx)
    return obs, {"reason": "ok", "trigger_index": idx, "n_prefix": len(prefix),
                 "n_messages": len(messages)}
