from __future__ import annotations

from typing import Any, Callable

from dar.labels import Label
from dar.taskmodel import load_task

_LABEL_TO_SIBLING: dict[Label, str] = {
    Label.TRANSIENT: "P1",
    Label.PERSISTENT: "P2",
}


def tool_call_succeeded(tool_result: Any) -> bool:
    return not (isinstance(tool_result, dict) and "error" in tool_result)


def perturbed_victims(task: dict[str, Any]) -> frozenset[str]:
    vs: set[str] = set()
    for s in (task.get("execution_trace") or []):
        if s.get("is_perturbed"):
            vs.add(s.get("tool_name"))
    for p in (task.get("valid_paths") or []):
        for s in (p.get("execution_trace") or []):
            if s.get("is_perturbed"):
                vs.add(s.get("tool_name"))
    return frozenset(v for v in vs if v)


def np_trigger(cat: str, base_id: str, label: Label,
               loader: Callable[[str, str, str], dict[str, Any]] = load_task,
               ) -> dict[str, Any] | None:
    mode = _LABEL_TO_SIBLING.get(label)
    if mode is None:
        if label is Label.NP:
            return None
        raise ValueError("")
    sibling = loader(cat, base_id, mode)
    victims = perturbed_victims(sibling)
    if not victims:
        raise ValueError("")
    return {"victim_tools": sorted(victims), "source_sibling": mode}
