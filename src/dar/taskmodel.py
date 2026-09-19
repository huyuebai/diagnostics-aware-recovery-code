from __future__ import annotations

import copy
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DATASET_ROOT_ENV = "DAR_TOOLMAZE_DATASET"
DEFAULT_DATASET = Path("/Users/bai/Code/dissertation_tool/ToolMaze_dataset")
VENDOR_TOOLS = Path(__file__).resolve().parents[2] / "vendor" / "toolmaze" / "tools" / "definitions"


def dataset_root() -> Path:
    return Path(os.environ.get(DATASET_ROOT_ENV, str(DEFAULT_DATASET)))


def load_task(cat: str, task_id: str, mode: str) -> dict[str, Any]:
    p = dataset_root() / "perturbed_tasks" / cat / f"{task_id}_{mode}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def iter_paths(task: dict[str, Any]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    vps = task.get("valid_paths") or []
    if vps:
        for p in vps:
            yield str(p.get("path_id")), list(p.get("execution_trace") or [])
    else:
        yield "__single__", list(task.get("execution_trace") or [])


def victims_of(task: dict[str, Any]) -> frozenset[str]:
    vs: set[str] = set()
    for _, steps in iter_paths(task):
        vs.update(s.get("tool_name") for s in steps if s.get("is_perturbed"))
    return frozenset(v for v in vs if v)


def has_alternative_path(task: dict[str, Any]) -> bool:
    return bool(task.get("valid_paths"))


@functools.lru_cache(maxsize=1)
def tool_categories() -> dict[str, str]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("") from e
    if not VENDOR_TOOLS.is_dir():
        raise FileNotFoundError(
            "")
    out: dict[str, str] = {}
    for f in sorted(VENDOR_TOOLS.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for tool in data.get("tools") or []:
            name = tool.get("name")
            cat = tool.get("category")
            if name and cat:
                out[str(name)] = str(cat)
    if not out:
        raise RuntimeError("")
    return out


def is_action(tool_name: str) -> bool:
    return tool_categories().get(tool_name, "").lower() == "action"


@functools.lru_cache(maxsize=1)
def tool_specs() -> dict[str, dict[str, Any]]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("") from e
    if not VENDOR_TOOLS.is_dir():
        raise FileNotFoundError(
            "")
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(VENDOR_TOOLS.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for tool in data.get("tools") or []:
            name = tool.get("name")
            if not name:
                continue
            spec = ((tool.get("paradigms") or {}).get("function_call") or {}).get("spec")
            if not isinstance(spec, dict) or spec.get("name") != name:
                raise RuntimeError(
                    "")
            out[str(name)] = spec
    if not out:
        raise RuntimeError("")
    return out


def trace_tool_names(task: dict[str, Any]) -> list[str]:
    if str(task.get("complexity") or "C1") in ("C2", "C3", "C4"):
        steps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for vp in task.get("valid_paths") or []:
            for s in vp.get("execution_trace") or []:
                tn = s.get("tool_name")
                if tn and tn not in seen:
                    seen.add(tn)
                    steps.append(s)
    else:
        steps = list(task.get("execution_trace") or [])
    return sorted({str(s["tool_name"]) for s in steps if s.get("tool_name")})


@dataclass(frozen=True)
class ToolSpecSelection:
    specs: tuple[dict[str, Any], ...]
    dropped: tuple[str, ...]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


def tool_specs_for_task(cat: str, task_id: str, mode: str, *,
                        loader: Any = None,
                        specs_by_name: dict[str, dict[str, Any]] | None = None,
                        ) -> ToolSpecSelection:
    task = (loader or load_task)(cat, task_id, mode)
    registry = tool_specs() if specs_by_name is None else specs_by_name
    names = trace_tool_names(task)
    specs = tuple(copy.deepcopy(registry[n]) for n in names if n in registry)
    dropped = tuple(n for n in names if n not in registry)
    return ToolSpecSelection(specs=specs, dropped=dropped)


_DEGENERATE = {True, False, None, 0, 1, "", "on", "off", "yes", "no", "true", "false"}


def leaf_scalars(obj: Any) -> set[Any]:
    out: set[Any] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= leaf_scalars(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= leaf_scalars(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        if obj not in _DEGENERATE:
            out.add(obj)
    elif isinstance(obj, str):
        s = obj.strip()
        if s.lower() not in {"", "on", "off", "yes", "no", "true", "false", "0", "1"}:
            out.add(s)
    return out
