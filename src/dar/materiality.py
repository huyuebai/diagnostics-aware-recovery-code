from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from dar.taskmodel import is_action, iter_paths, leaf_scalars, load_task

AXES = {"transient": "P1", "persistent": "P2"}


def _query_obtainable(value: Any, query_text: str) -> bool:
    s = str(value).strip()
    if not s:
        return True
    return re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", query_text) is not None


def _clean_output_map(task_p0: dict[str, Any]) -> dict[tuple[str, str], set[Any]]:
    out: dict[tuple[str, str], set[Any]] = {}
    for pid, steps in iter_paths(task_p0):
        for s in steps:
            key = (pid, s.get("tool_name"))
            out.setdefault(key, set()).update(leaf_scalars(s.get("output")))
    return out


@dataclass
class VictimMateriality:
    victim: str
    material_paths: list[str] = field(default_factory=list)
    checked_paths: list[str] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def material(self) -> bool:
        return bool(self.material_paths)


def axis_materiality(cat: str, task_id: str, axis: str) -> dict[str, Any]:
    mode = AXES[axis]
    task_mode = load_task(cat, task_id, mode)
    task_p0 = load_task(cat, task_id, "P0")
    clean = _clean_output_map(task_p0)
    query_text = json.dumps(task_p0.get("user_input"), ensure_ascii=False)

    per_victim: dict[str, VictimMateriality] = {}
    for pid, steps in iter_paths(task_mode):
        first_at: dict[str, int] = {}
        perturbed_args: dict[str, list[Any]] = {}
        for i, s in enumerate(steps):
            if s.get("is_perturbed"):
                first_at.setdefault(s.get("tool_name"), i)
                perturbed_args.setdefault(s.get("tool_name"), []).append(s.get("arguments"))
        for victim, vi in first_at.items():
            vm = per_victim.setdefault(victim, VictimMateriality(victim=victim))
            vm.checked_paths.append(pid)
            values = clean.get((pid, victim))
            if values is None:
                values = set()
                for (p2, t2), v2 in clean.items():
                    if t2 == victim:
                        values |= v2
            values = {v for v in values if not _query_obtainable(v, query_text)}
            hit = False
            for j in range(vi + 1, len(steps)):
                tool_j = steps[j].get("tool_name")
                if not is_action(tool_j):
                    continue
                if tool_j == victim and steps[j].get("arguments") in perturbed_args[victim]:
                    continue
                arg_leaves = leaf_scalars(steps[j].get("arguments"))
                inter = values & arg_leaves
                if inter:
                    hit = True
                    vm.matches.append({"path": pid, "to_tool": tool_j,
                                       "values": sorted(map(str, inter))[:6]})
            if hit:
                vm.material_paths.append(pid)

    victims = sorted(per_victim.values(), key=lambda v: v.victim)
    n_pairs = sum(len(v.checked_paths) for v in victims)
    n_material = sum(len(v.material_paths) for v in victims)
    return {
        "axis": axis,
        "mode_file": mode,
        "victims": [{
            "victim": v.victim,
            "material": v.material,
            "material_paths": v.material_paths,
            "checked_paths": v.checked_paths,
            "matches": v.matches,
        } for v in victims],
        "any_material": any(v.material for v in victims),
        "pair_frac": (n_material / n_pairs) if n_pairs else None,
    }
