from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class VLadderError(ValueError):
    pass


@dataclass(frozen=True)
class DiagnoserPoint:
    name: str
    cost: float
    delta_p: float


@dataclass(frozen=True)
class LadderRung:
    point: DiagnoserPoint
    icer_vs_prev: float | None


@dataclass(frozen=True)
class VLadder:
    rungs: tuple[LadderRung, ...]
    strictly_dominated: tuple[tuple[str, str], ...]
    extended_dominated: tuple[tuple[str, str, str], ...]


def _validate(points: Sequence[DiagnoserPoint]) -> None:
    if not points:
        raise VLadderError("empty input: a ladder needs at least one diagnoser point")
    names = [p.name for p in points]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise VLadderError(f"duplicate diagnoser names: {dupes}")
    for p in points:
        for attr in ("cost", "delta_p"):
            v = getattr(p, attr)
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise VLadderError(f"{p.name}.{attr} must be a finite number, got {v!r}")


def _strictly_dominates(a: DiagnoserPoint, b: DiagnoserPoint) -> bool:
    return (a.cost <= b.cost and a.delta_p >= b.delta_p
            and (a.cost < b.cost or a.delta_p > b.delta_p))


def build_v_ladder(points: Sequence[DiagnoserPoint]) -> VLadder:
    _validate(points)

    survivors = list(points)
    strictly_dropped: list[tuple[str, str]] = []
    for p in points:
        dominator = next((q for q in points if q is not p and _strictly_dominates(q, p)), None)
        if dominator is not None:
            survivors.remove(p)
            strictly_dropped.append((p.name, dominator.name))

    seen: dict[tuple[float, float], str] = {}
    for p in survivors:
        key = (float(p.cost), float(p.delta_p))
        if key in seen:
            raise VLadderError(
                f"{seen[key]!r} and {p.name!r} have identical (cost, delta_p)={key}: "
                f"equal-Δp frontier points give a zero ICER denominator; the ladder "
                f"cannot rank them (fail-loud clause) — deduplicate upstream."
            )
        seen[key] = p.name

    survivors.sort(key=lambda p: p.delta_p)

    extended_dropped: list[tuple[str, str, str]] = []
    changed = True
    while changed and len(survivors) >= 3:
        changed = False
        for k in range(1, len(survivors) - 1):
            lo, mid, hi = survivors[k - 1], survivors[k], survivors[k + 1]
            icer_in = (mid.cost - lo.cost) / (mid.delta_p - lo.delta_p)
            icer_out = (hi.cost - mid.cost) / (hi.delta_p - mid.delta_p)
            if icer_in >= icer_out:
                extended_dropped.append((mid.name, lo.name, hi.name))
                del survivors[k]
                changed = True
                break

    rungs: list[LadderRung] = []
    for k, p in enumerate(survivors):
        if k == 0:
            rungs.append(LadderRung(point=p, icer_vs_prev=None))
            continue
        prev = survivors[k - 1]
        denom = p.delta_p - prev.delta_p
        if denom <= 0:
            raise VLadderError(f"non-increasing Δp between {prev.name!r} and {p.name!r}")
        rungs.append(LadderRung(point=p, icer_vs_prev=(p.cost - prev.cost) / denom))

    icers = [r.icer_vs_prev for r in rungs if r.icer_vs_prev is not None]
    if any(b <= a for a, b in zip(icers, icers[1:])):
        raise VLadderError(f"internal error: ICER sequence not strictly increasing: {icers}")

    return VLadder(rungs=tuple(rungs),
                   strictly_dominated=tuple(strictly_dropped),
                   extended_dominated=tuple(extended_dropped))


def endogenous_v_reference(task_reexecution_cost: float) -> float:
    if (math.isnan(task_reexecution_cost) or math.isinf(task_reexecution_cost)
            or task_reexecution_cost < 0):
        raise VLadderError(
            f"task_reexecution_cost must be a finite non-negative cost, "
            f"got {task_reexecution_cost!r}")
    return task_reexecution_cost


def optimal_for_value(ladder: VLadder, v: float) -> DiagnoserPoint | None:
    if math.isnan(v) or math.isinf(v):
        raise VLadderError(f"v must be finite, got {v!r}")
    if not ladder.rungs:
        raise VLadderError("empty ladder")
    best: DiagnoserPoint | None = None
    best_nb = 0.0
    for rung in ladder.rungs:
        p = rung.point
        nb = v * p.delta_p - p.cost
        if nb > best_nb or (nb == best_nb and best is not None and p.cost < best.cost):
            best, best_nb = p, nb
    return best
