from __future__ import annotations

from typing import Any, Sequence

from . import cost as C
from . import synthesis as S
from . import vladder as V

SCOPE_STATEMENT = (
    ""
    ""
)
SCOPE_STATEMENT_COST_BOUND_NOTE = (
    " cost_bound = ——lab/docs/b6_findings.md §8"
)


def pareto_payload(points: Sequence[V.DiagnoserPoint], *,
                   includes_np_rows: bool,
                   ledger_a_based: bool = True) -> dict[str, Any]:
    ladder = V.build_v_ladder(points)
    statements: list[str] = [SCOPE_STATEMENT, SCOPE_STATEMENT_COST_BOUND_NOTE]
    if includes_np_rows:
        statements.append(S.NP_ROW_QUALIFIER)
    if ledger_a_based:
        statements += [C.ML_TRAINING_COST_DISCLOSURE, C.LATENCY_CALIBER_NOTE]
    return {
        "points": [{"name": p.name, "cost": p.cost, "delta_p": p.delta_p} for p in points],
        "frontier": [{"name": r.point.name, "cost": r.point.cost,
                      "delta_p": r.point.delta_p, "icer_vs_prev": r.icer_vs_prev}
                     for r in ladder.rungs],
        "strictly_dominated": [list(t) for t in ladder.strictly_dominated],
        "extended_dominated": [list(t) for t in ladder.extended_dominated],
        "statements": statements,
    }


def v_scan_payload(points: Sequence[V.DiagnoserPoint], *,
                   includes_np_rows: bool) -> dict[str, Any]:
    ladder = V.build_v_ladder(points)
    scan = []
    for v in C.v_grid():
        best = V.optimal_for_value(ladder, v)
        scan.append({"v": v, "optimal": None if best is None else best.name})
    statements = [SCOPE_STATEMENT]
    if includes_np_rows:
        statements.append(S.NP_ROW_QUALIFIER)
    return {"grid_expr": "[10 ** (-1 + 0.25 * i) for i in range(17)]", "scan": scan,
            "statements": statements}
