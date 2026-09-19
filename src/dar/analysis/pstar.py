from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

from . import synthesis as S

REASON_DENOM_ZERO = "denominator_zero_no_breakeven"
REASON_DBAR_NEGATIVE = "d_bar_negative"
REASON_DBAR_ZERO = "d_bar_zero"


def p_grid() -> list[float]:
    return [i / 10_000 for i in range(10_001)]


@dataclass(frozen=True)
class PStarResult:
    value: float | None
    reason: str | None
    out_of_unit_interval: bool
    r: float
    d_bar: float
    raw: dict[str, Any] = field(default_factory=dict)


def r_of(mu_row: Mapping[Label, float], true_type: TrueType) -> float:
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    return mu_row[star] - mu_row[Label.NP]


def d_components(mu_row: Mapping[Label, float]) -> dict[Label, float]:
    return {lab: mu_row[Label.NP] - mu_row[lab] for lab in S.LABELS}


def d_bar_of(mu_row: Mapping[Label, float], true_type: TrueType,
             T: S.TransitionMatrix) -> float:
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    _, w = S.confusion_weights(T, true_type)
    d = d_components(mu_row)
    return sum(w[lab] * d[lab] for lab in S.LABELS if lab != star)


def pstar_closed(r: float, d_bar: float) -> PStarResult:
    if math.isnan(r) or math.isnan(d_bar):
        raise ValueError(f"r/d_bar must not be NaN, got r={r!r}, d_bar={d_bar!r}")
    denom = r + d_bar
    if denom == 0:
        return PStarResult(value=None, reason=REASON_DENOM_ZERO,
                           out_of_unit_interval=False, r=r, d_bar=d_bar)
    if d_bar == 0:
        return PStarResult(value=0.0, reason=REASON_DBAR_ZERO,
                           out_of_unit_interval=False, r=r, d_bar=d_bar)
    value = d_bar / denom
    out = (d_bar < 0) or not (0.0 <= value <= 1.0)
    reason = REASON_DBAR_NEGATIVE if d_bar < 0 else None
    return PStarResult(value=value, reason=reason, out_of_unit_interval=out,
                       r=r, d_bar=d_bar)


def pstar_for(mu_row: Mapping[Label, float], true_type: TrueType,
              T: S.TransitionMatrix) -> PStarResult:
    return pstar_closed(r_of(mu_row, true_type), d_bar_of(mu_row, true_type, T))


def breakeven_gap(p: float, r: float, d_bar: float) -> float:
    return p * r - (1.0 - p) * d_bar


def grid_root(r: float, d_bar: float, grid: list[float] | None = None) -> float | None:
    if r == 0 and d_bar == 0:
        return None
    pts = p_grid() if grid is None else grid
    prev_p, prev_g = pts[0], breakeven_gap(pts[0], r, d_bar)
    if prev_g == 0.0:
        return prev_p
    for p in pts[1:]:
        g = breakeven_gap(p, r, d_bar)
        if g == 0.0:
            return p
        if (prev_g < 0) != (g < 0):
            return prev_p if abs(prev_g) <= abs(g) else p
        prev_p, prev_g = p, g
    return None


def empirical_pstar(mu_row: Mapping[Label, float], true_type: TrueType,
                    payload: Mapping[str, Any]) -> S.EmpiricalGridPoint:
    T = S.matrix_from_b6_counts(payload)
    zero_cols = S.zero_observed_columns(payload)
    res = pstar_for(mu_row, true_type, T)
    warnings = S.empirical_warnings(payload, true_type)
    reason = S.zero_column_guard(zero_cols, true_type)
    if reason is not None:
        return S.EmpiricalGridPoint(
            value=None, reason=reason,
            zero_columns=tuple(c.value for c in zero_cols),
            warnings=tuple(warnings),
            raw={"value_if_zero_weight": res.value, "r": res.r, "d_bar": res.d_bar,
                 "closed_form_reason": res.reason})
    return S.EmpiricalGridPoint(
        value=res.value, reason=res.reason,
        zero_columns=tuple(c.value for c in zero_cols),
        warnings=tuple(warnings),
        raw={"r": res.r, "d_bar": res.d_bar,
             "out_of_unit_interval": res.out_of_unit_interval})
