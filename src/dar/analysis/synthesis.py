from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

LABELS: tuple[Label, ...] = (Label.TRANSIENT, Label.PERSISTENT, Label.NP)
LABEL_INDEX: dict[Label, int] = {t: i for i, t in enumerate(LABELS)}
_N = len(LABELS)

BD50_T_ROW_WARNING = (
    "⚠️ [BD-50] (a)2026-08-24 T  transientpersistent "
    "—— p  w_ℓ "
    " p* "
    " = lab/docs/b6_findings.md §8"
)
BD50_ML_NP_ROW_WARNING = (
    "⚠️ [BD-50] V3 ml  np ——ml  P3/P4  P0 "
    "190/192 np np "
    " = lab/docs/b6_findings.md §8"
)
NP_ROW_QUALIFIER = (
    "NP×transient  P1 victim NP×persistent  P2 victim U6 first-touch"
    "⇒ "
)
B6_NP_ROW_QUALIFIER = (
    " ≠  B5  μ̂(NP,·) "
    " = lab/docs/b6_findings.md §8"
)
NP_SINGLE_RUN_WARNING = (
    "⚠️ [BD-62]② V1-4NP rule/llm/api "
    " run —— NP  w_ℓ  np_unitsnp_runs "
    " = lab/docs/b6_findings.md §8"
)


def _acc_vector(accuracy: float | Mapping[Label, float]) -> list[float]:
    vec = ([float(accuracy[LABELS[i]]) for i in range(_N)] if isinstance(accuracy, Mapping)
           else [float(accuracy)] * _N)
    for a in vec:
        if math.isnan(a) or not 0.0 <= a <= 1.0:
            raise ValueError(f"per-type accuracy must be in [0,1], got {a}")
    return vec


def _sample(weights: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for j, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return j
    return len(weights) - 1


def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    wsum = sum(weights)
    if total == 0:
        return [0] * len(weights)
    if wsum <= 0:
        raise ValueError(f"cannot allocate {total} units over non-positive weights {weights}")
    exact = [total * w / wsum for w in weights]
    base = [int(x) for x in exact]
    short = total - sum(base)
    order = sorted(range(len(weights)), key=lambda j: (-(exact[j] - base[j]), j))
    for j in order[:short]:
        base[j] += 1
    return base


@dataclass
class TransitionMatrix:
    rows: list[list[float]]

    def __post_init__(self) -> None:
        if len(self.rows) != _N:
            raise ValueError(f"expected {_N} rows, got {len(self.rows)}")
        for i, row in enumerate(self.rows):
            if len(row) != _N:
                raise ValueError(f"row {i} has {len(row)} entries, expected {_N}")
            if any(math.isnan(w) for w in row):
                raise ValueError(f"row {i} contains NaN")
            if any(w < 0 for w in row):
                raise ValueError(f"row {i} has a negative probability")
            if abs(sum(row) - 1.0) > 1e-9:
                raise ValueError(f"row {i} is not stochastic (sum={sum(row)})")

    def per_type_accuracy(self) -> dict[Label, float]:
        return {LABELS[i]: self.rows[i][i] for i in range(_N)}

    def corrupt(self, true_label: Label, rng: random.Random) -> Label:
        i = LABEL_INDEX.get(true_label)
        if i is None:
            return true_label
        return LABELS[_sample(self.rows[i], rng)]

    def to_config(self) -> dict:
        return {"labels": [t.value for t in LABELS], "rows": [list(r) for r in self.rows]}

    @classmethod
    def from_config(cls, config: dict) -> "TransitionMatrix":
        labels = config.get("labels")
        if labels is not None and labels != [t.value for t in LABELS]:
            raise ValueError("config label order does not match the taxonomy")
        return cls([list(r) for r in config["rows"]])


def uniform_matrix(accuracy: float | Mapping[Label, float]) -> TransitionMatrix:
    acc = _acc_vector(accuracy)
    rows = []
    for i in range(_N):
        off = (1.0 - acc[i]) / (_N - 1)
        row = [off] * _N
        row[i] = acc[i]
        rows.append(row)
    return TransitionMatrix(rows)


def random_baseline_matrix() -> TransitionMatrix:
    return uniform_matrix(1.0 / _N)


def empirical_matrix(confusion: Sequence[Sequence[float]],
                     accuracy: float | Mapping[Label, float] | None = None,
                     smoothing_alpha: float = 0.0) -> TransitionMatrix:
    if math.isnan(smoothing_alpha) or smoothing_alpha < 0:
        raise ValueError(f"smoothing_alpha must be >= 0, got {smoothing_alpha}")
    if len(confusion) != _N or any(len(r) != _N for r in confusion):
        raise ValueError(
            f"confusion must be {_N}x{_N}, got {len(confusion)} rows of lengths "
            f"{[len(r) for r in confusion]}")
    smoothed = [[float(confusion[i][j]) + smoothing_alpha for j in range(_N)]
                for i in range(_N)]
    totals = [sum(smoothed[i]) for i in range(_N)]
    for i, total in enumerate(totals):
        if total <= 0:
            raise ValueError(
                f"row {i} ({LABELS[i].value}) has zero observations under "
                f"alpha={smoothing_alpha}: refusing to fabricate a row "
                f"(zero-observation rows must be rejected loudly, prereg §2.3)")
    if accuracy is None:
        rows = [[smoothed[i][j] / totals[i] for j in range(_N)] for i in range(_N)]
        return TransitionMatrix(rows)
    acc = _acc_vector(accuracy)
    rows = []
    for i in range(_N):
        off_total = sum(smoothed[i][j] for j in range(_N) if j != i)
        row = [0.0] * _N
        row[i] = acc[i]
        if off_total > 0:
            for j in range(_N):
                if j != i:
                    row[j] = (1.0 - acc[i]) * (smoothed[i][j] / off_total)
        else:
            for j in range(_N):
                if j != i:
                    row[j] = (1.0 - acc[i]) / (_N - 1)
        rows.append(row)
    return TransitionMatrix(rows)


def adversarial_matrix(accuracy: float | Mapping[Label, float],
                       target: Mapping[Label, Label]) -> TransitionMatrix:
    acc = _acc_vector(accuracy)
    rows = []
    for i in range(_N):
        true_t = LABELS[i]
        row = [0.0] * _N
        row[i] = acc[i]
        tgt = target.get(true_t)
        if tgt is None or tgt == true_t or tgt not in LABEL_INDEX:
            for j in range(_N):
                if j != i:
                    row[j] = (1.0 - acc[i]) / (_N - 1)
        else:
            row[LABEL_INDEX[tgt]] = 1.0 - acc[i]
        rows.append(row)
    return TransitionMatrix(rows)


def matrix_from_b6_counts(payload: Mapping[str, Any]) -> TransitionMatrix:
    if payload.get("kind") != "counts":
        raise ValueError(f"expected a counts payload, got kind={payload.get('kind')!r}")
    diag = payload.get("diagnoser")
    if not isinstance(diag, str) or not diag:
        raise ValueError("")
    if payload.get("face") != "head_to_head":
        raise ValueError(
            "")
    if payload.get("smoothing_alpha") != 0:
        raise ValueError(
            f"counts payload smoothing_alpha must be 0, got {payload.get('smoothing_alpha')!r}")
    expected_order = [t.value for t in LABELS]
    if payload.get("axis_order") != expected_order:
        raise ValueError(
            f"axis_order mismatch: file={payload.get('axis_order')!r}, "
            f"consumer={expected_order!r}")
    m = payload["matrix"]
    confusion = [[float(m[row.value][col.value]) for col in LABELS] for row in LABELS]
    return empirical_matrix(confusion, smoothing_alpha=0.0)


def zero_observed_columns(payload: Mapping[str, Any]) -> tuple[Label, ...]:
    m = payload["matrix"]
    out = []
    for col in LABELS:
        if sum(float(m[row.value][col.value]) for row in LABELS) == 0:
            out.append(col)
    return tuple(out)


def np_unit_count(payload: Mapping[str, Any]) -> int:
    m = payload["matrix"]
    return int(sum(float(m[Label.NP.value][col.value]) for col in LABELS))


def confusion_weights(T: TransitionMatrix, true_type: TrueType) -> tuple[float, dict[Label, float]]:
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    i = LABEL_INDEX[star]
    p = T.rows[i][i]
    w: dict[Label, float] = {lab: 0.0 for lab in LABELS}
    if p >= 1.0:
        return p, w
    for j, lab in enumerate(LABELS):
        if j != i:
            w[lab] = T.rows[i][j] / (1.0 - p)
    return p, w


def expected_outcome(mu_row: Mapping[Label, float], true_type: TrueType,
                     T: TransitionMatrix) -> float:
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    i = LABEL_INDEX[star]
    acc = 0.0
    for j, lab in enumerate(LABELS):
        acc += T.rows[i][j] * mu_row[lab]
    return acc


def family_matrix(source: TransitionMatrix, p: float) -> TransitionMatrix:
    if math.isnan(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0,1], got {p}")
    rows = []
    for i in range(_N):
        row = [0.0] * _N
        row[i] = p
        if p < 1.0:
            off_total = sum(source.rows[i][j] for j in range(_N) if j != i)
            if off_total <= 0:
                raise ValueError(
                    f"source row {i} ({LABELS[i].value}) has zero off-diagonal mass: "
                    f"no confusion shape to inherit for p={p} < 1")
            for j in range(_N):
                if j != i:
                    row[j] = (1.0 - p) * (source.rows[i][j] / off_total)
        rows.append(row)
    return TransitionMatrix(rows)


def pi_uniform() -> dict[TrueType, float]:
    return {t: 1.0 / len(TrueType) for t in TrueType}


def pi_np_grid() -> list[float]:
    return [i / 20 for i in range(20)]


def pi_for_np(pi_np: float) -> dict[TrueType, float]:
    if math.isnan(pi_np) or not 0.0 <= pi_np <= 1.0:
        raise ValueError(f"pi_np must be in [0,1], got {pi_np}")
    rest = (1.0 - pi_np) / 4
    out = {t: rest for t in TrueType}
    out[TrueType.NP] = pi_np
    return out


def aggregate(mu_table: Mapping[TrueType, Mapping[Label, float]],
              T: TransitionMatrix,
              pi: Mapping[TrueType, float]) -> float:
    missing = [t for t in TrueType if t not in pi]
    if missing:
        raise ValueError(f"pi missing true types: {[t.value for t in missing]}")
    total = sum(pi[t] for t in TrueType)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"pi must sum to 1, got {total}")
    return sum(pi[t] * expected_outcome(mu_table[t], t, T) for t in TrueType)


@dataclass(frozen=True)
class EmpiricalGridPoint:
    value: float | None
    reason: str | None
    zero_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict)


def zero_column_guard(zero_cols: Sequence[Label], true_type: TrueType) -> str | None:
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    for col in zero_cols:
        if col != star:
            return f"zero_observed_column_{col.value}"
    return None


def empirical_warnings(payload: Mapping[str, Any], true_type: TrueType) -> list[str]:
    warnings = [BD50_T_ROW_WARNING]
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    if star == Label.NP:
        warnings += [NP_ROW_QUALIFIER, B6_NP_ROW_QUALIFIER, NP_SINGLE_RUN_WARNING]
        if payload.get("diagnoser") == "ml":
            warnings.append(BD50_ML_NP_ROW_WARNING)
    return warnings


def empirical_outcome(mu_row: Mapping[Label, float], true_type: TrueType,
                      payload: Mapping[str, Any]) -> EmpiricalGridPoint:
    T = matrix_from_b6_counts(payload)
    zero_cols = zero_observed_columns(payload)
    would_be = expected_outcome(mu_row, true_type, T)
    warnings = empirical_warnings(payload, true_type)
    star = TRUE_TYPE_TO_LABEL[TrueType(true_type)]
    reason = zero_column_guard(zero_cols, true_type)
    if reason is not None:
        p, w = confusion_weights(T, true_type)
        r_val = mu_row[star] - mu_row[Label.NP]
        d_bar_val = sum(w[lab] * (mu_row[Label.NP] - mu_row[lab])
                        for lab in LABELS if lab != star)
        return EmpiricalGridPoint(
            value=None, reason=reason,
            zero_columns=tuple(c.value for c in zero_cols),
            warnings=tuple(warnings),
            raw={"value_if_zero_weight": would_be, "p": p,
                 "r": r_val, "d_bar": d_bar_val})
    return EmpiricalGridPoint(
        value=would_be, reason=None,
        zero_columns=tuple(c.value for c in zero_cols),
        warnings=tuple(warnings), raw={})
