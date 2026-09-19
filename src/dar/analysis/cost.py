from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from dar.labels import Label

from . import synthesis as S

ML_TRAINING_COST_DISCLOSURE = (
    "⚠️ [BD-51] (a)2026-08-24ML  AMac CPU "
    " lab/docs/b6_findings.md §5.1  A  ml  rule "
    "ml  latency_ms ≡ 0 "
)
LATENCY_CALIBER_NOTE = (
    "⚠️ [BD-70] (iv)2026-08-24——API  latency_ms "
    "ml  latency_ms T1b "
    " = disclosure-only[BD-34]"
)

_T1_MAIN_AXES = frozenset({"main", "transient", "persistent"})
_T1_ALL_AXES = _T1_MAIN_AXES | {"extended"}


def select_ledger_a(rows: Iterable[Mapping[str, Any]], diagnoser: str) -> list[dict[str, Any]]:
    if diagnoser not in {"rule", "ml", "oracle"}:
        raise ValueError(f"ledger_a diagnosers are rule/ml/oracle, got {diagnoser!r}")
    return [dict(r) for r in rows if r["diagnoser"] == diagnoser]


def select_t1_main(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        axis = r["axis"]
        if axis not in _T1_ALL_AXES:
            raise ValueError(f"unknown t1 ledger axis {axis!r} (row {r.get('run_id')!r}): "
                             f"refusing to classify silently")
        if axis in _T1_MAIN_AXES:
            out.append(dict(r))
    return out


def select_t2_main(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def assert_basis_nonempty(rows: Iterable[Mapping[str, Any]]) -> None:
    for r in rows:
        basis = r.get("basis")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("")


def native_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("")
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "calls": sum(int(r["calls"]) for r in rows),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in rows),
        "completion_tokens": sum(int(r["completion_tokens"]) for r in rows),
    }
    if rows and all("latency_ms" in r and r["latency_ms"] is not None for r in rows):
        out["latency_ms"] = sum(float(r["latency_ms"]) for r in rows)
    return out


def usd_overlay(native: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    basis = rates.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("rates.basis must be a non-empty source string ([LM6]/[BD-11])")
    for key in ("usd_per_1k_prompt", "usd_per_1k_completion"):
        v = rates.get(key)
        if not isinstance(v, (int, float)) or math.isnan(float(v)) or float(v) < 0:
            raise ValueError(f"rates.{key} must be a non-negative number, got {v!r}")
    usd = (native["prompt_tokens"] / 1000.0 * float(rates["usd_per_1k_prompt"])
           + native["completion_tokens"] / 1000.0 * float(rates["usd_per_1k_completion"]))
    return {"usd": usd, "basis": basis}


def lambda_grid() -> list[float]:
    return [10 ** (-2 + 0.25 * i) for i in range(17)]


def v_grid() -> list[float]:
    return [10 ** (-1 + 0.25 * i) for i in range(17)]


def combined_cost(diagnoser_cost: float, downstream_cost: float, lam: float) -> float:
    for name, v in (("diagnoser_cost", diagnoser_cost),
                    ("downstream_cost", downstream_cost), ("lam", lam)):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        if v < 0 and name != "downstream_cost":
            raise ValueError(f"{name} must be non-negative, got {v!r}")
    return diagnoser_cost + lam * downstream_cost


def scan_lambda(diagnoser_cost: float, downstream_cost: float) -> list[dict[str, float]]:
    return [{"lam": lam, "total": combined_cost(diagnoser_cost, downstream_cost, lam)}
            for lam in lambda_grid()]


_WASTE_LABELS = (Label.TRANSIENT.value, Label.PERSISTENT.value)


def np_waste_per_label(per_run_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_label: dict[str, list[Mapping[str, Any]]] = {lab: [] for lab in _WASTE_LABELS}
    for r in per_run_rows:
        lab = r["label"]
        if lab not in by_label:
            raise ValueError(f"unexpected np_waste label {lab!r} "
                             f"(expected one of {_WASTE_LABELS})")
        by_label[lab].append(r)
    out: dict[str, dict[str, Any]] = {}
    for lab, rows in by_label.items():
        if not rows:
            raise ValueError(f"np_waste label {lab!r} has zero rows: refusing an empty mean")
        tasks = {(r["cat"], r["tid"]) for r in rows}
        if len(tasks) != len(rows):
            raise ValueError(
                "")
        out[lab] = {
            "mean_waste_calls": sum(float(r["np_waste_calls"]) for r in rows) / len(tasks),
            "n_tasks": len(tasks),
            "n_pairs": len(rows),
            "denominator": "n_tasks",
        }
    return out


def np_waste_pi_weighted(waste_by_label: Mapping[str, Mapping[str, Any]],
                         T: S.TransitionMatrix) -> dict[str, Any]:
    i = S.LABEL_INDEX[Label.NP]
    value = 0.0
    for j, lab in enumerate(S.LABELS):
        if lab == Label.NP:
            continue
        value += T.rows[i][j] * float(waste_by_label[lab.value]["mean_waste_calls"])
    return {
        "value": value,
        "estimand": "pi_weighted_mixture (presentation layer, not headline)",
        "denominator": "n_tasks",
        "qualifiers": (S.NP_ROW_QUALIFIER, S.NP_SINGLE_RUN_WARNING),
    }


def np_waste_risk_class_disclosure(
        per_run_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for r in per_run_rows:
        if r["label"] not in _WASTE_LABELS:
            raise ValueError(f"unexpected np_waste label {r['label']!r} "
                             f"(expected one of {_WASTE_LABELS})")
        by_class.setdefault(str(r["risk_class"]), []).append(r)
    if not by_class:
        raise ValueError("")
    out: dict[str, dict[str, Any]] = {}
    for cls, rows in sorted(by_class.items()):
        tasks = {(r["cat"], r["tid"]) for r in rows}
        out[cls] = {
            "mean_waste_calls": sum(float(r["np_waste_calls"]) for r in rows) / len(rows),
            "n_tasks": len(tasks),
            "n_rows": len(rows),
            "denominator": "n_rows ℓ ——/ n  n_tasks "
                           "",
        }
    return out
