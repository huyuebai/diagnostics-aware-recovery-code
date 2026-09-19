from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from dar.b5_stats import is_replicate_row
from dar.ga_semantic import main_estimator
from dar.headroom import AXIS_OF, EXCLUDE_DISPOSITIONS
from dar.np_trigger import _LABEL_TO_SIBLING, perturbed_victims

NP_SIBLING_OF_LABEL = {lab.value: sib for lab, sib in _LABEL_TO_SIBLING.items()}


FIXED_TIER_ZERO_BASIS = "fixed-tier-zero"


@dataclass(frozen=True)
class LedgerA:
    usd: float
    basis: str
    diagnoser: str = ""
    run_id: str = ""
    axis: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0

    @classmethod
    def b6_row(cls, *, diagnoser: str, run_id: str, axis: str, usd: float, basis: str,
               prompt_tokens: int = 0, completion_tokens: int = 0, calls: int = 0,
               latency_ms: float = 0.0) -> "LedgerA":
        for name, value in (("diagnoser", diagnoser), ("run_id", run_id), ("axis", axis)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "")
        return cls(usd=usd, basis=basis, diagnoser=diagnoser, run_id=run_id, axis=axis,
                   prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                   calls=calls, latency_ms=latency_ms)

    def __post_init__(self) -> None:
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError(
                "")
        if isinstance(self.usd, bool) or not isinstance(self.usd, (int, float)):
            raise ValueError("")
        if self.usd < 0:
            raise ValueError("")
        if self.basis == FIXED_TIER_ZERO_BASIS and self.usd != 0.0:
            raise ValueError(
                "")
        for name in ("prompt_tokens", "completion_tokens", "calls"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError("")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)) \
                or self.latency_ms < 0:
            raise ValueError("")


def ledger_a_fixed_zero() -> LedgerA:
    return LedgerA(usd=0.0, basis=FIXED_TIER_ZERO_BASIS)


RC_NA_NO_HIT = "N/A:no_hit"
RC_NA_NOT_ACHIEVED = "N/A:not_achieved"
RC_NA_ORACLE_UNDEFINED = "N/A:oracle_undefined"
RC_BASIS_COMPUTED = "excess_calls_vs_oracle"

LEDGER_B_FIELDS = (
    "run_id", "cell_id", "batch", "truth", "label", "cat", "tid", "mode", "replicate_rep",
    "input_tokens", "output_tokens", "n_tool_calls", "wall_s_amortized",
    "judge_pass", "ga_verdict", "ga_achieved",
    "hit", "prr_hits", "prr_resolved",
    "actual_recovery_calls", "oracle_recovery_calls", "excess_calls_vs_oracle",
    "rc", "rc_basis",
)

_MANIFEST_KEYS = ("run_id", "cell_id", "batch", "truth", "label", "cat", "tid",
                  "mode", "replicate_rep")


def _require_int(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("")
    return v


def _require_nonneg_int(v: Any, what: str) -> int:
    n = _require_int(v, what)
    if n < 0:
        raise ValueError("")
    return n


def tool_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        events.append({
            "tool_name": msg.get("name"),
            "success": not (isinstance(content, dict) and "error" in content),
            "perturbed": (msg.get("metadata") or {}).get("perturbation_status") == "perturbed",
        })
    return events


def unique_first_hits(events: list[dict[str, Any]]) -> int:
    seen: set[Any] = set()
    n = 0
    for e in events:
        if e["perturbed"] and e["tool_name"] not in seen:
            seen.add(e["tool_name"])
            n += 1
    return n


def actual_recovery_calls(events: list[dict[str, Any]]) -> int | None:
    for idx, e in enumerate(events):
        if e["perturbed"]:
            return len(events) - idx
    return None


DISPATCH_META_KEY = "dar_dispatch"


def dispatch_labels_in_trace(messages: list[dict[str, Any]] | None) -> list[str | None]:
    out: list[str | None] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        meta = (msg.get("metadata") or {}).get(DISPATCH_META_KEY)
        if isinstance(meta, dict):
            out.append(meta.get("diagnosed_label"))
    return out


def build_ledger_b_row(manifest_row: dict[str, Any], inference: dict[str, Any],
                       evaluation: dict[str, Any], *, ga_verdict: str,
                       prr_resolved: int, oracle_recovery_calls: int | None = None,
                       wall_s_amortized: float | None = None) -> dict[str, Any]:
    for k in _MANIFEST_KEYS:
        if k not in manifest_row:
            raise ValueError("")
    tid, mode = manifest_row["tid"], manifest_row["mode"]
    for what, doc in (("inference", inference), ("evaluation", evaluation)):
        if doc.get("task_id") != tid or doc.get("mode") != mode:
            raise ValueError(
                "")

    label = manifest_row["label"]
    seen_labels = dispatch_labels_in_trace(inference.get("messages"))
    if label == "np":
        if seen_labels:
            raise ValueError(
                "")
    else:
        bad = [x for x in seen_labels if x != label]
        if bad:
            raise ValueError(
                "")

    tokens = inference.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("")
    input_tokens = _require_nonneg_int(tokens.get("input_tokens"), "tokens.input_tokens")
    output_tokens = _require_nonneg_int(tokens.get("output_tokens"), "tokens.output_tokens")

    judge_pass = evaluation.get("pass")
    if not isinstance(judge_pass, bool):
        raise ValueError("")

    ga_achieved = main_estimator(ga_verdict)

    events = tool_events(inference.get("messages") or [])
    n_tool_calls = len(events)
    prr_hits = unique_first_hits(events)
    hit = any(e["perturbed"] for e in events)
    _require_nonneg_int(prr_resolved, "prr_resolved")
    if prr_resolved > prr_hits:
        raise ValueError("")

    arc = actual_recovery_calls(events)
    if oracle_recovery_calls is not None:
        _require_nonneg_int(oracle_recovery_calls, "oracle_recovery_calls")
    excess = (arc - oracle_recovery_calls
              if (arc is not None and oracle_recovery_calls is not None) else None)

    if wall_s_amortized is not None:
        if isinstance(wall_s_amortized, bool) or not isinstance(wall_s_amortized, (int, float)):
            raise ValueError("")
        if wall_s_amortized < 0:
            raise ValueError("")
        wall_s_amortized = float(wall_s_amortized)

    if not hit:
        rc, rc_basis = None, RC_NA_NO_HIT
    elif not ga_achieved:
        rc, rc_basis = None, RC_NA_NOT_ACHIEVED
    elif excess is None:
        rc, rc_basis = None, RC_NA_ORACLE_UNDEFINED
    else:
        rc, rc_basis = float(excess), RC_BASIS_COMPUTED

    row = {k: manifest_row[k] for k in _MANIFEST_KEYS}
    row.update({
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "n_tool_calls": n_tool_calls, "wall_s_amortized": wall_s_amortized,
        "judge_pass": judge_pass, "ga_verdict": ga_verdict, "ga_achieved": ga_achieved,
        "hit": hit, "prr_hits": prr_hits, "prr_resolved": prr_resolved,
        "actual_recovery_calls": arc, "oracle_recovery_calls": oracle_recovery_calls,
        "excess_calls_vs_oracle": excess, "rc": rc, "rc_basis": rc_basis,
    })
    assert set(row) == set(LEDGER_B_FIELDS), ""
    return row


def _prr_ratio(row: dict[str, Any]) -> float | None:
    h = _require_nonneg_int(row["prr_hits"], "prr_hits")
    s = _require_nonneg_int(row["prr_resolved"], "prr_resolved")
    if s > h:
        raise ValueError("")
    return (s / h) if h > 0 else None


def prr_per_run(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    n_rows = 0
    for r in rows:
        n_rows += 1
        ratio = _prr_ratio(r)
        if ratio is not None:
            ratios.append(ratio)
    return {"estimand": "per-run (mean-of-ratios); §14.5②  2026-07-23",
            "n_runs": n_rows, "n_runs_with_hit": len(ratios),
            "value": (sum(ratios) / len(ratios)) if ratios else None}


def prr_per_hit_pooled(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    hits = resolved = n_rows = 0
    for r in rows:
        n_rows += 1
        h = _require_nonneg_int(r["prr_hits"], "prr_hits")
        s = _require_nonneg_int(r["prr_resolved"], "prr_resolved")
        if s > h:
            raise ValueError("")
        hits += h
        resolved += s
    return {"estimand": "per-hit (ratio-of-sums)—— §14.5② estimand",
            "n_runs": n_rows, "n_hits": hits,
            "value": (resolved / hits) if hits else None}


def _calls_of(row: dict[str, Any]) -> int:
    return _require_nonneg_int(row["n_tool_calls"], f"n_tool_calls{row.get('run_id')!r}")


def _total_tokens_of(row: dict[str, Any]) -> int:
    return (_require_nonneg_int(row["input_tokens"], "input_tokens")
            + _require_nonneg_int(row["output_tokens"], "output_tokens"))


def np_waste_calls(injected_rows: list[dict[str, Any]],
                   np_rows: list[dict[str, Any]]) -> dict[str, Any]:
    np_calls: dict[tuple[str, str, str], int] = {}
    for r in np_rows:
        if r.get("truth") != "NP" or r.get("label") != "np":
            raise ValueError("")
        if is_replicate_row(r):
            raise ValueError("")
        k = (r["batch"], r["cat"], r["tid"])
        if k in np_calls:
            raise ValueError("")
        np_calls[k] = _calls_of(r)

    per: list[dict[str, Any]] = []
    n_unpaired = 0
    seen: set[tuple[str, str, str, str]] = set()
    for r in injected_rows:
        if r.get("truth") != "NP" or r.get("label") not in ("transient", "persistent"):
            raise ValueError("")
        ident = (r["batch"], r["cat"], r["tid"], r["label"])
        if ident in seen:
            raise ValueError("")
        seen.add(ident)
        k = (r["batch"], r["cat"], r["tid"])
        if k not in np_calls:
            n_unpaired += 1
            continue
        inj = _calls_of(r)
        per.append({"run_id": r.get("run_id"), "cell_id": r.get("cell_id"),
                    "batch": r["batch"], "cat": r["cat"], "tid": r["tid"],
                    "label": r["label"], "injected_calls": inj, "np_calls": np_calls[k],
                    "np_waste_calls": inj - np_calls[k]})
    return {"per_run": per, "n_pairs": len(per), "n_unpaired": n_unpaired}


RISK_CLASSES = ("write", "read")


def victim_risk_class(cat: str, tid: str, sibling: str, *,
                      loader: Any = None, is_action_fn: Any = None) -> str:
    if sibling not in set(NP_SIBLING_OF_LABEL.values()):
        raise ValueError("")
    if loader is None or is_action_fn is None:
        from dar.taskmodel import is_action, load_task
        loader = loader or load_task
        is_action_fn = is_action_fn or is_action
    victims = perturbed_victims(loader(cat, tid, sibling))
    if not victims:
        raise ValueError("")
    return "write" if any(is_action_fn(v) for v in sorted(victims)) else "read"


def np_waste_by_risk_class(injected_rows: list[dict[str, Any]],
                           np_rows: list[dict[str, Any]],
                           victim_class_by_task: dict[tuple[str, str, str], str]
                           ) -> dict[str, Any]:
    base = np_waste_calls(injected_rows, np_rows)
    per: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, Any]] = {
        c: {"n_pairs": 0, "n_tasks": 0, "n_shared_baseline_tasks": 0,
            "total_np_waste_calls": None, "mean_np_waste_calls": None}
        for c in RISK_CLASSES}
    totals: dict[str, int] = {c: 0 for c in RISK_CLASSES}
    pairs_by_task: dict[str, dict[tuple[str, str, str], int]] = {c: {} for c in RISK_CLASSES}
    for row in base["per_run"]:
        sibling = NP_SIBLING_OF_LABEL[row["label"]]
        k = (row["cat"], row["tid"], sibling)
        if k not in victim_class_by_task:
            raise ValueError("")
        cls = victim_class_by_task[k]
        if cls not in RISK_CLASSES:
            raise ValueError("")
        per.append({**row, "sibling": sibling, "risk_class": cls})
        by_class[cls]["n_pairs"] += 1
        totals[cls] += row["np_waste_calls"]
        bk = (row["batch"], row["cat"], row["tid"])
        pairs_by_task[cls][bk] = pairs_by_task[cls].get(bk, 0) + 1
    for c in RISK_CLASSES:
        n = by_class[c]["n_pairs"]
        by_class[c]["n_tasks"] = len(pairs_by_task[c])
        by_class[c]["n_shared_baseline_tasks"] = sum(
            1 for v in pairs_by_task[c].values() if v > 1)
        if n:
            by_class[c]["total_np_waste_calls"] = totals[c]
            by_class[c]["mean_np_waste_calls"] = totals[c] / n
    return {"per_run": per, "by_class": by_class,
            "n_pairs": base["n_pairs"], "n_unpaired": base["n_unpaired"],
            "n_tasks": len({(p["batch"], p["cat"], p["tid"]) for p in per})}


COST_ONLY_CAT = "c1"
COST_ONLY_TRUTHS = ("P2", "P4")


def avoided_waste(dispatch_rows: list[dict[str, Any]],
                  no_dispatch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    nd: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in no_dispatch_rows:
        if (r.get("cat") != COST_ONLY_CAT or r.get("truth") not in COST_ONLY_TRUTHS
                or r.get("label") != "np"):
            raise ValueError(
                "")
        k = (r["batch"], r["truth"], r["tid"])
        if k in nd:
            raise ValueError("")
        nd[k] = r

    per: list[dict[str, Any]] = []
    n_unpaired = 0
    seen: set[tuple[str, str, str]] = set()
    for r in dispatch_rows:
        if (r.get("cat") != COST_ONLY_CAT or r.get("truth") not in COST_ONLY_TRUTHS
                or r.get("label") != "persistent"):
            raise ValueError(
                "")
        k = (r["batch"], r["truth"], r["tid"])
        if k in seen:
            raise ValueError("")
        seen.add(k)
        if k not in nd:
            n_unpaired += 1
            continue
        b = nd[k]
        per.append({
            "batch": r["batch"], "truth": r["truth"], "cat": r["cat"], "tid": r["tid"],
            "no_dispatch_calls": _calls_of(b), "dispatch_calls": _calls_of(r),
            "avoided_calls": _calls_of(b) - _calls_of(r),
            "no_dispatch_total_tokens": _total_tokens_of(b),
            "dispatch_total_tokens": _total_tokens_of(r),
            "avoided_total_tokens": _total_tokens_of(b) - _total_tokens_of(r),
        })
    return {"per_task": per, "n_pairs": len(per), "n_unpaired": n_unpaired}


def build_p0_pools(rows: Iterable[dict[str, Any]], *, batch: str = "main") -> dict[str, Any]:
    sel: list[dict[str, Any]] = []
    n_rep_all = 0
    n_rep_in_batch = 0
    for r in rows:
        if r.get("truth") != "NP" or r.get("label") != "np":
            continue
        if is_replicate_row(r):
            n_rep_all += 1
            if r.get("batch") == batch:
                n_rep_in_batch += 1
            continue
        if r.get("batch") != batch:
            continue
        sel.append(r)
    if not sel:
        raise ValueError(
            "")
    ga_pool: list[tuple[str, str]] = []
    judge_pool: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in sel:
        k = (r["cat"], r["tid"])
        if k in seen:
            raise ValueError("")
        seen.add(k)
        if main_estimator(r["ga_verdict"]):
            ga_pool.append(k)
        jp = r["judge_pass"]
        if not isinstance(jp, bool):
            raise ValueError("")
        if jp:
            judge_pool.append(k)
    return {"batch": batch, "n_runs": len(sel),
            "n_replicates_seen_all_batches": n_rep_all,
            "n_replicates_excluded_in_batch": n_rep_in_batch,
            "ga_p0_pool": sorted(ga_pool), "judge_p0_pool": sorted(judge_pool)}


MATERIAL_AXES = ("transient", "persistent")


def material_axis_for_row(manifest_row: dict[str, Any], *,
                          pairing_axis: str | None = None) -> str:
    for k in ("truth", "label"):
        if k not in manifest_row:
            raise ValueError("")
    truth, label = manifest_row["truth"], manifest_row["label"]
    if truth in AXIS_OF:
        axis = AXIS_OF[truth]
    elif truth == "NP":
        if label in MATERIAL_AXES:
            axis = label
        elif label == "np":
            if pairing_axis is None:
                raise ValueError(
                    "")
            if pairing_axis not in MATERIAL_AXES:
                raise ValueError(f"pairing_axis {pairing_axis!r} ∉ {MATERIAL_AXES}")
            return pairing_axis
        else:
            raise ValueError("")
    else:
        raise ValueError("")
    if pairing_axis is not None:
        raise ValueError(
            "")
    return axis


REGISTER_RAW_DISPOSITION_Z = "exclude_cell_if_victim_is_availability"

_CELL_DISPOSITION_VALUES = ("keep", "exclude_cell")

DISPOSITION_VOCAB = ("clean", "keep_with_note") + EXCLUDE_DISPOSITIONS


def defect_disposition_for_axis(entry: dict[str, Any] | None, axis: str) -> str:
    if axis not in MATERIAL_AXES:
        raise ValueError("")
    disp = (entry or {}).get("disposition", "clean")
    if disp == REGISTER_RAW_DISPOSITION_Z:
        cell = (entry or {}).get("cell_disposition") or {}
        if axis not in cell:
            raise ValueError(
                "")
        if cell[axis] not in _CELL_DISPOSITION_VALUES:
            raise ValueError(
                "")
        return "exclude_cell" if cell[axis] == "exclude_cell" else "keep_with_note"
    if disp not in DISPOSITION_VOCAB:
        raise ValueError("")
    return disp


def cell_level_sieve(task_keys: Iterable[tuple[str, str]], *,
                     ga_p0_pool: Iterable[tuple[str, str]],
                     material_by_task: dict[tuple[str, str], bool],
                     disposition_by_task: dict[tuple[str, str], str]) -> set[tuple[str, str]]:
    pool = set(ga_p0_pool)
    out: set[tuple[str, str]] = set()
    for k in task_keys:
        if k not in material_by_task:
            raise ValueError("")
        mv = material_by_task[k]
        if not isinstance(mv, bool):
            raise ValueError(
                "")
        disp = disposition_by_task.get(k, "clean")
        if disp not in DISPOSITION_VOCAB:
            hint = (f" gt_defect_register ****—— "
                    f"`defect_disposition_for_axis(entry, axis)`"
                    f" 'exclude_cell' ACC-2"
                    if disp == REGISTER_RAW_DISPOSITION_Z else
                    "[LE1]")
            raise ValueError("")
        if mv is not True:
            continue
        if k not in pool:
            continue
        if disp in EXCLUDE_DISPOSITIONS:
            continue
        out.add(k)
    return out


def amortized_wall_clock_s(span_s: float, n_runs: int) -> float:
    if isinstance(span_s, bool) or not isinstance(span_s, (int, float)) or span_s < 0:
        raise ValueError("")
    n = _require_int(n_runs, "n_runs")
    if n <= 0:
        raise ValueError("")
    return float(span_s) / n


def call_log_totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    n_calls = prompt = completion = n_missing = 0
    for rec in records:
        n_calls += 1
        p, c = rec.get("prompt_tokens"), rec.get("completion_tokens")
        if isinstance(p, int) and not isinstance(p, bool) \
                and isinstance(c, int) and not isinstance(c, bool):
            prompt += p
            completion += c
        else:
            n_missing += 1
    return {"n_calls": n_calls, "prompt_tokens": prompt,
            "completion_tokens": completion, "n_missing_usage": n_missing}
