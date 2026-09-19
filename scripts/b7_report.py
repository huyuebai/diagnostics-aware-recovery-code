from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "src") not in sys.path:
    sys.path.insert(0, str(LAB / "src"))

from dar import b5_stats
from dar.b5_stats import B5_BOOTSTRAP_SEED
from dar import headroom
from dar.analysis import cost as COST
from dar.analysis import pareto as PAR
from dar.analysis import pstar as PS
from dar.analysis import synthesis as SYN
from dar.analysis import vladder as VL
from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

B7_BOOTSTRAP_SEED = B5_BOOTSTRAP_SEED

PRODUCT_FILES = (
    "identity_selfcheck.json", "synthesis.json", "pstar.json", "cost_lambda.json",
    "pi_np_scan.json", "pareto.json", "np_waste.json", "manifest.json",
)

SUPPORT_BATCH = "main"
EXCLUDED_CAT = "c1"

B5_MODEL = "qwen3-8b"

COUNTS_FILES = ("rule", "ml", "llm", "api")

ORACLE_NOT_A_CANDIDATE = (
    "oracle.counts.json lab/docs/b6_findings.md §8"
)

REPLAY_SIEVE_NOTE = ""

CONSTRUCTIVE_IDENTITY_STATEMENTS = (
    "μ̂(NP,np) ≡ 1.000 [LM13]B5 "
    "prereg §2.1",
    "r(NP) = μ̂(NP,ℓ*) − μ̂(NP,np)  ℓ*(NP) = np  0prereg §2.1",
    "p*(NP) = 1.0  μ̂ +  Tp<1——+"
    "[LM13]NP prereg §2.2bis",
)

C1_DISCLOSURE_TAG = "· headline p*"
PLAIN_NARROW_READING_TAG = " headline  p*"

TWO_LAYER_SCOPE_STATEMENT = (
    "two_layer —— =  q  referent ⑭-G V3A-1 §6 C-2 "
    "(A)  0"
    "prereg §2.5 "
)

LAMBDA_COST_COLUMN = "calls"

R_UNITS: dict[TrueType, str] = {
    TrueType.P1: "r(P1)", TrueType.P2: "r(P2)",
    TrueType.P3: "r(P3)", TrueType.P4: "r(P4)",
}
D_UNITS: dict[TrueType, dict[Label, str]] = {
    TrueType.P1: {Label.PERSISTENT: "d(P1,persistent)"},
    TrueType.P2: {Label.TRANSIENT: "d(P2,transient)"},
    TrueType.P3: {Label.PERSISTENT: "d(P3,persistent)"},
    TrueType.P4: {Label.TRANSIENT: "d(P4,transient)"},
    TrueType.NP: {Label.TRANSIENT: "d(NP,transient)",
                  Label.PERSISTENT: "d(NP,persistent)"},
}


def load_b5_report():
    path = LAB / "scripts" / "b5_report.py"
    spec = importlib.util.spec_from_file_location("b5_report_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deserialize_pools(pool_strings: list[str]) -> set[tuple[str, str]]:
    pools = {tuple(s.split("|", 1)) for s in pool_strings}
    if not pools:
        raise ValueError("")
    for t in pools:
        if len(t) != 2 or not t[0] or not t[1]:
            raise ValueError(f"pool element {t!r} is not a (cat, tid) pair")
    return pools


def filter_c1(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if r["cat"] != EXCLUDED_CAT]


def retake_unit_ci(diffs: list[Mapping[str, Any]], *, exclude_c1: bool) -> dict[str, Any]:
    vals = [float(x["diff"]) for x in diffs
            if not exclude_c1 or x["cat"] != EXCLUDED_CAT]
    return _frozen_bootstrap(vals)


def compose_d_bar(d_by_label: Mapping[Label, float], w: Mapping[Label, float],
                  star: Label) -> float:
    return sum(w[lab] * d_by_label[lab] for lab in d_by_label if lab != star)


def deserialize_pools_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {batch: {k: (deserialize_pools(v) if isinstance(v, list) else v)
                    for k, v in block.items()}
            for batch, block in payload.items()}


def _frozen_bootstrap(vals: list[float]) -> dict[str, Any]:
    return headroom.bootstrap_ci(vals, alpha=0.10, n_boot=10000, seed=B7_BOOTSTRAP_SEED)


def c1_only_unit_ci(diffs: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _frozen_bootstrap([float(x["diff"]) for x in diffs if x["cat"] == EXCLUDED_CAT])


def mu_lookup(mu_rows: list[Mapping[str, Any]], batch: str) -> dict[TrueType, dict[Label, float]]:
    table: dict[TrueType, dict[Label, float]] = {}
    seen: dict[tuple[str, str], float] = {}
    for m in mu_rows:
        if m["batch"] != batch:
            continue
        key = (m["truth"], m["label"])
        if key in seen:
            if key == ("NP", "np"):
                if m["mu_ga"] != seen[key]:
                    raise SystemExit(
                        "")
                continue
            raise SystemExit("")
        seen[key] = m["mu_ga"]
        table.setdefault(TrueType(m["truth"]), {})[Label(m["label"])] = float(m["mu_ga"])
    missing = [(t.value, l.value) for t in TrueType for l in Label
               if l not in table.get(t, {})]
    if missing:
        raise SystemExit("")
    return table


def gridpoint_dict(gp: "SYN.EmpiricalGridPoint") -> dict[str, Any]:
    return {"value": gp.value, "reason": gp.reason,
            "zero_observed_columns": list(gp.zero_columns),
            "warnings": list(gp.warnings), "raw": dict(gp.raw)}


def aggregate_gridpoints(mu_table: Mapping[TrueType, Mapping[Label, float]],
                         payload: Mapping[str, Any],
                         pi: Mapping[TrueType, float]) -> dict[str, Any]:
    total = sum(pi[t] for t in TrueType)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"pi must sum to 1, got {total}")
    points = {t: SYN.empirical_outcome(mu_table[t], t, payload) for t in TrueType}
    would = sum(pi[t] * (p.value if p.value is not None
                         else p.raw["value_if_zero_weight"])
                for t, p in points.items())
    warnings = sorted({w for p in points.values() for w in p.warnings})
    nones = [(t, p) for t, p in points.items() if p.value is None]
    if nones:
        return {"value": None, "reason": nones[0][1].reason,
                "affected_truths": [t.value for t, _ in nones],
                "warnings": warnings, "raw": {"value_if_zero_weight": would}}
    return {"value": would, "reason": None, "affected_truths": [],
            "warnings": warnings, "raw": {}}


def retake_all_units(analysis: Mapping[str, Any], *, exclude_c1: bool) -> dict[str, dict[str, Any]]:
    return {u["unit"]: retake_unit_ci(u["diffs"], exclude_c1=exclude_c1)
            for u in analysis["paired"]}


def rd_components(retake: Mapping[str, Mapping[str, Any]]) -> tuple[
        dict[TrueType, float], dict[TrueType, dict[Label, float]]]:
    r_by = {t: float(retake[u]["mean"]) for t, u in R_UNITS.items()}
    r_by[TrueType.NP] = 0.0
    d_by: dict[TrueType, dict[Label, float]] = {}
    for t, units in D_UNITS.items():
        d_by[t] = {lab: float(retake[u]["mean"]) for lab, u in units.items()}
        d_by[t][Label.NP] = 0.0
    return r_by, d_by


def headline_pstar_point(true_type: TrueType, r: float,
                         d_by_label: Mapping[Label, float],
                         payload: Mapping[str, Any]) -> dict[str, Any]:
    T = SYN.matrix_from_b6_counts(payload)
    zero_cols = SYN.zero_observed_columns(payload)
    star = TRUE_TYPE_TO_LABEL[true_type]
    p, w = SYN.confusion_weights(T, true_type)
    d_bar = compose_d_bar(d_by_label, w, star)
    res = PS.pstar_closed(r, d_bar)
    warnings = SYN.empirical_warnings(payload, true_type)
    reason = SYN.zero_column_guard(zero_cols, true_type)
    if reason is not None:
        return {"value": None, "reason": reason,
                "zero_observed_columns": [c.value for c in zero_cols],
                "warnings": warnings,
                "raw": {"value_if_zero_weight": res.value, "p": p, "r": r,
                        "d_bar": d_bar, "closed_form_reason": res.reason,
                        "out_of_unit_interval": res.out_of_unit_interval}}
    return {"value": res.value, "reason": res.reason,
            "out_of_unit_interval": res.out_of_unit_interval,
            "r": r, "d_bar": d_bar, "p": p,
            "zero_observed_columns": [c.value for c in zero_cols],
            "warnings": warnings, "raw": {}}


def read_lab_stamp() -> str:
    payload = json.loads((LAB / "provenance" / "lab_code_provenance.json")
                         .read_text(encoding="utf-8"))
    return payload["fingerprint_sha256"]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_check_paired(analysis: Mapping[str, Any]) -> dict[str, Any]:
    results = []
    all_ok = True
    for unit in analysis["paired"]:
        frozen_ci = unit["ci"]
        recomputed = retake_unit_ci(unit["diffs"], exclude_c1=False)
        mismatch = [k for k in ("mean", "lo", "hi", "n", "n_pos", "n_neg", "n_zero")
                    if recomputed.get(k) != frozen_ci.get(k)]
        ok = not mismatch
        all_ok = all_ok and ok
        results.append({"unit": unit["unit"], "ok": ok, "mismatch_keys": mismatch})
    return {
        "pass": all_ok,
        "scope": "(B)  bootstrap  diffsdiffs  JSON"
                 " join§2.5  (B) ",
        "units": results,
    }


def identity_check_mu(mu_recomputed: list[Mapping[str, Any]],
                      mu_frozen: list[Mapping[str, Any]]) -> dict[str, Any]:
    ok = mu_recomputed == mu_frozen
    detail: dict[str, Any] = {}
    if not ok:
        detail["n_recomputed"] = len(mu_recomputed)
        detail["n_frozen"] = len(mu_frozen)
    return {
        "pass": ok,
        "scope": "(A)  → join →  → μ̂  mu "
                 " 2 §2.5  (A) ",
        **detail,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_idempotent(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == text:
            return
        raise SystemExit(
            "")
    path.write_text(text, encoding="utf-8")


class ProductWriter:
    def __init__(self, tree_root: Path) -> None:
        self.tree_root = tree_root
        self.entries: list[dict[str, Any]] = []

    def write(self, filename: str, text: str) -> Path:
        if filename not in PRODUCT_FILES:
            raise ValueError(f"{filename!r} is not a frozen §8bis product name "
                             f"(expected one of {PRODUCT_FILES})")
        path = self.tree_root / filename
        _write_idempotent(path, text)
        self.entries.append({"path": str(path.relative_to(self.tree_root.parent)),
                             "sha256": _sha256(text), "bytes": len(text.encode("utf-8"))})
        return path

    def finalize(self, provenance_path: Path, *, tree_id: str, lab_stamp: str,
                 grid_echo: Mapping[str, Any]) -> None:
        if not self.entries:
            raise ValueError("no products written — refusing an empty provenance emission")
        payload: dict[str, Any] = {"schema": "b7_products/v1", "trees": {}}
        if provenance_path.exists():
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        record = {"files": sorted(self.entries, key=lambda e: e["path"]),
                  "lab_stamp": lab_stamp, "grids": dict(grid_echo)}
        existing = payload["trees"].get(tree_id)
        if existing is not None and existing != record:
            raise SystemExit(f"[b7-provenance] REFUSE: tree {tree_id!r} already recorded "
                             f"with different content")
        payload["trees"][tree_id] = record
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(text, encoding="utf-8")


def grid_echo() -> dict[str, Any]:
    from dar.analysis import cost as C
    from dar.analysis import pstar as P
    from dar.analysis import synthesis as S
    return {
        "p_grid": {"expr": "[i / 10_000 for i in range(10_001)]",
                   "n": len(P.p_grid()), "first": P.p_grid()[0], "last": P.p_grid()[-1]},
        "lambda_grid": {"expr": "[10 ** (-2 + 0.25 * i) for i in range(17)]",
                        "n": len(C.lambda_grid()), "first": C.lambda_grid()[0],
                        "last": C.lambda_grid()[-1]},
        "v_grid": {"expr": "[10 ** (-1 + 0.25 * i) for i in range(17)]",
                   "n": len(C.v_grid()), "first": C.v_grid()[0], "last": C.v_grid()[-1]},
        "pi_np_grid": {"expr": "[i / 20 for i in range(20)]",
                       "n": len(S.pi_np_grid()), "first": S.pi_np_grid()[0],
                       "last": S.pi_np_grid()[-1]},
    }


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


_CI_KEYS = ("mean", "lo", "hi", "n", "n_pos", "n_neg", "n_zero")


def run(data: Path, out_tree: Path, b6_tree: Path,
        np_identity: Path | None = None) -> int:
    import stamp_lab
    if stamp_lab.do_check(str(LAB)) != 0:
        raise SystemExit("")
    lab_stamp = read_lab_stamp()

    analysis = json.loads((data / "analysis" / "b5_analysis.json").read_text(encoding="utf-8"))

    mod = load_b5_report()
    v = mod.stage_verify(data)
    vmap, layer_a = v["_vmap"], v["_layer_a"]
    rows = mod.stage_join(data, B5_MODEL, vmap, layer_a)
    pools = deserialize_pools_payload(
        json.loads((data / "analysis" / "p0_pools.json").read_text(encoding="utf-8")))

    mu_full = mod.stage_mu(rows, pools)
    check_a = identity_check_mu(mu_full, analysis["mu"])
    check_b = identity_check_paired(analysis)
    invariance = []
    for u in analysis["paired"]:
        if EXCLUDED_CAT in u["cats"]:
            continue
        rec = retake_unit_ci(u["diffs"], exclude_c1=True)
        mismatch = [k for k in _CI_KEYS if rec.get(k) != u["ci"].get(k)]
        invariance.append({"unit": u["unit"], "ok": not mismatch,
                           "mismatch_keys": mismatch})
    inv_ok = all(x["ok"] for x in invariance)
    gate_pass = bool(check_a["pass"] and check_b["pass"] and inv_ok)

    writer = ProductWriter(out_tree)
    writer.write("identity_selfcheck.json", _dumps({
        "pass": gate_pass,
        "mu_full_support": check_a,
        "paired_bootstrap": check_b,
        "two_layer_scope": TWO_LAYER_SCOPE_STATEMENT,
        "already_excluded_units_invariance": {
            "scope": " c1 cats  c1§2.5 ",
            "units": invariance},
    }))
    if not gate_pass:
        writer.finalize(LAB / "provenance" / "b7_products.json", tree_id=out_tree.name,
                        lab_stamp=lab_stamp, grid_echo=grid_echo())
        raise SystemExit("")

    rows_ex = filter_c1(rows)
    mu_ex_rows = mod.stage_mu(rows_ex, pools)
    mu_table = mu_lookup(mu_ex_rows, SUPPORT_BATCH)
    retake = retake_all_units(analysis, exclude_c1=True)
    r_by, d_by = rd_components(retake)
    holm_ex = b5_stats.holm_report(
        [{"name": u["unit"],
          "diffs_by_task": [x["diff"] for x in u["diffs"] if x["cat"] != EXCLUDED_CAT]}
         for u in analysis["paired"] if u["tier"] == "confirmatory"],
        alpha=0.05, B=10000, seed=B7_BOOTSTRAP_SEED)

    rows_c1 = [r for r in rows
               if r["cat"] == EXCLUDED_CAT and r["batch"] == SUPPORT_BATCH]
    mu_c1_rows = mod.stage_mu(rows_c1, pools)
    c1_units = {u["unit"]: c1_only_unit_ci(u["diffs"]) for u in analysis["paired"]}

    payloads: dict[str, dict[str, Any]] = {}
    inputs_sha: dict[str, str] = {
        "analysis/b5_analysis.json": _sha256_file(data / "analysis" / "b5_analysis.json"),
        "analysis/p0_pools.json": _sha256_file(data / "analysis" / "p0_pools.json"),
    }
    for name in COUNTS_FILES:
        p = b6_tree / "matrix" / f"{name}.counts.json"
        payloads[name] = json.loads(p.read_text(encoding="utf-8"))
        if payloads[name].get("diagnoser") != name:
            raise SystemExit("")
        inputs_sha[f"matrix/{name}.counts.json"] = _sha256_file(p)
    oracle_path = b6_tree / "matrix" / "oracle.counts.json"
    oracle_payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    inputs_sha["matrix/oracle.counts.json"] = _sha256_file(oracle_path)
    T_ora = SYN.matrix_from_b6_counts(oracle_payload)
    n = len(SYN.LABELS)
    if any(T_ora.rows[i][j] != (1.0 if i == j else 0.0)
           for i in range(n) for j in range(n)):
        raise SystemExit("")

    def _jsonl(p: Path) -> list[dict[str, Any]]:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip()]

    rows_a = _jsonl(b6_tree / "ledger_a.jsonl")
    rows_t1 = _jsonl(b6_tree / "ledger_a_t1.jsonl")
    rows_t2 = _jsonl(b6_tree / "ledger_a_t2.jsonl")
    for fname in ("ledger_a.jsonl", "ledger_a_t1.jsonl", "ledger_a_t2.jsonl"):
        inputs_sha[fname] = _sha256_file(b6_tree / fname)
    selections = {
        "rule": COST.select_ledger_a(rows_a, "rule"),
        "ml": COST.select_ledger_a(rows_a, "ml"),
        "oracle": COST.select_ledger_a(rows_a, "oracle"),
        "llm": COST.select_t1_main(rows_t1),
        "api": COST.select_t2_main(rows_t2),
    }
    ledger_a_native: dict[str, dict[str, Any]] = {}
    for name, sel in selections.items():
        COST.assert_basis_nonempty(sel)
        ledger_a_native[name] = {
            "native": COST.native_totals(sel),
            "usd_ledger_sum": sum(float(r["usd"]) for r in sel),
            "basis_values": sorted({r["basis"] for r in sel}),
        }
    per_run = analysis["cost"]["np_waste_by_risk_class"]["per_run"]
    waste = COST.np_waste_per_label(per_run)
    pi_weighted: dict[str, dict[str, Any]] = {}
    for name in COUNTS_FILES + ("oracle",):
        pl = payloads[name] if name in payloads else oracle_payload
        pw = COST.np_waste_pi_weighted(waste, SYN.matrix_from_b6_counts(pl))
        pw["zero_observed_columns"] = [c.value for c in SYN.zero_observed_columns(pl)]
        pw["warnings"] = SYN.empirical_warnings(pl, TrueType.NP)
        pi_weighted[name] = pw

    def _rownorm_dict(T: SYN.TransitionMatrix) -> dict[str, dict[str, float]]:
        return {row.value: {col.value: T.rows[i][j]
                            for j, col in enumerate(SYN.LABELS)}
                for i, row in enumerate(SYN.LABELS)}

    def _np_units_of(pl: Mapping[str, Any]) -> dict[str, int]:
        return {"diagonal": int(pl["matrix"][Label.NP.value][Label.NP.value]),
                "row_total": SYN.np_unit_count(pl)}

    empirical_blocks: dict[str, dict[str, Any]] = {}
    for name in COUNTS_FILES:
        pl = payloads[name]
        T = SYN.matrix_from_b6_counts(pl)
        pw_by_truth = {}
        for t in TrueType:
            p_val, w = SYN.confusion_weights(T, t)
            pw_by_truth[t.value] = {"p": p_val,
                                    "w": {lab.value: w[lab] for lab in SYN.LABELS}}
        empirical_blocks[name] = {
            "T_rownorm": _rownorm_dict(T),
            "zero_observed_columns": [c.value for c in SYN.zero_observed_columns(pl)],
            "np_units": _np_units_of(pl),
            "p_w_by_truth": pw_by_truth,
            "outcome_by_truth": {t.value: gridpoint_dict(
                SYN.empirical_outcome(mu_table[t], t, pl)) for t in TrueType},
            "aggregate_uniform_pi": aggregate_gridpoints(mu_table, pl, SYN.pi_uniform()),
        }
    np_np_values = sorted({m["mu_ga"] for m in analysis["mu"]
                           if m["batch"] == SUPPORT_BATCH
                           and m["truth"] == "NP" and m["label"] == "np"})
    if np_np_values != [1.0]:
        raise SystemExit("")
    writer.write("synthesis.json", _dumps({
        "support_set": f'batch == "{SUPPORT_BATCH}" ∧ cat != "{EXCLUDED_CAT}"§2.5',
        "statements": [CONSTRUCTIVE_IDENTITY_STATEMENTS[0],
                       SYN.NP_ROW_QUALIFIER, SYN.B6_NP_ROW_QUALIFIER,
                       SYN.BD50_T_ROW_WARNING, ORACLE_NOT_A_CANDIDATE],
        "mu_headline_ex_c1": {t.value: {lab.value: val for lab, val in row.items()}
                              for t, row in mu_table.items()},
        "mu_headline_rows_ex_c1": [m for m in mu_ex_rows if m["batch"] == SUPPORT_BATCH],
        "np_np_axes_identity": {"distinct_values": np_np_values,
                                "statement": CONSTRUCTIVE_IDENTITY_STATEMENTS[0]},
        "c1_disclosure_parallel": {"tag": C1_DISCLOSURE_TAG,
                                   "mu_rows": mu_c1_rows,
                                   "unit_ci": c1_units},
        "plain_input_face": {"tag": PLAIN_NARROW_READING_TAG,
                             "mu_rows": [m for m in analysis["mu"]
                                         if m["batch"] == "plain"]},
        "empirical": empirical_blocks,
    }))

    pstar_per_file = {name: {t.value: headline_pstar_point(t, r_by[t], d_by[t],
                                                           payloads[name])
                             for t in TrueType} for name in COUNTS_FILES}
    r_c1_by: dict[TrueType, float | None] = {
        t: c1_units[u]["mean"] for t, u in R_UNITS.items()}
    r_c1_by[TrueType.NP] = 0.0
    d_c1_by: dict[TrueType, dict[Label, float | None]] = {}
    for t, units in D_UNITS.items():
        d_c1_by[t] = {lab: c1_units[u]["mean"] for lab, u in units.items()}
        d_c1_by[t][Label.NP] = 0.0
    c1_pstar: dict[str, dict[str, Any]] = {}
    for name in COUNTS_FILES:
        block = {}
        for t in TrueType:
            r_val = r_c1_by[t]
            d_vals = d_c1_by[t]
            if r_val is None or any(x is None for x in d_vals.values()):
                block[t.value] = {"value": None, "reason": "empty_c1_support",
                                  "input_ci": {u: c1_units[u]
                                               for u in ([R_UNITS[t]] if t in R_UNITS else [])
                                               + list(D_UNITS[t].values())}}
                continue
            block[t.value] = headline_pstar_point(t, r_val, d_vals, payloads[name])
        c1_pstar[name] = block
    writer.write("pstar.json", _dumps({
        "statements": list(CONSTRUCTIVE_IDENTITY_STATEMENTS)
        + [SYN.NP_ROW_QUALIFIER, SYN.BD50_T_ROW_WARNING],
        "components_ex_c1": {
            "r": {t.value: r_by[t] for t in TrueType},
            "d": {t.value: {lab.value: val for lab, val in d_by[t].items()}
                  for t in TrueType},
            "unit_ci": retake,
        },
        "holm_ex_c1": holm_ex,
        "per_file": pstar_per_file,
        "c1_disclosure_parallel": {"tag": C1_DISCLOSURE_TAG, "per_file": c1_pstar},
    }))

    writer.write("cost_lambda.json", _dumps({
        "statements": [PAR.SCOPE_STATEMENT, PAR.SCOPE_STATEMENT_COST_BOUND_NOTE,
                       COST.ML_TRAINING_COST_DISCLOSURE, COST.LATENCY_CALIBER_NOTE,
                       ORACLE_NOT_A_CANDIDATE],
        "diagnoser_cost_column": {
            "column": LAMBDA_COST_COLUMN,
            "basis": "§8ter V  = #12—— A  "
                     "calls  B  waste  "
                     "ledger_a_native"},
        "lambda_grid_expr": "[10 ** (-2 + 0.25 * i) for i in range(17)]",
        "ledger_a_native": ledger_a_native,
        "per_file": {name: {
            "np_waste_pi_weighted": pi_weighted[name],
            "lambda_scan": COST.scan_lambda(
                float(ledger_a_native[name]["native"][LAMBDA_COST_COLUMN]),
                float(pi_weighted[name]["value"])),
        } for name in COUNTS_FILES + ("oracle",)},
    }))

    scan_per_file = {}
    for name in COUNTS_FILES:
        pts = []
        for g in SYN.pi_np_grid():
            agg = aggregate_gridpoints(mu_table, payloads[name], SYN.pi_for_np(g))
            pts.append({"pi_np": g, "value": agg["value"], "reason": agg["reason"],
                        "raw": agg["raw"]})
        agg_u = aggregate_gridpoints(mu_table, payloads[name], SYN.pi_uniform())
        scan_per_file[name] = {"warnings": agg_u["warnings"], "points": pts}
    writer.write("pi_np_scan.json", _dumps({
        "grid_expr": "[i / 20 for i in range(20)]",
        "headline_pi_np": 0.20,
        "headline_on_grid": True,
        "allocation_statement": " (1 − π_NP)/4"
                                "π_NP = 0.20 ——prereg §8ter"
                                "U14③ 2026-08-14 ",
        "per_file": scan_per_file,
        "np_waste_pi_weighted": pi_weighted,
    }))

    pi_u = SYN.pi_uniform()
    baseline = sum(pi_u[t] * mu_table[t][Label.NP] for t in TrueType)
    delta: dict[str, dict[str, Any]] = {}
    for name in COUNTS_FILES:
        agg = aggregate_gridpoints(mu_table, payloads[name], pi_u)
        if agg["value"] is None:
            delta[name] = {"value": None, "reason": agg["reason"],
                           "raw": {"value_if_zero_weight":
                                   agg["raw"]["value_if_zero_weight"] - baseline}}
        else:
            delta[name] = {"value": agg["value"] - baseline, "reason": None, "raw": {}}
    per_lambda = []
    for lam in COST.lambda_grid():
        pts = [VL.DiagnoserPoint(
            name=name,
            cost=COST.combined_cost(
                float(ledger_a_native[name]["native"][LAMBDA_COST_COLUMN]),
                float(pi_weighted[name]["value"]), lam),
            delta_p=float(delta[name]["value"]))
            for name in COUNTS_FILES if delta[name]["value"] is not None]
        excluded = ([{"file": name, "reason": delta[name]["reason"],
                      "raw": delta[name]["raw"]}
                     for name in COUNTS_FILES if delta[name]["value"] is None]
                    + [{"file": "oracle", "reason": ORACLE_NOT_A_CANDIDATE}])
        per_lambda.append({"lam": lam,
                           "pareto": PAR.pareto_payload(
                               pts, includes_np_rows=True, ledger_a_based=True),
                           "v_scan": PAR.v_scan_payload(pts, includes_np_rows=True),
                           "excluded": excluded})
    writer.write("pareto.json", _dumps({
        "statements": [PAR.SCOPE_STATEMENT, PAR.SCOPE_STATEMENT_COST_BOUND_NOTE,
                       SYN.NP_ROW_QUALIFIER, SYN.BD50_T_ROW_WARNING,
                       COST.ML_TRAINING_COST_DISCLOSURE, COST.LATENCY_CALIBER_NOTE,
                       " order  λ  ladder "
                       "prereg §8quater"],
        "delta_p_baseline_no_dispatch": baseline,
        "delta_p": delta,
        "cost_axis": {"column": LAMBDA_COST_COLUMN,
                      "composition": "combined_cost( A calls, λ· B np_waste "
                                     "π )#12 "},
        "per_lambda": per_lambda,
    }))

    writer.write("np_waste.json", _dumps({
        "estimand": "waste(ℓ) = mean_{τ ∈ NP } np_waste_calls(τ, ℓ)"
                    "§6bis headlineper-label = ",
        "population": " = np  hit —— r/d "
                      " np_wasteμ̂(NP,ℓ) ",
        "statements": [SYN.NP_ROW_QUALIFIER, SYN.NP_SINGLE_RUN_WARNING,
                       "§6bis"],
        "per_label": waste,
        "pi_weighted_per_file": pi_weighted,
        "risk_class_disclosure": COST.np_waste_risk_class_disclosure(per_run),
    }))

    probe_block: dict[str, Any]
    if np_identity is not None and np_identity.exists():
        probe = json.loads(np_identity.read_text(encoding="utf-8"))
        np_runs = {name: probe["files"][name]["np_runs"]
                   for name in COUNTS_FILES if name in probe.get("files", {})}
        detail = {name: {k: blk[k] for k in
                         ("offdiag_units", "distinct_base_runs",
                          "n_np_units_in_grid", "n_np_base_runs_in_grid", "np_runs")}
                  for name, blk in probe.get("files", {}).items()}
        probe_block = {"probe": probe.get("probe"),
                       "source_path": str(np_identity),
                       "source_sha256": _sha256_file(np_identity),
                       "conclusion": probe.get("conclusion"),
                       "files": detail,
                       "np_runs_by_file": np_runs}
    else:
        probe_block = {"value": None,
                       "reason": "np_identity_probe_not_supplied",
                       "np_runs_by_file": None}
    writer.write("manifest.json", _dumps({
        "tree_id": out_tree.name,
        "b5_tree": data.name,
        "b6_tree_id": b6_tree.name,
        "inputs_sha256": inputs_sha,
        "lab_stamp": lab_stamp,
        "grids": grid_echo(),
        "stage_join_model_arg": {"value": B5_MODEL,
                                 "note": "stage_join prereg §2.5  1"
                                         " b5_report main "},
        "support_set": f'batch == "{SUPPORT_BATCH}" ∧ cat != "{EXCLUDED_CAT}"§2.5',
        "plain_narrow_reading": PLAIN_NARROW_READING_TAG,
        "pi_np_allocation": " (1 − π_NP)/4U14③ ",
        "replay_inertia_sieve": {"applied": False, "statement": REPLAY_SIEVE_NOTE},
        "np_row_two_calibers": {
            "np_units_by_file": {name: _np_units_of(payloads[name])
                                 for name in COUNTS_FILES},
            "np_identity_probe": probe_block,
            "caliber_statement": SYN.NP_SINGLE_RUN_WARNING},
        "oracle_role": ORACLE_NOT_A_CANDIDATE,
    }))

    writer.finalize(LAB / "provenance" / "b7_products.json", tree_id=out_tree.name,
                    lab_stamp=lab_stamp, grid_echo=grid_echo())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out-tree", required=True, type=Path)
    ap.add_argument("--b6-tree", required=True, type=Path)
    ap.add_argument("--np-identity", type=Path, default=None)
    args = ap.parse_args()

    def _abs(p: Path | None) -> Path | None:
        return None if p is None else (p if p.is_absolute() else (LAB / p))

    return run(_abs(args.data), _abs(args.out_tree), _abs(args.b6_tree),
               _abs(args.np_identity))


if __name__ == "__main__":
    raise SystemExit(main())
