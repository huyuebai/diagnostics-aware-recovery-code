#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b6_subset as SUB
from b6_products import ProductEmitter
from dar.accounting import LedgerA
from dar.diagnoser import ml as ML
from dar.diagnoser.oracle import OracleDiagnoser
from dar.diagnoser.rule import RuleDiagnoser
from dar.labels import Label, true_type_from_mode

AXIS_ORDER: tuple[str, ...] = tuple(l.value for l in Label)

DENOMINATOR_NOTE = " = "
DISCLOSURE_FACE_NOTE = " head-to-head "
NP_ROW_NOTE = ("NP NP×transient  P1 victim NP×persistent  P2 "
               "victim §2ter / DECISIONS §13#5(b)"
               "** ≠ **——B6  run"
               "B5  μ̂(NP,·) ")
FRAME_NOTE = ("n_units = 500n_diagnoses ≤ 600——"
              " NP ")
SELF_DIAGNOSIS_NOTE = (" LLM  = qwen3-8b**** "
                       "⇒ ")
RULE_PERSISTENT_ZERO_NOTE = (
    " persistent  0 victim "
    "`fa/none`  `max_consecutive_same_tool_errors` ∈ {0,1} ")

TOOL_SCHEMA_NOTES: tuple[str, ...] = (
    "① live dispatch B3–B5  tool_schemas "
    "`dispatch.py`  `DiagnosisInput.from_messages` ⇒ B6 **"
    "** live dispatch ",
    "② vendor  `_build_tool_definitions`  `set()` "
    " PYTHONHASHSEED⇒ B6 ****"
    "** agent **",
    "③  + vendor  YAML  `if tool_def:` "
    "** run **",
)

_TOOL_SCHEMA_NOTE_DIAGNOSERS = frozenset({"llm", "api"})

_A8_FACES = {("rule", "head_to_head")}


def empty_matrix() -> dict[str, dict[str, int]]:
    return {r: {c: 0 for c in AXIS_ORDER} for r in AXIS_ORDER}


def rownorm(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in AXIS_ORDER:
        total = sum(counts[r].values())
        if total == 0:
            raise SystemExit(
                "")
        out[r] = {c: counts[r][c] / total for c in AXIS_ORDER}
    return out


def assert_matrix_wellformed(counts: dict[str, dict[str, int]],
                             norm: dict[str, dict[str, float]]) -> None:
    for r in AXIS_ORDER:
        total = 0
        for c in AXIS_ORDER:
            v = counts[r][c]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise SystemExit("")
            total += v
        if total <= 0:
            raise SystemExit("")
        s = sum(norm[r].values())
        if abs(s - 1.0) > 1e-9:
            raise SystemExit("")


def assert_oracle_identity(counts: dict[str, dict[str, int]],
                           norm: dict[str, dict[str, float]]) -> None:
    for r in AXIS_ORDER:
        for c in AXIS_ORDER:
            if r == c:
                continue
            if counts[r][c] != 0:
                raise SystemExit(
                    "")
            if norm[r][c] != 0.0:
                raise SystemExit("")
        if norm[r][r] != 1.0:
            raise SystemExit("")


def matrix_payload(diagnoser: str, face: str, counts: dict[str, dict[str, int]], *,
                    n_unparseable: int, unparseable_reasons: dict[str, int],
                    excluded: dict[str, int], unparseable_by_cell: dict[str, int],
                    extra: dict[str, Any] | None = None,
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    norm = rownorm(counts)
    assert_matrix_wellformed(counts, norm)
    if diagnoser == "oracle":
        assert_oracle_identity(counts, norm)
    by_cell = dict(sorted(unparseable_by_cell.items()))
    if sum(by_cell.values()) != n_unparseable:
        raise SystemExit(
            "")
    n_parsed = sum(sum(row.values()) for row in counts.values())
    common = {
        "diagnoser": diagnoser,
        "face": face,
        "axis_order": list(AXIS_ORDER),
        "row_coordinate": "TRUE_TYPE_TO_LABEL[]",
        "smoothing_alpha": 0,
        "denominator_note": DENOMINATOR_NOTE,
        "frame_note": FRAME_NOTE,
        "np_row_note": NP_ROW_NOTE,
        "n_parsed": n_parsed,
        "n_unparseable": n_unparseable,
        "unparseable_rate": (n_unparseable / (n_parsed + n_unparseable)
                             if (n_parsed + n_unparseable) else 0.0),
        "unparseable_reasons": dict(sorted(unparseable_reasons.items())),
        "unparseable_by_cell": by_cell,
        "excluded_counts": dict(sorted(excluded.items())),
    }
    if face == "disclosure":
        common["face_note"] = DISCLOSURE_FACE_NOTE
    if (diagnoser, face) in _A8_FACES:
        common["rule_persistent_zero_note"] = RULE_PERSISTENT_ZERO_NOTE
    if diagnoser in _TOOL_SCHEMA_NOTE_DIAGNOSERS:
        common["tool_schema_notes"] = list(TOOL_SCHEMA_NOTES)
    if extra:
        common.update(extra)
    return ({**common, "kind": "counts", "matrix": counts},
            {**common, "kind": "rownorm", "matrix": norm})


def predict_rule(units: list[dict[str, Any]]) -> tuple[list[str | None], list[LedgerA]]:
    d = RuleDiagnoser()
    labels: list[str | None] = []
    ledger: list[LedgerA] = []
    for u in units:
        if u["obs"] is None:
            labels.append(None)
            continue
        t0 = time.monotonic()
        out = d.diagnose(u["obs"])
        labels.append(out.label.value)
        ledger.append(LedgerA.b6_row(
            diagnoser="rule", run_id=u["run_id"], axis=u["axis"], usd=0.0,
            basis=out.cost.basis, calls=out.cost.calls,
            latency_ms=(time.monotonic() - t0) * 1000.0))
    return labels, ledger


def predict_oracle(units: list[dict[str, Any]]) -> tuple[list[str | None], list[LedgerA]]:
    labels: list[str | None] = []
    ledger: list[LedgerA] = []
    for u in units:
        if u["obs"] is None:
            labels.append(None)
            continue
        t0 = time.monotonic()
        out = OracleDiagnoser(true_type_from_mode(u["mode"])).diagnose(u["obs"])
        labels.append(out.label.value)
        ledger.append(LedgerA.b6_row(
            diagnoser="oracle", run_id=u["run_id"], axis=u["axis"], usd=0.0,
            basis=out.cost.basis, calls=out.cost.calls,
            latency_ms=(time.monotonic() - t0) * 1000.0))
    return labels, ledger


def ml_out_of_fold(train_units: list[dict[str, Any]], cfg: dict[str, Any],
                   ) -> tuple[dict[str, str], dict[str, Any]]:
    mlc = cfg["ml"]
    usable = [u for u in train_units if u["obs"] is not None]
    if not usable:
        raise SystemExit("")
    by_cat: dict[str, list[str]] = {}
    for u in usable:
        by_cat.setdefault(u["cat"], []).append(u["tid"])
    folds = ML.fold_of_base_ids(by_cat, seed=int(mlc["fold_seed"]), k=int(mlc["k_folds"]))
    packed = [(u["tid"], ML.feature_vector(u["obs"]),
               ML.LABEL_ORDER.index(Label(u["true_label"]))) for u in usable]
    preds, reports = ML.cross_fitted_predictions(
        packed, folds=folds, k_folds=int(mlc["k_folds"]), l2=float(mlc["l2"]),
        lr=float(mlc["learning_rate"]), max_epochs=int(mlc["max_epochs"]),
        tol=float(mlc["tol"]), regularize_bias=bool(mlc["regularize_bias"]))
    label_of = {u["run_id"]: ML.LABEL_ORDER[p].value for u, p in zip(usable, preds)}
    report = {"n_train_units": len(usable), "k_folds": int(mlc["k_folds"]),
              "feature_names": list(ML.FEATURE_NAMES), "label_order":
                  [l.value for l in ML.LABEL_ORDER],
              "fold_sizes": {str(f): sum(1 for u in usable if folds[u["tid"]] == f)
                             for f in range(int(mlc["k_folds"]))},
              "per_fold": {str(f): {"n_samples": r.n_samples, "epochs_ran": r.epochs_ran,
                                    "initial_objective": r.initial_objective,
                                    "final_objective": r.final_objective,
                                    "stopped_by_tolerance": r.stopped_by_tolerance,
                                    "max_abs_weight": r.max_abs_weight}
                           for f, r in sorted(reports.items())}}
    return label_of, report


def predict_ml(units: list[dict[str, Any]], label_of: dict[str, str],
               ) -> tuple[list[str | None], list[LedgerA]]:
    labels: list[str | None] = []
    ledger: list[LedgerA] = []
    for u in units:
        if u["obs"] is None:
            labels.append(None)
            continue
        rid = u["run_id"]
        if rid not in label_of:
            raise SystemExit("")
        labels.append(label_of[rid])
        ledger.append(LedgerA.b6_row(
            diagnoser="ml", run_id=rid, axis=u["axis"], usd=0.0,
            basis=ML.ZERO_BASIS, calls=0))
    return labels, ledger


def cell_key(unit: dict[str, Any]) -> str:
    return f"{unit['cat']}|{unit['mode']}|{unit['axis']}"


def tally(units: list[dict[str, Any]], labels: list[str | None],
          ) -> tuple[dict[str, dict[str, int]], dict[str, int], int, dict[str, int]]:
    counts = empty_matrix()
    excluded: dict[str, int] = {}
    n_unparseable = 0
    by_cell: dict[str, int] = {}
    for u, lab in zip(units, labels):
        if u["obs"] is None:
            excluded[u["reason"]] = excluded.get(u["reason"], 0) + 1
            continue
        if lab is None:
            n_unparseable += 1
            k = cell_key(u)
            by_cell[k] = by_cell.get(k, 0) + 1
            continue
        counts[u["true_label"]][lab] += 1
    return counts, excluded, n_unparseable, dict(sorted(by_cell.items()))


def run(corpus: Path, out: Path, cfg: dict[str, Any], *, diagnosers: Iterable[str],
        emitter: ProductEmitter, full_scan: bool = True) -> dict[str, Any]:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    main_arm = str(manifest["sampling"]["main_arm"])
    cfg_arm = cfg.get("measurement", {}).get("main_arm")
    if cfg_arm is not None and str(cfg_arm) != main_arm:
        raise SystemExit(
            "")
    sub_units = SUB.enumerate_units(corpus, main_arm, manifest["subset_ids"])
    if len(sub_units) != manifest["n_diagnoses"]:
        raise SystemExit("")

    ledger_rows: list[LedgerA] = []
    summary: dict[str, Any] = {"tree_id": manifest["tree_id"], "faces": {},
                               "main_arm": main_arm, "full_scan": bool(full_scan)}

    train_units: list[dict[str, Any]] = []
    ml_labels: dict[str, str] = {}
    ml_report: dict[str, Any] = {}
    if "ml" in diagnosers:
        for arm in SUB.MAIN_BATCH_ARMS:
            train_units.extend(SUB.enumerate_units(corpus, arm, None,
                                                   with_task_description=False,
                                                   with_tool_schemas=False))
        ml_labels, ml_report = ml_out_of_fold(train_units, cfg)

    predictors: dict[str, Callable[[list[dict[str, Any]]],
                                   tuple[list[str | None], list[LedgerA]]]] = {
        "rule": predict_rule,
        "oracle": predict_oracle,
        "ml": lambda us: predict_ml(us, ml_labels),
    }

    for name in diagnosers:
        if name not in predictors:
            raise SystemExit("")
        labels, ledger = predictors[name](sub_units)
        ledger_rows.extend(ledger)
        counts, excluded, n_unparse, unparse_cells = tally(sub_units, labels)
        extra = {"ml_training": ml_report} if name == "ml" else None
        c_doc, n_doc = matrix_payload(name, "head_to_head", counts, n_unparseable=n_unparse,
                                      unparseable_reasons={}, excluded=excluded,
                                      unparseable_by_cell=unparse_cells, extra=extra)
        emitter.write_json(out / "matrix" / f"{name}.counts.json", c_doc)
        emitter.write_json(out / "matrix" / f"{name}.rownorm.json", n_doc)
        summary["faces"].setdefault("head_to_head", {})[name] = {
            "counts": counts, "excluded": excluded}

        if full_scan and name in ("rule", "ml"):
            per_arm: dict[str, Any] = {}
            for arm in SUB.MAIN_BATCH_ARMS:
                arm_units = ([u for u in train_units if u["arm"] == arm] if train_units
                             else SUB.enumerate_units(corpus, arm, None,
                                                      with_task_description=False,
                                                      with_tool_schemas=False))
                a_labels, _ = predictors[name](arm_units)
                a_counts, a_excl, a_unparse, a_cells = tally(arm_units, a_labels)
                per_arm[arm] = {"counts": a_counts, "excluded": a_excl,
                                "n_unparseable": a_unparse, "unparseable_by_cell": a_cells}
            merged = empty_matrix()
            merged_excl: dict[str, int] = {}
            merged_cells: dict[str, int] = {}
            merged_unparse = sum(b["n_unparseable"] for b in per_arm.values())
            for arm, blob in per_arm.items():
                for r in AXIS_ORDER:
                    for c in AXIS_ORDER:
                        merged[r][c] += blob["counts"][r][c]
                for k, v in blob["excluded"].items():
                    merged_excl[k] = merged_excl.get(k, 0) + v
                for k, v in blob["unparseable_by_cell"].items():
                    merged_cells[k] = merged_cells.get(k, 0) + v
            c_doc, n_doc = matrix_payload(
                name, "disclosure", merged, n_unparseable=merged_unparse, unparseable_reasons={},
                excluded=merged_excl, unparseable_by_cell=merged_cells,
                extra={"per_arm_counts": {a: b["counts"] for a, b in per_arm.items()},
                       "per_arm_excluded": {a: b["excluded"] for a, b in per_arm.items()},
                       "per_arm_unparseable_by_cell": {a: b["unparseable_by_cell"]
                                                       for a, b in per_arm.items()}})
            emitter.write_json(out / "matrix" / f"{name}.disclosure.counts.json", c_doc)
            emitter.write_json(out / "matrix" / f"{name}.disclosure.rownorm.json", n_doc)
            summary["faces"].setdefault("disclosure", {})[name] = {"counts": merged}

    if not ledger_rows:
        raise SystemExit("")
    for r in ledger_rows:
        if not (r.diagnoser.strip() and r.run_id.strip() and r.axis.strip()):
            raise SystemExit(
                "")
    emitter.write_jsonl(out / "ledger_a.jsonl",
                        [{"diagnoser": r.diagnoser, "run_id": r.run_id, "axis": r.axis,
                          "prompt_tokens": r.prompt_tokens,
                          "completion_tokens": r.completion_tokens, "calls": r.calls,
                          "latency_ms": r.latency_ms, "usd": r.usd, "basis": r.basis}
                         for r in ledger_rows])
    summary["n_ledger_rows"] = len(ledger_rows)
    summary["self_diagnosis_note"] = SELF_DIAGNOSIS_NOTE
    summary["np_row_note"] = NP_ROW_NOTE
    summary["frame_note"] = FRAME_NOTE
    summary["denominator_note"] = DENOMINATOR_NOTE
    summary["disclosure_face_note"] = DISCLOSURE_FACE_NOTE
    summary["rule_persistent_zero_note"] = RULE_PERSISTENT_ZERO_NOTE
    summary["rule_persistent_zero_note_faces"] = sorted(f"{d}/{f}" for d, f in _A8_FACES)
    emitter.write_json(out / "measure_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diagnosers", default="rule,oracle,ml")
    ap.add_argument("--no-full-scan", action="store_true")
    args = ap.parse_args(argv)

    cfg = SUB.load_config()
    corpus = Path(args.data) if args.data else (LAB / cfg["measurement"]["corpus_root"])
    if not corpus.is_absolute():
        corpus = LAB / corpus
    out = Path(args.out)
    if not out.is_absolute():
        out = LAB / out

    emitter = ProductEmitter(out.name, lab_stamp=SUB.lab_stamp_fingerprint())
    summary = run(corpus, out, cfg, diagnosers=[d for d in args.diagnosers.split(",") if d],
                  emitter=emitter, full_scan=not args.no_full_scan)
    emitter.emit()
    for face, blob in summary["faces"].items():
        for name, b in blob.items():
            print(f"[measure] {face}/{name}: "
                  + json.dumps(b["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
