#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b6_llm_diagnose as DRV
import b6_subset as SUB
import measure_diagnosers as MD
from b6_products import ProductEmitter
from dar.accounting import LedgerA
from dar.diagnoser import llm as LLM

LLM_LOCAL_ZERO_BASIS = ("llm-local=0( vLLM qwen3-8b  $ "
                        " = token/calls/latency_ms)")

BD34_CALIBER_RULE = ("T1b  latency_ms  [BD-34] "
                     "token T1 ——"
                     "[LM24] = b6_preregistration.md §3ter ②")

SERIAL_ROW_KEY_NOTE = (" =  run_pass == 't1b_latency' "
                       "concurrency == 1——T1  b6_preregistration.md "
                       "§3ter U20/U22 ")

TOPLEVEL_LATENCY_SAMPLE_NOTE = "T1b  latency_sample =  latency_subset.latency_sample"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


TREE_FIXED_FILES = ("responses.jsonl", "t1_run_stamp.json", "lab_code_provenance.json",
                    "env_echo/env_echo.json")


def tree_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for name in TREE_FIXED_FILES:
        p = root / name
        if not p.is_file():
            raise SystemExit("")
        out.append(p)
    out.extend(sorted(root.glob("dar-*.out")))
    return out


def load_tree(root: Path) -> dict[str, Any]:
    resp = root / "responses.jsonl"
    if not resp.is_file():
        raise SystemExit("")
    rows = []
    for i, line in enumerate(resp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            raise SystemExit(
                "")
    ids = [r.get("run_id") for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("")
    stamp_p = root / "t1_run_stamp.json"
    try:
        stamp = json.loads(stamp_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(
            "")
    prov = json.loads((root / "lab_code_provenance.json").read_text(encoding="utf-8"))
    fp = prov.get("fingerprint_sha256")
    if not fp:
        raise SystemExit("")
    return {"root": root, "tree_id": root.name, "rows": rows, "stamp": stamp,
            "fingerprint": str(fp)}


def arm_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main = [r for r in rows if r.get("axis") != "extended"]
    ext = [r for r in rows if r.get("axis") == "extended"]
    return main, ext


def split_request_errors(rows: list[dict[str, Any]],
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok = [r for r in rows if r.get("error") is None]
    err = [r for r in rows if r.get("error") is not None]
    return ok, err


def request_error_buckets(err_rows: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    by_cell: dict[str, int] = {}
    for r in err_rows:
        k = MD.cell_key(r)
        by_cell[k] = by_cell.get(k, 0) + 1
    return len(err_rows), dict(sorted(by_cell.items()))


def assert_usage_present(rows: list[dict[str, Any]], where: str) -> None:
    bad = [r["run_id"] for r in rows
           if r.get("error") is None
           and int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0) <= 0]
    if bad:
        raise SystemExit(
            "")


def parse_arm(ok_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    parses: dict[str, Any] = {}
    reasons: dict[str, int] = {}
    for r in ok_rows:
        pr = LLM.parse_response(r.get("text"))
        parses[r["run_id"]] = pr
        if pr.label is None:
            reasons[pr.reason] = reasons.get(pr.reason, 0) + 1
    return parses, dict(sorted(reasons.items()))


def unparseable_by_cell_of_rows(ok_rows: list[dict[str, Any]],
                                parses: dict[str, Any]) -> dict[str, int]:
    by_cell: dict[str, int] = {}
    for r in ok_rows:
        if parses[r["run_id"]].label is None:
            k = MD.cell_key(r)
            by_cell[k] = by_cell.get(k, 0) + 1
    return dict(sorted(by_cell.items()))


def main_arm_matrix(units: list[dict[str, Any]], ok_rows: list[dict[str, Any]],
                    err_rows: list[dict[str, Any]], parses: dict[str, Any],
                    n_diagnoses: int, *, parse_protocol: dict[str, Any],
                    diagnoser: str, extra_notes: dict[str, str],
                    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(units) != n_diagnoses:
        raise SystemExit("")
    obs_ids = {u["run_id"] for u in units if u["obs"] is not None}
    resp_ids = {r["run_id"] for r in ok_rows} | {r["run_id"] for r in err_rows}
    if obs_ids != resp_ids:
        raise SystemExit(
            "")
    err_ids = {r["run_id"] for r in err_rows}
    tally_units: list[dict[str, Any]] = []
    labels: list[str | None] = []
    for u in units:
        if u["obs"] is not None and u["run_id"] in err_ids:
            continue
        tally_units.append(u)
        if u["obs"] is None:
            labels.append(None)
        else:
            pr = parses[u["run_id"]]
            labels.append(None if pr.label is None else pr.label.value)
    counts, excluded, n_unparse, unparse_cells = MD.tally(tally_units, labels)
    n_request_error, request_error_by_cell = request_error_buckets(err_rows)
    n_parsed = sum(sum(row.values()) for row in counts.values())
    n_requests = len(ok_rows) + len(err_rows)
    if n_parsed + n_unparse + n_request_error != n_requests:
        raise SystemExit(
            "")
    reasons: dict[str, int] = {}
    for r in ok_rows:
        pr = parses[r["run_id"]]
        if pr.label is None:
            reasons[pr.reason] = reasons.get(pr.reason, 0) + 1
    extra = {**extra_notes,
             "n_request_error": n_request_error,
             "request_error_by_cell": request_error_by_cell,
             "parse_protocol": dict(parse_protocol)}
    c_doc, n_doc = MD.matrix_payload(
        diagnoser, "head_to_head", counts, n_unparseable=n_unparse,
        unparseable_reasons=dict(sorted(reasons.items())), excluded=excluded,
        unparseable_by_cell=unparse_cells, extra=extra)
    bits = {"counts": counts, "excluded": excluded, "n_parsed": n_parsed,
            "n_unparseable": n_unparse, "unparseable_by_cell": unparse_cells,
            "unparseable_reasons": dict(sorted(reasons.items())),
            "n_request_error": n_request_error,
            "request_error_by_cell": request_error_by_cell}
    return c_doc, n_doc, bits


def serial_latency_by_cell(t1b_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for r in t1b_rows:
        if r.get("concurrency") != 1:
            raise SystemExit(
                "")
    out: dict[str, dict[str, Any]] = {}
    for r in t1b_rows:
        if r.get("error") is not None:
            continue
        k = MD.cell_key(r)
        blob = out.setdefault(k, {"n": 0, "sum_ms": 0.0})
        blob["n"] += 1
        blob["sum_ms"] += float(r["latency_ms"])
    for blob in out.values():
        blob["mean_ms"] = blob["sum_ms"] / blob["n"]
    return {k: out[k] for k in sorted(out)}


def cost_bound_payload(ext_ok: list[dict[str, Any]], ext_err: list[dict[str, Any]],
                       t1b_ext_rows: list[dict[str, Any]], t1b_stamp: dict[str, Any],
                       *, t1_tree_id: str, t1b_tree_id: str) -> dict[str, Any]:
    n_request_error, request_error_by_cell = request_error_buckets(ext_err)
    lat = serial_latency_by_cell(t1b_ext_rows)
    per_cell: dict[str, dict[str, Any]] = {}
    for r in ext_ok + ext_err:
        k = MD.cell_key(r)
        blob = per_cell.setdefault(k, {"n_requests": 0, "n_request_error": 0,
                                       "prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        blob["n_requests"] += 1
        if r.get("error") is not None:
            blob["n_request_error"] += 1
        else:
            blob["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
            blob["completion_tokens"] += int(r.get("completion_tokens") or 0)
            blob["calls"] += int(r.get("attempt") or 1)
    for k, blob in per_cell.items():
        blob["t1b_latency"] = lat.get(k, {"n": 0, "sum_ms": 0.0, "mean_ms": None})
    ls = t1b_stamp["latency_subset"]
    coverage_note = (
        f" {n_request_error} context "
        "")
    return {
        "schema": "b6_cost_bound/v1",
        "diagnoser": "llm",
        "face": "extended_arm",
        "tokens_calls_source": f"T1 {t1_tree_id} concurrency ",
        "wall_clock_source": (f"T1b {t1b_tree_id}run_pass=t1b_latency"
                              f"n={sum(b['n'] for b in lat.values())}"),
        "caliber_rule": BD34_CALIBER_RULE,
        "serial_row_key_note": SERIAL_ROW_KEY_NOTE,
        "n_requests": len(ext_ok) + len(ext_err),
        "n_ok": len(ext_ok),
        "n_request_error": n_request_error,
        "request_error_by_cell": request_error_by_cell,
        "coverage_note": coverage_note,
        "totals": {"prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in ext_ok),
                   "completion_tokens": sum(int(r.get("completion_tokens") or 0)
                                            for r in ext_ok),
                   "calls": sum(int(r.get("attempt") or 1) for r in ext_ok)},
        "per_cell": {k: per_cell[k] for k in sorted(per_cell)},
        "t1b_subset": {"n": int(ls["arms"]["extended"]["n"]),
                       "n_contributing": sum(b["n"] for b in lat.values()),
                       "n_request_error": sum(1 for r in t1b_ext_rows
                                              if r.get("error") is not None),
                       "request_error_run_ids": sorted(
                           r["run_id"] for r in t1b_ext_rows
                           if r.get("error") is not None),
                       "recipe": ls["arms"]["extended"]["recipe"],
                       "seed_domain": ls["arms"]["extended"]["seed_domain"],
                       "subset_seed": int(ls["subset_seed"]),
                       "ids_pointer": f"{t1b_tree_id}/t1_run_stamp.json:latency_subset.arms.extended.ids"},
        "self_diagnosis_note": MD.SELF_DIAGNOSIS_NOTE,
        "tool_schema_notes": list(MD.TOOL_SCHEMA_NOTES),
    }


def ledger_rows_payload(t1_ok_rows: list[dict[str, Any]], *, t1_tree_id: str,
                        ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in sorted(t1_ok_rows, key=lambda x: x["run_id"]):
        led = LedgerA.b6_row(
            diagnoser="llm", run_id=r["run_id"], axis=r["axis"], usd=0.0,
            basis=LLM_LOCAL_ZERO_BASIS,
            prompt_tokens=int(r.get("prompt_tokens") or 0),
            completion_tokens=int(r.get("completion_tokens") or 0),
            calls=int(r.get("attempt") or 1),
            latency_ms=float(r["latency_ms"]))
        if not (led.diagnoser.strip() and led.run_id.strip() and led.axis.strip()):
            raise SystemExit("")
        out.append({"diagnoser": led.diagnoser, "run_id": led.run_id, "axis": led.axis,
                    "prompt_tokens": led.prompt_tokens,
                    "completion_tokens": led.completion_tokens, "calls": led.calls,
                    "latency_ms": led.latency_ms, "usd": led.usd, "basis": led.basis,
                    "run_pass": "t1", "concurrency": int(r["concurrency"]),
                    "attempt": int(r.get("attempt") or 1), "source_tree": t1_tree_id})
    return out


def apply_bd34_ruling(ledger: list[dict[str, Any]], dest: str,
                      t1b_rows: list[dict[str, Any]], *, t1b_tree_id: str,
                      ) -> dict[str, Any] | None:
    ext = [r["run_id"] for r in t1b_rows if r.get("axis") == "extended"]
    if ext:
        raise SystemExit(
            "")
    if dest == "disclosure-only":
        return None
    serial_ok = [r for r in t1b_rows if r.get("error") is None]
    if dest == "ledger-backfill":
        by_id = {row["run_id"]: row for row in ledger}
        for r in serial_ok:
            row = by_id.get(r["run_id"])
            if row is None:
                raise SystemExit("")
            if "serial_latency_ms" in row:
                raise SystemExit("")
            row["serial_latency_ms"] = float(r["latency_ms"])
            row["serial_source"] = f"t1b_latency({t1b_tree_id})"
        return None
    if dest.startswith("new-file:"):
        name = dest.split(":", 1)[1].strip()
        if not name:
            raise SystemExit("")
        return {"path": name,
                "payload": {"schema": "b6_serial_latency/v1",
                            "caliber_rule": BD34_CALIBER_RULE,
                            "serial_row_key_note": SERIAL_ROW_KEY_NOTE,
                            "source_tree": t1b_tree_id,
                            "rows": [{"run_id": r["run_id"], "cell": MD.cell_key(r),
                                      "axis": r["axis"],
                                      "latency_ms": float(r["latency_ms"])}
                                     for r in sorted(serial_ok,
                                                     key=lambda x: x["run_id"])]}}
    raise SystemExit("")


def assert_tree_facts(t1: dict[str, Any], t1b: dict[str, Any],
                      bundle_manifest: dict[str, Any], bundle_sha: str) -> None:
    for t in (t1, t1b):
        got = t["stamp"].get("bundle_sha256")
        if got != bundle_sha:
            raise SystemExit("")
        skel = t["stamp"].get("prompt_skeleton_fingerprint")
        if skel != bundle_manifest.get("prompt_skeleton_fingerprint"):
            raise SystemExit("")
        arm = t["stamp"].get("arm")
        if arm not in (None, "both"):
            raise SystemExit("")
    got_pass = t1b["stamp"].get("run_pass")
    if got_pass != DRV.T1B_PASS:
        raise SystemExit("")
    t1_pass = t1["stamp"].get("run_pass")
    if t1_pass not in (None, "t1"):
        raise SystemExit("")
    t1_ids = {r["run_id"] for r in t1["rows"]}
    t1b_ids = {r["run_id"] for r in t1b["rows"]}
    if not t1b_ids <= t1_ids:
        raise SystemExit("")


def reconcile_t1b_subset(bundle_main_rows: list[dict[str, Any]],
                         bundle_ext_rows: list[dict[str, Any]],
                         t1b_stamp: dict[str, Any]) -> dict[str, bool]:
    ls = t1b_stamp["latency_subset"]
    lat_sample = int(ls["latency_sample"])
    seed = int(ls["subset_seed"])
    cells = DRV.t1b_cells(bundle_main_rows)
    per = -(-lat_sample // len(cells))
    want = {"main": DRV.t1b_stratified_ids(bundle_main_rows, subset_seed=seed,
                                           per_cell=per, domain=DRV.T1B_SEED_PREFIX_MAIN),
            "extended": DRV.t1b_subset_ids(
                bundle_ext_rows, subset_seed=seed,
                n=DRV.t1b_sample_size(len(bundle_main_rows), len(bundle_ext_rows),
                                      lat_sample))}
    out: dict[str, bool] = {}
    for arm in ("main", "extended"):
        got = list(ls["arms"][arm]["ids"])
        out[arm] = (got == want[arm])
        if not out[arm]:
            raise SystemExit("")
    return out


def collect_context(*, bundle: Path, t1_root: Path, t1b_root: Path,
                    canonical_manifest: dict[str, Any],
                    units: list[dict[str, Any]] | None = None,
                    corpus_root: Path | None = None,
                    run_selfproof: bool = True) -> dict[str, Any]:
    bman, bmain, bext = DRV.load_bundle(bundle)
    bundle_sha = sha256_of_file(bundle)
    t1 = load_tree(t1_root)
    t1b = load_tree(t1b_root)
    assert_tree_facts(t1, t1b, bman, bundle_sha)
    ls = t1b["stamp"]["latency_subset"]
    selfproof_rcs: dict[str, Any] = {"t1": "not_run", "t1b": "not_run"}
    if run_selfproof:
        rc1 = DRV.selfproof(t1_root, bundle, "both")
        rc2 = DRV.selfproof(t1b_root, bundle, "both", latency_subset=True,
                            latency_sample=int(ls["latency_sample"]))
        selfproof_rcs = {"t1": rc1, "t1b": rc2}
        if rc1 or rc2:
            raise SystemExit("")
    ids_match = reconcile_t1b_subset(bmain, bext, t1b["stamp"])

    t1_main, t1_ext = arm_split(t1["rows"])
    t1b_main, t1b_ext = arm_split(t1b["rows"])
    for rows, where in ((t1_main, "T1 "), (t1_ext, "T1 "), (t1b["rows"], "T1b")):
        assert_usage_present(rows, where)

    main_ok, main_err = split_request_errors(t1_main)
    ext_ok, ext_err = split_request_errors(t1_ext)
    main_parses, _ = parse_arm(main_ok)
    ext_parses, ext_reasons = parse_arm(ext_ok)

    if units is None:
        if corpus_root is None:
            raise SystemExit("")
        main_arm = str(canonical_manifest["sampling"]["main_arm"])
        units = SUB.enumerate_units(corpus_root, main_arm, canonical_manifest["subset_ids"],
                                    with_task_description=False, with_tool_schemas=False)

    parse_protocol = {
        "implementation": "dar.diagnoser.llm.parse_response",
        "driver_parse_retries": int(t1["stamp"]["driver_parse_retries"]),
        "collect_reparse_retries": 0,
    }
    c_doc, n_doc, main_bits = main_arm_matrix(
        units, main_ok, main_err, main_parses,
        int(canonical_manifest["n_diagnoses"]), parse_protocol=parse_protocol,
        diagnoser="llm", extra_notes={"self_diagnosis_note": MD.SELF_DIAGNOSIS_NOTE})

    cost_bound = cost_bound_payload(ext_ok, ext_err, t1b_ext, t1b["stamp"],
                                    t1_tree_id=t1["tree_id"], t1b_tree_id=t1b["tree_id"])
    ledger = ledger_rows_payload(main_ok + ext_ok, t1_tree_id=t1["tree_id"])
    if len(ledger) != len(t1["rows"]) - len(main_err) - len(ext_err):
        raise SystemExit("")

    n_ext_unparse = sum(1 for r in ext_ok if ext_parses[r["run_id"]].label is None)
    n_ext_req_err, ext_req_cells = request_error_buckets(ext_err)

    serial_main = serial_latency_by_cell(t1b_main)
    serial_t1_prefix = [r for r in t1["rows"] if r.get("concurrency") == 1]

    manifest_t1 = {
        "schema": "b6_manifest_t1/v1",
        "tree_id": canonical_manifest["tree_id"],
        "jobids": {"t1": str(t1["stamp"].get("slurm_job_id") or t1["tree_id"]),
                   "t1b": str(t1b["stamp"].get("slurm_job_id") or t1b["tree_id"])},
        "hpc_trees": {t["tree_id"]: {
            "fingerprint_sha256": t["fingerprint"],
            "files": {str(p.relative_to(t["root"])): {"sha256": sha256_of_file(p),
                                                      "bytes": p.stat().st_size}
                      for p in tree_files(t["root"])}} for t in (t1, t1b)},
        "bundle": {"name": bundle.name, "sha256": bundle_sha,
                   "bundle_lab_stamp": t1b["stamp"].get("bundle_lab_stamp")},
        "reconciliation": {
            "main": {"n_bundle": int(bman["arms"]["main"]["n_requests"]),
                     "n_tree": len(t1_main)},
            "extended": {"n_bundle": int(bman["arms"]["extended"]["n_requests"]),
                         "n_tree": len(t1_ext)},
            "selfproof_rc": selfproof_rcs,
            "t1b": {"n": len(t1b["rows"]),
                    "ids_match_recomputed": ids_match,
                    "run_id_overlap_with_t1": len({r["run_id"] for r in t1b["rows"]}
                                                  & {r["run_id"] for r in t1["rows"]}),
                    "subset_of_t1": True}},
        "parse": {**parse_protocol,
                  "bd31_note": ("[BD-31] 0 §1.4bis③ "
                                " = §1.4bis⑤ "),
                  "prompt_skeleton_fingerprint": bman["prompt_skeleton_fingerprint"],
                  "system_prompt_sha256": bman["system_prompt_sha256"]},
        "counts": {
            "n_diagnoses": int(canonical_manifest["n_diagnoses"]),
            "main_arm": {"n_requests": len(t1_main),
                         "n_truncated": sum(1 for r in main_ok
                                            if r.get("finish_reason") == "length"),
                         "n_parsed": main_bits["n_parsed"],
                         "n_unparseable": main_bits["n_unparseable"],
                         "unparseable_by_cell": main_bits["unparseable_by_cell"],
                         "unparseable_reasons": main_bits["unparseable_reasons"],
                         "n_request_error": main_bits["n_request_error"],
                         "request_error_by_cell": main_bits["request_error_by_cell"],
                         "excluded_counts": main_bits["excluded"]},
            "extended_arm": {"n_requests": len(t1_ext),
                             "n_truncated": sum(1 for r in ext_ok
                                                if r.get("finish_reason") == "length"),
                             "n_parsed": len(ext_ok) - n_ext_unparse,
                             "n_unparseable": n_ext_unparse,
                             "unparseable_by_cell": unparseable_by_cell_of_rows(ext_ok,
                                                                               ext_parses),
                             "unparseable_reasons": ext_reasons,
                             "n_request_error": n_ext_req_err,
                             "request_error_by_cell": ext_req_cells}},
        "latency_calibers": {
            "serial_row_key": SERIAL_ROW_KEY_NOTE,
            "t1_serial_prefix": {"n": len(serial_t1_prefix),
                                 "usable_as_pure_latency": False,
                                 "ruling_pointer": "b6_preregistration.md §3ter U20/U22 "},
            "t1b": {"main": {"n": int(ls["arms"]["main"]["n"]),
                             "recipe": ls["arms"]["main"]["recipe"],
                             "per_cell": int(ls["arms"]["main"].get("per_cell") or 0),
                             "n_cells": int(ls["arms"]["main"].get("n_cells") or 0),
                             "seed_domain": ls["arms"]["main"]["seed_domain"],
                             "n_request_error": sum(1 for r in t1b_main
                                                    if r.get("error") is not None)},
                    "extended": {"n": int(ls["arms"]["extended"]["n"]),
                                 "recipe": ls["arms"]["extended"]["recipe"],
                                 "seed_domain": ls["arms"]["extended"]["seed_domain"],
                                 "n_request_error": sum(1 for r in t1b_ext
                                                        if r.get("error") is not None)}},
            "latency_sample": {"value": int(ls["latency_sample"]),
                               "rate_source": ls.get("rate_source"),
                               "bd35_note": "[BD-35] [BD-31] "},
            "t1b_stamp_toplevel_latency_sample": {
                "value": int(t1b["stamp"].get("latency_sample") or 0),
                "meaning": TOPLEVEL_LATENCY_SAMPLE_NOTE}},
        "bd34_caliber_rule": BD34_CALIBER_RULE,
        "main_arm_serial_latency": {"destination": None,
                                    "n": sum(b["n"] for b in serial_main.values()),
                                    "per_cell": serial_main},
        "self_diagnosis_note": MD.SELF_DIAGNOSIS_NOTE,
        "git_head": SUB.git_head(),
    }
    return {"bundle": bundle, "bundle_sha": bundle_sha, "t1": t1, "t1b": t1b,
            "t1_main_ok": main_ok, "t1_main_err": main_err,
            "t1_ext_ok": ext_ok, "t1_ext_err": ext_err,
            "t1b_main": t1b_main, "t1b_ext": t1b_ext,
            "counts_doc": c_doc, "rownorm_doc": n_doc, "main_bits": main_bits,
            "cost_bound": cost_bound, "ledger": ledger, "manifest_t1": manifest_t1,
            "serial_main": serial_main}


def _dist(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))


def halt_text(ctx: dict[str, Any]) -> str:
    m = ctx["manifest_t1"]
    mc = m["counts"]["main_arm"]
    ec = m["counts"]["extended_arm"]
    lc = m["latency_calibers"]
    serial_t1 = [r for r in ctx["t1"]["rows"] if r.get("concurrency") == 1]
    n_t1b = len(ctx["t1b"]["rows"])
    n_t1b_main = len(ctx["t1b_main"])
    n_t1b_ext = len(ctx["t1b_ext"])
    n_main_ok_serial = sum(1 for r in ctx["t1b_main"] if r.get("error") is None)
    lat_sample = lc["latency_sample"]["value"]
    n_cells = lc["t1b"]["main"]["n_cells"]
    n_main_req = m["reconciliation"]["main"]["n_tree"]
    n_ext_req = m["reconciliation"]["extended"]["n_tree"]
    lines = [
        "========== T1-collect HALT report  ==========",
        " = BACKLOG ",
        "",
        "BD-31§1.4bis⑤ →  → ",
        f"  n_unparseable={mc['n_unparseable']} / n_parsed={mc['n_parsed']}"
        f" / ={mc['unparseable_by_cell'] or ' 0'}"
        f" / reasons={mc['unparseable_reasons'] or '{}'}",
        f"  n_unparseable={ec['n_unparseable']} / n_parsed={ec['n_parsed']}"
        f" / ={ec['unparseable_by_cell'] or ' 0'}"
        f" / reasons={ec['unparseable_reasons'] or '{}'}",
        "  (a) 0 (b) selfproof "
        " (run_id,attempt) +  +  HPC + "
        " + collect  calls=attempt  attempt=2  1 "
        "(c) ",
        "",
        f"BD-34 {n_main_ok_serial} [BD-34] "
        f" {n_t1b_ext}  cost_bound T1b  n={n_t1b}",
        f"  {len(ctx['serial_main'])}  × "
        f"per_cell={lc['t1b']['main']['per_cell']}"
        f"cat ={_dist(ctx['t1b_main'], 'cat')}",
        f"  T1  n={len(serial_t1)}cat ={_dist(serial_t1, 'cat')}"
        "§3ter ",
        "  (a) disclosure-only—— manifest_t1"
        f"(b) ledger-backfill—— {n_main_ok_serial} run "
        " serial_latency_ms+serial_source "
        "(c) new-file:<>——U17 ",
        "",
        f"BD-35latency_sample={lat_sample} ",
        f"  rate_source= {lc['latency_sample']['rate_source']}",
        f"  per=ceil({lat_sample}/{n_cells})={lc['t1b']['main']['per_cell']}"
        f" ⇒  {lc['t1b']['main']['n']}"
        f"ext=ceil({n_ext_req}×{lat_sample}/{n_main_req})={lc['t1b']['extended']['n']}",
        "  (a) (b) "
        "(c) ",
        "",
        "",
        f"  (A)-9T1  N={ec['n_request_error']}"
        f"={ec['request_error_by_cell']} N={mc['n_request_error']}"
        f"T1b  N={lc['t1b']['main']['n_request_error'] + lc['t1b']['extended']['n_request_error']}"
        " run per [BD-34]  cost_bound/manifest ",
        "  usage error is None payload ",
        "  lab_stamp HPC canonical ",
        "",
        "emituv run python scripts/b6_t1_collect.py emit --bundle … --t1 …"
        " --t1b … --out … --bd31 \"<>\" --bd34-main-dest"
        " <disclosure-only|ledger-backfill|new-file:> --bd35 \"<>\"",
    ]
    return "\n".join(lines)


NEW_PRODUCT_RELPATHS = ("ledger_a_t1.jsonl", "manifest_t1.json",
                        "matrix/llm.counts.json", "matrix/llm.rownorm.json",
                        "matrix/llm.extended_arm.cost_bound.json")


def emit_products(ctx: dict[str, Any], rulings: dict[str, str], *, out: Path,
                  emitter_factory: Any = ProductEmitter,
                  lab_stamp_now: str | None = None) -> list[Path]:
    for k in ("bd31", "bd34_main_dest", "bd35"):
        if not (rulings.get(k) or "").strip():
            raise SystemExit("")
    lab_stamp_now = lab_stamp_now or SUB.lab_stamp_fingerprint()

    em_t1 = emitter_factory(ctx["t1"]["tree_id"], lab_stamp=ctx["t1"]["fingerprint"])
    em_t1b = emitter_factory(ctx["t1b"]["tree_id"], lab_stamp=ctx["t1b"]["fingerprint"])
    em_c = emitter_factory(out.name, lab_stamp=lab_stamp_now)

    prov_p = Path(getattr(em_c, "provenance_path", ""))
    existing: set[str] = set()
    if str(prov_p) and prov_p.exists():
        prev = json.loads(prov_p.read_text(encoding="utf-8"))
        existing = {p["path"] for p in
                    (prev.get("trees", {}).get(out.name) or {}).get("products", [])}
    env_names = {k: f"env_echo-{ctx[k]['tree_id']}.json" for k in ("t1", "t1b")}
    if env_names["t1"] == env_names["t1b"]:
        raise SystemExit("")
    new_rel = [f"data/{out.name}/{rel}" for rel in NEW_PRODUCT_RELPATHS]
    new_rel += [f"data/{out.name}/env_echo/{env_names[k]}" for k in ("t1", "t1b")]
    extra_file = apply_bd34_ruling(ctx["ledger"], rulings["bd34_main_dest"],
                                   ctx["t1b_main"], t1b_tree_id=ctx["t1b"]["tree_id"])
    if extra_file:
        new_rel.append(f"data/{out.name}/{extra_file['path']}")
    overlap = sorted(set(new_rel) & existing)
    if overlap:
        raise SystemExit("")

    manifest = dict(ctx["manifest_t1"])
    manifest["user_rulings"] = {"bd31": rulings["bd31"],
                                "bd34_main_dest": rulings["bd34_main_dest"],
                                "bd35": rulings["bd35"]}
    manifest["main_arm_serial_latency"] = {**manifest["main_arm_serial_latency"],
                                           "destination": rulings["bd34_main_dest"]}
    manifest["new_products"] = sorted(new_rel)
    manifest["lab_stamp_current"] = lab_stamp_now

    written: list[Path] = []
    for em_t, t in ((em_t1, ctx["t1"]), (em_t1b, ctx["t1b"])):
        for p in tree_files(t["root"]):
            written.append(em_t.write(p, p.read_text(encoding="utf-8")))
        em_t.emit()
    em = em_c
    written.append(em.write_json(out / "matrix" / "llm.counts.json", ctx["counts_doc"]))
    written.append(em.write_json(out / "matrix" / "llm.rownorm.json", ctx["rownorm_doc"]))
    written.append(em.write_json(out / "matrix" / "llm.extended_arm.cost_bound.json",
                                 ctx["cost_bound"]))
    written.append(em.write_jsonl(out / "ledger_a_t1.jsonl", ctx["ledger"]))
    if extra_file:
        written.append(em.write_json(out / extra_file["path"], extra_file["payload"]))
    for key in ("t1", "t1b"):
        t = ctx[key]
        src = t["root"] / "env_echo" / "env_echo.json"
        written.append(em.write(out / "env_echo" / env_names[key],
                                src.read_text(encoding="utf-8")))
    written.append(em.write_json(out / "manifest_t1.json", manifest))
    em.emit()
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        )
    ap.add_argument("mode", choices=("report", "emit"))
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t1b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--bd31", default=None)
    ap.add_argument("--bd34-main-dest", default=None)
    ap.add_argument("--bd35", default=None)
    args = ap.parse_args(argv)

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else LAB / p

    out = _p(args.out)
    man_p = out / "manifest.json"
    if not man_p.is_file():
        raise SystemExit("")
    canonical_manifest = json.loads(man_p.read_text(encoding="utf-8"))
    cfg = SUB.load_config()
    corpus = _p(args.data) if args.data else (LAB / cfg["measurement"]["corpus_root"])

    ctx = collect_context(bundle=_p(args.bundle), t1_root=_p(args.t1),
                          t1b_root=_p(args.t1b), canonical_manifest=canonical_manifest,
                          corpus_root=corpus)
    if args.mode == "report":
        print(halt_text(ctx))
        return 0
    rulings = {"bd31": args.bd31 or "", "bd34_main_dest": args.bd34_main_dest or "",
               "bd35": args.bd35 or ""}
    written = emit_products(ctx, rulings, out=out)
    for p in written:
        print(f"[t1-collect]   {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
