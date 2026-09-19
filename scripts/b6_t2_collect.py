#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b6_api_diagnose as API
import b6_llm_diagnose as DRV
import b6_subset as SUB
import b6_t1_collect as COL
import measure_diagnosers as MD
from b6_products import ProductEmitter
from dar.accounting import LedgerA
from dar.diagnoser import llm as LLM

T2_TREE_FIXED_FILES = ("responses.jsonl", "batch_chunks_t2.json", "t2_run_stamp.json")
T2_TREE_OPTIONAL_FILES = ("batch_usage_t2.json",)
NEW_PRODUCT_RELPATHS = ("ledger_a_t2.jsonl", "manifest_t2.json",
                        "matrix/api.counts.json", "matrix/api.rownorm.json")


LATENCY_ABSENT_NOTE = (
    "API Batch 24h driver docstring⇒  latency_ms"
    "concurrency  = prompt_tokenscompletion_tokenscalls")

ACTUAL_USD_CALIBER_NOTE = (
    "actual_usd = usage(prompt_tokens, completion_tokens) × configs/b6_diagnosers.yaml::"
    "pricing  in/out  × batch_discount_factor"
    "cached_tokens  config[LM6]⇒  cached_tokens > 0"
    "actual_usd completion_tokens  reasoning  batch_usage_t2.json "
    " reasoning_tokens ")

PREFLIGHT_NOTE = (
    "submit  create  1-request live preflight  "
    "actual_usd usage  = BACKLOG [BD-44] canary ①"
    "⚠️  b4_gate._api_retry(tries=6)  ⇒ ****")

PRICING_RECEIPT_NOTE = (
    "canary submitt2_run_stamp.json  mtime"
    "driver **** batches.create "
    "batch_chunks_t2.json  batch_id  8  hex  unix  "
    "checked_utc  create §6.5 "
    "⚠️  =  0  "
    "t2_run_stamp.json mtime  responses.jsonl mtime"
    " config  matches_config  ⇒ ")

STAMP_ATTRIBUTION_NOTE = (
    "T2  lab_stamp =  mtime  git  "
    "lab_code_provenance.json  evidence_commands driver "
    " verify BACKLOG [BD-45] ⚠️ ")

STAMP_SEGMENT_KEYS_NOTE = (
    "segments[].files = segments[].chunks = "
    "—— 0  responses.jsonl "
    "")

COUNTS_ROW_SET_NOTE = (
    "`n_requests`  responses.jsonl `n_truncated`"
    "`n_text_none_error_free`  error `n_parsed``n_unparseable`"
    "`n_request_error`  = `excluded_counts` ****"
    "obs is None  (A)-7 =  "
    "matrix/api.*.json")

LEDGER_ROW_SET_NOTE = (
    " = T2 ** error ******error (A)-9"
    "⚠️  ≠ NP  ⇒ §2ter⇒ "
    " n_diagnoses **** reconciliation.subset.n_units  500 n_units "
    "§6.5 ledger_a_t1.jsonl "
    " axis ——")

USAGE_RECON_POWER_NOTE = (
    "d_usd  usd_from_batch_usage ** config ** ⇒ d_usd  (d_in, d_out) "
    " Batch.usage ** $ **usd_from_batch_usage = "
    " token ×  token "
    " ⇒ collect chunk ")

BILL_LEVEL_OPEN_NOTE = (
    "[BD-44] canary ③  batches.retrieve(...).usage"
    " = batch_usage_t2.json batch ****"
    " ⇒ actual_usd ")

PROMPT_CACHE_CALIBER_NOTE = (
    "configs/b6_diagnosers.yaml  prompt_cache: false  §5 prompt caching"
    "**** caching batch_usage_t2.json  cached_tokens "
    " batch_usage_reconciliation.totals.cached——"
    "§5 cached/uncached token + Anthropic  **** "
    " ⇒  = BACKLOG.md [BD-47] "
    " (a)2026-08-23 D2  ✅  2026-08-24  "
    "[BD-78]②  [BD-47] ")

STAMP_RULE_NOTE = (
    " = 2026-08-21  = BACKLOG.md "
    "[BD-45] T2-collect (ii) user_rulings  hex "
    "——")

UNPARSEABLE_NOTE = (
    "[BD-31] API  0 §1.4bis③  = "
    "§1.4bis⑤ ")

STAMP_EVIDENCE_COMMANDS = (
    "git show b0d66f9:lab/provenance/lab_code_provenance.json | "
    "python3 -c \"import json,sys;print(json.load(sys.stdin)['fingerprint_sha256'])\"",
    "git show 7bc4872:lab/provenance/lab_code_provenance.json | "
    "python3 -c \"import json,sys;print(json.load(sys.stdin)['fingerprint_sha256'])\"",
    "git diff --stat b0d66f9 7bc4872 -- lab/",
    "git log -1 --format='%h %ad' --date=iso-strict b0d66f9; "
    "git log -1 --format='%h %ad' --date=iso-strict 7bc4872",
    "stat -f '%Sm %N' -t '%Y-%m-%dT%H:%M:%S%z' lab/data/b6-t2api-20260820/*",
    "cd lab && uv run python -c \"import json,datetime as dt;"
    "[print(c['idx'], dt.datetime.fromtimestamp("
    "int(c['batch_id'].split('_')[1][:8],16), dt.timezone.utc).isoformat()) "
    "for c in json.load(open('data/b6-t2api-20260820/batch_chunks_t2.json'))['chunks']]\"",
    "cd lab && uv run python scripts/stamp_lab.py --check",
)

PRICING_RECEIPT = {
    "checked_utc": "2026-08-21T11:46Z",
    "source_url": "https://developers.openai.com/api/docs/pricing",
    "standard_in_per_m": 2.5,
    "standard_out_per_m": 15.0,
    "batch_discount": 0.5,
    "source": "BACKLOG [BD-44] canary ②",
}

_HEX64 = re.compile(r"[0-9a-f]{64}")


def _product_rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(LAB))
    except ValueError:
        return str(Path(path))


def t2_tree_files(root: Path, *, with_usage: bool) -> list[Path]:
    out = [root / n for n in T2_TREE_FIXED_FILES]
    if with_usage:
        out += [root / n for n in T2_TREE_OPTIONAL_FILES]
    return out


def load_t2_tree(root: Path, *, require_batch_usage: bool) -> dict[str, Any]:
    for name in T2_TREE_FIXED_FILES:
        if not (root / name).is_file():
            raise SystemExit("")
    resp = root / "responses.jsonl"
    rows: list[dict[str, Any]] = []
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

    stamp_p = root / "t2_run_stamp.json"
    try:
        stamp = json.loads(stamp_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(
            "")
    if stamp.get("run_pass") != API.T2_PASS:
        raise SystemExit("")

    led = json.loads((root / "batch_chunks_t2.json").read_text(encoding="utf-8"))
    if led.get("run_pass") != API.T2_PASS or led.get("model") != stamp.get("model"):
        raise SystemExit("")
    if list(led.get("chunk_sizes") or []) != list(stamp.get("chunk_sizes") or []):
        raise SystemExit("")
    if int(led.get("n_requests") or -1) != int(stamp.get("n_requests") or -2):
        raise SystemExit("")
    if int(stamp["n_requests"]) != len(rows):
        raise SystemExit("")

    chunks = list(led.get("chunks") or [])
    bad = [c["idx"] for c in chunks
           if c.get("status") != "collected" or c.get("actual_usd") is None]
    if bad:
        raise SystemExit(
            "")
    sizes = list(led["chunk_sizes"])
    if len(chunks) != len(sizes):
        raise SystemExit("")
    if [int(c["idx"]) for c in chunks] != list(range(len(chunks))):
        raise SystemExit("")
    if [int(c["n"]) for c in chunks] != [int(n) for n in sizes]:
        raise SystemExit("")
    per_chunk: dict[Any, int] = {}
    for r in rows:
        per_chunk[r.get("chunk_idx")] = per_chunk.get(r.get("chunk_idx"), 0) + 1
    got = [per_chunk.get(i, 0) for i in range(len(sizes))]
    if got != [int(n) for n in sizes]:
        raise SystemExit("")

    usage_p = root / "batch_usage_t2.json"
    usage_doc = None
    if usage_p.is_file():
        if not require_batch_usage:
            raise SystemExit(
                "")
        usage_doc = json.loads(usage_p.read_text(encoding="utf-8"))
        if usage_doc.get("run_pass") != API.T2_PASS \
                or usage_doc.get("items_sha256") != led.get("items_sha256"):
            raise SystemExit(
                "")
        want = [(int(c["idx"]), c.get("batch_id")) for c in chunks]
        got_ids = [(int(c["idx"]), c.get("batch_id")) for c in (usage_doc.get("chunks") or [])]
        if got_ids != want:
            raise SystemExit("")
    elif require_batch_usage:
        raise SystemExit(
            "")
    return {"root": root, "tree_id": root.name, "rows": rows, "stamp": stamp,
            "ledger": led, "usage_doc": usage_doc}


def assert_t2_tree_facts(t2: dict[str, Any], bundle_manifest: dict[str, Any],
                         canonical_manifest: dict[str, Any], cfg: dict[str, Any],
                         bundle_sha: str) -> None:
    stamp = t2["stamp"]
    if stamp.get("bundle_sha256") != bundle_sha:
        raise SystemExit("")
    skel_now = LLM.skeleton_fingerprint()
    for where, got in (("", bundle_manifest.get("prompt_skeleton_fingerprint")),
                       ("canonical manifest", canonical_manifest["prompt_skeleton_fingerprint"]),
                       (" llm.py", skel_now)):
        if stamp.get("prompt_skeleton_fingerprint") != got:
            raise SystemExit("")
    if stamp.get("system_prompt_sha256") != bundle_manifest.get("system_prompt_sha256"):
        raise SystemExit("")
    if stamp.get("model") != cfg["model"]:
        raise SystemExit("")
    if stamp.get("basis") != cfg["basis"]:
        raise SystemExit("")
    if int(stamp["api_parse_retries"]) != API.API_PARSE_RETRIES:
        raise SystemExit("")


def ledger_rows_payload_t2(ok_rows: list[dict[str, Any]], *, t2_tree_id: str,
                           cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in sorted(ok_rows, key=lambda x: x["run_id"]):
        p_tok = int(r.get("prompt_tokens") or 0)
        c_tok = int(r.get("completion_tokens") or 0)
        led = LedgerA.b6_row(
            diagnoser="api", run_id=r["run_id"], axis=r["axis"],
            usd=API.usd_of_tokens(p_tok, c_tok, cfg), basis=cfg["basis"],
            prompt_tokens=p_tok, completion_tokens=c_tok,
            calls=int(r.get("attempt") or 1))
        if not (led.diagnoser.strip() and led.run_id.strip() and led.axis.strip()):
            raise SystemExit("")
        out.append({"diagnoser": led.diagnoser, "run_id": led.run_id, "axis": led.axis,
                    "prompt_tokens": led.prompt_tokens,
                    "completion_tokens": led.completion_tokens, "calls": led.calls,
                    "usd": led.usd, "basis": led.basis,
                    "run_pass": API.T2_PASS, "attempt": int(r.get("attempt") or 1),
                    "batch_id": r.get("batch_id"), "chunk_idx": int(r["chunk_idx"]),
                    "source_tree": t2_tree_id})
    return out


def reconcile_ledger_by_chunk(ledger: list[dict[str, Any]],
                              led_doc: dict[str, Any]) -> list[dict[str, Any]]:
    by_chunk: dict[int, float] = {}
    for row in ledger:
        k = int(row["chunk_idx"])
        by_chunk[k] = by_chunk.get(k, 0.0) + float(row["usd"])
    out: list[dict[str, Any]] = []
    for ch in led_doc["chunks"]:
        idx = int(ch["idx"])
        got = by_chunk.get(idx, 0.0)
        actual = float(ch["actual_usd"])
        diff = got - actual
        if abs(diff) > API._USD_RECONCILE_TOL:
            raise SystemExit(
                "")
        out.append({"idx": idx, "ledger_usd": got, "actual_usd": actual, "diff": diff})
    return out


def batch_usage_reconciliation(usage_doc: dict[str, Any] | None,
                               ok_rows: list[dict[str, Any]], led_doc: dict[str, Any],
                               cfg: dict[str, Any]) -> dict[str, Any]:
    if usage_doc is None:
        return {"available": False,
                "reason": "batch_usage_t2.json --without-batch-usage "
                          " files.content 2026-08-21 ",
                "caliber_note": ACTUAL_USD_CALIBER_NOTE,
                "power_note": USAGE_RECON_POWER_NOTE,
                "actual_usd_is_upper_bound": None}
    rows_in: dict[int, int] = {}
    rows_out: dict[int, int] = {}
    for r in ok_rows:
        k = int(r["chunk_idx"])
        rows_in[k] = rows_in.get(k, 0) + int(r.get("prompt_tokens") or 0)
        rows_out[k] = rows_out.get(k, 0) + int(r.get("completion_tokens") or 0)
    chunks: list[dict[str, Any]] = []
    tot = {"rows_in": 0, "rows_out": 0, "api_in": 0, "api_out": 0,
           "cached": 0, "reasoning": 0}
    n_avail = 0
    any_cached_known = False
    for uc in usage_doc.get("chunks") or []:
        idx = int(uc["idx"])
        blob: dict[str, Any] = {"idx": idx, "rows_in": rows_in.get(idx, 0),
                                "rows_out": rows_out.get(idx, 0),
                                "api_status": uc.get("api_status"),
                                "request_counts": uc.get("request_counts")}
        usage = uc.get("usage")
        api_in = (usage or {}).get("input_tokens")
        api_out = (usage or {}).get("output_tokens")
        if usage is None or api_in is None or api_out is None:
            blob.update({"available": False,
                         "reason": uc.get("usage_absent_reason")
                         or "usage  input_tokens/output_tokens ⇒  0[LE1]",
                         "api_in": None, "api_out": None, "cached": None,
                         "reasoning": None, "d_in": None, "d_out": None,
                         "usd_from_batch_usage": None, "d_usd": None})
            chunks.append(blob)
            continue
        cached = ((usage.get("input_tokens_details") or {}).get("cached_tokens"))
        reasoning = ((usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
        usd_api = API.usd_of_tokens(int(api_in), int(api_out), cfg)
        usd_rows = API.usd_of_tokens(blob["rows_in"], blob["rows_out"], cfg)
        blob.update({"available": True, "reason": None,
                     "api_in": int(api_in), "api_out": int(api_out),
                     "cached": None if cached is None else int(cached),
                     "reasoning": None if reasoning is None else int(reasoning),
                     "d_in": blob["rows_in"] - int(api_in),
                     "d_out": blob["rows_out"] - int(api_out),
                     "usd_from_batch_usage": usd_api, "d_usd": usd_rows - usd_api})
        n_avail += 1
        tot["rows_in"] += blob["rows_in"]
        tot["rows_out"] += blob["rows_out"]
        tot["api_in"] += int(api_in)
        tot["api_out"] += int(api_out)
        if cached is not None:
            any_cached_known = True
            tot["cached"] += int(cached)
        if reasoning is not None:
            tot["reasoning"] += int(reasoning)
        chunks.append(blob)
    tot["d_in"] = tot["rows_in"] - tot["api_in"]
    tot["d_out"] = tot["rows_out"] - tot["api_out"]
    tot["usd_rows"] = API.usd_of_tokens(tot["rows_in"], tot["rows_out"], cfg)
    tot["usd_from_batch_usage"] = API.usd_of_tokens(tot["api_in"], tot["api_out"], cfg)
    tot["d_usd"] = tot["usd_rows"] - tot["usd_from_batch_usage"]
    if abs(tot["d_usd"]) <= API._USD_RECONCILE_TOL:
        direction = "equal"
    elif tot["d_usd"] > 0:
        direction = "rows_over_api"
    else:
        direction = "api_over_rows"
    upper = None if not any_cached_known else bool(tot["cached"] > 0)
    return {"available": n_avail > 0, "reason": None if n_avail else " usage ",
            "caliber_note": ACTUAL_USD_CALIBER_NOTE,
            "n_chunks_available": n_avail, "chunks": chunks, "totals": tot,
            "direction": direction, "actual_usd_is_upper_bound": upper,
            "power_note": USAGE_RECON_POWER_NOTE,
            "sdk_openai_version": usage_doc.get("sdk_openai_version"),
            "source": f"{usage_doc.get('schema')}driver usage  batches.retrieve"}


def preflight_block(ctx: dict[str, Any], *, prompt_tokens: int,
                    completion_tokens: int) -> dict[str, Any]:
    run_id = ctx["first_request_run_id"]
    by_id = {r["run_id"]: r for r in ctx["t2"]["rows"]}
    got = int((by_id.get(run_id) or {}).get("prompt_tokens") or -1)
    if got != int(prompt_tokens):
        raise SystemExit(
            "")
    pricing = SUB.load_config()["pricing"]
    usd = (int(prompt_tokens) * float(pricing["openai_price_in_per_m_usd"]) / 1e6
           + int(completion_tokens) * float(pricing["openai_price_out_per_m_usd"]) / 1e6)
    return {"prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens),
            "request_run_id": run_id,
            "usd_est_direct": usd, "source": "",
            "included_in_actual_usd": False, "note": PREFLIGHT_NOTE}


def stamp_attribution(ctx: dict[str, Any], *, canary: str | None,
                      full: str | None) -> dict[str, Any]:
    n_chunks = len(ctx["t2"]["ledger"]["chunks"])
    segs = [{"files": ["t2_run_stamp.json"], "chunks": [0], "lab_stamp": canary,
             "basis": ""},
            {"files": sorted(["responses.jsonl", "batch_chunks_t2.json"]),
             "chunks": list(range(1, n_chunks)), "lab_stamp": full,
             "basis": ""}]
    if ctx["t2"]["usage_doc"] is not None:
        segs.append({"files": ["batch_usage_t2.json"], "chunks": [],
                     "lab_stamp": ctx["t2"]["usage_doc"].get("lab_stamp"),
                     "basis": ""})
    return {"t2_tree_stamp_recorded": False, "segments": segs,
            "segment_keys_note": STAMP_SEGMENT_KEYS_NOTE,
            "rule_note": STAMP_RULE_NOTE,
            "evidence_commands": list(STAMP_EVIDENCE_COMMANDS),
            "driver_diff_scope": STAMP_ATTRIBUTION_NOTE}


def _hist(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return dict(sorted(out.items()))


def collect_context(*, bundle: Path, t2_root: Path, canonical_root: Path,
                    canonical_manifest: dict[str, Any], corpus_root: Path,
                    cfg: dict[str, Any], require_batch_usage: bool = True,
                    units: list[dict[str, Any]] | None = None,
                    run_verify: bool = True) -> dict[str, Any]:
    reqs = API.build_requests(canonical_root, corpus_root, cfg)
    bundle_sha = API.assert_same_input_face_as_bundle(reqs, bundle)
    bman, _bmain, _bext = DRV.load_bundle(bundle)
    t2 = load_t2_tree(t2_root, require_batch_usage=require_batch_usage)
    assert_t2_tree_facts(t2, bman, canonical_manifest, cfg, bundle_sha)

    verify_rc: Any = "not_run"
    if run_verify:
        verify_rc = API.verify(t2_root, canonical_root, corpus_root, cfg)
        if verify_rc:
            raise SystemExit("")
    rows = t2["rows"]
    COL.assert_usage_present(rows, "")
    API.assert_model_echo(rows, cfg)

    sizes = [int(n) for n in t2["ledger"]["chunk_sizes"]]
    chunks, at = [], 0
    for n in sizes:
        chunks.append(reqs[at:at + n])
        at += n

    ok_rows, err_rows = COL.split_request_errors(rows)
    parses, _reasons = COL.parse_arm(ok_rows)
    if units is None:
        main_arm = str(canonical_manifest["sampling"]["main_arm"])
        units = SUB.enumerate_units(corpus_root, main_arm, canonical_manifest["subset_ids"],
                                    with_task_description=False, with_tool_schemas=False)
    parse_protocol = {
        "implementation": "dar.diagnoser.llm.parse_response",
        "api_parse_retries": int(t2["stamp"]["api_parse_retries"]),
        "collect_reparse_retries": 0,
    }
    c_doc, n_doc, main_bits = COL.main_arm_matrix(
        units, ok_rows, err_rows, parses, int(canonical_manifest["n_diagnoses"]),
        parse_protocol=parse_protocol, diagnoser="api", extra_notes={})

    ledger = ledger_rows_payload_t2(ok_rows, t2_tree_id=t2["tree_id"], cfg=cfg)
    if len(ledger) != len(rows) - len(err_rows):
        raise SystemExit("")
    usd_by_chunk = reconcile_ledger_by_chunk(ledger, t2["ledger"])
    usage_recon = batch_usage_reconciliation(t2["usage_doc"], ok_rows, t2["ledger"], cfg)

    stamp = t2["stamp"]
    led_doc = t2["ledger"]
    tree_files = t2_tree_files(t2_root, with_usage=t2["usage_doc"] is not None)
    sum_actual = sum(float(c["actual_usd"]) for c in led_doc["chunks"])
    pricing = SUB.load_config()["pricing"]
    receipt_ok = (float(pricing["openai_price_in_per_m_usd"]) == PRICING_RECEIPT["standard_in_per_m"]
                  and float(pricing["openai_price_out_per_m_usd"]) == PRICING_RECEIPT["standard_out_per_m"]
                  and float(pricing["batch_discount_factor"]) == PRICING_RECEIPT["batch_discount"])
    if not receipt_ok:
        raise SystemExit(
            "")

    frozen_dip = int(canonical_manifest["n_excluded_dispatch_in_prefix"])
    tally_dip = int(main_bits["excluded"].get("dispatch_in_prefix", 0))
    if tally_dip != frozen_dip:
        raise SystemExit(
            "")

    bd45 = {
        "n_len": sum(1 for r in rows if r.get("finish_reason") != "stop"),
        "n_echo": sum(1 for r in ok_rows if (r.get("model_echo") or "") != stamp["model"]),
        "n_none": sum(1 for r in ok_rows if r.get("text") is None),
        "n_unparse": main_bits["n_unparseable"],
        "row_sets": {"n_len": " error  finish_reason  None ⇒ ",
                     "n_echo": " error ", "n_none": " error ",
                     "n_unparse": " error = main_arm  n_unparseable"},
        "source": "BACKLOG [BD-45]③ ",
    }
    manifest_t2 = {
        "schema": "b6_manifest_t2/v1",
        "tree_id": canonical_manifest["tree_id"],
        "t2_tree": {
            "tree_id": t2["tree_id"],
            "fingerprint_recorded": False,
            "files": {str(p.relative_to(t2_root)): {"sha256": COL.sha256_of_file(p),
                                                    "bytes": p.stat().st_size}
                      for p in tree_files}},
        "bundle": {"name": bundle.name, "sha256": bundle_sha},
        "batch": {
            "model": stamp["model"], "cap_usd": float(stamp["cap_usd"]),
            "completion_window": stamp["batch_completion_window"],
            "reasoning_effort": stamp["reasoning_effort"],
            "max_completion_tokens": int(stamp["max_completion_tokens"]),
            "n_requests": int(stamp["n_requests"]), "chunk_sizes": sizes,
            "items_sha256": led_doc["items_sha256"],
            "batch_ids": [c["batch_id"] for c in led_doc["chunks"]],
            "chunks": [{"idx": int(c["idx"]), "n": int(c["n"]), "batch_id": c["batch_id"],
                        "status": c["status"], "n_ok": c["n_ok"], "n_errors": c["n_errors"],
                        "n_no_usage": c["n_no_usage"],
                        "actual_usd": float(c["actual_usd"]),
                        "n_rows_in_tree": sum(1 for r in rows
                                              if int(r["chunk_idx"]) == int(c["idx"]))}
                       for c in led_doc["chunks"]],
            "sum_actual_usd": sum_actual,
            "cap_headroom_usd_excl_preflight": float(stamp["cap_usd"]) - sum_actual},
        "reconciliation": {
            "main": {"n_bundle": int(bman["arms"]["main"]["n_requests"]),
                     "n_tree": len(rows)},
            "verify_rc": verify_rc,
            "input_face_anchor": "b6_api_diagnose.assert_same_input_face_as_bundle",
            "usd_by_chunk": usd_by_chunk,
            "bd45_counts": bd45,
            "subset": {"n_units": int(canonical_manifest["n_units"]),
                       "n_diagnoses": int(canonical_manifest["n_diagnoses"]),
                       "n_diagnoses_ok": int(canonical_manifest["n_diagnoses_ok"]),
                       "subset_seed": int(canonical_manifest["sampling"]["subset_seed"])}},
        "parse": {**parse_protocol,
                  "prompt_skeleton_fingerprint": stamp["prompt_skeleton_fingerprint"],
                  "system_prompt_sha256": stamp["system_prompt_sha256"],
                  "unparseable_note": UNPARSEABLE_NOTE},
        "counts": {
            "n_diagnoses": int(canonical_manifest["n_diagnoses"]),
            "main_arm": {
                "n_requests": len(rows),
                "n_truncated": sum(1 for r in ok_rows
                                   if r.get("finish_reason") == "length"),
                "finish_reason_histogram": _hist(ok_rows, "finish_reason"),
                "model_echo_histogram": _hist(ok_rows, "model_echo"),
                "n_text_none_error_free": sum(1 for r in ok_rows if r.get("text") is None),
                "n_parsed": main_bits["n_parsed"],
                "n_unparseable": main_bits["n_unparseable"],
                "unparseable_by_cell": main_bits["unparseable_by_cell"],
                "unparseable_reasons": main_bits["unparseable_reasons"],
                "n_request_error": main_bits["n_request_error"],
                "request_error_by_cell": main_bits["request_error_by_cell"],
                "excluded_counts": main_bits["excluded"],
                "n_excluded_dispatch_in_prefix": frozen_dip,
                "row_set_note": COUNTS_ROW_SET_NOTE}},
        "ledger_schema": {
            "columns": sorted(ledger[0].keys()) if ledger else [],
            "row_set": LEDGER_ROW_SET_NOTE,
            "latency_ms": LATENCY_ABSENT_NOTE, "concurrency": "absent",
            "usd_formula": "b6_api_diagnose.usd_of_tokens", "basis": cfg["basis"]},
        "pricing": {
            "price_in_per_tok": cfg["price_in"], "price_out_per_tok": cfg["price_out"],
            "basis": cfg["basis"], "basis_matches_stamp": stamp["basis"] == cfg["basis"],
            "actual_usd_caliber": ACTUAL_USD_CALIBER_NOTE,
            "prompt_cache_caliber": PROMPT_CACHE_CALIBER_NOTE,
            "bill_level_reconciliation": BILL_LEVEL_OPEN_NOTE,
            "pricing_receipt": {**PRICING_RECEIPT, "matches_config": receipt_ok,
                                "timing_note": PRICING_RECEIPT_NOTE}},
        "batch_usage_reconciliation": usage_recon,
        "a9_coverage_sentence": {
            "carrier": None,
            "reason": "API §6.5 ⇒ (A)-9 "
                      "n_request_error + request_error_by_cell",
            "annotation": "b6_preregistration.md §0ter (A)-9 🔵 T2-collect "},
    }
    ctx = {"bundle": bundle, "bundle_sha": bundle_sha, "cfg": cfg, "t2": t2,
           "reqs": reqs, "chunks": chunks, "units": units, "verify_rc": verify_rc,
           "ok_rows": ok_rows, "err_rows": err_rows, "parses": parses,
           "counts_doc": c_doc, "rownorm_doc": n_doc, "main_bits": main_bits,
           "ledger": ledger, "usd_by_chunk": usd_by_chunk, "usage_recon": usage_recon,
           "first_request_run_id": chunks[0][0]["custom_id"],
           "canonical_manifest": canonical_manifest, "manifest_t2": manifest_t2}
    manifest_t2["lab_stamp_attribution"] = stamp_attribution(ctx, canary=None, full=None)
    return ctx


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def halt_text(ctx: dict[str, Any]) -> str:
    m = ctx["manifest_t2"]
    mc = m["counts"]["main_arm"]
    t2root = ctx["t2"]["root"]
    ur = ctx["usage_recon"]
    lines = [
        "========== T2-collect HALT report  ==========",
        " = ",
        "",
        "H1lab_stamp [BD-45] ⚠️T2  lab_stamp ⇒ ",
        f"   lab  = {SUB.lab_stamp_fingerprint()}",
        "   mtimeUTC" + "".join(
            f"{n}={_mtime(t2root / n)}" for n in T2_TREE_FIXED_FILES),
        "   = " + "".join(
            f"{'/'.join(s['files'])}→{s['chunks'] or '—'}"
            for s in m["lab_stamp_attribution"]["segments"]),
        "  ⚠️  mtime  commit ****"
        "——`git show <commit>:…` ****"
        " commit  submit "
        "",
        "  ",
    ] + [f"    $ {c}" for c in STAMP_EVIDENCE_COMMANDS] + [
        f"  {STAMP_RULE_NOTE}",
        "   = --stamp-canary-segment <64hex> --stamp-full-segment <64hex>",
        "",
        "H2(A)-9  API ",
        f"   = a9_coverage_sentence.carrier={m['a9_coverage_sentence']['carrier']}"
        f"reason={m['a9_coverage_sentence']['reason']}",
        "   plan §5.8 b6_preregistration.md §0ter (A)-9 "
        " --a9-note \"<>\"  user_rulings",
        "",
        "H3Batch.usage [BD-44] canary ③actual_usd ",
    ]
    if not ur.get("available"):
        lines.append(f"  available=falsereason={ur.get('reason')}"
                     " files.content ——")
    else:
        for b in ur["chunks"]:
            lines.append(
                f"   {b['idx']}rows_in={b['rows_in']} api_in={b['api_in']} "
                f"d_in={b['d_in']}rows_out={b['rows_out']} api_out={b['api_out']} "
                f"d_out={b['d_out']}cached={b['cached']} reasoning={b['reasoning']}"
                f"d_usd={b['d_usd']}" if b["available"] else
                f"   {b['idx']}available=false{b['reason']}")
        t = ur["totals"]
        lines.append(f"   available "
                     f"{ur['n_chunks_available']}/{len(ur['chunks'])}"
                     f"d_in={t['d_in']} d_out={t['d_out']} cached={t['cached']} "
                     f"reasoning={t['reasoning']} d_usd={t['d_usd']}"
                     f"direction={ur['direction']}"
                     f"actual_usd_is_upper_bound={ur['actual_usd_is_upper_bound']}")
        if ur["direction"] == "api_over_rows":
            lines.append(f"  ⚠️ direction=api_over_rows ⇒  "
                         f"${m['batch']['cap_headroom_usd_excl_preflight']:.4f}")
    lines += [
        "  (a)  basis (b) cached  config "
        "(c)  batch ——****"
        "[BD-44] ③ ",
        "   = --batch-usage-ruling \"<>\"",
        "",
        f"H4§1.4bis⑤  →  → ",
        f"  n_unparseable={mc['n_unparseable']} / n_parsed={mc['n_parsed']}"
        f" / ={mc['unparseable_by_cell'] or ' 0'}"
        f" / reasons={mc['unparseable_reasons'] or '{}'}",
        "  (a) (b) ****——API "
        "§1.4bis③Batch  = ",
        f"   = --unparseable-ruling \"<>\" n_unparseable > 0 ",
        "",
        "H5preflight  usage  = ",
        f"   = {ctx['first_request_run_id']} 0  Batch  "
        f"prompt_tokens={next((r['prompt_tokens'] for r in ctx['t2']['rows'] if r['run_id'] == ctx['first_request_run_id']), None)}",
        "   = --preflight-usage \"p=<int>,c=<int>\"collect  p",
        "",
        "",
        f"  verify rc = {ctx['verify_rc']}[BD-45]③  = {m['reconciliation']['bd45_counts']}",
        f"  n_truncated={mc['n_truncated']}finish_reason={mc['finish_reason_histogram']}"
        f"  text=Noneerror ={mc['n_text_none_error_free']}",
        f"  Σactual_usd=${m['batch']['sum_actual_usd']:.4f} /  ${m['batch']['cap_usd']:.0f}"
        f" ${m['batch']['cap_headroom_usd_excl_preflight']:.4f}"
        "**** actual_usd  preflight ",
        f"  parsed={mc['n_parsed']} / unparseable={mc['n_unparseable']} / "
        f"request_error={mc['n_request_error']}={mc['excluded_counts']}",
        "  (A)-2/-3/-8  api tool_schema_notes "
        " manifest_t2.pricing.pricing_receipt",
        f"  [BD-46]  = "
        f"{[c['status'] for c in ctx['t2']['ledger']['chunks']]}",
        f"  api  counts =  = = "
        f"{ctx['counts_doc']['matrix']}",
        "",
        "emituv run python scripts/b6_t2_collect.py emit --t2 … --out … "
        "--bundle … --stamp-canary-segment <64hex> --stamp-full-segment <64hex> "
        "--a9-note \"<>\" --batch-usage-ruling \"<>\" --preflight-usage \"p=…,c=…\"",
    ]
    return "\n".join(lines)


_RULING_KEYS = ("stamp_canary_segment", "stamp_full_segment", "a9_note",
                "batch_usage", "preflight_usage")


def _parse_preflight(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*p=(\d+)\s*,\s*c=(\d+)\s*", s or "")
    if not m:
        raise SystemExit("")
    return int(m.group(1)), int(m.group(2))


def emit_products(ctx: dict[str, Any], rulings: dict[str, str], *, out: Path,
                  emitter_factory: Any = ProductEmitter,
                  lab_stamp_now: str | None = None) -> list[Path]:
    for k in ("stamp_canary_segment", "stamp_full_segment"):
        v = rulings.get(k)
        if not (isinstance(v, str) and _HEX64.fullmatch(v)):
            raise SystemExit(
                "")
    for k in _RULING_KEYS:
        if not (rulings.get(k) or "").strip():
            raise SystemExit("")
    n_unparse = int(ctx["main_bits"]["n_unparseable"])
    given = (rulings.get("unparseable") or "").strip()
    if n_unparse and not given:
        raise SystemExit("")
    if not n_unparse and given:
        raise SystemExit("")
    p_tok, c_tok = _parse_preflight(rulings["preflight_usage"])
    lab_stamp_now = lab_stamp_now or SUB.lab_stamp_fingerprint()

    t2 = ctx["t2"]
    em_canary = emitter_factory(t2["tree_id"], lab_stamp=rulings["stamp_canary_segment"])
    em_full = emitter_factory(t2["tree_id"], lab_stamp=rulings["stamp_full_segment"])
    em_c = emitter_factory(out.name, lab_stamp=lab_stamp_now)

    prov_p = Path(em_c.provenance_path)
    existing: set[str] = set()
    if prov_p.exists():
        prev = json.loads(prov_p.read_text(encoding="utf-8"))
        existing = {p["path"] for p in
                    (prev.get("trees", {}).get(out.name) or {}).get("products", [])}
    new_rel = [_product_rel(out / rel) for rel in NEW_PRODUCT_RELPATHS]
    overlap = sorted(set(new_rel) & existing)
    if overlap:
        raise SystemExit(
            "")

    manifest = dict(ctx["manifest_t2"])
    manifest["user_rulings"] = {
        **{k: rulings[k] for k in _RULING_KEYS},
        "unparseable": given if n_unparse else "n/a: n_unparseable == 0"}
    manifest["preflight_direct_call"] = preflight_block(ctx, prompt_tokens=p_tok,
                                                        completion_tokens=c_tok)
    manifest["lab_stamp_attribution"] = {
        **stamp_attribution(ctx, canary=rulings["stamp_canary_segment"],
                            full=rulings["stamp_full_segment"]),
        "products_binding": "T2 canonical "}
    manifest["new_products"] = sorted(f"data/{out.name}/{rel}"
                                      for rel in NEW_PRODUCT_RELPATHS)
    manifest["lab_stamp_current"] = lab_stamp_now

    written: list[Path] = []
    seg = [(em_canary, ["t2_run_stamp.json"]),
           (em_full, ["responses.jsonl", "batch_chunks_t2.json"])]
    if t2["usage_doc"] is not None:
        em_usage = emitter_factory(t2["tree_id"],
                                   lab_stamp=str(t2["usage_doc"]["lab_stamp"]))
        seg.append((em_usage, list(T2_TREE_OPTIONAL_FILES)))
    for em, names in seg:
        for name in names:
            p = t2["root"] / name
            written.append(em.write(p, p.read_text(encoding="utf-8")))
        em.emit()
    written.append(em_c.write_json(out / "matrix" / "api.counts.json", ctx["counts_doc"]))
    written.append(em_c.write_json(out / "matrix" / "api.rownorm.json", ctx["rownorm_doc"]))
    written.append(em_c.write_jsonl(out / "ledger_a_t2.jsonl", ctx["ledger"]))
    written.append(em_c.write_json(out / "manifest_t2.json", manifest))
    em_c.emit()
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        )
    ap.add_argument("mode", choices=("report", "emit"))
    ap.add_argument("--t2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default="data/b6-input-a5f90686.tar.gz")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--without-batch-usage", action="store_true")
    ap.add_argument("--stamp-canary-segment", default=None)
    ap.add_argument("--stamp-full-segment", default=None)
    ap.add_argument("--a9-note", default=None)
    ap.add_argument("--batch-usage-ruling", default=None)
    ap.add_argument("--preflight-usage", default=None)
    ap.add_argument("--unparseable-ruling", default=None)
    args = ap.parse_args(argv)

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else LAB / p

    out = _p(args.out)
    man_p = out / "manifest.json"
    if not man_p.is_file():
        raise SystemExit("")
    canonical_manifest = json.loads(man_p.read_text(encoding="utf-8"))
    cfg = API.load_api_cfg()
    doc = SUB.load_config()
    corpus = _p(args.corpus) if args.corpus else (LAB / doc["measurement"]["corpus_root"])

    ctx = collect_context(bundle=_p(args.bundle), t2_root=_p(args.t2),
                          canonical_root=out, canonical_manifest=canonical_manifest,
                          corpus_root=corpus, cfg=cfg,
                          require_batch_usage=not args.without_batch_usage)
    if args.mode == "report":
        print(halt_text(ctx))
        return 0
    rulings = {"stamp_canary_segment": args.stamp_canary_segment or "",
               "stamp_full_segment": args.stamp_full_segment or "",
               "a9_note": args.a9_note or "",
               "batch_usage": args.batch_usage_ruling or "",
               "preflight_usage": args.preflight_usage or "",
               "unparseable": args.unparseable_ruling or ""}
    written = emit_products(ctx, rulings, out=out)
    for p in written:
        print(f"[t2-collect]   {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
