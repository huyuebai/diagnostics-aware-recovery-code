#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_items as I
import b5_judge as J
import run_matrix
from dar import accounting as A
from dar.taskmodel import load_task
from dar.vendorpath import VENDOR_ROOT

MANIFEST_GLOB = "b5_run_manifest.chunk*.json"
STAGING_GLOB = "staging_manifest.chunk*of*.json"
ANALYSIS_DIRNAME = "analysis"

WALL_LOWER_BOUND_NOTE = (
    " GPU·h **** ≳0.27 GPU·h .out "
    " .out  ⇒  ≥ "
    " §7-f  70  §7-e  [est] ")


def _load_vendor_metrics():
    p = VENDOR_ROOT / "evaluation" / "core" / "metrics.py"
    if not p.is_file():
        raise SystemExit("")
    spec = importlib.util.spec_from_file_location("_vendor_metrics", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MetricsCalculator


def load_manifest_rows(data: Path) -> list[dict[str, Any]]:
    paths = sorted(data.glob(MANIFEST_GLOB))
    if not paths:
        raise SystemExit("")
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    rows = first["runs"]
    for p in paths[1:]:
        other = json.loads(p.read_text(encoding="utf-8"))["runs"]
        if other != rows:
            raise SystemExit("")
    return rows


_OUT_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_out_file(p: Path) -> tuple[int | None, float | None, str | None]:
    m = re.search(r"_(\d+)\.out$", p.name)
    if not m:
        return None, None, "no_chunk_index"
    text = p.read_text(encoding="utf-8", errors="replace")
    ts = [x.group(1) for x in (_OUT_TS.match(ln) for ln in text.splitlines()) if x]
    if len(ts) < 2:
        return int(m.group(1)), None, "too_few_timestamps"
    import datetime as _dt
    try:
        a = _dt.datetime.fromisoformat(ts[0])
        b = _dt.datetime.fromisoformat(ts[-1])
    except ValueError:
        return int(m.group(1)), None, "bad_timestamp"
    return int(m.group(1)), max(0.0, (b - a).total_seconds()), None


def chunk_wall_seconds(data: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    logs = data / "logs"
    if not logs.is_dir():
        return out
    for p in sorted(logs.glob("*.out")):
        k, span, reason = parse_out_file(p)
        if reason is not None or k is None or span is None:
            continue
        out[k] = out.get(k, 0.0) + span
    return out


_OUT_RUN = re.compile(r" - tm_run_eval - INFO - Processing ")


def out_file_scan(data: Path) -> dict[str, Any]:
    logs = data / "logs"
    scan: dict[str, Any] = {"n_out_files": 0, "n_out_parsed": 0,
                            "unparsed_out_files": [], "n_executions_by_chunk": {},
                            "n_exec_units_by_chunk": {}}
    if not logs.is_dir():
        return scan
    for p in sorted(logs.glob("*.out")):
        scan["n_out_files"] += 1
        k, _span, reason = parse_out_file(p)
        if reason is not None:
            scan["unparsed_out_files"].append(
                {"name": p.name, "bytes": p.stat().st_size, "reason": reason})
            continue
        scan["n_out_parsed"] += 1
        text = p.read_text(encoding="utf-8", errors="replace")
        key = str(k)
        scan["n_executions_by_chunk"][key] = (
            scan["n_executions_by_chunk"].get(key, 0) + len(_OUT_RUN.findall(text)))
        scan["n_exec_units_by_chunk"][key] = (
            scan["n_exec_units_by_chunk"].get(key, 0) + text.count("Running in parallel with"))
    scan["n_executions_by_chunk"] = dict(sorted(scan["n_executions_by_chunk"].items()))
    scan["n_exec_units_by_chunk"] = dict(sorted(scan["n_exec_units_by_chunk"].items()))
    return scan


def cat_to_chunk_from_staging(data: Path,
                              rows: list[dict[str, Any]]) -> tuple[dict[str, int] | None, str]:
    by_run = {r["run_id"]: r["cat"] for r in rows}
    derived: dict[str, int] = {}
    paths = sorted(data.glob(STAGING_GLOB))
    if not paths:
        raise SystemExit("")
    for p in paths:
        ch = json.loads(p.read_text(encoding="utf-8"))["chunk"]
        idx, ids = int(ch["index"]), ch["chunk_run_ids"]
        cats = {by_run[i] for i in ids if i in by_run}
        if len(cats) != 1:
            return None, (f"unavailable(chunk {idx}  {len(cats)}  cat{sorted(cats)}"
                          f" cat  chunk ⇒ )")
        cat = cats.pop()
        if derived.get(cat, idx) != idx:
            return None, (f"unavailable(cat={cat!r}  chunk {derived[cat]}  {idx}"
                          f" cat  ⇒ )")
        derived[cat] = idx
    positional = {c: i + 1 for i, c in enumerate(sorted({r["cat"] for r in rows}))}
    if derived != positional:
        raise SystemExit(
            "")
    return derived, f"{STAGING_GLOB}  chunk.chunk_run_ids  + "


def build_ledger(data: Path, model: str, vmap: dict[str, dict[tuple, str]],
                 layer_a: dict[str, dict[str, Any]],
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from dar.b5_stats import run_verdict_from_item_map
    MC = _load_vendor_metrics()
    rows = load_manifest_rows(data)
    walls = chunk_wall_seconds(data)
    scan = out_file_scan(data)
    n_per_chunk: dict[str, int] = {}
    for r in rows:
        n_per_chunk[r["cat"]] = n_per_chunk.get(r["cat"], 0) + 1
    cat_to_chunk, cat_chunk_src = cat_to_chunk_from_staging(data, rows)
    missing = (sorted(c for c, k in cat_to_chunk.items() if k not in walls)
               if cat_to_chunk else [])
    if missing and walls:
        raise SystemExit(
            "")
    if cat_to_chunk:
        for cat, k in sorted(cat_to_chunk.items()):
            key = str(k)
            if key not in scan["n_executions_by_chunk"]:
                continue
            n_exec, n_deliv = scan["n_executions_by_chunk"][key], n_per_chunk[cat]
            if n_exec < n_deliv:
                raise SystemExit(
                    "")

    out: list[dict[str, Any]] = []
    mismatch_hits: list[str] = []
    mismatch_calls: list[str] = []
    n_layer_b = 0
    for row in rows:
        rid = row["run_id"]
        la = layer_a.get(rid)
        if la is None:
            raise SystemExit("")
        item_map = vmap.get(rid)
        if item_map:
            ga_verdict = run_verdict_from_item_map(item_map)
            n_layer_b += 1
        else:
            if B4.is_item_eligible(la["verdict"], la.get("reason_code")):
                raise SystemExit(
                    "")
            ga_verdict = la["verdict"]
        inf_p = data / run_matrix.inference_relpath(row, model)
        ev_p = data / run_matrix.eval_relpath(row, model)
        inference = json.loads(inf_p.read_text(encoding="utf-8"))
        evaluation = json.loads(ev_p.read_text(encoding="utf-8"))
        task = load_task(row["cat"], row["tid"], row["mode"])

        calc = MC()
        calc.add_result(task, inference, evaluation)
        vres = calc.results[-1]
        bad = run_matrix.inference_health(inference)
        if bad:
            raise SystemExit("")
        events = A.tool_events(inference["messages"])
        ours_hits = A.unique_first_hits(events)
        if int(vres.get("prr_hits") or 0) != ours_hits:
            mismatch_hits.append(f"{rid}(vendor={vres.get('prr_hits')} ours={ours_hits})")
        ours_calls = A.actual_recovery_calls(events)
        v_calls = vres.get("actual_recovery_calls")
        if (v_calls is None) != (ours_calls is None) or (
                v_calls is not None and ours_calls is not None and int(v_calls) != int(ours_calls)):
            mismatch_calls.append(f"{rid}(vendor={v_calls} ours={ours_calls})")

        k = cat_to_chunk[row["cat"]] if cat_to_chunk else None
        wall = (A.amortized_wall_clock_s(walls[k], n_per_chunk[row["cat"]])
                if k is not None and k in walls else None)
        for key in ("prr_resolved", "oracle_recovery_calls"):
            if key not in vres:
                raise SystemExit(
                    "")
        out.append(A.build_ledger_b_row(
            row, inference, evaluation, ga_verdict=ga_verdict,
            prr_resolved=int(vres["prr_resolved"] or 0),
            oracle_recovery_calls=vres["oracle_recovery_calls"],
            wall_s_amortized=wall))
    if mismatch_hits or mismatch_calls:
        raise SystemExit(
            "")
    meta = {"n_runs": len(out), "n_layer_b": n_layer_b, "n_layer_a_terminal": len(out) - n_layer_b,
            "wall_seconds_by_chunk": {str(k): v for k, v in sorted(walls.items())},
            "wall_source": "logs/*_<k>.out ",
            "n_runs_by_cat": n_per_chunk, "model": model,
            "vendor_replay": "MetricsCalculator().add_result → results[-1]",
            "dual_impl_reconciled": ["prr_hits", "actual_recovery_calls"],
            "n_out_files": scan["n_out_files"],
            "n_out_parsed": scan["n_out_parsed"],
            "unparsed_out_files": scan["unparsed_out_files"],
            "missing_wall_chunks": (sorted(set(cat_to_chunk.values()) - set(walls))
                                    if cat_to_chunk else None),
            "n_executions_by_chunk": scan["n_executions_by_chunk"],
            "n_exec_units_by_chunk": scan["n_exec_units_by_chunk"],
            "n_exec_units_note": (
                "= `run_matrix.plan_units` ****vendor "
                "****`B5PairedUnit` §3 ****"
                "——5  4 worker  2 "
                " ≈2.11  `run_matrix.plan_units` docstring"),
            "n_delivered_by_chunk": ({str(cat_to_chunk[c]): n
                                      for c, n in sorted(n_per_chunk.items())}
                                     if cat_to_chunk else {}),
            "cat_to_chunk": dict(sorted(cat_to_chunk.items())) if cat_to_chunk else None,
            "cat_to_chunk_source": cat_chunk_src,
            }
    if walls:
        meta["wall_lower_bound_note"] = WALL_LOWER_BOUND_NOTE
    if walls and cat_to_chunk:
        meta["amortization_note"] = (
            "`wall_s_amortized`  =  chunk  n_executions_by_chunk "
            "**** run = n_delivered_by_chunk****"
            "⇒  > B-27")
    return out, meta


def write_ledgers(data: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Path]:
    outdir = data / ANALYSIS_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    payloads = {
        "ledger_b.jsonl": "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                  for r in rows),
        "ledger_a.json": json.dumps(
            {"usd": A.ledger_a_fixed_zero().usd, "basis": A.ledger_a_fixed_zero().basis,
             "note": "B5  fixed/none tier ⇒  0 = B6§6  A",
             "meta": meta}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    }
    for name, text in payloads.items():
        p = outdir / name
        if p.exists() and p.read_text(encoding="utf-8") != text:
            raise SystemExit("")
        p.write_text(text, encoding="utf-8")
        paths[name] = p
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    args = ap.parse_args()
    data = Path(args.data)

    I.assert_corpus_identity_matches_local(data)
    I.assert_code_stamped()

    items = J.load_b5_judge_items(data)
    vmap = J.load_verdicts_singlesource(
        data, expected_run_ids=sorted({i["run_id"] for i in items}),
        expected_items=J.items_universe(items))
    layer_a = {}
    for line in (data / "ga_layerA.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            layer_a[r["run_id"]] = r
    rows, meta = build_ledger(data, args.model, vmap, layer_a)
    paths = write_ledgers(data, rows, meta)
    for k, p in paths.items():
        print(f"           {k} → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
