#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterator

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b6_subset as SUB
from b6_products import ProductEmitter, sha256_text
from dar.diagnoser import llm as LLM
from dar.diagnoser import trace_adapter as TA
from dar.taskmodel import tool_specs_for_task

BUNDLE_SCHEMA = "b6_t1_bundle/v1"


def _est_tokens(text: str) -> float:
    return LLM.estimate_tokens_chars4(text)


def main_arm_requests(units: list[dict[str, Any]], system_sha: str) -> Iterator[dict[str, Any]]:
    for u in units:
        if u["obs"] is None:
            continue
        _sys, usr = LLM.prompt_pair(u["obs"])
        yield {"run_id": u["run_id"], "arm": u["arm"], "cat": u["cat"], "tid": u["tid"],
               "mode": u["mode"], "axis": u["axis"], "trigger_index": u["trigger_index"],
               "n_tool_schemas": u["n_tool_schemas"], "user": usr, "system_sha256": system_sha}


def extended_arm_requests(corpus: Path, arm: str, subset: dict[str, list[str]],
                          system_sha: str) -> Iterator[dict[str, Any]]:
    for cat, base, mode, path in SUB.iter_inference_files(corpus, arm):
        if base not in subset.get(f"{cat}|{mode}", ()):
            continue
        inf = TA.load_inference(path)
        if inf["mode"] != mode:
            raise SystemExit("")
        desc = SUB.task_description(cat, base, mode)
        sel = tool_specs_for_task(cat, base, mode)
        msgs = inf["messages"]
        step = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "tool":
                continue
            prefix = TA.prefix_at(msgs, i)
            if TA.has_dispatch_message(prefix):
                continue
            obs = TA.to_diagnosis_input(prefix, task_description=desc,
                                        tool_schemas=sel.specs, trigger_round=i)
            _sys, usr = LLM.prompt_pair(obs)
            step += 1
            yield {"run_id": f"{arm}|{cat}|{base}|{mode}|ext{step:03d}", "arm": arm,
                   "cat": cat, "tid": base, "mode": mode, "axis": "extended",
                   "step_index": step, "trigger_index": i,
                   "n_tool_schemas": len(sel.specs), "user": usr, "system_sha256": system_sha}


def _arm_stats(rows: list[dict[str, Any]], system_tokens: float) -> dict[str, Any]:
    toks = sorted(system_tokens + _est_tokens(r["user"]) for r in rows)
    n = len(toks)
    return {"n_requests": n,
            "input_tokens_chars4": {
                "total": sum(toks),
                "mean": (sum(toks) / n) if n else 0.0,
                "p50": toks[n // 2] if n else 0.0,
                "p90": toks[int(n * 0.9)] if n else 0.0,
                "max": toks[-1] if n else 0.0},
            "cells": sorted({f"{r['cat']}|{r['mode']}|{r['axis']}" for r in rows})}


def build_bundle_manifest(cfg: dict[str, Any], tree_id: str, source_manifest: dict[str, Any],
                          source_manifest_sha: str, main_rows: list[dict[str, Any]],
                          ext_rows: list[dict[str, Any]], units: list[dict[str, Any]],
                          ) -> dict[str, Any]:
    sysp = LLM.system_prompt()
    sys_tok = _est_tokens(sysp)
    local = cfg["llm"]["local"]
    return {
        "schema": BUNDLE_SCHEMA,
        "source_tree_id": tree_id,
        "source_manifest_sha256": source_manifest_sha,
        "lab_stamp": SUB.lab_stamp_fingerprint(),
        "prompt_skeleton_fingerprint": LLM.skeleton_fingerprint(),
        "prompt_skeleton_token_estimate": LLM.skeleton_token_estimate(),
        "system_prompt": sysp,
        "system_prompt_sha256": sha256_text(sysp),
        "tool_schemas": {**SUB.tool_schema_disclosure(units),
                         "source": source_manifest.get("tool_schemas", {}).get("source"),
                         "vendor_manifest_sha256":
                             source_manifest.get("tool_schemas", {}).get("vendor_manifest_sha256")},
        "sampling": {"model": local["model"], "temperature": local["temperature"],
                     "top_p": local["top_p"], "top_k": local["top_k"],
                     "min_p": local["min_p"], "max_tokens": local["max_tokens"],
                     "parse_retries": local["parse_retries"]},
        "corpus": source_manifest.get("corpus"),
        "sampling_frame": {"subset_seed": source_manifest["sampling"]["subset_seed"],
                           "main_arm": source_manifest["sampling"]["main_arm"],
                           "n_units": source_manifest["n_units"],
                           "n_diagnoses": source_manifest["n_diagnoses"],
                           "n_diagnoses_ok": source_manifest["n_diagnoses_ok"]},
        "main_arm_excluded": {
            "reason_counts_by_cell": SUB.reason_counts(units),
            "n_excluded_no_trigger": sum(1 for u in units if u["reason"] == "no_trigger"),
            "n_excluded_armed_not_touched": sum(
                1 for u in units if u["reason"] == "armed_not_touched"),
            "n_excluded_dispatch_in_prefix": sum(
                1 for u in units if u["reason"] == "dispatch_in_prefix")},
        "arms": {"main": _arm_stats(main_rows, sys_tok),
                 "extended": {**_arm_stats(ext_rows, sys_tok),
                              "face": " = ",
                              "note": "****——T1-collect "}},
    }


def _tar_add(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def pack(manifest: dict[str, Any], main_rows: list[dict[str, Any]],
         ext_rows: list[dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    import gzip
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        _tar_add(tar, "manifest.json",
                 json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for name, rows in (("requests/main.jsonl", main_rows),
                           ("requests/extended.jsonl", ext_rows)):
            _tar_add(tar, name, "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return buf.getvalue()


def build(corpus: Path, cfg: dict[str, Any], tree: Path, *, extended: bool = True,
          ) -> tuple[dict[str, Any], bytes]:
    src_path = tree / "manifest.json"
    if not src_path.is_file():
        raise SystemExit("")
    src_text = src_path.read_text(encoding="utf-8")
    source_manifest = json.loads(src_text)
    main_arm = str(source_manifest["sampling"]["main_arm"])
    subset = source_manifest["subset_ids"]

    units = SUB.enumerate_units(corpus, main_arm, subset)
    if len(units) != source_manifest["n_diagnoses"]:
        raise SystemExit(
            "")
    sysp = LLM.system_prompt()
    system_sha = sha256_text(sysp)
    main_rows = list(main_arm_requests(units, system_sha))
    ext_rows = (list(extended_arm_requests(corpus, main_arm, subset, system_sha))
                if extended else [])
    if not main_rows:
        raise SystemExit("")
    if extended and not ext_rows:
        raise SystemExit("")
    empty = [r["run_id"] for r in (main_rows + ext_rows) if not r["n_tool_schemas"]]
    if empty:
        raise SystemExit(
            "")

    assert_rendered_rows_are_clean(main_rows + ext_rows)

    manifest = build_bundle_manifest(cfg, tree.name, source_manifest,
                                     sha256_text(src_text), main_rows, ext_rows, units)
    return manifest, pack(manifest, main_rows, ext_rows)


def assert_rendered_rows_are_clean(rows: list[dict[str, Any]]) -> None:
    from dar.accounting import DISPATCH_META_KEY
    from dar.diagnoser.base import GROUND_TRUTH_KEYS
    needles = sorted(GROUND_TRUTH_KEYS) + [DISPATCH_META_KEY]
    for r in rows:
        hits = [n for n in needles if n in r["user"]]
        if hits:
            raise SystemExit(
                "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument("--emit-provenance", action="store_true")
    args = ap.parse_args(argv)

    cfg = SUB.load_config()
    corpus = Path(args.data) if args.data else (LAB / cfg["measurement"]["corpus_root"])
    if not corpus.is_absolute():
        corpus = LAB / corpus
    tree = Path(args.tree)
    if not tree.is_absolute():
        tree = LAB / tree

    manifest, blob = build(corpus, cfg, tree, extended=not args.no_extended)
    sha = hashlib.sha256(blob).hexdigest()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = LAB / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"b6-input-{sha[:8]}.tar.gz"
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() != sha:
        raise SystemExit("")
    out.write_bytes(blob)

    m, e = manifest["arms"]["main"], manifest["arms"]["extended"]
    print(f"[t1-bundle] → {out}")
    print(f"[t1-bundle] sha256 = {sha}")
    if args.emit_provenance:
        emitter = ProductEmitter(tree.name, lab_stamp=manifest["lab_stamp"])
        emitter.records[str(out.relative_to(LAB))] = {
            "path": str(out.relative_to(LAB)), "sha256": sha, "bytes": len(blob),
            "lab_stamp": manifest["lab_stamp"]}
        emitter.emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
