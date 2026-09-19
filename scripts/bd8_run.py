#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import bd8_matrix
import run_matrix
from run_matrix import MatrixRunError


def validate_manifest(man: dict) -> list[dict]:
    if man.get("schema") != bd8_matrix.MANIFEST_SCHEMA:
        raise MatrixRunError(
            f"[bd8-run] REFUSING: schema {man.get('schema')!r} != {bd8_matrix.MANIFEST_SCHEMA}")
    rows = man.get("runs") or []
    total = bd8_matrix.EXPECTED_TOTAL
    if man.get("expected_total") != total or len(rows) != total:
        raise MatrixRunError(
            "")
    if man.get("seed_base") != bd8_matrix.SEED_BASE:
        raise MatrixRunError(
            "")
    known = set(bd8_matrix.CELLS and [c for _, _, _, _, c in bd8_matrix.CELLS])
    seen_ids: set[str] = set()
    seen_blocks: set[tuple[str, str]] = set()
    prev: tuple[str, str] | None = None
    for idx, r in enumerate(rows):
        if r["staging_index"] != idx:
            raise MatrixRunError(
                "")
        if not r.get("router"):
            raise MatrixRunError(
                "")
        if r["router"] not in known:
            raise MatrixRunError(
                "")
        try:
            want_dispatch, want_spec = bd8_matrix.expected_wiring(
                r["truth"], r["framing_label"], r["instruction_label"])
        except SystemExit as e:
            raise MatrixRunError("")
        if (r["dispatch"], r["router"]) != (want_dispatch, want_spec):
            raise MatrixRunError(
                "")
        if r.get("cell_id") != bd8_matrix.cell_id(
                r["truth"], r["framing_label"], r["instruction_label"]):
            raise MatrixRunError(
                "")
        if r["run_id"] in seen_ids:
            raise MatrixRunError("")
        seen_ids.add(r["run_id"])
        blk = (r["cat"], r["tid"])
        if blk != prev:
            if blk in seen_blocks:
                raise MatrixRunError(
                    "")
            if prev is not None:
                seen_blocks.add(prev)
            prev = blk
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cell_id"]] = counts.get(r["cell_id"], 0) + 1
    want = {bd8_matrix.cell_id(t, f, i): bd8_matrix.EXPECTED_PER_CELL
            for t, f, i, _, _ in bd8_matrix.CELLS}
    if counts != want:
        raise MatrixRunError(
            "")
    if man.get("cell_counts") != counts:
        raise MatrixRunError(
            "")
    return rows


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    import json
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    return man, validate_manifest(man)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--op-config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--out-root", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--chunk", default="1/1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--i-am-on-hpc", action="store_true")
    ap.add_argument("--selfproof-only", action="store_true")
    args = ap.parse_args()

    man, rows = load_manifest(Path(args.manifest))
    chunk = run_matrix.parse_chunk(args.chunk)
    rows_chunk = run_matrix.chunk_rows(rows, *chunk)

    if args.emit_only:
        return run_matrix._emit_only(args, rows)
    if args.selfproof_only:
        return run_matrix._selfproof_only(args, rows_chunk)
    if args.dry_run:
        model = run_matrix._raw_model_name(Path(args.config))
        classified = run_matrix.classify(rows_chunk, Path(args.out_root), model, require_eval=True)
        units = run_matrix.plan_units(classified)
        specs = sorted({u.get("router") for u in units})
        if units and any(u.get("router") is None for u in units):
            raise MatrixRunError("")
        return 0
    return run_matrix._exec_chunk(args, man, rows_chunk, chunk)


if __name__ == "__main__":
    raise SystemExit(main())
