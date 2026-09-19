#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import bd8_matrix
import bd8_run
import run_matrix
import stamp_lab

from matrix_audit import (
    CHUNK_PROV_NAME,
    FREEZE_DIR,
    _FREEZE_FILES,
    _load_json,
    _named,
    split_provenance_files,
)

from dar import config_snapshot as cs

PREREG_PATH = LAB / "docs" / "bd8_ablation_prereg.md"


def expected_cell_keys() -> dict[tuple[str, str], int]:
    keys: dict[tuple[str, str], int] = {}
    for truth, framing, instruction, dispatch, spec in bd8_matrix.CELLS:
        probe = {"batch": "bd8", "force_prompt": bd8_matrix.FORCE_PROMPT,
                 "dispatch": dispatch, "router": spec}
        sub = run_matrix.subtree_of(probe)
        for cat in bd8_matrix.CATS:
            keys[(sub, cat)] = bd8_matrix.EXPECTED_TASKS_PER_CAT
    if len(keys) != len(bd8_matrix.CELLS) * len(bd8_matrix.CATS):
        raise bd8_matrix.Bd8MatrixError(
            "")
    return keys


def _disk_cell_counts(out_root: Path, model: str) -> dict[tuple[str, str], int]:
    safe = model.replace("/", "_").replace(":", "_")
    got: dict[tuple[str, str], int] = {}
    for p in out_root.glob("*/*/*/fc/*/inferences/*_inference.json"):
        rel = p.relative_to(out_root)
        sub = "%s/%s" % (rel.parts[0], rel.parts[1])
        if rel.parts[2] != safe:
            continue
        got[(sub, rel.parts[4])] = got.get((sub, rel.parts[4]), 0) + 1
    return got


def audit(out_root: Path, manifest_path: Path, model: str, *,
          allow_missing_stamp: bool = False, freeze_dir: Path | None = None,
          prereg: Path | None = None, allow_unbound_prereg: bool = False) -> dict:
    out_root = Path(out_root)
    freeze_dir = Path(freeze_dir) if freeze_dir is not None else FREEZE_DIR
    prereg = Path(prereg) if prereg is not None else PREREG_PATH
    problems: list[str] = []
    notes: list[str] = []
    counts: dict[str, int] = {}

    try:
        man, rows = bd8_run.load_manifest(Path(manifest_path))
    except SystemExit as e:
        problems.append("")
        return {"green": False, "problems": problems, "notes": notes, "counts": counts,
                "expected_total": bd8_matrix.EXPECTED_TOTAL}
    counts["manifest_rows"] = len(rows)

    for sp in sorted(out_root.glob("prereg_stamp*.json")) or [out_root / "prereg_stamp.json"]:
        if not sp.is_file():
            if allow_missing_stamp:
                notes.append("prereg_stamp.json —— --allow-missing-stamp "
                             "")
            else:
                problems.append(
                    "")
            continue
        stamp, err = _load_json(sp)
        if err:
            problems.append(f"② {sp.name} {err}")
            continue
        if stamp.get("expected_total") != bd8_matrix.EXPECTED_TOTAL:
            problems.append("")
        if stamp.get("seed") != bd8_matrix.SEED_BASE:
            problems.append("")

    missing_inf: list[str] = []
    bad_inf: list[str] = []
    missing_ev: list[str] = []
    bad_ev: list[str] = []
    ident_bad: list[str] = []
    by_cell_missing: dict[tuple[str, str], int] = {}
    expected_inf: set[Path] = set()
    expected_ev: set[Path] = set()
    cell_dirs: set[Path] = set()
    for r in rows:
        rel_i = run_matrix.inference_relpath(r, model)
        rel_e = run_matrix.eval_relpath(r, model)
        expected_inf.add(rel_i)
        expected_ev.add(rel_e)
        cell_dirs.add((out_root / rel_i).parent.parent)
        for kind, rel, miss, bad in (("inference", rel_i, missing_inf, bad_inf),
                                     ("evaluation", rel_e, missing_ev, bad_ev)):
            doc, err = _load_json(out_root / rel)
            if err == "missing":
                miss.append(r["run_id"])
                k = (kind, r["cell_id"])
                by_cell_missing[k] = by_cell_missing.get(k, 0) + 1
            elif err:
                bad.append("")
            elif doc.get("task_id") != r["tid"] or doc.get("mode") != r["mode"]:
                ident_bad.append("")
        if r["batch"] != "bd8":
            problems.append("")

    def _per_cell(kind: str) -> str:
        return ", ".join(f"{c} ×{n}" for (k, c), n in sorted(by_cell_missing.items())
                         if k == kind)
    if missing_inf:
        problems.append("")
    if missing_ev:
        problems.append("")
    if bad_inf:
        problems.append("")
    if bad_ev:
        problems.append("")
    if ident_bad:
        problems.append("")

    want_keys = expected_cell_keys()
    got_keys = _disk_cell_counts(out_root, model)
    counts["disk_cells"] = len(got_keys)
    missing_keys = sorted(k for k in want_keys if k not in got_keys)
    extra_keys = sorted(k for k in got_keys if k not in want_keys)
    wrong_n = sorted((k, got_keys[k], want_keys[k]) for k in want_keys
                     if k in got_keys and got_keys[k] != want_keys[k])
    if missing_keys:
        problems.append("")
    if extra_keys:
        problems.append("")
    if wrong_n:
        problems.append("")

    missing_freeze = [f for f in _FREEZE_FILES if not (freeze_dir / f).is_file()]
    if missing_freeze:
        problems.append(
            "")
    for cell in sorted(cell_dirs):
        rel = cell.relative_to(out_root)
        snap, err = _load_json(cell / "config_snapshot.json")
        if err:
            problems.append(f"② {rel}/config_snapshot.json {err}")
        elif not isinstance(snap, dict):
            problems.append("")
        elif not missing_freeze:
            try:
                frozen_bad = cs.verify_snapshot_frozen(snap, freeze_dir)
            except Exception as e:
                frozen_bad = [""]
            if frozen_bad:
                problems.append("")
        if not (cell / "run_manifest.jsonl").is_file():
            problems.append("")
        else:
            try:
                bij = cs.audit_bijection(cell / "inferences", cell / "run_manifest.jsonl")
            except Exception as e:
                bij = [""]
            if bij:
                problems.append("")

    staging = sorted(out_root.glob("staging_manifest.chunk*of*.json"))
    if not staging:
        problems.append("")
    else:
        union: dict[str, str] = {}
        dup: list[str] = []
        for sp in staging:
            doc, err = _load_json(sp)
            if err:
                problems.append(f"② {sp.name} {err}")
                continue
            if doc.get("expected_total") != bd8_matrix.EXPECTED_TOTAL:
                problems.append("")
            for rid in (doc.get("chunk") or {}).get("chunk_run_ids") or []:
                if rid in union:
                    dup.append(rid)
                union[rid] = sp.name
        man_ids = {r["run_id"] for r in rows}
        if dup:
            problems.append("")
        if set(union) != man_ids:
            miss = sorted(man_ids - set(union))
            extra = sorted(set(union) - man_ids)
            if miss:
                problems.append("")
            if extra:
                problems.append("")

    orphan_inf = sorted(str(p.relative_to(out_root))
                        for p in out_root.glob("*/*/*/fc/*/inferences/*_inference.json")
                        if p.relative_to(out_root) not in expected_inf)
    orphan_ev = sorted(str(p.relative_to(out_root))
                       for p in out_root.glob("*/*/*/fc/*/evaluations/*_eval.json")
                       if p.relative_to(out_root) not in expected_ev)
    if orphan_inf:
        problems.append("")
    if orphan_ev:
        problems.append("")
    quarantined = sorted(out_root.glob("*/*/*/fc/*/inferences/*.corrupt-*"))
    if quarantined:
        notes.append(f" *.corrupt-* × {len(quarantined)}"
                     "run_matrix  run ")

    classified = run_matrix.classify(rows, out_root, model)
    c = run_matrix.status_counts(classified)
    counts.update(c)
    units = run_matrix.plan_units(classified)
    if c["pending"] or c["corrupt"] or units:
        pend = [e["row"]["run_id"] for e in classified if e["status"] != "done"]
        problems.append("")

    prov_files, near_miss = split_provenance_files(out_root)
    if near_miss:
        notes.append(f"⑤  {len(near_miss)} **** chunk "
                     f"{_named(near_miss)}")
    prov_by_chunk: dict[str, str] = {}
    if not prov_files:
        problems.append("")
    else:
        for pp in prov_files:
            doc, err = _load_json(pp)
            if err:
                problems.append(f"⑤ {pp.name} {err}")
                continue
            fp, files = doc.get("fingerprint_sha256"), doc.get("files")
            if not fp or not isinstance(files, dict) or not files:
                problems.append("")
                continue
            if stamp_lab.aggregate(files) != fp:
                problems.append("")
                continue
            prov_by_chunk[CHUNK_PROV_NAME.fullmatch(pp.name).group(1)] = fp
        staged_chunks = {sp.name.split("chunk")[-1].split("of")[0] for sp in staging}
        miss_prov = sorted(staged_chunks - set(prov_by_chunk))
        if miss_prov:
            problems.append("")
        fps = sorted(set(prov_by_chunk.values()))
        if len(fps) > 1:
            notes.append("⑤  **** chunk "
                         "****" + "".join(
                             f"chunk{k}={v[:16]}" for k, v in sorted(prov_by_chunk.items())))
        elif fps:
            notes.append(f"⑤ {fps[0][:16]}{len(prov_by_chunk)} chunk")

    if not prereg.is_file():
        problems.append("")
    else:
        want_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
        stamps = sorted(out_root.glob("prereg_stamp*.json"))
        checked = 0
        bound_ok = True
        if not stamps:
            if allow_unbound_prereg:
                notes.append("⑥  prereg_stamp*.json ⇒ ↔****"
                             "—— --allow-unbound-prereg ")
            else:
                problems.append(
                    "")
            bound_ok = False
        for sp in stamps:
            doc, err = _load_json(sp)
            if err:
                bound_ok = False
                continue
            checked += 1
            got = doc.get("prereg_sha256")
            if got != want_sha:
                bound_ok = False
                problems.append(
                    "")
        if bound_ok and checked:
            notes.append(f"⑥ {checked}  stamp  "
                         f"{want_sha[:16]}…{prereg.name} ")

    return {"green": not problems, "problems": problems, "notes": notes,
            "counts": counts, "expected_total": bd8_matrix.EXPECTED_TOTAL}


def render_report(res: dict, out_root: Path, manifest_path: Path, model: str) -> str:
    lines = [
        "# [BD-8] matrix_audit_bd8C-2 ",
        f"- out_root: `{out_root}`",
        f"- manifest: `{manifest_path}`rows={res['counts'].get('manifest_rows', '?')}",
        f"- model: `{model}`",
        f"-  = bd8_matrix.EXPECTED_TOTAL = {res['expected_total']}"
        f"[BD-13]  = "
        f"{len(bd8_matrix.CELLS)}  × {len(bd8_matrix.CATS)} cat × "
        f"{bd8_matrix.EXPECTED_TASKS_PER_CAT}",
        f"- classify: {({k: v for k, v in res['counts'].items() if k != 'manifest_rows'})}",
        "",
        ("✅ ① ②+ ③resume  "
         "④==+ ⑤ ⑥ "
         if res["green"]
         else f"🔴 {len(res['problems'])} "),
    ]
    for p in res["problems"]:
        lines.append(f"  - 🔴 {p}")
    for n in res["notes"]:
        lines.append(f"  - ℹ️ {n}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-missing-stamp", action="store_true")
    ap.add_argument("--allow-unbound-prereg", action="store_true")
    ap.add_argument("--freeze-dir", default=None)
    ap.add_argument("--prereg", default=None)
    args = ap.parse_args()
    res = audit(Path(args.out_root), Path(args.manifest), args.model,
                allow_missing_stamp=args.allow_missing_stamp,
                freeze_dir=Path(args.freeze_dir) if args.freeze_dir else None,
                prereg=Path(args.prereg) if args.prereg else None,
                allow_unbound_prereg=args.allow_unbound_prereg)
    report = render_report(res, Path(args.out_root), Path(args.manifest), args.model)
    print(report, end="")
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
    return 0 if res["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
