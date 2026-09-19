#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import matrix
import run_matrix
import stamp_lab

from dar import config_snapshot as cs
from dar.b5_stats import is_replicate_row

MAX_NAMED = 12

FREEZE_DIR = LAB / "tests" / "data"
_FREEZE_FILES = ("prompt_freeze.json", "ga_semantic_freeze.json")


CHUNK_PROV_NAME = re.compile(r"lab_code_provenance\.chunk(\d+)\.json\Z")


def split_provenance_files(out_root: Path) -> tuple[list[Path], list[str]]:
    cands = sorted(out_root.glob("lab_code_provenance.chunk*.json"))
    ok = [p for p in cands if CHUNK_PROV_NAME.fullmatch(p.name)]
    near = [p.name for p in cands if not CHUNK_PROV_NAME.fullmatch(p.name)]
    return ok, near


def eval_relpath(row: dict, model: str) -> Path:
    p = run_matrix.inference_relpath(row, model)
    return p.parent.parent / "evaluations" / f"{row['tid']}_{row['mode']}_eval.json"


def _named(items: list[str]) -> str:
    shown = items[:MAX_NAMED]
    more = f" … {len(items)}" if len(items) > MAX_NAMED else ""
    return "; ".join(shown) + more


def _load_json(p: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"unparsable: {e.__class__.__name__}"


def audit(out_root: Path, manifest_path: Path, model: str, *,
          allow_missing_stamp: bool = False, freeze_dir: Path | None = None) -> dict:
    out_root = Path(out_root)
    freeze_dir = Path(freeze_dir) if freeze_dir is not None else FREEZE_DIR
    problems: list[str] = []
    notes: list[str] = []
    counts: dict[str, int] = {}

    try:
        man, rows = run_matrix.load_manifest(Path(manifest_path))
        matrix._assert_frozen_counts(rows)
        if man.get("seed_base") != matrix.SEED_BASE:
            problems.append("")
    except SystemExit as e:
        problems.append("")
        return {"green": False, "problems": problems, "notes": notes, "counts": counts,
                "expected_total": matrix.EXPECTED_TOTAL}
    counts["manifest_rows"] = len(rows)

    stamp_p = out_root / "prereg_stamp.json"
    if stamp_p.is_file():
        stamp, err = _load_json(stamp_p)
        if err:
            problems.append(f"② prereg_stamp.json {err}")
        else:
            if stamp.get("expected_total") != matrix.EXPECTED_TOTAL:
                problems.append("")
            if stamp.get("seed") != matrix.SEED_BASE:
                problems.append("")
    elif allow_missing_stamp:
        notes.append("prereg_stamp.json —— --allow-missing-stamp "
                     "")
    else:
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
        rel_e = eval_relpath(r, model)
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
        if (r["batch"] == "rep") != is_replicate_row(r):
            problems.append("")
        if r["batch"] == "rep" and r.get("replicate_rep") not in (1, 2, 3, 4):
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
            if doc.get("expected_total") != matrix.EXPECTED_TOTAL:
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
                     f"{_named(near_miss)}—— `quarantine-*/pre-rerun-backup/` ")
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

    return {"green": not problems, "problems": problems, "notes": notes,
            "counts": counts, "expected_total": matrix.EXPECTED_TOTAL}


def render_report(res: dict, out_root: Path, manifest_path: Path, model: str) -> str:
    lines = [
        "# B5 matrix_audit§8-4 ",
        f"- out_root: `{out_root}`",
        f"- manifest: `{manifest_path}`rows={res['counts'].get('manifest_rows', '?')}",
        f"- model: `{model}`",
        f"-  = matrix.EXPECTED_TOTAL = {res['expected_total']}[BD-13] ",
        f"- classify: {({k: v for k, v in res['counts'].items() if k != 'manifest_rows'})}",
        "",
        ("✅ ① ② ③resume  ④==+ "
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
    ap.add_argument("--freeze-dir", default=None)
    args = ap.parse_args()
    res = audit(Path(args.out_root), Path(args.manifest), args.model,
                allow_missing_stamp=args.allow_missing_stamp,
                freeze_dir=Path(args.freeze_dir) if args.freeze_dir else None)
    report = render_report(res, Path(args.out_root), Path(args.manifest), args.model)
    print(report, end="")
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
    return 0 if res["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
