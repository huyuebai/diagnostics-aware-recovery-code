#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED_BASE = 20260730
REPLICATE_SAMPLE_SEED = SEED_BASE
EXPECTED_TOTAL = 7600
EXPECTED_TASKS_PER_CAT = 100
EXPECTED_PER_CELL = 400
REPLICATE_PER_CAT = 25
REPLICATE_REPS = 4
EXPECTED_REPLICATE_ROWS = 400

MANIFEST_SCHEMA = "b5_run_manifest/v1"

CATS = ("c1", "c2", "c3", "c4")
TRUTHS = ("NP", "P1", "P2", "P3", "P4")
LABELS = ("transient", "persistent", "np")
TRUTH_TO_MODE = {"NP": "P0", "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}
LABEL_TO_DISPATCH = {"transient": "fixed:transient", "persistent": "fixed:persistent",
                     "np": "none"}
NP_TRIGGER_SIBLING = {"transient": "P1", "persistent": "P2"}
PLAIN_CELLS = (("NP", "np"), ("P2", "np"), ("P2", "persistent"))
REPLICATE_CELL = ("NP", "np")


class MatrixError(SystemExit):
    pass


def enumerate_tasks(dataset_root: Path) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    for cat in CATS:
        d = Path(dataset_root) / "perturbed_tasks" / cat
        ids = sorted({p.name.rsplit("_P", 1)[0] for p in d.glob("*_P0.json")})
        if len(ids) != EXPECTED_TASKS_PER_CAT:
            raise MatrixError(
                "")
        tasks[cat] = ids
    return tasks


def _validate_task_ids(task_ids: dict[str, list[str]]) -> dict[str, list[str]]:
    if sorted(task_ids) != sorted(CATS):
        raise MatrixError("")
    out = {}
    for cat in CATS:
        ids = sorted(task_ids[cat])
        if len(ids) != EXPECTED_TASKS_PER_CAT or len(set(ids)) != EXPECTED_TASKS_PER_CAT:
            raise MatrixError(
                "")
        out[cat] = ids
    return out


def replicate_task_ids(tasks: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cat in CATS:
        pool = sorted(tasks[cat])
        out[cat] = sorted(random.Random(REPLICATE_SAMPLE_SEED).sample(pool, REPLICATE_PER_CAT))
    return out


def _row(i: int, batch: str, truth: str, label: str, cat: str, tid: str,
         force_prompt: str, rep: int = 0) -> dict:
    mode = TRUTH_TO_MODE[truth]
    cell = f"{batch}:{truth}x{label}"
    run_id = f"{cell}:{tid}" + (f":rep{rep}" if rep else "")
    return {
        "staging_index": i,
        "run_id": run_id,
        "cell_id": cell,
        "batch": batch,
        "truth": truth,
        "label": label,
        "dispatch": LABEL_TO_DISPATCH[label],
        "force_prompt": force_prompt,
        "cat": cat,
        "tid": tid,
        "mode": mode,
        "mode_file": f"{tid}_{mode}.json",
        "replicate_rep": rep,
        "np_trigger_source_sibling": (
            NP_TRIGGER_SIBLING[label] if truth == "NP" and label != "np" else None),
    }


def build_rows(tasks: dict[str, list[str]]) -> list[dict]:
    rep_ids = replicate_task_ids(tasks)
    rows: list[dict] = []
    for cat in CATS:
        rep_set = set(rep_ids[cat])
        for tid in tasks[cat]:
            i = len(rows)
            for truth in TRUTHS:
                rows.append(_row(len(rows), "main", truth, "np", cat, tid, "fault_aware"))
            if tid in rep_set:
                for rep in range(1, REPLICATE_REPS + 1):
                    rows.append(_row(len(rows), "rep", *REPLICATE_CELL, cat, tid,
                                     "fault_aware", rep=rep))
            for label in ("transient", "persistent"):
                for truth in TRUTHS:
                    rows.append(_row(len(rows), "main", truth, label, cat, tid, "fault_aware"))
            for truth, label in PLAIN_CELLS:
                rows.append(_row(len(rows), "plain", truth, label, cat, tid, "p0"))
            assert rows[i]["tid"] == tid
    _assert_frozen_counts(rows)
    return rows


def _assert_frozen_counts(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise MatrixError("")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cell_id"]] = counts.get(r["cell_id"], 0) + 1
    expected = {f"main:{t}x{l}": EXPECTED_PER_CELL for t in TRUTHS for l in LABELS}
    expected.update({f"plain:{t}x{l}": EXPECTED_PER_CELL for t, l in PLAIN_CELLS})
    expected[f"rep:{REPLICATE_CELL[0]}x{REPLICATE_CELL[1]}"] = EXPECTED_REPLICATE_ROWS
    if counts != expected:
        diff = {k: (counts.get(k), expected.get(k))
                for k in sorted(set(counts) | set(expected)) if counts.get(k) != expected.get(k)}
        raise MatrixError("")
    for idx, r in enumerate(rows):
        if r["staging_index"] != idx:
            raise MatrixError("")


def build_manifest(dataset_root: Path | str | None = None,
                   task_ids: dict[str, list[str]] | None = None) -> dict:
    if (dataset_root is None) == (task_ids is None):
        raise MatrixError("")
    tasks = enumerate_tasks(Path(dataset_root)) if dataset_root else _validate_task_ids(task_ids)
    rows = build_rows(tasks)
    cell_counts: dict[str, int] = {}
    for r in rows:
        cell_counts[r["cell_id"]] = cell_counts.get(r["cell_id"], 0) + 1
    return {
        "schema": MANIFEST_SCHEMA,
        "seed_base": SEED_BASE,
        "replicate_sample_seed": REPLICATE_SAMPLE_SEED,
        "expected_total": EXPECTED_TOTAL,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "cell_counts": cell_counts,
        "runs": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, help=".../ToolMaze_dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = build_manifest(dataset_root=args.dataset_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[matrix] seed_base={SEED_BASE} cells={len(man['cell_counts'])} "
          f"rep_tasks={REPLICATE_PER_CAT}×{len(CATS)}×{REPLICATE_REPS}rep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
