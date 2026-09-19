#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED_BASE = 20260730
EXPECTED_TASKS_PER_CAT = 100
EXPECTED_PER_CELL = 400
EXPECTED_TOTAL = 1600

MANIFEST_SCHEMA = "bd8_run_manifest/v1"

CATS = ("c1", "c2", "c3", "c4")
FORCE_PROMPT = "fault_aware"

CELLS: tuple[tuple[str, str, str, str, str], ...] = (
    ("P1", "transient", "persistent", "fixed:persistent",
     "decoupled:content=truth,action=diagnosis"),
    ("P1", "persistent", "transient", "fixed:persistent",
     "decoupled:content=diagnosis,action=truth"),
    ("P2", "transient", "persistent", "fixed:transient",
     "decoupled:content=diagnosis,action=truth"),
    ("P2", "persistent", "transient", "fixed:transient",
     "decoupled:content=truth,action=diagnosis"),
)

TRUTH_TO_CORRECT_LABEL = {"P1": "transient", "P2": "persistent"}


class Bd8MatrixError(SystemExit):
    pass


def expected_wiring(truth: str, framing: str, instruction: str) -> tuple[str, str]:
    try:
        correct = TRUTH_TO_CORRECT_LABEL[truth]
    except KeyError:
        raise Bd8MatrixError(
            "") from None
    if framing == instruction:
        raise Bd8MatrixError(
            "")
    if framing == correct:
        content_from, action_from, fixed = "truth", "diagnosis", instruction
    elif instruction == correct:
        content_from, action_from, fixed = "diagnosis", "truth", framing
    else:
        raise Bd8MatrixError(
            "")
    return f"fixed:{fixed}", f"decoupled:content={content_from},action={action_from}"


def _assert_cells_are_consistent() -> None:
    seen = set()
    for truth, framing, instruction, dispatch, spec in CELLS:
        want_dispatch, want_spec = expected_wiring(truth, framing, instruction)
        if (spec, dispatch) != (want_spec, want_dispatch):
            raise Bd8MatrixError(
                "")
        key = (truth, framing, instruction)
        if key in seen:
            raise Bd8MatrixError("")
        seen.add(key)
    if len(CELLS) * EXPECTED_PER_CELL != EXPECTED_TOTAL:
        raise Bd8MatrixError(
            "")


_assert_cells_are_consistent()


def enumerate_tasks(dataset_root: Path) -> dict[str, list[str]]:
    tasks: dict[str, list[str]] = {}
    for cat in CATS:
        d = Path(dataset_root) / "perturbed_tasks" / cat
        ids = sorted({p.name.rsplit("_P", 1)[0] for p in d.glob("*_P0.json")})
        if len(ids) != EXPECTED_TASKS_PER_CAT:
            raise Bd8MatrixError(
                "")
        tasks[cat] = ids
    return tasks


def _validate_task_ids(task_ids: dict[str, list[str]]) -> dict[str, list[str]]:
    if sorted(task_ids) != sorted(CATS):
        raise Bd8MatrixError(
            "")
    out = {}
    for cat in CATS:
        ids = sorted(task_ids[cat])
        if len(ids) != EXPECTED_TASKS_PER_CAT or len(set(ids)) != EXPECTED_TASKS_PER_CAT:
            raise Bd8MatrixError(
                "")
        out[cat] = ids
    return out


def cell_id(truth: str, framing: str, instruction: str) -> str:
    return f"bd8:{truth}xF-{framing}.I-{instruction}"


def wrong_label(truth: str, framing: str, instruction: str) -> str:
    correct = TRUTH_TO_CORRECT_LABEL[truth]
    wrong = [x for x in (framing, instruction) if x != correct]
    if len(wrong) != 1:
        raise Bd8MatrixError(
            "")
    return wrong[0]


def _row(i: int, cell: tuple[str, str, str, str, str], cat: str, tid: str) -> dict:
    truth, framing, instruction, dispatch, spec = cell
    cid = cell_id(truth, framing, instruction)
    return {
        "staging_index": i,
        "run_id": f"{cid}:{tid}",
        "cell_id": cid,
        "batch": "bd8",
        "truth": truth,
        "framing_label": framing,
        "instruction_label": instruction,
        "dispatch": dispatch,
        "router": spec,
        "force_prompt": FORCE_PROMPT,
        "cat": cat,
        "tid": tid,
        "mode": truth,
        "mode_file": f"{tid}_{truth}.json",
        "b5_anchor_cell": f"main:{truth}x{wrong_label(truth, framing, instruction)}",
    }


def build_rows(tasks: dict[str, list[str]]) -> list[dict]:
    rows: list[dict] = []
    for cat in CATS:
        for tid in tasks[cat]:
            i = len(rows)
            for cell in CELLS:
                rows.append(_row(len(rows), cell, cat, tid))
            assert rows[i]["tid"] == tid
    _assert_frozen_counts(rows)
    return rows


def _assert_frozen_counts(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise Bd8MatrixError(
            "")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cell_id"]] = counts.get(r["cell_id"], 0) + 1
    expected = {cell_id(t, f, i): EXPECTED_PER_CELL for t, f, i, _, _ in CELLS}
    if counts != expected:
        diff = {k: (counts.get(k), expected.get(k))
                for k in sorted(set(counts) | set(expected)) if counts.get(k) != expected.get(k)}
        raise Bd8MatrixError("")
    for idx, r in enumerate(rows):
        if r["staging_index"] != idx:
            raise Bd8MatrixError(
                "")


def build_manifest(dataset_root: Path | str | None = None,
                   task_ids: dict[str, list[str]] | None = None) -> dict:
    if (dataset_root is None) == (task_ids is None):
        raise Bd8MatrixError("")
    tasks = enumerate_tasks(Path(dataset_root)) if dataset_root else _validate_task_ids(task_ids)
    rows = build_rows(tasks)
    cell_counts: dict[str, int] = {}
    for r in rows:
        cell_counts[r["cell_id"]] = cell_counts.get(r["cell_id"], 0) + 1
    return {
        "schema": MANIFEST_SCHEMA,
        "seed_base": SEED_BASE,
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
    print(f"[bd8-matrix] seed_base={SEED_BASE} cells={len(man['cell_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
