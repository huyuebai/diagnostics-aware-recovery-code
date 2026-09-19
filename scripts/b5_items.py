#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_judge as J
import matrix
import matrix_audit
import run_matrix
import stamp_lab

from dar.taskmodel import load_task

DEFAULT_MODEL = "qwen3-8b"
MANIFEST_GLOB = "b5_run_manifest.chunk*.json"
DEFAULT_MANIFEST = "b5_run_manifest.chunk1.json"
LAYER_A_FILENAME = "ga_layerA.jsonl"

_LIST_CAP = 12


class B5ItemsError(SystemExit):
    pass


def _named(items: Iterable[Any]) -> str:
    s = sorted(str(x) for x in items)
    head = s[:_LIST_CAP]
    return ", ".join(head) + (f" …( {len(s)} )" if len(s) > len(head) else "")


def _arm_name(batch: str, truth: str, label: str, force_prompt: str, rep: int = 0) -> str:
    row = matrix._row(0, batch, truth, label, matrix.CATS[0], "_probe", force_prompt, rep)
    return run_matrix.subtree_of(row)


def _plain_arm_modes() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for truth, label in matrix.PLAIN_CELLS:
        out.setdefault(_arm_name("plain", truth, label, "p0"), set()).add(
            matrix.TRUTH_TO_MODE[truth])
    return {a: frozenset(m) for a, m in out.items()}


ARMS_FA = tuple(_arm_name("main", matrix.TRUTHS[0], label, "fault_aware")
                for label in matrix.LABELS)
FA_MODES = frozenset(matrix.TRUTH_TO_MODE[t] for t in matrix.TRUTHS)
ARMS_PLAIN = _plain_arm_modes()
ARMS_REP = tuple(_arm_name("rep", *matrix.REPLICATE_CELL, "fault_aware", rep)
                 for rep in range(1, matrix.REPLICATE_REPS + 1))
REP_MODE = matrix.TRUTH_TO_MODE[matrix.REPLICATE_CELL[0]]
ARM_REP_ORIGIN = _arm_name("main", *matrix.REPLICATE_CELL, "fault_aware")
ALL_ARMS = (*ARMS_FA, *sorted(ARMS_PLAIN), *ARMS_REP)


def assert_code_stamped(lab_root: Path = LAB) -> None:
    rc = stamp_lab.do_check(str(lab_root))
    if rc != 0:
        raise B5ItemsError(
            "")


_STARTED_MARKERS: tuple[tuple[str, str], ...] = (
    (J.CHUNK_MANIFEST_FILENAME, ""),
    ("chunks", ""),
    (J.VERDICTS_FILENAME, ""),
    (J.RECON_FILENAME, ""),
    ("batch_submit.json", ""),
    ("batch_input.jsonl", ""),
)


def assert_round_not_started(data: Path) -> None:
    data = Path(data)
    hits: list[str] = []
    for name, why in _STARTED_MARKERS:
        p = data / name
        if name == "chunks":
            if p.is_dir() and any(p.iterdir()):
                hits.append("")
        elif p.exists():
            hits.append("")
    if hits:
        raise B5ItemsError(
            "")


def load_b5_manifest(data: Path, manifest: Path | None = None) -> tuple[dict, list[dict]]:
    data = Path(data)
    chosen = Path(manifest) if manifest is not None else data / DEFAULT_MANIFEST
    if not chosen.is_file():
        raise B5ItemsError("")
    siblings = sorted(data.glob(MANIFEST_GLOB))
    if not siblings:
        raise B5ItemsError("")
    ref = chosen.read_bytes()
    diff = [str(p.name) for p in siblings if p != chosen and p.read_bytes() != ref]
    if diff:
        raise B5ItemsError(
            "")
    man, rows = run_matrix.load_manifest(chosen)
    matrix._assert_frozen_counts(rows)
    if man.get("seed_base") != matrix.SEED_BASE:
        raise B5ItemsError("")
    return man, rows


def _rowsets_by_arm(rows: list[dict]) -> dict[str, set[tuple[str, str, str]]]:
    out: dict[str, set[tuple[str, str, str]]] = {}
    for r in rows:
        out.setdefault(run_matrix.subtree_of(r), set()).add((r["cat"], r["tid"], r["mode"]))
    return out


def _fmt_rows(rs: Iterable[tuple[str, str, str]]) -> str:
    return _named("/".join(x) for x in rs)


def _cmp(arm: str, got: set, want: set, role: str) -> list[str]:
    miss, extra = want - got, got - want
    if not (miss or extra):
        return []
    return [""]


def assert_nine_arm_rowsets(rows: list[dict]) -> None:
    if not rows:
        raise B5ItemsError("")
    by_arm = _rowsets_by_arm(rows)
    if set(by_arm) != set(ALL_ARMS):
        raise B5ItemsError(
            "")

    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r["cat"], set()).add(r["tid"])
    tasks_by_cat = {c: sorted(v) for c, v in seen.items()}
    if sorted(tasks_by_cat) != sorted(matrix.CATS):
        raise B5ItemsError("")
    bad_n = {c: len(v) for c, v in tasks_by_cat.items()
             if len(v) != matrix.EXPECTED_TASKS_PER_CAT}
    if bad_n:
        raise B5ItemsError("")
    all_tasks = {(c, t) for c, ts in tasks_by_cat.items() for t in ts}

    problems: list[str] = []
    want_fa = {(c, t, m) for c, t in all_tasks for m in FA_MODES}
    for arm in ARMS_FA:
        problems += _cmp(arm, by_arm[arm], want_fa, "")
    for a, b in itertools.combinations(ARMS_FA, 2):
        only_a, only_b = by_arm[a] - by_arm[b], by_arm[b] - by_arm[a]
        if only_a or only_b:
            problems.append(
                "")
    for arm, modes in sorted(ARMS_PLAIN.items()):
        want = {(c, t, m) for c, t in all_tasks for m in modes}
        problems += _cmp(arm, by_arm[arm], want, "")
    rep_ids = matrix.replicate_task_ids(tasks_by_cat)
    want_rep = {(c, t, REP_MODE) for c, ts in rep_ids.items() for t in ts}
    for arm in ARMS_REP:
        problems += _cmp(arm, by_arm[arm], want_rep,
                         "")
    for a, b in itertools.combinations(ARMS_REP, 2):
        if by_arm[a] != by_arm[b]:
            problems.append("")
    if problems:
        raise B5ItemsError(
            "")


def assert_manifest_is_canonical_expansion(rows: list[dict]) -> None:
    tasks_by_cat = {c: sorted({r["tid"] for r in rows if r["cat"] == c})
                    for c in {r["cat"] for r in rows}}
    canon = matrix.build_rows(tasks_by_cat)
    if canon != rows:
        if len(canon) != len(rows):
            raise B5ItemsError("")
        for i, (a, b) in enumerate(zip(canon, rows)):
            if a != b:
                keys = sorted(set(a) | set(b))
                d = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
                raise B5ItemsError(
                    "")


def assert_tree_matches_manifest(rows: list[dict], data: Path, model: str) -> None:
    data = Path(data)
    want_by_arm = _rowsets_by_arm(rows)
    row_by_arm = {run_matrix.subtree_of(r): r for r in rows}
    problems: list[str] = []
    for arm in ALL_ARMS:
        want_fc = (data / run_matrix.inference_relpath(row_by_arm[arm], model)).parents[2]
        try:
            cells = list(B4._cells(data, arm))
        except FileNotFoundError as e:
            raise B5ItemsError(
                "")
        if not cells:
            raise B5ItemsError(
                "")
        if cells[0].parent != want_fc:
            raise B5ItemsError(
                "")
        disk = {(cat, tid, mode) for cat, tid, mode, _ in B4._iter_runs(data, arm)}
        miss, orphan = want_by_arm[arm] - disk, disk - want_by_arm[arm]
        if miss:
            problems.append("")
        if orphan:
            problems.append("")
        for cell in cells:
            sp = cell / "config_snapshot.json"
            if not sp.is_file():
                problems.append("")
                continue
            got = (json.loads(sp.read_text(encoding="utf-8")) or {}).get("arm")
            if got != arm:
                problems.append("")
    if problems:
        raise B5ItemsError(
            "")


_IDENTITY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("dataset_aggregate", "toolmaze_dataset_manifest.json",
     ""),
    ("vendor_aggregate", "toolmaze_manifest.json",
     ""),
)


def assert_corpus_identity_matches_local(data: Path) -> None:
    from run_dispatch import PROVENANCE, _read_aggregate
    data = Path(data)
    local: dict[str, str] = {}
    for field, fname, _why in _IDENTITY_FIELDS:
        p = PROVENANCE / fname
        if not p.is_file():
            raise B5ItemsError(
                "")
        local[field] = _read_aggregate(p)
    seen: dict[str, dict[Any, list[str]]] = {f: {} for f, _, _ in _IDENTITY_FIELDS}
    for arm in ALL_ARMS:
        for cell in B4._cells(data, arm):
            snap = json.loads((cell / "config_snapshot.json").read_text(encoding="utf-8"))
            for field, _, _why in _IDENTITY_FIELDS:
                seen[field].setdefault(snap.get(field), []).append(f"{arm}/{cell.name}")
    problems: list[str] = []
    for field, fname, why in _IDENTITY_FIELDS:
        vals = seen[field]
        if len(vals) != 1:
            problems.append(
                "")
            continue
        got = next(iter(vals))
        if got != local[field]:
            problems.append(
                "")
    if problems:
        raise B5ItemsError(
            "")


def runid_map(rows: list[dict]) -> dict[tuple[str, str, str, str], str]:
    fwd: dict[tuple[str, str, str, str], str] = {}
    rev: dict[str, tuple[str, str, str, str]] = {}
    for r in rows:
        k = (run_matrix.subtree_of(r), r["cat"], r["tid"], r["mode"])
        rid = r["run_id"]
        if k in fwd:
            raise B5ItemsError(
                "")
        if rid in rev:
            raise B5ItemsError(
                "")
        fwd[k] = rid
        rev[rid] = k
    return fwd


def emit_layer_a(rows: list[dict], data: Path, model: str,
                 rid_of: dict[tuple[str, str, str, str], str], *,
                 mat: dict, defects: dict, descs: dict[str, str],
                 task_loader: Callable[[str, str, str], dict] = load_task,
                 message_loader: Callable[[Path], list] = B4._load_messages,
                 progress_every: int = 500) -> tuple[list[dict], list[dict]]:
    ga_rows: list[dict] = []
    items: list[dict] = []
    for n, r in enumerate(rows, 1):
        cat, tid, mode = r["cat"], r["tid"], r["mode"]
        arm = run_matrix.subtree_of(r)
        rid = rid_of[(arm, cat, tid, mode)]
        msgs = message_loader(data / run_matrix.inference_relpath(r, model))
        p0 = task_loader(cat, tid, "P0")
        task_mode = p0 if mode == "P0" else task_loader(cat, tid, mode)
        row, its = B4.layer_a_row_and_items(arm, cat, tid, mode, msgs, p0, task_mode,
                                            mat=mat, defects=defects, descs=descs)
        row["run_id"] = rid
        ga_rows.append(row)
        for it in its:
            it["run_id"] = rid
        items.extend(its)
        if progress_every and n % progress_every == 0:
            pass
    return ga_rows, items


def _render(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def write_outputs(data: Path, ga_rows: list[dict], items: list[dict]) -> bool:
    data = Path(data)
    if not items:
        raise B5ItemsError(
            "")
    payload = {LAYER_A_FILENAME: _render(ga_rows), J.ITEMS_FILENAME: _render(items)}
    drift = [n for n, text in payload.items()
             if (data / n).exists() and (data / n).read_text(encoding="utf-8") != text]
    if drift:
        raise B5ItemsError(
            "")
    wrote = False
    for name, text in payload.items():
        p = data / name
        nbytes = len(text.encode("utf-8"))
        if p.exists():
            continue
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
        wrote = True
    return wrote


def print_summary(ga_rows: list[dict], items: list[dict]) -> None:
    import collections
    print("| arm | mode | achieved | undetermined | not_achieved | n |")
    print("|---|---|---|---|---|---|")
    g: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in ga_rows:
        g[(r["arm"], r["mode"])].append(r)
    for k in sorted(g):
        gg = g[k]
        c = collections.Counter(r["verdict"] for r in gg)
        print(f"| {k[0]} | {k[1]} | {c['achieved']} | {c['undetermined']} | "
              f"{c['not_achieved']} | {len(gg)} |")
    rc = collections.Counter(r["reason_code"] for r in ga_rows)
    kinds = collections.Counter(it["kind"] for it in items)


def run(data: Path, manifest: Path | None, model: str) -> int:
    assert_code_stamped()
    assert_round_not_started(data)
    _man, rows = load_b5_manifest(data, manifest)
    assert_nine_arm_rowsets(rows)
    assert_manifest_is_canonical_expansion(rows)
    assert_tree_matches_manifest(rows, data, model)
    assert_corpus_identity_matches_local(data)
    man_path = Path(manifest) if manifest else Path(data) / DEFAULT_MANIFEST
    res = matrix_audit.audit(data, man_path, model)
    if not res["green"]:
        raise B5ItemsError(
            "")
    for n in res.get("notes") or []:
        print(f"[b5-items] matrix_audit ℹ️ {n}")
    rid_of = runid_map(rows)

    mat = B4._materiality(data)
    defects = B4._defects()
    descs = B4._tool_descriptions()
    ga_rows, items = emit_layer_a(rows, data, model, rid_of,
                                  mat=mat, defects=defects, descs=descs)
    write_outputs(data, ga_rows, items)
    print_summary(ga_rows, items)

    plan = J.plan_submit(Path(data))
    return 0


CLI_OPTION_STRINGS = frozenset({"-h", "--help", "--data", "--manifest", "--model"})


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = args.data if args.data.is_absolute() else (LAB / args.data)
    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = LAB / manifest
    return run(data, manifest, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
