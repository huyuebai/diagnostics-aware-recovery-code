#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_items as I5
import b5_judge as J
import bd8_matrix
import bd8_run
import matrix_audit_bd8
import run_matrix

DEFAULT_MODEL = "qwen3-8b"
MANIFEST_GLOB = "bd8_run_manifest.chunk*.json"
DEFAULT_MANIFEST = "bd8_run_manifest.chunk1.json"

_LIST_CAP = I5._LIST_CAP


class Bd8ItemsError(I5.B5ItemsError):
    pass


def _named(items) -> str:
    return I5._named(items)


def _arm_name(cell: tuple[str, str, str, str, str]) -> str:
    return run_matrix.subtree_of(bd8_matrix._row(0, cell, bd8_matrix.CATS[0], "_probe"))


ARM_MODE: dict[str, str] = {}
ARM_LABEL: dict[str, str] = {}
for _cell in bd8_matrix.CELLS:
    _arm = _arm_name(_cell)
    if _arm in ARM_MODE:
        raise Bd8ItemsError(
            "")
    ARM_MODE[_arm] = _cell[0]
    ARM_LABEL[_arm] = bd8_matrix.cell_id(_cell[0], _cell[1], _cell[2])
ALL_ARMS: tuple[str, ...] = tuple(ARM_MODE)
del _cell, _arm


def _assert_arm_labels_are_bijective() -> None:
    if len(set(ARM_LABEL.values())) != len(ARM_LABEL):
        raise Bd8ItemsError(
            "")
    clash = sorted(set(ARM_LABEL.values()) & set(ARM_LABEL))
    if clash:
        raise Bd8ItemsError(
            "")


_assert_arm_labels_are_bijective()


def load_bd8_manifest(data: Path, manifest: Path | None = None) -> tuple[dict, list[dict]]:
    data = Path(data)
    chosen = Path(manifest) if manifest is not None else data / DEFAULT_MANIFEST
    if not chosen.is_file():
        raise Bd8ItemsError("")
    siblings = sorted(data.glob(MANIFEST_GLOB))
    if not siblings:
        raise Bd8ItemsError("")
    ref = chosen.read_bytes()
    diff = [str(p.name) for p in siblings if p != chosen and p.read_bytes() != ref]
    if diff:
        raise Bd8ItemsError(
            "")
    man, rows = bd8_run.load_manifest(chosen)
    return man, rows


def assert_four_arm_rowsets(rows: list[dict]) -> None:
    if not rows:
        raise Bd8ItemsError("")
    by_arm = I5._rowsets_by_arm(rows)
    if set(by_arm) != set(ALL_ARMS):
        raise Bd8ItemsError(
            "")
    tasks = sorted({(r["cat"], r["tid"]) for r in rows})
    n_cat: dict[str, int] = {}
    for cat, _tid in tasks:
        n_cat[cat] = n_cat.get(cat, 0) + 1
    if sorted(n_cat) != sorted(bd8_matrix.CATS) or set(n_cat.values()) != {
            bd8_matrix.EXPECTED_TASKS_PER_CAT}:
        raise Bd8ItemsError(
            "")
    problems: list[str] = []
    for arm in ALL_ARMS:
        mode = ARM_MODE[arm]
        want = {(cat, tid, mode) for cat, tid in tasks}
        problems += I5._cmp(arm, by_arm[arm], want, "")
    by_truth: dict[str, list[str]] = {}
    for arm in ALL_ARMS:
        by_truth.setdefault(ARM_MODE[arm], []).append(arm)
    for truth, arms in sorted(by_truth.items()):
        if len(arms) != 2:
            problems.append("")
            continue
        a, b = arms
        pa = {(c, t) for c, t, _ in by_arm[a]}
        pb = {(c, t) for c, t, _ in by_arm[b]}
        if pa != pb:
            problems.append(
                "")
    if problems:
        raise Bd8ItemsError(
            "")


def assert_manifest_is_canonical_expansion(rows: list[dict]) -> None:
    tasks_by_cat = {c: sorted({r["tid"] for r in rows if r["cat"] == c})
                    for c in {r["cat"] for r in rows}}
    canon = bd8_matrix.build_rows(tasks_by_cat)
    if canon != rows:
        if len(canon) != len(rows):
            raise Bd8ItemsError("")
        for i, (a, b) in enumerate(zip(canon, rows)):
            if a != b:
                keys = sorted(set(a) | set(b))
                d = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
                raise Bd8ItemsError(
                    "")


def assert_tree_matches_manifest(rows: list[dict], data: Path, model: str) -> None:
    data = Path(data)
    want_by_arm = I5._rowsets_by_arm(rows)
    row_by_arm = {run_matrix.subtree_of(r): r for r in rows}
    problems: list[str] = []
    for arm in ALL_ARMS:
        want_fc = (data / run_matrix.inference_relpath(row_by_arm[arm], model)).parents[2]
        try:
            cells = list(B4._cells(data, arm))
        except FileNotFoundError as e:
            raise Bd8ItemsError(
                "")
        if not cells:
            raise Bd8ItemsError(
                "")
        if cells[0].parent != want_fc:
            raise Bd8ItemsError(
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
        raise Bd8ItemsError(
            "")


def assert_corpus_identity_matches_local(data: Path) -> None:
    from run_dispatch import PROVENANCE, _read_aggregate
    data = Path(data)
    local: dict[str, str] = {}
    for field, fname, _why in I5._IDENTITY_FIELDS:
        p = PROVENANCE / fname
        if not p.is_file():
            raise Bd8ItemsError(
                "")
        local[field] = _read_aggregate(p)
    seen: dict[str, dict[Any, list[str]]] = {f: {} for f, _, _ in I5._IDENTITY_FIELDS}
    for arm in ALL_ARMS:
        for cell in B4._cells(data, arm):
            snap = json.loads((cell / "config_snapshot.json").read_text(encoding="utf-8"))
            for field, _, _why in I5._IDENTITY_FIELDS:
                seen[field].setdefault(snap.get(field), []).append(f"{arm}/{cell.name}")
    problems: list[str] = []
    for field, fname, why in I5._IDENTITY_FIELDS:
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
        raise Bd8ItemsError(
            "")


def relabel_arms(ga_rows: list[dict], items: list[dict]) -> None:
    if len(ga_rows) != bd8_matrix.EXPECTED_TOTAL:
        raise Bd8ItemsError(
            "")
    if not items:
        raise Bd8ItemsError("")
    if {r["arm"] for r in ga_rows} != set(ALL_ARMS):
        raise Bd8ItemsError(
            "")
    for coll, keep_subtree in ((ga_rows, True), (items, False)):
        for r in coll:
            sub = r["arm"]
            label = ARM_LABEL.get(sub)
            if label is None:
                raise Bd8ItemsError(
                    "")
            if keep_subtree:
                r["arm_subtree"] = sub
            r["arm"] = label


def assert_seam3_under_labels(ga_rows: list[dict]) -> None:
    if len(ga_rows) != bd8_matrix.EXPECTED_TOTAL:
        raise Bd8ItemsError(
            "")
    labels = {r["arm"] for r in ga_rows}
    if labels != set(ARM_LABEL.values()):
        raise Bd8ItemsError(
            "")
    fwd: dict[tuple, str] = {}
    rev: dict[str, tuple] = {}
    for r in ga_rows:
        k = (r["arm"], r["cat"], r["tid"], r["mode"])
        rid = r["run_id"]
        if k in fwd:
            raise Bd8ItemsError(
                "")
        if rid in rev:
            raise Bd8ItemsError(
                "")
        fwd[k] = rid
        rev[rid] = k


def assert_custom_ids_fit(items: list[dict]) -> None:
    if not items:
        raise Bd8ItemsError("")
    labels = {it["arm"] for it in items}
    if not labels <= set(ARM_LABEL.values()) or not labels:
        raise Bd8ItemsError(
            "")
    seen: dict[str, int] = {}
    dup: list[str] = []
    longest = ""
    for i, it in enumerate(items):
        cid = B4._custom_id(it)
        B4._custom_id(it, recon=True)
        if len(cid) > len(longest):
            longest = cid
        if cid in seen:
            dup.append("")
        seen[cid] = i
    if dup:
        raise Bd8ItemsError(
            "")


def run(data: Path, manifest: Path | None, model: str) -> int:
    I5.assert_code_stamped()
    I5.assert_round_not_started(data)
    _man, rows = load_bd8_manifest(data, manifest)
    assert_four_arm_rowsets(rows)
    assert_manifest_is_canonical_expansion(rows)
    assert_tree_matches_manifest(rows, data, model)
    assert_corpus_identity_matches_local(data)
    man_path = Path(manifest) if manifest else Path(data) / DEFAULT_MANIFEST
    res = matrix_audit_bd8.audit(data, man_path, model)
    if not res["green"]:
        raise Bd8ItemsError(
            "")
    for n in res.get("notes") or []:
        print(f"[bd8-items] matrix_audit_bd8 ℹ️ {n}")
    rid_of = I5.runid_map(rows)

    mat = B4._materiality(data)
    defects = B4._defects()
    descs = B4._tool_descriptions()
    ga_rows, items = I5.emit_layer_a(rows, data, model, rid_of,
                                     mat=mat, defects=defects, descs=descs)
    relabel_arms(ga_rows, items)
    assert_seam3_under_labels(ga_rows)
    assert_custom_ids_fit(items)
    I5.write_outputs(data, ga_rows, items)
    I5.print_summary(ga_rows, items)

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
