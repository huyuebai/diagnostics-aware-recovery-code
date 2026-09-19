#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

from b6_products import ProductEmitter, sha256_text
from dar.diagnoser import llm as LLM
from dar.diagnoser import ml as ML
from dar.diagnoser import trace_adapter as TA
from dar.labels import TRUE_TYPE_TO_LABEL, Label, true_type_from_mode
from dar.np_trigger import np_trigger
from dar.taskmodel import load_task, tool_specs_for_task

MODES = ("P0", "P1", "P2", "P3", "P4")
CATS = ("c1", "c2", "c3", "c4")
NP_AXES = (("transient", Label.TRANSIENT), ("persistent", Label.PERSISTENT))
MAIN_BATCH_ARMS = ("fa/none", "fa/fixed-transient", "fa/fixed-persistent")


def load_config(path: Path | None = None) -> dict[str, Any]:
    import yaml
    p = path or (LAB / "configs" / "b6_diagnosers.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def lab_stamp_fingerprint() -> str:
    p = LAB / "provenance" / "lab_code_provenance.json"
    if not p.is_file():
        raise SystemExit("")
    stamp = json.loads(p.read_text(encoding="utf-8"))
    fp = stamp.get("fingerprint_sha256")
    if not fp:
        raise SystemExit("")
    import stamp_lab
    actual = stamp_lab.aggregate(stamp_lab.hash_scope(LAB))
    if actual != fp:
        raise SystemExit(
            "")
    return str(fp)


def arm_root(corpus_root: Path, arm: str) -> Path:
    return corpus_root / arm / "qwen3-8b" / "fc"


def iter_inference_files(corpus_root: Path, arm: str) -> Iterator[tuple[str, str, str, Path]]:
    root = arm_root(corpus_root, arm)
    if not root.is_dir():
        raise SystemExit("")
    for cat in CATS:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            raise SystemExit("")
        for f in sorted((cat_dir / "inferences").glob("*_inference.json")):
            stem = f.name[: -len("_inference.json")]
            base, _, mode = stem.rpartition("_")
            if mode not in MODES:
                raise SystemExit("")
            yield cat, base, mode, f


def cell_pools(corpus_root: Path, arm: str) -> dict[tuple[str, str], list[str]]:
    pools: dict[tuple[str, str], list[str]] = {}
    for cat, base, mode, _ in iter_inference_files(corpus_root, arm):
        pools.setdefault((cat, mode), []).append(base)
    return {k: sorted(v) for k, v in sorted(pools.items())}


def cell_seed(subset_seed: int, cat: str, mode: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{subset_seed}:{cat}:{mode}".encode()).digest()[:8], "big")


def sample_cell(pool: list[str], *, subset_seed: int, cat: str, mode: str,
                k: int) -> list[str]:
    pool = sorted(pool)
    if len(pool) < k:
        raise SystemExit("")
    return sorted(random.Random(cell_seed(subset_seed, cat, mode)).sample(pool, k))


def build_subset(corpus_root: Path, cfg: dict[str, Any],
                 arm: str = "fa/none") -> dict[str, list[str]]:
    m = cfg["measurement"]
    pools = cell_pools(corpus_root, arm)
    out: dict[str, list[str]] = {}
    for cat in CATS:
        for mode in MODES:
            pool = pools.get((cat, mode))
            if not pool:
                raise SystemExit("")
            out[f"{cat}|{mode}"] = sample_cell(pool, subset_seed=int(m["subset_seed"]),
                                               cat=cat, mode=mode,
                                               k=int(m["tasks_per_cell"]))
    return out


def task_description(cat: str, base_id: str, mode: str) -> str:
    task = load_task(cat, base_id, mode)
    user_input = task.get("user_input") or {}
    return str(user_input.get("query", task.get("task_description", "")) or "")


def diagnosis_units(corpus_root: Path, arm: str, cat: str, base_id: str, mode: str, *,
                    with_task_description: bool = True,
                    with_tool_schemas: bool = True) -> list[dict[str, Any]]:
    path = arm_root(corpus_root, arm) / cat / "inferences" / f"{base_id}_{mode}_inference.json"
    inf = TA.load_inference(path)
    if inf["mode"] != mode:
        raise SystemExit("")
    desc = task_description(cat, base_id, mode) if with_task_description else ""
    if with_tool_schemas:
        sel = tool_specs_for_task(cat, base_id, mode)
        schemas: tuple[dict[str, Any], ...] = sel.specs
        dropped: tuple[str, ...] = sel.dropped
    else:
        schemas, dropped = (), ()
    label = TRUE_TYPE_TO_LABEL[true_type_from_mode(mode)]

    def _row(axis: str, obs: Any, meta: dict[str, Any]) -> dict[str, Any]:
        return {"run_id": f"{arm}|{cat}|{base_id}|{mode}|{axis}", "arm": arm, "cat": cat,
                "tid": base_id, "mode": mode, "axis": axis, "true_label": label.value,
                "reason": meta["reason"], "trigger_index": meta.get("trigger_index"),
                "n_tool_schemas": len(schemas), "dropped_tools": list(dropped),
                "obs": obs}

    if mode != "P0":
        obs, meta = TA.build_detection_oracle_input(inf, task_description=desc,
                                                    tool_schemas=schemas)
        return [_row("main", obs, meta)]

    rows = []
    for axis_name, axis_label in NP_AXES:
        spec = np_trigger(cat, base_id, axis_label)
        if spec is None:
            raise SystemExit("")
        obs, meta = TA.build_detection_oracle_input(
            inf, victim_tools=spec["victim_tools"], task_description=desc,
            tool_schemas=schemas)
        rows.append(_row(axis_name, obs, meta))
    return rows


def enumerate_units(corpus_root: Path, arm: str,
                    subset: dict[str, list[str]] | None = None,
                    *, with_task_description: bool = True,
                    with_tool_schemas: bool = True) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for cat, base, mode, _ in iter_inference_files(corpus_root, arm):
        if subset is not None and base not in subset.get(f"{cat}|{mode}", ()):
            continue
        units.extend(diagnosis_units(corpus_root, arm, cat, base, mode,
                                     with_task_description=with_task_description,
                                     with_tool_schemas=with_tool_schemas))
    return units


def tool_schema_disclosure(units: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    by_cell: dict[str, int] = {}
    names: dict[str, int] = {}
    n_schemas: list[int] = []
    for u in units:
        n_schemas.append(int(u.get("n_tool_schemas") or 0))
        for name in (u.get("dropped_tools") or []):
            total += 1
            key = f"{u['cat']}|{u['mode']}|{u['axis']}"
            by_cell[key] = by_cell.get(key, 0) + 1
            names[name] = names.get(name, 0) + 1
    return {"n_dropped_total": total,
            "dropped_by_cell": dict(sorted(by_cell.items())),
            "dropped_tool_names": dict(sorted(names.items())),
            "n_tool_schemas_min": min(n_schemas) if n_schemas else 0,
            "n_tool_schemas_max": max(n_schemas) if n_schemas else 0,
            "n_tool_schemas_mean": (sum(n_schemas) / len(n_schemas)) if n_schemas else 0.0}


def reason_counts(units: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for u in units:
        key = f"{u['cat']}|{u['mode']}|{u['axis']}"
        out.setdefault(key, {})
        out[key][u["reason"]] = out[key].get(u["reason"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _vendor_manifest_sha() -> str:
    p = LAB / "provenance" / "toolmaze_manifest.json"
    if not p.is_file():
        raise SystemExit("")
    return sha256_text(p.read_text(encoding="utf-8"))


def corpus_digest(corpus_root: Path, arm: str) -> dict[str, Any]:
    h = hashlib.sha256()
    n = 0
    for _cat, _base, _mode, f in iter_inference_files(corpus_root, arm):
        h.update(f.name.encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode())
        h.update(b"\n")
        n += 1
    return {"arm": arm, "n_files": n, "files_digest_sha256": h.hexdigest()}


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(LAB.parent),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "(git-head-unavailable)"


def build_manifest(corpus_root: Path, cfg: dict[str, Any], tree_id: str, *,
                   full_scan: bool = True) -> dict[str, Any]:
    m = cfg["measurement"]
    main_arm = str(m["main_arm"])
    subset = build_subset(corpus_root, cfg, arm=main_arm)
    sub_units = enumerate_units(corpus_root, main_arm, subset)
    n_ok = sum(1 for u in sub_units if u["reason"] == "ok")

    disclosure: dict[str, Any] = {}
    if full_scan:
        for arm in MAIN_BATCH_ARMS:
            arm_units = enumerate_units(corpus_root, arm, None, with_task_description=False,
                                        with_tool_schemas=False)
            disclosure[arm] = {
                "face": " head-to-head ",
                "n_units_attempted": len(arm_units),
                "n_units_ok": sum(1 for u in arm_units if u["reason"] == "ok"),
                "reason_counts_by_cell": reason_counts(arm_units),
                "n_excluded_dispatch_in_prefix": sum(
                    1 for u in arm_units if u["reason"] == "dispatch_in_prefix"),
            }

    return {
        "schema": "b6_manifest/v1",
        "tree_id": tree_id,
        "git_head": git_head(),
        "lab_stamp": lab_stamp_fingerprint(),
        "prompt_skeleton_fingerprint": LLM.skeleton_fingerprint(),
        "prompt_skeleton_token_estimate": LLM.skeleton_token_estimate(),
        "tool_schemas": {**tool_schema_disclosure(sub_units),
                         "source": "vendor/toolmaze/tools/definitions/*.yaml"
                                   "::paradigms.function_call.spec",
                         "selection_rule": "vendor sandbox._build_tool_definitions "
                                           "complexity∈{C2,C3,C4}  valid_paths "
                                           " execution_trace****",
                         "vendor_manifest_sha256": _vendor_manifest_sha()},
        "ml": {"feature_names": list(ML.FEATURE_NAMES),
               "label_order": [l.value for l in ML.LABEL_ORDER],
               "fold_seed": int(cfg["ml"]["fold_seed"]),
               "k_folds": int(cfg["ml"]["k_folds"]),
               "l2": float(cfg["ml"]["l2"]),
               "learning_rate": float(cfg["ml"]["learning_rate"]),
               "max_epochs": int(cfg["ml"]["max_epochs"]),
               "tol": float(cfg["ml"]["tol"])},
        "config": {"path": "configs/b6_diagnosers.yaml",
                   "sha256": sha256_text((LAB / "configs" / "b6_diagnosers.yaml")
                                         .read_text(encoding="utf-8"))},
        "sampling": {
            "subset_seed": int(m["subset_seed"]),
            "tasks_per_cell": int(m["tasks_per_cell"]),
            "cells": [f"{c}|{md}" for c in CATS for md in MODES],
            "cell_seeds": {f"{c}|{md}": cell_seed(int(m["subset_seed"]), c, md)
                           for c in CATS for md in MODES},
            "main_arm": m["main_arm"],
        },
        "subset_ids": subset,
        "n_units": sum(len(v) for v in subset.values()),
        "n_units_expected": int(m["expected_units"]),
        "n_diagnoses": len(sub_units),
        "n_diagnoses_ok": n_ok,
        "n_diagnoses_max": int(m["max_expected_diagnoses"]),
        "np_row_axes": [a for a, _ in NP_AXES],
        "subset_reason_counts_by_cell": reason_counts(sub_units),
        "n_excluded_dispatch_in_prefix": sum(
            1 for u in sub_units if u["reason"] == "dispatch_in_prefix"),
        "n_excluded_armed_not_touched": sum(
            1 for u in sub_units if u["reason"] == "armed_not_touched"),
        "n_excluded_no_trigger": sum(1 for u in sub_units if u["reason"] == "no_trigger"),
        "corpus": {"root": str(corpus_root.relative_to(LAB)) if corpus_root.is_relative_to(LAB)
                   else str(corpus_root),
                   "arms": [corpus_digest(corpus_root, a) for a in MAIN_BATCH_ARMS]},
        "disclosure_face": disclosure,
    }


def assert_frame(manifest: dict[str, Any]) -> None:
    if manifest["n_units"] != manifest["n_units_expected"]:
        raise SystemExit("")
    if manifest["n_diagnoses"] > manifest["n_diagnoses_max"]:
        raise SystemExit("")
    arms = set((manifest.get("disclosure_face") or {}))
    if arms != set(MAIN_BATCH_ARMS):
        raise SystemExit(
            "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-full-scan", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    corpus = Path(args.data) if args.data else (LAB / cfg["measurement"]["corpus_root"])
    if not corpus.is_absolute():
        corpus = LAB / corpus
    out = Path(args.out)
    if not out.is_absolute():
        out = LAB / out
    tree_id = out.name

    emitter = ProductEmitter(tree_id, lab_stamp=lab_stamp_fingerprint())
    manifest = build_manifest(corpus, cfg, tree_id, full_scan=not args.no_full_scan)
    assert_frame(manifest)
    emitter.write_json(out / "manifest.json", manifest)
    emitter.emit()
    print(f"[b6-subset] manifest → {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
