#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from dar.materiality import AXES, axis_materiality
from dar.taskmodel import dataset_root

OUT = LAB / "provenance" / "materiality_manifest.json"
OVERRIDES = LAB / "provenance" / "materiality_overrides.json"
CATS = ["c1", "c2", "c3", "c4"]


def _apply_overrides(body: dict) -> dict:
    if not OVERRIDES.is_file():
        raise SystemExit("")
    reg = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    applied, noop = [], []
    for o in reg.get("overrides") or []:
        cat, tid, ax = o["cat"], o["task_id"], o["axis"]
        cell = (body.get(cat) or {}).get(tid, {}).get(ax)
        if cell is None:
            raise SystemExit("")
        if o["any_material"] is not True:
            raise SystemExit("")
        tag = f"{cat}/{tid}/{ax}"
        if cell["any_material"] is True:
            noop.append(tag)
            continue
        cell["any_material"] = True
        cell["override"] = {"source": OVERRIDES.name, "reason": "F2_embedded_flow",
                            "machine_fields_pre_override": True,
                            "evidence": o["evidence"]}
        applied.append(tag)
    if noop:
        pass
    return {"source": OVERRIDES.name,
            "source_sha256": hashlib.sha256(OVERRIDES.read_bytes()).hexdigest(),
            "applied": applied, "noop": noop}


def build() -> dict:
    root = dataset_root() / "perturbed_tasks"
    body: dict = {}
    for cat in CATS:
        ids = sorted({p.name.rsplit("_P", 1)[0] for p in (root / cat).glob("*_P0.json")})
        if not ids:
            raise SystemExit("")
        body[cat] = {tid: {ax: axis_materiality(cat, tid, ax) for ax in AXES} for tid in ids}
    overrides = _apply_overrides(body)
    summary: dict = {}
    for cat in CATS:
        for ax in AXES:
            n = len(body[cat])
            mat = sum(1 for t in body[cat].values() if t[ax]["any_material"])
            summary[f"{cat}.{ax}"] = {"tasks": n, "any_material": mat, "non_material": n - mat}
    payload = {
        "schema": "dar-materiality-manifest/1",
        "definition": ("victim  ⟺  P0  path  Action "
                       " args** victim **——F1  2026-08-04"
                       " = transient(P1)/persistent(P2) victim "
                       " F2  overrides ——"
                       "** `any_material` ** `pair_frac`/"
                       "`victims[*].material`/`material_paths`/`matches` ****"
                       " `override.machine_fields_pre_override: true` "
                       " `any_material`**** frac/victims 2026-08-05 "
                       " lab/src/dar/materiality.py docstring §14.5 "),
        "dataset_root": str(dataset_root()),
        "overrides": overrides,
        "summary": summary,
        "tasks": body,
    }
    hashable = {k: v for k, v in payload.items() if k != "dataset_root"}
    blob = json.dumps(hashable, sort_keys=True, ensure_ascii=False).encode()
    payload["aggregate_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    payload = build()
    if args.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old["aggregate_sha256"] != payload["aggregate_sha256"]:
            return 1
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[materiality] → {OUT}")
    for k, v in payload["summary"].items():
        pass
    print(f"[materiality] aggregate={payload['aggregate_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
