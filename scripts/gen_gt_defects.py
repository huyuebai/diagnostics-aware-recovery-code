#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from dar.materiality import _query_obtainable
from dar.taskmodel import dataset_root, is_action, iter_paths, leaf_scalars, load_task

OUT = LAB / "provenance" / "gt_defect_register.json"
CATS = ["c1", "c2", "c3", "c4"]

MANUAL = {
    "C3_task_001": {"forms": ["G_hard", "Z"], "note": "querybook Conference Room A canonical  build_room_booking_requestavailability=0 "},
    "C4_task_094": {"forms": ["G_hard"], "note": "build a text report and email it to mecanonical  build"},
    "C4_task_002": {"forms": ["V", "Q_hard"], "note": "place_sell_order_from_converted_quote  0.92 185.5 USD ≈170.7 EURquery "},
    "C4_task_100": {"forms": ["T", "Z"], "note": "s6  Conference Room As7-8  Meeting Room 1  ticketavailability=0 "},
    "C3_task_061": {"forms": ["Q_hard"], "note": " 'coupon_code'  coupon  query"},
    "C4_task_015": {"forms": ["Q_soft", "R"], "note": "choose the smart lock, and unlock lock_main valid path  set_lock_state args  lock_main ⇒ GA keep"},
    "C2_task_019": {"forms": ["G_soft"], "note": "coupon SAVE15  GT "},
    "C3_task_067": {"forms": ["G_soft"], "note": "note the currency as EUR GT "},
    "C4_task_061": {"forms": ["R"], "note": "ritual "},
    "C4_task_064": {"forms": ["R"], "note": ""},
    "C4_task_084": {"forms": ["R"], "note": ""},
    "C4_task_099": {"forms": ["R"], "note": ""},
}

EXCLUDE_FORMS = {"G_hard", "V", "T", "Q_hard", "N"}


def detect_n(task_p0: dict) -> bool:
    n_by_path = [sum(1 for s in steps if is_action(s.get("tool_name") or ""))
                 for _, steps in iter_paths(task_p0)]
    return bool(n_by_path) and all(n == 0 for n in n_by_path)


def detect_z(task_p0: dict) -> list[dict]:
    hits = []
    for pid, steps in iter_paths(task_p0):
        for i, s in enumerate(steps):
            tool = s.get("tool_name") or ""
            out = s.get("output") or {}
            if "availability" not in tool or not isinstance(out, dict):
                continue
            empty = out.get("total_slots") == 0 or (
                isinstance(out.get("available_slots"), list) and not out["available_slots"])
            if empty and any(is_action(t.get("tool_name") or "") for t in steps[i + 1:]):
                hits.append({"path": pid, "tool": tool})
    return hits


def detect_r(task_p0: dict, query_text: str) -> list[dict]:
    hits = []
    for pid, steps in iter_paths(task_p0):
        for i, s in enumerate(steps):
            tool = s.get("tool_name") or ""
            if not tool.startswith(("select_", "resolve_")):
                continue
            values = {v for v in leaf_scalars(s.get("output"))
                      if not _query_obtainable(v, query_text)}
            if not values:
                continue
            used = False
            for t in steps[i + 1:]:
                if values & leaf_scalars(t.get("arguments")):
                    used = True
                    break
            if not used:
                hits.append({"path": pid, "tool": tool})
    return hits


def build() -> dict:
    root = dataset_root() / "perturbed_tasks"
    tasks: dict = {}
    seen_ids: set[str] = set()
    for cat in CATS:
        ids = sorted({p.name.rsplit("_P", 1)[0] for p in (root / cat).glob("*_P0.json")})
        seen_ids.update(ids)
        for tid in ids:
            p0 = load_task(cat, tid, "P0")
            query_text = json.dumps(p0.get("user_input"), ensure_ascii=False)
            entry: dict = {"cat": cat, "forms": [], "auto": {}, "note": ""}
            if detect_n(p0):
                entry["forms"].append("N")
                entry["auto"]["N"] = {"paths_all_zero_action": True}
            z = detect_z(p0)
            if z:
                entry["forms"].append("Z")
                entry["auto"]["Z"] = z
            r = detect_r(p0, query_text)
            if r:
                entry["forms"].append("R")
                entry["auto"]["R"] = r
            manual = MANUAL.get(tid)
            if manual:
                entry["forms"] = sorted(set(entry["forms"]) | set(manual["forms"]))
                entry["note"] = manual["note"]
                entry["manual_source"] = "devils-advocate-survey-2026-07-22"
            if not entry["forms"]:
                continue
            if set(entry["forms"]) & EXCLUDE_FORMS:
                entry["disposition"] = "exclude"
            elif "Z" in entry["forms"]:
                avail_tools = {h["tool"] for h in entry["auto"].get("Z", [])}
                cells = {}
                for axis, mode in [("transient", "P1"), ("persistent", "P2")]:
                    tm = load_task(cat, tid, mode)
                    victims = set()
                    for _, steps in iter_paths(tm):
                        victims |= {s.get("tool_name") for s in steps if s.get("is_perturbed")}
                    cells[axis] = "exclude_cell" if victims & avail_tools else "keep"
                entry["disposition"] = "exclude_cell_if_victim_is_availability"
                entry["cell_disposition"] = cells
            else:
                entry["disposition"] = "keep_with_note"
            tasks[tid] = entry

    orphans = sorted(set(MANUAL) - seen_ids)
    if orphans:
        raise SystemExit("")

    summary: dict = {}
    for cat in CATS:
        rows = [e for e in tasks.values() if e["cat"] == cat]
        summary[cat] = {
            "flagged": len(rows),
            "exclude": sum(1 for e in rows if e["disposition"] == "exclude"),
            "cell_conditional": sum(1 for e in rows
                                    if e["disposition"] == "exclude_cell_if_victim_is_availability"),
            "keep_with_note": sum(1 for e in rows if e["disposition"] == "keep_with_note"),
        }
    payload = {
        "schema": "dar-gt-defect-register/1",
        "policy": "G_hard/V/T/Q_hard/N=exclude; Z=exclude_cell_if_victim_is_availability; "
                  "R/G_soft/Q_soft=keep_with_note docstringB4 ",
        "summary": summary,
        "tasks": tasks,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    payload["aggregate_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payload = build()
    if args.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old["aggregate_sha256"] != payload["aggregate_sha256"]:
            print("[gt-defects] DRIFT")
            return 1
        return 0
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[gt-defects] → {OUT}")
    for cat, v in payload["summary"].items():
        print(f"  {cat}: flagged {v['flagged']}  exclude {v['exclude']}  "
              f"cell-conditional {v['cell_conditional']}  keep {v['keep_with_note']}")
    print(f"[gt-defects] aggregate={payload['aggregate_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
