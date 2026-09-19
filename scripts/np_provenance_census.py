#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from dar.labels import Label
from dar.np_trigger import np_trigger

_ARM_LABEL = {"fixed-transient": Label.TRANSIENT, "fixed-persistent": Label.PERSISTENT}

_UNARMED_TAILS = frozenset({"none"})
_UNARMED_TAIL_RE = re.compile(r"rep\d+\Z")


def _is_known_unarmed(tail: str) -> bool:
    return tail in _UNARMED_TAILS or _UNARMED_TAIL_RE.match(tail) is not None


def _arm_label(snapshot_arm: str, accepted: frozenset[str] = frozenset()) -> Label | None:
    tail = snapshot_arm.rsplit("/", 1)[-1]
    if tail in _ARM_LABEL:
        return _ARM_LABEL[tail]
    if _is_known_unarmed(tail) or snapshot_arm in accepted or tail in accepted:
        return None
    raise SystemExit(
        "")


def census(data: Path, allow_unarmed: frozenset[str] | None = None,
           allow_no_armed_arm: bool = False) -> dict:
    accepted = frozenset(allow_unarmed or ())
    rows: list[dict] = []
    seen_arms: set[str] = set()
    for snap in sorted(data.glob("*/*/qwen3-8b/fc/*/config_snapshot.json")):
        cell = snap.parent
        cat = cell.name
        arm = json.loads(snap.read_text(encoding="utf-8"))["arm"]
        seen_arms.add(arm)
        label = _arm_label(arm, accepted)
        for inf in sorted((cell / "inferences").glob("*_P0_inference.json")):
            tid = inf.name[: -len("_P0_inference.json")]
            msgs = json.loads(inf.read_text(encoding="utf-8"))["messages"]
            fired = any((m.get("metadata") or {}).get("dar_dispatch", {}).get("np_trigger")
                        for m in msgs)
            if label is None:
                state = "unarmed_label_np"
                victims: list[str] = []
            else:
                spec = np_trigger(cat, tid, label)
                victims = list(spec["victim_tools"]) if spec else []
                state = "fired" if fired else "armed_not_touched"
            rows.append({"arm": arm, "cat": cat, "tid": tid, "state": state,
                         "n_victims": len(victims)})
    if not seen_arms:
        raise SystemExit(
            "")
    unmatched = sorted(a for a in accepted if a not in seen_arms
                       and a not in {x.rsplit("/", 1)[-1] for x in seen_arms})
    if unmatched:
        pass
    if not allow_no_armed_arm and not any(_arm_label(a, accepted) for a in seen_arms):
        raise SystemExit(
            "")
    return {"data": str(data), "rows": rows,
            "accepted_unarmed": sorted(accepted),
            "arms_recognized": sorted(a for a in seen_arms if _arm_label(a, accepted)),
            "arms_unarmed": sorted(a for a in seen_arms if not _arm_label(a, accepted)),
            "summary": dict(Counter(r["state"] for r in rows)),
            "by_arm": {a: dict(Counter(r["state"] for r in rows if r["arm"] == a))
                       for a in sorted({r["arm"] for r in rows})}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--allow-unarmed", default="", metavar="ARM[,ARM…]")
    ap.add_argument("--allow-no-armed-arm", action="store_true")
    args = ap.parse_args()

    accepted = frozenset(a.strip() for a in args.allow_unarmed.split(",") if a.strip())
    out = census(args.data, allow_unarmed=accepted,
                 allow_no_armed_arm=args.allow_no_armed_arm)
    armed = sum(v for k, v in out["summary"].items() if k != "unarmed_label_np")
    for arm, d in out["by_arm"].items():
        print(f"  {arm}: " + "  ".join(f"{k}={v}" for k, v in sorted(d.items())))
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
