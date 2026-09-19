#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B
from dar.ga_semantic import VERDICT_VOCAB

ITEMS_FILENAME = "judge_items.jsonl"
VERDICTS_FILENAME = "judge_verdicts.jsonl"
RECON_FILENAME = "judge_recon.jsonl"

_LIST_CAP = 20


def _fmt_names(names: list[str]) -> str:
    shown = names[:_LIST_CAP]
    more = len(names) - len(shown)
    return ", ".join(shown) + (f" …(+{more} more)" if more > 0 else "")


def _item_name(k: tuple[str, str, Any]) -> str:
    return f"{k[0]}|{k[1]}#{k[2]}"


def _read_jsonl_loud(path: Path) -> list[dict]:
    rows: list[dict] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit("")
    return rows


def iter_verdict_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise SystemExit(
            "")
    if not root.is_dir():
        raise SystemExit(
            "")
    linked: list[str] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        for d in sorted(dirnames):
            p = Path(dirpath) / d
            if p.is_symlink():
                linked.append(str(p))
    if linked:
        raise SystemExit(
            "")
    return sorted(root.rglob(VERDICTS_FILENAME))


def _rows_with_run_id(path: Path) -> list[dict]:
    rows = _read_jsonl_loud(path)
    missing = [i for i, r in enumerate(rows, 1) if not r.get("run_id")]
    if missing:
        raise SystemExit(
            "")
    return rows


def scan_existing_verdicts(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in iter_verdict_files(root):
        for r in _rows_with_run_id(p):
            srcs = out.setdefault(r["run_id"], [])
            if p not in srcs:
                srcs.append(p)
    return out


def assert_single_round(planned: Iterable[tuple[str, bool]],
                        existing: dict[str, list[Path]]) -> None:
    dup: dict[str, list[Path]] = {}
    for run_id, is_recon in planned:
        if is_recon:
            continue
        if run_id in existing and run_id not in dup:
            dup[run_id] = existing[run_id]
    if dup:
        names = sorted(dup)
        srcs = sorted({str(p) for ps in dup.values() for p in ps})
        raise SystemExit(
            "")


def load_b5_judge_items(data: Path) -> list[dict]:
    p = Path(data) / ITEMS_FILENAME
    if not p.exists():
        raise SystemExit("")
    rows = _read_jsonl_loud(p)
    if not rows:
        raise SystemExit("")
    bad = [i for i, r in enumerate(rows, 1) if not r.get("run_id")]
    if bad:
        raise SystemExit(
            "")
    return rows


def items_universe(items: Iterable[dict]) -> list[tuple[str, str, Any]]:
    seen: set[tuple[str, str, Any]] = set()
    dup: list[str] = []
    out: list[tuple[str, str, Any]] = []
    for i, it in enumerate(items, 1):
        for f in ("run_id", "path_id", "idx"):
            if f not in it:
                raise SystemExit("")
        k = (it["run_id"], it["path_id"], it["idx"])
        if k in seen:
            dup.append(_item_name(k))
            continue
        seen.add(k)
        out.append(k)
    if dup:
        raise SystemExit("")
    return out


def plan_submit(data: Path, scan_root: Path | None = None) -> dict[str, Any]:
    items = load_b5_judge_items(data)
    root = Path(scan_root) if scan_root else Path(data)
    existing = scan_existing_verdicts(root)
    assert_single_round(((it["run_id"], False) for it in items), existing)
    run_ids = sorted({it["run_id"] for it in items})
    return {"n_items": len(items), "n_runs": len(run_ids), "run_ids": run_ids,
            "item_keys": items_universe(items),
            "scan_root": str(root), "n_already_judged_in_scan": len(existing)}


CHUNK_MANIFEST_FILENAME = "batch_chunks.json"


def assert_force_wastes_nothing(data: Path) -> None:
    mpath = Path(data) / CHUNK_MANIFEST_FILENAME
    if not mpath.is_file():
        return
    try:
        man = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit("")
    spent = [c for c in (man.get("chunks") or [])
             if c.get("status") != "pending" or c.get("batch_id") is not None]
    if spent:
        names = [f"chunk{c.get('idx')}(status={c.get('status')!r},"
                 f"batch_id={c.get('batch_id')!r},usd={c.get('actual_usd')!r})"
                 for c in spent]
        total = sum(c.get("actual_usd") or 0.0 for c in (man.get("chunks") or []))
        raise SystemExit(
            "")


def attach_run_ids(data: Path) -> None:
    data = Path(data)
    key_to_rid: dict[tuple, str] = {}
    for it in load_b5_judge_items(data):
        k = B._judge_key(it)
        rid = it["run_id"]
        if key_to_rid.get(k, rid) != rid:
            raise SystemExit("")
        key_to_rid[k] = rid
    for fname in (VERDICTS_FILENAME, RECON_FILENAME):
        p = data / fname
        if not p.exists():
            continue
        rows = _read_jsonl_loud(p)
        changed = False
        for i, r in enumerate(rows, 1):
            try:
                k = B._judge_key(r)
            except KeyError as e:
                raise SystemExit("")
            rid = key_to_rid.get(k)
            if rid is None:
                raise SystemExit("")
            if r.get("run_id") is None:
                r["run_id"] = rid
                changed = True
            elif r["run_id"] != rid:
                raise SystemExit("")
        if changed:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                           encoding="utf-8")
            tmp.replace(p)


_UNSET = object()


def load_verdicts_singlesource(root: Path, *,
                               expected_run_ids: Iterable[str] | None,
                               expected_items: Iterable[tuple] | None | Any = _UNSET
                               ) -> dict[str, dict[tuple, str]]:
    if expected_items is _UNSET:
        if expected_run_ids is not None:
            raise TypeError(
                "")
        expected_items = None
    elif (expected_run_ids is None) != (expected_items is None):
        raise TypeError(
            "")
    src_by_run: dict[str, list[Path]] = {}
    vmap: dict[str, dict[tuple, str]] = {}
    dup_items: list[str] = []
    for p in iter_verdict_files(Path(root)):
        for i, r in enumerate(_rows_with_run_id(p), 1):
            rid = r["run_id"]
            v = r.get("verdict")
            if v not in VERDICT_VOCAB:
                raise SystemExit("")
            if "path_id" not in r or "idx" not in r:
                raise SystemExit("")
            srcs = src_by_run.setdefault(rid, [])
            if p not in srcs:
                srcs.append(p)
            ik = (r["path_id"], r["idx"])
            per = vmap.setdefault(rid, {})
            if ik in per:
                dup_items.append(f"{rid}:{ik}@{p.name}")
            per[ik] = v
    forked = {rid: ps for rid, ps in src_by_run.items() if len(ps) >= 2}
    if forked or dup_items:
        lines = [f"    {rid} ← {', '.join(str(p) for p in ps)}"
                 for rid, ps in sorted(forked.items())[:_LIST_CAP]]
        raise SystemExit(
            "")
    if expected_run_ids is not None:
        missing = sorted(set(expected_run_ids) - set(vmap))
        if missing:
            raise SystemExit(
                "")
    if expected_items is not None:
        actual_items = {(rid, ik[0], ik[1]) for rid, per in vmap.items() for ik in per}
        exp_items = {tuple(k) for k in expected_items}
        miss_i = sorted(exp_items - actual_items)
        extra_i = sorted(actual_items - exp_items)
        if miss_i or extra_i:
            lines = []
            if miss_i:
                lines.append("")
            if extra_i:
                lines.append("")
            raise SystemExit(
                "")
    return vmap


def cmd_submit(data: Path, scan_root: Path | None, budget: int,
               poll_seconds: int, force: bool) -> int:
    plan = plan_submit(data, scan_root)
    if force:
        assert_force_wastes_nothing(data)
    rc = B.stage_judge_chunked(data, budget, poll_seconds, force)
    if rc != 0:
        return rc
    attach_run_ids(data)
    load_verdicts_singlesource(data, expected_run_ids=plan["run_ids"],
                               expected_items=plan["item_keys"])
    return 0


def cmd_collect(data: Path, budget: int, poll_seconds: int) -> int:
    if not (Path(data) / CHUNK_MANIFEST_FILENAME).exists():
        raise SystemExit("")
    rc = B.stage_judge_chunked(data, budget, poll_seconds, force=False)
    if rc != 0:
        return rc
    attach_run_ids(data)
    items = load_b5_judge_items(data)
    ids = sorted({it["run_id"] for it in items})
    load_verdicts_singlesource(data, expected_run_ids=ids,
                               expected_items=items_universe(items))
    return 0


def recon_channel_report(data: Path, vmap: dict[str, dict[tuple, str]], *,
                         items: list[dict] | None = None) -> dict[str, Any]:
    rows = _read_jsonl_loud(data / RECON_FILENAME)
    need = ("run_id", "path_id", "idx", "verdict_main", "verdict_recon", "agree", "kind")
    n = agree = 0
    by_kind: dict[str, list[int]] = {}
    bad_agree: list[str] = []
    off_source: list[str] = []
    for i, r in enumerate(rows):
        miss = [k for k in need if k not in r]
        if miss:
            raise SystemExit("")
        for tok in (r["verdict_main"], r["verdict_recon"]):
            if tok not in VERDICT_VOCAB:
                raise SystemExit("")
        recomputed = (r["verdict_main"] == r["verdict_recon"])
        if bool(r["agree"]) != recomputed:
            bad_agree.append(f"{r['run_id']}|{r['path_id']}|{r['idx']}")
        src = vmap.get(r["run_id"], {}).get((r["path_id"], r["idx"]))
        if src is None or src != r["verdict_main"]:
            off_source.append("")
        n += 1
        agree += recomputed
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += 1
        by_kind[r["kind"]][1] += recomputed
    if bad_agree:
        raise SystemExit("")
    if off_source:
        raise SystemExit(
            "")
    if n == 0:
        raise SystemExit("")
    out: dict[str, Any] = {
        "n_recon": n, "n_agree": agree, "agree_rate": agree / n, "q_recon": 1 - agree / n,
        "by_kind": {k: {"n": v[0], "agree_rate": v[1] / v[0], "q": 1 - v[1] / v[0]}
                    for k, v in sorted(by_kind.items())},
        "q_recon_definition": "10%  = 1 − mean(agree) =  custom_id "
                              " |R  byte  prompt",
        "q_prompt": None, "q_prompt_definition": None,
    }
    if items is not None:
        by_prompt: dict[str, list[str]] = {}
        for it in items:
            v = vmap.get(it["run_id"], {}).get((it["path_id"], it["idx"]))
            if v is not None:
                by_prompt.setdefault(it["prompt"], []).append(v)
        npair = ndis = 0
        for vs in by_prompt.values():
            for a, b in itertools.combinations(vs, 2):
                npair += 1
                ndis += (a != b)
        out["q_prompt"] = (ndis / npair) if npair else None
        out["q_prompt_n_pairs"] = npair
        out["q_prompt_definition"] = (" prompt "
                                      "`b4_gate._judge_channel_section` "
                                      " B4/B4R ")
    return out


JUDGE_BINDING_KEYS = ("arm", "cat", "tid", "mode")
LEDGER_BINDING_KEYS = ("cat", "tid", "mode")
BRIDGE_FILENAME = "ga_layerA.jsonl"


def verify_run_id_binding(data: Path, layer_a_rows: list[dict],
                          scan_root: Path | None = None) -> dict[str, int]:
    bridge: dict[str, tuple] = {}
    dup_bridge: list[str] = []
    for r in layer_a_rows:
        rid = r.get("run_id")
        if not rid:
            raise SystemExit("")
        lack = [k for k in JUDGE_BINDING_KEYS if k not in r]
        if lack:
            raise SystemExit(
                "")
        key = tuple(r[k] for k in JUDGE_BINDING_KEYS)
        if rid in bridge and bridge[rid] != key:
            dup_bridge.append(rid)
        bridge[rid] = key
    if dup_bridge:
        raise SystemExit(
            "")

    seen: dict[tuple, str] = {}
    collide: list[str] = []
    for _rid, _key in bridge.items():
        if _key in seen:
            collide.append(f"{_key} → {seen[_key]} / {_rid}")
        seen[_key] = _rid
    if collide:
        raise SystemExit(
            "")
    root = Path(scan_root) if scan_root else Path(data)
    mismatched: list[str] = []
    orphan: list[str] = []
    n_rows = 0
    for p in iter_verdict_files(root):
        for r in _rows_with_run_id(p):
            n_rows += 1
            rid = r["run_id"]
            missing = [k for k in JUDGE_BINDING_KEYS if k not in r]
            if missing:
                raise SystemExit(
                    "")
            if rid not in bridge:
                orphan.append(rid)
                continue
            if tuple(r[k] for k in JUDGE_BINDING_KEYS) != bridge[rid]:
                mismatched.append(
                    f"{rid}verdict  {tuple(r[k] for k in JUDGE_BINDING_KEYS)} vs  {bridge[rid]}")
    if n_rows == 0:
        raise SystemExit(
            "")
    if orphan or mismatched:
        raise SystemExit(
            "")

    n_ledger = 0
    led_path = Path(data) / "analysis" / "ledger_b.jsonl"
    arm_to_bl: dict[str, set[tuple]] = {}
    if led_path.exists():
        bridge_sub = {rid: tuple(r.get(k) for k in LEDGER_BINDING_KEYS)
                      for rid, r in ((r["run_id"], r) for r in layer_a_rows)}
        bad_led: list[str] = []
        bad_rep: list[str] = []
        led_ids: set[str] = set()
        for line in led_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n_ledger += 1
            rid = r.get("run_id")
            led_ids.add(rid)
            if rid in bridge_sub and tuple(r.get(k) for k in LEDGER_BINDING_KEYS) != bridge_sub[rid]:
                bad_led.append(rid)
            arm = bridge.get(rid, (None,))[0]
            if arm is not None:
                arm_to_bl.setdefault(arm, set()).add((r.get("batch"), r.get("label")))
                rep = r.get("replicate_rep")
                expect_rep = arm.rsplit("/rep", 1)[1] if "/rep" in arm else None
                if expect_rep is not None:
                    if str(rep) != expect_rep:
                        bad_rep.append("")
                elif rep not in (0, None):
                    bad_rep.append("")
        only_bridge = sorted(set(bridge) - led_ids)
        only_led = sorted(led_ids - set(bridge))
        forked = {a: v for a, v in arm_to_bl.items() if len(v) > 1}
        if bad_led or bad_rep or forked or only_bridge or only_led:
            raise SystemExit(
                "")
    return {"n_verdict_rows": n_rows, "n_bridge": len(bridge), "n_ledger_rows": n_ledger,
            "n_arms": len(arm_to_bl), "ledger_checked": led_path.exists()}


def cmd_verify(data: Path, scan_root: Path | None) -> int:
    items = load_b5_judge_items(data)
    ids = sorted({it["run_id"] for it in items})
    vmap = load_verdicts_singlesource(Path(scan_root) if scan_root else Path(data),
                                      expected_run_ids=ids,
                                      expected_items=items_universe(items))
    import b4_gate as _B4
    la_path = Path(data) / BRIDGE_FILENAME
    layer_a_rows = [json.loads(line) for line in
                    la_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = {r["run_id"] for r in layer_a_rows
                if _B4.is_item_eligible(r["verdict"], r.get("reason_code"))}
    only_layer_a = sorted(eligible - set(ids))
    only_items = sorted(set(ids) - eligible)
    if only_layer_a or only_items:
        raise SystemExit(
            "")
    bind = verify_run_id_binding(data, layer_a_rows, scan_root)
    n_items = sum(len(m) for m in vmap.values())
    if bind["ledger_checked"]:
        bd30 = (f"[BD-30] verdict  {bind['n_verdict_rows']} / "
                f" {bind['n_bridge']} runs /  {bind['n_ledger_rows']}  / "
                f"arm→(batch,label) {bind['n_arms']}  arm")
    else:
        bd30 = (f"[BD-30] ** ① **verdict  {bind['n_verdict_rows']} / "
                f" {bind['n_bridge']} runs ****——"
                "②③ `analysis/ledger_b.jsonl`  B5 "
                "`verify` ")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["submit", "collect", "verify"])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--scan-root", type=Path, default=None)
    ap.add_argument("--enqueued-budget", type=int, default=B._DEFAULT_ENQUEUED_BUDGET)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    data = args.data if args.data.is_absolute() else (LAB / args.data)
    scan_root = args.scan_root
    if scan_root is not None and not scan_root.is_absolute():
        scan_root = LAB / scan_root
    if args.stage == "submit":
        return cmd_submit(data, scan_root, args.enqueued_budget,
                          args.poll_seconds, args.force)
    if args.stage == "collect":
        if args.force:
            raise SystemExit(
                "")
        return cmd_collect(data, args.enqueued_budget, args.poll_seconds)
    if args.stage == "verify":
        return cmd_verify(data, scan_root)
    raise SystemExit(f"unreachable stage {args.stage}")


if __name__ == "__main__":
    sys.exit(main())
