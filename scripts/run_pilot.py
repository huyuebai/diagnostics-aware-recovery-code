#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from dar.agents import get_dar_vllm_agent_class
from dar.vendorpath import add_vendor_to_syspath

MODES = ["P0", "P1", "P2", "P3", "P4"]
CATEGORIES = ["c1", "c2", "c3", "c4"]


def sample_base_ids(task_dir: Path, k: int, seed: int) -> list[str]:
    ids = sorted({p.name.rsplit("_P", 1)[0] for p in task_dir.glob("*_P0.json")})
    if len(ids) < k:
        raise SystemExit("")
    return random.Random(seed).sample(ids, k)


def stage_category(task_dir: Path, stage_cat: Path, ids: list[str]) -> int:
    if stage_cat.is_dir():
        shutil.rmtree(stage_cat)
    stage_cat.mkdir(parents=True, exist_ok=True)
    n = 0
    for tid in ids:
        for mode in MODES:
            src = task_dir / f"{tid}_{mode}.json"
            if not src.is_file():
                raise SystemExit("")
            shutil.copy2(src, stage_cat / src.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--dataset-root", required=True, help=".../ToolMaze_dataset")
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    bad = set(cats) - set(CATEGORIES)
    if bad:
        raise SystemExit("")
    if not cats:
        raise SystemExit("")
    if args.k <= 0:
        raise SystemExit("")

    dataset = Path(args.dataset_root)
    stage = Path(args.stage_dir)

    manifest: dict = {"seed": args.seed, "k": args.k, "modes": MODES, "ids": {}}
    for cat in cats:
        task_dir = dataset / "perturbed_tasks" / cat
        ids = sample_base_ids(task_dir, args.k, args.seed)
        n = stage_category(task_dir, stage / cat, ids)
        manifest["ids"][cat] = ids
        print(f"[pilot] {cat}: staged {n} files ({len(ids)} tasks × {len(MODES)} modes)")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    vendor_root = add_vendor_to_syspath()
    spec = importlib.util.spec_from_file_location(
        "tm_run_eval", vendor_root / "evaluation" / "scripts" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    import evaluation.agents as agents_pkg
    DarVLLMAgent = get_dar_vllm_agent_class()
    agents_pkg.VLLMAgent = DarVLLMAgent
    config = run_eval.load_config(args.config)
    base_url = str((config.get("agent") or {}).get("base_url", ""))
    if "${" in base_url:
        raise SystemExit("")
    dar_cfg = config.get("dar") or {}
    DarVLLMAgent.configure(timeout_s=dar_cfg.get("timeout_s"),
                           enable_thinking=dar_cfg.get("enable_thinking", ...),
                           strip_history_thinking=dar_cfg.get("strip_history_thinking"))
    strip_on = bool(dar_cfg.get("strip_history_thinking"))
    from dar.agents import CallLog
    CallLog.configure(os.environ.get("DAR_CALL_LOG"))

    if args.dry_run:
        agent = run_eval.create_agent(config)
        ok = type(agent).__name__ == "DarVLLMAgent"
        wired = getattr(agent.client.chat.completions, "_strip", None)
        flagged = getattr(agent.client, "dar_strip_history_thinking", None)
        if wired != strip_on or flagged != strip_on:
            return 1
        print(f"[pilot:dry] agent={'OK' if ok else type(agent).__name__} "
              f"timeout={getattr(agent.client, 'dar_timeout_s', '?')}s "
              f"strip_history_thinking={wired} "
              f"max_tokens={config['agent'].get('max_tokens')} "
              f"modes={config['evaluation'].get('modes')} "
              f"stage={stage} manifest={stage / 'pilot_manifest.json'}")
        return 0 if ok else 1

    os.chdir(vendor_root)
    rc = 0
    for cat in cats:
        print(f"[pilot] ===== category {cat} =====", flush=True)
        sys.argv = ["run_eval.py", "--config", args.config,
                    "--task-dir", str(stage / cat)]
        try:
            run_eval.main()
        except SystemExit as e:
            if e.code not in (0, None):
                rc = int(e.code)
                print(f"[pilot] {cat} exited {e.code}")
        except Exception as e:
            rc = 1
            print(f"[pilot] {cat} FAILED: {e!r}")

    if strip_on:
        from dar.agents import StripStats
        if StripStats.chars_removed < 1000:
            rc = rc or 3

    from dar.agents import CallLog
    if CallLog.enabled():
        if CallLog.n_written == 0:
            rc = rc or 4
    print(f"[pilot] done rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
