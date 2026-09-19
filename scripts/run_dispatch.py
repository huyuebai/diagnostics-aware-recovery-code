#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import yaml

from dar import config_snapshot as cs
from dar.agents import CallLog, get_dar_vllm_agent_class
from dar.dispatch import (
    DispatchConfigError,
    configure_agent_from_operation_point,
    wire_dispatch_engine,
)
from dar.ga_semantic import (
    judge_prompt_fingerprint,
    substitution_prompt_fingerprint,
)
from dar.router import template_fingerprint
from dar.vendorpath import add_vendor_to_syspath
from run_pilot import CATEGORIES, MODES, sample_base_ids, stage_category

PROVENANCE = LAB / "provenance"


def _validate_tier(tier: str) -> str:
    if tier == "none" or tier == "oracle" or (
        tier.startswith("fixed:") and tier.split(":", 1)[1] in ("transient", "persistent", "np")
    ):
        return tier
    raise SystemExit("")


def _bridge_dataset_source(dataset_root: str) -> Path:
    p = Path(dataset_root).resolve()
    os.environ["DAR_TOOLMAZE_DATASET"] = str(p)
    return p


def _load_run_eval(vendor_root: Path):
    spec = importlib.util.spec_from_file_location(
        "tm_run_eval", vendor_root / "evaluation" / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_op_consistency(config: dict, op_config: dict) -> None:
    va, oa = config.get("agent") or {}, op_config.get("agent") or {}
    for k in ("max_tokens", "temperature"):
        if va.get(k) != oa.get(k):
            raise SystemExit(
                "")


def _prompt_sha256(agent_cls, force_prompt: str) -> str:
    if force_prompt == "fault_aware":
        s = agent_cls._build_perturbation_system_message(None)
    elif force_prompt == "p0":
        s = agent_cls._build_p0_system_message(None)
    else:
        raise SystemExit("")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_aggregate(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["aggregate_sha256"]


def _build_run_snapshot(agent_cls, config: dict, op_config: dict, tier: str, args) -> dict:
    force_prompt = agent_cls.dar_force_prompt or "fault_aware"
    return cs.build_batch_snapshot(
        arm=tier, force_prompt=force_prompt,
        system_prompt_sha256=_prompt_sha256(agent_cls, force_prompt),
        op_config=op_config,
        judge_prompt_fingerprint=judge_prompt_fingerprint(),
        substitution_prompt_fingerprint=substitution_prompt_fingerprint(),
        template_fingerprint=template_fingerprint(), sample_seed=args.seed, k=args.k,
        vendor_aggregate=_read_aggregate(PROVENANCE / "toolmaze_manifest.json"),
        dataset_aggregate=_read_aggregate(PROVENANCE / "toolmaze_dataset_manifest.json"))


def _emit_cells(config: dict, snapshot: dict, cats: list[str]) -> tuple[int, int, list]:
    output_root = Path(config["data"]["output_dir"])
    catset = set(cats)
    n_cells, n_rows, problems = 0, 0, []
    for inf_dir in sorted(output_root.rglob("inferences")):
        if not inf_dir.is_dir() or inf_dir.parent.name not in catset:
            continue
        cell_dir = inf_dir.parent
        paths = cs.emit_cell(inf_dir, cell_dir, snapshot)
        rows = cs.read_manifest(paths[1])
        bij = cs.audit_bijection(inf_dir, paths[1])
        if bij:
            problems.append((str(cell_dir), bij))
        n_cells += 1
        n_rows += len(rows)
    return n_cells, n_rows, problems


def _write_staging_manifest(config: dict, args, staging_ids: dict, tier: str) -> Path:
    output_root = Path(config["data"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    p = output_root / "staging_manifest.json"
    p.write_text(json.dumps({"tier": tier, "seed": args.seed, "k": args.k,
                             "modes": config["evaluation"].get("modes"), "ids": staging_ids},
                            ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--op-config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--dispatch", required=True,
                    help="none | oracle | fixed:<transient|persistent|np>")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--force-prompt", choices=("fault_aware", "p0"), default="fault_aware")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tier = _validate_tier(args.dispatch)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not cats or (set(cats) - set(CATEGORIES)):
        raise SystemExit("")
    if args.k <= 0:
        raise SystemExit("")

    dataset = _bridge_dataset_source(args.dataset_root)
    stage = Path(args.stage_dir)
    staging_ids: dict[str, list[str]] = {}
    for cat in cats:
        ids = sample_base_ids(dataset / "perturbed_tasks" / cat, args.k, args.seed)
        n = stage_category(dataset / "perturbed_tasks" / cat, stage / cat, ids)
        staging_ids[cat] = ids
        print(f"[dispatch] {cat}: staged {n} files ({len(ids)} tasks × {len(MODES)} modes)")

    vendor_root = add_vendor_to_syspath()
    run_eval = _load_run_eval(vendor_root)

    import evaluation.agents as agents_pkg
    DarVLLMAgent = get_dar_vllm_agent_class()
    agents_pkg.VLLMAgent = DarVLLMAgent
    config = run_eval.load_config(args.config)
    base_url = str((config.get("agent") or {}).get("base_url", ""))
    if "${" in base_url:
        raise SystemExit("")
    dar_cfg = config.get("dar") or {}
    DarVLLMAgent.configure(timeout_s=dar_cfg.get("timeout_s"),
                           enable_thinking=dar_cfg.get("enable_thinking", ...))
    op_config = yaml.safe_load(Path(args.op_config).read_text())
    configure_agent_from_operation_point(DarVLLMAgent, op_config,
                                         force_prompt=args.force_prompt)
    CallLog.configure(os.environ.get("DAR_CALL_LOG"))

    _assert_op_consistency(config, op_config)
    snapshot = _build_run_snapshot(DarVLLMAgent, config, op_config, tier, args)
    print(f"[dispatch] config_snapshot fp={snapshot['snapshot_fingerprint']} "
          f"arm={snapshot['arm']} force_prompt={snapshot['force_prompt']} "
          f"prompt_sha={snapshot['system_prompt_sha256'][:16]}")
    if not args.dry_run and (op_config.get("agent") or {}).get("temperature") is None:
        raise SystemExit("")

    try:
        wire_dispatch_engine(run_eval, tier)
    except DispatchConfigError as e:
        raise SystemExit("")

    if args.dry_run:
        return _dry_run_check(run_eval, config, DarVLLMAgent, op_config, tier, stage)

    os.chdir(vendor_root)
    rc = 0
    for cat in cats:
        print(f"[dispatch] ===== category {cat} (tier={tier}) =====", flush=True)
        sys.argv = ["run_eval.py", "--config", args.config, "--task-dir", str(stage / cat)]
        try:
            run_eval.main()
        except SystemExit as e:
            if e.code not in (0, None):
                rc = int(e.code)
                print(f"[dispatch] {cat} exited {e.code}")
        except Exception as e:
            rc = 1
            print(f"[dispatch] {cat} FAILED: {e!r}")

    n_cells, n_rows, problems = _emit_cells(config, snapshot, cats)
    man = _write_staging_manifest(config, args, staging_ids, tier)
    expected = len(cats) * args.k * len(config["evaluation"].get("modes") or [])
    print(f"[dispatch] emitted into {n_cells} cells; run_manifest rows={n_rows}/{expected}; "
          f"staging={man}")
    if problems:
        rc = rc or 5
        for cell, bij in problems:
            pass
    if n_rows != expected:
        rc = rc or 6
    print(f"[dispatch] done rc={rc}")
    return rc


def _dry_run_check(run_eval, config, DarVLLMAgent, op_config, tier, stage) -> int:
    from dar.dispatch import DispatchEngine
    agent = run_eval.create_agent(config)
    checks = {
        "agent_is_dar": type(agent).__name__ == "DarVLLMAgent",
        "engine_is_dispatch": run_eval.ExecutionEngine is DispatchEngine,
        "tier_set": DispatchEngine._tier == tier,
        "force_prompt_unified": DarVLLMAgent.dar_force_prompt is not None,
        "strip_matches_op": bool(DarVLLMAgent.dar_strip_history_thinking)
        == bool((op_config.get("agent") or {}).get("strip_history_thinking")),
        "writeback_matches_op": bool(DarVLLMAgent.dar_writeback_reasoning)
        == bool((op_config.get("writeback") or {}).get("reasoning_content_to_trace")),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        return 1
    print(f"[dispatch:dry] OK  tier={tier} agent=DarVLLMAgent engine=DispatchEngine "
          f"force_prompt={DarVLLMAgent.dar_force_prompt} "
          f"strip={DarVLLMAgent.dar_strip_history_thinking} "
          f"writeback={DarVLLMAgent.dar_writeback_reasoning} "
          f"modes={config['evaluation'].get('modes')} stage={stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
