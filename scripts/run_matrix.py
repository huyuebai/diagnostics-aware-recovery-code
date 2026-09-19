#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import yaml

from matrix import (
    CATS, EXPECTED_TASKS_PER_CAT, EXPECTED_TOTAL, MANIFEST_SCHEMA, REPLICATE_PER_CAT, SEED_BASE,
)

MAX_ATTEMPTS = 2
MODES_ALL = ("P0", "P1", "P2", "P3", "P4")

VENDOR_ERROR_MARKER = "Error during execution"

VENDOR_EMPTY_RESPONSE_MARKER = "Model repeatedly returned empty responses"

VENDOR_ERROR_MARKERS = (VENDOR_ERROR_MARKER, VENDOR_EMPTY_RESPONSE_MARKER)

RC_EMIT, RC_GAP, RC_FAKE = 5, 6, 7

MAX_NAMED = 12


class MatrixRunError(SystemExit):
    pass


def validate_manifest(man: dict) -> list[dict]:
    if man.get("schema") != MANIFEST_SCHEMA:
        raise MatrixRunError(f"[run_matrix] REFUSING: schema {man.get('schema')!r} != {MANIFEST_SCHEMA}")
    rows = man.get("runs") or []
    if man.get("expected_total") != EXPECTED_TOTAL or len(rows) != EXPECTED_TOTAL:
        raise MatrixRunError(
            "")
    seen_ids: set[str] = set()
    seen_blocks: set[tuple[str, str]] = set()
    prev: tuple[str, str] | None = None
    for idx, r in enumerate(rows):
        if r["staging_index"] != idx:
            raise MatrixRunError(
                "")
        if r["run_id"] in seen_ids:
            raise MatrixRunError("")
        seen_ids.add(r["run_id"])
        blk = (r["cat"], r["tid"])
        if blk != prev:
            if blk in seen_blocks:
                raise MatrixRunError(
                    "")
            if prev is not None:
                seen_blocks.add(prev)
            prev = blk
    return rows


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    return man, validate_manifest(man)


def task_blocks(rows: list[dict]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for r in rows:
        blk = (r["cat"], r["tid"])
        if not blocks or blocks[-1] != blk:
            blocks.append(blk)
    return blocks


def parse_chunk(s: str) -> tuple[int, int]:
    try:
        i_s, n_s = s.split("/", 1)
        i, n = int(i_s), int(n_s)
    except ValueError:
        raise MatrixRunError("")
    if not (1 <= i <= n):
        raise MatrixRunError("")
    return i, n


def chunk_rows(rows: list[dict], i: int, n: int) -> list[dict]:
    blocks = task_blocks(rows)
    if n > len(blocks):
        raise MatrixRunError("")
    lo, hi = (i - 1) * len(blocks) // n, i * len(blocks) // n
    picked = set(blocks[lo:hi])
    return [r for r in rows if (r["cat"], r["tid"]) in picked]


def _router_segment(spec: str | None) -> str:
    if spec is None or spec == "template":
        return ""
    body = spec.split(":", 1)[1] if ":" in spec else spec
    parts = dict(kv.split("=", 1) for kv in body.split(","))
    return f"__C{parts['content']}-A{parts['action']}"


def subtree_of(row: dict) -> str:
    if row["batch"] == "rep":
        return f"rep/rep{row['replicate_rep']}"
    fp = "fa" if row["force_prompt"] == "fault_aware" else "p0"
    return f"{fp}/{row['dispatch'].replace(':', '-')}{_router_segment(row.get('router'))}"


def inference_relpath(row: dict, model: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return (Path(subtree_of(row)) / safe / "fc" / row["cat"] / "inferences"
            / f"{row['tid']}_{row['mode']}_inference.json")


def eval_relpath(row: dict, model: str) -> Path:
    p = inference_relpath(row, model)
    return p.parent.parent / "evaluations" / f"{row['tid']}_{row['mode']}_eval.json"


def _json_or_none(p: Path) -> tuple[dict | None, str]:
    if not p.is_file():
        return None, "missing"
    try:
        return json.loads(p.read_text(encoding="utf-8")), "ok"
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "corrupt"


def inference_health(doc: dict) -> str | None:
    ms = doc.get("messages")
    if not isinstance(ms, list) or len(ms) < 2:
        return f"messages  list  <2 {len(ms) if isinstance(ms, list) else type(ms).__name__}"
    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in ms):
        return " assistant message"
    for i, m in enumerate(ms):
        if not isinstance(m, dict):
            return f"message[{i}]  dict"
        content = str(m.get("content") or "")
        for mark in VENDOR_ERROR_MARKERS:
            if mark in content:
                return f"message[{i}]  vendor {mark!r}=  final_answer"
    if not str(ms[-1].get("content") or "").strip():
        return " message content "
    tk = (doc.get("tokens") or {}).get("total_tokens")
    if isinstance(tk, bool) or not isinstance(tk, int) or tk <= 0:
        return f"tokens.total_tokens={tk!r} server  0"
    return None


def classify(rows: list[dict], out_root: Path, model: str, *,
             require_eval: bool = False) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        p = Path(out_root) / inference_relpath(r, model)
        pe = Path(out_root) / eval_relpath(r, model)
        _, inf_state = _json_or_none(p)
        status = {"missing": "pending", "corrupt": "corrupt", "ok": "done"}[inf_state]
        ev_state = None
        if require_eval and status == "done":
            _, ev_state = _json_or_none(pe)
            if ev_state != "ok":
                status = "eval_missing"
        out.append({"row": r, "status": status, "path": p,
                    "eval_path": pe, "eval_state": ev_state})
    return out


def status_counts(classified: list[dict]) -> dict[str, int]:
    c = {"done": 0, "corrupt": 0, "eval_missing": 0, "pending": 0}
    for e in classified:
        c[e["status"]] += 1
    return c


def assert_fresh_or_resume(classified: list[dict], resume: bool) -> None:
    c = status_counts(classified)
    existing = c["done"] + c["corrupt"] + c["eval_missing"]
    if not resume and existing:
        raise MatrixRunError(
            "")


def _quarantine_one(p: Path, kind: str) -> None:
    k = 0
    while (q := p.with_name(f"{p.name}.corrupt-{k}")).exists():
        k += 1
    p.rename(q)


def quarantine_corrupt(classified: list[dict]) -> int:
    n = 0
    for e in classified:
        if e["status"] == "corrupt":
            _quarantine_one(e["path"], "inference")
            e["status"] = "pending"
            n += 1
        elif e["status"] == "eval_missing" and e.get("eval_state") == "corrupt":
            _quarantine_one(e["eval_path"], "evaluation")
            e["eval_state"] = "missing"
            n += 1
    return n


def plan_units(classified: list[dict]) -> list[dict]:
    units: list[dict] = []
    for e in classified:
        if e["status"] == "done":
            continue
        r = e["row"]
        key = (r["cat"], r["tid"], r["dispatch"], r["force_prompt"], r.get("router"), subtree_of(r))
        if units and units[-1]["key"] == key:
            units[-1]["rows"].append(r)
            units[-1]["modes"].append(r["mode"])
        else:
            units.append({"key": key, "cat": r["cat"], "tid": r["tid"], "dispatch": r["dispatch"],
                          "force_prompt": r["force_prompt"], "router": r.get("router"),
                          "subtree": subtree_of(r),
                          "rows": [r], "modes": [r["mode"]]})
    return units


def _raw_model_name(config_path: Path) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    agent = cfg.get("agent") or {}
    model = agent.get("model")
    if not model:
        raise MatrixRunError("")
    if agent.get("result_tag"):
        model = f"{model}_{agent['result_tag']}"
    return model


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stage_task(dataset: Path, stage_root: Path, cat: str, tid: str) -> Path:
    import shutil
    src_dir = dataset / "perturbed_tasks" / cat
    dst = stage_root / cat / tid
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for mode in MODES_ALL:
        src = src_dir / f"{tid}_{mode}.json"
        if not src.is_file():
            raise SystemExit("")
        shutil.copy2(src, dst / src.name)
    return dst


def _run_unit(run_eval, config_path: str, unit: dict, stage_root: Path, out_root: Path,
              agent_cls, op_config: dict) -> int:
    from dar.dispatch import DispatchConfigError, configure_agent_from_operation_point, wire_dispatch_engine
    from dar.router import RouterWiringError, router_from_spec
    os.environ["DAR_OUTPUT_DIR"] = str(out_root / unit["subtree"])
    try:
        wire_dispatch_engine(run_eval, unit["dispatch"], router=router_from_spec(unit.get("router")))
    except (DispatchConfigError, RouterWiringError) as e:
        raise SystemExit("")
    configure_agent_from_operation_point(agent_cls, op_config, force_prompt=unit["force_prompt"])
    task_dir = stage_root / unit["cat"] / unit["tid"]
    sys.argv = ["run_eval.py", "--config", config_path, "--task-dir", str(task_dir),
                "--task-category", unit["cat"], "--modes", *unit["modes"]]
    try:
        run_eval.main()
    except SystemExit as e:
        return 0 if e.code in (0, None) else int(e.code)
    except Exception as e:
        print(f"[run_matrix] unit {unit['subtree']}/{unit['cat']}/{unit['tid']} FAILED: {e!r}")
        return 1
    return 0


def _build_subtree_snapshot(agent_cls, op_config: dict, subtree: str, force_prompt: str, k: int):
    from run_dispatch import PROVENANCE, _prompt_sha256, _read_aggregate

    from dar import config_snapshot as cs
    from dar.ga_semantic import judge_prompt_fingerprint, substitution_prompt_fingerprint
    from dar.router import template_fingerprint
    return cs.build_batch_snapshot(
        arm=subtree, force_prompt=force_prompt,
        system_prompt_sha256=_prompt_sha256(agent_cls, force_prompt),
        op_config=op_config,
        judge_prompt_fingerprint=judge_prompt_fingerprint(),
        substitution_prompt_fingerprint=substitution_prompt_fingerprint(),
        template_fingerprint=template_fingerprint(),
        sample_seed=SEED_BASE, k=k,
        vendor_aggregate=_read_aggregate(PROVENANCE / "toolmaze_manifest.json"),
        dataset_aggregate=_read_aggregate(PROVENANCE / "toolmaze_dataset_manifest.json"))


def _emit_subtrees(rows: list[dict], out_root: Path, agent_cls, op_config: dict) -> list[str]:
    from dar import config_snapshot as cs
    seen: dict[str, dict] = {}
    for r in rows:
        st = subtree_of(r)
        if st not in seen:
            k = REPLICATE_PER_CAT if r["batch"] == "rep" else EXPECTED_TASKS_PER_CAT
            seen[st] = {"fp": r["force_prompt"], "k": k}
    cats_scope = {r["cat"] for r in rows}
    problems: list[str] = []
    for st in sorted(seen):
        meta = seen[st]
        snap = _build_subtree_snapshot(agent_cls, op_config, st, meta["fp"], meta["k"])
        root = out_root / st
        for inf_dir in sorted(root.rglob("inferences")):
            if not inf_dir.is_dir() or inf_dir.parent.name not in CATS:
                continue
            if inf_dir.parent.name not in cats_scope:
                continue
            try:
                paths = cs.emit_cell(inf_dir, inf_dir.parent, snap)
                bij = cs.audit_bijection(inf_dir, paths[1])
                if bij:
                    problems.append(f"{inf_dir.parent}: {bij}")
            except Exception as e:
                problems.append(f"{inf_dir.parent}: emit FAILED {e!r}")
    return problems


def _write_staging_manifest(man: dict, rows_chunk: list[dict], out_root: Path,
                            chunk: tuple[int, int]) -> Path:
    i, n = chunk
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / f"staging_manifest.chunk{i}of{n}.json"
    payload = {**{k: v for k, v in man.items() if k != "runs"},
               "chunk": {"index": i, "n": n,
                         "chunk_run_ids": [r["run_id"] for r in rows_chunk]},
               "runs": man["runs"]}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def _named(items: list[str]) -> str:
    more = f" … {len(items)}" if len(items) > MAX_NAMED else ""
    return "; ".join(items[:MAX_NAMED]) + more


def selfproof(rows: list[dict], out_root: Path, model: str) -> tuple[int, list[str], list[str]]:
    unhealthy: list[str] = []
    missing: list[str] = []
    n = 0
    for r in rows:
        p = Path(out_root) / inference_relpath(r, model)
        doc, state = _json_or_none(p)
        if state == "missing":
            missing.append(r["run_id"])
            continue
        n += 1
        why = "json " if state == "corrupt" else inference_health(doc)
        if why:
            unhealthy.append(f"{r['run_id']}: {why}")
    return n, unhealthy, missing


def _report_selfproof(n: int, unhealthy: list[str], missing: list[str]) -> int:
    if unhealthy:
        pass
    if missing:
        pass
    if unhealthy:
        return RC_FAKE
    return RC_GAP if missing else 0


def chunk_exit_code(rc_selfproof: int, *, gap: bool, emit_problems: bool) -> int:
    return max(rc_selfproof, RC_GAP if gap else 0, RC_EMIT if emit_problems else 0)


def _selfproof_only(args, rows_chunk: list[dict]) -> int:
    model = _raw_model_name(Path(args.config))
    return _report_selfproof(*selfproof(rows_chunk, Path(args.out_root), model))


def _exec_chunk(args, man: dict, rows_chunk: list[dict], chunk: tuple[int, int]) -> int:
    from run_dispatch import _assert_op_consistency, _bridge_dataset_source, _load_run_eval

    from dar.agents import CallLog, get_dar_vllm_agent_class
    from dar.vendorpath import add_vendor_to_syspath

    out_root = Path(args.out_root)
    stage_root = Path(args.stage_dir).resolve()
    dataset = _bridge_dataset_source(args.dataset_root)
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
    _assert_op_consistency(config, op_config)
    if (op_config.get("agent") or {}).get("temperature") is None:
        raise SystemExit("")
    CallLog.configure(os.environ.get("DAR_CALL_LOG"))
    model = _raw_model_name(Path(args.config))

    classified = classify(rows_chunk, out_root, model, require_eval=True)
    assert_fresh_or_resume(classified, args.resume)
    n_quarantined = quarantine_corrupt(classified)
    units = plan_units(classified)
    c0 = status_counts(classified)
    _write_staging_manifest(man, rows_chunk, out_root, chunk)

    staged: set[tuple[str, str]] = set()
    for u in units:
        blk = (u["cat"], u["tid"])
        if blk not in staged:
            _stage_task(dataset, stage_root, *blk)
            staged.add(blk)

    os.chdir(vendor_root)
    fail_log = out_root / f"failures.chunk{chunk[0]}of{chunk[1]}.jsonl"
    n_failed = 0
    for u in units:
        rc = 1
        attempts = 0
        while attempts < MAX_ATTEMPTS and rc != 0:
            attempts += 1
            rc = _run_unit(run_eval, args.config, u, stage_root, out_root, DarVLLMAgent, op_config)
        if rc != 0:
            n_failed += 1
            with fail_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"unit": f"{u['subtree']}/{u['cat']}/{u['tid']}",
                                    "run_ids": [r["run_id"] for r in u["rows"]],
                                    "attempts": attempts, "rc": rc, "error": "unit rc != 0",
                                    "utc": _utc()}, ensure_ascii=False) + "\n")

    final = status_counts(classify(rows_chunk, out_root, model, require_eval=True))
    rc_proof = _report_selfproof(*selfproof(rows_chunk, out_root, model))
    problems = _emit_subtrees(rows_chunk, out_root, DarVLLMAgent, op_config)
    gap = bool(n_failed or final["pending"] or final["corrupt"] or final["eval_missing"])
    for p in problems:
        pass
    if gap:
        pass
    rc_out = chunk_exit_code(rc_proof, gap=gap, emit_problems=bool(problems))
    print(f"[run_matrix] chunk {chunk[0]}/{chunk[1]} done rc={rc_out} "
          f"done={final['done']}/{len(rows_chunk)}")
    return rc_out


def assert_emit_target_is_hpc(out_root: Path, i_am_on_hpc: bool) -> None:
    if i_am_on_hpc or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        return
    target = Path(out_root).resolve()
    data_root = (LAB / "data").resolve()
    if target == data_root or data_root in target.parents:
        raise MatrixRunError(
            "")


def _emit_only(args, rows: list[dict]) -> int:
    from dar.agents import get_dar_vllm_agent_class
    from dar.vendorpath import add_vendor_to_syspath
    assert_emit_target_is_hpc(Path(args.out_root), args.i_am_on_hpc)
    add_vendor_to_syspath()
    op_config = yaml.safe_load(Path(args.op_config).read_text())
    problems = _emit_subtrees(rows, Path(args.out_root), get_dar_vllm_agent_class(), op_config)
    for p in problems:
        pass
    print(f"[run_matrix] emit-only done rc={5 if problems else 0}")
    return 5 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--op-config", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--out-root", required=True, type=lambda p: str(Path(p).resolve()))
    ap.add_argument("--chunk", default="1/1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--i-am-on-hpc", action="store_true")
    ap.add_argument("--selfproof-only", action="store_true")
    args = ap.parse_args()

    chunk = parse_chunk(args.chunk)
    man, rows = load_manifest(Path(args.manifest))
    if man.get("seed_base") != SEED_BASE:
        raise MatrixRunError(
            "")
    rows_chunk = chunk_rows(rows, *chunk)

    if args.emit_only:
        return _emit_only(args, rows)
    if args.selfproof_only:
        return _selfproof_only(args, rows_chunk)
    if args.dry_run:
        model = _raw_model_name(Path(args.config))
        classified = classify(rows_chunk, Path(args.out_root), model, require_eval=True)
        units = plan_units(classified)
        c = status_counts(classified)
        print(f"[run_matrix:dry] chunk {chunk[0]}/{chunk[1]} rows={len(rows_chunk)} "
              f"done={c['done']} corrupt={c['corrupt']} eval_missing={c['eval_missing']} "
              f"pending={c['pending']} units={len(units)}")
        per_cell: dict[str, int] = {}
        for e in classified:
            if e["status"] != "done":
                per_cell[e["row"]["cell_id"]] = per_cell.get(e["row"]["cell_id"], 0) + 1
        for cell in sorted(per_cell):
            pass
        return 0
    return _exec_chunk(args, man, rows_chunk, chunk)


if __name__ == "__main__":
    sys.exit(main())
