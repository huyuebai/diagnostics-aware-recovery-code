from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "dar-config-snapshot/1"

_REQUIRED = (
    "arm", "force_prompt", "system_prompt_sha256",
    "judge_prompt_fingerprint", "substitution_prompt_fingerprint",
    "template_fingerprint", "sample_seed", "k",
    "vendor_aggregate", "dataset_aggregate",
)

_OP_REQUIRED = (
    "reasoning_parser", "tool_call_parser", "vllm_version", "image_sha256",
    "thinking", "tool_choice", "max_tokens",
)


class ConfigSnapshotError(RuntimeError):
    pass


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    body = {k: v for k, v in snapshot.items() if k != "snapshot_fingerprint"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()[:16]


def assert_snapshot_body_intact(snapshot: dict[str, Any]) -> None:
    stored = snapshot.get("snapshot_fingerprint")
    if stored is None:
        raise ConfigSnapshotError("")
    actual = snapshot_fingerprint(snapshot)
    if actual != stored:
        raise ConfigSnapshotError(
            "")


def _operation_point(op_config: dict[str, Any]) -> dict[str, Any]:
    serve = op_config.get("serve") or {}
    agent = op_config.get("agent") or {}
    wb = op_config.get("writeback") or {}
    op = {
        "reasoning_parser": serve.get("reasoning_parser"),
        "tool_call_parser": serve.get("tool_call_parser"),
        "vllm_version": serve.get("vllm_version"),
        "image_sha256": serve.get("image_sha256"),
        "thinking": agent.get("thinking"),
        "tool_choice": agent.get("tool_choice"),
        "max_tokens": agent.get("max_tokens"),
        "temperature": agent.get("temperature"),
        "seed": agent.get("seed"),
        "sampling": agent.get("sampling"),
        "seed_mode": agent.get("seed_mode"),
        "strip_history_thinking": agent.get("strip_history_thinking"),
        "writeback_reasoning_to_trace": wb.get("reasoning_content_to_trace"),
        "writeback_reasoning_to_prompt": wb.get("reasoning_content_to_prompt"),
    }
    missing = [k for k in _OP_REQUIRED if op.get(k) in (None, "")]
    if missing:
        raise ConfigSnapshotError(
            "")
    return op


def build_batch_snapshot(
    *,
    arm: str,
    force_prompt: str,
    system_prompt_sha256: str,
    op_config: dict[str, Any],
    judge_prompt_fingerprint: str,
    substitution_prompt_fingerprint: str,
    template_fingerprint: str,
    sample_seed: int,
    k: int,
    vendor_aggregate: str,
    dataset_aggregate: str,
) -> dict[str, Any]:
    params = {
        "arm": arm, "force_prompt": force_prompt, "system_prompt_sha256": system_prompt_sha256,
        "judge_prompt_fingerprint": judge_prompt_fingerprint,
        "substitution_prompt_fingerprint": substitution_prompt_fingerprint,
        "template_fingerprint": template_fingerprint, "sample_seed": sample_seed, "k": k,
        "vendor_aggregate": vendor_aggregate, "dataset_aggregate": dataset_aggregate,
    }
    missing = [f for f in _REQUIRED if params[f] in (None, "")]
    if missing:
        raise ConfigSnapshotError("")
    snap: dict[str, Any] = {
        "schema": SCHEMA,
        "arm": arm,
        "force_prompt": force_prompt,
        "system_prompt_sha256": system_prompt_sha256,
        "operation_point": _operation_point(op_config),
        "ga": {
            "judge_prompt_fingerprint": judge_prompt_fingerprint,
            "substitution_prompt_fingerprint": substitution_prompt_fingerprint,
        },
        "template_fingerprint": template_fingerprint,
        "sample_seed": sample_seed,
        "k": k,
        "vendor_aggregate": vendor_aggregate,
        "dataset_aggregate": dataset_aggregate,
    }
    snap["snapshot_fingerprint"] = snapshot_fingerprint(snap)
    return snap


def _iter_inferences(inferences_dir: Path) -> list[Path]:
    return sorted(inferences_dir.glob("*_inference.json"))


def _parse_inf_name(p: Path) -> tuple[str, str]:
    stem = p.name[: -len("_inference.json")]
    tid, mode = stem.rsplit("_", 1)
    return tid, mode


def emit_cell(inferences_dir: Path, cell_dir: Path, snapshot: dict[str, Any]) -> list[Path]:
    infs = _iter_inferences(inferences_dir)
    if not infs:
        raise ConfigSnapshotError(
            "")
    if "snapshot_fingerprint" not in snapshot:
        raise ConfigSnapshotError("")

    cell_dir.mkdir(parents=True, exist_ok=True)
    snap_path = cell_dir / "config_snapshot.json"
    if snap_path.is_file():
        prev = json.loads(snap_path.read_text(encoding="utf-8"))
        if prev.get("snapshot_fingerprint") != snapshot["snapshot_fingerprint"]:
            raise ConfigSnapshotError(
                "")
    snap_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
        encoding="utf-8")

    lines = []
    for p in infs:
        tid, mode = _parse_inf_name(p)
        lines.append(json.dumps({
            "task_id": tid,
            "mode": mode,
            "inference_file": p.name,
            "inference_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        }, ensure_ascii=False))
    manifest_path = cell_dir / "run_manifest.jsonl"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [snap_path, manifest_path]


def read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in
            manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def audit_bijection(inferences_dir: Path, manifest_path: Path) -> list[str]:
    problems: list[str] = []
    if not manifest_path.is_file():
        return [f"missing_manifest:{manifest_path.name}"]
    rows = read_manifest(manifest_path)
    by_file = {r["inference_file"]: r for r in rows}
    if len(by_file) != len(rows):
        problems.append(f"duplicate_rows:{len(rows) - len(by_file)}")
    actual = {p.name for p in _iter_inferences(inferences_dir)}
    for name in sorted(actual - set(by_file)):
        problems.append(f"missing_row:{name}")
    for name in sorted(set(by_file) - actual):
        problems.append(f"orphan_row:{name}")
    for name in sorted(actual & set(by_file)):
        digest = hashlib.sha256((inferences_dir / name).read_bytes()).hexdigest()
        if digest != by_file[name].get("inference_sha256"):
            problems.append(f"sha_mismatch:{name}")
    return problems


def verify_snapshot_frozen(snapshot: dict[str, Any], freeze_dir: Path) -> list[str]:
    prompt = json.loads((freeze_dir / "prompt_freeze.json").read_text(encoding="utf-8"))
    gaf = json.loads((freeze_dir / "ga_semantic_freeze.json").read_text(encoding="utf-8"))
    key = ("fault_aware_sha256" if snapshot.get("force_prompt") == "fault_aware"
           else "p0_plain_sha256")
    problems: list[str] = []
    if snapshot_fingerprint(snapshot) != snapshot.get("snapshot_fingerprint"):
        problems.append("")
    if snapshot.get("system_prompt_sha256") != prompt[key]:
        problems.append(f"prompt_sha != frozen[{key}]")
    ga = snapshot.get("ga") or {}
    if ga.get("judge_prompt_fingerprint") != gaf["judge_prompt_fingerprint"]:
        problems.append("judge_prompt_fingerprint drift")
    if ga.get("substitution_prompt_fingerprint") != gaf["substitution_prompt_fingerprint"]:
        problems.append("substitution_prompt_fingerprint drift")
    return problems


def resolve_run_config(row: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    assert_snapshot_body_intact(snapshot)
    rs = row.get("snapshot_fingerprint")
    if rs != snapshot["snapshot_fingerprint"]:
        raise ConfigSnapshotError(
            "")
    return {
        "task_id": row["task_id"],
        "mode": row["mode"],
        "inference_file": row["inference_file"],
        "inference_sha256": row["inference_sha256"],
        "arm": snapshot["arm"],
        "force_prompt": snapshot["force_prompt"],
        "system_prompt_sha256": snapshot["system_prompt_sha256"],
        "operation_point": snapshot["operation_point"],
        "ga": snapshot["ga"],
        "template_fingerprint": snapshot["template_fingerprint"],
        "sample_seed": snapshot["sample_seed"],
        "vendor_aggregate": snapshot["vendor_aggregate"],
        "dataset_aggregate": snapshot["dataset_aggregate"],
    }
