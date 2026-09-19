#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

from dar.diagnoser import llm as LLM

BUNDLE_SCHEMA = "b6_t1_bundle/v1"

DRIVER_PARSE_RETRIES = 0

DEFAULT_LATENCY_SAMPLE = 30

T1B_PASS = "t1b_latency"
T1B_SEED_PREFIX = "B6_T1B_LATENCY:"
T1B_SEED_PREFIX_MAIN = "B6_T1B_LATENCY_MAIN:"


def t1b_sample_size(n_main: int, n_ext: int, latency_sample: int) -> int:
    if min(n_main, n_ext, latency_sample) <= 0:
        raise SystemExit(
            "")
    raw = -(-n_ext * latency_sample // n_main)
    if raw > n_ext:
        raise SystemExit(
            "")
    return raw


def t1b_cells(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = {}
    for r in rows:
        cells.setdefault(f"{r['cat']}|{r['mode']}|{r['axis']}", []).append(r["run_id"])
    return {k: sorted(v) for k, v in sorted(cells.items())}


def t1b_stratified_ids(rows: list[dict[str, Any]], *, subset_seed: int, per_cell: int,
                       domain: str) -> list[str]:
    if per_cell <= 0:
        raise SystemExit("")
    out: list[str] = []
    for cell, pool in t1b_cells(rows).items():
        if len(pool) < per_cell:
            raise SystemExit(
                "")
        seed = int.from_bytes(hashlib.sha256(
            f"{domain}{subset_seed}:{cell}".encode("utf-8")).digest()[:8], "big")
        out.extend(sorted(random.Random(seed).sample(pool, per_cell)))
    return sorted(out)


def t1b_subset_ids(ext_rows: list[dict[str, Any]], *, subset_seed: int, n: int,
                   domain: str = T1B_SEED_PREFIX) -> list[str]:
    pool = sorted(r["run_id"] for r in ext_rows)
    if len(pool) != len(set(pool)):
        raise SystemExit("")
    if n > len(pool):
        raise SystemExit("")
    seed = int.from_bytes(
        hashlib.sha256(f"{domain}{subset_seed}".encode("utf-8")).digest()[:8], "big")
    return sorted(random.Random(seed).sample(pool, n))


def load_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                     list[dict[str, Any]]]:
    blob = path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)), mode="r") as tar:
        def _read(name: str) -> str:
            f = tar.extractfile(name)
            if f is None:
                raise SystemExit(
                    "")
            return f.read().decode("utf-8")
        manifest = json.loads(_read("manifest.json"))
        main_rows = [json.loads(l) for l in _read("requests/main.jsonl").splitlines() if l]
        ext_rows = [json.loads(l) for l in _read("requests/extended.jsonl").splitlines() if l]
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise SystemExit(
            "")
    sysp = manifest["system_prompt"]
    got = hashlib.sha256(sysp.encode("utf-8")).hexdigest()
    if got != manifest["system_prompt_sha256"]:
        raise SystemExit(
            "")
    bad = [r["run_id"] for r in (main_rows + ext_rows)
           if r.get("system_sha256") != manifest["system_prompt_sha256"]]
    if bad:
        raise SystemExit(
            "")
    for arm, rows in (("main", main_rows), ("extended", ext_rows)):
        n = manifest["arms"][arm]["n_requests"]
        if len(rows) != n:
            raise SystemExit(
                "")
    return manifest, main_rows, ext_rows


def assert_sampling_matches_config(manifest: dict[str, Any], cfg_local: dict[str, Any]) -> None:
    got = manifest["sampling"]
    for k in ("model", "temperature", "top_p", "top_k", "min_p", "max_tokens",
              "parse_retries"):
        if got.get(k) != cfg_local.get(k):
            raise SystemExit(
                "")
    if int(got["max_tokens"]) != 2048:
        raise SystemExit(
            "")


def make_client(base_url: str) -> Any:
    if not base_url:
        raise SystemExit("")
    if "${" in base_url:
        raise SystemExit(
            "")
    from openai import OpenAI
    return OpenAI(api_key="EMPTY", base_url=base_url)


class ResponseWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def answered_run_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise SystemExit(
                "") from None
        rid = row.get("run_id")
        if not rid:
            raise SystemExit("")
        out.add(str(rid))
    return out


def run_requests(rows: list[dict[str, Any]], system: str, complete_fn: Any,
                 writer: ResponseWriter, *, concurrency: int, latency_sample: int,
                 progress_every: int = 100) -> dict[str, Any]:
    done = {"n": 0, "n_error": 0, "n_zero_usage": 0}
    t_start = time.monotonic()
    lock = threading.Lock()

    def _one(row: dict[str, Any], conc: int) -> None:
        t0 = time.monotonic()
        err = None
        resp: dict[str, Any] = {}
        try:
            resp = complete_fn(system, row["user"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        latency = (time.monotonic() - t0) * 1000.0
        p = int(resp.get("prompt_tokens") or 0)
        c = int(resp.get("completion_tokens") or 0)
        out = {"run_id": row["run_id"], "arm": row["arm"], "cat": row["cat"],
               "tid": row["tid"], "mode": row["mode"], "axis": row["axis"],
               "step_index": row.get("step_index"),
               "trigger_index": row.get("trigger_index"),
               "text": resp.get("text"), "prompt_tokens": p, "completion_tokens": c,
               "finish_reason": resp.get("finish_reason"),
               "reasoning_len": resp.get("reasoning_len"),
               "n_choices": resp.get("n_choices"),
               "attempt": DRIVER_PARSE_RETRIES + 1,
               "latency_ms": latency, "concurrency": conc, "error": err}
        writer.write(out)
        with lock:
            done["n"] += 1
            done["n_error"] += int(err is not None)
            done["n_zero_usage"] += int(err is None and p + c <= 0)
            if done["n"] % progress_every == 0:
                el = time.monotonic() - t_start

    head, tail = rows[:latency_sample], rows[latency_sample:]
    for row in head:
        _one(row, 1)
    if tail:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda r: _one(r, concurrency), tail))
    done["wall_s"] = time.monotonic() - t_start
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="both", choices=("main", "extended", "both"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--latency-sample", type=int, default=DEFAULT_LATENCY_SAMPLE)
    ap.add_argument("--latency-subset", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.concurrency < 1:
        raise SystemExit("")
    import yaml
    cfg_path = Path(args.config) if args.config else (LAB / "configs" / "b6_diagnosers.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    local = cfg["llm"]["local"]

    bundle = Path(args.bundle)
    manifest, main_rows, ext_rows = load_bundle(bundle)
    assert_sampling_matches_config(manifest, local)
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()

    rows = {"main": main_rows, "extended": ext_rows,
            "both": main_rows + ext_rows}[args.arm]
    if args.arm in ("extended", "both") and not ext_rows:
        raise SystemExit(
            "")
    if args.limit:
        rows = rows[:args.limit]

    concurrency, latency_sample = args.concurrency, args.latency_sample
    t1b: dict[str, Any] | None = None
    if args.latency_subset:
        if args.limit:
            raise SystemExit("")
        seed_bundle = int(manifest["sampling_frame"]["subset_seed"])
        seed_cfg = int(cfg["measurement"]["subset_seed"])
        if seed_bundle != seed_cfg:
            raise SystemExit(
                "")
        by_id = {r["run_id"]: r for r in main_rows + ext_rows}
        rows, arms = [], {}
        if args.arm in ("main", "both"):
            cells = t1b_cells(main_rows)
            per = -(-args.latency_sample // len(cells))
            arm_ids = t1b_stratified_ids(main_rows, subset_seed=seed_bundle, per_cell=per,
                                         domain=T1B_SEED_PREFIX_MAIN)
            arms["main"] = {"n": len(arm_ids), "seed_domain": T1B_SEED_PREFIX_MAIN,
                            "recipe": "stratified", "per_cell": per, "n_cells": len(cells),
                            "ids": arm_ids}
            rows.extend(by_id[i] for i in arm_ids)
        if args.arm in ("extended", "both"):
            n_arm = t1b_sample_size(len(main_rows), len(ext_rows), args.latency_sample)
            arm_ids = t1b_subset_ids(ext_rows, subset_seed=seed_bundle, n=n_arm,
                                     domain=T1B_SEED_PREFIX)
            arms["extended"] = {"n": n_arm, "seed_domain": T1B_SEED_PREFIX,
                                "recipe": "srs", "ids": arm_ids}
            rows.extend(by_id[i] for i in arm_ids)
        ids = [r["run_id"] for r in rows]
        concurrency, latency_sample = 1, len(rows)
        t1b = {"run_pass": T1B_PASS, "n": len(rows), "arms": arms,
               "recipe": ("n_arm = -(-len(pool) * latency_sample // n_main); "
                          "ids = sorted(random.Random(int.from_bytes(hashlib.sha256("
                          "(domain + str(subset_seed)).encode('utf-8')).digest()[:8], 'big'))"
                          ".sample(sorted(run_ids_of_arm), n_arm))"),
               "subset_seed": seed_bundle, "n_main": len(main_rows), "n_ext": len(ext_rows),
               "rate_source": ("T1 latency_sample / n_main"
                               "latency_sample "),
               "latency_sample": args.latency_sample,
               "ids": ids}

    out = Path(args.out)
    if not out.is_absolute():
        out = LAB / out
    resp_path = out / "responses.jsonl"
    this_pass = T1B_PASS if args.latency_subset else "t1"
    if resp_path.is_file():
        stamp_path = out / "t1_run_stamp.json"
        prev_pass = None
        if stamp_path.is_file():
            try:
                prev_pass = json.loads(stamp_path.read_text(encoding="utf-8")).get("run_pass")
            except ValueError as e:
                raise SystemExit(
                    "")
        if prev_pass is not None and prev_pass != this_pass:
            raise SystemExit(
                "")
        if prev_pass is None and args.latency_subset:
            raise SystemExit(
                "")
    already = answered_run_ids(resp_path)
    todo = [r for r in rows if r["run_id"] not in already]

    if t1b is not None:
        for _a, _d in sorted(t1b["arms"].items()):
            pass
    if not todo:
        return 0

    client = make_client(os.environ.get("DAR_VLLM_BASE_URL", ""))
    complete_fn = LLM.make_local_vllm_complete_fn(
        client, str(local["model"]), temperature=float(local["temperature"]),
        top_p=float(local["top_p"]), top_k=int(local["top_k"]),
        min_p=float(local["min_p"]), max_tokens=int(local["max_tokens"]))

    (out).mkdir(parents=True, exist_ok=True)
    _stamp_text = json.dumps({
        "bundle": bundle.name, "bundle_sha256": bundle_sha,
        "bundle_lab_stamp": manifest["lab_stamp"],
        "prompt_skeleton_fingerprint": manifest["prompt_skeleton_fingerprint"],
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "sampling": manifest["sampling"], "arm": args.arm,
        "driver_parse_retries": DRIVER_PARSE_RETRIES,
        "concurrency": concurrency, "latency_sample": latency_sample,
        "run_pass": "t1" if t1b is None else T1B_PASS,
        **({"latency_subset": t1b} if t1b is not None else {}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "base_url": os.environ.get("DAR_VLLM_BASE_URL", ""),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _stamp_path = out / "t1_run_stamp.json"
    _stamp_path.write_text(_stamp_text, encoding="utf-8")

    writer = ResponseWriter(resp_path)
    try:
        stats = run_requests(todo, manifest["system_prompt"], complete_fn, writer,
                             concurrency=concurrency,
                             latency_sample=min(latency_sample, len(todo)))
    finally:
        writer.close()

    print(f"[b6-llm] → {resp_path}")
    return 0


def selfproof(out: Path, bundle: Path, arm: str, *,
              latency_subset: bool = False,
              latency_sample: int = DEFAULT_LATENCY_SAMPLE) -> int:
    manifest, main_rows, ext_rows = load_bundle(bundle)
    expect = {"main": main_rows, "extended": ext_rows,
              "both": main_rows + ext_rows}[arm]
    if latency_subset:
        seed = int(manifest["sampling_frame"]["subset_seed"])
        want_ids: set[str] = set()
        if arm in ("main", "both"):
            _cells = t1b_cells(main_rows)
            want_ids |= set(t1b_stratified_ids(
                main_rows, subset_seed=seed, domain=T1B_SEED_PREFIX_MAIN,
                per_cell=-(-latency_sample // len(_cells))))
        if arm in ("extended", "both"):
            want_ids |= set(t1b_subset_ids(
                ext_rows, subset_seed=seed,
                n=t1b_sample_size(len(main_rows), len(ext_rows), latency_sample)))
        expect = [r for r in main_rows + ext_rows if r["run_id"] in want_ids]
        stamp = out / "t1_run_stamp.json"
        if stamp.is_file():
            try:
                got = json.loads(stamp.read_text(encoding="utf-8")).get("run_pass")
            except ValueError as e:
                return 8
            if got != T1B_PASS:
                return 8
    resp_path = out / "responses.jsonl"
    if not resp_path.is_file():
        return 6
    rows = []
    for i, line in enumerate(resp_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            return 7
    ids = [r.get("run_id") for r in rows]
    dup = len(ids) - len(set(ids))
    want = {r["run_id"] for r in expect}
    missing = sorted(want - set(ids))
    extra = sorted(set(ids) - want)
    zero_tok = [r["run_id"] for r in rows
                if not r.get("error") and (r.get("prompt_tokens", 0)
                                           + r.get("completion_tokens", 0)) <= 0]
    errors = [r["run_id"] for r in rows if r.get("error")]
    truncated = [r["run_id"] for r in rows if r.get("finish_reason") == "length"]
    empty_text = [r["run_id"] for r in rows
                  if not r.get("error") and not (r.get("text") or "").strip()]
    for name, lst in (("", missing), ("", extra), (" token", zero_tok),
                      ("", errors), ("", truncated), ("", empty_text)):
        if lst:
            pass
    rc = 0
    if dup:
        rc = 7
    if missing:
        rc = 6
    if extra:
        rc = 6
    if zero_tok:
        rc = 7
    if errors:
        pass
    if rc == 0:
        pass
    return rc


if __name__ == "__main__":
    if "--selfproof" in sys.argv:
        _ap = argparse.ArgumentParser()
        _ap.add_argument("--selfproof", action="store_true")
        _ap.add_argument("--bundle", required=True)
        _ap.add_argument("--out", required=True)
        _ap.add_argument("--arm", default="both", choices=("main", "extended", "both"))
        _ap.add_argument("--latency-subset", action="store_true")
        _ap.add_argument("--latency-sample", type=int, default=DEFAULT_LATENCY_SAMPLE)
        _a, _ = _ap.parse_known_args()
        _out = Path(_a.out)
        raise SystemExit(selfproof(_out if _out.is_absolute() else LAB / _out,
                                   Path(_a.bundle), _a.arm,
                                   latency_subset=_a.latency_subset,
                                   latency_sample=_a.latency_sample))
    raise SystemExit(main())
