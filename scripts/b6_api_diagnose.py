#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B
import b6_llm_diagnose as DRV
import b6_subset as SUB
from dar.diagnoser import llm as LLM

API_PARSE_RETRIES = 0

T2_PASS = "t2_api"

_MAX_CUSTOM_ID = 64

_USD_ROUNDING_MAX_ERR = 0.5e-6
_USD_RECONCILE_TOL = _USD_ROUNDING_MAX_ERR + 1e-9

_PIN_RE = re.compile(r"gpt-5\.4-\d{4}-\d{2}-\d{2}")


def load_api_cfg() -> dict[str, Any]:
    doc = SUB.load_config()
    api = doc["llm"]["api"]
    pricing = doc["pricing"]
    model = str(api["model"])
    forb = [str(s) for s in api["forbidden_model_substrings"]]
    if not forb:
        raise SystemExit("")
    hit = [s for s in forb if s.lower() in model.lower()]
    if hit:
        raise SystemExit("")
    if not _PIN_RE.fullmatch(model):
        raise SystemExit("")
    if str(api["provider"]) != "openai" or not bool(api["use_batch"]):
        raise SystemExit("")
    if int(api["parse_retries"]) != API_PARSE_RETRIES:
        raise SystemExit("")
    if pricing.get("prompt_cache") is not False:
        raise SystemExit("")
    d = float(pricing["batch_discount_factor"])
    return {"model": model,
            "mct": int(api["max_completion_tokens"]),
            "effort": str(api["reasoning_effort"]),
            "window": str(api["batch_completion_window"]),
            "cap_usd": float(api["max_budget_usd"]),
            "price_in": float(pricing["openai_price_in_per_m_usd"]) * d / 1e6,
            "price_out": float(pricing["openai_price_out_per_m_usd"]) * d / 1e6,
            "allin_per_tok": float(pricing["openai_batch_effective_usd_per_mtok"]) / 1e6,
            "basis": ("openai batchusage × in/out  × "
                      "configs/b6_diagnosers.yaml::pricingest-enqueued $0.690/Mtok "
                      "")}


def usd_of_tokens(p_tok: int, c_tok: int, cfg: dict[str, Any]) -> float:
    return p_tok * cfg["price_in"] + c_tok * cfg["price_out"]


def parse_run_id(cid: str) -> dict[str, str]:
    parts = cid.split("|")
    if len(parts) != 5:
        raise SystemExit("")
    return {"arm": parts[0], "cat": parts[1], "tid": parts[2],
            "mode": parts[3], "axis": parts[4]}


def build_requests(canonical: Path, corpus: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    man_p = canonical / "manifest.json"
    if not man_p.is_file():
        raise SystemExit("")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    main_arm = str(manifest["sampling"]["main_arm"])
    units = SUB.enumerate_units(corpus, main_arm, manifest["subset_ids"])
    if len(units) != int(manifest["n_diagnoses"]):
        raise SystemExit("")
    reqs: list[dict[str, Any]] = []
    for u in units:
        if u["obs"] is None:
            continue
        cid = u["run_id"]
        if len(cid) > _MAX_CUSTOM_ID:
            raise SystemExit("")
        body = LLM.batch_request_body(u["obs"], model=cfg["model"], max_tokens=cfg["mct"],
                                      reasoning_effort=cfg["effort"])
        reqs.append({"custom_id": cid, "method": "POST",
                     "url": "/v1/chat/completions", "body": body})
    ids = [r["custom_id"] for r in reqs]
    if len(ids) != len(set(ids)):
        raise SystemExit("")
    return reqs


def assert_same_input_face_as_bundle(reqs: list[dict[str, Any]], bundle: Path) -> str:
    bman, bmain, _ = DRV.load_bundle(bundle)
    by_id = {r["run_id"]: r for r in bmain}
    got = {q["custom_id"] for q in reqs}
    if got != set(by_id):
        raise SystemExit(
            "")
    sysp = bman["system_prompt"]
    for q in reqs:
        msgs = q["body"]["messages"]
        if msgs[0]["content"] != sysp:
            raise SystemExit("")
        if msgs[1]["content"] != by_id[q["custom_id"]]["user"]:
            raise SystemExit("")
    h = hashlib.sha256()
    h.update(bundle.read_bytes())
    return h.hexdigest()


def est_input_tokens_of_body(body: dict[str, Any]) -> float:
    return sum(len(m["content"]) for m in body["messages"]) / 4.0


def chunk_requests(reqs: list[dict[str, Any]], cfg: dict[str, Any],
                   budget: int = B._DEFAULT_ENQUEUED_BUDGET) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_t = 0.0
    for r in reqs:
        t = est_input_tokens_of_body(r["body"]) + cfg["mct"]
        if cur and cur_t + t > budget:
            chunks.append(cur)
            cur, cur_t = [], 0.0
        cur.append(r)
        cur_t += t
    if cur:
        chunks.append(cur)
    return chunks


def chunk_estimates(chunk: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, float]:
    in_tok = sum(est_input_tokens_of_body(r["body"]) for r in chunk)
    out_cap = len(chunk) * cfg["mct"]
    return {"est_in_tok": in_tok,
            "est_enqueued": in_tok + out_cap,
            "est_usd_upper": in_tok * cfg["price_in"] + out_cap * cfg["price_out"],
            "est_usd_allin": (in_tok + out_cap) * cfg["allin_per_tok"]}


def items_sha256(reqs: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in reqs:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def ledger_path(out: Path) -> Path:
    return out / "batch_chunks_t2.json"


def load_or_init_ledger(out: Path, chunks: list[list[dict[str, Any]]],
                        reqs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    p = ledger_path(out)
    sizes = [len(c) for c in chunks]
    sha = items_sha256(reqs)
    if p.is_file():
        led = json.loads(p.read_text(encoding="utf-8"))
        if led.get("chunk_sizes") != sizes or led.get("items_sha256") != sha:
            raise SystemExit(
                "")
        return led
    led = {"schema": "b6_batch_chunks_t2/v1", "run_pass": T2_PASS,
           "model": cfg["model"], "cap_usd": cfg["cap_usd"],
           "enqueued_budget": B._DEFAULT_ENQUEUED_BUDGET,
           "n_requests": len(reqs), "chunk_sizes": sizes, "items_sha256": sha,
           "chunks": [{"idx": i, "n": len(c), "batch_id": None, "status": "pending",
                       "input_file_id": None, "submitted_intent": False,
                       "actual_usd": None, "n_ok": None, "n_errors": None,
                       "n_no_usage": None,
                       **{k: round(v, 4) for k, v in chunk_estimates(c, cfg).items()}}
                      for i, c in enumerate(chunks)]}
    B._atomic_write_json(p, led)
    return led


def _save(out: Path, led: dict[str, Any]) -> None:
    B._atomic_write_json(ledger_path(out), led)


def spent_usd(led: dict[str, Any]) -> float:
    return sum(float(c["actual_usd"]) for c in led["chunks"] if c["actual_usd"] is not None)


def row_from_batch_line(line: dict[str, Any], *, chunk_idx: int,
                        cfg: dict[str, Any]) -> tuple[dict[str, Any], float]:
    cid = line.get("custom_id")
    if not cid:
        raise SystemExit("")
    fields = parse_run_id(cid)
    err_obj = line.get("error")
    resp = line.get("response") or {}
    status = resp.get("status_code")
    body = resp.get("body") or {}
    usd = 0.0
    if err_obj or status != 200:
        error = f"batch_error:{json.dumps(err_obj, ensure_ascii=False)[:160]}" \
            if err_obj else f"http_{status}"
        text, p_tok, c_tok, finish, model_echo = None, 0, 0, None, None
    else:
        choices = body.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        text = msg.get("content")
        finish = choices[0].get("finish_reason") if choices else None
        usage = body.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        model_echo = body.get("model")
        error = None
        usd = usd_of_tokens(p_tok, c_tok, cfg)
    return ({"run_id": cid, **fields, "text": text, "error": error,
             "prompt_tokens": p_tok, "completion_tokens": c_tok,
             "finish_reason": finish, "attempt": API_PARSE_RETRIES + 1,
             "batch_id": line.get("_batch_id"), "chunk_idx": chunk_idx,
             "model_echo": model_echo}, usd)


def assert_model_echo(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    doc = SUB.load_config()
    forb = [str(s).lower() for s in doc["llm"]["api"]["forbidden_model_substrings"]]
    for r in rows:
        if r.get("error") is not None:
            continue
        echo = r.get("model_echo")
        if not echo:
            raise SystemExit("")
        if cfg["model"] not in str(echo):
            raise SystemExit("")
        low = str(echo).lower()
        bad = [s for s in forb if s in low]
        if bad:
            raise SystemExit("")


def append_rows(out: Path, rows: list[dict[str, Any]]) -> int:
    resp = out / "responses.jsonl"
    have: set[str] = set()
    if resp.is_file():
        have = {json.loads(l)["run_id"] for l in resp.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    n = 0
    with open(resp, "a", encoding="utf-8") as fh:
        for r in rows:
            if r["run_id"] in have:
                continue
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_stamp(out: Path, cfg: dict[str, Any], *, bundle: Path, bundle_sha: str,
                n_requests: int, chunk_sizes: list[int]) -> None:
    p = out / "t2_run_stamp.json"
    stamp = {"run_pass": T2_PASS, "model": cfg["model"],
             "max_completion_tokens": cfg["mct"], "reasoning_effort": cfg["effort"],
             "batch_completion_window": cfg["window"], "cap_usd": cfg["cap_usd"],
             "api_parse_retries": API_PARSE_RETRIES,
             "bundle": bundle.name, "bundle_sha256": bundle_sha,
             "prompt_skeleton_fingerprint": LLM.skeleton_fingerprint(),
             "system_prompt_sha256": hashlib.sha256(
                 LLM.system_prompt().encode("utf-8")).hexdigest(),
             "n_requests": n_requests, "chunk_sizes": chunk_sizes,
             "basis": cfg["basis"]}
    text = json.dumps(stamp, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if p.is_file():
        if p.read_text(encoding="utf-8") == text:
            return
        raise SystemExit("")
    p.write_text(text, encoding="utf-8")


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("")
    from openai import OpenAI
    return OpenAI()


def preflight_one(client, req: dict[str, Any]) -> None:
    body = dict(req["body"])
    r = B._api_retry(client.chat.completions.create, what="preflight", **body)
    text = r.choices[0].message.content if r.choices else None
    pr = LLM.parse_response(text)
    if pr.label is None:
        raise SystemExit(
            "")


def collect_chunk(client, out: Path, led: dict[str, Any], ch: dict[str, Any],
                  chunk: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    lines = B._read_batch_output_lines(client, ch.get("output_file_id"))
    err_lines = B._read_batch_output_lines(client, ch.get("error_file_id"))
    by_cid: dict[str, dict[str, Any]] = {}
    for ln in lines + err_lines:
        ln["_batch_id"] = ch["batch_id"]
        by_cid[ln.get("custom_id")] = ln
    rows: list[dict[str, Any]] = []
    usd_sum, n_err, n_no_usage = 0.0, 0, 0
    for req in chunk:
        cid = req["custom_id"]
        ln = by_cid.get(cid)
        if ln is None:
            rows.append({"run_id": cid, **parse_run_id(cid), "text": None,
                         "error": "batch_error:missing_output", "prompt_tokens": 0,
                         "completion_tokens": 0, "finish_reason": None,
                         "attempt": API_PARSE_RETRIES + 1, "batch_id": ch["batch_id"],
                         "chunk_idx": ch["idx"], "model_echo": None})
            n_err += 1
            continue
        row, usd = row_from_batch_line(ln, chunk_idx=ch["idx"], cfg=cfg)
        if row["error"] is not None:
            n_err += 1
        elif row["prompt_tokens"] + row["completion_tokens"] == 0:
            n_no_usage += 1
        rows.append(row)
        usd_sum += usd
    assert_model_echo(rows, cfg)
    n_new = append_rows(out, rows)
    ch.update({"status": "collected", "actual_usd": round(usd_sum, 6),
               "n_ok": len(rows) - n_err, "n_errors": n_err, "n_no_usage": n_no_usage})
    _save(out, led)


def _answered_run_ids(out: Path) -> set[str]:
    resp = out / "responses.jsonl"
    if not resp.is_file():
        return set()
    return {json.loads(l)["run_id"] for l in resp.read_text(encoding="utf-8").splitlines()
            if l.strip()}


def run_chunks(out: Path, chunks: list[list[dict[str, Any]]], led: dict[str, Any],
               cfg: dict[str, Any], *, allow_create: bool, max_chunks: int | None,
               poll_seconds: int) -> None:
    client = _client()
    did_preflight = any(c["submitted_intent"] for c in led["chunks"])
    n_done = 0
    for ch, chunk in zip(led["chunks"], chunks):
        if ch["status"] == "collected":
            continue
        if max_chunks is not None and n_done >= max_chunks:
            return
        if ch["batch_id"] is None and not allow_create:
            if ch["submitted_intent"]:
                pass
            else:
                pass
            return
        if allow_create and ch["batch_id"] is None:
            answered = _answered_run_ids(out) & {r["custom_id"] for r in chunk}
            if answered:
                raise SystemExit(
                    "")
            est = float(ch["est_usd_upper"])
            if spent_usd(led) + est > cfg["cap_usd"]:
                _save(out, led)
                raise SystemExit(
                    "")
            if not did_preflight:
                preflight_one(client, chunk[0])
                did_preflight = True
            if ch["input_file_id"] is None:
                blob = "".join(json.dumps({k: r[k] for k in
                                           ("custom_id", "method", "url", "body")},
                                          ensure_ascii=False) + "\n" for r in chunk)
                data = blob.encode("utf-8")
                if len(data) > 200 * 1024 * 1024:
                    raise SystemExit("")
                f = B._api_retry(client.files.create, what=f" {ch['idx']} files.create",
                                 file=(f"b6_t2_chunk_{ch['idx']:03d}.jsonl", data),
                                 purpose="batch")
                ch["input_file_id"] = f.id
                _save(out, led)
            existing = None
            if ch["submitted_intent"]:
                existing = B._find_batch_by_input_file(client, ch["input_file_id"], ch["idx"])
            if existing is not None:
                ch["batch_id"] = existing.id
            else:
                ch["submitted_intent"] = True
                _save(out, led)
                b = B._api_retry(client.batches.create, what=f" {ch['idx']} batches.create",
                                 input_file_id=ch["input_file_id"],
                                 endpoint="/v1/chat/completions",
                                 completion_window=cfg["window"])
                ch["batch_id"] = b.id
            ch["status"] = "submitted"
            _save(out, led)
        while True:
            b = B._api_retry(client.batches.retrieve, ch["batch_id"],
                             what=f" {ch['idx']} batches.retrieve")
            st = b.status
            if st in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(poll_seconds)
        ch["status"] = st
        ch["output_file_id"] = getattr(b, "output_file_id", None)
        ch["error_file_id"] = getattr(b, "error_file_id", None)
        _save(out, led)
        if st != "completed":
            errs = getattr(getattr(b, "errors", None), "data", None) or []
            for e in errs:
                print(f"[t2] 🔴 batch error [{getattr(e, 'code', '?')}] "
                      f"{getattr(e, 'message', '?')}")
            raise SystemExit("")
        collect_chunk(client, out, led, ch, chunk, cfg)
        n_done += 1


class _RetrieveOnlyBatches:
    def __init__(self, batches: Any) -> None:
        self._batches = batches

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return self._batches.retrieve(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        raise SystemExit("")


class _RetrieveOnly:
    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def batches(self) -> _RetrieveOnlyBatches:
        return _RetrieveOnlyBatches(self._client.batches)

    def __getattr__(self, name: str) -> Any:
        raise SystemExit("")


def write_batch_usage(out: Path, doc: dict[str, Any]) -> Path:
    p = out / "batch_usage_t2.json"
    text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if p.is_file():
        if p.read_text(encoding="utf-8") == text:
            return p
        raise SystemExit("")
    p.write_text(text, encoding="utf-8")
    return p


def usage_verb(out: Path, cfg: dict[str, Any]) -> int:
    import openai

    led_p = ledger_path(out)
    if not led_p.is_file():
        raise SystemExit("")
    led = json.loads(led_p.read_text(encoding="utf-8"))
    missing = [c["idx"] for c in led["chunks"] if not c.get("batch_id")]
    if missing:
        raise SystemExit("")
    client = _RetrieveOnly(_client())
    chunks: list[dict[str, Any]] = []
    for ch in led["chunks"]:
        b = B._api_retry(client.batches.retrieve, ch["batch_id"],
                         what=f" {ch['idx']} batches.retrieve")
        raw = getattr(b, "usage", None)
        reason = None
        if raw is None:
            usage = None
            reason = "SDK  usage Batch  0 files.content "
        elif hasattr(raw, "model_dump"):
            usage = raw.model_dump()
        elif isinstance(raw, dict):
            usage = dict(raw)
        else:
            usage = None
            reason = f"usage  {type(raw).__name__}  dict/pydantic ⇒ "
        counts = getattr(b, "request_counts", None)
        if counts is not None and hasattr(counts, "model_dump"):
            counts = counts.model_dump()
        elif counts is not None and not isinstance(counts, dict):
            counts = None
        chunks.append({"idx": int(ch["idx"]), "batch_id": ch["batch_id"],
                       "api_status": getattr(b, "status", None),
                       "request_counts": counts, "usage": usage,
                       "usage_absent_reason": reason})
    doc = {"schema": "b6_batch_usage_t2/v1", "run_pass": T2_PASS, "model": cfg["model"],
           "items_sha256": led["items_sha256"], "n_requests": led["n_requests"],
           "sdk_openai_version": getattr(openai, "__version__", "(unknown)"),
           "lab_stamp": SUB.lab_stamp_fingerprint(), "chunks": chunks}
    p = write_batch_usage(out, doc)
    return 0


def verify(out: Path, canonical: Path, corpus: Path, cfg: dict[str, Any]) -> int:
    reqs = build_requests(canonical, corpus, cfg)
    led_p = ledger_path(out)
    rc = 0
    if not led_p.is_file():
        return 6
    led = json.loads(led_p.read_text(encoding="utf-8"))
    if led.get("items_sha256") != items_sha256(reqs):
        rc = 7
    resp = out / "responses.jsonl"
    rows = [json.loads(l) for l in resp.read_text(encoding="utf-8").splitlines()
            if l.strip()] if resp.is_file() else []
    ids = [r["run_id"] for r in rows]
    want = {r["custom_id"] for r in reqs}
    dup = len(ids) - len(set(ids))
    missing = sorted(want - set(ids))
    extra = sorted(set(ids) - want)
    zero_tok = [r["run_id"] for r in rows
                if r.get("error") is None
                and int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0) <= 0]
    errors = [r["run_id"] for r in rows if r.get("error") is not None]
    for name, lst in (("", missing), ("", extra), ("token", zero_tok),
                      ("error", errors)):
        if lst:
            pass
    if dup or extra:
        rc = 7
    if missing and all(c["status"] == "collected" for c in led["chunks"]):
        rc = 6
    if zero_tok:
        rc = 7
    spent = spent_usd(led)
    if spent > cfg["cap_usd"]:
        rc = 7
    usd_by_chunk: dict[int, float] = {}
    rows_by_chunk: dict[Any, int] = {}
    for r in rows:
        k = r.get("chunk_idx")
        if not isinstance(k, int) or isinstance(k, bool):
            k = "</ chunk_idx>"
        rows_by_chunk[k] = rows_by_chunk.get(k, 0) + 1
        if r.get("error") is None and isinstance(k, int):
            usd_by_chunk[k] = usd_by_chunk.get(k, 0.0) + usd_of_tokens(
                int(r.get("prompt_tokens") or 0),
                int(r.get("completion_tokens") or 0), cfg)
    known = {int(ch["idx"]) for ch in led["chunks"]}
    stray = {k: n for k, n in rows_by_chunk.items() if k not in known}
    if stray:
        rc = 7
    for ch in led["chunks"]:
        idx = int(ch["idx"])
        got = usd_by_chunk.get(idx, 0.0)
        if ch.get("actual_usd") is None:
            if rows_by_chunk.get(idx, 0):
                rc = 7
            continue
        diff = abs(got - float(ch["actual_usd"]))
        if diff > _USD_RECONCILE_TOL:
            rc = 7
    if rc == 0:
        pass
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("plan", "submit", "collect", "verify",
                                     "usage"))
    ap.add_argument("--data", default="data/b6-20260814")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle", default="data/b6-input-a5f90686.tar.gz")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args(argv)

    def _p(s: str) -> Path:
        q = Path(s)
        return q if q.is_absolute() else LAB / q

    cfg = load_api_cfg()
    canonical = _p(args.data)
    doc = SUB.load_config()
    corpus = _p(args.corpus) if args.corpus else (LAB / doc["measurement"]["corpus_root"])
    bundle = _p(args.bundle)

    reqs = build_requests(canonical, corpus, cfg)
    bundle_sha = assert_same_input_face_as_bundle(reqs, bundle)
    chunks = chunk_requests(reqs, cfg)
    ests = [chunk_estimates(c, cfg) for c in chunks]
    tot_upper = sum(e["est_usd_upper"] for e in ests)
    tot_allin = sum(e["est_usd_allin"] for e in ests)

    if args.verb == "plan":
        for i, (c, e) in enumerate(zip(chunks, ests)):
            pass
        scale = 600 / len(reqs) if reqs else 1.0
        if tot_upper * scale > cfg["cap_usd"]:
            return 2
        return 0

    if not args.out:
        raise SystemExit("")
    out = _p(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.verb == "verify":
        return verify(out, canonical, corpus, cfg)
    if args.verb == "usage":
        return usage_verb(out, cfg)

    write_stamp(out, cfg, bundle=bundle, bundle_sha=bundle_sha,
                n_requests=len(reqs), chunk_sizes=[len(c) for c in chunks])
    led = load_or_init_ledger(out, chunks, reqs, cfg)
    if args.verb == "submit":
        if len(reqs) > 50000:
            raise SystemExit("")
        scale = 600 / len(reqs)
        if tot_upper * scale > cfg["cap_usd"]:
            raise SystemExit("")
        run_chunks(out, chunks, led, cfg, allow_create=True,
                   max_chunks=args.max_chunks, poll_seconds=args.poll_seconds)
    else:
        run_chunks(out, chunks, led, cfg, allow_create=False,
                   max_chunks=args.max_chunks, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
