#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from dar.ga import ga_verdict
from dar.ga_semantic import ga_ordered_verdict, semantic_items
from dar.taskmodel import VENDOR_TOOLS, load_task

MODES_PERT = ["P1", "P2", "P3", "P4"]
AXIS_OF = {"P1": "transient", "P3": "transient", "P2": "persistent", "P4": "persistent"}
ARMS_DISPATCH = ["none", "oracle"]
ARM_P0 = "p0-baseline"


def _cells(data: Path, arm: str) -> Iterator[Path]:
    base = data / arm / "qwen3-8b" / "fc"
    if not base.is_dir():
        raise FileNotFoundError("")
    yield from sorted(base.glob("c*"))


def _iter_runs(data: Path, arm: str) -> Iterator[tuple[str, str, str, Path]]:
    for cell in _cells(data, arm):
        cat = cell.name
        inf_dir = cell / "inferences"
        for inf in sorted(inf_dir.glob("*_inference.json")):
            stem = inf.name[: -len("_inference.json")]
            tid, mode = stem.rsplit("_", 1)
            yield cat, tid, mode, inf


def _verify_frozen(data: Path) -> None:
    from dar import config_snapshot as cs
    recognized = (*ARMS_DISPATCH, ARM_P0)
    expected = {c for arm in recognized for c in _cells(data, arm)}
    snaps = sorted(sp for arm in recognized
                   for sp in data.glob(f"{arm}/qwen3-8b/fc/*/config_snapshot.json"))
    got = {sp.parent for sp in snaps}
    if not snaps or got != expected:
        missing = sorted(str(c.relative_to(data)) for c in expected - got)
        extra = sorted(str(c.relative_to(data)) for c in got - expected)
        raise SystemExit(
            "")
    unrecognized = sorted(d.name for d in data.iterdir()
                          if d.is_dir() and d.name not in recognized
                          and (d / "qwen3-8b" / "fc").is_dir())
    if unrecognized:
        pass
    freeze_dir = LAB / "tests" / "data"
    bad: list[str] = []
    for sp in snaps:
        snap = json.loads(sp.read_text(encoding="utf-8"))
        cell = sp.parent
        rel = sp.relative_to(data)
        bad += [f"{rel}: {p}" for p in cs.verify_snapshot_frozen(snap, freeze_dir)]
        bad += [f"{rel.parent}/run_manifest.jsonl: {p}"
                for p in cs.audit_bijection(cell / "inferences", cell / "run_manifest.jsonl")]
    if bad:
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        raise SystemExit("")


def _reconcile(data: Path, arm: str, expected_modes: tuple[str, ...]) -> None:
    per_cell_counts = []
    for cell in _cells(data, arm):
        cat = cell.name
        inf_ids = {p.name[: -len("_inference.json")]
                   for p in (cell / "inferences").glob("*_inference.json")}
        ev_ids = {p.name[: -len("_eval.json")]
                  for p in (cell / "evaluations").glob("*_eval.json")}
        missing_ev = inf_ids - ev_ids
        missing_inf = ev_ids - inf_ids
        if missing_ev or missing_inf:
            raise SystemExit(
                "")
        n = len(inf_ids)
        if n % len(expected_modes) != 0:
            raise SystemExit(
                "")
        per_cell_counts.append((cat, n))
    ks = {n // len(expected_modes) for _, n in per_cell_counts}
    if len(ks) != 1:
        raise SystemExit("")


def _assert_cross_arm_rowsets(data: Path) -> None:
    sets: dict[str, set[tuple[str, str, str]]] = {
        arm: {(cat, tid, mode) for cat, tid, mode, _ in _iter_runs(data, arm)}
        for arm in ARMS_DISPATCH}
    if not any(sets.values()):
        raise SystemExit(
            "")

    def _fmt(rows: set[tuple[str, str, str]], k: int = 10) -> str:
        s = sorted("/".join(r) for r in rows)
        return f"{s[:k]}{'…' if len(s) > k else ''}"

    for a, b in itertools.combinations(ARMS_DISPATCH, 2):
        only_a, only_b = sets[a] - sets[b], sets[b] - sets[a]
        if only_a or only_b:
            raise SystemExit(
                "")

    dispatch_tasks = {(cat, tid) for s in sets.values() for cat, tid, _ in s}
    p0_tasks = {(cat, tid) for cat, tid, _, _ in _iter_runs(data, ARM_P0)}
    extra_p0 = p0_tasks - dispatch_tasks
    if extra_p0:
        raise SystemExit(
            "")
    missing_p0 = dispatch_tasks - p0_tasks
    if missing_p0:
        pass


def _load_messages(inf: Path) -> list[dict[str, Any]]:
    return json.loads(inf.read_text(encoding="utf-8"))["messages"]


def _tool_descriptions() -> dict[str, str]:
    import yaml
    out: dict[str, str] = {}
    for f in sorted(VENDOR_TOOLS.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for tool in data.get("tools") or []:
            name, desc = tool.get("name"), tool.get("description")
            if name:
                out[str(name)] = str(desc or "")
    return out


def _materiality(data_root_lab: Path) -> dict:
    return json.loads((LAB / "provenance" / "materiality_manifest.json")
                      .read_text(encoding="utf-8"))["tasks"]


def _defects() -> dict:
    return json.loads((LAB / "provenance" / "gt_defect_register.json")
                      .read_text(encoding="utf-8"))["tasks"]


def _defect_disposition(defects: dict, tid: str, axis: str | None) -> str:
    d = defects.get(tid) or {}
    disp = d.get("disposition", "clean")
    if disp == "exclude_cell_if_victim_is_availability" and axis:
        return ("exclude_cell" if d["cell_disposition"][axis] == "exclude_cell"
                else "keep_with_note")
    return disp


ITEM_ELIGIBLE_REASON_CODES = ("arg_divergence", "action_substitution_candidate")


def is_item_eligible(verdict: str, reason_code: Any) -> bool:
    return verdict == "undetermined" and reason_code in ITEM_ELIGIBLE_REASON_CODES


def layer_a_row_and_items(arm: str, cat: str, tid: str, mode: str,
                          msgs: list[dict[str, Any]], p0: dict[str, Any],
                          task_mode: dict[str, Any], *,
                          mat: dict, defects: dict, descs: dict[str, str]
                          ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    axis = AXIS_OF.get(mode)
    v = ga_ordered_verdict(msgs, p0, task_mode)
    row = {"arm": arm, "cat": cat, "tid": tid, "mode": mode,
           "axis": axis, "verdict": v["verdict"],
           "reason_code": v.get("reason_code"), "layer": v.get("layer"),
           "material": (mat[cat][tid][axis]["any_material"] if axis else None),
           "defect": _defect_disposition(defects, tid, axis)}
    items: list[dict[str, Any]] = []
    if is_item_eligible(v["verdict"], v.get("reason_code")):
        user_query = p0.get("user_input")
        for _pid, its in semantic_items(msgs, p0).items():
            for it in its:
                items.append({
                    "arm": arm, "cat": cat, "tid": tid, "mode": mode,
                    "path_id": it.path_id, "idx": it.idx, "kind": it.kind,
                    "tool": it.tool, "arg": it.arg,
                    "prompt": it.render_prompt(user_query if isinstance(user_query, str)
                                               else json.dumps(user_query, ensure_ascii=False),
                                               descs.get(it.tool, "")),
                })
    return row, items


def stage_a(data: Path) -> int:
    mat = _materiality(data)
    defects = _defects()
    descs = _tool_descriptions()

    for arm in ARMS_DISPATCH:
        _reconcile(data, arm, tuple(MODES_PERT))
    _reconcile(data, ARM_P0, ("P0",))
    _assert_cross_arm_rowsets(data)
    _verify_frozen(data)

    rows: list[dict[str, Any]] = []
    items_out: list[dict[str, Any]] = []
    n_item_total = 0

    def _emit(arm: str, cat: str, tid: str, mode: str, inf: Path) -> None:
        nonlocal n_item_total
        msgs = _load_messages(inf)
        p0 = load_task(cat, tid, "P0")
        task_mode = p0 if mode == "P0" else load_task(cat, tid, mode)
        row, its = layer_a_row_and_items(arm, cat, tid, mode, msgs, p0, task_mode,
                                         mat=mat, defects=defects, descs=descs)
        rows.append(row)
        items_out.extend(its)
        n_item_total += len(its)

    for arm in ARMS_DISPATCH:
        for cat, tid, mode, inf in _iter_runs(data, arm):
            _emit(arm, cat, tid, mode, inf)
    for cat, tid, mode, inf in _iter_runs(data, ARM_P0):
        _emit(ARM_P0, cat, tid, mode, inf)

    (data / "ga_layerA.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (data / "judge_items.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in items_out) + "\n", encoding="utf-8")

    by_arm = collections.Counter(r["arm"] for r in rows)
    print("| arm | mode | achieved | undetermined | not_achieved | n |")
    print("|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["arm"], r["mode"])].append(r)
    for k in sorted(g):
        gg = g[k]
        ach = sum(1 for r in gg if r["verdict"] == "achieved")
        und = sum(1 for r in gg if r["verdict"] == "undetermined")
        nach = sum(1 for r in gg if r["verdict"] == "not_achieved")
        print(f"| {k[0]} | {k[1]} | {ach} | {und} | {nach} | {len(gg)} |")
    rc = collections.Counter(r["reason_code"] for r in rows)
    subst = sum(1 for it in items_out if it["kind"] == "action_substitution")
    return 0


def _forbidden_substrings(doc: dict) -> list[str]:
    out: list[str] = []
    for src in (doc.get("forbidden_model_substrings"),
                (doc.get("judge") or {}).get("forbidden_model_substrings")):
        for s in src or []:
            if str(s) not in out:
                out.append(str(s))
    return out


def _judge_cfg_from_doc(doc: dict) -> dict[str, Any]:
    cfg = doc["judge"]
    model = str(cfg["model"])
    for bad in _forbidden_substrings(doc):
        if bad.lower() in model.lower():
            raise SystemExit("")
    import re
    if not re.fullmatch(r"gpt-5\.4-\d{4}-\d{2}-\d{2}", model):
        raise SystemExit("")
    return cfg


def _load_judge_cfg() -> dict[str, Any]:
    import yaml
    return _judge_cfg_from_doc(
        yaml.safe_load((LAB / "configs" / "judge_b4.yaml").read_text(encoding="utf-8")))


def _parse_verdict(text: str) -> str:
    from dar.ga_semantic import VERDICT_VOCAB
    up = (text or "").strip().upper()
    for tok in VERDICT_VOCAB:
        if up == tok or up.startswith(tok):
            return tok
    raise ValueError("")


_ITEM_FIELDS = ("arm", "cat", "tid", "mode", "path_id", "idx", "kind", "tool")


def _judge_key(d: dict) -> tuple:
    return (d["arm"], d["cat"], d["tid"], d["mode"], d["path_id"], d["idx"])


def _custom_id(it: dict, recon: bool = False) -> str:
    cid = "|".join(str(it[f]) for f in ("arm", "cat", "tid", "mode", "path_id", "idx"))
    if recon:
        cid += "|R"
    if len(cid) > 64:
        raise SystemExit("")
    return cid


def _parse_custom_id(cid: str) -> tuple[tuple, bool]:
    recon = cid.endswith("|R")
    core = cid[:-2] if recon else cid
    arm, cat, tid, mode, path_id, idx = core.split("|")
    return (arm, cat, tid, mode, path_id, int(idx)), recon


def _err_record(cid: str, kind: str, note: str = "") -> dict:
    try:
        _, recon = _parse_custom_id(cid)
    except Exception:
        recon = cid.endswith("|R")
    return {"cid": cid, "recon": recon, "kind": kind, "note": note}


def _norm_err(e: Any) -> dict:
    if isinstance(e, dict):
        return e
    s = str(e)
    cid, _, rest = s.partition("(")
    note = rest.rstrip(")")
    return _err_record(cid, "write" if note.startswith("no-matching-item") else "request", note)


def _err_brief(e: dict) -> str:
    tail = str(e.get("kind", "?")) + (f":{e['note']}" if e.get("note") else "")
    return f"{e.get('cid', '?')}[{tail}]"


def _covered_main(n_main_ok: int, errors: list[dict]) -> int:
    return n_main_ok + sum(1 for e in errors
                           if e.get("kind") == "request" and not e.get("recon"))


def _build_body(cfg: dict, prompt: str) -> dict:
    body = {"model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": cfg.get("reasoning_effort", "low"),
            "max_completion_tokens": int(cfg["max_completion_tokens"])}
    temp = cfg.get("temperature")
    if temp is not None:
        body["temperature"] = float(temp)
    return body


def _est_input_tokens(prompt: str) -> float:
    return len(prompt) / 4.0


def _prices(cfg: dict, batch: bool) -> tuple[float, float]:
    disc = float(cfg["batch_discount_factor"]) if batch else 1.0
    return (float(cfg["price_in_per_m_usd"]) / 1e6 * disc,
            float(cfg["price_out_per_m_usd"]) / 1e6 * disc)


def _judge_setup():
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("")
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("")
    return OpenAI(), _load_judge_cfg()


def _load_judge_items(data: Path, limit: int | None) -> list[dict]:
    items = [json.loads(x) for x in
             (data / "judge_items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if not items:
        raise SystemExit("")
    return items[:limit] if limit else items


def stage_judge(data: Path, limit: int | None, resume: bool) -> int:
    import time
    vpath = data / "judge_verdicts.jsonl"
    if vpath.exists() and not resume:
        raise SystemExit(
            "")
    client, cfg = _judge_setup()
    price_in, price_out = _prices(cfg, batch=False)
    cap = float(cfg["max_budget_usd"])
    items = _load_judge_items(data, limit)

    done: dict[tuple, str] = {}
    if resume and vpath.exists():
        for line in vpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                done[_judge_key(j)] = j["verdict"]

    spent = 0.0
    fout = open(vpath, "a", encoding="utf-8")

    def _one_call(prompt: str) -> tuple[str, float]:
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(**_build_body(cfg, prompt))
                u = resp.usage
                usd = (u.prompt_tokens * price_in + u.completion_tokens * price_out)
                return _parse_verdict(resp.choices[0].message.content), usd
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("")

    n_new = 0
    for it in items:
        k = _judge_key(it)
        if k in done:
            continue
        if spent >= cap:
            fout.close()
            raise SystemExit("")
        v, usd = _one_call(it["prompt"])
        spent += usd
        rec = {**{f: it[f] for f in _ITEM_FIELDS}, "verdict": v, "usd": round(usd, 5)}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_new += 1
        if n_new % 50 == 0:
            pass
    fout.close()
    return 0


_ACTIVE_BATCH_STATUSES = ("validating", "in_progress", "finalizing", "completed")
_MAX_BATCH_REQUESTS = 50000
_MAX_BATCH_INPUT_BYTES = 200 * 1024 * 1024


def stage_judge_submit(data: Path, limit: int | None, force: bool) -> int:
    from dar.ga_disclosure import reconsistency_subsample
    client, cfg = _judge_setup()
    subp = data / "batch_submit.json"

    if subp.exists() and not force:
        prior = json.loads(subp.read_text(encoding="utf-8"))
        try:
            st = client.batches.retrieve(prior["batch_id"]).status
        except Exception as e:
            st = f"unknown({e})"
        if any(st == s for s in _ACTIVE_BATCH_STATUSES):
            raise SystemExit(
                "")

    items = _load_judge_items(data, limit)

    frac = float(cfg["reconsistency_subsample"])
    recon_ids = (set(reconsistency_subsample([_custom_id(it) for it in items], frac=frac))
                 if 0 < frac < 1 else set())

    req: list[dict] = []
    seen: set[str] = set()
    for it in items:
        cid = _custom_id(it)
        if cid in seen:
            raise SystemExit("")
        seen.add(cid)
        body = _build_body(cfg, it["prompt"])
        req.append({"custom_id": cid, "method": "POST",
                    "url": "/v1/chat/completions", "body": body})
        if cid in recon_ids:
            req.append({"custom_id": _custom_id(it, recon=True), "method": "POST",
                        "url": "/v1/chat/completions", "body": body})

    price_in, price_out = _prices(cfg, batch=True)
    in_tok = sum(_est_input_tokens(it["prompt"]) for it in items)
    in_tok += sum(_est_input_tokens(it["prompt"]) for it in items if _custom_id(it) in recon_ids)
    out_tok = len(req) * int(cfg["max_completion_tokens"])
    est = in_tok * price_in + out_tok * price_out
    cap = float(cfg["max_budget_usd"])
    n_recon = len(req) - len(items)
    if len(req) > _MAX_BATCH_REQUESTS:
        raise SystemExit("")
    if est > cap:
        raise SystemExit("")

    inp = data / "batch_input.jsonl"
    inp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in req) + "\n",
                   encoding="utf-8")
    size = inp.stat().st_size
    if size > _MAX_BATCH_INPUT_BYTES:
        raise SystemExit("")

    try:
        pf = client.chat.completions.create(**req[0]["body"])
        _parse_verdict(pf.choices[0].message.content)
    except Exception as e:
        raise SystemExit("")

    with open(inp, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                              completion_window=cfg.get("batch_completion_window", "24h"))
    sub = {"batch_id": b.id, "input_file_id": up.id, "model": cfg["model"],
           "n_requests": len(req), "n_main": len(items), "n_recon": n_recon,
           "est_usd_upper": round(est, 2), "batch_discount_factor": float(cfg["batch_discount_factor"]),
           "status_at_submit": b.status}
    subp.write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[judge-submit] batch_id={b.id} status={b.status} → {subp}")
    return 0


def _read_batch_output_lines(client, file_id) -> list[dict]:
    if not file_id:
        return []
    text = client.files.content(file_id).text
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def stage_judge_collect(data: Path) -> int:
    client, cfg = _judge_setup()
    subp = data / "batch_submit.json"
    if not subp.exists():
        raise SystemExit("")
    sub = json.loads(subp.read_text(encoding="utf-8"))
    b = client.batches.retrieve(sub["batch_id"])
    print(f"[judge-collect] batch {sub['batch_id']} status={b.status} "
          f"counts={getattr(b, 'request_counts', None)}")
    if b.status != "completed":
        if b.status in ("failed", "expired", "cancelled", "cancelling"):
            errs = getattr(b, "errors", None)
            if errs is not None:
                data_ = getattr(errs, "data", None) or (
                    errs.get("data") if isinstance(errs, dict) else None) or []
                for e in data_:
                    code = getattr(e, "code", None) or (e.get("code") if isinstance(e, dict) else None)
                    msg = getattr(e, "message", None) or (e.get("message") if isinstance(e, dict) else None)
                    line = getattr(e, "line", None) or (e.get("line") if isinstance(e, dict) else None)
                if not data_:
                    pass
            raise SystemExit(
                "")
        return 0

    price_in, price_out = _prices(cfg, batch=True)
    main: dict[tuple, tuple] = {}
    recon: dict[tuple, float] = {}
    recon_v: dict[tuple, str] = {}
    errors: list[dict] = []
    no_usage: list[str] = []

    def _cost(u: dict) -> float:
        return (u.get("prompt_tokens", 0) * price_in + u.get("completion_tokens", 0) * price_out)

    for j in _read_batch_output_lines(client, b.output_file_id):
        cid = j["custom_id"]
        resp = j.get("response") or {}
        if j.get("error") or resp.get("status_code") != 200:
            errors.append(_err_record(cid, "request", f"status:{resp.get('status_code')}"))
            continue
        body = resp.get("body") or {}
        try:
            v = _parse_verdict(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError) as e:
            errors.append(_err_record(cid, "request", f"parse:{e}"))
            continue
        u = body.get("usage")
        if not u:
            no_usage.append(cid)
            u = {}
        usd = _cost(u)
        key, is_recon = _parse_custom_id(cid)
        if is_recon:
            recon_v[key], recon[key] = v, usd
        else:
            main[key] = (v, usd)
    for j in _read_batch_output_lines(client, getattr(b, "error_file_id", None)):
        errors.append(_err_record(j.get("custom_id", "?"), "request", "error_file"))

    items = {_judge_key(it): it for it in
             (json.loads(x) for x in
              (data / "judge_items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    vpath = data / "judge_verdicts.jsonl"
    if vpath.exists():
        existing = {_judge_key(json.loads(x)) for x in
                    vpath.read_text(encoding="utf-8").splitlines() if x.strip()}
        extra = existing - set(main)
        if extra:
            raise SystemExit(
                "")
    written = 0
    with open(vpath, "w", encoding="utf-8") as f:
        for key, (v, usd) in sorted(main.items(), key=lambda kv: repr(kv[0])):
            it = items.get(key)
            if it is None:
                errors.append(_err_record("|".join(map(str, key)), "write", "no-matching-item"))
                continue
            rec = {**{fld: it[fld] for fld in _ITEM_FIELDS}, "verdict": v, "usd": round(usd, 6)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    rpath = data / "judge_recon.jsonl"
    with open(rpath, "w", encoding="utf-8") as f:
        for key in sorted(recon_v, key=repr):
            it = items.get(key)
            f.write(json.dumps(
                {**({fld: it[fld] for fld in _ITEM_FIELDS} if it else
                    dict(zip(("arm", "cat", "tid", "mode", "path_id", "idx"), key))),
                 "verdict_recon": recon_v[key],
                 "verdict_main": main.get(key, (None,))[0],
                 "agree": (key in main and recon_v[key] == main[key][0]),
                 "usd": round(recon.get(key, 0.0), 6)}, ensure_ascii=False) + "\n")

    actual = sum(u for _, u in main.values()) + sum(recon.values())
    if recon_v:
        both = [k for k in recon_v if k in main]
        agree = sum(1 for k in both if recon_v[k] == main[k][0])
        for k in both:
            if recon_v[k] != main[k][0]:
                pass
        if len(both) < len(recon_v):
            pass
    if no_usage:
        pass

    n_main = sub.get("n_main")
    covered = _covered_main(len(main), errors)
    if n_main is not None and covered < n_main:
        pass
    if errors:
        pass
    return 0


def _apply_layer_b(rows: list[dict[str, Any]], data: Path) -> tuple[list[dict[str, Any]], bool]:
    jpath = data / "judge_verdicts.jsonl"
    if not jpath.exists():
        return rows, False
    from dar.ga_semantic import apply_semantic_verdicts, semantic_items
    jv: dict[tuple, str] = {}
    jmeta: dict[tuple, tuple] = {}
    for line in jpath.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        k = (j["arm"], j["cat"], j["tid"], j["mode"], j["path_id"], j["idx"])
        jv[k] = j["verdict"]
        jmeta[k] = (j.get("kind"), j.get("tool"))
    upgraded = 0
    for r in rows:
        if not is_item_eligible(r["verdict"], r.get("reason_code")):
            continue
        mode = r["mode"]
        p0 = load_task(r["cat"], r["tid"], "P0")
        inf = (data / r["arm"] / "qwen3-8b" / "fc" / r["cat"] / "inferences"
               / f"{r['tid']}_{mode}_inference.json")
        msgs = _load_messages(inf)
        items_by_path = semantic_items(msgs, p0)
        vmap = {}
        for its in items_by_path.values():
            for it in its:
                jk = (r["arm"], r["cat"], r["tid"], mode, it.path_id, it.idx)
                if jk not in jv:
                    raise SystemExit(
                        "")
                mk, mt = jmeta.get(jk, (None, None))
                if (mk is not None and mk != it.kind) or (mt is not None and mt != it.tool):
                    raise SystemExit(
                        "")
                vmap[(it.path_id, it.idx)] = jv[jk]
        b = apply_semantic_verdicts(items_by_path, vmap)
        r["verdict"], r["layer"], r["reason_code"] = b["verdict"], "B", "semantic"
        upgraded += 1
    return rows, True


def _run_verdict_fast(paths: list[list[tuple]], vmap: dict[tuple, str]) -> str:
    res = []
    for keys in paths:
        vs = [vmap[k] for k in keys]
        if not vs:
            res.append("undecided")
        elif all(v == "EQUIVALENT" for v in vs):
            res.append("pass")
        elif any(v == "NOT_EQUIVALENT" for v in vs):
            res.append("fail")
        else:
            res.append("undecided")
    if any(r == "pass" for r in res):
        return "achieved"
    if res and all(r == "fail" for r in res):
        return "not_achieved"
    return "undetermined"


def _judge_channel_section(rows: list[dict[str, Any]], data: Path) -> list[str]:
    import random as _random
    from dar import headroom as H
    from dar.ga_semantic import main_estimator, semantic_items, VERDICT_VOCAB
    from dar.taskmodel import load_task

    jpath = data / "judge_verdicts.jsonl"
    if not jpath.exists():
        return []
    jrows = [json.loads(x) for x in jpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    jv = {_judge_key(j): j["verdict"] for j in jrows}
    items = [json.loads(x) for x in
             (data / "judge_items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    by_prompt: dict[str, list[str]] = collections.defaultdict(list)
    for it in items:
        k = _judge_key(it)
        if k in jv:
            by_prompt[it["prompt"]].append(jv[k])
    npair = ndis = 0
    for vs in by_prompt.values():
        for a, b_ in itertools.combinations(vs, 2):
            npair += 1
            ndis += (a != b_)
    q = (ndis / npair) if npair else 0.0

    rows = [json.loads(x) for x in
            (data / "ga_layerA.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    struct: dict[tuple, list[list[tuple]]] = {}
    for r in rows:
        rk = (r["arm"], r["cat"], r["tid"], r["mode"])
        if not is_item_eligible(r["verdict"], r.get("reason_code")):
            continue
        p0 = load_task(r["cat"], r["tid"], "P0")
        inf = (data / r["arm"] / "qwen3-8b" / "fc" / r["cat"] / "inferences"
               / f"{r['tid']}_{r['mode']}_inference.json")
        ibp = semantic_items(_load_messages(inf), p0)
        struct[rk] = [[(rk[0], rk[1], rk[2], rk[3], it.path_id, it.idx) for it in its]
                      for its in ibp.values()]

    rowmap = {(r["arm"], r["cat"], r["tid"], r["mode"]): r for r in rows}
    base = {rk: (_run_verdict_fast(struct[rk], jv) if rk in struct else rowmap[rk]["verdict"])
            for rk in rowmap}

    from dar.ga_semantic import apply_semantic_verdicts as _auth
    for r in rows:
        rk = (r["arm"], r["cat"], r["tid"], r["mode"])
        if rk not in struct:
            continue
        p0 = load_task(r["cat"], r["tid"], "P0")
        inf = (data / r["arm"] / "qwen3-8b" / "fc" / r["cat"] / "inferences"
               / f"{r['tid']}_{r['mode']}_inference.json")
        ibp = semantic_items(_load_messages(inf), p0)
        vm = {(it.path_id, it.idx): jv[(rk[0], rk[1], rk[2], rk[3], it.path_id, it.idx)]
              for its in ibp.values() for it in its}
        if _auth(ibp, vm)["verdict"] != base[rk]:
            raise SystemExit("")

    def _gate(vmap: dict[tuple, str], n_boot: int = 10000) -> dict:
        mk = [{**rowmap[rk], "verdict": vmap[rk]} for rk in rowmap]
        return H.gate_decision(mk, n_boot=n_boot)

    g0 = _gate(base)
    base_red = g0["n_fail"] >= 3
    base_7b = g0["n_fail"] >= 2
    crit_red, crit_7b = [], []
    for j in jrows:
        k = _judge_key(j)
        rk = k[:4]
        if rk not in struct:
            continue
        for alt in sorted(VERDICT_VOCAB - {j["verdict"]}):
            vm_i = dict(jv)
            vm_i[k] = alt
            nv = _run_verdict_fast(struct[rk], vm_i)
            if main_estimator(nv) == main_estimator(base[rk]):
                continue
            g = _gate({**base, rk: nv})
            if (g["n_fail"] >= 3) != base_red:
                crit_red.append((k, j["verdict"], alt))
            elif (g["n_fail"] >= 2) != base_7b:
                crit_7b.append((k, j["verdict"], alt, tuple(g["fails"])))

    red_w = "" if base_red else ""
    b7_w = "" if base_7b else ""
    L = ["", "## ****", "",
         f"-  **q = {q:.1%}** prompt {ndis}/{npair} ", ""]
    L += [f"### ① gate  item  item ×  token →  gate base ****", "",
          f"- basen_fail={g0['n_fail']}/3{'' if base_red else ''}",
          f"- ** #8 {red_w}{len(crit_red)} ** "
          + ("⇒ ****" if not crit_red else f"🔴 **{red_w}**"),
          f"-  **#7②** {b7_w}n_fail≥2 **{len(crit_7b)} **"
          f" **{len({c[0] for c in crit_7b})}  item**"
          "#7②  §3  2 "]
    if crit_7b:
        n_it = len({c[0] for c in crit_7b})
        L += ["", "| item |  |  |  fails |", "|---|---|---|---|"]
        for k, frm, to, fails in crit_7b:
            L.append(f"| `{'|'.join(map(str,k))}` | {frm} | {to} | {list(fails)} |")
        L += ["", f"> ⚠️ **q={q:.1%}  item  ≈ "
              f"1−(1−q)^{n_it} = {1-(1-q)**n_it:.1%}**"
              f"#7②  §3  2 B4R "
              "",
              ">  item  **p0-baseline ** ⇒ ****"
              ""]

    L += ["", "### ②  CI**** #8 ", "",
          "| mode |  CIverdict  | judge-only  item  q  |",
          "|---|---|---|"]
    rng = _random.Random(20260724)
    alts = {v: sorted(VERDICT_VOCAB - {v}) for v in VERDICT_VOCAB}
    for mode in ("P1", "P2", "P3", "P4"):
        means = []
        for _ in range(600):
            vm_i = {k: (rng.choice(alts[v]) if rng.random() < q else v) for k, v in jv.items()}
            vmap = {rk: (_run_verdict_fast(struct[rk], vm_i) if rk in struct else base[rk])
                    for rk in rowmap}
            mk = [{**rowmap[rk], "verdict": vmap[rk]} for rk in rowmap]
            mh = H.mode_headroom(mk, mode, n_boot=1)
            if mh.ci.get("mean") is not None:
                means.append(mh.ci["mean"])
        frozen = g0["per_mode"][mode].ci
        if means:
            means.sort()
            lo, hi = means[int(0.05 * len(means))], means[int(0.95 * len(means))]
            L.append(f"| {mode} | {_fmt_ci(frozen)} | [{lo:+.3f}, {hi:+.3f}]judge  only |")
    L += ["", ">  CI  **judge** CI ****judge "
          " ⇒ ****"
          " B5 `BACKLOG [BD-5]`"]
    return L


def _disclosure_section(rows: list[dict[str, Any]], data: Path) -> list[str]:
    from dar import headroom as H
    from dar.ga_disclosure import (c_counts_by_arm_cell, manski_bounds,
                                   split_decision_rates)
    from dar.ga_semantic import main_estimator
    L = ["", "## §14.5③④", ""]
    jpath = data / "judge_verdicts.jsonl"
    if not jpath.exists():
        L += ["_ B judge_verdicts.jsonl _"]
        return L

    pert = [r for r in rows if r["arm"] in ("none", "oracle")]
    counts = c_counts_by_arm_cell(pert)
    L += ["### (C)  B  undetermined ×  + Manski ", "",
          "| cat | mode | none (ach/und/n) | oracle (ach/und/n) |  Δ | Manski [lo, hi] | (C)- |",
          "|---|---|---|---|---|---|---|"]
    n_flagged = 0
    for cat in ("c1", "c2", "c3", "c4"):
        for mode in MODES_PERT:
            ov = [r["verdict"] for r in pert
                  if r["arm"] == "oracle" and r["cat"] == cat and r["mode"] == mode]
            nv = [r["verdict"] for r in pert
                  if r["arm"] == "none" and r["cat"] == cat and r["mode"] == mode]
            if not ov or not nv:
                continue
            mb = manski_bounds(ov, nv)
            cn = counts.get(("none", cat, mode), {})
            co = counts.get(("oracle", cat, mode), {})
            flag = "⚠️ " if mb["flagged"] else ""
            if mb["flagged"]:
                n_flagged += 1
            L.append(
                f"| {cat} | {mode} "
                f"| {cn.get('achieved',0)}/{cn.get('undetermined',0)}/{cn.get('n',0)} "
                f"| {co.get('achieved',0)}/{co.get('undetermined',0)}/{co.get('n',0)} "
                f"| {mb['main']:+.2f} | [{mb['lo']:+.2f}, {mb['hi']:+.2f}] | {flag} |")
    ref_c = 0
    for mode in MODES_PERT:
        for cat in ("c1", "c2", "c3", "c4"):
            ref = H.reference_tasks(rows, cat, mode)
            for r in rows:
                if (r["arm"] in ("none", "oracle") and r["cat"] == cat and r["mode"] == mode
                        and (r["cat"], r["tid"]) in ref and r["verdict"] == "undetermined"):
                    ref_c += 1
    L += ["", f"**(C)- = {n_flagged}** ⇒  (C) "
          "** gate**",
          f"******** gate ****——"
          f" (C) run = **{ref_c}** ⇒ "
          + ("undetermined→not_achieved** gate **"
             if ref_c == 0 else " (C)  gate "), ""]

    items = [json.loads(x) for x in
             (data / "judge_items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kind_of = {(j["arm"], j["cat"], j["tid"], j["mode"], j["path_id"], j["idx"]): j["kind"]
               for j in items}
    jrows = [json.loads(x) for x in jpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    rates = split_decision_rates([{"kind": kind_of[_judge_key(j)], "verdict": j["verdict"]}
                                  for j in jrows if _judge_key(j) in kind_of])
    L += ["### judge substitution  vs arg ", "",
          "|  | n | EQUIVALENT | NOT_EQUIVALENT | CANNOT_DETERMINE |", "|---|---|---|---|---|"]
    for bucket in ("action_substitution", "arg"):
        b = rates.get(bucket, {})
        c = b.get("counts", {})
        n = b.get("n", 0)
        if not n:
            continue
        L.append(f"| {bucket} | {n} | {c.get('EQUIVALENT',0)} | {c.get('NOT_EQUIVALENT',0)} "
                 f"| {c.get('CANNOT_DETERMINE',0)} |")

    L += ["", "### judge item  × [LM22] ", "",
          "| arm | substitution | arg |  |", "|---|---|---|---|"]
    for arm in ("none", "oracle", ARM_P0):
        sub = sum(1 for j in items if j["arm"] == arm and j["kind"] == "action_substitution")
        arg = sum(1 for j in items if j["arm"] == arm and j["kind"] != "action_substitution")
        L.append(f"| {arm} | {sub} | {arg} | {sub+arg} |")
    L += ["", ">  item  = ****[LM22] "
          " (C)  →  Manski "]

    by_prompt: dict[str, list[str]] = collections.defaultdict(list)
    vmap_j = {_judge_key(j): j["verdict"] for j in jrows}
    for it in items:
        k = _judge_key(it)
        if k in vmap_j:
            by_prompt[it["prompt"]].append(vmap_j[k])
    dup = {p: vs for p, vs in by_prompt.items() if len(vs) > 1}
    incons = {p: vs for p, vs in dup.items() if len(set(vs)) > 1}
    n_pairs = sum(len(vs) for vs in dup.values())
    L += ["", "### judge ** prompt**  10% ", "",
          f"-  prompt {len(by_prompt)} >1  prompt {len(dup)} {n_pairs}  item",
          f"- ** prompt {len(incons)}**"
          f"= {len(incons)/len(dup):.1%} of " if dup else "-  prompt",
          "", ">  ⇒ judge ****temperature reasoning "
          "**** per-run GA "]

    deliv: dict[tuple, int] = collections.Counter()
    nodisp: list[tuple] = []
    for arm in ("oracle",):
        for cat, tid, mode, inf in _iter_runs(data, arm):
            msgs = _load_messages(inf)
            got = any((m.get("metadata") or {}).get("dar_dispatch") for m in msgs)
            deliv[(cat, mode)] += int(got)
            if not got:
                nodisp.append((cat, tid, mode))
    L += ["", "### oracle  dispatch  run+ ", "",
          "| cat | P1 | P2 | P3 | P4 |", "|---|---|---|---|---|"]
    denom: dict[tuple, int] = collections.Counter()
    n_oracle_runs = 0
    for cat, tid, mode, _inf in _iter_runs(data, "oracle"):
        denom[(cat, mode)] += 1
        n_oracle_runs += 1
    for cat in ("c1", "c2", "c3", "c4"):
        L.append(f"| {cat} | " + " | ".join(
            f"{deliv[(cat,m)]}/{denom.get((cat,m),0)}" for m in MODES_PERT) + " |")
    pl = []
    vd = {(r["arm"], r["cat"], r["tid"], r["mode"]): main_estimator(r["verdict"]) for r in rows}
    for cat, tid, mode in nodisp:
        a, b_ = vd.get(("none", cat, tid, mode)), vd.get(("oracle", cat, tid, mode))
        if a is not None and b_ is not None:
            pl.append(int(b_) - int(a))
    L += ["",
          f"- ** dispatch  oracle run = {len(nodisp)}/{n_oracle_runs}** ⇒  run ",
          f"-  GA ** = **n={len(pl)}"
          f" {sum(1 for x in pl if x)}  **{(sum(pl)/len(pl) if pl else 0):+.3f}**",
          "", "> ⚠️ **** = post-treatment selection",
          ">  ≠ 0 ⇒ CRN  seed  bit vLLM "
          "**run **** |Δ| **"]
    return L


def _fmt_ci(ci: dict[str, Any]) -> str:
    if ci.get("lo") is None:
        return "—"
    return (f"{ci['mean']:+.3f} [90% {ci['lo']:+.3f}, {ci['hi']:+.3f}] "
            f"n={ci['n']} (+{ci['n_pos']}/−{ci['n_neg']}/={ci['n_zero']})")


def stage_gate(data: Path, out: str | None) -> int:
    from dar import headroom as H
    from dar.ga_semantic import main_estimator

    _verify_frozen(data)
    _assert_cross_arm_rowsets(data)

    rows = [json.loads(x) for x in
            (data / "ga_layerA.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows, used_judge = _apply_layer_b(rows, data)
    layer = "Bjudge" if used_judge else "Aundetermined→not_achieved "

    g = H.gate_decision(rows)
    L = [f"# B4 headroom gate GA  {layer} [LM17]", "",
         ">  =  GA  = P0-GA=achieved ∩ (any_material) ∩ r>0",
         ">  bootstrap 90% CI#8C1×persistent cost-only §2.3", ""]

    MIN_N = 20
    L += ["## #8 gate ", "",
          "| mode |  | headroom Δ(oracle−none) | 90%CI >0? | n ? |",
          "|---|---|---|---|---|"]
    small_n_modes = []
    for mode in ("P1", "P2", "P3", "P4"):
        mh = g["per_mode"][mode]
        passes = mh.passes
        mark = "" if mode == "P1" else ("✅ " if passes else "❌ ")
        n = mh.ci.get("n") or 0
        nflag = "✅" if n >= MIN_N else f"⚠️ n={n}<{MIN_N}"
        if n < MIN_N and mode != "P1":
            small_n_modes.append(mode)
        contrib = sorted({d["cat"] for d in mh.diffs})
        cats_s = ",".join(contrib) if contrib else ""
        if set(contrib) != set(mh.cats):
            cats_s += f" {','.join(mh.cats)} 0"
        L.append(f"| {mode} | {cats_s} | {_fmt_ci(mh.ci)} | {mark} | {nflag} |")
    if small_n_modes:
        L += ["",
              f"> ⚠️ **{small_n_modes}  n<{MIN_N}** §4 "
              "** `passes` **b4r  §0  ⑦-1"
              " headroom",
              "> percentile bootstrap  **k/n **n=6–36 >0 "
              " **k≥3** `bootstrap_ci`  n  `[1]*k+[0]*(n-k)` "
              "n≤3 ** CI** +1 ⇒ lo=hi=1>0 "]
    red = g["red_line"]
    L += ["",
          f"**P2/P3/P4  90%CI>0**{'🔴  → ' if red else '🟢 '}",
          f" gate {g['fails'] or ''}n_fail={g['n_fail']}/3", "",
          f"**#7② prompt ≥2  P2/P3/P4 CI  0 →  P0 **"
          f"{'⚠️ ' if g['prompt_rule_2_triggered'] else ''}"
          "🗄️ ** 2026-07-30 **② B4R  = "
          "fault_aware headline + ⇒ ****"
          "******** fault_aware "
          " B4 B4R "
          " 2026-08-05  §1-9② fault_aware "
          " :1038/:1046 ——[LM8]", ""]

    L += ["##  achieved-rate#7③ ", "",
          "| cat | mode |  n | none ach | oracle ach | Δ |", "|---|---|---|---|---|---|"]
    for mode in ("P1", "P2", "P3", "P4"):
        for cat in ("c1", "c2", "c3", "c4"):
            if cat in H.MODE_EXCLUDE_CATS.get(mode, frozenset()):
                continue
            ref = H.reference_tasks(rows, cat, mode)
            d = H.paired_diffs(rows, cat, mode, ref)
            if not d:
                continue
            na = sum(1 for x in d if x["none"]) / len(d)
            oa = sum(1 for x in d if x["oracle"]) / len(d)
            L.append(f"| {cat} | {mode} | {len(d)} | {na:.2f} | {oa:.2f} | {oa-na:+.2f} |")

    L += ["", "## c1/c2  P3/P4=", "",
          "| cat | mode | n(,∩P0) | none ach | oracle ach | Δ | ? |",
          "|---|---|---|---|---|---|---|"]
    n_valid_nc = 0
    for mode in ("P3", "P4"):
        for cat in ("c1", "c2"):
            axis = AXIS_OF[mode]
            p0_ach = {(r["cat"], r["tid"]) for r in rows
                      if r["arm"] == "p0-baseline" and r["cat"] == cat
                      and main_estimator(r["verdict"])}
            nm = {(r["cat"], r["tid"]) for r in rows
                  if r["arm"] in ("none", "oracle") and r["cat"] == cat and r["mode"] == mode
                  and r.get("material") is False
                  and r.get("defect") not in H.EXCLUDE_DISPOSITIONS}
            ref = (p0_ach & nm)
            d = H.paired_diffs(rows, cat, mode, ref)
            valid = cat not in H.MODE_EXCLUDE_CATS.get(mode, frozenset())
            n_valid_nc += valid
            vs = "✅" if valid else "❌ "
            if not d:
                L.append(f"| {cat} | {mode} | 0 | — | — | — | {vs} |")
                continue
            na = sum(1 for x in d if x["none"]) / len(d)
            oa = sum(1 for x in d if x["oracle"]) / len(d)
            L.append(f"| {cat} | {mode} | {len(d)} | {na:.2f} | {oa:.2f} | {oa-na:+.2f} | {vs} |")

    L += ["",
          "> ⚠️ **c1×persistentP2/P4**§2.3  oracle  = `GRACEFUL_ABORT`"
          " ⇒  Δ  ⇒ "
          " 2026-07-30B4R  U-7Action  ⇒ GA "
          "Δ  −(none )****——GA ****abort "
          " canonical Action abort run  achievedB4R primary c1-P4  Δ=−3/7"
          "  −5/7****"
          "** §14.4 **§14.4  §2.3  `BACKLOG [BD-4]`",
          f"> ⇒ ** = {n_valid_nc}**** (N3)  null-check "
          " n **——",
          "> ⚠️ ****"]

    L += ["", "## P0-GA achieved  = ", ""]
    for cat in ("c1", "c2", "c3", "c4"):
        p0 = [r for r in rows if r["arm"] == "p0-baseline" and r["cat"] == cat]
        ach = sum(1 for r in p0 if main_estimator(r["verdict"]))
        contrib = sum(1 for m in MODES_PERT
                      if cat not in H.MODE_EXCLUDE_CATS.get(m, frozenset())
                      for _ in H.paired_diffs(rows, cat, m, H.reference_tasks(rows, cat, m)))
        note = ""
        if ach and not contrib:
            note = "  ← ** P0  0 **/"
        L.append(f"- {cat}: P0-GA achieved {ach}/{len(p0)} {contrib}{note}")

    L += _disclosure_section(rows, data)
    L += _judge_channel_section(rows, data)

    text = "\n".join(L) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"[b4-gate/gate] → {out}")
    else:
        print(text)
    return 0


_DEFAULT_ENQUEUED_BUDGET = 800_000


def _api_retry(fn, *a, what: str = "API", tries: int = 6, **kw):
    import time
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last = e
            if i == tries - 1:
                break
            wait = min(5 * 2 ** i, 300)
            time.sleep(wait)
    raise SystemExit("")


_BATCH_LIST_PAGE = 100
_BATCH_LIST_MAX_PAGES = 50


def _find_batch_by_input_file(client, input_file_id: str, idx: int):
    seen = 0
    after: str | None = None
    for page in range(_BATCH_LIST_MAX_PAGES):
        kw = {"limit": _BATCH_LIST_PAGE}
        if after is not None:
            kw["after"] = after
        resp = _api_retry(client.batches.list, what=f" {idx} batches.list p{page}", **kw)
        if not hasattr(resp, "data"):
            raise SystemExit(
                "")
        data = list(resp.data or [])
        for cand in data:
            seen += 1
            if getattr(cand, "input_file_id", None) == input_file_id:
                return cand
        more = getattr(resp, "has_more", None)
        if not data or more is False or (more is None and len(data) < _BATCH_LIST_PAGE):
            return None
        after = getattr(data[-1], "id", None)
        if after is None:
            raise SystemExit(
                "")
    raise SystemExit(
        "")


def _build_all_requests(data: Path, cfg: dict) -> tuple[list[dict], list[dict], int]:
    from dar.ga_disclosure import reconsistency_subsample
    items = _load_judge_items(data, None)
    frac = float(cfg["reconsistency_subsample"])
    recon_ids = (set(reconsistency_subsample([_custom_id(it) for it in items], frac=frac))
                 if 0 < frac < 1 else set())
    reqs: list[dict] = []
    seen: set[str] = set()
    for it in items:
        cid = _custom_id(it)
        if cid in seen:
            raise SystemExit("")
        seen.add(cid)
        body = _build_body(cfg, it["prompt"])
        reqs.append({"custom_id": cid, "method": "POST",
                     "url": "/v1/chat/completions", "body": body})
        if cid in recon_ids:
            reqs.append({"custom_id": _custom_id(it, recon=True), "method": "POST",
                         "url": "/v1/chat/completions", "body": body})
    return items, reqs, len(recon_ids)


def _chunk_by_enqueued(reqs: list[dict], budget: int, mct: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_tok = 0.0
    for r in reqs:
        t = _est_input_tokens(r["body"]["messages"][0]["content"]) + mct
        if cur and cur_tok + t > budget:
            chunks.append(cur)
            cur, cur_tok = [], 0.0
        cur.append(r)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _items_sha256(data: Path) -> str:
    import hashlib
    return hashlib.sha256((data / "judge_items.jsonl").read_bytes()).hexdigest()


def _check_chunk_manifest_identity(man: dict, sizes: list[int], items_sha: str) -> None:
    if man.get("chunk_sizes") != sizes:
        raise SystemExit(
            "")
    stored = man.get("items_sha256")
    if stored is None:
        pass
    elif stored != items_sha:
        raise SystemExit(
            "")


def stage_judge_chunked(data: Path, budget: int, poll_seconds: int, force: bool) -> int:
    import time
    client, cfg = _judge_setup()
    mct = int(cfg["max_completion_tokens"])
    price_in, price_out = _prices(cfg, batch=True)
    cap = float(cfg["max_budget_usd"])

    items, reqs, n_recon = _build_all_requests(data, cfg)
    item_by_key = {_judge_key(it): it for it in items}
    chunks = _chunk_by_enqueued(reqs, budget, mct)
    sizes = [len(c) for c in chunks]

    mpath = data / "batch_chunks.json"
    cdir = data / "chunks"
    cdir.mkdir(exist_ok=True)

    items_sha = _items_sha256(data)
    if mpath.exists() and not force:
        man = json.loads(mpath.read_text(encoding="utf-8"))
        _check_chunk_manifest_identity(man, sizes, items_sha)
    else:
        man = {"enqueued_budget": budget, "n_requests": len(reqs),
               "n_main": len(items), "n_recon": n_recon, "chunk_sizes": sizes,
               "items_sha256": items_sha,
               "chunks": [{"idx": i, "n": len(c), "batch_id": None,
                           "status": "pending", "actual_usd": None} for i, c in enumerate(chunks)]}
        _atomic_write_json(mpath, man)

    def _save() -> None:
        _atomic_write_json(mpath, man)

    spent = sum(c["actual_usd"] or 0.0 for c in man["chunks"])
    n_done = sum(1 for c in man["chunks"] if c["status"] == "collected")

    if n_done == 0:
        try:
            pf = client.chat.completions.create(**reqs[0]["body"])
            _parse_verdict(pf.choices[0].message.content)
        except Exception as e:
            raise SystemExit("")

    for ch, reqs_i in zip(man["chunks"], chunks):
        if ch["status"] == "collected":
            continue
        i = ch["idx"]
        in_tok = sum(_est_input_tokens(r["body"]["messages"][0]["content"]) for r in reqs_i)
        est = in_tok * price_in + len(reqs_i) * mct * price_out
        if spent + est > cap:
            _save()
            raise SystemExit(
                "")

        if ch["batch_id"] is None:
            inp = cdir / f"chunk_{i:03d}.input.jsonl"
            inp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs_i) + "\n",
                           encoding="utf-8")
            if ch.get("input_file_id") is None:
                with open(inp, "rb") as fh:
                    up = _api_retry(client.files.create, file=fh, purpose="batch",
                                    what=f" {i} files.create")
                ch["input_file_id"] = up.id
                _save()
            found = None
            if ch.get("submitted_intent"):
                found = _find_batch_by_input_file(client, ch["input_file_id"], i)
                if found is not None:
                    pass
            if found is None:
                ch["submitted_intent"] = True
                _save()
                found = _api_retry(client.batches.create, input_file_id=ch["input_file_id"],
                                   endpoint="/v1/chat/completions",
                                   completion_window=cfg.get("batch_completion_window", "24h"),
                                   what=f" {i} batches.create")
            b = found
            ch.update(batch_id=b.id, status=b.status, est_usd_upper=round(est, 4),
                      est_enqueued=int(in_tok + len(reqs_i) * mct))
            _save()

        while True:
            b = _api_retry(client.batches.retrieve, ch["batch_id"],
                           what=f" {i} batches.retrieve")
            if b.status in ("completed", "failed", "expired", "cancelled"):
                break
            ch["status"] = b.status
            _save()
            time.sleep(poll_seconds)

        if b.status != "completed":
            ch["status"] = b.status
            _save()
            errs = getattr(b, "errors", None)
            data_ = (getattr(errs, "data", None) or
                     (errs.get("data") if isinstance(errs, dict) else None) or []) if errs else []
            for e in data_:
                code = getattr(e, "code", None) or (e.get("code") if isinstance(e, dict) else None)
                msg = getattr(e, "message", None) or (e.get("message") if isinstance(e, dict) else None)
                print(f"  - [{code}] {msg}", file=sys.stderr)
            raise SystemExit("")

        main_c: dict[str, tuple] = {}
        recon_c: dict[str, tuple] = {}
        errs_c: list[dict] = []
        no_usage_c: list[str] = []
        for j in _read_batch_output_lines(client, b.output_file_id):
            cid = j["custom_id"]
            resp = j.get("response") or {}
            if j.get("error") or resp.get("status_code") != 200:
                errs_c.append(_err_record(cid, "request", f"status:{resp.get('status_code')}"))
                continue
            body = resp.get("body") or {}
            try:
                v = _parse_verdict(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, ValueError) as e:
                errs_c.append(_err_record(cid, "request", f"parse:{e}"))
                continue
            u = body.get("usage")
            if not u:
                no_usage_c.append(cid)
                u = {}
            usd = u.get("prompt_tokens", 0) * price_in + u.get("completion_tokens", 0) * price_out
            key, is_recon = _parse_custom_id(cid)
            (recon_c if is_recon else main_c)["|".join(map(str, key))] = (v, usd)
        for j in _read_batch_output_lines(client, getattr(b, "error_file_id", None)):
            errs_c.append(_err_record(j.get("custom_id", "?"), "request", "error_file"))

        (cdir / f"chunk_{i:03d}.parsed.json").write_text(
            json.dumps({"main": main_c, "recon": recon_c, "errors": errs_c,
                        "no_usage": no_usage_c}, ensure_ascii=False), encoding="utf-8")
        actual = sum(u for _, u in main_c.values()) + sum(u for _, u in recon_c.values())
        ch.update(status="collected", actual_usd=round(actual, 6),
                  n_main_ok=len(main_c), n_errors=len(errs_c), n_no_usage=len(no_usage_c))
        spent += actual
        _save()
        if no_usage_c:
            pass

    main_all: dict[str, tuple] = {}
    recon_all: dict[str, tuple] = {}
    errors: list[dict] = []
    no_usage_all: list[str] = []
    for ch in man["chunks"]:
        p = cdir / f"chunk_{ch['idx']:03d}.parsed.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        main_all.update(d["main"])
        recon_all.update(d["recon"])
        errors += [_norm_err(e) for e in d["errors"]]
        no_usage_all += d.get("no_usage") or []

    vpath, rpath = data / "judge_verdicts.jsonl", data / "judge_recon.jsonl"
    if vpath.exists():
        existing = {"|".join(str(json.loads(x)[f]) for f in
                             ("arm", "cat", "tid", "mode", "path_id", "idx"))
                    for x in vpath.read_text(encoding="utf-8").splitlines() if x.strip()}
        extra = existing - set(main_all)
        if extra:
            raise SystemExit("")
    written = 0
    with open(vpath, "w", encoding="utf-8") as f:
        for ks, (v, usd) in sorted(main_all.items()):
            a, c, t, m, pth, ix = ks.split("|")
            it = item_by_key.get((a, c, t, m, pth, int(ix)))
            if it is None:
                errors.append(_err_record(ks, "write", "no-matching-item"))
                continue
            f.write(json.dumps({**{fld: it[fld] for fld in _ITEM_FIELDS},
                                "verdict": v, "usd": round(usd, 6)}, ensure_ascii=False) + "\n")
            written += 1
    with open(rpath, "w", encoding="utf-8") as f:
        for ks, (v, usd) in sorted(recon_all.items()):
            a, c, t, m, pth, ix = ks.split("|")
            it = item_by_key.get((a, c, t, m, pth, int(ix)))
            f.write(json.dumps(
                {**({fld: it[fld] for fld in _ITEM_FIELDS} if it else
                    {"arm": a, "cat": c, "tid": t, "mode": m, "path_id": pth, "idx": int(ix)}),
                 "verdict_recon": v, "verdict_main": (main_all.get(ks) or (None,))[0],
                 "agree": (ks in main_all and v == main_all[ks][0]),
                 "usd": round(usd, 6)}, ensure_ascii=False) + "\n")

    both = [k for k in recon_all if k in main_all]
    if both:
        agree = sum(1 for k in both if recon_all[k][0] == main_all[k][0])
    if no_usage_all:
        pass
    covered = _covered_main(len(main_all), errors)
    if covered < man["n_main"]:
        pass
    if errors:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["a", "judge", "judge-submit", "judge-collect",
                                      "judge-chunked", "gate"])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--enqueued-budget", type=int, default=_DEFAULT_ENQUEUED_BUDGET)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()
    data = args.data if args.data.is_absolute() else (LAB / args.data)
    if args.stage == "a":
        return stage_a(data)
    if args.stage == "judge":
        return stage_judge(data, args.limit, args.resume)
    if args.stage == "judge-submit":
        return stage_judge_submit(data, args.limit, args.force)
    if args.stage == "judge-collect":
        return stage_judge_collect(data)
    if args.stage == "judge-chunked":
        return stage_judge_chunked(data, args.enqueued_budget, args.poll_seconds, args.force)
    if args.stage == "gate":
        return stage_gate(data, args.out)
    raise SystemExit(f"unreachable stage {args.stage}")


if __name__ == "__main__":
    sys.exit(main())
