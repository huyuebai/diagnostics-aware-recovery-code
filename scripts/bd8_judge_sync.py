#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

import b4_gate as B4
import b5_items as I5
import b5_judge as J

BUDGET_CAP_USD = 30.0

PASSED = 0
_LOCK = threading.Lock()


class SyncJudgeError(SystemExit):
    pass


def _rec_main(it: dict, verdict: str, usd: float) -> dict:
    return {**{f: it[f] for f in B4._ITEM_FIELDS}, "verdict": verdict, "usd": round(usd, 6)}


def _rec_recon(it: dict, v_recon: str, v_main: str | None, usd: float) -> dict:
    return {**{f: it[f] for f in B4._ITEM_FIELDS}, "verdict_recon": v_recon,
            "verdict_main": v_main, "agree": (v_main is not None and v_recon == v_main),
            "usd": round(usd, 6)}


def _load_done(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            out[B4._judge_key(j)] = j
    return out


def run(data: Path, *, cap_usd: float, concurrency: int, limit: int | None,
        resume: bool, progress_every: int) -> int:
    global PASSED
    I5.assert_code_stamped()

    vpath = data / J.VERDICTS_FILENAME
    rpath = data / J.RECON_FILENAME
    if vpath.exists() and not resume:
        raise SyncJudgeError(
            "")

    client, cfg = B4._judge_setup()
    price_in, price_out = B4._prices(cfg, batch=False)
    items = J.load_b5_judge_items(data)
    if limit:
        items = items[:limit]

    from dar.ga_disclosure import reconsistency_subsample
    frac = float(cfg["reconsistency_subsample"])
    recon_ids = (set(reconsistency_subsample([B4._custom_id(it) for it in items], frac=frac))
                 if 0 < frac < 1 else set())
    recon_items = [it for it in items if B4._custom_id(it) in recon_ids]

    done_main = _load_done(vpath) if resume else {}
    done_recon = _load_done(rpath) if resume else {}
    todo_main = [it for it in items if B4._judge_key(it) not in done_main]
    todo_recon = [it for it in recon_items if B4._judge_key(it) not in done_recon]
    spent = sum(float(j.get("usd") or 0) for j in done_main.values()) + \
        sum(float(j.get("usd") or 0) for j in done_recon.values())


    def one(it: dict) -> tuple[dict, str, float]:
        last: Exception | None = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(**B4._build_body(cfg, it["prompt"]))
                u = resp.usage
                usd = u.prompt_tokens * price_in + u.completion_tokens * price_out
                return it, B4._parse_verdict(resp.choices[0].message.content), usd
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise SyncJudgeError("")

    def drain(todo: list[dict], out_path: Path, make_rec, label: str) -> int:
        nonlocal spent
        n = 0
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as fh:
            for i in range(0, len(todo), concurrency):
                if spent >= cap_usd:
                    raise SyncJudgeError(
                        "")
                wave = todo[i:i + concurrency]
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    for it, v, usd in ex.map(one, wave):
                        with _LOCK:
                            fh.write(json.dumps(make_rec(it, v, usd), ensure_ascii=False) + "\n")
                            fh.flush()
                            spent += usd
                            n += 1
                if progress_every and n % progress_every < concurrency:
                    rate = n / max(1e-9, time.time() - t0)
                    left = (len(todo) - n) / max(1e-9, rate)
        return n

    n_main = drain(todo_main, vpath, lambda it, v, usd: _rec_main(it, v, usd), "")
    main_by_key = {**done_main}
    for line in vpath.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            main_by_key[B4._judge_key(j)] = j
    n_rec = drain(todo_recon, rpath,
                  lambda it, v, usd: _rec_recon(
                      it, v, (main_by_key.get(B4._judge_key(it)) or {}).get("verdict"), usd),
                  "")

    if limit:
        PASSED = 1
        print("SYNC JUDGE SMOKE OK")
        return 0

    J.attach_run_ids(data)
    plan_ids = sorted({it["run_id"] for it in items})
    J.load_verdicts_singlesource(data, expected_run_ids=plan_ids,
                                 expected_items=J.items_universe(items))
    PASSED = 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--cap-usd", type=float, default=BUDGET_CAP_USD)
    ap.add_argument("--i-have-a-new-ruling", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()
    if args.cap_usd > BUDGET_CAP_USD and not args.i_have_a_new_ruling:
        raise SyncJudgeError(
            "")
    if args.concurrency < 1:
        raise SyncJudgeError("")
    return run(args.data, cap_usd=args.cap_usd, concurrency=args.concurrency,
               limit=args.limit, resume=args.resume, progress_every=args.progress_every)


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        if not PASSED:
            pass
    sys.exit(rc)
