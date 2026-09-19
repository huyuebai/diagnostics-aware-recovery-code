#!/usr/bin/env python3
from __future__ import annotations

import atexit
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

PASSED = False

if not __debug__:
    raise SystemExit("")

LAB = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(LAB / _p) not in sys.path:
        sys.path.insert(0, str(LAB / _p))

import yaml
from dar.analysis import cost as COST

B6 = LAB / "data" / "b6-20260814"
CFG = LAB / "configs" / "b6_diagnosers.yaml"

SCOPE_AXIS_EXCLUDED = "extended"
SCOPE_SENTENCE = f' = `axis != "{SCOPE_AXIS_EXCLUDED}"`b7 §4 T7 '

LEDGERS = {"llm": "ledger_a_t1.jsonl", "api": "ledger_a_t2.jsonl"}

RATE_SKU = "Qwen3 8B= b6_preregistration.md  LLM  `qwen3-8b`"
RATE_VENDOR = "Alibaba Cloud Int.OpenRouter  provider"
RATE_CALIBER = ("****"
                "—— = Stanage  vLLM ")

LLM_BASIS_PREFIX = "llm-local=0"


def _sentinel() -> None:
    print("GREENLIGHT PASS" if PASSED else "GREENLIGHT FAILED (no verdict)")


atexit.register(_sentinel)


def _rows(name: str) -> list[dict]:
    p = B6 / LEDGERS[name]
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _native(rows: list[dict]) -> dict[str, int]:
    return {"prompt_tokens": sum(int(r["prompt_tokens"]) for r in rows),
            "completion_tokens": sum(int(r["completion_tokens"]) for r in rows),
            "calls": sum(int(r["calls"]) for r in rows),
            "n_rows": len(rows)}


def _per_1k(per_m: float) -> float:
    return per_m / 1000.0


def main() -> int:
    global PASSED

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))["pricing"]
    assert cfg, ""

    q_in_m = float(cfg["qwen3_8b_price_in_per_m_usd"])
    q_out_m = float(cfg["qwen3_8b_price_out_per_m_usd"])
    q_day = str(cfg["qwen3_8b_price_quoted_on"])
    a_in_m = float(cfg["openai_price_in_per_m_usd"]) * float(cfg["batch_discount_factor"])
    a_out_m = float(cfg["openai_price_out_per_m_usd"]) * float(cfg["batch_discount_factor"])

    qwen_basis = (f"qwen3-8b {RATE_VENDOR}"
                  f"${q_in_m}/${q_out_m} per 1M {q_day}"
                  f" = {RATE_CALIBER}"
                  "usd  usage × config in/out ")
    api_basis = (f"openai batch configs/b6_diagnosers.yaml::pricing  in/out "
                 f" × batch_discount_factor${a_in_m}/${a_out_m} per 1M"
                 "")

    print("=" * 96)
    print("=" * 96)

    sel: dict[str, list[dict]] = {}
    nat: dict[str, dict[str, int]] = {}
    for name in LEDGERS:
        allrows = _rows(name)
        assert allrows, ""
        keep = [r for r in allrows if r["axis"] != SCOPE_AXIS_EXCLUDED]
        assert keep, ""
        sel[name], nat[name] = keep, _native(keep)
    ns = {n: nat[n]["n_rows"] for n in nat}
    assert len(set(ns.values())) == 1, (
        "")
    dropped = {n: len(_rows(n)) - nat[n]["n_rows"] for n in LEDGERS}

    q_rates = {"usd_per_1k_prompt": _per_1k(q_in_m),
               "usd_per_1k_completion": _per_1k(q_out_m),
               "basis": qwen_basis}
    a_rates = {"usd_per_1k_prompt": _per_1k(a_in_m),
               "usd_per_1k_completion": _per_1k(a_out_m),
               "basis": api_basis}
    for tag, rr, im, om in (("qwen3-8b", q_rates, q_in_m, q_out_m),
                            ("openai-batch()", a_rates, a_in_m, a_out_m)):
        assert math.isclose(rr["usd_per_1k_prompt"] * 1000.0, im, rel_tol=1e-12), \
            ""
        assert math.isclose(rr["usd_per_1k_completion"] * 1000.0, om, rel_tol=1e-12), \
            ""
    bad = {"usd_per_1k_prompt": q_in_m, "usd_per_1k_completion": q_out_m,
           "basis": qwen_basis}
    assert not math.isclose(bad["usd_per_1k_prompt"] * 1000.0, q_in_m, rel_tol=1e-12), \
        ""

    api_overlay = COST.usd_overlay(nat["api"], a_rates)
    api_ledger = sum(float(r["usd"]) for r in sel["api"])
    assert math.isclose(api_overlay["usd"], api_ledger, rel_tol=1e-12), (
        "")

    llm_usd = {float(r["usd"]) for r in sel["llm"]}
    assert llm_usd == {0.0}, ""
    llm_basis = {str(r["basis"]) for r in sel["llm"]}
    assert llm_basis and all(b.startswith(LLM_BASIS_PREFIX) for b in llm_basis), \
        ""

    q_overlay = COST.usd_overlay(nat["llm"], q_rates)
    print()
    _emit: list[str] = []

    def _p(line: str) -> None:
        _emit.append(line)
        print(line)

    _p("──  5$ ****")
    _p(f"          : qwen3-8b ${q_in_m} in  ${q_out_m} out per 1M")
    _p(f"   SKU    : {RATE_SKU}")
    _p(f"        : {RATE_VENDOR}")
    _p(f"        : {q_day}")
    _p(f"          : {RATE_CALIBER}")
    _p(f"  api   : ${a_in_m} in  ${a_out_m} out per 1M"
       f"=  × batch_discount_factor****")
    for f in ("llm", "api"):
        n = nat[f]
        _p(f"  {f:<4}: n_rows={n['n_rows']} calls={n['calls']} "
           f"prompt={n['prompt_tokens']} completion={n['completion_tokens']}")
    _p(f"  → llmqwen = ${q_overlay['usd']:.6f}")
    _p(f"  → apiconfig = ${api_overlay['usd']:.6f}"
       f"  ==  usd H5")
    ratio = api_overlay["usd"] / q_overlay["usd"]
    _p(f"  → api / llm = {ratio:.4f}×")
    emitted = "\n".join(_emit)
    assert emitted.strip(), ""
    for lbl, tok in ((" in", str(q_in_m)), (" out", str(q_out_m)),
                     ("SKU", RATE_SKU), ("", RATE_VENDOR),
                     ("", q_day), ("", RATE_CALIBER)):
        assert tok, ""
        assert tok in emitted, (
            "")
    assert q_day in q_rates["basis"] and "" in q_rates["basis"], \
        ""

    flip = COST.usd_overlay(nat["llm"], a_rates)
    comp_ratio = nat["llm"]["completion_tokens"] / nat["api"]["completion_tokens"]
    prom_ratio = nat["llm"]["prompt_tokens"] / nat["api"]["prompt_tokens"]
    print()
    assert flip["usd"] > api_overlay["usd"], (
        "")
    assert comp_ratio > 1.0, ""

    src = (LAB / "scripts" / "b7_report.py").read_text(encoding="utf-8")
    neg = len(re.findall(r"pricing|rates|usd_per_1k", src))
    pos = len(re.findall(r"usd|cost", src))
    assert pos > 0, ""
    assert neg == 0, (
        "")
    print()

    live = hashlib.sha256(CFG.read_bytes()).hexdigest()
    man = json.loads((B6 / "manifest.json").read_text(encoding="utf-8"))
    rec = str(man["config"]["sha256"])
    guards = subprocess.run(
        ["/usr/bin/grep", "-rl", rec, "--include=*.py", "--include=*.md",
         str(LAB / "scripts"), str(LAB / "tests"), str(LAB / "src"), str(LAB / "docs")],
        capture_output=True, text=True)
    watchers = [ln for ln in guards.stdout.splitlines() if ln.strip()]
    print()

    print()
    PASSED = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
