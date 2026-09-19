#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

if not __debug__:
    raise SystemExit("")

sys.path.insert(0, "src")

from dar import accounting, headroom
from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

_CI_CALLS: list[dict] = []
_raw_bootstrap_ci = headroom.bootstrap_ci


def bootstrap_ci(diffs, *, alpha, n_boot, seed):
    _CI_CALLS.append({"alpha": alpha, "n_boot": n_boot, "seed": seed, "n": len(diffs)})
    return _raw_bootstrap_ci(diffs, alpha=alpha, n_boot=n_boot, seed=seed)

LEDGER = Path("data/b5-11090655/analysis/ledger_b.jsonl")

BD100_SEED = 20260903
N_BOOT = 10000
ALPHA = 0.10
EXCLUDED_CAT = "c1"
FAIL_TYPES = (TrueType.P1, TrueType.P2, TrueType.P3, TrueType.P4)
EXPECTED_FAIL_TYPES = tuple(t for t in TrueType if t is not TrueType.NP)

MAIN_ROW_KEYS = frozenset({"truth", "n_pairs", "d_calls", "d_ga",
                           "n_achieved_by_arm", "ci_calls", "ci_ga"})

MANDATORY_T14 = (
    "① `W_disp` ⇒ ****"
    "** c1 **——c1×{P2,P4}  GRACEFUL_ABORT",
    "② ** `W_mis`  exploratory ** retry "
    "switch_path ",
    "③ ⛔ ** Δcalls** [LM22] ——"
    " Δcalls  ΔGA ****",
    "④  ITT  **[BD-103] **  `b5_findings.md` §7 "
    "**** `np_waste` —— hit  NP  fault ****"
    "2026-09-03  E = ′",
)
MANDATORY_T15 = (
    "****`n_tool_calls`**** "
    "`DECISIONS.md` §3.1 ****`excess_calls_vs_oracle` ITT "
    " ⟺ `hit == False`⇒ ",
    "c1 `c1×{P2,P4}`  `b4_recovery_cost_ga.md` §2.3  **cost-only **"
    " = `accounting.py::avoided_waste` `b5_analysis.json['cost']`——"
    "****",
)


ANCHORS_T14: tuple[str, ...] = ('', ' `W_mis`  exploratory ', ' Δcalls', '** `np_waste` ')
RECIPE = Path("docs/bd100_gap_recipe.md")

_fails: list[str] = []


def check(name: str, ok: bool, detail: object) -> None:
    print(("  [PASS] " if ok else "  [FAIL] ") + name + " — " + str(detail))
    if not ok:
        _fails.append(name)


_emitted: list[str] = []


def emit(line: str) -> None:
    _emitted.append(line)
    print("   " + line)


def wrong_label(t: TrueType) -> Label:
    diag = TRUE_TYPE_TO_LABEL[t]
    others = [l for l in (Label.TRANSIENT, Label.PERSISTENT) if l is not diag]
    if len(others) != 1:
        raise RuntimeError("")
    return others[0]


def load_itt_rows() -> list[dict]:
    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["batch"] == "main"]


def paired(idx, t: TrueType, arm_a: Label, arm_b: Label, field: str):
    ka = {(c, tid) for (tt, lb, c, tid) in idx if tt == t.value and lb == arm_a.value}
    kb = {(c, tid) for (tt, lb, c, tid) in idx if tt == t.value and lb == arm_b.value}
    members = sorted(ka & kb)
    diffs = [float(idx[(t.value, arm_a.value, c, tid)][field])
             - float(idx[(t.value, arm_b.value, c, tid)][field]) for (c, tid) in members]
    return diffs, members, (arm_a.value, arm_b.value)


def main() -> int:
    print("=" * 96)
    print("=" * 96)
    if not LEDGER.exists():
        print("\nGAP2 FAILED")
        return 1
    allmain = load_itt_rows()

    np_hit = {r["hit"] for r in allmain if r["truth"] == TrueType.NP.value}
    check("T-7 NP  `hit`  False", np_hit == {False},
          "NP  hit  = %s ⇒  fault victim ⇒ `hit` "
          "**NP ** = " % sorted(np_hit))

    itt = [r for r in allmain
           if r["cat"] != EXCLUDED_CAT and r["truth"] != TrueType.NP.value]
    idx = {(r["truth"], r["label"], r["cat"], r["tid"]): r for r in itt}
    cells = collections.Counter((r["truth"], r["label"]) for r in itt)

    leaked = sorted({r["cat"] for r in itt if r["cat"] == accounting.COST_ONLY_CAT})
    check("T-5b cost-only cat  =  `accounting.COST_ONLY_CAT`",
          not leaked and EXCLUDED_CAT == accounting.COST_ONLY_CAT,
          "ITT  cost-only cat  = %s EXCLUDED_CAT=%r  %r"
          % (leaked or "", EXCLUDED_CAT, accounting.COST_ONLY_CAT))

    expect_cells = {(t.value, l.value) for t in FAIL_TYPES for l in Label}
    n_values = set(cells.values())
    check("T-5T-9  =  ∧  n",
          set(cells) == expect_cells and len(n_values) == 1 and n_values != {0}
          and len(idx) == len(itt),
          " %d %s n  = %s(,cat,tid)  %s"
          % (len(cells), set(cells) == expect_cells, n_values, len(idx) == len(itt)))

    cross = sum(1 for r in itt
                if (r.get("actual_recovery_calls") is None) != (r["hit"] is False))
    miss_by_cell = collections.Counter((r["truth"], r["label"]) for r in itt
                                       if r.get("excess_calls_vs_oracle") is None)
    check("T-6  `null(actual_recovery_calls) ⟺ hit == False` ", cross == 0,
          " = %d=  no-hit****= %s"
          % (cross, dict(sorted(miss_by_cell.items()))))
    for f in ("n_tool_calls", "input_tokens", "output_tokens", "ga_achieved"):
        n_null = sum(1 for r in itt if r.get(f) is None)
        check("T-6b ITT  `%s`  ITT " % f, n_null == 0,
              " %d 0 ⇒  ITT headline " % n_null)

    rows_out: list[dict] = []
    _printed_main: list[str] = []
    arm_ids: dict[str, tuple[str, str]] = {}
    for kind in ("W_mis", "W_disp"):
        for t in FAIL_TYPES:
            diag, wrong = TRUE_TYPE_TO_LABEL[t], wrong_label(t)
            a, b = (wrong, diag) if kind == "W_mis" else (diag, Label.NP)
            d_calls, mem, used_arms = paired(idx, t, a, b, "n_tool_calls")
            d_ga, mem2, used_arms2 = paired(idx, t, a, b, "ga_achieved")
            arm_ids["%s/%s" % (kind, t.value)] = used_arms
            if used_arms != used_arms2 or mem != mem2:
                check("/calls vs GA", False,
                      "%s/%s  %s vs %s" % (kind, t.value, used_arms, used_arms2))
                continue
            ci_c = bootstrap_ci(d_calls, alpha=ALPHA, n_boot=N_BOOT, seed=BD100_SEED)
            ci_g = bootstrap_ci([float(x) for x in d_ga],
                                         alpha=ALPHA, n_boot=N_BOOT, seed=BD100_SEED)
            n_ach = {arm.value: sum(1 for (c, tid) in mem
                                    if idx[(t.value, arm.value, c, tid)]["ga_achieved"])
                     for arm in (a, b)}
            rows_out.append({"truth": t.value, "n_pairs": len(mem),
                             "d_calls": ci_c["mean"], "d_ga": ci_g["mean"],
                             "n_achieved_by_arm": n_ach,
                             "ci_calls": (ci_c["lo"], ci_c["hi"]),
                             "ci_ga": (ci_g["lo"], ci_g["hi"])})
            _line = ("     %s  n_pairs=%3d  Δcalls %+8.4f [%+.4f, %+.4f]  ΔGA %+8.4f [%+.4f, %+.4f]"
                     "  n_achieved %s→%d / %s→%d"
                     % (t.value, len(mem), ci_c["mean"], ci_c["lo"], ci_c["hi"],
                        ci_g["mean"], ci_g["lo"], ci_g["hi"],
                        a.value, n_ach[a.value], b.value, n_ach[b.value]))
            _printed_main.append(_line)
            print(_line)
        for unit in ("input_tokens", "output_tokens"):
            for t in FAIL_TYPES:
                diag, wrong = TRUE_TYPE_TO_LABEL[t], wrong_label(t)
                a, b = (wrong, diag) if kind == "W_mis" else (diag, Label.NP)
                d, mem, _ = paired(idx, t, a, b, unit)
                ci = bootstrap_ci(d, alpha=ALPHA, n_boot=N_BOOT, seed=BD100_SEED)
                print("        %s  n=%3d  Δ %+11.2f [%+.2f, %+.2f]"
                      % (t.value, len(mem), ci["mean"], ci["lo"], ci["hi"]))

    bad_keys = [r["truth"] for r in rows_out if set(r) != MAIN_ROW_KEYS]
    ratio_keys = sorted({k for r in rows_out for k in r if "ratio" in k or "per_" in k})
    n_expect = 2 * len(EXPECTED_FAIL_TYPES)
    no_dga = [l for l in _printed_main if "ΔGA" not in l or "Δcalls" not in l]
    check("T-12  =  ∧  ∧ ** Δcalls  ΔGA**",
          not bad_keys and not ratio_keys and len(rows_out) == n_expect
          and len(_printed_main) == n_expect and not no_dga,
          " %d %d %d** `TrueType` ** %s"
          " %s %d"
          % (len(rows_out), len(_printed_main), n_expect, bad_keys or "",
             ratio_keys or "", len(no_dga)))

    bad_arm = []
    for t in FAIL_TYPES:
        diag, wrong = TRUE_TYPE_TO_LABEL[t].value, wrong_label(t).value
        if arm_ids["W_mis/%s" % t.value] != (wrong, diag):
            bad_arm.append("")
        if arm_ids["W_disp/%s" % t.value] != (diag, Label.NP.value):
            bad_arm.append("")
    check("T-13 W_mis =(ℓ_wrong, ℓ_diag)W_disp =(ℓ_diag, np)", not bad_arm,
          " `TRUE_TYPE_TO_LABEL`%s" % (bad_arm or ""))

    diag_rows: list[dict] = []
    for t in FAIL_TYPES:
        diag, wrong = TRUE_TYPE_TO_LABEL[t], wrong_label(t)
        d, mem, _ = paired(idx, t, wrong, diag, "n_tool_calls")
        sub = [(c, tid) for (c, tid) in mem
               if idx[(t.value, wrong.value, c, tid)].get("excess_calls_vs_oracle") is not None
               and idx[(t.value, diag.value, c, tid)].get("excess_calls_vs_oracle") is not None]
        dd = [float(idx[(t.value, wrong.value, c, tid)]["excess_calls_vs_oracle"])
              - float(idx[(t.value, diag.value, c, tid)]["excess_calls_vs_oracle"]) for (c, tid) in sub]
        ci = bootstrap_ci(dd, alpha=ALPHA, n_boot=N_BOOT, seed=BD100_SEED)
        diag_rows.append({"truth": t.value, "n_main": len(mem), "n_sub": len(sub),
                          "n_excluded": len(mem) - len(sub)})
    ok_t8 = (len(diag_rows) == len(FAIL_TYPES)
             and all(r["n_sub"] <= r["n_main"] for r in diag_rows)
             and any(r["n_sub"] < r["n_main"] for r in diag_rows)
             and all(r["n_excluded"] == r["n_main"] - r["n_sub"] for r in diag_rows))
    check("T-8 ", ok_t8,
          " %d %d n_sub ≤ n_main ****"
          " = n_main − n_sub  %s"
          % (len(diag_rows), len(FAIL_TYPES),
             all(r["n_excluded"] == r["n_main"] - r["n_sub"] for r in diag_rows)))

    bad_ci = [c for c in _CI_CALLS
              if (c["alpha"], c["n_boot"], c["seed"]) != (ALPHA, N_BOOT, BD100_SEED)]
    check("T-3 ****",
          _CI_CALLS and not bad_ci and BD100_SEED != 20260730,
          " %d (α, B, seed) = (%.2f, %d, %d) %s"
          "seed ≠  20260730 %s[LE13]③"
          % (len(_CI_CALLS), ALPHA, N_BOOT, BD100_SEED, bad_ci or "",
             BD100_SEED != 20260730))

    for line in MANDATORY_T14 + MANDATORY_T15:
        emit(line)
    _blob = "\n".join(_emitted)
    _rec = RECIPE.read_text(encoding="utf-8") if RECIPE.exists() else ""
    miss_emit = [a for a in ANCHORS_T14 if a not in _blob]
    miss_rec = [a for a in ANCHORS_T14 if a not in _rec]
    check("T-14  ∧  ∧ ",
          len(ANCHORS_T14) == 4 and not miss_emit and not miss_rec,
          " %d 4 %s %s"
          "→/ print→→"
          % (len(ANCHORS_T14), miss_emit or "", miss_rec or ""))
    check("T-15   c1 ****", set(MANDATORY_T15) <= set(_emitted),
          " %d/%d" % (len(set(MANDATORY_T15) & set(_emitted)), len(MANDATORY_T15)))

    print("\n" + "=" * 96)
    if _fails:
        print("GAP2 FAILED")
        return 1
    print("GAP2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
