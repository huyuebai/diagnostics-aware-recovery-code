#!/usr/bin/env python3
from __future__ import annotations

import atexit
import json
import sys
from pathlib import Path

PASSED = False

if not __debug__:
    raise SystemExit("")

LAB = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(LAB / _p) not in sys.path:
        sys.path.insert(0, str(LAB / _p))

from dar.analysis import pstar as PS
from dar.analysis import synthesis as S
from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

B7 = LAB / "data" / "b7-20260824-d"
B6M = LAB / "data" / "b6-20260814" / "matrix"
B5A = LAB / "data" / "b5-11090655" / "analysis"

SCOPE_SENTENCE = (
    " = P1–P4× **NP **——"
    "NP×transient  P1 victim NP×persistent  P2 victim "
    "U6 first-touch ⇒ [BD-62]② V1-4"
)

THESIS_FRAGMENTS = (
    "c1  headline  persistent ",
    "n = 14",
)
THESIS_SENTENCE = "".join(THESIS_FRAGMENTS) + ""

FORBIDDEN_2 = (
    "② ——** vs np ** vs "
    "**** §2 −"
    " CI ⇒ ****"
)

def _forbidden2_hits() -> int:
    import re as _re
    import unicodedata as _ud
    _norm = lambda t: _re.sub(r"\s+", "", _ud.normalize("NFKC", t))
    doc = (LAB / "docs" / "b5_findings.md").read_text(encoding="utf-8")
    return _norm(doc).count(_norm(FORBIDDEN_2))


A8_SENTENCE = (
    " persistent  0 victim "
    "`fa/none`  `max_consecutive_same_tool_errors` ∈ {0,1} "
)
A8_FACE = ("rule", "head_to_head")

STATE_W_ZERO = "w_wrong = 0"
STATE_D_ZERO = "d̂ = 0CI  = "
STATE_D_NEG = "d̂ < 0"
STATE_D_POS = "d̂ > 0 ∧ w > 0"
STATES = (STATE_W_ZERO, STATE_D_ZERO, STATE_D_NEG, STATE_D_POS)


def _sentinel() -> None:
    print("GREENLIGHT PASS" if PASSED else "GREENLIGHT FAILED (no verdict)")


atexit.register(_sentinel)


def _state_of(w_wrong: float, d_wrong: float) -> str:
    if d_wrong == 0.0:
        return STATE_D_ZERO
    if w_wrong == 0.0:
        return STATE_W_ZERO
    return STATE_D_NEG if d_wrong < 0.0 else STATE_D_POS


def _wrong_label(t: TrueType) -> Label:
    star = TRUE_TYPE_TO_LABEL[t]
    cand = [lab for lab in S.LABELS if lab is not star and lab is not Label.NP]
    assert len(cand) == 1, ""
    return cand[0]


def _rownorm_path(diagnoser: str) -> Path:
    return B6M / f"{diagnoser}.rownorm.json"


def _T_from_rownorm(path: Path) -> S.TransitionMatrix:
    m = json.loads(path.read_text(encoding="utf-8"))["matrix"]
    return S.TransitionMatrix.from_config(
        {"rows": [[m[r.value][c.value] for c in S.LABELS] for r in S.LABELS]})


def _fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _ci_str(ci: dict) -> str:
    if ci.get("mean") is None:
        return "n=0"
    return (f"{ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
            f"n={ci['n']} (+{ci['n_pos']}/−{ci['n_neg']}/0:{ci['n_zero']})")


def _crosses_zero(ci: dict) -> bool:
    return ci["lo"] <= 0.0 <= ci["hi"]


def main() -> int:
    global PASSED

    syn = json.loads((B7 / "synthesis.json").read_text(encoding="utf-8"))
    ps = json.loads((B7 / "pstar.json").read_text(encoding="utf-8"))
    an = json.loads((B5A / "b5_analysis.json").read_text(encoding="utf-8"))

    files = sorted(syn["empirical"])
    fails = [t for t in TrueType if t is not TrueType.NP]
    assert files, ""
    assert fails, ""
    wrong = {t: _wrong_label(t) for t in fails}

    d_prod = ps["components_ex_c1"]["d"]
    r_prod = ps["components_ex_c1"]["r"]
    ci_ex = ps["components_ex_c1"]["unit_ci"]
    c1_ci = syn["c1_disclosure_parallel"]["unit_ci"]
    c1_tag = ps["c1_disclosure_parallel"]["tag"]
    mu = syn["mu_headline_ex_c1"]
    paired = {u["unit"]: u for u in an["paired"]}
    assert d_prod and r_prod and ci_ex and c1_ci and mu and paired, \
        ""

    b7doc = (LAB / "docs" / "b7_findings.md").read_text(encoding="utf-8")
    assert b7doc, ""
    miss = [f for f in THESIS_FRAGMENTS if f not in b7doc]
    assert not miss, (
        "")
    assert "b7t:T9" in b7doc, ""

    print("=" * 100)
    print("=" * 100)
    print(S.BD50_T_ROW_WARNING)
    print()

    wA: dict[tuple[str, TrueType], float] = {}
    wB: dict[tuple[str, TrueType], float] = {}
    dA: dict[TrueType, float] = {}
    dB: dict[TrueType, float] = {}
    faces: list[str] = []

    for t in fails:
        lw = wrong[t]
        assert set(d_prod[t.value]) == {Label.NP.value, lw.value}, (
            "")
        dA[t] = d_prod[t.value][lw.value]
        u = paired[f"d({t.value},{lw.value})"]
        ex = [x["diff"] for x in u["diffs"] if x["cat"] != "c1"]
        assert ex, ""
        dB[t] = sum(ex) / len(ex)

    for f in files:
        rn = _rownorm_path(f)
        assert rn.is_file(), ""
        face = json.loads(rn.read_text(encoding="utf-8"))["face"]
        faces.append(face)
        T = _T_from_rownorm(rn)
        for t in fails:
            lw = wrong[t]
            wA[(f, t)] = syn["empirical"][f]["p_w_by_truth"][t.value]["w"][lw.value]
            _, wb = S.confusion_weights(T, t)
            wB[(f, t)] = wb[lw]

    stateA = {(f, t): _state_of(wA[(f, t)], dA[t]) for f in files for t in fails}
    stateB = {(f, t): _state_of(wB[(f, t)], dB[t]) for f in files for t in fails}
    assert stateA and stateB, ""
    diverge = sorted(k for k in stateA if stateA[k] != stateB[k])
    assert not diverge, (
        "")
    both = sorted((f, t.value) for f in files for t in fails
                  if wA[(f, t)] == 0.0 and dA[t] == 0.0)
    assert not both, (
        "")

    n_cells = len(files) * len(fails)
    assert len(stateA) == n_cells, ""
    assert set(stateA.values()) <= set(STATES), \
        ""
    covered = set(stateA.values())
    assert len(covered) == len(STATES), (
        "")
    assert not any(t is TrueType.NP for _, t in stateA), ""
    print(f"     {SCOPE_SENTENCE}")

    assert faces, ""
    bad_face = sorted({x for x in faces if x != A8_FACE[1]})
    assert not bad_face, (
        "")
    worst = 0.0
    for f in files:
        T = _T_from_rownorm(_rownorm_path(f))
        for t in fails:
            mu_row = {lab: mu[t.value][lab.value] for lab in S.LABELS}
            rhs = PS.d_bar_of(mu_row, t, T)
            lhs = wA[(f, t)] * dA[t]
            worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-12, (
        "")

    p4 = [t for t in fails if dA[t] == 0.0]
    assert p4, ""
    for t in p4:
        zero_w = sorted(f for f in files if wA[(f, t)] == 0.0)
        assert not zero_w, (
            "")
    blew_up = False
    try:
        _ = (wA[(files[0], p4[0])] * dA[p4[0]]) / dA[p4[0]]
    except ZeroDivisionError:
        blew_up = True
    assert blew_up, (
        "")

    in_open = sorted((f, t.value) for f in files for t in fails
                     if (v := ps["per_file"][f][t.value]["value"]) is not None
                     and 0.0 < v < 1.0)
    assert in_open, ""
    open_types = {tv for _, tv in in_open}
    assert len(open_types) == 1, \
        ""
    tv = open_types.pop()
    assert len(in_open) == len(files), (
        "")
    lw_open = wrong[TrueType(tv)]
    ci_open = ci_ex[f"d({tv},{lw_open.value})"]
    assert _crosses_zero(ci_open), (
        "")

    axis: dict[Label, list[tuple[str, TrueType]]] = {}
    for t in fails:
        axis.setdefault(TRUE_TYPE_TO_LABEL[t], []).append(("", t))
    pairs = [(a, b) for lab, xs in axis.items() if len(xs) == 2
             for (_, a), (_, b) in [(xs[0], xs[1])]]
    assert pairs, ""
    for f in files:
        for a, b in pairs:
            assert wA[(f, a)] == wA[(f, b)], (
                "")
    n_indep = len(files) * len(pairs)

    wz = sorted((f, t) for f in files for t in fails if stateA[(f, t)] == STATE_W_ZERO)
    assert wz, ""
    reasons: dict[str, list[str]] = {}
    for f, t in wz:
        reasons.setdefault(str(ps["per_file"][f][t.value]["reason"]), []) \
            .append(f"{f}/{t.value}")
    assert len(reasons) > 1, (
        "")
    for rsn, cells in sorted(reasons.items()):
        pass

    print()
    hdr = (f"{'':<6}{'':<6}{'ℓ_wrong':<12}{'w_wrong':>10}  "
           f"{'d(t,ℓ_wrong)  [CI] n (+/−/0)':<46}{'':<34} p* T2 ")
    print(hdr)
    print("─" * len(hdr))
    for f in files:
        for t in fails:
            lw = wrong[t]
            ci = ci_ex[f"d({t.value},{lw.value})"]
            cell = ps["per_file"][f][t.value]
            mark = (f"{cell['value']:.6f}" if cell["value"] is not None else "None")
            mark += f" / {cell['reason']}"
            print(f"{f:<6}{t.value:<6}{lw.value:<12}{wA[(f, t)]:>10.4f}  "
                  f"{_ci_str(ci):<46}{stateA[(f, t)]:<34}{mark}")

    print()
    axis_types = [t for t in fails if TRUE_TYPE_TO_LABEL[t] is TRUE_TYPE_TO_LABEL[fails[0]]]
    assert len(axis_types) >= 2, ""
    hdr2 = f"{'':<22}{'':<10}{' [CI] n (+/−/0)'}"
    print(hdr2)
    print("─" * 78)
    eight: list[tuple[str, str]] = []
    for t in axis_types:
        lw = wrong[t]
        for qty in (f"d({t.value},{lw.value})", f"r({t.value})"):
            for tag, src in ((" c1", ci_ex), ("c1", c1_ci)):
                assert qty in src, ""
                print(f"{qty:<22}{tag:<10}{_ci_str(src[qty])}")
                eight.append((qty, tag))
    assert len(eight) == len(axis_types) * 2 * 2, (
        "")

    print()
    flip, degen = [], []
    for t in axis_types:
        key = f"d({t.value},{wrong[t].value})"
        a, b = ci_ex[key], c1_ci[key]
        if b["mean"] is None:
            continue
        if (a["mean"] < 0 < b["mean"] or b["mean"] < 0 < a["mean"]) \
                and not _crosses_zero(a) and not _crosses_zero(b):
            flip.append(t.value)
        if b["mean"] == 0.0 and b["lo"] == 0.0 and b["hi"] == 0.0 \
                and b["n_zero"] == b["n"]:
            degen.append((t.value, b["n"]))
    assert flip, (
        "")
    assert degen, (
        "")

    print()
    c1p = ps["c1_disclosure_parallel"]["per_file"]
    shown = [(f, t) for f in files for t in fails
             if (v := c1p[f][t.value]["value"]) is not None and 0.0 < v < 1.0]
    assert shown, ""
    for f, t in shown:
        rk = f"r({t.value})"
        assert rk in c1_ci, ""
        rci = c1_ci[rk]
        assert _crosses_zero(rci), (
            "")

    print()
    n7 = 0
    for t in fails:
        lw = wrong[t]
        line = f"  Δ_corr({t.value})  c1 = {r_prod[t.value] + dA[t]:+.10f}"
        rk, dk = f"r({t.value})", f"d({t.value},{lw.value})"
        if c1_ci.get(rk, {}).get("mean") is not None \
                and c1_ci.get(dk, {}).get("mean") is not None:
            line += f"   c1 = {c1_ci[rk]['mean'] + c1_ci[dk]['mean']:+.10f}"
        else:
            line += "   c1 = —"
        print(line)
        n7 += 1
    assert n7 == len(fails), ""
    _n_f2 = _forbidden2_hits()
    assert _n_f2 == 1, (
        "")
    print(f"     {FORBIDDEN_2}")

    probe_cell = (files[0], fails[0])
    poisons = [
        ("P4  d  0.0  1e-4", wA[(files[0], fails[-1])], 1e-4, None,
         " `d̂ = 0` "),
        ("w  0", 0.0, dA[fails[1]], STATE_W_ZERO,
         "`d̂ > 0 ∧ w > 0`  `w_wrong = 0`"),
        ("d ", 0.5, abs(dA[fails[0]]), STATE_D_POS,
         "`d̂ < 0`  `d̂ > 0 ∧ w > 0`"),
        ("d  0w ", 0.5, 0.0, STATE_D_ZERO,
         " `d̂ = 0`"),
        ("w  d ", 0.0, 0.0, STATE_D_ZERO,
         " d**** if  `w_wrong = 0`"),
    ]
    assert poisons, ""
    base_w, base_d = wA[probe_cell], dA[probe_cell[1]]
    base_state = _state_of(base_w, base_d)
    n11 = 0
    for lbl, pw, pd, want, why in poisons:
        assert (pw, pd) != (base_w, base_d), \
            ""
        got = _state_of(pw, pd)
        if want is None:
            assert got != STATE_D_ZERO, (
                "")
        else:
            assert got == want, (
                "")
        n11 += 1
    assert _state_of(base_w, base_d) == base_state, ""
    print()

    print()
    print(S.BD50_ML_NP_ROW_WARNING)
    PASSED = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
