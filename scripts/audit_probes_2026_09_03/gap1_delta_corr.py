#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

if not __debug__:
    raise SystemExit("")

sys.path.insert(0, "src")

from dar import headroom
from dar.labels import TRUE_TYPE_TO_LABEL, Label, TrueType

_CI_CALLS: list[dict] = []
_raw_bootstrap_ci = headroom.bootstrap_ci


def bootstrap_ci(diffs, *, alpha, n_boot, seed):
    _CI_CALLS.append({"alpha": alpha, "n_boot": n_boot, "seed": seed})
    return _raw_bootstrap_ci(diffs, alpha=alpha, n_boot=n_boot, seed=seed)

ANALYSIS = Path("data/b5-11090655/analysis/b5_analysis.json")
B7_DOC = Path("docs/b7_findings.md")
G7_PROBE = Path("scripts/audit_probes_2026_08_28/pstar_degeneracy_map.py")

BD100_SEED = 20260903
N_BOOT = 10000
ALPHA = 0.10
EXCLUDED_CAT = "c1"
FAIL_TYPES = (TrueType.P1, TrueType.P2, TrueType.P3, TrueType.P4)
EXPECTED_FAIL_TYPES = tuple(t for t in TrueType if t is not TrueType.NP)
IDENTITY_RTOL = 1e-12

MANDATORY_T10 = (
    "① ****switch_path  ⇒ "
    "⇒ ****",
    "② ****——`b7_findings.md` T4 "
    " `pstar_degeneracy_map.py` G7  ⇒ ****",
    "③ ** = CI******——"
    " retry ",
    "④ ** q** `headroom.bootstrap_ci` T4 ——"
    "[BD-102]  ⑭-G V3A-1 ****",
)
MANDATORY_T11 = (
    "******** G8b"
    " CI ",
    "P1P3 ********"
    "——[BD-29] C-2 ",
)


ANCHORS_T10: tuple[str, ...] = ('', '', '', ' q')
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


def t4_from_doc(doc: str) -> dict[str, float]:
    m = re.search(r"<!-- b7t:T4 -->(.*?)<!-- b7t:/T4 -->", doc, re.S)
    out: dict[str, float] = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cells) == 5:
            try:
                out[cells[0]] = float(cells[1])
            except ValueError:
                continue
    return out


def g7_from_probe() -> dict[str, float]:
    r = subprocess.run(["uv", "run", "python", str(G7_PROBE)],
                       capture_output=True, text=True)
    out: dict[str, float] = {}
    for line in r.stdout.splitlines():
        m = re.search(r"Δ_corr\((P[1-4])\)  c1 = ([+-][0-9.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main() -> int:
    print("=" * 96)
    print("=" * 96)
    for q in (ANALYSIS, B7_DOC, G7_PROBE):
        if not q.exists():
            print("\nGAP1 FAILED")
            return 1
    an = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    units = {u["unit"]: u for u in an["paired"]}
    t4 = t4_from_doc(B7_DOC.read_text(encoding="utf-8"))
    g7 = g7_from_probe()

    check("T-3 ",
          BD100_SEED == 20260903 and N_BOOT == 10000 and abs(ALPHA - 0.10) < 1e-15
          and BD100_SEED != headroom.bootstrap_ci.__defaults__[2] and BD100_SEED != 20260730,
          "seed=%d≠ `bootstrap_ci`  %d≠ 20260730B=%dα=%.2f"
          % (BD100_SEED, headroom.bootstrap_ci.__defaults__[2], N_BOOT, ALPHA))

    n_emitted = 0
    id_fail, mem_fail, arm_fail, doc_fail, g7_fail, pair_fail = [], [], [], [], [], []
    for t in FAIL_TYPES:
        diag, wrong = TRUE_TYPE_TO_LABEL[t], wrong_label(t)
        ru, du = units.get("r(%s)" % t.value), units.get("d(%s,%s)" % (t.value, wrong.value))
        if ru is None or du is None:
            id_fail.append("")
            continue
        if not (ru["oracle_arm"].endswith(diag.value) and ru["none_arm"] == "fa/none"
                and du["oracle_arm"] == "fa/none" and du["none_arm"].endswith(wrong.value)):
            arm_fail.append("%s: r=(%s,%s) d=(%s,%s)" % (t.value, ru["none_arm"], ru["oracle_arm"],
                                                        du["none_arm"], du["oracle_arm"]))
        rd = {(x["cat"], x["tid"]): x for x in ru["diffs"] if x["cat"] != EXCLUDED_CAT}
        dd = {(x["cat"], x["tid"]): x for x in du["diffs"] if x["cat"] != EXCLUDED_CAT}
        if set(rd) != set(dd):
            mem_fail.append("%s: |r∖d|=%d |d∖r|=%d" % (t.value, len(set(rd) - set(dd)),
                                                       len(set(dd) - set(rd))))
            continue
        members = sorted(rd)
        np_mismatch = sum(1 for k in members if float(rd[k]["none"]) != float(dd[k]["oracle"]))
        if np_mismatch:
            arm_fail.append("")
        diffs = [float(rd[k]["oracle"]) - float(dd[k]["none"]) for k in members]
        ci = bootstrap_ci(diffs, alpha=ALPHA, n_boot=N_BOOT, seed=BD100_SEED)
        vals = set(diffs)
        if not vals <= {-1.0, 0.0, 1.0}:
            pair_fail.append("")
        mis = [i for i, k in enumerate(members)
               if diffs[i] != float(rd[k]["oracle"]) - float(dd[k]["none"])]
        if mis or len(diffs) != len(members):
            pair_fail.append("")
        if ci["n_pos"] + ci["n_neg"] + ci["n_zero"] != ci["n"]:
            pair_fail.append("")
        rm = sum(float(rd[k]["diff"]) for k in members) / len(members)
        dm = sum(float(dd[k]["diff"]) for k in members) / len(members)
        if not math.isclose(ci["mean"], rm + dm, rel_tol=IDENTITY_RTOL, abs_tol=IDENTITY_RTOL):
            id_fail.append("%s: %.17g ≠ %.17g" % (t.value, ci["mean"], rm + dm))
        dr, dd4 = t4.get("r(%s)" % t.value), t4.get("d(%s,%s)" % (t.value, wrong.value))
        if dr is None or dd4 is None or abs(ci["mean"] - (dr + dd4)) > 5e-4:
            doc_fail.append("")
        if t.value not in g7 or abs(ci["mean"] - g7[t.value]) > 5e-9:
            g7_fail.append("")
        n_emitted += 1

    check("T-1a Δ_corr == r(t) + d(t,ℓ_wrong)", not id_fail,
          "rel_tol=%g%s⚠️ ****——**CI  CI **"
          % (IDENTITY_RTOL, id_fail or ""))
    check("T-1b · T4 `<!-- b7t:T4 -->`", not doc_fail,
          "doc  4dp%s" % (doc_fail or ""))
    check("T-1c ·G7 `pstar_degeneracy_map.py`", not g7_fail,
          "%s ⇒ ** CI**" % (g7_fail or ""))
    check("T-2  n[LM24]", not mem_fail, mem_fail or "")
    check("T-3b  ⊆ {-1,0,+1} ∧ ****", not pair_fail,
          "%s****—— ⇒ T-1a/b/c "
          "" % (pair_fail or ""))
    check("T-2b   np ", not arm_fail, arm_fail or "")
    bad_ci = [c for c in _CI_CALLS
              if (c["alpha"], c["n_boot"], c["seed"]) != (ALPHA, N_BOOT, BD100_SEED)]
    check("T-3c ****[LE13]②",
          bool(_CI_CALLS) and not bad_ci,
          " %d (α, B, seed) = (%.2f, %d, %d) %s"
          % (len(_CI_CALLS), ALPHA, N_BOOT, BD100_SEED, bad_ci or ""))
    check("T-4T-11 ",
          tuple(FAIL_TYPES) == EXPECTED_FAIL_TYPES and n_emitted == len(EXPECTED_FAIL_TYPES),
          " %d %d** `TrueType` ** FAIL_TYPES "
          "FAIL_TYPES  %s G8b"
          % (n_emitted, len(EXPECTED_FAIL_TYPES), tuple(FAIL_TYPES) == EXPECTED_FAIL_TYPES))

    for line in MANDATORY_T10 + MANDATORY_T11:
        emit(line)
    _blob = "\n".join(_emitted)
    _rec = RECIPE.read_text(encoding="utf-8") if RECIPE.exists() else ""
    miss_emit = [a for a in ANCHORS_T10 if a not in _blob]
    miss_rec = [a for a in ANCHORS_T10 if a not in _rec]
    check("T-10  ∧  ∧ ",
          len(ANCHORS_T10) == 4 and not miss_emit and not miss_rec,
          " %d 4 %s %s"
          "→/ print→→"
          % (len(ANCHORS_T10), miss_emit or "", miss_rec or ""))
    check("T-11 ****", set(MANDATORY_T11) <= set(_emitted),
          " %d/%d" % (len(set(MANDATORY_T11) & set(_emitted)), len(MANDATORY_T11)))

    print("\n" + "=" * 96)
    if _fails:
        print("GAP1 FAILED")
        return 1
    print("GAP1 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
