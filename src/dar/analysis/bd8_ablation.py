from __future__ import annotations

from typing import Any, Iterable

from dar import accounting as A
from dar.ga_semantic import main_estimator
from dar.headroom import bootstrap_ci

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260730
ALPHA = 0.10

ALL_CATS: tuple[str, ...] = ("c1", "c2", "c3", "c4")
HEADLINE_CATS: tuple[str, ...] = ("c2", "c3", "c4")
CROSSBATCH_CATS: tuple[str, ...] = ALL_CATS
C1_ONLY: tuple[str, ...] = ("c1",)

DELTA1 = ("bd8:P1xF-transient.I-persistent", "bd8:P1xF-persistent.I-transient")
DELTA2 = ("bd8:P2xF-transient.I-persistent", "bd8:P2xF-persistent.I-transient")

_TRUTH_CELLS = {"P1": DELTA1, "P2": DELTA2}


class Bd8AnalysisError(ValueError):
    pass


def support_set(layer_a_rows: Iterable[dict[str, Any]], *, truth: str,
                ga_p0_pool: Iterable[tuple[str, str]],
                cats: Iterable[str]) -> set[tuple[str, str]]:
    cats = tuple(cats)
    rows = [r for r in layer_a_rows if r.get("mode") == truth and r.get("cat") in cats]
    if not rows:
        raise Bd8AnalysisError(
            "")
    keys = {(r["cat"], r["tid"]) for r in rows}
    mat_by: dict[tuple[str, str], Any] = {}
    dis_by: dict[tuple[str, str], Any] = {}
    for r in rows:
        k = (r["cat"], r["tid"])
        for by, field in ((mat_by, "material"), (dis_by, "defect")):
            if k in by and by[k] != r[field]:
                raise Bd8AnalysisError(
                    "")
            by[k] = r[field]
    return A.cell_level_sieve(keys, ga_p0_pool=ga_p0_pool,
                              material_by_task=mat_by, disposition_by_task=dis_by)


def support_sets(layer_a_rows: list[dict[str, Any]], *,
                 ga_p0_pool: Iterable[tuple[str, str]]) -> dict[str, dict[str, set]]:
    pool = set(ga_p0_pool)
    out: dict[str, dict[str, set]] = {}
    for name, cats in (("headline", HEADLINE_CATS), ("crossbatch", CROSSBATCH_CATS),
                       ("c1_only", C1_ONLY)):
        out[name] = {t: support_set(layer_a_rows, truth=t, ga_p0_pool=pool, cats=cats)
                     for t in sorted(_TRUTH_CELLS)}
    return out


def cell_mu(achieved: dict[tuple[str, str, str], bool], cell_id: str,
            support: set[tuple[str, str]]) -> dict[str, Any]:
    vals: list[bool] = []
    missing: list[tuple[str, str]] = []
    for cat, tid in sorted(support):
        k = (cell_id, cat, tid)
        if k not in achieved:
            missing.append((cat, tid))
        else:
            vals.append(achieved[k])
    if missing:
        raise Bd8AnalysisError(
            "")
    if not vals:
        raise Bd8AnalysisError("")
    return {"cell_id": cell_id, "n": len(vals),
            "mu": sum(1 for v in vals if v) / len(vals)}


def paired_diffs(achieved: dict[tuple[str, str, str], bool], pair: tuple[str, str],
                 support: set[tuple[str, str]]) -> list[dict[str, Any]]:
    a, b = pair
    out: list[dict[str, Any]] = []
    for cat, tid in sorted(support):
        ka, kb = (a, cat, tid), (b, cat, tid)
        if ka not in achieved or kb not in achieved:
            raise Bd8AnalysisError(
                "")
        out.append({"cat": cat, "tid": tid, "a": achieved[ka], "b": achieved[kb],
                    "diff": float(achieved[ka]) - float(achieved[kb])})
    if not out:
        raise Bd8AnalysisError("")
    return out


def delta_ci(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    return bootstrap_ci([d["diff"] for d in diffs], alpha=ALPHA,
                        n_boot=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)


VERDICT_CONSISTENT = " instruction "
VERDICT_INCONSISTENT = " framing "
VERDICT_INDISTINGUISHABLE = " n "
VERDICT_UNIDENTIFIED = ("⛔ ——"
                        " =  γ")


def _excludes_zero(ci: dict[str, Any]) -> bool:
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return False
    return lo > 0 or hi < 0


def discriminant(ci1: dict[str, Any], ci2: dict[str, Any]) -> dict[str, Any]:
    e1, e2 = _excludes_zero(ci1), _excludes_zero(ci2)
    if not (e1 and e2):
        return {"verdict": VERDICT_INDISTINGUISHABLE, "code": "indistinguishable",
                "delta1_excludes_zero": e1, "delta2_excludes_zero": e2,
                "why": " 90% CI  0 ⇒ "}
    s1 = 1 if ci1["lo"] > 0 else -1
    s2 = 1 if ci2["lo"] > 0 else -1
    if s1 == s2:
        return {"verdict": VERDICT_UNIDENTIFIED, "code": "unidentified",
                "delta1_excludes_zero": True, "delta2_excludes_zero": True,
                "why": "§7.1 "}
    if s1 < 0 and s2 > 0:
        return {"verdict": VERDICT_CONSISTENT, "code": "consistent",
                "delta1_excludes_zero": True, "delta2_excludes_zero": True,
                "why": "Δ₁ < 0  Δ₂ > 0§7.1 "}
    return {"verdict": VERDICT_INCONSISTENT, "code": "inconsistent",
            "delta1_excludes_zero": True, "delta2_excludes_zero": True,
            "why": "Δ₁ > 0  Δ₂ < 0§7.1 "}


def run_discriminant(achieved: dict[tuple[str, str, str], bool],
                     supports: dict[str, set[tuple[str, str]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"support_label": "headline c1",
                           "cats": list(HEADLINE_CATS),
                           "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED,
                                         "alpha": ALPHA, "unit": ""}}
    for name, truth, pair in (("delta1", "P1", DELTA1), ("delta2", "P2", DELTA2)):
        sup = supports[truth]
        diffs = paired_diffs(achieved, pair, sup)
        out[name] = {"pair": list(pair), "truth": truth, "n": len(diffs),
                     "mu_a": cell_mu(achieved, pair[0], sup)["mu"],
                     "mu_b": cell_mu(achieved, pair[1], sup)["mu"],
                     "ci": delta_ci(diffs)}
    out["discriminant"] = discriminant(out["delta1"]["ci"], out["delta2"]["ci"])
    return out
