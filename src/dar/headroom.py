from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from dar.ga_semantic import main_estimator

AXIS_OF = {"P1": "transient", "P3": "transient", "P2": "persistent", "P4": "persistent"}

MODE_EXCLUDE_CATS: dict[str, frozenset[str]] = {
    "P2": frozenset({"c1"}), "P4": frozenset({"c1"}),
    "P1": frozenset(), "P3": frozenset(),
}
GATE_MODES = ("P2", "P3", "P4")

EXCLUDE_DISPOSITIONS = ("exclude", "exclude_cell")


def _key(r: dict[str, Any]) -> tuple[str, str]:
    return (r["cat"], r["tid"])


def reference_tasks(rows: list[dict[str, Any]], cat: str, mode: str) -> set[tuple[str, str]]:
    axis = AXIS_OF[mode]
    p0_ach = {_key(r) for r in rows
              if r["arm"] == "p0-baseline" and r["cat"] == cat
              and main_estimator(r["verdict"])}
    ok: set[tuple[str, str]] = set()
    for r in rows:
        if r["arm"] not in ("none", "oracle") or r["cat"] != cat or r["mode"] != mode:
            continue
        if r.get("material") is not True:
            continue
        if r.get("defect") in EXCLUDE_DISPOSITIONS:
            continue
        k = _key(r)
        if k in p0_ach:
            ok.add(k)
    return ok


def paired_diffs(rows: list[dict[str, Any]], cat: str, mode: str,
                 ref: set[tuple[str, str]]) -> list[dict[str, Any]]:
    def _by(arm: str) -> dict[tuple[str, str], bool]:
        return {_key(r): main_estimator(r["verdict"])
                for r in rows if r["arm"] == arm and r["cat"] == cat and r["mode"] == mode}
    none_a, orac_a = _by("none"), _by("oracle")
    out = []
    for k in sorted(ref):
        if k in none_a and k in orac_a:
            n, o = none_a[k], orac_a[k]
            out.append({"cat": k[0], "tid": k[1], "none": n, "oracle": o,
                        "diff": int(o) - int(n)})
    return out


def bootstrap_ci(diffs: list[float], alpha: float = 0.10,
                 n_boot: int = 10000, seed: int = 20260722) -> dict[str, Any]:
    n = len(diffs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0,
                "n_pos": 0, "n_neg": 0, "n_zero": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": sum(diffs) / n, "lo": lo, "hi": hi, "n": n,
            "n_pos": sum(1 for d in diffs if d > 0),
            "n_neg": sum(1 for d in diffs if d < 0),
            "n_zero": sum(1 for d in diffs if d == 0)}


@dataclass
class ModeHeadroom:
    mode: str
    cats: list[str]
    diffs: list[dict[str, Any]] = field(default_factory=list)
    ci: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool | None:
        lo = self.ci.get("lo")
        return None if lo is None else lo > 0


def mode_headroom(rows: list[dict[str, Any]], mode: str,
                  alpha: float = 0.10, n_boot: int = 10000,
                  seed: int = 20260722) -> ModeHeadroom:
    exclude = MODE_EXCLUDE_CATS.get(mode, frozenset())
    cats = [c for c in ("c1", "c2", "c3", "c4") if c not in exclude]
    all_diffs: list[dict[str, Any]] = []
    for cat in cats:
        ref = reference_tasks(rows, cat, mode)
        all_diffs.extend(paired_diffs(rows, cat, mode, ref))
    ci = bootstrap_ci([d["diff"] for d in all_diffs], alpha, n_boot, seed)
    return ModeHeadroom(mode=mode, cats=cats, diffs=all_diffs, ci=ci)


def gate_decision(rows: list[dict[str, Any]], **kw) -> dict[str, Any]:
    per: dict[str, ModeHeadroom] = {}
    for mode in ("P1",) + GATE_MODES:
        per[mode] = mode_headroom(rows, mode, **kw)
    gate = {m: per[m] for m in GATE_MODES}
    fails = [m for m in GATE_MODES if not (per[m].passes is True)]
    n_fail = len(fails)
    return {"per_mode": per,
            "red_line": n_fail == len(GATE_MODES),
            "n_fail": n_fail, "fails": fails,
            "prompt_rule_2_triggered": n_fail >= 2}


OFFDIAG_WL = {"P1": "persistent", "P3": "persistent", "P2": "transient", "P4": "transient"}


def attribution_delta_of_delta(
    oracle_ga: dict[tuple[str, str], bool],
    wl_ga: dict[tuple[str, str], bool],
    ref: set[tuple[str, str]],
    *,
    exclude_cats: frozenset[str] = frozenset(),
    n_boot: int = 10000,
) -> dict[str, Any]:
    diffs = [int(oracle_ga[k]) - int(wl_ga[k])
             for k in sorted(ref)
             if k[0] not in exclude_cats and k in oracle_ga and k in wl_ga]
    return {"diffs": diffs, **bootstrap_ci(diffs, n_boot=n_boot)}


def position_index(staging_ids: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    pos: dict[tuple[str, str], int] = {}
    p = 0
    for cat in sorted(staging_ids):
        for tid in sorted(staging_ids[cat]):
            pos[(cat, tid)] = p
            p += 1
    return pos


def position_drift_regression(
    pairs: list[dict[str, Any]],
    pos: dict[tuple[str, str], int],
    *,
    warmup_cat: str = "c1",
) -> dict[str, Any]:
    def _ols(sub: list[dict[str, Any]]) -> dict[str, Any]:
        if len(sub) < 3:
            return {"n": len(sub), "slope": None, "se": None, "p": None, "significant": None}
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for d in sub:
            by_cat.setdefault(d["cat"], []).append(d)
        xs, ys = [], []
        for cat, ds in by_cat.items():
            mx = sum(pos[(d["cat"], d["tid"])] for d in ds) / len(ds)
            my = sum(d["diff"] for d in ds) / len(ds)
            for d in ds:
                xs.append(pos[(d["cat"], d["tid"])] - mx)
                ys.append(d["diff"] - my)
        sxx = sum(x * x for x in xs)
        if sxx == 0:
            return {"n": len(sub), "slope": None, "se": None, "p": None, "significant": None}
        slope = sum(x * y for x, y in zip(xs, ys)) / sxx
        dof = len(xs) - len(by_cat) - 1
        if dof <= 0:
            return {"n": len(sub), "slope": slope, "se": None, "p": None, "significant": None}
        rss = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
        se = (rss / dof / sxx) ** 0.5
        if se == 0:
            return {"n": len(sub), "slope": slope, "se": 0.0, "p": 0.0 if slope else 1.0,
                    "significant": bool(slope)}
        z = slope / se
        import math
        pval = math.erfc(abs(z) / math.sqrt(2))
        return {"n": len(sub), "slope": slope, "se": se, "p": pval,
                "significant": pval < 0.10}

    full = _ols(pairs)
    no_warm = _ols([d for d in pairs if d["cat"] != warmup_cat])
    return {"full": full, "excl_warmup": no_warm, "warmup_cat": warmup_cat, "alpha": 0.10}
