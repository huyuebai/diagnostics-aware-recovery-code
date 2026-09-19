from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable

from dar.ga_semantic import VERDICT_VOCAB, SemanticItem, apply_semantic_verdicts, main_estimator
from dar.headroom import AXIS_OF, MODE_EXCLUDE_CATS, bootstrap_ci, paired_diffs

B5_BOOTSTRAP_SEED = 20260730

REPLICATE_FIELD = "replicate_rep"

MIN_FLOOR_REP_N = 5

CROSS_CELL_CAVEAT = (" NP×np +"
                     "b5_preregistration §5-4/§7-g⑵")

_RUN_VOCAB = ("achieved", "not_achieved", "undetermined")
_RUN_ALTS = {v: tuple(x for x in _RUN_VOCAB if x != v) for v in _RUN_VOCAB}

_ITEM_VOCAB = ("EQUIVALENT", "NOT_EQUIVALENT", "CANNOT_DETERMINE")
_ITEM_ALTS = {v: tuple(x for x in _ITEM_VOCAB if x != v) for v in _ITEM_VOCAB}
assert set(_ITEM_VOCAB) == set(VERDICT_VOCAB), ""


def _boot_means(diffs: list[float], n_boot: int, seed: int) -> list[float]:
    n = len(diffs)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    return means


def _as_number_diff(x: Any) -> float:
    if isinstance(x, bool):
        raise ValueError("")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict) and "diff" in x:
        return _as_number_diff(x["diff"])
    raise ValueError("")


def _arm_value(rec: dict[str, Any], key: str) -> bool | str:
    if key not in rec:
        raise ValueError("")
    v = rec[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        main_estimator(v)
        return v
    raise ValueError("")


def _est(v: bool | str) -> bool:
    return main_estimator(v) if isinstance(v, str) else v


def _flip_token(v: bool | str, q: float, rng: random.Random,
                alts: dict[str, tuple[str, ...]]) -> bool | str:
    if rng.random() >= q:
        return v
    if isinstance(v, bool):
        return not v
    return rng.choice(alts[v])


def _arm_form(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, dict):
        return "item"
    if isinstance(v, str):
        return "token"
    raise ValueError("")


def _arm_form_summary(forms: set[str]) -> str | None:
    if not forms:
        return None
    if len(forms) == 1:
        return next(iter(forms))
    return "mixed(" + "+".join(sorted(forms)) + ")"


def _flip_arm(v: bool | str, q: float, rng: random.Random) -> bool | str:
    return _flip_token(v, q, rng, _RUN_ALTS)


def two_layer_ci(diffs_by_task: Iterable[dict[str, Any]], q: float,
                 B: int = 10000, *, seed: int = B5_BOOTSTRAP_SEED,
                 inner_levels: tuple[float, float] = (0.05, 0.95)) -> dict[str, Any]:
    if not (isinstance(q, (int, float)) and not isinstance(q, bool) and 0.0 <= q <= 1.0):
        raise ValueError("")
    lo_lv, hi_lv = inner_levels
    if not (0.0 <= lo_lv < hi_lv <= 1.0):
        raise ValueError("")
    if isinstance(diffs_by_task, (dict, str)):
        raise ValueError("")

    arms: list[tuple[bool | str, bool | str]] = []
    base: list[int] = []
    forms: set[str] = set()
    for rec in diffs_by_task:
        nv, ov = _arm_value(rec, "none"), _arm_value(rec, "oracle")
        d = int(_est(ov)) - int(_est(nv))
        if "diff" in rec and rec["diff"] != d:
            raise ValueError("")
        arms.append((nv, ov))
        base.append(d)
        forms.update((_arm_form(nv), _arm_form(ov)))

    n = len(base)
    meta = {"q": float(q), "n_boot": B, "inner_levels": (lo_lv, hi_lv), "seed": seed,
            "inner_unit": "run_verdict(approx)", "arm_form": _arm_form_summary(forms)}
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0,
                "n_pos": 0, "n_neg": 0, "n_zero": 0, **meta}

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(B):
        if q > 0:
            world = []
            for nv, ov in arms:
                fn = _flip_arm(nv, q, rng)
                fo = _flip_arm(ov, q, rng)
                world.append(int(_est(fo)) - int(_est(fn)))
        else:
            world = base
        s = 0.0
        for _ in range(n):
            s += world[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(lo_lv * B)]
    hi = means[min(B - 1, int(hi_lv * B))]
    return {"mean": sum(base) / n, "lo": lo, "hi": hi, "n": n,
            "n_pos": sum(1 for d in base if d > 0),
            "n_neg": sum(1 for d in base if d < 0),
            "n_zero": sum(1 for d in base if d == 0), **meta}


def run_verdict_from_item_map(item_map: dict[tuple[str, int], str]) -> str:
    if not item_map:
        raise ValueError("")
    items_by_path: dict[str, list[SemanticItem]] = {}
    for pid, idx in sorted(item_map):
        items_by_path.setdefault(pid, []).append(
            SemanticItem(pid, idx, -1, "", "", "", None, None))
    return apply_semantic_verdicts(items_by_path, dict(item_map))["verdict"]


def paired_item_maps(rows: list[dict[str, Any]],
                     verdicts_by_run: dict[str, dict[tuple, str]],
                     cat: str, mode: str, ref: set[tuple[str, str]],
                     ) -> list[dict[str, Any]]:
    def _by(arm: str) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            if r["arm"] != arm or r["cat"] != cat or r["mode"] != mode:
                continue
            k = (r["cat"], r["tid"])
            if k in out:
                raise ValueError("")
            out[k] = r
        return out

    def _arm_val(r: dict[str, Any]) -> dict | str:
        rid = r.get("run_id")
        if not rid:
            raise ValueError("")
        im = verdicts_by_run.get(rid)
        if not im:
            return r["verdict"]
        recomputed = run_verdict_from_item_map(im)
        if recomputed != r["verdict"]:
            raise ValueError(
                "")
        return dict(im)

    none_a, orac_a = _by("none"), _by("oracle")
    out: list[dict[str, Any]] = []
    for k in sorted(ref):
        if k not in none_a or k not in orac_a:
            continue
        nv, ov = _arm_val(none_a[k]), _arm_val(orac_a[k])
        n_est = _est(run_verdict_from_item_map(nv) if isinstance(nv, dict) else nv)
        o_est = _est(run_verdict_from_item_map(ov) if isinstance(ov, dict) else ov)
        out.append({"cat": k[0], "tid": k[1], "none": nv, "oracle": ov,
                    "diff": int(o_est) - int(n_est),
                    "none_form": _arm_form(nv), "oracle_form": _arm_form(ov)})
    return out


def two_layer_ci_items(items_by_task: Iterable[dict[str, Any]], q: float,
                       B: int = 10000, *, seed: int = B5_BOOTSTRAP_SEED,
                       inner_levels: tuple[float, float] = (0.05, 0.95),
                       allow_layer_a_only: bool = False,
                       run_verdict_fn: Callable[[dict[tuple[str, int], str]], str]
                       = run_verdict_from_item_map) -> dict[str, Any]:
    if not (isinstance(q, (int, float)) and not isinstance(q, bool) and 0.0 <= q <= 1.0):
        raise ValueError("")
    lo_lv, hi_lv = inner_levels
    if not (0.0 <= lo_lv < hi_lv <= 1.0):
        raise ValueError("")
    if isinstance(items_by_task, (dict, str)):
        raise ValueError("")

    def _arm_items(rec: dict[str, Any], key: str) -> dict | bool | str:
        if key not in rec:
            raise ValueError("")
        v = rec[key]
        if isinstance(v, dict):
            for k, tok in v.items():
                if not (isinstance(k, tuple) and len(k) == 2):
                    raise ValueError("")
                if tok not in VERDICT_VOCAB:
                    raise ValueError("")
            return v
        return _arm_value(rec, key)

    def _base_est(v: dict | bool | str) -> bool:
        return _est(run_verdict_fn(v) if isinstance(v, dict) else v)

    def _flipped_est(v: dict | bool | str, rng: random.Random) -> bool:
        if isinstance(v, dict):
            flipped = {k: _flip_token(v[k], q, rng, _ITEM_ALTS) for k in sorted(v)}
            return _est(run_verdict_fn(flipped))
        return _est(v)

    arms: list[tuple[dict | bool | str, dict | bool | str]] = []
    base: list[int] = []
    forms: set[str] = set()
    n_arms_with_items = 0
    n_items_inner = 0
    for rec in items_by_task:
        nv, ov = _arm_items(rec, "none"), _arm_items(rec, "oracle")
        d = int(_base_est(ov)) - int(_base_est(nv))
        if "diff" in rec and rec["diff"] != d:
            raise ValueError("")
        arms.append((nv, ov))
        base.append(d)
        for v in (nv, ov):
            forms.add(_arm_form(v))
            if isinstance(v, dict):
                n_arms_with_items += 1
                n_items_inner += len(v)

    n = len(base)
    meta = {"q": float(q), "n_boot": B, "inner_levels": (lo_lv, hi_lv), "seed": seed,
            "inner_unit": "judge_item", "arm_form": _arm_form_summary(forms),
            "n_arms_with_items": n_arms_with_items, "n_items_inner": n_items_inner,
            "item_coverage": (n_arms_with_items / (2 * n)) if n else None,
            "allow_layer_a_only": bool(allow_layer_a_only)}
    if n > 0 and q > 0 and n_arms_with_items == 0 and not allow_layer_a_only:
        raise ValueError(
            "")
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0,
                "n_pos": 0, "n_neg": 0, "n_zero": 0, **meta}

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(B):
        if q > 0:
            world = []
            for nv, ov in arms:
                fn_ = _flipped_est(nv, rng)
                fo_ = _flipped_est(ov, rng)
                world.append(int(fo_) - int(fn_))
        else:
            world = base
        s = 0.0
        for _ in range(n):
            s += world[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(lo_lv * B)]
    hi = means[min(B - 1, int(hi_lv * B))]
    return {"mean": sum(base) / n, "lo": lo, "hi": hi, "n": n,
            "n_pos": sum(1 for d in base if d > 0),
            "n_neg": sum(1 for d in base if d < 0),
            "n_zero": sum(1 for d in base if d == 0), **meta}


def holm_report(estimates: list[dict[str, Any]], *, alpha: float = 0.05,
                B: int = 10000, seed: int = B5_BOOTSTRAP_SEED) -> dict[str, Any]:
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            and 0.0 < alpha <= 1.0):
        raise ValueError("")
    ci_alpha = min(2.0 * alpha, 1.0)
    per: list[dict[str, Any]] = []
    names: set[str] = set()
    for e in estimates:
        name = e["name"]
        if name in names:
            raise ValueError("")
        names.add(name)
        diffs = [_as_number_diff(x) for x in e["diffs_by_task"]]
        if diffs:
            bm = _boot_means(diffs, B, seed)
            k_le0 = sum(1 for m_ in bm if m_ <= 0)
            p = (1 + k_le0) / (B + 1)
        else:
            k_le0, p = None, None
        per.append({"name": name, "n": len(diffs), "p": p, "n_boot_le_zero": k_le0,
                    "ci": bootstrap_ci(diffs, alpha=ci_alpha, n_boot=B, seed=seed),
                    "p_adj": None, "reject": None, "rank": None})

    m = len(per)
    order = sorted(range(m), key=lambda i: (per[i]["p"] is None,
                                            per[i]["p"] if per[i]["p"] is not None else 1.0))
    running = 0.0
    failed = False
    for rank0, i in enumerate(order):
        h = per[i]
        h["rank"] = rank0 + 1
        if h["p"] is None:
            continue
        running = max(running, min(1.0, (m - rank0) * h["p"]))
        h["p_adj"] = running
        if failed or h["p"] > alpha / (m - rank0):
            failed = True
            h["reject"] = False
        else:
            h["reject"] = True
    return {"alpha": alpha, "alpha_sided": "one-sided", "ci_alpha": ci_alpha,
            "m": m, "n_boot": B, "seed": seed, "p_smoothing": "(1+k)/(B+1)",
            "family": (f"confirmatory: r(t)>0 × {{P1,P2,P3,P4}}§7-c "
                       f" =  bootstrap p α={alpha:g}****"
                       f" ↔ {(1 - ci_alpha) * 100:g}%  CI  > 0"
                       f"α=0.05  = #7②/#8 "),
            "per_hypothesis": per}


def is_replicate_row(row: dict[str, Any]) -> bool:
    return bool(row.get(REPLICATE_FIELD))


def replicate_noise_floor(replicate_rows: list[dict[str, Any]],
                          original_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for r in replicate_rows:
        if not is_replicate_row(r):
            raise ValueError("")
    for r in original_rows:
        if is_replicate_row(r):
            raise ValueError("")

    orig: dict[tuple[str, str], bool] = {}
    for r in original_rows:
        k = (r["cat"], r["tid"])
        if k in orig:
            raise ValueError("")
        orig[k] = main_estimator(r["verdict"])

    per_task: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[tuple[str, str], Any]] = set()
    n_unpaired = 0
    for r in replicate_rows:
        k = (r["cat"], r["tid"])
        rep = r[REPLICATE_FIELD]
        if (k, rep) in seen:
            raise ValueError("")
        seen.add((k, rep))
        if k not in orig:
            n_unpaired += 1
            continue
        ach = main_estimator(r["verdict"])
        rec = per_task.setdefault(k, {"orig": orig[k], "reps": {}, "deltas": {}})
        rec["reps"][rep] = ach
        rec["deltas"][rep] = int(ach) - int(orig[k])

    per_cat: dict[str, dict[str, Any]] = {}
    for (cat, _tid), rec in sorted(per_task.items()):
        pc = per_cat.setdefault(cat, {"n_tasks": 0, "n_deltas": 0,
                                      "abs_counts": {0: 0, 1: 0},
                                      "_sums": {}, "_ns": {}})
        pc["n_tasks"] += 1
        for rep, d in rec["deltas"].items():
            pc["n_deltas"] += 1
            pc["abs_counts"][abs(d)] += 1
            pc["_sums"][rep] = pc["_sums"].get(rep, 0) + d
            pc["_ns"][rep] = pc["_ns"].get(rep, 0) + 1

    floor_by_cat: dict[str, float] = {}
    for cat, pc in per_cat.items():
        rep_mean = {rep: pc["_sums"][rep] / pc["_ns"][rep] for rep in sorted(pc["_sums"])}
        rep_n = {rep: pc["_ns"][rep] for rep in sorted(pc["_ns"])}
        del pc["_sums"], pc["_ns"]
        pc["rep_mean"] = rep_mean
        pc["rep_n"] = rep_n
        pc["flip_rate"] = pc["abs_counts"][1] / pc["n_deltas"]
        pc["max_abs_delta"] = 1 if pc["abs_counts"][1] else 0
        pc["max_abs_rep_mean"] = max(abs(v) for v in rep_mean.values())
        pc["floor_rep"] = min(r for r in rep_mean
                              if abs(rep_mean[r]) == pc["max_abs_rep_mean"])
        floor_by_cat[cat] = pc["max_abs_rep_mean"]

    n_pairs = sum(pc["n_deltas"] for pc in per_cat.values())
    n_flips = sum(pc["abs_counts"][1] for pc in per_cat.values())
    return {"per_task": per_task, "per_cat": per_cat, "floor_by_cat": floor_by_cat,
            "n_pairs": n_pairs, "n_unpaired": n_unpaired,
            "flip_rate": (n_flips / n_pairs) if n_pairs else None}


def noise_floor_calibration(replicate_result: dict[str, Any],
                            n_negctl: int | dict[str, int], *,
                            B: int = 10000,
                            seed: int = B5_BOOTSTRAP_SEED) -> dict[str, Any]:
    if not isinstance(replicate_result, dict) or "per_task" not in replicate_result:
        raise ValueError("")
    per_cat_in = replicate_result.get("per_cat") or {}

    def _n_for(cat: str) -> int:
        n = n_negctl[cat] if isinstance(n_negctl, dict) else n_negctl
        if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
            raise ValueError("")
        return n

    pools: dict[str, list[int]] = {}
    for (cat, _tid), rec in replicate_result["per_task"].items():
        pools.setdefault(cat, []).extend(rec["deltas"].values())

    out: dict[str, dict[str, Any]] = {}
    for cat in sorted(set(pools) | set(per_cat_in)):
        pool = pools.get(cat, [])
        pc = per_cat_in.get(cat, {})
        rep_n = pc.get("rep_n") or {}
        rep_mean = pc.get("rep_mean") or {}
        n_cell = _n_for(cat)
        if not pool or not rep_n:
            out[cat] = {"n_negctl": n_cell, "n_pool": len(pool), "n_reps": len(rep_n),
                        "rep_n": dict(rep_n), "pool_mean": None,
                        "floor": pc.get("max_abs_rep_mean"),
                        "h0_exceed_rate": None, "n_boot": B,
                        "loro_exceed_count": None, "loro_n": len(rep_mean)}
            continue
        rng = random.Random(f"{seed}:{cat}")
        L = len(pool)
        exceed = 0
        for _ in range(B):
            floor_star = 0.0
            for _rep, m in rep_n.items():
                s = 0.0
                for _ in range(m):
                    s += pool[rng.randrange(L)]
                floor_star = max(floor_star, abs(s / m))
            s2 = 0.0
            for _ in range(n_cell):
                s2 += pool[rng.randrange(L)]
            if abs(s2 / n_cell) > floor_star:
                exceed += 1
        reps = sorted(rep_mean)
        loro = sum(1 for r in reps
                   if len(reps) > 1
                   and abs(rep_mean[r]) > max(abs(rep_mean[x]) for x in reps if x != r))
        out[cat] = {"n_negctl": n_cell, "n_pool": L, "n_reps": len(rep_n),
                    "rep_n": dict(rep_n), "pool_mean": sum(pool) / L,
                    "floor": pc.get("max_abs_rep_mean"),
                    "h0_exceed_rate": exceed / B, "n_boot": B,
                    "loro_exceed_count": (loro if len(reps) > 1 else None),
                    "loro_n": len(reps)}
    return {"per_cat": out, "n_boot": B, "seed": seed,
            "reading": ("H0 |Δ_| > floor——**** "
                        "`expected_ci_false_positive_cells` 90% CI  0 ")}


def _resolve_floors(floor_src: Any) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if not isinstance(floor_src, dict):
        raise ValueError("")
    if "floor_by_cat" in floor_src and "per_cat" in floor_src:
        floors = dict(floor_src["floor_by_cat"])
        prov: dict[str, dict[str, Any]] = {}
        for cat in floors:
            pc = (floor_src["per_cat"] or {}).get(cat) or {}
            rep_n = dict(pc.get("rep_n") or {})
            frep = pc.get("floor_rep")
            prov[cat] = {"source": "replicate_noise_floor", "rep_n": rep_n,
                         "floor_rep": frep,
                         "floor_n": rep_n.get(frep) if frep is not None else None}
        return floors, prov
    return dict(floor_src), {cat: {"source": "bare", "rep_n": None,
                                   "floor_rep": None, "floor_n": None}
                             for cat in floor_src}


def negctl_verdicts(negctl_cells: list[dict[str, Any]],
                    floor_src: dict[str, Any], *, alpha: float = 0.10,
                    B: int = 10000, seed: int = B5_BOOTSTRAP_SEED,
                    min_floor_rep_n: int = MIN_FLOOR_REP_N,
                    calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    floors, prov = _resolve_floors(floor_src)
    cal_by_cat = ((calibration or {}).get("per_cat") or {}) if calibration else {}
    cal_by_cell = ((calibration or {}).get("by_cell") or {}) if calibration else {}
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    n_data = 0
    n_judged = 0
    exp_floor_rule = 0.0
    have_cal_rate = False
    for c in negctl_cells:
        for k in ("name", "cat", "diffs"):
            if k not in c:
                raise ValueError("")
        key = (c["name"], c["cat"])
        if key in seen:
            raise ValueError("")
        seen.add(key)
        cat = c["cat"]
        if cat not in floors:
            raise ValueError("")
        floor = floors[cat]
        if floor < 0:
            raise ValueError("")
        pv = prov[cat]
        diffs = [_as_number_diff(x) for x in c["diffs"]]
        ci = bootstrap_ci(diffs, alpha=alpha, n_boot=B, seed=seed)
        floor_n = pv["floor_n"]
        cal = cal_by_cat.get(cat) or {}
        rate = cal_by_cell.get(c["name"], cal.get("h0_exceed_rate"))
        if ci["n"] == 0:
            verdict, reason = None, "n=0"
        elif floor_n is not None and floor_n < min_floor_rep_n:
            n_data += 1
            verdict, reason = None, (
                f"floor floor  rep {pv['floor_rep']!r} "
                f"n={floor_n} < {min_floor_rep_n}MIN_FLOOR_REP_N—— "
                "floor [LE1]")
        else:
            n_data += 1
            n_judged += 1
            verdict = "" if abs(ci["mean"]) <= floor else ""
            reason = None
            if rate is not None:
                exp_floor_rule += rate
                have_cal_rate = True
        cells.append({"name": c["name"], "cat": cat, "n": ci["n"],
                      "mean": ci["mean"], "ci": ci, "floor": floor,
                      "verdict": verdict, "reason": reason,
                      "floor_provenance": pv["source"], "floor_rep": pv["floor_rep"],
                      "floor_n": floor_n, "floor_n_per_rep": pv["rep_n"],
                      "floor_rule_h0_rate": rate,
                      "caveat": CROSS_CELL_CAVEAT})
    out = {"cells": cells, "n_cells": len(cells), "n_cells_with_data": n_data,
           "n_cells_judged": n_judged, "alpha": alpha,
           "min_floor_rep_n": min_floor_rep_n,
           "expected_ci_false_positive_cells": alpha * n_data}
    if have_cal_rate or (calibration is not None):
        out["expected_floor_rule_false_positive_cells"] = (
            exp_floor_rule if have_cal_rate else None)
    return out


def position_drift_regression_b5(pairs: list[dict[str, Any]],
                                 positions: dict[tuple[str, str], int],
                                 cats: Iterable[str], *,
                                 warmup_cat: str | Iterable[str] | None = None,
                                 alpha: float = 0.10) -> dict[str, Any]:
    cats = list(cats)
    if not cats:
        raise ValueError("")
    if warmup_cat is None:
        warmup_cat = cats[0]
    warmup_set = {warmup_cat} if isinstance(warmup_cat, str) else set(warmup_cat)
    if not warmup_set <= set(cats):
        raise ValueError("")
    if warmup_set == set(cats):
        raise ValueError("")
    for d in pairs:
        for k in ("cell", "cat", "tid", "diff"):
            if k not in d:
                raise ValueError("")
        if d["cat"] not in cats:
            raise ValueError("")
        if (d["cat"], d["tid"]) not in positions:
            raise ValueError("")

    def _fe_ols(sub: list[dict[str, Any]],
                gkey: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        if len(sub) < 3:
            return {"n": len(sub), "slope": None, "se": None, "p": None, "significant": None}
        groups: dict[Any, list[dict[str, Any]]] = {}
        for d in sub:
            groups.setdefault(gkey(d), []).append(d)
        xs, ys = [], []
        for ds in groups.values():
            mx = sum(positions[(d["cat"], d["tid"])] for d in ds) / len(ds)
            my = sum(d["diff"] for d in ds) / len(ds)
            for d in ds:
                xs.append(positions[(d["cat"], d["tid"])] - mx)
                ys.append(d["diff"] - my)
        sxx = sum(x * x for x in xs)
        if sxx == 0:
            return {"n": len(sub), "slope": None, "se": None, "p": None, "significant": None}
        slope = sum(x * y for x, y in zip(xs, ys)) / sxx
        dof = len(xs) - len(groups) - 1
        if dof <= 0:
            return {"n": len(sub), "slope": slope, "se": None, "p": None, "significant": None}
        rss = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
        se = (rss / dof / sxx) ** 0.5
        if se == 0:
            return {"n": len(sub), "slope": slope, "se": 0.0,
                    "p": 0.0 if slope else 1.0, "significant": bool(slope)}
        z = slope / se
        pval = math.erfc(abs(z) / math.sqrt(2))
        return {"n": len(sub), "slope": slope, "se": se, "p": pval,
                "significant": pval < alpha}

    per_cell: dict[Any, dict[str, Any]] = {}
    for cell in sorted({d["cell"] for d in pairs}, key=str):
        sub = [d for d in pairs if d["cell"] == cell]
        per_cell[cell] = {
            "full": _fe_ols(sub, lambda d: d["cat"]),
            "excl_warmup": _fe_ols([x for x in sub if x["cat"] not in warmup_set],
                                   lambda d: d["cat"]),
        }
    pooled = {
        "full": _fe_ols(list(pairs), lambda d: (d["cell"], d["cat"])),
        "excl_warmup": _fe_ols([x for x in pairs if x["cat"] not in warmup_set],
                               lambda d: (d["cell"], d["cat"])),
    }
    return {"per_cell": per_cell, "pooled": pooled,
            "warmup_cat": (warmup_cat if isinstance(warmup_cat, str)
                           else sorted(warmup_set)),
            "alpha": alpha}


_CATS: tuple[str, ...] = ("c1", "c2", "c3", "c4")


def map_arms_to_b4_slots(rows: list[dict[str, Any]], *, none_arm: str, oracle_arm: str,
                         ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if none_arm == oracle_arm:
        raise ValueError("")
    seen: set[str] = set()
    mapped: list[dict[str, Any]] = []
    n_none = n_oracle = 0
    for r in rows:
        a = r["arm"]
        seen.add(a)
        if a in ("none", "oracle"):
            raise ValueError(
                "")
        if a == none_arm:
            mapped.append({**r, "arm": "none"})
            n_none += 1
        elif a == oracle_arm:
            mapped.append({**r, "arm": "oracle"})
            n_oracle += 1
    if n_none == 0 or n_oracle == 0:
        raise ValueError(
            "")
    return mapped, {"none_arm": none_arm, "oracle_arm": oracle_arm,
                    "n_none": n_none, "n_oracle": n_oracle,
                    "n_dropped": len(rows) - n_none - n_oracle}


class B5PairedUnit:
    __slots__ = ("name", "none_arm", "oracle_arm", "mode", "batch", "axis", "cats", "tier")

    def __init__(self, name: str, none_arm: str, oracle_arm: str, mode: str,
                 batch: str, axis: str, cats: tuple[str, ...], tier: str) -> None:
        self.name = name
        self.none_arm = none_arm
        self.oracle_arm = oracle_arm
        self.mode = mode
        self.batch = batch
        self.axis = axis
        self.cats = cats
        self.tier = tier

    def __repr__(self) -> str:
        return f"B5PairedUnit({self.name!r}, {self.batch}/{self.mode}, tier={self.tier})"


def _cats_for(mode: str) -> tuple[str, ...]:
    ex = MODE_EXCLUDE_CATS.get(mode, frozenset())
    return tuple(c for c in _CATS if c not in ex)


B5_PAIRED_UNITS: tuple[B5PairedUnit, ...] = (
    B5PairedUnit("r(P1)", "fa/none", "fa/fixed-transient", "P1", "main",
                 AXIS_OF["P1"], _cats_for("P1"), "confirmatory"),
    B5PairedUnit("r(P2)", "fa/none", "fa/fixed-persistent", "P2", "main",
                 AXIS_OF["P2"], _cats_for("P2"), "confirmatory"),
    B5PairedUnit("r(P3)", "fa/none", "fa/fixed-transient", "P3", "main",
                 AXIS_OF["P3"], _cats_for("P3"), "confirmatory"),
    B5PairedUnit("r(P4)", "fa/none", "fa/fixed-persistent", "P4", "main",
                 AXIS_OF["P4"], _cats_for("P4"), "confirmatory"),
    B5PairedUnit("d(P1,persistent)", "fa/fixed-persistent", "fa/none", "P1", "main",
                 AXIS_OF["P1"], _CATS, "exploratory"),
    B5PairedUnit("d(P2,transient)", "fa/fixed-transient", "fa/none", "P2", "main",
                 AXIS_OF["P2"], _CATS, "exploratory"),
    B5PairedUnit("d(P3,persistent)", "fa/fixed-persistent", "fa/none", "P3", "main",
                 AXIS_OF["P3"], _CATS, "exploratory"),
    B5PairedUnit("d(P4,transient)", "fa/fixed-transient", "fa/none", "P4", "main",
                 AXIS_OF["P4"], _CATS, "exploratory"),
    B5PairedUnit("d(NP,transient)", "fa/fixed-transient", "fa/none", "P0", "main",
                 "transient", _CATS, "exploratory"),
    B5PairedUnit("d(NP,persistent)", "fa/fixed-persistent", "fa/none", "P0", "main",
                 "persistent", _CATS, "exploratory"),
    B5PairedUnit("r_plain(P2)", "p0/none", "p0/fixed-persistent", "P2", "plain",
                 AXIS_OF["P2"], _cats_for("P2"), "estimation"),
)


def _unit_rows(unit: B5PairedUnit, rows: list[dict[str, Any]], cat: str,
               ) -> list[dict[str, Any]]:
    sub = [r for r in rows if r["cat"] == cat and r["mode"] == unit.mode
           and r["arm"] in (unit.none_arm, unit.oracle_arm)]
    bad = {r.get("batch") for r in sub} - {unit.batch, None}
    if bad:
        raise ValueError("")
    mapped, _meta = map_arms_to_b4_slots(sub, none_arm=unit.none_arm,
                                         oracle_arm=unit.oracle_arm)
    return mapped


def unit_paired_diffs(unit: B5PairedUnit, rows: list[dict[str, Any]],
                      ref_by_cat: dict[str, set[tuple[str, str]]],
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    per_cat: dict[str, Any] = {}
    for cat in unit.cats:
        ref = ref_by_cat.get(cat, set())
        mapped = _unit_rows(unit, rows, cat)
        d = paired_diffs(mapped, cat, unit.mode, ref)
        out.extend(d)
        rec: dict[str, Any] = {"n_ref": len(ref), "n_paired": len(d)}
        if len(d) < len(ref):
            paired_keys = {(x["cat"], x["tid"]) for x in d}
            missing = sorted(t for c, t in ref if (c, t) not in paired_keys)
            rec["n_unpaired"] = len(ref) - len(d)
            rec["unpaired_tids"] = missing
        per_cat[cat] = rec
    return out, {"unit": unit.name, "none_arm": unit.none_arm,
                 "oracle_arm": unit.oracle_arm, "mode": unit.mode,
                 "batch": unit.batch, "axis": unit.axis, "tier": unit.tier,
                 "cats": list(unit.cats), "per_cat": per_cat, "n": len(out)}


def unit_paired_item_maps(unit: B5PairedUnit, rows: list[dict[str, Any]],
                          verdicts_by_run: dict[str, dict[tuple, str]],
                          ref_by_cat: dict[str, set[tuple[str, str]]],
                          ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cat in unit.cats:
        ref = ref_by_cat.get(cat, set())
        mapped = _unit_rows(unit, rows, cat)
        out.extend(paired_item_maps(mapped, verdicts_by_run, cat, unit.mode, ref))
    return out


def staging_task_ids(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    if not runs:
        raise ValueError("")
    idx = [r["staging_index"] for r in runs]
    if sorted(idx) != list(range(len(runs))):
        raise ValueError("")
    if idx != sorted(idx):
        raise ValueError("")
    first: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for r in runs:
        k = (r["cat"], r["tid"])
        if k in seen:
            continue
        seen.add(k)
        first.setdefault(r["cat"], []).append(r["tid"])
    for cat, tids in first.items():
        if tids != sorted(tids):
            raise ValueError(
                "")
    return first


def injected_minus_np_sign(unit: B5PairedUnit) -> int:
    none_is_np = unit.none_arm.endswith("/none")
    oracle_is_np = unit.oracle_arm.endswith("/none")
    if none_is_np == oracle_is_np:
        raise ValueError(
            "")
    return 1 if none_is_np else -1


def calibration_by_cell(replicate_result: dict[str, Any],
                        cells: list[dict[str, Any]], *,
                        B: int = 10000, seed: int = B5_BOOTSTRAP_SEED,
                        ) -> dict[str, Any]:
    by_cell: dict[str, float | None] = {}
    per_cell_n: dict[str, int] = {}
    cache: dict[tuple[str, int], Any] = {}
    for c in cells:
        cat = c["cat"]
        n = len([d for d in c.get("diffs") or []])
        per_cell_n[c["name"]] = n
        key = (cat, n)
        if key not in cache:
            all_cats = sorted((replicate_result.get("per_cat") or {}))
            cache[key] = noise_floor_calibration(
                replicate_result, {c: n for c in all_cats} or {cat: n}, B=B, seed=seed)
        per_cat = (cache[key].get("per_cat") or {}).get(cat) or {}
        by_cell[c["name"]] = per_cat.get("h0_exceed_rate")
    fallback: dict[str, Any] = {}
    for (cat, n), res in sorted(cache.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        fallback.setdefault(cat, (res.get("per_cat") or {}).get(cat) or {})
    return {"by_cell": by_cell, "per_cell_n": per_cell_n,
            "calls": sorted(cache), "per_cat": fallback,
            "per_cat_note": (" cat ** n** "
                             " `by_cell` cat  n ")}
