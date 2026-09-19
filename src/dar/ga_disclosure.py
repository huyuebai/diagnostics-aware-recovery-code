from __future__ import annotations

import collections
import random
from typing import Any, Hashable, Iterable

from dar.ga_semantic import VERDICT_VOCAB

_ACH, _NOT, _UND = "achieved", "not_achieved", "undetermined"

_ARG_KINDS = {"scalar_variant", "free_text"}
_SUBST_KINDS = {"action_substitution"}


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def c_counts_by_arm_cell(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, int]]:
    out: dict[tuple, dict[str, int]] = collections.defaultdict(
        lambda: {_ACH: 0, _NOT: 0, _UND: 0, "n": 0})
    for r in rows:
        key = (r["arm"], r["cat"], r["mode"])
        v = r["verdict"]
        if v not in (_ACH, _NOT, _UND):
            raise ValueError("")
        out[key][v] += 1
        out[key]["n"] += 1
    return dict(out)


def _rates(verdicts: list[str]) -> tuple[int, int, int]:
    bad = [v for v in verdicts if v not in (_ACH, _NOT, _UND)]
    if bad:
        raise ValueError("")
    n = len(verdicts)
    ach = sum(1 for v in verdicts if v == _ACH)
    und = sum(1 for v in verdicts if v == _UND)
    return n, ach, und


def manski_bounds(oracle_verdicts: list[str], none_verdicts: list[str]) -> dict[str, Any]:
    no, ao, uo = _rates(oracle_verdicts)
    nn, an, un = _rates(none_verdicts)
    if no == 0 or nn == 0:
        return {"lo": None, "hi": None, "main": None, "flagged": None,
                "n_oracle": no, "n_none": nn}
    o_lo, o_hi = ao / no, (ao + uo) / no
    n_lo, n_hi = an / nn, (an + un) / nn
    lo, hi, main = o_lo - n_hi, o_hi - n_lo, o_lo - n_lo
    return {"lo": lo, "hi": hi, "main": main,
            "flagged": _sign(lo) != _sign(hi),
            "n_oracle": no, "n_none": nn}


def split_decision_rates(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, collections.Counter] = {"action_substitution": collections.Counter(),
                                               "arg": collections.Counter()}
    for it in items:
        kind = it["kind"]
        if kind in _SUBST_KINDS:
            b = "action_substitution"
        elif kind in _ARG_KINDS:
            b = "arg"
        else:
            raise ValueError("")
        v = it["verdict"]
        if v not in VERDICT_VOCAB:
            raise ValueError("")
        buckets[b][v] += 1
    out: dict[str, dict[str, Any]] = {}
    for b, cnt in buckets.items():
        n = sum(cnt.values())
        out[b] = {"counts": dict(cnt), "n": n,
                  "rates": {k: cnt[k] / n for k in cnt} if n else {}}
    return out


def reconsistency_subsample(keys: list[Hashable], frac: float = 0.10,
                            seed: int = 20260722) -> list[Hashable]:
    if not 0 < frac < 1:
        raise ValueError("")
    uniq = sorted(set(keys), key=lambda x: repr(x))
    size = round(frac * len(uniq))
    if size == 0:
        return []
    return sorted(random.Random(seed).sample(uniq, size), key=lambda x: repr(x))
