from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from dar.taskmodel import is_action, iter_paths, leaf_scalars

FREE_TEXT_LEN = 40

_MIN_ECHO_TOKEN_LEN = 4

_DEGENERATE_NUM_KEYS = frozenset({"0", "1"})

_NUM_LEFT_BOUNDARY = r"(?<![\w.])(?:(?<!-)|(?<=\w-))"


def _successful_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    pending: dict[str, dict[str, Any]] = {}
    anon = itertools.count()
    for m in messages:
        if m.get("role") == "assistant" and m.get("type") == "tool_call":
            tc = m.get("tool_call") or {}
            cid = tc.get("id") or f"anon-{next(anon)}"
            pending[cid] = {"tool": tc.get("name"), "args": tc.get("arguments") or {}}
        elif m.get("role") == "tool":
            cid = m.get("call_id")
            if cid:
                call = pending.pop(cid, None)
            elif len(pending) == 1:
                call = pending.pop(next(iter(pending)))
            else:
                call = None
            if call is None:
                continue
            content = m.get("content")
            ok = not (isinstance(content, dict) and "error" in content)
            if ok:
                out.append(call)
    return out


def _comparable(v: Any) -> bool:
    if isinstance(v, str):
        return len(v.strip()) <= FREE_TEXT_LEN
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _num_norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num_norm(x) for x in v]
    return v


def _canon(v: Any) -> str:
    import json as _json
    return _json.dumps(_num_norm(v), sort_keys=True, ensure_ascii=False)


def _values_equal(gt: Any, run: Any) -> bool:
    if isinstance(gt, str) and isinstance(run, str):
        return gt.strip() == run.strip()
    if (isinstance(gt, (int, float)) and not isinstance(gt, bool)
            and isinstance(run, (int, float)) and not isinstance(run, bool)):
        return gt == run
    return _canon(gt) == _canon(run)


@dataclass
class PathVerdict:
    path_id: str
    n_actions: int = 0
    missing_actions: list[str] = field(default_factory=list)
    scalar_mismatches: list[dict[str, Any]] = field(default_factory=list)
    free_text_divergent: list[dict[str, Any]] = field(default_factory=list)
    free_text_matched: int = 0

    @property
    def fully_matched(self) -> bool:
        return (not self.missing_actions and not self.scalar_mismatches
                and not self.free_text_divergent)


def _match_candidate(c: dict[str, Any], gt_args: dict[str, Any]) -> tuple[list, list, int]:
    mism: list = []
    diverg: list = []
    matched_ft = 0
    for k, v in gt_args.items():
        present = k in c["args"]
        equal = present and _values_equal(v, c["args"][k])
        if equal:
            if not _comparable(v):
                matched_ft += 1
            continue
        rec = {"arg": k, "gt": v, "run": c["args"].get(k)}
        if _comparable(v):
            mism.append(rec)
        else:
            diverg.append(rec)
    return mism, diverg, matched_ft


def _match_path(pid: str, steps: list[dict[str, Any]],
                calls: list[dict[str, Any]]) -> PathVerdict:
    pv = PathVerdict(path_id=pid)
    for i, s in enumerate(steps):
        tool = s.get("tool_name")
        if not is_action(tool):
            continue
        pv.n_actions += 1
        cands = [c for c in calls if c["tool"] == tool]
        if not cands:
            pv.missing_actions.append(tool)
            continue
        gt_args = s.get("arguments") or {}
        best: tuple[list, list, int] | None = None
        for c in cands:
            got = _match_candidate(c, gt_args)
            if best is None or (len(got[0]), len(got[1])) < (len(best[0]), len(best[1])):
                best = got
            if not got[0] and not got[1]:
                break
        assert best is not None
        mism, diverg, matched_ft = best
        pv.free_text_matched += matched_ft
        for rec in mism:
            pv.scalar_mismatches.append({"step": i, "tool": tool, **rec})
        for rec in diverg:
            pv.free_text_divergent.append({"step": i, "tool": tool, **rec})
    return pv


def _num_key(v: Any) -> str | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        s = v.strip()
        if not re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", s):
            return None
        f = float(s)
    else:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return str(int(f)) if f.is_integer() else repr(f)


def _echo_pattern(token: str) -> re.Pattern[str]:
    nk = _num_key(token)
    if nk is None:
        return re.compile(rf"(?<![\w]){re.escape(token)}(?![\w])")
    if nk in _DEGENERATE_NUM_KEYS or "e" in nk or "E" in nk:
        return re.compile(_NUM_LEFT_BOUNDARY + re.escape(token) + r"(?!\w)(?!\.\d)")
    esc = re.escape(nk)
    if "." in nk:
        k = max(0, _MIN_ECHO_TOKEN_LEN - len(nk))
        rendering = rf"{esc}0{{{k},}}" if k else rf"{esc}0*"
    elif len(nk) >= _MIN_ECHO_TOKEN_LEN:
        rendering = rf"{esc}(?:\.0+)?"
    else:
        k = max(1, _MIN_ECHO_TOKEN_LEN - len(nk) - 1)
        rendering = rf"{esc}\.0{{{k},}}"
    body = (rf"(?:{re.escape(token)}|{rendering})"
            if len(token) >= _MIN_ECHO_TOKEN_LEN else rendering)
    return re.compile(_NUM_LEFT_BOUNDARY + body + r"(?!\w)(?!\.\d)")


def _echo_query_obtainable(token: str, query_text: str) -> bool:
    return _echo_pattern(token).search(query_text) is not None


def corrupt_distinctive_tokens(task_mode: dict[str, Any],
                               task_p0: dict[str, Any]) -> set[str]:
    import json as _json
    corrupt: set[Any] = set()
    for _, steps in iter_paths(task_mode):
        for s in steps:
            if s.get("is_perturbed"):
                corrupt |= leaf_scalars(s.get("output"))
    clean_all: set[Any] = set()
    for _, steps in iter_paths(task_p0):
        for s in steps:
            clean_all |= leaf_scalars(s.get("output"))
            clean_all |= leaf_scalars(s.get("arguments"))
    clean_str = [str(v) for v in clean_all]
    clean_exact = set(clean_str)
    clean_nums = {k for k in (_num_key(v) for v in clean_all) if k is not None}
    qt = _json.dumps(task_p0.get("user_input"), ensure_ascii=False)

    out: set[str] = set()
    for v in corrupt:
        t = str(v)
        if t in clean_exact:
            continue
        nk = _num_key(t)
        if nk is not None and nk in clean_nums:
            continue
        pat = _echo_pattern(t)
        if any(pat.search(c) for c in clean_str):
            continue
        if _echo_query_obtainable(t, qt):
            continue
        out.add(t)
    return out


def _string_leaves(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_string_leaves(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_string_leaves(v))
    elif isinstance(obj, str):
        out.append(obj)
    elif obj is not None:
        out.append(str(obj))
    return out


def corrupt_echo_in_actions(messages: list[dict[str, Any]],
                            tokens: set[str]) -> list[dict[str, str]]:
    pats = {t: _echo_pattern(t) for t in tokens if len(t) >= _MIN_ECHO_TOKEN_LEN}
    hits = []
    for c in _successful_calls(messages):
        if not is_action(c["tool"]):
            continue
        leaves = _string_leaves(c["args"])
        for t, pat in pats.items():
            if any(pat.search(leaf) for leaf in leaves):
                hits.append({"tool": c["tool"], "token": t[:60]})
    return hits


def ga_verdict(messages: list[dict[str, Any]], task_p0: dict[str, Any]) -> dict[str, Any]:
    calls = _successful_calls(messages)
    all_verdicts = [_match_path(pid, steps, calls) for pid, steps in iter_paths(task_p0)]
    verdicts = [v for v in all_verdicts if v.n_actions > 0]
    if not verdicts:
        return {"verdict": "undetermined", "reason_code": "no_canonical_path",
                "reason": "task  canonical path  Action GA "
                          "GT  N ", "paths": []}

    full = [v for v in verdicts if v.fully_matched]
    if full:
        return {"verdict": "achieved", "reason_code": "all_args_match",
                "reason": f"path {full[0].path_id}  Action  gt args ",
                "paths": [v.path_id for v in full]}
    actions_present = [v for v in verdicts if not v.missing_actions]
    if not actions_present:
        best = min(verdicts, key=lambda v: len(v.missing_actions))
        run_actions = [c for c in calls if is_action(c["tool"])]
        if run_actions:
            return {"verdict": "undetermined",
                    "reason_code": "action_substitution_candidate",
                    "reason": " path  Action  run  Action "
                              "——",
                    "closest_path": best.path_id,
                    "missing_actions": best.missing_actions,
                    "run_actions": [c["tool"] for c in run_actions]}
        return {"verdict": "not_achieved", "reason_code": "actions_missing",
                "reason": " path  Action  run  Action ",
                "closest_path": best.path_id,
                "missing_actions": best.missing_actions}
    best = min(actions_present,
               key=lambda v: (len(v.scalar_mismatches), len(v.free_text_divergent)))
    return {"verdict": "undetermined", "reason_code": "arg_divergence",
            "reason": "Action  gt arg dar.ga_semantic",
            "closest_path": best.path_id,
            "scalar_mismatches": best.scalar_mismatches,
            "free_text_divergent": best.free_text_divergent}
