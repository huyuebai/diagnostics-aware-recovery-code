from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dar.diagnoser.base import DiagnoserCostRecord, DiagnosisInput, DiagnosisOutput
from dar.labels import Label
from dar.np_trigger import tool_call_succeeded

LABEL_ORDER: tuple[Label, ...] = (Label.TRANSIENT, Label.PERSISTENT, Label.NP)

FEATURE_NAMES: tuple[str, ...] = (
    "prefix_rounds",
    "n_tool_calls",
    "n_distinct_tools",
    "n_error_results",
    "error_result_ratio",
    "max_consecutive_same_tool_errors",
    "last_tool_result_is_error",
    "max_same_tool_same_args_repeats",
    "n_distinct_tools_with_error",
    "n_assistant_messages",
)

_LOG1P_FEATURES = frozenset(FEATURE_NAMES) - {"error_result_ratio", "last_tool_result_is_error"}

B6_ML_FOLD_SEED = 20260813

ZERO_BASIS = "ml=0( python  API )"


def _is_error(content: Any) -> bool:
    return not tool_call_succeeded(content)


def raw_features(obs: DiagnosisInput) -> dict[str, float]:
    n_tool_results = 0
    n_err = 0
    n_asst = 0
    n_calls = 0
    tools_called: list[str] = []
    tools_with_error: set[str] = set()
    run_len: dict[str, int] = {}
    max_run: dict[str, int] = {}
    call_sig: dict[str, int] = {}
    last_is_error = 0

    for m in obs.trace_prefix:
        role = m.get("role")
        if role == "assistant":
            n_asst += 1
            tc = m.get("tool_call") or {}
            name = tc.get("name")
            if name:
                n_calls += 1
                tools_called.append(str(name))
                try:
                    args = json.dumps(tc.get("arguments"), sort_keys=True, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = repr(tc.get("arguments"))
                sig = f"{name}\x00{args}"
                call_sig[sig] = call_sig.get(sig, 0) + 1
        elif role == "tool":
            n_tool_results += 1
            tool = str(m.get("name") or "?")
            if _is_error(m.get("content")):
                n_err += 1
                last_is_error = 1
                tools_with_error.add(tool)
                run_len[tool] = run_len.get(tool, 0) + 1
                max_run[tool] = max(max_run.get(tool, 0), run_len[tool])
            else:
                last_is_error = 0
                run_len[tool] = 0

    return {
        "prefix_rounds": float(n_tool_results),
        "n_tool_calls": float(n_calls),
        "n_distinct_tools": float(len(set(tools_called))),
        "n_error_results": float(n_err),
        "error_result_ratio": (n_err / n_tool_results) if n_tool_results else 0.0,
        "max_consecutive_same_tool_errors": float(max(max_run.values()) if max_run else 0),
        "last_tool_result_is_error": float(last_is_error),
        "max_same_tool_same_args_repeats": float(max(call_sig.values()) if call_sig else 0),
        "n_distinct_tools_with_error": float(len(tools_with_error)),
        "n_assistant_messages": float(n_asst),
    }


def feature_vector(obs: DiagnosisInput) -> list[float]:
    raw = raw_features(obs)
    return [math.log1p(raw[k]) if k in _LOG1P_FEATURES else raw[k] for k in FEATURE_NAMES]


def fold_of_base_ids(base_ids_by_cat: dict[str, Iterable[str]], *,
                     seed: int = B6_ML_FOLD_SEED, k: int = 5) -> dict[str, int]:
    if k < 2:
        raise ValueError("")
    out: dict[str, int] = {}
    for cat in sorted(base_ids_by_cat):
        ids = sorted(set(base_ids_by_cat[cat]),
                     key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
        for i, tid in enumerate(ids):
            if tid in out:
                raise ValueError("")
            out[tid] = i % k
    return out


@dataclass(frozen=True)
class TrainReport:
    n_samples: int
    n_features: int
    epochs_ran: int
    initial_objective: float
    final_objective: float
    stopped_by_tolerance: bool
    max_abs_weight: float


class MultinomialLogistic:
    def __init__(self, *, l2: float = 1e-4, lr: float = 0.1, max_epochs: int = 20000,
                 tol: float = 1e-9, regularize_bias: bool = False,
                 n_classes: int = len(LABEL_ORDER)) -> None:
        self.l2 = float(l2)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.tol = float(tol)
        self.regularize_bias = bool(regularize_bias)
        self.n_classes = int(n_classes)
        self.W: list[list[float]] = []
        self.b: list[float] = []
        self.report: TrainReport | None = None

    def _objective_and_grad(self, X: Sequence[Sequence[float]], y: Sequence[int],
                            ) -> tuple[float, list[list[float]], list[float]]:
        n = len(X)
        d = len(self.W[0])
        c = self.n_classes
        gW = [[0.0] * d for _ in range(c)]
        gb = [0.0] * c
        nll = 0.0
        W, b = self.W, self.b
        for xi, yi in zip(X, y):
            logits = [b[k] + sum(w * v for w, v in zip(W[k], xi)) for k in range(c)]
            mx = max(logits)
            exps = [math.exp(z - mx) for z in logits]
            s = sum(exps)
            nll += math.log(s) + mx - logits[yi]
            for k in range(c):
                err = exps[k] / s - (1.0 if k == yi else 0.0)
                if err:
                    gk = gW[k]
                    for j, v in enumerate(xi):
                        gk[j] += err * v
                    gb[k] += err
        obj = nll / n
        for k in range(c):
            gk, wk = gW[k], W[k]
            for j in range(d):
                gk[j] = gk[j] / n + self.l2 * wk[j]
            gb[k] /= n
        obj += 0.5 * self.l2 * sum(w * w for wk in W for w in wk)
        if self.regularize_bias:
            obj += 0.5 * self.l2 * sum(v * v for v in b)
            for k in range(c):
                gb[k] += self.l2 * b[k]
        return obj, gW, gb

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> TrainReport:
        if not X:
            raise ValueError("")
        d = len(X[0])
        if any(len(xi) != d for xi in X):
            raise ValueError("")
        if any(not (0 <= yi < self.n_classes) for yi in y):
            raise ValueError("")
        self.W = [[0.0] * d for _ in range(self.n_classes)]
        self.b = [0.0] * self.n_classes

        obj, gW, gb = self._objective_and_grad(X, y)
        initial = obj
        stopped_by_tol = False
        epochs = 0
        for _ in range(self.max_epochs):
            for k in range(self.n_classes):
                wk, gk = self.W[k], gW[k]
                for j in range(d):
                    wk[j] -= self.lr * gk[j]
                self.b[k] -= self.lr * gb[k]
            epochs += 1
            new_obj, gW, gb = self._objective_and_grad(X, y)
            improvement = obj - new_obj
            obj = new_obj
            if improvement < self.tol:
                stopped_by_tol = True
                break

        max_w = max((abs(w) for wk in self.W for w in wk), default=0.0)
        rep = TrainReport(n_samples=len(X), n_features=d, epochs_ran=epochs,
                          initial_objective=initial, final_objective=obj,
                          stopped_by_tolerance=stopped_by_tol,
                          max_abs_weight=max_w)
        self.report = rep
        assert_converged(rep, self)
        return rep

    def predict(self, x: Sequence[float]) -> int:
        if not self.W:
            raise RuntimeError("")
        logits = [self.b[k] + sum(w * v for w, v in zip(self.W[k], x))
                  for k in range(self.n_classes)]
        best = 0
        for k in range(1, self.n_classes):
            if logits[k] > logits[best]:
                best = k
        return best


def assert_converged(rep: TrainReport, model: "MultinomialLogistic") -> None:
    if not (rep.final_objective <= rep.initial_objective):
        raise RuntimeError(
            "")
    if not all(math.isfinite(w) for wk in model.W for w in wk) or \
       not all(math.isfinite(v) for v in model.b):
        raise RuntimeError("")
    if not rep.stopped_by_tolerance:
        raise RuntimeError(
            "")


class MLDiagnoser:
    name = "ml"

    def __init__(self, model: MultinomialLogistic, *, fold: int | None = None) -> None:
        self.model = model
        self.fold = fold

    def reset(self) -> None:
        pass

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        x = feature_vector(obs)
        k = self.model.predict(x)
        return DiagnosisOutput(
            label=LABEL_ORDER[k],
            cost=DiagnoserCostRecord(calls=0, basis=ZERO_BASIS),
            detail={"source": "ml", "fold": self.fold,
                    "features": dict(zip(FEATURE_NAMES, x))},
        )


def cross_fitted_predictions(units: Sequence[tuple[str, list[float], int]], *,
                             folds: dict[str, int], k_folds: int = 5,
                             l2: float = 1e-4, lr: float = 0.1,
                             max_epochs: int = 20000, tol: float = 1e-9,
                             regularize_bias: bool = False,
                             ) -> tuple[list[int], dict[int, TrainReport]]:
    missing = sorted({t for t, _, _ in units if t not in folds})
    if missing:
        raise KeyError("")
    preds: list[int | None] = [None] * len(units)
    reports: dict[int, TrainReport] = {}
    for f in range(k_folds):
        tr_x, tr_y, te_idx = [], [], []
        for i, (tid, x, y) in enumerate(units):
            if folds[tid] == f:
                te_idx.append(i)
            else:
                tr_x.append(x)
                tr_y.append(y)
        if not te_idx:
            raise ValueError("")
        model = MultinomialLogistic(l2=l2, lr=lr, max_epochs=max_epochs, tol=tol,
                                    regularize_bias=regularize_bias)
        reports[f] = model.fit(tr_x, tr_y)
        for i in te_idx:
            preds[i] = model.predict(units[i][1])
    if any(p is None for p in preds):
        raise RuntimeError("")
    return [int(p) for p in preds], reports
