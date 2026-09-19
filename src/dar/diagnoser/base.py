from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from dar.labels import Label

ZERO_COST_BASIS = "zero-cost($ )"

GROUND_TRUTH_KEYS = frozenset({"perturbation_status", "perturbation_mode", "is_perturbed"})


def _strip_gt(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_gt(v) for k, v in obj.items() if k not in GROUND_TRUTH_KEYS}
    if isinstance(obj, list):
        return [_strip_gt(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_gt(v) for v in obj)
    return obj


def sanitize_message(msg: dict[str, Any]) -> dict[str, Any]:
    return _strip_gt(msg)


def sanitize_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_message(m) for m in messages]


@dataclass(frozen=True)
class DiagnosisInput:
    trace_prefix: list[dict[str, Any]]
    task_description: str = ""
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    trigger_round: int = -1

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]], **kw: Any) -> "DiagnosisInput":
        return cls(trace_prefix=sanitize_trace(messages), **kw)


@dataclass(frozen=True)
class DiagnoserCostRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    usd: float = 0.0
    basis: str = ZERO_COST_BASIS

    def __post_init__(self) -> None:
        if not self.basis.strip():
            raise ValueError(
                "")
        if self.usd != 0.0 and self.basis == ZERO_COST_BASIS:
            raise ValueError(
                "")


@dataclass(frozen=True)
class DiagnosisOutput:
    label: Label
    cost: DiagnoserCostRecord = field(default_factory=DiagnoserCostRecord)
    detail: dict[str, Any] = field(default_factory=dict)


class Diagnoser(Protocol):
    name: str

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        ...
