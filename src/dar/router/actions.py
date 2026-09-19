from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dar.labels import Label


class RecoveryAction(str, Enum):
    NO_OP = "no_op"
    RETRY_SAME_TOOL = "retry_same_tool"
    SWITCH_PATH = "switch_path"
    GRACEFUL_ABORT = "graceful_abort"


LABEL_TO_ACTION: dict[Label, RecoveryAction | None] = {
    Label.NP: RecoveryAction.NO_OP,
    Label.TRANSIENT: RecoveryAction.RETRY_SAME_TOOL,
    Label.PERSISTENT: None,
}


@dataclass
class RecoveryEvidence:
    last_action: dict[str, Any] | None
    last_error: str | None
    step_index: int
    has_alternative_path: bool = False
    available_tools: list[dict[str, Any]] | None = None
    oracle_label: Label | None = None


@dataclass
class RoutingDecision:
    action: RecoveryAction
    recovery_context: str = ""
    fields_rendered: dict[str, str | None] = field(default_factory=dict)
