from __future__ import annotations

from typing import Literal, Protocol

from dar.labels import Label
from dar.router.actions import (
    LABEL_TO_ACTION,
    RecoveryAction,
    RecoveryEvidence,
    RoutingDecision,
)
from dar.router.templates import (
    ORACLE_TOKENS,
    extract_facts,
    generic_context,
    generic_split_context,
    render_context,
    render_split_context,
    template_fingerprint,
)

LabelSource = Literal["diagnosis", "truth"]


class RouterWiringError(RuntimeError):
    pass


def _resolve_action(label: Label, evidence: RecoveryEvidence | None) -> RecoveryAction:
    mapped = LABEL_TO_ACTION.get(label, "__unmapped__")
    if mapped == "__unmapped__":
        raise RouterWiringError(
            f"no recovery action mapped for label {label!r}; map covers "
            f"{sorted(l.value for l in LABEL_TO_ACTION)}"
        )
    if mapped is not None:
        return mapped
    if evidence is None:
        raise RouterWiringError(
            ""
        )
    return (
        RecoveryAction.SWITCH_PATH
        if evidence.has_alternative_path
        else RecoveryAction.GRACEFUL_ABORT
    )


def _decide(
    content_label: Label, action_label: Label, evidence: RecoveryEvidence | None
) -> RoutingDecision:
    action = _resolve_action(action_label, evidence)
    if action is RecoveryAction.NO_OP:
        return RoutingDecision(action=action, recovery_context="")
    content_action = _resolve_action(content_label, evidence)
    if content_action is RecoveryAction.NO_OP:
        return RoutingDecision(action=action, recovery_context=generic_context(action))
    if evidence is None:
        return RoutingDecision(
            action=action,
            recovery_context=generic_split_context(content_action, action),
        )
    context, fields = render_split_context(content_action, action, evidence)
    return RoutingDecision(action=action, recovery_context=context, fields_rendered=fields)


class Router(Protocol):
    def route(self, label: Label, evidence: RecoveryEvidence | None = None) -> RoutingDecision: ...


class TemplateRouter:
    def route(self, label: Label, evidence: RecoveryEvidence | None = None) -> RoutingDecision:
        return _decide(label, label, evidence)


class DecoupledRouter:
    def __init__(
        self, content_from: LabelSource = "diagnosis", action_from: LabelSource = "diagnosis"
    ) -> None:
        for name, value in (("content_from", content_from), ("action_from", action_from)):
            if value not in ("diagnosis", "truth"):
                raise ValueError("")
        self.content_from = content_from
        self.action_from = action_from

    def _label(self, source: LabelSource, diagnosed: Label, evidence: RecoveryEvidence | None) -> Label:
        if source == "diagnosis":
            return diagnosed
        if evidence is None or evidence.oracle_label is None:
            raise RouterWiringError(
                ""
            )
        return evidence.oracle_label

    def route(self, label: Label, evidence: RecoveryEvidence | None = None) -> RoutingDecision:
        content_label = self._label(self.content_from, label, evidence)
        action_label = self._label(self.action_from, label, evidence)
        return _decide(content_label, action_label, evidence)


ROUTER_SPECS: dict[str, tuple[LabelSource, LabelSource]] = {
    "decoupled:content=truth,action=diagnosis": ("truth", "diagnosis"),
    "decoupled:content=diagnosis,action=truth": ("diagnosis", "truth"),
}


def router_from_spec(spec: str | None):
    if spec is None or spec == "template":
        return TemplateRouter()
    try:
        content_from, action_from = ROUTER_SPECS[spec]
    except KeyError:
        raise RouterWiringError(
            ""
        ) from None
    return DecoupledRouter(content_from=content_from, action_from=action_from)


__all__ = [
    "DecoupledRouter",
    "LabelSource",
    "Router",
    "RouterWiringError",
    "TemplateRouter",
    "RecoveryAction",
    "RecoveryEvidence",
    "RoutingDecision",
    "ORACLE_TOKENS",
    "extract_facts",
    "generic_context",
    "generic_split_context",
    "render_context",
    "render_split_context",
    "template_fingerprint",
    "ROUTER_SPECS",
    "router_from_spec",
]
