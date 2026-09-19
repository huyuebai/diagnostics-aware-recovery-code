from __future__ import annotations

from enum import Enum


class TrueType(str, Enum):
    NP = "NP"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Label(str, Enum):
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    NP = "np"


TRUE_TYPE_TO_LABEL: dict[TrueType, Label] = {
    TrueType.NP: Label.NP,
    TrueType.P1: Label.TRANSIENT,
    TrueType.P3: Label.TRANSIENT,
    TrueType.P2: Label.PERSISTENT,
    TrueType.P4: Label.PERSISTENT,
}

EXPLICIT_TYPES = frozenset({TrueType.P1, TrueType.P2})
IMPLICIT_TYPES = frozenset({TrueType.P3, TrueType.P4})

CELLS: tuple[tuple[TrueType, Label], ...] = tuple(
    (t, l) for t in TrueType for l in Label
)


def correct_label(true_type: TrueType | str) -> Label:
    t = TrueType(true_type)
    return TRUE_TYPE_TO_LABEL[t]


def is_diagonal(true_type: TrueType | str, label: Label | str) -> bool:
    return correct_label(true_type) is Label(label)


def true_type_from_mode(mode: str) -> TrueType:
    head = str(mode).split("_")[0].upper()
    if head == "P0":
        return TrueType.NP
    return TrueType(head)
