from __future__ import annotations

from dar.diagnoser.base import DiagnoserCostRecord, DiagnosisInput, DiagnosisOutput
from dar.labels import Label


class FixedLabelDiagnoser:
    TRACE_INDEPENDENT = True

    def __init__(self, label: Label | str) -> None:
        self.label = Label(label)
        self.name = f"fixed:{self.label.value}"

    def reset(self) -> None:
        pass

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        return DiagnosisOutput(
            label=self.label,
            cost=DiagnoserCostRecord(basis="fixed=0"),
            detail={"assigned_label": self.label.value, "source": "fixed"},
        )
