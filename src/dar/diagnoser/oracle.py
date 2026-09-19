from __future__ import annotations

from dar.diagnoser.base import DiagnoserCostRecord, DiagnosisInput, DiagnosisOutput
from dar.labels import Label, TrueType, correct_label


class OracleDiagnoser:
    TRACE_INDEPENDENT = True

    def __init__(self, true_type: TrueType | str) -> None:
        self.true_type = TrueType(true_type)
        self.label: Label = correct_label(self.true_type)
        self.name = "oracle"

    def reset(self) -> None:
        pass

    def diagnose(self, obs: DiagnosisInput) -> DiagnosisOutput:
        return DiagnosisOutput(
            label=self.label,
            cost=DiagnoserCostRecord(basis="oracle=0()"),
            detail={"true_type": self.true_type.value, "source": "oracle"},
        )
