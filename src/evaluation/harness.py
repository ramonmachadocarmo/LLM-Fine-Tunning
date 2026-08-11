from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Sequence

from src.evaluation.cases import EvalCase
from src.evaluation.metrics import MetricResult, score_case

GenerateFn = Callable[[str], str]


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    prediction: str
    metric: MetricResult


@dataclass
class EvalReport:
    eval_set: str
    target: str
    results: List[CaseResult] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.metric.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0


def run_harness(
    cases: Sequence[EvalCase],
    *,
    generate: GenerateFn | None = None,
    predictions: Dict[str, str] | None = None,
    eval_set: str = "",
    target: str = "unknown",
) -> EvalReport:
    if generate is None and predictions is None:
        raise ValueError("Provide generate= or predictions=")

    results: List[CaseResult] = []
    for case in cases:
        if predictions is not None and case.id in predictions:
            prediction = predictions[case.id]
        elif generate is not None:
            prediction = generate(case.instruction)
        elif predictions is not None:
            raise KeyError(f"Missing prediction for case id={case.id}")
        else:
            raise RuntimeError("unreachable")

        results.append(
            CaseResult(
                case=case,
                prediction=prediction,
                metric=score_case(prediction, case),
            )
        )

    return EvalReport(eval_set=eval_set, target=target, results=results)
