from __future__ import annotations

import json
from dataclasses import dataclass

from src.evaluation.cases import EvalCase


@dataclass(frozen=True)
class MetricResult:
    passed: bool
    metric: str
    detail: str


def score_case(prediction: str, case: EvalCase) -> MetricResult:
    pred = (prediction or "").strip()
    expected = (case.expected or "").strip()

    if case.match == "exact":
        passed = pred == expected
        return MetricResult(
            passed=passed,
            metric="exact",
            detail="exact match" if passed else "prediction != expected",
        )

    if case.match == "contains":
        passed = expected.lower() in pred.lower()
        return MetricResult(
            passed=passed,
            metric="contains",
            detail="expected substring found" if passed else "expected substring missing",
        )

    return _score_json(pred, expected)


def _score_json(prediction: str, expected: str) -> MetricResult:
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError as exc:
        return MetricResult(
            passed=False,
            metric="json",
            detail=f"invalid JSON: {exc.msg}",
        )

    if not isinstance(parsed, (dict, list)):
        return MetricResult(
            passed=False,
            metric="json",
            detail="JSON root must be object or array",
        )

    try:
        expected_obj = json.loads(expected)
    except json.JSONDecodeError:
        return MetricResult(
            passed=True,
            metric="json",
            detail="valid JSON (expected was not JSON; validity-only check)",
        )

    if parsed == expected_obj:
        return MetricResult(passed=True, metric="json", detail="JSON equals expected")

    if isinstance(parsed, dict) and isinstance(expected_obj, dict):
        missing = [k for k in expected_obj if k not in parsed]
        if missing:
            return MetricResult(
                passed=False,
                metric="json",
                detail=f"missing keys: {', '.join(missing)}",
            )
        mismatches = [
            k for k, v in expected_obj.items() if parsed.get(k) != v
        ]
        if mismatches:
            return MetricResult(
                passed=False,
                metric="json",
                detail=f"value mismatch on: {', '.join(mismatches)}",
            )
        return MetricResult(passed=True, metric="json", detail="JSON contains expected fields")

    return MetricResult(
        passed=False,
        metric="json",
        detail="valid JSON but does not match expected structure",
    )
