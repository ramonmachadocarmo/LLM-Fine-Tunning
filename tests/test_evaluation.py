from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.cases import EvalCase, load_eval_cases
from src.evaluation.harness import run_harness
from src.evaluation.metrics import score_case
from src.evaluation.report import write_report
from src.paths import ROOT


def test_load_eval_sample():
    cases = load_eval_cases(ROOT / "data" / "eval.sample.jsonl")
    assert len(cases) >= 3
    assert {c.match for c in cases} >= {"exact", "contains", "json"}


def test_score_exact_and_contains():
    exact = EvalCase(id="e", instruction="x", expected="ACK", match="exact")
    assert score_case("ACK", exact).passed
    assert not score_case("ACK.", exact).passed

    contains = EvalCase(id="c", instruction="x", expected="LoRA", match="contains")
    assert score_case("LoRA adapts models cheaply.", contains).passed
    assert not score_case("full fine-tune only", contains).passed


def test_score_json_validity_and_fields():
    case = EvalCase(
        id="j",
        instruction="x",
        expected='{"name": "Alice", "score": 92}',
        match="json",
    )
    assert score_case('{"name": "Alice", "score": 92}', case).passed
    assert score_case('{"score": 92, "name": "Alice", "extra": true}', case).passed
    assert not score_case('{"name": "Alice"}', case).passed
    assert not score_case("not-json", case).passed


def test_harness_and_report(tmp_path: Path):
    cases = [
        EvalCase(id="a", instruction="Say ACK", expected="ACK", match="exact"),
        EvalCase(id="b", instruction="Mention LoRA", expected="LoRA", match="contains"),
    ]
    report = run_harness(
        cases,
        predictions={"a": "ACK", "b": "About LoRA"},
        eval_set="inline",
        target="predictions",
    )
    assert report.passed == 2
    path = write_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 2
    assert (tmp_path / path.with_suffix(".md").name).exists()
