from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.harness import EvalReport
from src.paths import EVAL_LOGS_DIR


def write_report(report: EvalReport, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else EVAL_LOGS_DIR
    out.mkdir(parents=True, exist_ok=True)

    stamp = report.created_at.replace(":", "").replace("+00:00", "Z")
    safe_target = "".join(c if c.isalnum() or c in "-_" else "_" for c in report.target)
    base = f"eval_{safe_target}_{stamp}"

    payload = {
        "created_at": report.created_at,
        "eval_set": report.eval_set,
        "target": report.target,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": round(report.pass_rate, 4),
        },
        "cases": [
            {
                "id": item.case.id,
                "match": item.case.match,
                "instruction": item.case.instruction,
                "expected": item.case.expected,
                "prediction": item.prediction,
                "passed": item.metric.passed,
                "metric": item.metric.metric,
                "detail": item.metric.detail,
            }
            for item in report.results
        ],
    }

    json_path = out / f"{base}.json"
    md_path = out / f"{base}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return json_path


def _to_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        f"# Eval report — {payload['target']}",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- Eval set: `{payload['eval_set']}`",
        f"- Passed: **{summary['passed']}/{summary['total']}** ({summary['pass_rate']:.1%})",
        "",
        "| id | match | passed | detail |",
        "|----|-------|--------|--------|",
    ]
    for case in payload["cases"]:
        flag = "yes" if case["passed"] else "no"
        detail = str(case["detail"]).replace("|", "\\|")
        lines.append(f"| `{case['id']}` | {case['match']} | {flag} | {detail} |")
    lines.append("")
    return "\n".join(lines)
