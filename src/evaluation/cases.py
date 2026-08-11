from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal

MatchKind = Literal["exact", "contains", "json"]


@dataclass(frozen=True)
class EvalCase:
    id: str
    instruction: str
    expected: str
    match: MatchKind = "contains"


def load_eval_cases(path: str | Path) -> List[EvalCase]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval set not found: {path}")

    cases: List[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            cases.append(_parse_case(row, path, line_no))
    if not cases:
        raise ValueError(f"Eval set is empty: {path}")
    return cases


def iter_eval_cases(path: str | Path) -> Iterator[EvalCase]:
    yield from load_eval_cases(path)


def _parse_case(row: dict, path: Path, line_no: int) -> EvalCase:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{line_no}: expected object")

    instruction = row.get("instruction")
    expected = row.get("expected", row.get("output"))
    if not instruction or expected is None:
        raise ValueError(
            f"{path}:{line_no}: requires 'instruction' and 'expected' (or 'output')"
        )

    match = str(row.get("match", "contains")).strip().lower()
    if match not in ("exact", "contains", "json"):
        raise ValueError(f"{path}:{line_no}: match must be exact|contains|json")

    case_id = str(row.get("id") or f"case-{line_no}")
    return EvalCase(
        id=case_id,
        instruction=str(instruction),
        expected=str(expected),
        match=match,  # type: ignore[arg-type]
    )
