from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from src.config import load_yaml
from src.evaluation import load_eval_cases, run_harness, write_report
from src.evaluation.generate import build_hf_generator
from src.paths import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline eval harness (base vs adapter)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--eval-set",
        default="data/eval.sample.jsonl",
        help="JSONL with instruction + expected (+ match)",
    )
    parser.add_argument(
        "--target",
        choices=("base", "adapter"),
        default="adapter",
        help="Which model to score",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional JSON map {case_id: prediction} to score without loading a model",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Report directory (default: logs/eval)",
    )
    args, _unknown = parser.parse_known_args()

    eval_path = Path(args.eval_set)
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path

    cases = load_eval_cases(eval_path)
    config = load_yaml(args.config)

    if args.predictions:
        pred_path = Path(args.predictions)
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        report = run_harness(
            cases,
            predictions=predictions,
            eval_set=str(eval_path),
            target=f"predictions:{pred_path.name}",
        )
    else:
        base_model = config["model"]["base_model"]
        adapter_path = config["project"]["output_dir"]
        system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        load_in_4bit = config["model"].get("load_in_4bit", True)
        generate = build_hf_generator(
            base_model=base_model,
            adapter_path=adapter_path,
            system_prompt=system_prompt,
            load_in_4bit=load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            use_adapter=args.target == "adapter",
        )
        report = run_harness(
            cases,
            generate=generate,
            eval_set=str(eval_path),
            target=args.target,
        )

    out_path = write_report(report, args.out_dir)
    print(
        f"Eval {report.passed}/{report.total} passed "
        f"({report.pass_rate:.1%}) → {out_path}"
    )
    for item in report.results:
        flag = "PASS" if item.metric.passed else "FAIL"
        print(f"  [{flag}] {item.case.id} ({item.metric.metric}): {item.metric.detail}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
