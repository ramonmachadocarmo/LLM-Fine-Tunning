from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLES = [
    {
        "instruction": "Summarize what QLoRA is in two sentences.",
        "output": (
            "QLoRA loads the base model in 4-bit quantization and trains LoRA adapters on top. "
            "It enables fine-tuning larger models on consumer GPUs with much lower VRAM."
        ),
    },
    {
        "instruction": 'Reply with JSON: {"ok": true, "message": "ready"}',
        "output": '{"ok": true, "message": "ready"}',
    },
    {
        "instruction": "Give one reason to use a system prompt during fine-tuning.",
        "output": (
            "A consistent system prompt teaches the model the expected role and output style, "
            "so inference matches training."
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a tiny generic sample JSONL dataset")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--repeat", type=int, default=20, help="Repeat sample block N times")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _ in range(max(1, args.repeat)):
            for row in SAMPLES:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.repeat * len(SAMPLES)} lines -> {out}")


if __name__ == "__main__":
    main()
