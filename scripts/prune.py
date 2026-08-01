"""Downsample a JSONL dataset to a target size with light diversity by instruction prefix."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def bucket_key(instruction: str, width: int = 48) -> str:
    text = " ".join((instruction or "").lower().split())
    return text[:width] or "empty"


def prune_dataset(input_path: str, output_path: str, target: int) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(input_path)

    lines = path.read_text(encoding="utf-8").splitlines()
    groups: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        groups[bucket_key(str(data.get("instruction", "")))].append(line)

    if not groups:
        raise ValueError("No valid JSONL rows found")

    per_group = max(1, int((target / len(groups)) * 1.5))
    selected: list[str] = []
    for rows in groups.values():
        if len(rows) > per_group:
            selected.extend(random.sample(rows, per_group))
        else:
            selected.extend(rows)

    random.shuffle(selected)
    if len(selected) > target:
        selected = selected[:target]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"{len(lines)} -> {len(selected)} lines ({len(groups)} buckets) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance / downsample JSONL dataset")
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--output", default="data/train_balanced.jsonl")
    parser.add_argument("--target", type=int, default=5000)
    args = parser.parse_args()
    prune_dataset(args.input, args.output, args.target)


if __name__ == "__main__":
    main()
