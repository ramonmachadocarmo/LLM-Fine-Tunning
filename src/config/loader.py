from __future__ import annotations

import argparse
from typing import Any

import yaml

from src.config.builder import normalize_config


def load_yaml(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return normalize_config(data)


def get_config() -> dict[str, Any]:
    """Parse CLI args, load YAML and apply training overrides."""
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Engine")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate")
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
        print(f"Override: Epochs set to {args.epochs}")
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        print(f"Override: Batch Size set to {args.batch_size}")
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
        print(f"Override: Learning Rate set to {args.learning_rate}")

    return config
