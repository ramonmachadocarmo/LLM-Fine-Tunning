from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIGS_DIR, ROOT, UI_CONFIGS_DIR


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip()).strip("_").lower()
    return cleaned or "experiment"


def default_config() -> dict[str, Any]:
    return {
        "project": {
            "name": "my_experiment",
            "output_dir": "./adapters/my_experiment",
        },
        "model": {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "load_in_4bit": True,
        },
        "training": {
            "dataset_paths": ["data/train.jsonl"],
            "epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "save_strategy": "steps",
            "save_steps": 50,
            "learning_rate": 2.0e-4,
            "max_seq_length": 1024,
        },
        "system_prompt": (
            "You are a helpful assistant. Follow the user's instructions carefully. "
            "Respond with the format requested in the prompt (plain text or JSON when asked)."
        ),
        "export": {
            "adapter_path": "./adapters/my_experiment",
            "merged_path": "./models/my_experiment_merged",
            "gguf_filename": "./models/my_experiment.gguf",
        },
    }


def normalize_config(data: dict[str, Any]) -> dict[str, Any]:
    cfg = default_config()
    cfg["project"].update(data.get("project") or {})
    cfg["model"].update(data.get("model") or {})

    training = dict(data.get("training") or {})
    if "dataset_path" in training and "dataset_paths" not in training:
        path = training.pop("dataset_path")
        training["dataset_paths"] = [path] if isinstance(path, str) else list(path or [])
    cfg["training"].update(training)

    if "system_prompt" in data:
        cfg["system_prompt"] = data["system_prompt"]
    cfg["export"].update(data.get("export") or {})
    return cfg


def build_config_from_form(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = default_config()
    project_name = safe_name(str(payload.get("project_name") or "new_experiment"))
    output_dir = payload.get("output_dir") or f"./adapters/{project_name}"

    cfg["project"]["name"] = project_name
    cfg["project"]["output_dir"] = output_dir
    cfg["model"]["base_model"] = payload.get("base_model") or cfg["model"]["base_model"]
    cfg["model"]["load_in_4bit"] = bool(payload.get("load_in_4bit", True))

    datasets = payload.get("dataset_paths") or []
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",") if d.strip()]

    training = cfg["training"]
    training["dataset_paths"] = list(datasets)
    training["epochs"] = int(payload.get("epochs", training["epochs"]))
    training["batch_size"] = int(payload.get("batch_size", training["batch_size"]))
    training["gradient_accumulation_steps"] = int(
        payload.get("gradient_accumulation_steps", training["gradient_accumulation_steps"])
    )
    training["save_strategy"] = payload.get("save_strategy") or training["save_strategy"]
    training["save_steps"] = int(payload.get("save_steps", training["save_steps"]))
    training["learning_rate"] = float(payload.get("learning_rate", training["learning_rate"]))
    training["max_seq_length"] = int(payload.get("max_seq_length", training["max_seq_length"]))

    if payload.get("max_steps") not in (None, "", 0, "0"):
        training["max_steps"] = int(payload["max_steps"])

    cfg["system_prompt"] = payload.get("system_prompt") or cfg["system_prompt"]

    adapter_path = payload.get("adapter_path") or output_dir
    merged_path = payload.get("merged_path") or f"./merged_models/{project_name}"
    gguf_filename = payload.get("gguf_filename") or f"./models/{project_name}.gguf"

    cfg["export"]["adapter_path"] = adapter_path
    cfg["export"]["merged_path"] = merged_path
    cfg["export"]["gguf_filename"] = gguf_filename
    return cfg


def save_config(cfg: dict[str, Any], filename: str | None = None) -> str:
    UI_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_name(filename or cfg["project"]["name"])
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = UI_CONFIGS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return path.relative_to(ROOT).as_posix()


def resolve_config_path(rel_path: str) -> Path:
    configs = CONFIGS_DIR.resolve()
    path = (ROOT / rel_path).resolve()
    if not path.is_relative_to(configs):
        raise ValueError("Config path must be under configs/")
    return path


def load_config_file(rel_path: str) -> dict[str, Any]:
    path = resolve_config_path(rel_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {rel_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return normalize_config(data)
