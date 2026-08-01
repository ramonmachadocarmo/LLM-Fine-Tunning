from __future__ import annotations

import json
from typing import Any

import yaml


def config_template_yaml(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)


def dataset_sample_jsonl() -> str:
    rows = [
        {
            "instruction": "Explain what LoRA fine-tuning is in one short paragraph.",
            "output": (
                "LoRA (Low-Rank Adaptation) freezes the base model weights and trains small "
                "rank-decomposition matrices injected into attention layers. This cuts trainable "
                "parameters and VRAM while adapting the model to a domain or task."
            ),
        },
        {
            "instruction": 'Return JSON {"status": "ok"}.',
            "output": '{"status": "ok"}',
        },
    ]
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
