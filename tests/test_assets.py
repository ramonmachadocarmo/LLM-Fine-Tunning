from __future__ import annotations

from web.application.assets import preferred_config_path, suggested_hf_models
from web.application.templates import dataset_sample_jsonl


def test_preferred_config_prefers_default_yaml():
    items = [
        {"path": "configs/default.template.yaml"},
        {"path": "configs/ui/foo.yaml"},
        {"path": "configs/default.yaml"},
    ]
    assert preferred_config_path(items) == "configs/default.yaml"


def test_preferred_config_falls_back_to_local():
    items = [
        {"path": "configs/default.template.yaml"},
        {"path": "configs/ui/exp.yaml"},
    ]
    assert preferred_config_path(items) == "configs/ui/exp.yaml"


def test_preferred_config_empty():
    assert preferred_config_path([]) is None


def test_suggested_hf_models_nonempty():
    assert any("Llama" in m for m in suggested_hf_models())


def test_dataset_sample_jsonl_rows():
    body = dataset_sample_jsonl()
    lines = [ln for ln in body.strip().splitlines() if ln]
    assert len(lines) >= 2
    assert '"instruction"' in lines[0]
    assert '"output"' in lines[0]
