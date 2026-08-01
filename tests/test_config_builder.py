from __future__ import annotations

import pytest

from src.config.builder import (
    build_config_from_form,
    default_config,
    load_config_file,
    normalize_config,
    resolve_config_path,
    safe_name,
)


def test_safe_name_sanitizes():
    assert safe_name(" My Exp! ") == "my_exp"
    assert safe_name("@@@") == "experiment"


def test_default_config_shape():
    cfg = default_config()
    assert cfg["project"]["name"] == "my_experiment"
    assert cfg["model"]["load_in_4bit"] is True
    assert "dataset_paths" in cfg["training"]
    assert "gguf_filename" in cfg["export"]


def test_normalize_legacy_dataset_path():
    cfg = normalize_config(
        {
            "training": {"dataset_path": "data/a.jsonl", "epochs": 5},
            "system_prompt": "Hi",
        }
    )
    assert cfg["training"]["dataset_paths"] == ["data/a.jsonl"]
    assert cfg["training"]["epochs"] == 5
    assert cfg["system_prompt"] == "Hi"


def test_build_config_from_form():
    cfg = build_config_from_form(
        {
            "project_name": "Smoke Test!",
            "base_model": "meta-llama/Llama-3.2-1B-Instruct",
            "dataset_paths": "data/a.jsonl, data/b.jsonl",
            "epochs": 1,
            "max_steps": 10,
            "load_in_4bit": False,
        }
    )
    assert cfg["project"]["name"] == "smoke_test"
    assert cfg["model"]["base_model"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg["model"]["load_in_4bit"] is False
    assert cfg["training"]["dataset_paths"] == ["data/a.jsonl", "data/b.jsonl"]
    assert cfg["training"]["max_steps"] == 10
    assert cfg["export"]["gguf_filename"] == "./models/smoke_test.gguf"


def test_resolve_config_path_rejects_escape():
    with pytest.raises(ValueError, match="configs/"):
        resolve_config_path("../pyproject.toml")


def test_load_config_file_rejects_traversal():
    with pytest.raises(ValueError, match="configs/"):
        load_config_file("../../etc/passwd")


def test_load_config_file_reads_template():
    cfg = load_config_file("configs/default.template.yaml")
    assert "project" in cfg
    assert "training" in cfg
