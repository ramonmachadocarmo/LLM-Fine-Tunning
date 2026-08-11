from __future__ import annotations

from pathlib import Path

import pytest

from src.training.validate_model import InvalidBaseModelError, validate_base_model


def test_rejects_empty():
    with pytest.raises(InvalidBaseModelError, match="empty"):
        validate_base_model("  ")


def test_rejects_ollama_digest():
    with pytest.raises(InvalidBaseModelError, match="Ollama digest"):
        validate_base_model("7cdf5a0187d5")


def test_rejects_gguf_repo_name():
    with pytest.raises(InvalidBaseModelError, match="GGUF"):
        validate_base_model("yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF")


def test_rejects_gguf_filename():
    with pytest.raises(InvalidBaseModelError, match="GGUF"):
        validate_base_model("model-Q4_K_M.gguf")


def test_rejects_missing_local_path():
    with pytest.raises(InvalidBaseModelError, match="not found"):
        validate_base_model("./merged_models/does_not_exist_xyz")


def test_accepts_local_hf_folder(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}',
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    check = validate_base_model(str(tmp_path))
    assert check.ok
    assert check.kind == "local"
    assert len(check.checksum) == 16


def test_rejects_local_gguf_only(tmp_path: Path):
    (tmp_path / "README.md").write_text("gguf", encoding="utf-8")
    (tmp_path / "weights.gguf").write_bytes(b"fake")
    with pytest.raises(InvalidBaseModelError, match="GGUF"):
        validate_base_model(str(tmp_path))


def test_rejects_gemma4_on_transformers_v4(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}', encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(InvalidBaseModelError, match="Gemma 4"):
        validate_base_model(str(tmp_path))


def test_checksum_stable(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"model_type": "gemma2"}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    a = validate_base_model(str(tmp_path))
    b = validate_base_model(str(tmp_path))
    assert a.checksum == b.checksum
