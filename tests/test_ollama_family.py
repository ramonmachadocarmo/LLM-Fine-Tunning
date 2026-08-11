from __future__ import annotations

import json
from pathlib import Path

from web.application.ollama_chat import detect_gguf_family


def test_detect_family_from_merged_config(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "config.json").write_text(
        json.dumps({"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"]}),
        encoding="utf-8",
    )
    assert detect_gguf_family(gguf) == "gemma"
