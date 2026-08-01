from __future__ import annotations

from src.training.dataset import format_system_prompt


def test_format_system_prompt_llama3_chat():
    text = format_system_prompt("Be brief.", "Hello?", "Hi there.")
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert "Be brief." in text
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert "Hello?" in text
    assert "<|start_header_id|>assistant<|end_header_id|>" in text
    assert "Hi there.<|eot_id|>" in text
