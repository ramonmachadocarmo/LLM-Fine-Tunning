from __future__ import annotations

from src.training.chat_format import format_system_prompt


def test_format_system_prompt_llama3_fallback():
    text = format_system_prompt("Be brief.", "Hello?", "Hi there.")
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert "Be brief." in text
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert "Hello?" in text
    assert "<|start_header_id|>assistant<|end_header_id|>" in text
    assert "Hi there.<|eot_id|>" in text


def test_format_system_prompt_generate_omits_eot():
    text = format_system_prompt("Be brief.", "Hello?", "", generate=True)
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert not text.endswith("<|eot_id|>")


class _GemmaTok:
    chat_template = "gemma"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "".join(parts)


def test_format_system_prompt_uses_tokenizer_template():
    text = format_system_prompt("Sys", "Hello?", "Hi.", tokenizer=_GemmaTok())
    assert "<start_of_turn>user" in text
    assert "<start_of_turn>model" in text
    assert "Hi." in text
    assert "<|start_header_id|>" not in text
