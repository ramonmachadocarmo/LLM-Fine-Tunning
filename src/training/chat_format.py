from __future__ import annotations


def llama3_chat(system_prompt: str, instruction: str, output: str = "", *, generate: bool = False) -> str:
    text = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if generate:
        return text
    return f"{text}{output}<|eot_id|>"


def format_system_prompt(
    system_prompt: str,
    instruction: str,
    output: str = "",
    *,
    tokenizer=None,
    generate: bool | None = None,
) -> str:
    want_generate = (not output) if generate is None else generate
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": instruction},
        ]
        if output and not want_generate:
            messages.append({"role": "assistant", "content": output})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=want_generate
            )
        except Exception:
            user = f"{system_prompt.strip()}\n\n{instruction}" if system_prompt.strip() else instruction
            fallback = [{"role": "user", "content": user}]
            if output and not want_generate:
                fallback.append({"role": "assistant", "content": output})
            return tokenizer.apply_chat_template(
                fallback, tokenize=False, add_generation_prompt=want_generate
            )
    return llama3_chat(system_prompt, instruction, output, generate=want_generate)
