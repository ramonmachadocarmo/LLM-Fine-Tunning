from __future__ import annotations

import os
from typing import Callable

import torch
from peft import PeftModel

from src.training.dataset import format_system_prompt
from src.training.model import load_base_model, load_tokenizer


def build_hf_generator(
    *,
    base_model: str,
    adapter_path: str | None,
    system_prompt: str,
    load_in_4bit: bool = True,
    max_new_tokens: int = 256,
    use_adapter: bool = True,
) -> Callable[[str], str]:
    model = load_base_model(base_model, load_in_4bit=load_in_4bit)
    if use_adapter and adapter_path:
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = load_tokenizer(base_model)

    def generate(instruction: str) -> str:
        prompt = format_system_prompt(
            system_prompt, instruction, "", tokenizer=tokenizer, generate=True
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return generate
