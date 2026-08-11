import argparse
import os

import torch
from peft import PeftModel

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from src.config import load_yaml
from src.training.chat_format import format_system_prompt
from src.training.model import load_base_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Verify Fine-Tuned Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    args, _unknown = parser.parse_known_args()

    config = load_yaml(args.config)

    base_model = config["model"]["base_model"]
    adapter_path = config["project"]["output_dir"]
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")

    print(f"Loading Base Model: {base_model}")
    load_in_4bit = config["model"].get("load_in_4bit", True)
    model = load_base_model(base_model, load_in_4bit=load_in_4bit)

    print(f"Loading Adapter: {adapter_path}")
    if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"WARNING: Adapter not found at {adapter_path}. Running with Base Model only.")
    else:
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = load_tokenizer(base_model)
    instruction = args.prompt or "Explain LoRA fine-tuning in one short paragraph."
    print(f"Input Prompt: {instruction}")

    full_prompt = format_system_prompt(
        system_prompt, instruction, "", tokenizer=tokenizer, generate=True
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(0)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    print("-" * 50)
    print(response)
    print("-" * 50)

    with open("verification_output.txt", "w", encoding="utf-8") as f:
        f.write(response)

    print("VERIFICATION DONE")


if __name__ == "__main__":
    main()
