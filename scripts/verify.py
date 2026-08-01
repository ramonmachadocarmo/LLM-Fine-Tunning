import argparse
import os

import torch
from peft import PeftModel

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from src.config import load_yaml
from src.training.model import load_base_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Verify Fine-Tuned Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    args, _unknown = parser.parse_known_args()

    config = load_yaml(args.config)
    
    BASE_MODEL = config["model"]["base_model"]
    ADAPTER_PATH = config["project"]["output_dir"]
    SYSTEM_PROMPT = config.get("system_prompt", "You are a helpful assistant.")
    
    print(f"Loading Base Model: {BASE_MODEL}")
    # Load 4bit if configured
    load_in_4bit = config["model"].get("load_in_4bit", True)
    model = load_base_model(BASE_MODEL, load_in_4bit=load_in_4bit)
    
    print(f"Loading Adapter: {ADAPTER_PATH}")
    # Load Adapter
    # Check if adapter exists
    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"WARNING: Adapter not found at {ADAPTER_PATH}. Running with Base Model only.")
    else:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    tokenizer = load_tokenizer(BASE_MODEL)
    
    # Determine Instruction
    if args.prompt:
        instruction = args.prompt
    else:
        instruction = "Explain LoRA fine-tuning in one short paragraph."

    print(f"Input Prompt: {instruction}")

    # Format Prompt
    # We can reuse src.dataset.format_system_prompt but we need to handle the output part.
    # format_system_prompt expects (sys, instr, output). Here output is what we want to generate.
    # Llama 3 format for inference ends with <|start_header_id|>assistant<|end_header_id|>\n\n
    
    full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(0)
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    assistant_response = response
    if "assistant<|end_header_id|>" in response:
        # The decode might strip special tokens, so we might not see the header tag exactly like that depending on tokenizer.
        # But usually skip_special_tokens=True removes them. 
        # If we skipped special tokens, the tags are gone.
        # So we should rely on the text content. Llama 3 format is tricky with skip_special_tokens=True.
        # Let's decode properly.
        pass
        
    # Re-decode without skipping special tokens to find the split point, then decode the rest?
    # Or just print everything.
    # For now, let's just print the raw response if it's clean enough, or try to split by the prompt end.
    
    # Simple split approximation if tags are removed:
    # The system prompt and user prompt will be at the start.
    # We can just check what comes after the known prompt text or just print it all for debug.
    
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    with open("verification_output.txt", "w", encoding="utf-8") as f:
        f.write(response)
        
    print("VERIFICATION DONE")

if __name__ == "__main__":
    main()
