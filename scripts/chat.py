"""Interactive chat against a fine-tuned adapter (or base model)."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401
from src.config import load_yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class AdapterChat:
    def __init__(self, config_path: str) -> None:
        self.config = load_yaml(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history: list[dict[str, str]] = []
        self.system_prompt = self.config.get(
            "system_prompt",
            "You are a helpful assistant. Follow the user's instructions carefully.",
        )
        self.root_instruction: str | None = None
        self.last_interaction: dict[str, str] | None = None
        self._load_model()

    def _load_model(self) -> None:
        base_model_name = self.config["model"]["base_model"]
        adapter_path = self.config["project"]["output_dir"]
        print(f"Loading model: {base_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0} if self.device == "cuda" else "auto",
            dtype=torch.float16,
        )

        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            print(f"Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            print("No adapter found. Using base model only.")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat_loop(self) -> None:
        print("\n" + "=" * 60)
        print("ADAPTER CHAT")
        print(" commands: /save | /correct <text> | /clear | /exit")
        print("=" * 60 + "\n")

        self.history = [{"role": "system", "content": self.system_prompt}]

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"/exit", "quit", "exit"}:
                break
            if user_input.lower() == "/clear":
                self.history = [{"role": "system", "content": self.system_prompt}]
                self.root_instruction = None
                print("History cleared.")
                continue
            if user_input.lower() == "/save":
                self._save_last()
                continue
            if user_input.lower().startswith("/correct "):
                self._correct_last(user_input[9:].strip())
                continue
            if not user_input:
                continue

            if self.root_instruction is None:
                self.root_instruction = user_input

            self.history.append({"role": "user", "content": user_input})
            generated = self._generate()
            self.history.append({"role": "assistant", "content": generated})
            self.last_interaction = {"instruction": user_input, "output": generated}
            print("\n--- /save to dataset | /correct <text> to fix ---")

    def _generate(self) -> str:
        prompt_str = ""
        for msg in self.history:
            prompt_str += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        prompt_str += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            min_new_tokens=10,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        Thread(target=self.model.generate, kwargs=kwargs).start()

        generated = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            generated += chunk
        print()
        return generated

    def _save_last(self) -> None:
        if not self.last_interaction:
            print("Nothing to save yet.")
            return
        entry = {
            "instruction": self.root_instruction or self.last_interaction["instruction"],
            "output": self.last_interaction["output"],
            "source": "correction_chat",
        }
        os.makedirs("data", exist_ok=True)
        with open("data/corrections.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print("Saved to data/corrections.jsonl")

    def _correct_last(self, correction: str) -> None:
        if not self.last_interaction or not correction:
            print("Nothing to correct yet.")
            return
        entry = {
            "instruction": self.root_instruction or self.last_interaction["instruction"],
            "output": correction,
            "source": "correction_chat",
        }
        os.makedirs("data", exist_ok=True)
        with open("data/corrections.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.last_interaction["output"] = correction
        print("Correction saved to data/corrections.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive adapter chat")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    AdapterChat(args.config).chat_loop()


if __name__ == "__main__":
    main()
