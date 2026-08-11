from __future__ import annotations

import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.shared.logging import setup_logging

logger = setup_logging("ModelLoader")


def load_tokenizer(model_name: str):
    from src.shared.hf_auth import apply_hf_token_to_environ

    apply_hf_token_to_environ()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str, load_in_4bit: bool = True):
    """Load base causal LM with optional 4-bit QLoRA prep."""
    from src.shared.hf_auth import apply_hf_token_to_environ

    apply_hf_token_to_environ()
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA indisponível (torch={torch.__version__}). "
            "Install CUDA torch: make setup (or make fix-torch)"
        )

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model


def setup_lora_model(
    model,
    output_dir: str,
    resume_checkpoint: str | None = None,
    load_existing_adapter: bool = False,
):
    """Attach LoRA: resume checkpoint, incremental adapter, or fresh config."""
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
        return model, "resumed"

    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    if load_existing_adapter and os.path.exists(adapter_path):
        logger.info("Incremental training from adapter: %s", output_dir)
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
        return model, "incremental"

    logger.info("Initializing fresh LoRA adapters")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model, "fresh"
