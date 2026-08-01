from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.paths import LLAMA_CPP_DIR, ROOT
from src.shared.logging import setup_logging

logger = setup_logging("Export")

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
LLAMA3_EOT_ID = 128009


def merge_model(config: dict[str, Any]) -> str:
    """Merge LoRA adapter into base model and patch Llama 3 EOS token."""
    base_model_id = config["model"]["base_model"]
    adapter_path = config["export"]["adapter_path"]
    merged_dir = config["export"]["merged_path"]

    if os.path.exists(merged_dir):
        logger.info("Removing existing merged model at %s", merged_dir)
        shutil.rmtree(merged_dir)

    logger.info("Loading base model: %s", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path, device_map="cpu")
    model = model.merge_and_unload()

    logger.info("Saving merged model to %s", merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    _patch_llama3_eos(merged_dir)
    logger.info("Merge complete")
    return merged_dir


def _patch_llama3_eos(merged_dir: str) -> None:
    config_json_path = os.path.join(merged_dir, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["eos_token_id"] = LLAMA3_EOT_ID
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    tok_conf_path = os.path.join(merged_dir, "tokenizer_config.json")
    if os.path.exists(tok_conf_path):
        with open(tok_conf_path, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        tdata["eos_token"] = "<|eot_id|>"
        with open(tok_conf_path, "w", encoding="utf-8") as f:
            json.dump(tdata, f, indent=2)


def ensure_llama_cpp() -> Path:
    """Clone llama.cpp if missing. Never pip-install its requirements (they force CPU torch)."""
    llama_dir = Path(LLAMA_CPP_DIR)
    if not llama_dir.exists():
        logger.info("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(llama_dir)],
            check=True,
            cwd=str(ROOT),
        )

    try:
        import gguf  # noqa: F401
    except ImportError:
        logger.info("Installing gguf (only; keeps project CUDA torch intact)")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gguf"],
            check=True,
        )
    return llama_dir


def convert_to_gguf(config: dict[str, Any]) -> str:
    llama_dir = ensure_llama_cpp()
    merged_dir = config["export"]["merged_path"]
    gguf_file = config["export"]["gguf_filename"]

    os.makedirs(os.path.dirname(os.path.abspath(gguf_file)) or ".", exist_ok=True)
    logger.info("Converting to GGUF: %s", gguf_file)

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    env = os.environ.copy()
    gguf_py = str(llama_dir / "gguf-py")
    env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(convert_script),
        merged_dir,
        "--outfile",
        gguf_file,
        "--outtype",
        "f16",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    logger.info("Conversion complete: %s", gguf_file)
    return gguf_file


def run_export(config: dict[str, Any], *, merge: bool = True, convert: bool = True) -> None:
    if merge:
        merge_model(config)
    if convert:
        if not os.path.exists(config["export"]["merged_path"]):
            raise FileNotFoundError(
                f"{config['export']['merged_path']} not found. Run merge first."
            )
        convert_to_gguf(config)
