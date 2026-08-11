"""Project path constants shared by CLI, training, export and web UI."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = ROOT / "configs"
UI_CONFIGS_DIR = CONFIGS_DIR / "ui"
DATA_DIR = ROOT / "data"
ADAPTERS_DIR = ROOT / "adapters"
MERGED_DIR = ROOT / "merged_models"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs" / "jobs"
EVAL_LOGS_DIR = ROOT / "logs" / "eval"
LLAMA_CPP_DIR = ROOT / "llama.cpp"
