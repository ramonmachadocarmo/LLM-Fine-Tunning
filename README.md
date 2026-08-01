# LLM Fine-Tuning Engine

[![CI](https://github.com/ramonmachadocarmo/LLM-Fine-Tunning/actions/workflows/ci.yml/badge.svg)](https://github.com/ramonmachadocarmo/LLM-Fine-Tunning/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-Poetry-60A5FA.svg)](https://python-poetry.org/)
[![PyTorch CUDA 12.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A7EA4.svg)](tests/)

**Repo:** [ramonmachadocarmo/LLM-Fine-Tunning](https://github.com/ramonmachadocarmo/LLM-Fine-Tunning)

Generic **QLoRA / LoRA** fine-tuning engine: train adapters from JSONL, merge + export **GGUF**, and validate via **Ollama** — from CLI or a local web UI.

## Objective

Ship a **domain-agnostic** pipeline so you can fine-tune open LLMs on your own instruction/output data without wiring Hugging Face + PEFT + bitsandbytes + llama.cpp + Ollama by hand.

Typical loop:

1. Prepare JSONL (`instruction` / `output`)
2. Configure base model + hyperparameters (YAML or UI)
3. Train LoRA/QLoRA adapters
4. Merge + convert to GGUF
5. Chat / smoke-test with Ollama

## Features

| Area | What you get |
|------|----------------|
| **Training** | QLoRA (4-bit) or LoRA, PEFT + TRL, checkpoints, CLI overrides (`epochs`, `batch_size`, `learning_rate`) |
| **Data** | Multi-file JSONL, mixed schemas tolerated, train/val split, sample generator + prune helper |
| **Config** | YAML templates, UI-saved configs under `configs/ui/`, form → config builder |
| **Export** | Merge adapter into base → GGUF via `llama.cpp` (cloned on first export) |
| **Web UI** | Tabs: Configure → Train → Export → Chat; job runner, live logs, progress, asset browser; UI in **pt-BR / en / es** |
| **Chat** | List Ollama models, register GGUF, chat with system prompt |
| **Ops** | OS-agnostic `Makefile` + `activate.sh` / `activate.ps1`, `fix-torch`, port cleanup |
| **Tests** | Unit tests for config, schemas, checkpoints, prompt format, API health/templates |

## Requirements

- NVIDIA GPU + CUDA 12.1+ (Linux / Windows) for training
- Python **3.11.9** via [pyenv](https://github.com/pyenv/pyenv) / [pyenv-win](https://github.com/pyenv-win/pyenv-win)
- [Poetry](https://python-poetry.org/)
- `make`, `curl`
- [Ollama](https://ollama.com) (Chat tab / GGUF smoke tests)

macOS: fine for UI/export tooling; run QLoRA training on a Linux/Windows GPU host.

## Setup

### Windows (PowerShell)

```powershell
. .\activate.ps1
make setup
make check
make test
```

### Linux / macOS

```bash
source ./activate.sh
make setup
make check
make test
```

### Local artifacts (gitignored)

| Path | What to put there |
|------|-------------------|
| `configs/default.yaml` | Copy from template |
| `data/*.jsonl` | `instruction` + `output` rows |
| `merged_models/` | Optional local HF base (or HF ID in config) |
| `adapters/` | Created by training |
| `models/*.gguf` | Created by export |
| `llama.cpp/` | Auto-cloned on first export |
| `logs/` | UI job logs |

```bash
# Linux / macOS
cp configs/default.template.yaml configs/default.yaml
cp data/sample.jsonl data/train.jsonl
# or
make generate
```

```powershell
# Windows
copy configs\default.template.yaml configs\default.yaml
copy data\sample.jsonl data\train.jsonl
make generate
```

If CUDA torch gets overwritten:

```bash
make fix-torch
make check
```

## Web UI

```bash
# after activate.ps1 / activate.sh
make up
# http://127.0.0.1:7860
```

| Tab | Role |
|-----|------|
| **1 Configure** | Base model, datasets, system prompt, download config/dataset templates |
| **2 Train** | Start/stop jobs, hyperparameters, live logs + progress |
| **3 Export** | Merge adapter → GGUF |
| **4 Chat** | Ollama status, register GGUF, chat |

## CLI pipeline

| Step | Command |
|------|---------|
| Sample dataset | `make generate` |
| Balance | `make prune` |
| Train | `make train` |
| Verify | `make verify` |
| Export GGUF | `make export` |
| Adapter chat | `make chat` |
| Unit tests | `make test` |

Default config: `configs/default.yaml` (`CONFIG=path/to.yaml` to override).

## Tests

Unit tests live in `tests/` (no GPU required for the suite).

```bash
make test
# or
poetry run pytest
```

Coverage focus:

- `src/config` — `safe_name`, normalize, form builder, path guards
- `src/shared/checkpoints` — latest epoch selection
- `src/training/dataset` — Llama 3 chat formatting
- `web/application` — preferred config, sample JSONL
- `web/api` — health + template downloads, Pydantic schemas

CI runs the same pytest suite on **Python 3.11.9** with the Poetry lock (including the `torch 2.5.1+cu121` wheel). Runners have no GPU — unit tests must not require CUDA.

## Dataset format

```json
{"instruction": "...", "output": "..."}
```

## Ollama after export

UI **Chat** tab, or:

```bash
# edit FROM in Modelfile.example
ollama create my-ft -f Modelfile.example
ollama run my-ft "Explain LoRA in one paragraph."
```

## Versioned layout

```
configs/default.template.yaml
data/sample.jsonl
generators/sample_dataset.py
src/ web/ scripts/ tests/
activate.sh / activate.ps1 / Makefile / pyproject.toml
.github/workflows/ci.yml
Modelfile.example
docs/ARCHITECTURE.md
LICENSE
```

## Make

| Target | Action |
|--------|--------|
| `make setup` | pyenv + Poetry + CUDA torch |
| `make fix-torch` | Restore torch 2.5.1+cu121 |
| `make check` | Validate Python / CUDA |
| `make test` | Run pytest |
| `make up` / `down` | Start / stop UI |
| `make train` / `export` / `verify` / `chat` | Pipeline |
| `make generate` / `prune` | Dataset helpers |

Portable helpers: `scripts/dev_helpers.py` (`dirs`, `free-port`, `fix-torch`).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
