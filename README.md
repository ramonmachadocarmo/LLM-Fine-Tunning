# LLM Fine-Tuning Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-Poetry-60A5FA.svg)](https://python-poetry.org/)
[![PyTorch CUDA 12.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A7EA4.svg)](tests/)
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/ramonmachadocarmo)

**Repo:** [ramonmachadocarmo/LLM-Fine-Tunning](https://github.com/ramonmachadocarmo/LLM-Fine-Tunning)

Generic **QLoRA / LoRA** fine-tuning engine: train adapters from JSONL, run an **offline eval harness**, merge + export **GGUF**, and validate via **Ollama** — from CLI or a local web UI.

## Objective

Ship a **domain-agnostic** pipeline so you can fine-tune open LLMs on your own instruction/output data without wiring Hugging Face + PEFT + bitsandbytes + llama.cpp + Ollama by hand.

Typical loop:

1. Prepare JSONL (`instruction` / `output`)
2. Configure base model + hyperparameters (YAML or UI)
3. Train LoRA/QLoRA adapters
4. Eval base vs adapter (`make eval`)
5. Merge + convert to GGUF
6. Chat / smoke-test with Ollama

## Features

| Area | What you get |
|------|----------------|
| **Training** | QLoRA (4-bit) or LoRA, PEFT + TRL, checkpoints, CLI overrides (`epochs`, `batch_size`, `learning_rate`) |
| **Eval** | Fixed JSONL eval set, exact / contains / JSON metrics, `make eval` reports under `logs/eval/` |
| **Data** | Multi-file JSONL, mixed schemas tolerated, train/val split, sample generator + prune helper |
| **Config** | YAML templates, UI-saved configs under `configs/ui/`, form → config builder |
| **Export** | Merge adapter into base → GGUF via `llama.cpp` (cloned on first export) |
| **Web UI** | Tabs: Configure → Train → Export → Chat; job runner, live logs, progress, asset browser; UI in **pt-BR / en / es** |
| **Chat** | List Ollama models, register GGUF, chat with system prompt |
| **Ops** | OS-agnostic `Makefile` + `install.sh` / `install.ps1`, `fix-torch`, port cleanup |
| **Tests** | Unit tests for config, schemas, checkpoints, prompt format, evaluation metrics, API health/templates |

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
.\install.ps1
make setup
make check
make test
```

### Linux / macOS

```bash
chmod +x install.sh
./install.sh
make setup
make check
make test
```

`install.ps1` / `install.sh` check (and try to install) make, curl, git, pyenv, Poetry, and Python **3.11.9**. Pass `-Setup` / `--setup` to also run `make setup`. Use `make doctor` anytime to print the active toolchain. `make` alone shows the styled help.

### Local artifacts (gitignored)

| Path | What to put there |
|------|-------------------|
| `configs/default.yaml` | Copy from template |
| `data/*.jsonl` | `instruction` + `output` rows (training); see also tracked `eval.sample.jsonl` |
| `merged_models/` | Optional local HF base (or HF ID in config) |
| `adapters/` | Created by training |
| `models/*.gguf` | Created by export |
| `llama.cpp/` | Auto-cloned on first export |
| `logs/jobs/` | UI job logs |
| `logs/eval/` | Offline eval reports (`.json` + `.md`) |

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
# after: ./install.sh  OR  .\install.ps1   then make setup
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
| Eval (base/adapter) | `make eval` / `make eval EVAL_TARGET=base` |
| Verify (single prompt) | `make verify` |
| Export GGUF | `make export` |
| Adapter chat | `make chat` |
| Unit tests | `make test` |

Default config: `configs/default.yaml` (`CONFIG=path/to.yaml` to override). Eval set default: `data/eval.sample.jsonl` (`EVAL_SET=…` to override).

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
- `src/training/chat_format` — tokenizer chat template (Gemma / Llama fallback)
- `src/evaluation` — eval JSONL load, exact/contains/JSON metrics, report writer
- `web/application` — preferred config, sample JSONL
- `web/api` — health + template downloads, Pydantic schemas

## Dataset format

Training:

```json
{"instruction": "...", "output": "..."}
```

Eval (`data/eval.sample.jsonl`):

```json
{"id": "case-1", "instruction": "...", "expected": "...", "match": "contains"}
```

`match`: `contains` (default), `exact`, or `json`. Details: [docs/EVAL.md](docs/EVAL.md).

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
data/eval.sample.jsonl
generators/sample_dataset.py
src/ web/ scripts/ tests/
install.sh / install.ps1 / Makefile / pyproject.toml
Modelfile.example
docs/ARCHITECTURE.md
docs/CONFIGURATION.md
docs/EVAL.md
docs/TODO.md
LICENSE
```

## Make

| Target | Action |
|--------|--------|
| `make doctor` / `make env` | Print toolchain status |
| `make setup` | pyenv + Poetry + CUDA torch |
| `make fix-torch` | Restore torch 2.5.1+cu121 |
| `make check` | Validate Python / CUDA |
| `make test` | Run pytest |
| `make up` / `down` | Start / stop UI |
| `make train` / `eval` / `export` / `verify` / `chat` | Pipeline |
| `make generate` / `prune` | Dataset helpers |

First-time host deps: `.\install.ps1` (Windows) or `./install.sh` (Linux/macOS).

Portable helpers: `scripts/dev_helpers.py` (`dirs`, `free-port`, `fix-torch`).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/CONFIGURATION.md](docs/CONFIGURATION.md), [docs/EVAL.md](docs/EVAL.md), and the roadmap in [docs/TODO.md](docs/TODO.md).

## Support

If this project helps you, consider sponsoring:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub_Sponsors-ea4aaa?style=for-the-badge&logo=github)](https://github.com/sponsors/ramonmachadocarmo)
