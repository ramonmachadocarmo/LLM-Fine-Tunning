# Architecture — LLM Fine-Tuning Engine

## Versioned code

```
configs/default.template.yaml
data/sample.jsonl
generators/sample_dataset.py
src/
  paths.py
  shared/       # logging, checkpoints
  config/       # YAML load/build/save
  training/     # Trainer, model, dataset
  export/       # merge + GGUF
web/            # FastAPI + UI + jobs + Ollama chat
scripts/        # train, export, verify, chat, prune, web_ui, dev_helpers
tests/          # pytest unit tests (config, API, checkpoints, schemas)
Makefile / activate.sh / activate.ps1 / pyproject.toml
.github/workflows/ci.yml
```

## Local / gitignored

```
configs/*.yaml, configs/ui/*.yaml
data/*.jsonl
adapters/
merged_models/
models/*.gguf
llama.cpp/          # cloned on first export
logs/jobs/
.venv/
```

## Flow

1. Configure (`configs/` + `data/`) via UI or YAML
2. `scripts/train.py` → `adapters/`
3. `scripts/export.py` → merge → GGUF in `models/`
4. Chat tab / Ollama → test the GGUF

## Roadmap

See [TODO.md](TODO.md) for planned work beyond fine-tuning (eval, RAG, tracking, preference tuning, serving/safety).
