# Architecture — LLM Fine-Tuning Engine

## Versioned code

```
configs/default.template.yaml
data/sample.jsonl
data/eval.sample.jsonl
generators/sample_dataset.py
src/
  paths.py
  shared/       # logging, checkpoints
  config/       # YAML load/build/save
  training/     # Trainer, model, dataset
  evaluation/   # eval cases, metrics, harness, reports, HF generate
  export/       # merge + GGUF
web/            # FastAPI + UI + jobs + Ollama chat
scripts/        # train, eval, export, verify, chat, prune, web_ui, dev_helpers
tests/          # pytest (config, API, checkpoints, schemas, evaluation)
Makefile / install.sh / install.ps1 / pyproject.toml
```

## Local / gitignored

```
configs/*.yaml, configs/ui/*.yaml
data/*.jsonl          # tracked exceptions: sample.jsonl, *.sample.jsonl
adapters/
merged_models/
models/*.gguf
llama.cpp/            # cloned on first export
logs/jobs/
logs/eval/
.venv/
```

## Flow

1. Configure (`configs/` + `data/`) via UI or YAML
2. `scripts/train.py` → `adapters/`
3. `scripts/eval.py` / `make eval` → reports in `logs/eval/` (base or adapter)
4. `scripts/export.py` → merge → GGUF in `models/`
5. Chat tab / Ollama → smoke-test the GGUF (same instructions as the eval set when possible)

## Evaluation

- Package: `src/evaluation/`
- Entry: `scripts/eval.py`, `make eval` (`EVAL_TARGET=base|adapter`, `EVAL_SET=…`)
- How to compare targets: [EVAL.md](EVAL.md)

## Roadmap

See [TODO.md](TODO.md) for planned work beyond fine-tuning (eval UI, RAG, tracking, preference tuning, serving/safety).

## Configuration reference

Parameter-by-parameter YAML, datasets, Makefile/CLI flags, and examples: [CONFIGURATION.md](CONFIGURATION.md).
