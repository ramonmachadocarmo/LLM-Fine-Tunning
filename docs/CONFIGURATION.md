# Configuration reference

Complete reference for **YAML configs**, **datasets**, **eval sets**, **Makefile / CLI flags**, and related files. Defaults match [`configs/default.template.yaml`](../configs/default.template.yaml).

## Quick start

```bash
cp configs/default.template.yaml configs/default.yaml   # Linux/macOS
# copy configs\default.template.yaml configs\default.yaml   # Windows

cp data/sample.jsonl data/train.jsonl
# edit configs/default.yaml — then:
make train
make eval
```

Local YAML under `configs/` (except `*.template.yaml`) is **gitignored**. Prefer editing a copy of the template, or save from the UI into `configs/ui/`.

---

## Config YAML (`configs/*.yaml`)

Top-level sections: `project`, `model`, `training`, `system_prompt`, `export`.

### Full example

```yaml
project:
  name: my_experiment
  output_dir: ./adapters/my_experiment

model:
  base_model: meta-llama/Llama-3.2-3B-Instruct
  load_in_4bit: true

training:
  dataset_paths:
    - data/train.jsonl
  epochs: 2
  batch_size: 2
  gradient_accumulation_steps: 8
  save_strategy: steps
  save_steps: 50
  learning_rate: 2.0e-4
  max_seq_length: 1024
  # max_steps: 20          # optional smoke-test stop

system_prompt: |
  You are a helpful assistant. Follow the user's instructions carefully.
  Respond with the format requested in the prompt (plain text or JSON when asked).

export:
  adapter_path: ./adapters/my_experiment
  merged_path: ./models/my_experiment_merged
  gguf_filename: ./models/my_experiment.gguf
```

### `project`

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `name` | string | `my_experiment` | Labels this run. UI, safe filenames, and derived paths (`adapters/`, export names) come from it. Prefer `[a-z0-9_-]`. Changing it later does **not** rename folders already on disk. |
| `output_dir` | path | `./adapters/<name>` | Folder where LoRA adapters and checkpoints are written. Resume / incremental train looks here. |

```yaml
project:
  name: support_bot_v1
  output_dir: ./adapters/support_bot_v1
```

### `model`

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `base_model` | string | `meta-llama/Llama-3.2-3B-Instruct` | Which weights to fine-tune. Hugging Face id **or** local HF folder (`config.json` + tokenizer + safetensors). **Not** GGUF / Ollama / `*-GGUF` repos. Larger models (e.g. Gemma 9B) need more VRAM and take much longer per step. Training validates the ref and logs a checksum before load. |
| `load_in_4bit` | bool | `true` | `true` = QLoRA: base stays 4-bit (less VRAM, slightly slower/less precise). `false` = LoRA on full-precision base (more VRAM, often a bit higher quality). |

**Gated Hub models** (Gemma, Llama, …): accept the license on the model page, then set `HF_TOKEN` (UI under Base model, or `.env` / process env). See `.env.example`. Never commit the token.

```yaml
model:
  base_model: meta-llama/Llama-3.2-1B-Instruct   # smaller smoke test
  load_in_4bit: true
```

```yaml
model:
  base_model: ./merged_models/my_local_base
  load_in_4bit: false
```

### `training`

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `dataset_paths` | list[string] | `["data/train.jsonl"]` | JSONL files to train on. More rows → more steps per epoch → longer wall time. Only `instruction` + `output` are kept. |
| `dataset_path` | string \| list | — | **Legacy** alias. Normalized to `dataset_paths` on load. Same effect as listing those files above. |
| `epochs` | int | `2` | How many full passes over the dataset. `1` = see every example once; `2` ≈ twice the time. Extra epochs can help a bit, then overfit. Ignored if `max_steps` is reached first. |
| `batch_size` | int | `2` | Examples processed together on the GPU each micro-step. Higher = faster and more VRAM. OOM → lower this (often to `1`). |
| `gradient_accumulation_steps` | int | `8` | How many micro-batches to stack before an optimizer update. Effective batch ≈ `batch_size × gradient_accumulation_steps`. Raise this (not `batch_size`) when VRAM is tight but you still want a larger effective batch. |
| `learning_rate` | float | `2.0e-4` | How big each LoRA weight update is. Too high → loss spikes / unstable; too low → learns slowly. Typical LoRA range `1e-4`–`2e-4`. |
| `max_seq_length` | int | `1024` | Max tokens per example (prompt + answer). Longer context uses more VRAM and time; text past this is truncated. |
| `save_strategy` | string | `steps` | When to write checkpoints: `steps` (every `save_steps`) or `epoch` (end of each epoch). More frequent saves = safer resume, more disk I/O. |
| `save_steps` | int | `50` | If `save_strategy: steps`, write a checkpoint every N optimizer steps (e.g. `50` ≈ a snapshot often enough to resume after a crash). |
| `max_steps` | int \| null | omitted | Hard stop after this many optimizer steps. Use for smoke tests (`20`–`50`). When set, training may end before all `epochs` finish. |

```yaml
training:
  dataset_paths:
    - data/train.jsonl
    - data/extra.jsonl
  epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  max_seq_length: 512
  save_strategy: steps
  save_steps: 25
  max_steps: 30
```

**Effective batch size example:** `batch_size: 2` and `gradient_accumulation_steps: 8` → ~16 examples per optimizer step.

**Wall time:** steps per epoch ≈ dataset size ÷ `batch_size`. Total steps ≈ that × `epochs` (or `max_steps` if smaller). A 9B QLoRA run at ~28 s/step and 2700 steps/epoch is ~21 h per epoch.

### `system_prompt`

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `system_prompt` | string | short assistant prompt | System turn prepended to every training example (and should match Chat / Modelfile). Teaches role + output style (e.g. “JSON only”). Changing it without retraining will not change the adapter. |

```yaml
system_prompt: |
  You are a JSON-only API. Always answer with a single JSON object.
```

### `export`

Used by `make export` / UI Export tab after training.

| Key | Type | Default | What it does |
|-----|------|---------|--------------|
| `adapter_path` | path | same as `project.output_dir` | Which trained LoRA folder to merge into the base at export time. |
| `merged_path` | path | `./models/<name>_merged` | Where the full merged Hugging Face weights are written (intermediate before GGUF). |
| `gguf_filename` | path | `./models/<name>.gguf` | Final GGUF path for Ollama / llama.cpp. |

```yaml
export:
  adapter_path: ./adapters/support_bot_v1
  merged_path: ./merged_models/support_bot_v1
  gguf_filename: ./models/support_bot_v1.gguf
```

---

## Training dataset JSONL (`data/*.jsonl`)

Each line is one JSON object.

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | yes | User prompt / task. |
| `output` | yes | Target assistant completion. |
| other columns | no | Dropped on load (e.g. optional `input`). |

```json
{"instruction": "Explain LoRA in one short paragraph.", "output": "LoRA freezes the base weights and trains small low-rank adapters..."}
```

```json
{"instruction": "Return JSON {\"status\":\"ok\"}.", "output": "{\"status\":\"ok\"}"}
```

Tracked sample: [`data/sample.jsonl`](../data/sample.jsonl). Copy to `data/train.jsonl` or run `make generate`.

Multi-file tip: list several paths under `training.dataset_paths`; files are loaded separately then concatenated so mixed optional columns do not break casting.

---

## Eval dataset JSONL (`data/eval*.jsonl`)

Used by `make eval` / `scripts/eval.py`. See also [EVAL.md](EVAL.md).

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | yes | Prompt sent to the model. |
| `expected` | yes | Score signal. Alias: `output`. |
| `match` | no | `contains` (default), `exact`, or `json`. |
| `id` | no | Stable case id (default `case-<line>`). |

```json
{"id": "lora-brief", "instruction": "Explain LoRA fine-tuning in one short paragraph.", "expected": "LoRA", "match": "contains"}
```

```json
{"id": "exact-ack", "instruction": "Reply with exactly: ACK", "expected": "ACK", "match": "exact"}
```

```json
{"id": "json-student", "instruction": "Return JSON with keys name and score for Alice / 92.", "expected": "{\"name\": \"Alice\", \"score\": 92}", "match": "json"}
```

| `match` | Behavior |
|---------|----------|
| `contains` | Case-insensitive substring check of `expected` inside the prediction. |
| `exact` | Trimmed full-string equality. |
| `json` | Prediction must be valid JSON; if `expected` is JSON, required fields/values must match. |

Tracked sample: [`data/eval.sample.jsonl`](../data/eval.sample.jsonl).

Offline scoring without a GPU (canned predictions):

```bash
poetry run python scripts/eval.py --eval-set data/eval.sample.jsonl --predictions preds.json
```

`preds.json` shape:

```json
{
  "lora-brief": "LoRA adapts models by training small low-rank matrices...",
  "exact-ack": "ACK"
}
```

---

## Makefile variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `configs/default.yaml` | Training / eval / export / chat config path. |
| `EVAL_SET` | `data/eval.sample.jsonl` | Eval JSONL for `make eval`. |
| `EVAL_TARGET` | `adapter` | `adapter` or `base`. |
| `HOST` | `127.0.0.1` | UI bind host. |
| `PORT` | `7860` | UI port. |
| `PY_VER` | `3.11.9` | pyenv Python version for `make setup`. |
| `POETRY` | `poetry` | Poetry binary name/path. |
| `NO_COLOR` | unset | Set to `1` to disable colored `make` help. |

```bash
make train CONFIG=configs/ui/smoke.yaml
make eval EVAL_TARGET=base EVAL_SET=data/eval.sample.jsonl
make up HOST=0.0.0.0 PORT=8080
```

Common targets: `make doctor`, `setup`, `up`, `train`, `eval`, `export`, `verify`, `chat`, `test`. Run `make` for the full list.

---

## CLI overrides (`scripts/train.py` via `get_config`)

When calling the train entrypoint, these flags override YAML:

| Flag | Overrides |
|------|-----------|
| `--config PATH` | Config file (default `configs/default.yaml`) |
| `--epochs N` | `training.epochs` |
| `--batch_size N` | `training.batch_size` |
| `--learning_rate F` | `training.learning_rate` |

```bash
poetry run python scripts/train.py --config configs/default.yaml --epochs 1 --batch_size 1
```

### `scripts/eval.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/default.yaml` | Model / adapter paths. |
| `--eval-set` | `data/eval.sample.jsonl` | Cases JSONL. |
| `--target` | `adapter` | `base` or `adapter`. |
| `--max-new-tokens` | `256` | Generation length cap. |
| `--predictions` | — | JSON map to score without loading a model. |
| `--out-dir` | `logs/eval` | Report directory. |

Reports: `logs/eval/*.json` + `*.md`.

### `scripts/export.py`

| Flag | Description |
|------|-------------|
| `--config` | YAML with `export.*` and `model.base_model`. |
| `--merge` | Merge adapter into base. |
| `--convert` | Convert merged weights to GGUF. |
| `--all` | Merge + convert (`make export`). |

### `scripts/verify.py` / `scripts/chat.py`

| Flag | Description |
|------|-------------|
| `--config` | Same training YAML (`project.output_dir`, `model`, `system_prompt`). |
| `--prompt` | (`verify` only) Single smoke prompt. |

---

## Web UI / API payloads

The UI posts JSON shaped like `TrainRequest` ([`web/api/schemas.py`](../web/api/schemas.py)). Field names map onto the YAML sections above.

| UI / JSON field | YAML destination |
|-----------------|------------------|
| `project_name` | `project.name` |
| `output_dir` | `project.output_dir` |
| `base_model` | `model.base_model` |
| `load_in_4bit` | `model.load_in_4bit` |
| `dataset_paths` | `training.dataset_paths` |
| `epochs`, `batch_size`, … | `training.*` |
| `max_steps` | `training.max_steps` (optional) |
| `system_prompt` | `system_prompt` |
| `adapter_path`, `merged_path`, `gguf_filename` | `export.*` |
| `save_config_as` | Filename under `configs/ui/` |
| `start_training` | `true` starts a job; `false` save-only |

Export job body:

```json
{"config_path": "configs/ui/my_experiment.yaml"}
```

Ollama chat / register:

```json
{"model": "my-ft", "messages": [{"role": "user", "content": "Hello"}], "system_prompt": "..."}
```

```json
{"gguf_path": "models/my_experiment.gguf", "model_name": "my-ft", "system_prompt": "..."}
```

---

## Ollama `Modelfile.example`

After `make export`, point `FROM` at your GGUF and create a model:

```bash
# edit FROM in Modelfile.example
ollama create my-ft -f Modelfile.example
ollama run my-ft "Explain LoRA in one paragraph."
```

| Directive | Meaning |
|-----------|---------|
| `FROM` | Path to `.gguf` |
| `TEMPLATE` | Chat template (Llama or Gemma; UI Register detects family) |
| `SYSTEM` | Default system prompt (keep close to YAML `system_prompt`) |
| `PARAMETER stop` | Stop tokens |
| `PARAMETER temperature` / `top_p` | Sampling |

---

## Important paths

| Path | Role |
|------|------|
| `configs/default.template.yaml` | Tracked template |
| `configs/default.yaml` | Local default (gitignored) |
| `configs/ui/*.yaml` | UI-saved configs (gitignored) |
| `data/sample.jsonl` | Tracked training sample |
| `data/eval.sample.jsonl` | Tracked eval sample |
| `data/*.jsonl` | Your datasets (gitignored except samples) |
| `adapters/` | Training outputs |
| `merged_models/` | Optional local HF bases / merges |
| `models/*.gguf` | Exported GGUF |
| `logs/jobs/` | UI job logs |
| `logs/eval/` | Eval reports |
| `llama.cpp/` | Cloned on first export (gitignored) |
| `.python-version` | pyenv pin (`3.11.9`) |
| `install.ps1` / `install.sh` | Host dependency bootstrap |

---

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — layout and flow
- [EVAL.md](EVAL.md) — eval metrics and base vs adapter vs GGUF
- [TODO.md](TODO.md) — roadmap
- [README.md](../README.md) — setup and pipeline overview
