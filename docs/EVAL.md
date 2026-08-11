# Offline evaluation

Reproducible checks for **base** vs **adapter** (Hugging Face / PEFT). GGUF via Ollama is still a manual smoke path; wire it into the harness later if needed.

## Eval set

Tracked sample: [`data/eval.sample.jsonl`](../data/eval.sample.jsonl)

```json
{"id": "lora-brief", "instruction": "...", "expected": "LoRA", "match": "contains"}
```

| Field | Required | Notes |
|-------|----------|--------|
| `instruction` | yes | Prompt sent to the model |
| `expected` | yes | Signal to score (`output` accepted as alias) |
| `match` | no | `contains` (default), `exact`, or `json` |
| `id` | no | Stable case id (defaults to `case-<line>`) |

**Metrics**

- `exact` — prediction equals `expected` (trimmed)
- `contains` — `expected` appears in prediction (case-insensitive)
- `json` — prediction must be valid JSON; if `expected` is JSON, required fields/values must match

## Run

```bash
# adapter (default) or base — needs GPU + config like training
make eval
make eval EVAL_TARGET=base
make eval EVAL_SET=data/eval.sample.jsonl CONFIG=configs/default.yaml

# score canned predictions without loading a model (unit / offline)
poetry run python scripts/eval.py --eval-set data/eval.sample.jsonl --predictions preds.json
```

`preds.json` shape: `{ "case-id": "model output text", ... }`.

Reports land in `logs/eval/` as paired `.json` + `.md`.

## Compare base vs adapter vs GGUF

Use the **same** eval set and metrics for fair comparison.

| Target | How |
|--------|-----|
| **Base** | `make eval EVAL_TARGET=base` |
| **Adapter** | `make eval EVAL_TARGET=adapter` (loads LoRA from `project.output_dir` in config) |
| **GGUF / Ollama** | Export first (`make export`), register/create the model, then smoke-test the same `instruction`s in the Chat tab or `ollama run …`. Automated GGUF scoring is not in the harness yet. |

Workflow tip:

1. Freeze `data/eval.sample.jsonl` (or your own eval JSONL).
2. Run base → note pass rate in `logs/eval/`.
3. Train adapter → run adapter eval → compare reports.
4. Export GGUF → manually spot-check the failing / critical cases in Ollama.

`make verify` remains a single-prompt smoke test; prefer `make eval` for regressions.
