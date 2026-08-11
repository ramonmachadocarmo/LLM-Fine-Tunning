# Datasets

Put `.jsonl` files here (gitignored). New files show up in the UI after **Refresh**.

## Training

```json
{"instruction": "...", "output": "..."}
```

Tracked sample: `sample.jsonl`

```powershell
copy data\sample.jsonl data\train.jsonl
make generate
```

## Eval

Tracked sample: `eval.sample.jsonl`

```json
{"id": "case-1", "instruction": "...", "expected": "...", "match": "contains"}
```

| `match` | Behavior |
|---------|----------|
| `contains` | Expected substring in prediction (default) |
| `exact` | Full-string equality |
| `json` | Valid JSON; field checks when `expected` is JSON |

```bash
make eval
make eval EVAL_TARGET=base
```

Reports: `logs/eval/`. Full guide: [docs/EVAL.md](../docs/EVAL.md).
