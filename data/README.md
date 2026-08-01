# Datasets

Put `.jsonl` files here (gitignored). New files show up in the UI after **Refresh**.

```json
{"instruction": "...", "output": "..."}
```

Tracked sample: `sample.jsonl`

```powershell
copy data\sample.jsonl data\train.jsonl
make generate
```
