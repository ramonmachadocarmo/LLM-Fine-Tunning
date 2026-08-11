# Roadmap / TODO

English checklist for work **beyond** the current QLoRA fine-tuning loop.

## Purpose

**Today:** configure → train LoRA/QLoRA → offline eval (base/adapter) → export GGUF → chat via Ollama (CLI + web UI).

**Gap:** eval UI + automated GGUF scoring still open; no RAG path, weak experiment tracking, no preference tuning, limited serving/safety tooling.

Use this file to prioritize the next product slices. Mark items `- [x]` when done and add a line under [Done log](#done-log).

## Priority rule

| Need | Prefer |
|------|--------|
| Stable format / tone / skill | Fine-tune (this repo) |
| Fresh or large factual knowledge | RAG |
| Actions / tools | Agents + tool calling |
| Know if we improved | **Evals (always)** |

## Phases

```text
Eval → RAG → Tracking → Preference (DPO/ORPO) → Serving / safety / multi-adapter
```

---

### Phase 1 — Eval harness

**Goal:** Reproduceable offline checks for base vs adapter vs GGUF.

**Acceptance criteria**

- [x] Fixed eval set (`data/eval.sample.jsonl` or similar) with `instruction` + expected signal
- [x] Metrics: at least exact match **or** containment, plus JSON-validity when output should be JSON
- [x] CLI: `make eval` (and/or `scripts/eval.py`) writing a small report under `logs/`
- [ ] Optional light UI surface (panel or tab) to pick config/model and show last report
- [x] Document how to compare base vs adapter vs Ollama GGUF

**Suggested touchpoints:** [`scripts/eval.py`](../scripts/eval.py), [`src/evaluation/`](../src/evaluation/), [`docs/EVAL.md`](EVAL.md), [`scripts/verify.py`](../scripts/verify.py), [`web/`](../web/), [`Makefile`](../Makefile)

**In progress (branch `feat/phase-1-eval-harness`):** core harness + docs done; UI still open.

---

### Phase 2 — RAG (optional path)

**Goal:** Answer from local documents without another fine-tune when the problem is knowledge, not style.

**Acceptance criteria**

- [ ] Ingest docs (files/folder) → chunks → embeddings store (local)
- [ ] Retrieve top-k + inject into prompt (CLI and/or Chat tab option)
- [ ] Minimal config (paths, chunk size, top-k) in YAML or UI
- [ ] Short doc: when to use RAG vs fine-tuning

**Suggested touchpoints:** new `src/rag/` (or similar), [`web/application/ollama_chat.py`](../web/application/ollama_chat.py), [`configs/`](../configs/)

---

### Phase 3 — Experiment tracking

**Goal:** Every train/export run is discoverable with hparams and artifact paths.

**Acceptance criteria**

- [ ] Persist run metadata: project name, hparams, dataset paths, adapter/GGUF paths, status, timestamps
- [ ] Local backend first (JSON/YAML under `logs/runs/` **or** MLflow tracking URI local)
- [ ] UI jobs list links to run metadata
- [ ] README note on how to browse runs

**Suggested touchpoints:** [`web/application/jobs.py`](../web/application/jobs.py), [`src/training/trainer.py`](../src/training/trainer.py), [`src/shared/logging.py`](../src/shared/logging.py)

---

### Phase 4 — Preference tuning (DPO / ORPO)

**Goal:** Improve preference alignment after a stable SFT adapter.

**Acceptance criteria**

- [ ] Preference dataset format documented + sample file
- [ ] Training entrypoint (CLI + config keys) for DPO or ORPO on top of SFT adapter
- [ ] Same export path: merge → GGUF → Ollama
- [ ] Smoke eval (Phase 1) runnable before/after preference train

**Suggested touchpoints:** [`src/training/`](../src/training/), [`scripts/train.py`](../scripts/train.py), [`configs/default.template.yaml`](../configs/default.template.yaml)

---

### Phase 5 — Serving, safety, multi-adapter

**Goal:** Safer demos and more flexible deployment of adapters.

**Acceptance criteria**

- [ ] Optional extra quant paths where useful (e.g. AWQ/GPTQ) documented; GGUF remains default
- [ ] Basic guardrails (PII redaction and/or JSON schema check) on Chat/eval outputs
- [ ] Multi-adapter or simple domain router (select LoRA by project/tag)
- [ ] Notes on production inference (Ollama vs vLLM) without requiring a cluster in-repo

**Suggested touchpoints:** [`src/export/`](../src/export/), [`web/`](../web/), [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Out of scope / later

- Full human labeling UI
- Mandatory cloud W&B (local tracking first)
- Managed vLLM / multi-node serving
- Full red-team suite (can follow Phase 1 + Phase 5 guardrails)

## Done log

| Date | Item | PR / commit |
|------|------|-------------|
| 2026-08-10 | Phase 1: eval set + metrics + `make eval` / `scripts/eval.py` reports under `logs/eval/` | `feat/phase-1-eval-harness` |
| 2026-08-10 | Phase 1: docs — README, ARCHITECTURE, EVAL.md (base vs adapter vs GGUF) | `feat/phase-1-eval-harness` |

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [EVAL.md](EVAL.md)
- [README.md](../README.md)
