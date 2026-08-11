# LLM Fine-Tuning Engine - pyenv 3.11.9 + Poetry (Windows / Linux / macOS)
# First time:  ./install.sh   OR   .\install.ps1
# Then:        make setup && make up

HOST ?= 127.0.0.1
PORT ?= 7860
CONFIG ?= configs/default.yaml
PY_VER ?= 3.11.9
POETRY ?= poetry
EVAL_SET ?= data/eval.sample.jsonl
EVAL_TARGET ?= adapter

ifeq ($(OS),Windows_NT)
  PYENV_ROOT ?= $(USERPROFILE)/.pyenv/pyenv-win
  PY_HOME := $(PYENV_ROOT)/versions/$(PY_VER)
  export PYENV := $(PYENV_ROOT)
  export PYENV_ROOT := $(PYENV_ROOT)
  export PATH := $(PYENV_ROOT)/bin;$(PYENV_ROOT)/shims;$(PY_HOME);$(PY_HOME)/Scripts;$(PATH)
  PYTHON_BIN := $(PY_HOME)/python.exe
else
  PYENV_ROOT ?= $(HOME)/.pyenv
  export PYENV_ROOT := $(PYENV_ROOT)
  export PATH := $(PYENV_ROOT)/shims:$(PYENV_ROOT)/bin:$(PATH)
  PYTHON_BIN := $(shell command -v python3 2>/dev/null || command -v python)
endif

ifeq ($(wildcard $(PYTHON_BIN)),)
  HELPER_PY := python
else
  HELPER_PY := $(PYTHON_BIN)
endif

RUN = $(POETRY) run python
HELPER_SCRIPT := scripts/dev_helpers.py
BOOT = "$(HELPER_PY)" $(HELPER_SCRIPT)

ifeq ($(wildcard $(HELPER_SCRIPT)),)
  $(error $(HELPER_SCRIPT) is missing - incomplete checkout)
endif

.PHONY: help env up down setup check fix-torch dirs ui generate prune train verify eval chat export status clean test doctor

.DEFAULT_GOAL := help

help:
	@$(BOOT) help --py-ver "$(PY_VER)" --host "$(HOST)" --port "$(PORT)" --config "$(CONFIG)" --eval-set "$(EVAL_SET)" --eval-target "$(EVAL_TARGET)"

env: doctor

doctor:
	@$(BOOT) doctor --pyenv-root "$(PYENV_ROOT)" --python-bin "$(PYTHON_BIN)"

setup:
	@echo ">>> pyenv $(PY_VER)"
	pyenv install -s $(PY_VER)
	pyenv local $(PY_VER)
	@echo ">>> poetry env -> $(PYTHON_BIN)"
	$(POETRY) env use "$(PYTHON_BIN)"
	@echo ">>> poetry lock + sync"
	$(POETRY) lock
	$(POETRY) sync
	@$(MAKE) fix-torch
	@$(MAKE) check

fix-torch:
	@echo ">>> restore torch CUDA"
	$(BOOT) fix-torch

check:
	@echo ">>> check"
	@$(RUN) -c "import sys; print('python', sys.version.split()[0], sys.executable)"
	@$(RUN) -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count());\
import sys; sys.exit(0 if ('+cu' in torch.__version__ and torch.cuda.is_available()) else 1)" || (echo "ERROR: torch without CUDA. Run: make fix-torch" && exit 1)
	@$(RUN) -c "from transformers import AutoConfig; print('transformers ok')" || (echo "ERROR: transformers broken. Run: make fix-torch" && exit 1)

test:
	@echo ">>> pytest"
	$(POETRY) run pytest

up: down dirs
	@echo ">>> UI -> http://$(HOST):$(PORT)"
	$(RUN) -c "import uvicorn; uvicorn.run('web.app:app', host='$(HOST)', port=$(PORT), reload=False)"

down:
	@echo ">>> freeing port $(PORT)..."
	-$(BOOT) free-port $(PORT)

dirs:
	-$(BOOT) dirs

status:
	-@curl -s http://$(HOST):$(PORT)/api/health || echo "UI offline"

ui: up

generate: dirs
	$(RUN) generators/sample_dataset.py --output data/train.jsonl --repeat 40

prune:
	$(RUN) scripts/prune.py --input "data/train.jsonl" --output "data/train_balanced.jsonl" --target 5000

train: check
	$(RUN) scripts/train.py --config $(CONFIG)

verify:
	$(RUN) scripts/verify.py --config $(CONFIG) --prompt "Explain LoRA fine-tuning in one short paragraph."

eval: dirs
	$(RUN) scripts/eval.py --config $(CONFIG) --eval-set $(EVAL_SET) --target $(EVAL_TARGET)

chat:
	$(RUN) scripts/chat.py --config $(CONFIG)

export:
	$(RUN) scripts/export.py --config $(CONFIG) --all

clean:
	$(RUN) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(RUN) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
