#!/usr/bin/env bash
# Activate pyenv 3.11.9 + Poetry for this project (Linux / macOS)
# Usage: source ./activate.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY_VER="${PY_VER:-3.11.9}"
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found. Install: https://github.com/pyenv/pyenv" >&2
  return 1 2>/dev/null || exit 1
fi

eval "$(pyenv init -)"
pyenv install -s "$PY_VER"
pyenv local "$PY_VER"

echo "Python : $(python --version)  ($(command -v python))"
echo "Poetry : $(poetry --version 2>/dev/null || echo 'missing — pipx install poetry')"
if [[ -x .venv/bin/python ]]; then
  echo "Venv   : .venv  ($(.venv/bin/python --version 2>&1))"
else
  echo "Venv   : missing — run: make setup"
fi
echo ""
echo "Commands: make setup | make up | make train | make down"
