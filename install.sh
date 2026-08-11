#!/usr/bin/env bash
# Check and install host dependencies for LLM Fine-Tuning Engine (Linux / macOS).
# Usage: ./install.sh [--setup]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY_VER="${PY_VER:-3.11.9}"
DO_SETUP=0
for arg in "$@"; do
  case "$arg" in
    --setup|-s) DO_SETUP=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--setup]"
      echo "  Checks make, curl, git, pyenv, Poetry, Python ${PY_VER}."
      echo "  --setup  also runs: make setup"
      exit 0
      ;;
  esac
done

ok()   { printf "  [OK]  %s\n" "$*"; }
warn() { printf "  [!!]  %s\n" "$*"; }
fail() { printf "  [XX]  %s\n" "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> LLM Fine-Tuning Engine · dependency check"

MISSING=0

if have make; then ok "make $(make --version 2>/dev/null | head -1)"; else fail "make not found"; MISSING=1; fi
if have curl; then ok "curl"; else fail "curl not found"; MISSING=1; fi
if have git; then ok "git $(git --version 2>/dev/null | head -1)"; else fail "git not found"; MISSING=1; fi

# --- pyenv ---
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

if ! have pyenv; then
  warn "pyenv missing — installing via official installer"
  if have curl; then
    curl -fsSL https://pyenv.run | bash
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
  else
    fail "cannot install pyenv without curl"
    MISSING=1
  fi
fi

if have pyenv; then
  ok "pyenv $(pyenv --version 2>/dev/null | head -1)"
  eval "$(pyenv init -)" || true
  if pyenv versions --bare 2>/dev/null | grep -qx "$PY_VER"; then
    ok "Python $PY_VER already installed via pyenv"
  else
    warn "installing Python $PY_VER via pyenv (may take a few minutes)"
    pyenv install -s "$PY_VER"
  fi
  pyenv local "$PY_VER"
  ok "pyenv local -> $PY_VER"
else
  fail "pyenv still unavailable"
  MISSING=1
fi

# --- Poetry ---
if have poetry; then
  ok "poetry $(poetry --version 2>/dev/null)"
else
  warn "Poetry missing — installing"
  if have curl; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
  else
    fail "cannot install Poetry without curl"
    MISSING=1
  fi
  if have poetry; then
    ok "poetry $(poetry --version 2>/dev/null)"
  else
    fail "Poetry install finished but 'poetry' not on PATH (add ~/.local/bin)"
    MISSING=1
  fi
fi

if [[ "$MISSING" -ne 0 ]]; then
  echo ""
  echo "Some required tools are missing. Install them and re-run ./install.sh"
  exit 1
fi

echo ""
echo "==> host deps OK"
echo "    Next: make setup && make up"
echo "    Or:   ./install.sh --setup"

if [[ "$DO_SETUP" -eq 1 ]]; then
  echo ""
  echo "==> make setup"
  make setup
fi
