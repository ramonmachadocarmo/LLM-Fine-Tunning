from __future__ import annotations

import os
import re
from pathlib import Path

from src.paths import ROOT

ENV_FILE = ROOT / ".env"
TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def _strip_value(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        return v[1:-1]
    return v


def _read_dotenv(path: Path | None = None) -> dict[str, str]:
    target = path or ENV_FILE
    if not target.is_file():
        return {}
    out: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        out[m.group(1)] = _strip_value(m.group(2))
    return out


def _write_dotenv(values: dict[str, str], path: Path | None = None) -> None:
    target = path or ENV_FILE
    target.write_text(
        "".join(f"{k}={v}\n" for k, v in values.items()),
        encoding="utf-8",
    )


def resolve_hf_token() -> str | None:
    for key in TOKEN_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    file_vals = _read_dotenv()
    for key in TOKEN_KEYS:
        val = (file_vals.get(key) or "").strip()
        if val:
            return val
    return None


def apply_hf_token_to_environ(token: str | None = None) -> str | None:
    """Ensure HF hub clients see the token via env (HF_TOKEN)."""
    tok = (token if token is not None else resolve_hf_token()) or None
    if not tok:
        return None
    os.environ["HF_TOKEN"] = tok
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
    return tok


def load_dotenv_into_environ(path: Path | None = None) -> None:
    target = path or ENV_FILE
    for key, val in _read_dotenv(target).items():
        if key not in os.environ or not os.environ.get(key):
            os.environ[key] = val
    apply_hf_token_to_environ()


def mask_token(token: str) -> str:
    t = token.strip()
    if len(t) <= 8:
        return "••••••••"
    return f"{t[:4]}…{t[-4:]}"


def hf_auth_status() -> dict[str, object]:
    env_tok = ""
    for key in TOKEN_KEYS:
        env_tok = (os.environ.get(key) or "").strip()
        if env_tok:
            break
    file_vals = _read_dotenv()
    file_tok = ""
    for key in TOKEN_KEYS:
        file_tok = (file_vals.get(key) or "").strip()
        if file_tok:
            break
    token = env_tok or file_tok
    source = "env" if env_tok else ("file" if file_tok else "none")
    return {
        "configured": bool(token),
        "source": source,
        "masked": mask_token(token) if token else None,
        "env_file": ENV_FILE.relative_to(ROOT).as_posix(),
    }


def save_hf_token(token: str) -> dict[str, object]:
    tok = (token or "").strip()
    if not tok:
        raise ValueError("Token is empty")
    values = _read_dotenv()
    values["HF_TOKEN"] = tok
    values.pop("HUGGING_FACE_HUB_TOKEN", None)
    _write_dotenv(values)
    apply_hf_token_to_environ(tok)
    return hf_auth_status()


def clear_hf_token() -> dict[str, object]:
    values = _read_dotenv()
    for key in TOKEN_KEYS:
        values.pop(key, None)
    if values:
        _write_dotenv(values)
    elif ENV_FILE.exists():
        ENV_FILE.unlink()
    for key in TOKEN_KEYS:
        os.environ.pop(key, None)
    return hf_auth_status()
