from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import requests

from src.paths import MODELS_DIR, ROOT

OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Follow the user's instructions carefully. "
    "Respond with the format requested in the prompt."
)

LLAMA3_TEMPLATE = '''TEMPLATE """{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>
"""
'''

GEMMA_TEMPLATE = '''TEMPLATE """{{- $sys := "" }}
{{- if .System }}{{- $sys = .System }}{{- end }}
{{- range .Messages }}
{{- if eq .Role "system" }}{{- if not $sys }}{{- $sys = .Content }}{{- end }}
{{- else if eq .Role "user" }}<start_of_turn>user
{{ if $sys }}{{ $sys }}

{{ end }}{{ .Content }}<end_of_turn>
{{- $sys = "" }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}<end_of_turn>
{{- end }}
{{- end }}<start_of_turn>model
"""
'''

CHAT_TEMPLATES = {
    "llama": (LLAMA3_TEMPLATE, ['PARAMETER stop "<|eot_id|>"']),
    "gemma": (GEMMA_TEMPLATE, ['PARAMETER stop "<end_of_turn>"', 'PARAMETER stop "<eos>"']),
}


def detect_gguf_family(path: Path) -> str:
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        field = reader.get_field("general.architecture")
        raw = getattr(field, "contents", lambda: None)()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
        arch = str(raw or "").lower()
        if "gemma" in arch:
            return "gemma"
        if "llama" in arch:
            return "llama"
    except Exception:
        pass
    for cfg in [path.parent / "config.json", *path.parent.glob("*/config.json")]:
        if not cfg.is_file():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        blob = f"{data.get('model_type', '')} {' '.join(str(a) for a in (data.get('architectures') or []))}".lower()
        if "gemma" in blob:
            return "gemma"
        if "llama" in blob:
            return "llama"
    return "llama"


def _safe_model_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._\-]+", "-", name.strip().lower()).strip("-._")
    return cleaned or "ft-chat"


def status() -> dict[str, Any]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        return {"ok": True, "host": OLLAMA_HOST, "version": "reachable"}
    except requests.RequestException as exc:
        return {"ok": False, "host": OLLAMA_HOST, "error": str(exc)}


def list_ollama_models() -> list[dict[str, Any]]:
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    r.raise_for_status()
    models = []
    for m in r.json().get("models", []):
        models.append(
            {
                "name": m.get("name") or m.get("model"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "digest": m.get("digest"),
            }
        )
    return models


def list_gguf_files() -> list[dict[str, Any]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        out.append(
            {
                "path": p.relative_to(ROOT).as_posix(),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "suggested_model": _safe_model_name(p.stem),
            }
        )
    return out


def register_gguf(
    gguf_path: str,
    model_name: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    import subprocess

    path = Path(gguf_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists() or path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("GGUF must be inside project root") from exc

    name = _safe_model_name(model_name or path.stem)
    system = (system_prompt or DEFAULT_SYSTEM).strip()
    from_path = str(path).replace("\\", "/")
    family = detect_gguf_family(path)
    template, stops = CHAT_TEMPLATES.get(family, CHAT_TEMPLATES["llama"])
    modelfile = (
        f"FROM {from_path}\n\n"
        f"{template}\n"
        f'SYSTEM """{system}"""\n\n'
        + "".join(f"{s}\n" for s in stops)
        + "PARAMETER temperature 0.2\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        mf = Path(tmp) / "Modelfile"
        mf.write_text(modelfile, encoding="utf-8")
        proc = subprocess.run(
            ["ollama", "create", name, "-f", str(mf)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            r = requests.post(
                f"{OLLAMA_HOST}/api/create",
                json={"model": name, "from": from_path, "system": system, "stream": False},
                timeout=600,
            )
            if r.status_code >= 400:
                raise RuntimeError((proc.stderr or proc.stdout or r.text).strip())
            data = r.json() if r.content else {}
        else:
            data = {"status": "success", "cli": (proc.stdout or "").strip()}

    return {"model": name, "gguf": path.relative_to(ROOT).as_posix(), "status": data.get("status", "ok")}


def chat(
    model: str,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model is required")
    payload_messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant", "system"} and content:
            payload_messages.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in payload_messages):
        raise ValueError("at least one user message is required")

    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    data = r.json()
    msg = data.get("message") or {}
    return {
        "model": model,
        "message": {"role": msg.get("role", "assistant"), "content": msg.get("content", "")},
        "total_duration": data.get("total_duration"),
        "eval_count": data.get("eval_count"),
    }
