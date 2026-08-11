from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from packaging.version import Version

_MIN_TRANSFORMERS_BY_TYPE = {
    "gemma4": Version("5.5.0"),
}

_OLLAMA_DIGEST = re.compile(r"^[0-9a-f]{8,64}$", re.I)
_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ModelCheck:
    ok: bool
    ref: str
    kind: str
    checksum: str
    detail: str


class InvalidBaseModelError(ValueError):
    """Base model is not usable for transformers QLoRA/LoRA training."""


def _installed_transformers() -> Version:
    import transformers

    return Version(transformers.__version__.split("+", 1)[0])


def _assert_transformers_supports(model_type: str) -> None:
    blob = str(model_type).lower()
    installed = _installed_transformers()
    for name, minimum in _MIN_TRANSFORMERS_BY_TYPE.items():
        if name not in blob:
            continue
        if installed < minimum:
            raise InvalidBaseModelError(
                f"model_type={model_type} needs transformers>={minimum} "
                f"(installed {installed}). This stack cannot train Gemma 4. "
                "Use google/gemma-2-2b-it (or google/gemma-2-9b-it)."
            )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_gguf_ref(ref: str) -> bool:
    lower = ref.replace("\\", "/").lower()
    name = Path(lower).name
    return "gguf" in name or lower.endswith(".gguf")


def _local_files(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


def _has_weight(files: list[str]) -> bool:
    return any(f.lower().endswith(_WEIGHT_SUFFIXES) for f in files)


def _has_tokenizer(files: list[str]) -> bool:
    names = {Path(f).name.lower() for f in files}
    return any(n in names for n in _TOKENIZER_NAMES)


def _gguf_only(files: list[str]) -> bool:
    if not files:
        return False
    non_meta = [
        f
        for f in files
        if not f.lower().endswith((".md", ".txt", ".gitattributes", ".gitignore"))
        and Path(f).name not in {".gitattributes", "LICENSE", "license"}
    ]
    if not non_meta:
        return False
    return all(f.lower().endswith(".gguf") for f in non_meta)


def _validate_local(path: Path) -> ModelCheck:
    ref = str(path)
    if not path.is_dir():
        raise InvalidBaseModelError(
            f"Local base model must be a Hugging Face folder, not a file: {ref}"
        )
    files = _local_files(path)
    if _gguf_only(files) or any(f.lower().endswith(".gguf") for f in files) and not (
        (path / "config.json").is_file() and _has_weight(files)
    ):
        raise InvalidBaseModelError(
            f"GGUF folder/repo cannot be used for QLoRA training: {ref}. "
            "Use a transformers HF id or folder with config.json + weights "
            "(e.g. meta-llama/Llama-3.2-3B-Instruct). GGUF is only for Ollama after export."
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        raise InvalidBaseModelError(
            f"Missing config.json in local base model: {ref}. "
            "Train needs a Hugging Face causal-LM folder, not GGUF/Ollama."
        )
    if not _has_tokenizer(files):
        raise InvalidBaseModelError(
            f"Missing tokenizer files in {ref} "
            f"(need one of: {', '.join(_TOKENIZER_NAMES)})."
        )
    if not _has_weight(files):
        raise InvalidBaseModelError(
            f"Missing model weights in {ref} (.safetensors / .bin)."
        )
    cfg_raw = config_path.read_text(encoding="utf-8", errors="replace")
    try:
        cfg = json.loads(cfg_raw)
    except json.JSONDecodeError as exc:
        raise InvalidBaseModelError(f"Invalid config.json in {ref}: {exc}") from exc
    model_type = str(cfg.get("model_type") or cfg.get("architectures") or "unknown")
    _assert_transformers_supports(model_type)
    fingerprint = "|".join(
        [
            "local",
            path.resolve().as_posix(),
            model_type,
            _sha256_text(cfg_raw),
            str(sorted(f for f in files if f.lower().endswith(_WEIGHT_SUFFIXES))[:32]),
        ]
    )
    checksum = _sha256_text(fingerprint)[:16]
    return ModelCheck(
        ok=True,
        ref=ref,
        kind="local",
        checksum=checksum,
        detail=f"local HF folder ok (model_type={model_type})",
    )


def _validate_hf_id(repo_id: str) -> ModelCheck:
    if _looks_like_gguf_ref(repo_id):
        raise InvalidBaseModelError(
            f"Repo looks GGUF-only: {repo_id}. "
            "QLoRA needs the non-GGUF Hugging Face weights (config.json + safetensors), "
            "not a *-GGUF repo. Pick the base Instruct/Coder HF model, train, then export GGUF."
        )
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise InvalidBaseModelError(
            "huggingface_hub is required to validate remote models"
        ) from exc

    from src.shared.hf_auth import apply_hf_token_to_environ, resolve_hf_token

    token = apply_hf_token_to_environ()
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 — surface HF errors cleanly
        err = str(exc)
        if "401" in err or "gated" in err.lower() or "restricted" in err.lower():
            hint = (
                " This is a gated Hugging Face model. "
                "Accept the license on the model page, then set HF_TOKEN "
                "(UI: Base model → Hugging Face token, or env HF_TOKEN / .env)."
            )
            if not resolve_hf_token():
                hint += " No token is configured yet."
            raise InvalidBaseModelError(
                f"Cannot access gated/private model '{repo_id}'.{hint}"
            ) from exc
        raise InvalidBaseModelError(
            f"Cannot resolve Hugging Face model '{repo_id}': {exc}. "
            "Use owner/name (e.g. google/gemma-2-2b-it) or a local HF folder."
        ) from exc

    siblings = [s.rfilename for s in (info.siblings or [])]
    if _gguf_only(siblings) or (
        any(f.lower().endswith(".gguf") for f in siblings)
        and "config.json" not in siblings
    ):
        raise InvalidBaseModelError(
            f"'{repo_id}' is GGUF-only (no transformers config/weights). "
            "Cannot train QLoRA on GGUF. Use the original HF causal LM, then export."
        )
    if "config.json" not in siblings:
        raise InvalidBaseModelError(
            f"'{repo_id}' has no config.json — not a transformers training base."
        )
    if not _has_tokenizer(siblings):
        raise InvalidBaseModelError(
            f"'{repo_id}' has no tokenizer files for transformers training."
        )
    if not _has_weight(siblings):
        raise InvalidBaseModelError(
            f"'{repo_id}' has no .safetensors/.bin weights for training."
        )

    try:
        cfg_path = hf_hub_download(repo_id, "config.json", token=token)
        cfg_raw = Path(cfg_path).read_text(encoding="utf-8", errors="replace")
        cfg = json.loads(cfg_raw)
        model_type = str(cfg.get("model_type") or cfg.get("architectures") or "unknown")
        _assert_transformers_supports(model_type)
        cfg_hash = _sha256_text(cfg_raw)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if "401" in err or "gated" in err.lower() or "restricted" in err.lower():
            raise InvalidBaseModelError(
                f"Gated model '{repo_id}' needs HF auth. "
                "Accept the license on Hugging Face, then save HF_TOKEN in the UI "
                "(or set env HF_TOKEN / .env) and Validate again."
            ) from exc
        raise InvalidBaseModelError(
            f"Failed to read config.json for '{repo_id}': {exc}"
        ) from exc

    revision = getattr(info, "sha", None) or "unknown"
    fingerprint = "|".join(
        ["hf", repo_id, revision, model_type, cfg_hash, str(sorted(siblings)[:64])]
    )
    checksum = _sha256_text(fingerprint)[:16]
    return ModelCheck(
        ok=True,
        ref=repo_id,
        kind="huggingface",
        checksum=checksum,
        detail=f"HF model ok (revision={revision[:12]}, model_type={model_type})",
    )


def validate_base_model(model_ref: str) -> ModelCheck:
    """Validate that *model_ref* is a transformers-trainable base; return checksum."""
    ref = (model_ref or "").strip().strip('"').strip("'")
    if not ref:
        raise InvalidBaseModelError("base_model is empty")

    if _OLLAMA_DIGEST.fullmatch(ref):
        raise InvalidBaseModelError(
            f"'{ref}' looks like an Ollama digest. "
            "Train needs a Hugging Face id or local HF folder — not `ollama list` hashes. "
            "Ollama/GGUF is only for Chat after export."
        )

    if _looks_like_gguf_ref(ref):
        raise InvalidBaseModelError(
            f"'{ref}' looks like a GGUF artifact/repo. "
            "QLoRA training requires HF transformers weights (config.json + safetensors), "
            "not GGUF. Example: google/gemma-2-2b-it"
        )

    local = Path(ref)
    if local.exists():
        return _validate_local(local.resolve())

    # relative ./path that does not exist yet
    if ref.startswith((".", "/", "\\")) or (len(ref) > 1 and ref[1] == ":"):
        raise InvalidBaseModelError(f"Local base model path not found: {ref}")

    if not _HF_ID.match(ref):
        raise InvalidBaseModelError(
            f"Invalid base_model '{ref}'. Expected Hugging Face id (owner/name) "
            "or an existing local HF folder. Not Ollama names/digests or bare filenames."
        )
    return _validate_hf_id(ref)
