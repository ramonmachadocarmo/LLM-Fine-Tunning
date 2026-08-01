from __future__ import annotations

from pathlib import Path
from typing import Any

from src.paths import ADAPTERS_DIR, CONFIGS_DIR, DATA_DIR, MERGED_DIR, MODELS_DIR, ROOT, UI_CONFIGS_DIR


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _file_info(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    size = path.stat().st_size if path.exists() else 0
    info = {
        "path": _rel(path),
        "name": path.name,
        "size_bytes": size,
        "exists": path.exists(),
    }
    if kind:
        info["kind"] = kind
    return info


def list_datasets() -> list[dict[str, Any]]:
    """List every .jsonl in data/ (samples + real datasets)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.jsonl"))
    out: list[dict[str, Any]] = []
    for p in files:
        kind = "sample" if ".sample." in p.name or p.name.endswith(".sample.jsonl") else "dataset"
        out.append(_file_info(p, kind=kind))
    return out


def list_configs() -> list[dict[str, Any]]:
    """List YAML in configs/ and configs/ui/ (templates + local)."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    UI_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for folder in (CONFIGS_DIR, UI_CONFIGS_DIR):
        files.extend(sorted(folder.glob("*.yaml")))
        files.extend(sorted(folder.glob("*.yml")))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in files:
        rel = _rel(p)
        if rel in seen:
            continue
        seen.add(rel)
        kind = "template" if ".template." in p.name else "config"
        out.append(_file_info(p, kind=kind))
    return out


def list_local_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for base in (MERGED_DIR, MODELS_DIR):
        base.mkdir(parents=True, exist_ok=True)
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir() and (child / "config.json").exists():
                models.append(
                    {
                        "path": _rel(child),
                        "name": child.name,
                        "kind": "local_hf",
                        "source": _rel(base),
                    }
                )
            elif child.is_file() and child.suffix.lower() == ".gguf":
                models.append(
                    {
                        "path": _rel(child),
                        "name": child.name,
                        "kind": "gguf",
                        "source": _rel(base),
                    }
                )
    return models


def list_adapters() -> list[dict[str, Any]]:
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    adapters: list[dict[str, Any]] = []
    for child in sorted(ADAPTERS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        adapters.append(
            {
                "path": _rel(child),
                "name": child.name,
                "has_adapter": (child / "adapter_model.safetensors").exists(),
                "has_best_model": (child / "best_model" / "adapter_model.safetensors").exists(),
            }
        )
    return adapters


def suggested_hf_models() -> list[str]:
    return [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]


def preferred_config_path(configs: list[dict[str, Any]] | None = None) -> str | None:
    """Pick default config: default.yaml > any local yaml > template."""
    items = configs if configs is not None else list_configs()
    if not items:
        return None
    paths = [c["path"] for c in items]
    for preferred in (
        "configs/default.yaml",
        "configs/ui/default.yaml",
    ):
        if preferred in paths:
            return preferred
    locals_only = [p for p in paths if not p.endswith(".template.yaml")]
    if locals_only:
        return locals_only[0]
    return paths[0]


def scan_all() -> dict[str, Any]:
    configs = list_configs()
    return {
        "datasets": list_datasets(),
        "configs": configs,
        "local_models": list_local_models(),
        "adapters": list_adapters(),
        "suggested_hf_models": suggested_hf_models(),
        "preferred_config": preferred_config_path(configs),
        "root": str(ROOT),
    }
