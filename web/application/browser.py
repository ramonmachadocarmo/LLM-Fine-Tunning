from __future__ import annotations

from pathlib import Path
from typing import Any

from src.paths import ROOT


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_under_root(rel_path: str | None = None) -> Path:
    """Resolve a path that must stay inside the project root."""
    if not rel_path or rel_path in {".", "./", ""}:
        return ROOT.resolve()
    candidate = (ROOT / rel_path).resolve()
    root = ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path must be inside the project root")
    return candidate


def browse(
    path: str | None = None,
    *,
    mode: str = "all",
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    List directory entries for the UI explorer.

    mode: all | files | dirs
    extensions: e.g. [".yaml", ".jsonl"] (case-insensitive)
    """
    current = resolve_under_root(path)
    if not current.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not current.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or [])}
    entries: list[dict[str, Any]] = []

    for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        if child.name in {".venv", ".venv.bak", "llama.cpp", "graphify-out", "__pycache__", "node_modules"}:
            continue

        is_dir = child.is_dir()
        if mode == "dirs" and not is_dir:
            continue
        if mode == "files" and not is_dir and exts and child.suffix.lower() not in exts:
            continue
        if mode == "files" and not is_dir and not exts:
            pass
        # Keep directories always when mode=files so the UI can navigate.

        entries.append(
            {
                "name": child.name,
                "path": _rel(child),
                "is_dir": is_dir,
                "size_bytes": child.stat().st_size if child.is_file() else None,
            }
        )

    parent = None
    root = ROOT.resolve()
    if current.resolve() != root:
        parent_path = current.parent.resolve()
        parent = "." if parent_path == root else _rel(parent_path)

    return {
        "root": str(ROOT),
        "current": "." if current.resolve() == root else _rel(current),
        "parent": parent,
        "entries": entries,
    }
