#!/usr/bin/env python3
"""Cross-platform helpers for Makefile targets."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [
    "data",
    "adapters",
    "models",
    "merged_models",
    "configs/ui",
    "logs/jobs",
]


def ensure_dirs() -> None:
    for rel in DIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    print("dirs ok:", ", ".join(DIRS))


def _pids_on_port(port: int) -> list[int]:
    pids: set[int] = set()
    if sys.platform == "win32":
        out = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        needle = f":{port} "
        for line in out.splitlines():
            if needle not in line or "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if parts:
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
        return sorted(pids)

    for cmd in (
        ["lsof", "-ti", f"tcp:{port}"],
        ["fuser", f"{port}/tcp"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        for token in out.stdout.replace("\n", " ").split():
            token = token.strip()
            if token.isdigit():
                pids.add(int(token))
        if pids:
            break
    return sorted(pids)


def free_port(port: int) -> None:
    pids = _pids_on_port(port)
    if not pids:
        print(f"port {port}: free")
        return
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"port {port}: stopped pid {pid}")
        except (OSError, ProcessLookupError) as exc:
            print(f"port {port}: could not stop {pid}: {exc}")


def venv_python() -> Path:
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    if not candidate.exists():
        raise SystemExit("venv not found — run: make setup")
    return candidate


def site_packages(py: Path) -> Path:
    # Prefer querying the venv
    out = subprocess.run(
        [str(py), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip())
    # Fallback layout
    if sys.platform == "win32":
        return ROOT / ".venv" / "Lib" / "site-packages"
    return next((ROOT / ".venv" / "lib").glob("python*/site-packages"))


def fix_torch() -> None:
    py = venv_python()
    pkgs = site_packages(py)
    print(">>> removing inconsistent torch")
    subprocess.run([str(py), "-m", "pip", "uninstall", "-y", "torch"], check=False, capture_output=True)
    for name in ("torch", "torchgen", "functorch"):
        target = pkgs / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            print(f"  removed {name}")
    for dist in pkgs.glob("torch-*.dist-info"):
        shutil.rmtree(dist, ignore_errors=True)
        print(f"  removed {dist.name}")

    print(">>> installing torch==2.5.1+cu121")
    subprocess.run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.5.1+cu121",
        ],
        check=True,
    )
    print(">>> pin packaging <26 + gguf")
    subprocess.run(
        [str(py), "-m", "pip", "install", "packaging>=23,<26", "gguf>=0.10"],
        check=True,
    )
    print(">>> validating")
    subprocess.run(
        [
            str(py),
            "-c",
            "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
            "from transformers import AutoConfig; print('transformers ok')",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OS-agnostic Makefile helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dirs")
    p_port = sub.add_parser("free-port")
    p_port.add_argument("port", type=int)
    sub.add_parser("fix-torch")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.cmd == "dirs":
        ensure_dirs()
    elif args.cmd == "free-port":
        free_port(args.port)
    elif args.cmd == "fix-torch":
        fix_torch()


if __name__ == "__main__":
    main()
