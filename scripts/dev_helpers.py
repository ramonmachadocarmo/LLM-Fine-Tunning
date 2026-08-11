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
    "logs/eval",
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


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def show_help(
    *,
    py_ver: str,
    host: str,
    port: str,
    config: str,
    eval_set: str,
    eval_target: str,
) -> None:
    bold, dim = "1", "2"
    cyan, green, yellow, magenta, blue = "36", "32", "33", "35", "34"
    line = "=" * 58
    print(_c(bold + ";" + cyan, f" +{line}+"))
    print(_c(bold + ";" + cyan, f" |{'LLM Fine-Tuning Engine':^58}|"))
    print(_c(bold + ";" + cyan, f" |{'QLoRA / Eval / GGUF / Ollama':^58}|"))
    print(_c(bold + ";" + cyan, f" +{line}+"))
    print()
    print(_c(dim, f"  Python {py_ver} / Poetry / CUDA torch / make"))
    print()
    print(_c(bold + ";" + green, "  Setup"))
    print(f"    {_c(cyan, 'make doctor')}      check toolchain (pyenv / poetry / venv / cuda)")
    print(f"    {_c(cyan, 'make setup')}       pyenv {py_ver} + poetry sync + CUDA torch")
    print(f"    {_c(cyan, 'make fix-torch')}   restore torch 2.5.1+cu121")
    print(f"    {_c(cyan, 'make check')}       validate Python / Poetry / CUDA")
    print(f"    {_c(cyan, 'make env')}         alias for doctor")
    print()
    print(_c(bold + ";" + yellow, "  App"))
    print(f"    {_c(cyan, 'make up')}          UI at http://{host}:{port}")
    print(f"    {_c(cyan, 'make down')}        free port {port}")
    print(f"    {_c(cyan, 'make status')}      health check")
    print()
    print(_c(bold + ";" + magenta, "  Pipeline"))
    print(f"    {_c(cyan, 'make generate')}    sample train.jsonl")
    print(f"    {_c(cyan, 'make prune')}       balance dataset")
    print(f"    {_c(cyan, 'make train')}       train ({config})")
    print(f"    {_c(cyan, 'make eval')}        offline eval ({eval_set}, target={eval_target})")
    print(f"    {_c(cyan, 'make verify')}      single-prompt smoke test")
    print(f"    {_c(cyan, 'make export')}      merge + GGUF")
    print(f"    {_c(cyan, 'make chat')}        adapter chat CLI")
    print()
    print(_c(bold + ";" + blue, "  Quality"))
    print(f"    {_c(cyan, 'make test')}        pytest")
    print(f"    {_c(cyan, 'make clean')}       remove __pycache__")
    print()
    print(_c(bold, "  First install"))
    print(_c(dim, "    Windows:  .\\install.ps1"))
    print(_c(dim, "    Linux:    ./install.sh"))
    print(_c(dim, "    then:     make setup && make up"))
    print()
    print(_c(dim, "  Overrides: CONFIG=... EVAL_SET=... EVAL_TARGET=base|adapter HOST=... PORT=... NO_COLOR=1"))


def doctor(*, pyenv_root: str, python_bin: str) -> None:
    print(_c("1;36", ">> toolchain"))
    print(f"  pyenv root : {pyenv_root}")
    print(f"  python bin : {python_bin}")

    pyenv = shutil.which("pyenv")
    if pyenv:
        try:
            out = subprocess.run(["pyenv", "version"], capture_output=True, text=True, check=False)
            print(f"  pyenv      : {(out.stdout or out.stderr).strip() or 'ok'}")
        except OSError:
            print(_c("33", "  pyenv      : found but failed to run"))
    else:
        print(_c("33", "  pyenv      : not on PATH"))

    poetry = shutil.which("poetry")
    if poetry:
        out = subprocess.run(["poetry", "--version"], capture_output=True, text=True, check=False)
        print(f"  poetry     : {(out.stdout or '').strip() or poetry}")
    else:
        print(_c("33", "  poetry     : missing"))

    if sys.platform == "win32":
        venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        out = subprocess.run([str(venv_py), "--version"], capture_output=True, text=True, check=False)
        print(f"  venv       : {(out.stdout or '').strip()}  ({venv_py})")
    else:
        print(_c("33", "  venv       : missing - run make setup"))

    try:
        out = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )
        if out.returncode == 0:
            print(f"  torch      : {(out.stdout or '').strip()}")
        else:
            print(_c("33", "  torch      : not installed"))
    except OSError:
        print(_c("33", "  torch      : not installed"))


def main() -> None:
    parser = argparse.ArgumentParser(description="OS-agnostic Makefile helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dirs")
    p_port = sub.add_parser("free-port")
    p_port.add_argument("port", type=int)
    sub.add_parser("fix-torch")
    p_help = sub.add_parser("help")
    p_help.add_argument("--py-ver", default="3.11.9")
    p_help.add_argument("--host", default="127.0.0.1")
    p_help.add_argument("--port", default="7860")
    p_help.add_argument("--config", default="configs/default.yaml")
    p_help.add_argument("--eval-set", default="data/eval.sample.jsonl")
    p_help.add_argument("--eval-target", default="adapter")
    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--pyenv-root", default="")
    p_doc.add_argument("--python-bin", default="")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.cmd == "dirs":
        ensure_dirs()
    elif args.cmd == "free-port":
        free_port(args.port)
    elif args.cmd == "fix-torch":
        fix_torch()
    elif args.cmd == "help":
        show_help(
            py_ver=args.py_ver,
            host=args.host,
            port=args.port,
            config=args.config,
            eval_set=args.eval_set,
            eval_target=args.eval_target,
        )
    elif args.cmd == "doctor":
        doctor(pyenv_root=args.pyenv_root or "(unset)", python_bin=args.python_bin or sys.executable)


if __name__ == "__main__":
    main()
