from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.paths import LOGS_DIR, ROOT


@dataclass
class JobRecord:
    id: str
    kind: str
    config_path: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    log_path: str | None = None
    pid: int | None = None
    error: str | None = None
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start_train(self, config_path: str) -> JobRecord:
        return self._start(
            "train",
            [sys.executable, "scripts/train.py", "--config", config_path],
            config_path,
        )

    def start_export(self, config_path: str) -> JobRecord:
        return self._start(
            "export",
            [sys.executable, "scripts/export.py", "--config", config_path, "--all"],
            config_path,
        )

    def _start(self, kind: str, command: list[str], config_path: str) -> JobRecord:
        with self._lock:
            running = [j for j in self._jobs.values() if j.status == "running"]
            if running:
                raise RuntimeError(f"Another job is already running: {running[0].id}")

            job_id = uuid.uuid4().hex[:10]
            now = datetime.now(timezone.utc).isoformat()
            log_file = LOGS_DIR / f"{job_id}.log"
            job = JobRecord(
                id=job_id,
                kind=kind,
                config_path=config_path,
                status="queued",
                created_at=now,
                log_path=log_file.relative_to(ROOT).as_posix(),
                command=command,
            )
            self._jobs[job_id] = job

        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return job

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            command = list(job.command)
            log_path = ROOT / (job.log_path or f"logs/jobs/{job_id}.log")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        from src.shared.hf_auth import load_dotenv_into_environ

        load_dotenv_into_environ()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(command)}\n\n")
            log.flush()
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = datetime.now(timezone.utc).isoformat()
                return

            with self._lock:
                job.status = "running"
                job.started_at = datetime.now(timezone.utc).isoformat()
                job.pid = proc.pid
                self._procs[job_id] = proc

            exit_code = proc.wait()

            with self._lock:
                job.exit_code = exit_code
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.status = "completed" if exit_code == 0 else "failed"
                if exit_code != 0 and not job.error:
                    job.error = f"Process exited with code {exit_code}"
                self._procs.pop(job_id, None)

    def stop_job(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            proc = self._procs.get(job_id)
            if not job:
                raise KeyError(f"Job not found: {job_id}")
            if job.status != "running" or not proc:
                raise RuntimeError("Job is not running")
            proc.terminate()
            job.status = "stopped"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            return job

    def read_log(self, job_id: str, tail: int = 200) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job or not job.log_path:
            raise KeyError(f"Job not found: {job_id}")
        path = ROOT / job.log_path
        if not path.exists():
            return {"job_id": job_id, "lines": [], "status": job.status}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {
            "job_id": job_id,
            "status": job.status,
            "lines": lines[-tail:],
            "total_lines": len(lines),
        }


job_runner = JobRunner()
