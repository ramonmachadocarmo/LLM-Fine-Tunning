from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from src.config import build_config_from_form, default_config, save_config
from src.config.builder import load_config_file
from src.paths import CONFIGS_DIR, DATA_DIR, ROOT, UI_CONFIGS_DIR
from src.shared.hf_auth import clear_hf_token, hf_auth_status, save_hf_token
from src.training.validate_model import InvalidBaseModelError, validate_base_model
from web.api.schemas import (
    ExportRequest,
    HfTokenRequest,
    LoadConfigResponse,
    OllamaChatRequest,
    OllamaRegisterRequest,
    SaveConfigRequest,
    TrainRequest,
    ValidateModelRequest,
)
from web.application.assets import scan_all
from web.application.browser import browse
from web.application.jobs import job_runner
from web.application import ollama_chat
from web.application.templates import config_template_yaml, dataset_sample_jsonl

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/models/validate")
def validate_model(payload: ValidateModelRequest):
    try:
        check = validate_base_model(payload.base_model)
    except InvalidBaseModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": check.ok,
        "ref": check.ref,
        "kind": check.kind,
        "checksum": check.checksum,
        "detail": check.detail,
    }


@router.get("/hf/token")
def get_hf_token_status():
    return hf_auth_status()


@router.put("/hf/token")
def put_hf_token(payload: HfTokenRequest):
    try:
        return save_hf_token(payload.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/hf/token")
def delete_hf_token():
    return clear_hf_token()


@router.get("/templates/config")
def download_config_template():
    template_path = CONFIGS_DIR / "default.template.yaml"
    if template_path.exists():
        body = template_path.read_text(encoding="utf-8")
    else:
        body = config_template_yaml(default_config())
    return Response(
        content=body,
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="default.template.yaml"'},
    )


@router.get("/templates/dataset")
def download_dataset_template():
    sample_path = DATA_DIR / "sample.jsonl"
    body = sample_path.read_text(encoding="utf-8") if sample_path.exists() else dataset_sample_jsonl()
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="sample.jsonl"'},
    )

@router.get("/assets")
def get_assets():
    return scan_all()


@router.get("/browse")
def browse_path(
    path: str | None = Query(default=None),
    mode: str = Query(default="all"),
    extensions: str | None = Query(default=None, description="Comma-separated, e.g. .yaml,.jsonl"),
):
    try:
        exts = [e.strip() for e in extensions.split(",")] if extensions else None
        return browse(path, mode=mode, extensions=exts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload")
async def upload_file(
    kind: str = Query(..., description="config | dataset"),
    file: UploadFile = File(...),
):
    name = file.filename or "upload.bin"
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if kind == "config":
        if suffix not in {"yaml", "yml"}:
            raise HTTPException(status_code=400, detail="Config must be .yaml/.yml")
        UI_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UI_CONFIGS_DIR / name
    elif kind == "dataset":
        if suffix != "jsonl":
            raise HTTPException(status_code=400, detail="Dataset must be .jsonl")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_DIR / name
    else:
        raise HTTPException(status_code=400, detail="kind must be config or dataset")

    content = await file.read()
    dest.write_bytes(content)
    rel = dest.relative_to(ROOT).as_posix()
    return {"path": rel, "name": dest.name, "kind": kind}


@router.get("/configs/{config_path:path}", response_model=LoadConfigResponse)
def get_config(config_path: str):
    try:
        cfg = load_config_file(config_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"path": config_path, "config": cfg}


@router.post("/configs")
def create_config(payload: SaveConfigRequest):
    if not payload.dataset_paths:
        raise HTTPException(status_code=400, detail="Select at least one dataset")
    cfg = build_config_from_form(payload.model_dump())
    path = save_config(cfg, payload.save_config_as or payload.project_name)
    return {"config_path": path, "config": cfg}


@router.post("/jobs/train")
def start_train(payload: TrainRequest):
    if not payload.dataset_paths:
        raise HTTPException(status_code=400, detail="Select at least one dataset")
    try:
        model_check = validate_base_model(payload.base_model)
    except InvalidBaseModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cfg = build_config_from_form(payload.model_dump())
    path = save_config(cfg, payload.save_config_as or payload.project_name)
    if not payload.start_training:
        return {
            "config_path": path,
            "config": cfg,
            "job": None,
            "model_checksum": model_check.checksum,
        }
    try:
        job = job_runner.start_train(path)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "config_path": path,
        "config": cfg,
        "job": job.to_dict(),
        "model_checksum": model_check.checksum,
    }


@router.post("/jobs/export")
def start_export(payload: ExportRequest):
    try:
        job = job_runner.start_export(payload.config_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job.to_dict()}


@router.get("/jobs")
def list_jobs():
    return {"jobs": job_runner.list_jobs()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = 200):
    try:
        return job_runner.read_log(job_id, tail=tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        job = job_runner.stop_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@router.get("/ollama/status")
def ollama_status():
    return ollama_chat.status()


@router.get("/ollama/models")
def ollama_models():
    st = ollama_chat.status()
    if not st.get("ok"):
        raise HTTPException(status_code=503, detail=st.get("error") or "Ollama offline")
    try:
        return {
            "models": ollama_chat.list_ollama_models(),
            "ggufs": ollama_chat.list_gguf_files(),
            "default_system": ollama_chat.DEFAULT_SYSTEM,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/ollama/register")
def ollama_register(payload: OllamaRegisterRequest):
    st = ollama_chat.status()
    if not st.get("ok"):
        raise HTTPException(status_code=503, detail=st.get("error") or "Ollama offline")
    try:
        return ollama_chat.register_gguf(
            payload.gguf_path,
            model_name=payload.model_name,
            system_prompt=payload.system_prompt,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ollama/chat")
def ollama_chat_endpoint(payload: OllamaChatRequest):
    st = ollama_chat.status()
    if not st.get("ok"):
        raise HTTPException(status_code=503, detail=st.get("error") or "Ollama offline")
    try:
        return ollama_chat.chat(
            payload.model,
            [m.model_dump() for m in payload.messages],
            system_prompt=payload.system_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
