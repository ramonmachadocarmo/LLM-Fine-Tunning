from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    base_model: str = Field(..., min_length=1)
    dataset_paths: list[str] = Field(default_factory=list)
    load_in_4bit: bool = True
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    save_strategy: str = "steps"
    save_steps: int = 50
    learning_rate: float = 2.0e-4
    max_seq_length: int = 1024
    max_steps: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."
    output_dir: Optional[str] = None
    adapter_path: Optional[str] = None
    merged_path: Optional[str] = None
    gguf_filename: Optional[str] = None
    save_config_as: Optional[str] = None
    start_training: bool = True


class SaveConfigRequest(TrainRequest):
    start_training: bool = False


class ExportRequest(BaseModel):
    config_path: str


class LoadConfigResponse(BaseModel):
    path: str
    config: dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    system_prompt: Optional[str] = None


class OllamaRegisterRequest(BaseModel):
    gguf_path: str
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None

