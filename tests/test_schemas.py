from __future__ import annotations

import pytest
from pydantic import ValidationError

from web.api.schemas import ExportRequest, TrainRequest


def test_train_request_defaults():
    req = TrainRequest(project_name="demo", base_model="meta-llama/Llama-3.2-1B-Instruct")
    assert req.epochs == 2
    assert req.load_in_4bit is True
    assert req.start_training is True


def test_train_request_requires_name():
    with pytest.raises(ValidationError):
        TrainRequest(project_name="", base_model="x")


def test_export_request():
    req = ExportRequest(config_path="configs/default.yaml")
    assert req.config_path.endswith(".yaml")
