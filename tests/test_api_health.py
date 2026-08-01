from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_config_template_download():
    res = client.get("/api/templates/config")
    assert res.status_code == 200
    assert "project:" in res.text or "base_model" in res.text


def test_dataset_template_download():
    res = client.get("/api/templates/dataset")
    assert res.status_code == 200
    assert "instruction" in res.text


def test_get_config_rejects_traversal():
    # URL-encoded ".." so the router still receives a path under /api/configs/
    res = client.get("/api/configs/" + "..%2F" * 2 + "pyproject.toml")
    assert res.status_code == 400
    assert "configs/" in res.json()["detail"]
