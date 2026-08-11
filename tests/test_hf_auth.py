from __future__ import annotations

import os

from src.shared import hf_auth


def test_save_and_status(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.setattr(hf_auth, "ENV_FILE", env_file)
    monkeypatch.setattr(hf_auth, "ROOT", tmp_path)
    for key in hf_auth.TOKEN_KEYS:
        monkeypatch.delenv(key, raising=False)

    st = hf_auth.save_hf_token("hf_test_token_abcdefgh")
    assert st["configured"] is True
    assert st["source"] == "env"
    assert "hf_t" in st["masked"]
    assert env_file.read_text(encoding="utf-8").startswith("HF_TOKEN=")
    assert os.environ.get("HF_TOKEN") == "hf_test_token_abcdefgh"

    cleared = hf_auth.clear_hf_token()
    assert cleared["configured"] is False
    assert not env_file.exists() or "HF_TOKEN" not in env_file.read_text(encoding="utf-8")


def test_load_dotenv_into_environ(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    monkeypatch.setattr(hf_auth, "ENV_FILE", env_file)
    monkeypatch.setattr(hf_auth, "ROOT", tmp_path)
    for key in hf_auth.TOKEN_KEYS:
        monkeypatch.delenv(key, raising=False)

    hf_auth.load_dotenv_into_environ(env_file)
    assert os.environ["HF_TOKEN"] == "hf_from_file"
