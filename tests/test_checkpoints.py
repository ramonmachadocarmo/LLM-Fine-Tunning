from __future__ import annotations

from pathlib import Path

from src.shared.checkpoints import find_latest_checkpoint


def test_find_latest_missing_dir(tmp_path: Path):
    assert find_latest_checkpoint(str(tmp_path / "missing")) == (None, 0)


def test_find_latest_empty(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    assert find_latest_checkpoint(str(out)) == (None, 0)


def test_find_latest_picks_highest_epoch(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoint-epoch-1").mkdir()
    (out / "checkpoint-epoch-3").mkdir()
    (out / "checkpoint-epoch-2").mkdir()
    path, epoch = find_latest_checkpoint(str(out))
    assert epoch == 3
    assert path is not None
    assert path.endswith("checkpoint-epoch-3")
