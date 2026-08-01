from __future__ import annotations

import glob
import os
import re


def find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    """Return (checkpoint_path, epoch) for the latest epoch checkpoint, or (None, 0)."""
    if not os.path.exists(output_dir):
        return None, 0

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-epoch-*"))
    if not checkpoints:
        return None, 0

    def extract_epoch(path: str) -> int:
        match = re.search(r"checkpoint-epoch-(\d+)", path)
        return int(match.group(1)) if match else -1

    latest_checkpoint = max(checkpoints, key=extract_epoch)
    return latest_checkpoint, extract_epoch(latest_checkpoint)
