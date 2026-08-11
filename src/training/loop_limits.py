from __future__ import annotations


def parse_max_steps(raw) -> int | None:
    if raw in (None, "", False):
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def parse_epochs(raw) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 1
    return n if n > 0 else 1


def optimizer_step(micro_batches: int, accum: int) -> int:
    if accum < 1:
        return micro_batches
    return micro_batches // accum


def hit_max_steps(micro_batches: int, accum: int, max_steps: int | None) -> bool:
    if max_steps is None:
        return False
    return optimizer_step(micro_batches, accum) >= max_steps
