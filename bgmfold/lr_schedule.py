"""Torch-free LR policy for opt-in frozen-encoder warmup and fixed cosine decay."""

from __future__ import annotations

import math


def phased_cosine_multiplier(
    step: int,
    *,
    warmup_start: int,
    warmup_steps: int,
    decay_start: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Warm up this group, hold its peak, then share a fixed cosine horizon.

    ``step`` is the number of completed optimizer updates, not microbatches.
    Frozen encoder parameters have no gradients; their LR is also zero until
    ``warmup_start``. The schedule never depends on validation or restarts.
    """
    if not (
        0 <= warmup_start <= warmup_start + warmup_steps <= decay_start < total_steps
    ):
        raise ValueError("Invalid frozen/warmup/cosine step boundaries")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie within [0, 1]")
    if step < warmup_start:
        return 0.0
    if warmup_steps and step < warmup_start + warmup_steps:
        return max(0.0, (step - warmup_start) / warmup_steps)
    progress = min(max((step - decay_start) / (total_steps - decay_start), 0.0), 1.0)
    return (
        min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
    )
