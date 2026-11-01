"""Probability thresholds shared by training and evaluation; independent of PyTorch."""

import math


def default_threshold_candidates():
    return sorted(
        {i / 100 for i in range(1, 100)}
        | {i / 1000 for i in range(905, 1000, 10)}
        | {0.999}
    )


def scan_threshold_candidates(settings=None):
    """Allow an explicit threshold grid to override the default."""
    settings = settings or {}
    if not any(k in settings for k in ("threshold_step", "extra_thresholds")):
        return default_threshold_candidates()
    step = float(settings.get("threshold_step", 0.01))
    if not 0 < step < 1:
        raise ValueError("threshold_step must be in (0, 1)")
    values = [
        round(i * step, 8)
        for i in range(1, math.ceil(1 / step))
        if round(i * step, 8) < 1
    ]
    values += list(map(float, settings.get("extra_thresholds", [])))
    if any(not 0 < t < 1 for t in values):
        raise ValueError("Threshold probabilities must be in (0, 1)")
    return sorted(set(values))
