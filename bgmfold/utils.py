"""Generic utilities: seeding, logging, checkpointing, masks, optim/scheduler builders."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

# All log timestamps are rendered in Beijing time (UTC+8), regardless of the
# server's actual system timezone (this host runs in UTC), since that's the
# timezone the team actually reads logs in.
_BEIJING_TZ = timezone(timedelta(hours=8))


class _BeijingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=_BEIJING_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def now_cst() -> datetime:
    """Current time in China Standard Time (UTC+8), regardless of host system timezone."""
    return datetime.now(_BEIJING_TZ)


def now_cst_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Current CST time formatted as a string (default: for run-folder naming, use fmt='%Y%m%d_%H%M%S')."""
    return now_cst().strftime(fmt)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(
    name: str, log_file: Optional[str] = None, level=logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = _BeijingFormatter(
        "[%(asctime)s CST] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class AverageMeter:
    """Tracks a running average of a scalar value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def build_pair_mask(
    mask: torch.Tensor, min_sequence_separation: int = 1
) -> torch.Tensor:
    """Build a [B, L, L] boolean mask for valid RNA base-pair candidates.

    Only padding and the diagonal are excluded, in training and evaluation.
    ``min_sequence_separation`` remains an ignored compatibility argument for
    old launchers/configs; it cannot re-enable a minimum-loop prior.
    """

    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    length = mask.size(1)
    idx = torch.arange(length, device=mask.device)
    separation = (idx[:, None] - idx[None, :]).abs()
    return pair_mask & (separation > 0)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    base_lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    rnafm_lr = cfg.get("rna_encoder_lr", cfg.get("rnafm_lr"))
    if rnafm_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=base_lr, weight_decay=weight_decay)

    rnafm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("rnafm.") or ".rnafm." in name:
            rnafm_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    if rnafm_params:
        param_groups.append({"params": rnafm_params, "lr": rnafm_lr})
    if cfg.get("scheduler") == "cosine_with_encoder_warmup":
        # Only the new opt-in policy needs group identities. Legacy optimizers
        # and their checkpoint parameter groups are unchanged.
        for group in param_groups:
            group["group_name"] = (
                "encoder" if group["params"] is rnafm_params else "task"
            )
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer, cfg: Dict[str, Any], num_training_steps: int
) -> Optional[_LRScheduler]:
    scheduler_type = cfg.get("scheduler", "none")
    if scheduler_type == "cosine_with_encoder_warmup":
        from bgmfold.lr_schedule import phased_cosine_multiplier

        encoder_start = int(cfg["rna_encoder_warmup_start_steps"])
        encoder_warmup = int(cfg["rna_encoder_warmup_steps"])
        decay_start = encoder_start + encoder_warmup
        floor = float(cfg.get("min_lr_ratio", 0.1))

        def group_lambda(start, warmup):
            # Validate once before constructing LambdaLR (which evaluates step 0).
            phased_cosine_multiplier(
                0,
                warmup_start=start,
                warmup_steps=warmup,
                decay_start=decay_start,
                total_steps=num_training_steps,
                min_lr_ratio=floor,
            )
            return lambda step: phased_cosine_multiplier(
                step,
                warmup_start=start,
                warmup_steps=warmup,
                decay_start=decay_start,
                total_steps=num_training_steps,
                min_lr_ratio=floor,
            )

        schedules = []
        for group in optimizer.param_groups:
            if group.get("group_name") == "encoder":
                schedules.append(group_lambda(encoder_start, encoder_warmup))
            elif group.get("group_name") == "task":
                schedules.append(group_lambda(0, int(cfg["warmup_steps"])))
            else:
                raise ValueError(
                    "Phased cosine requires named task/encoder parameter groups"
                )
        return LambdaLR(optimizer, schedules)
    warmup_steps = cfg.get("warmup_steps", 0)
    base_lr = float(cfg.get("lr", optimizer.param_groups[0]["lr"]))
    if cfg.get("min_lr") is not None:
        min_lr_ratio = float(cfg["min_lr"]) / max(base_lr, 1e-12)
    else:
        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    if scheduler_type in (None, "none"):
        return None

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type in ("cosine", "cosine_with_min_lr"):
            decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        elif scheduler_type == "linear":
            decay = max(0.0, 1.0 - progress)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
    save_optimizer_state: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    if save_optimizer_state:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    if extra:
        state.update(extra)
    temporary = f"{path}.partial.{os.getpid()}"
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    # Project checkpoints include optimizer/scheduler state and scalar metadata,
    # so PyTorch 2.6+'s weights_only=True default is too restrictive here.
    state = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def _checkpoint_model_state(state: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    """Extract a model state dict from project, trainer, or raw checkpoints."""
    if isinstance(state, Mapping):
        for container_key in ("model", "state_dict", "model_state_dict"):
            candidate = state.get(container_key)
            if isinstance(candidate, Mapping):
                return candidate, container_key
        if state and all(isinstance(key, str) for key in state):
            tensor_values = {
                key: value for key, value in state.items() if torch.is_tensor(value)
            }
            if tensor_values:
                return tensor_values, "raw"
    raise ValueError(
        "Initialization checkpoint does not contain a model/state_dict/model_state_dict mapping"
    )


def _target_key_for_source(source_key: str, target_keys: set[str]) -> str | None:
    """Resolve common wrapper prefixes without rewriting real module names."""
    candidates = [source_key]
    current = source_key
    while True:
        stripped = None
        for prefix in ("module.", "_orig_mod."):
            if current.startswith(prefix):
                stripped = current[len(prefix) :]
                break
        if stripped is None:
            break
        candidates.append(stripped)
        current = stripped
    if current.startswith("model."):
        candidates.append(current[len("model.") :])
    return next(
        (candidate for candidate in candidates if candidate in target_keys), None
    )


def _state_component(key: str) -> str:
    if key.startswith("rnafm."):
        return "encoder"
    for component in (
        "seq_proj",
        "attn_proj",
        "fusion",
        "single_branch_norm",
        "local_smoother",
        "pairedness_head",
        "pair_type_head",
        "head",
    ):
        if key == component or key.startswith(f"{component}."):
            return component
    return "other"


def load_compatible_initialization(
    path: str,
    model: nn.Module,
    *,
    map_location: str = "cpu",
    require_encoder: bool = True,
    min_encoder_coverage: float = 0.99,
) -> Dict[str, Any]:
    """Warm-start every name-and-shape-compatible tensor from one checkpoint.

    This intentionally does not restore epoch, optimizer, scheduler, or best
    metric state. Incompatible source tensors are reported and ignored; target
    tensors absent from the source retain their freshly initialized values.
    """
    if not 0.0 <= float(min_encoder_coverage) <= 1.0:
        raise ValueError("min_encoder_coverage must be within [0, 1]")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    source_state, container_key = _checkpoint_model_state(checkpoint)
    raw_model = model.module if hasattr(model, "module") else model
    target_state = raw_model.state_dict()
    target_keys = set(target_state)
    compatible: dict[str, torch.Tensor] = {}
    loaded_entries = []
    skipped_entries = []

    for source_key, source_value in source_state.items():
        if not torch.is_tensor(source_value):
            skipped_entries.append(
                {
                    "source_key": source_key,
                    "reason": "not_a_tensor",
                }
            )
            continue
        target_key = _target_key_for_source(source_key, target_keys)
        if target_key is None:
            skipped_entries.append(
                {
                    "source_key": source_key,
                    "reason": "not_present_in_target",
                    "shape": list(source_value.shape),
                }
            )
            continue
        if target_key in compatible:
            skipped_entries.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "reason": "duplicate_target_key",
                    "shape": list(source_value.shape),
                }
            )
            continue
        target_value = target_state[target_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_entries.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "reason": "shape_mismatch",
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        compatible[target_key] = source_value
        loaded_entries.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "component": _state_component(target_key),
                "shape": list(source_value.shape),
                "numel": int(source_value.numel()),
            }
        )

    raw_model.load_state_dict(compatible, strict=False)
    loaded_keys = set(compatible)
    random_entries = [
        {
            "target_key": key,
            "component": _state_component(key),
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "reason": "missing_or_incompatible_source",
        }
        for key, value in target_state.items()
        if key not in loaded_keys
    ]

    components: dict[str, dict[str, int | float]] = {}
    component_names = sorted({_state_component(key) for key in target_state})
    for component in component_names:
        target_items = [
            (key, value)
            for key, value in target_state.items()
            if _state_component(key) == component
        ]
        target_numel = sum(int(value.numel()) for _, value in target_items)
        loaded_numel = sum(
            int(target_state[key].numel())
            for key, _ in target_items
            if key in loaded_keys
        )
        components[component] = {
            "loaded_tensors": sum(key in loaded_keys for key, _ in target_items),
            "target_tensors": len(target_items),
            "loaded_numel": loaded_numel,
            "target_numel": target_numel,
            "coverage": loaded_numel / max(target_numel, 1),
        }

    encoder_coverage = float(components.get("encoder", {}).get("coverage", 0.0))
    if require_encoder and encoder_coverage < float(min_encoder_coverage):
        raise RuntimeError(
            "Initialization checkpoint encoder coverage is too low: "
            f"{encoder_coverage:.2%} < {float(min_encoder_coverage):.2%}. "
            "Check the encoder family, checkpoint format, and parameter names."
        )
    if not compatible:
        raise RuntimeError(
            "Initialization checkpoint has no name-and-shape-compatible tensors"
        )

    return {
        "checkpoint": os.path.abspath(os.path.expanduser(path)),
        "container_key": container_key,
        "loaded_tensor_count": len(loaded_entries),
        "target_tensor_count": len(target_state),
        "source_tensor_count": sum(
            torch.is_tensor(value) for value in source_state.values()
        ),
        "encoder_coverage": encoder_coverage,
        "components": components,
        "loaded": loaded_entries,
        "skipped_source": skipped_entries,
        "random_target": random_entries,
    }
