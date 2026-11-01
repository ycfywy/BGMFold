"""Exact accumulation of one original padded batch over smaller micro-batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bgmfold.utils import build_pair_mask


def _slice(batch: dict, indices: list[int], padded_length: int) -> dict:
    out = dict(batch)
    out["ids"] = [batch["ids"][i] for i in indices]
    out["sequences"] = [batch["sequences"][i] for i in indices]
    local_length = max(map(len, out["sequences"]))
    for key in ("contact_map", "pair_type_map"):
        value = batch[key][indices, :local_length, :local_length]
        if local_length < padded_length:
            value = F.pad(
                value,
                (0, padded_length - local_length, 0, padded_length - local_length),
                value=-1 if key == "pair_type_map" else 0,
            )
        out[key] = value
    for key in ("paired_tokens", "mask"):
        value = batch[key][indices, :local_length]
        if local_length < padded_length:
            value = F.pad(value, (0, padded_length - local_length))
        out[key] = value
    return out


def _denominators(
    batch: dict, loss_cfg: dict, separation: int
) -> dict[str, torch.Tensor]:
    mask = batch["mask"]
    pair_mask = build_pair_mask(mask, separation).float()
    pos_weight = float(loss_cfg.get("pos_weight", 200.0))
    contact_weights = torch.where(batch["contact_map"] > 0.5, pos_weight, 1.0)
    contact = (contact_weights * pair_mask).sum()
    aux = loss_cfg.get("multitask", {})
    paired_weights = torch.where(
        batch["paired_tokens"] > 0.5, float(aux.get("pairedness_pos_weight", 1.0)), 1.0
    )
    pairedness = (paired_weights * mask).sum()
    valid_type = pair_mask.bool() & (batch["pair_type_map"] >= 0)
    class_weights = torch.tensor(
        aux.get("pair_type_class_weights", [1.0] * 4),
        dtype=torch.float32,
        device=mask.device,
    )
    pair_type = (class_weights[batch["pair_type_map"].clamp_min(0)] * valid_type).sum()
    return {"contact": contact, "pairedness": pairedness, "pair_type": pair_type}


def backward_original_batch(
    model,
    batch: dict,
    loss_fn,
    loss_cfg: dict,
    separation: int,
    device: torch.device,
    splits: int = 2,
) -> tuple[dict, torch.Tensor]:
    """Backprop one full batch while only materializing one split at a time."""
    if splits < 1:
        raise ValueError("splits must be positive")
    if splits > 1 and (
        loss_cfg.get("dice_weight", 0) or loss_cfg.get("degree_weight", 0)
    ):
        raise ValueError(
            "Exact split accumulation currently requires dice_weight=degree_weight=0"
        )
    if splits > 1 and (
        loss_cfg.get("normalization") == "per_sequence_class_balanced"
        or loss_cfg.get("multitask", {}).get("normalization")
        == "per_sequence_class_balanced"
    ):
        raise ValueError(
            "Exact split accumulation does not support per_sequence_class_balanced"
        )
    padded_length = batch["mask"].shape[1]
    full_denoms = _denominators(batch, loss_cfg, separation)
    totals: dict[str, torch.Tensor] = {}
    last_mask = None
    for part in range(min(splits, len(batch["sequences"]))):
        indices = list(range(part, len(batch["sequences"]), splits))
        micro = _slice(batch, indices, padded_length)
        for key in ("contact_map", "paired_tokens", "pair_type_map"):
            micro[key] = micro[key].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output, mask = model(
                micro["sequences"], return_auxiliary=True, pad_to_length=padded_length
            )
            local = loss_fn(output, micro, mask)
        local_denoms = _denominators(micro, loss_cfg, separation)
        total = (
            local["contact"]
            * local_denoms["contact"]
            / full_denoms["contact"].clamp_min(1.0)
        )
        for name, weight_key in (
            ("pairedness", "pairedness_weight"),
            ("pair_type", "pair_type_weight"),
        ):
            if name in local:
                weight = float(loss_cfg.get("multitask", {}).get(weight_key, 0.0))
                total = total + weight * local[name] * local_denoms[name] / full_denoms[
                    name
                ].clamp_min(1.0)
        total.backward()
        totals["total"] = (
            totals.get("total", torch.zeros((), device=device)) + total.detach()
        )
        last_mask = mask
    return totals, last_mask
