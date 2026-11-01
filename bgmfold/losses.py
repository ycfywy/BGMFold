"""Contact, pairedness and pair-type losses, unchanged from the research trainer."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from bgmfold.utils import build_pair_mask


def distributed_weighted_mean(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Return a globally weighted mean with the correct DDP gradient scale."""
    denominator = denominator.to(dtype=torch.float32)
    if not enabled:
        return numerator / denominator.clamp_min(1.0)
    stats = torch.stack((numerator.detach(), denominator.detach()))
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_numerator, global_denominator = stats.unbind()
    global_denominator = global_denominator.clamp_min(1.0)
    # DDP averages gradients across ranks, so compensate by world_size here.
    backward_value = numerator * (float(dist.get_world_size()) / global_denominator)
    reported_value = global_numerator / global_denominator
    return backward_value + (reported_value - backward_value.detach())


def _sequence_normalized_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    distributed_batch_reduction: bool,
) -> torch.Tensor:
    """Normalize each RNA independently, then average RNAs across all ranks."""
    if values.shape != weights.shape or values.ndim < 2:
        raise ValueError("values and weights must have the same [B, ...] shape")
    reduce_dims = tuple(range(1, values.ndim))
    per_sequence_denominator = weights.sum(dim=reduce_dims)
    valid_sequences = per_sequence_denominator > 0
    per_sequence_loss = (values * weights).sum(
        dim=reduce_dims
    ) / per_sequence_denominator.clamp_min(1.0)
    numerator = per_sequence_loss[valid_sequences].sum()
    denominator = valid_sequences.sum().to(dtype=torch.float32)
    return distributed_weighted_mean(
        numerator, denominator, enabled=distributed_batch_reduction
    )


def _sequence_class_balanced_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_fraction: float,
    distributed_batch_reduction: bool,
) -> torch.Tensor:
    """Average each class inside each RNA, then give every RNA equal weight.

    This removes three structural count effects from BCE reduction: sequence
    length, number of positive pairs, and number of negative candidate pairs.
    If one class is absent, the available class receives the full sequence
    weight instead of silently shrinking that RNA's loss.
    """
    if values.shape != target.shape or values.shape != valid_mask.shape:
        raise ValueError("values, target, and valid_mask must have the same shape")
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("class_balance_positive_fraction must be within [0, 1]")

    reduce_dims = tuple(range(1, values.ndim))
    positive_mask = valid_mask * (target > 0.5).to(valid_mask.dtype)
    negative_mask = valid_mask * (target <= 0.5).to(valid_mask.dtype)
    positive_count = positive_mask.sum(dim=reduce_dims)
    negative_count = negative_mask.sum(dim=reduce_dims)
    positive_mean = (values * positive_mask).sum(
        dim=reduce_dims
    ) / positive_count.clamp_min(1.0)
    negative_mean = (values * negative_mask).sum(
        dim=reduce_dims
    ) / negative_count.clamp_min(1.0)
    has_positive = positive_count > 0
    has_negative = negative_count > 0
    both = has_positive & has_negative
    per_sequence_loss = torch.where(
        both,
        positive_fraction * positive_mean + (1.0 - positive_fraction) * negative_mean,
        torch.where(has_positive, positive_mean, negative_mean),
    )
    valid_sequences = has_positive | has_negative
    return distributed_weighted_mean(
        per_sequence_loss[valid_sequences].sum(),
        valid_sequences.sum().to(dtype=torch.float32),
        enabled=distributed_batch_reduction,
    )


def masked_contact_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_cfg: dict,
    min_sequence_separation: int,
    distributed_batch_reduction: bool = False,
) -> torch.Tensor:
    pair_mask = build_pair_mask(
        mask, min_sequence_separation=min_sequence_separation
    ).to(logits.dtype)
    # Keep reductions and class weighting in FP32 under BF16 autocast. The
    # logits remain connected to the BF16 forward graph, while large reductions
    # are accumulated in a stable dtype.
    logits_fp32 = logits.float()
    target_fp32 = target.float()
    pair_mask_fp32 = pair_mask.float()
    bce = F.binary_cross_entropy_with_logits(logits_fp32, target_fp32, reduction="none")
    focal_gamma = float(loss_cfg.get("focal_gamma", 0.0))
    if focal_gamma > 0:
        pt = torch.exp(-bce)
        bce = (1.0 - pt).pow(focal_gamma) * bce
    normalization = loss_cfg.get("normalization", "pair")
    if normalization == "per_sequence_class_balanced":
        loss = _sequence_class_balanced_mean(
            bce,
            target_fp32,
            pair_mask_fp32,
            float(loss_cfg.get("class_balance_positive_fraction", 0.5)),
            distributed_batch_reduction,
        )
    else:
        pos_weight = float(loss_cfg.get("pos_weight", 200.0))
        weight = torch.where(
            target_fp32 > 0.5,
            torch.full_like(target_fp32, pos_weight),
            torch.ones_like(target_fp32),
        )
        weighted_mask = weight * pair_mask_fp32
    if normalization == "per_sequence":
        loss = _sequence_normalized_mean(
            bce, weighted_mask, distributed_batch_reduction
        )
    elif normalization == "pair":
        numerator = (bce * weighted_mask).sum()
        denominator = weighted_mask.sum()
        loss = distributed_weighted_mean(
            numerator, denominator, enabled=distributed_batch_reduction
        )
    elif normalization != "per_sequence_class_balanced":
        raise ValueError(
            "loss.normalization must be 'pair', 'per_sequence', or "
            "'per_sequence_class_balanced', "
            f"but got {normalization!r}"
        )

    dice_weight = float(loss_cfg.get("dice_weight", 0.0))
    if dice_weight > 0:
        probs = torch.sigmoid(logits_fp32) * pair_mask_fp32
        target_masked = target_fp32 * pair_mask_fp32
        if normalization in {"per_sequence", "per_sequence_class_balanced"}:
            reduce_dims = (1, 2)
            intersection = (probs * target_masked).sum(dim=reduce_dims)
            denominator = probs.sum(dim=reduce_dims) + target_masked.sum(
                dim=reduce_dims
            )
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            dice = distributed_weighted_mean(
                dice.sum(),
                torch.tensor(float(dice.numel()), device=dice.device),
                enabled=distributed_batch_reduction,
            )
        else:
            intersection = (probs * target_masked).sum()
            denominator = probs.sum() + target_masked.sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        loss = loss + dice_weight * dice

    degree_weight = float(loss_cfg.get("degree_weight", 0.0))
    if degree_weight > 0:
        probs = torch.sigmoid(logits_fp32) * pair_mask_fp32
        degree = probs.sum(dim=-1)
        valid_tokens = mask.to(torch.float32)
        excess_degree = torch.relu(degree - 1.0).square()
        if normalization in {"per_sequence", "per_sequence_class_balanced"}:
            degree = (excess_degree * valid_tokens).sum(dim=1) / valid_tokens.sum(
                dim=1
            ).clamp_min(1.0)
            degree = distributed_weighted_mean(
                degree.sum(),
                torch.tensor(float(degree.numel()), device=degree.device),
                enabled=distributed_batch_reduction,
            )
        else:
            degree = (excess_degree * valid_tokens).sum() / valid_tokens.sum().clamp(
                min=1.0
            )
        loss = loss + degree_weight * degree
    return loss


def masked_pairedness_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_cfg: dict,
    distributed_batch_reduction: bool = False,
) -> torch.Tensor:
    """Masked node-level BCE for whether a nucleotide has any partner."""
    logits_fp32 = logits.float()
    target_fp32 = target.float()
    mask_fp32 = mask.float()
    bce = F.binary_cross_entropy_with_logits(logits_fp32, target_fp32, reduction="none")
    focal_gamma = float(loss_cfg.get("pairedness_focal_gamma", 0.0))
    if focal_gamma > 0:
        bce = (1.0 - torch.exp(-bce)).pow(focal_gamma) * bce
    normalization = loss_cfg.get("normalization", "pair")
    if normalization == "per_sequence_class_balanced":
        return _sequence_class_balanced_mean(
            bce,
            target_fp32,
            mask_fp32,
            float(loss_cfg.get("class_balance_positive_fraction", 0.5)),
            distributed_batch_reduction,
        )
    pos_weight = float(loss_cfg.get("pairedness_pos_weight", 1.0))
    weights = torch.where(
        target_fp32 > 0.5,
        torch.full_like(target_fp32, pos_weight),
        torch.ones_like(target_fp32),
    )
    weighted_mask = weights * mask_fp32
    if normalization == "per_sequence":
        return _sequence_normalized_mean(
            bce, weighted_mask, distributed_batch_reduction
        )
    if normalization != "pair":
        raise ValueError(
            "loss.multitask.normalization must be 'pair', 'per_sequence', or "
            "'per_sequence_class_balanced', "
            f"but got {normalization!r}"
        )
    return distributed_weighted_mean(
        (bce * weighted_mask).sum(),
        weighted_mask.sum(),
        enabled=distributed_batch_reduction,
    )


def masked_pair_type_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_cfg: dict,
    min_sequence_separation: int,
    distributed_batch_reduction: bool = False,
) -> torch.Tensor:
    """Cross-entropy on true contact pairs; -1 labels and invalid pairs are ignored."""
    pair_mask = build_pair_mask(mask, min_sequence_separation=min_sequence_separation)
    valid = pair_mask & (target >= 0)
    class_weights = loss_cfg.get("pair_type_class_weights")
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(
            class_weights, dtype=torch.float32, device=logits.device
        )
    normalization = loss_cfg.get("normalization", "pair")
    if valid.any() and normalization in {
        "per_sequence",
        "per_sequence_class_balanced",
    }:
        safe_target = target.long().masked_fill(~valid, -1)
        per_pair_loss = F.cross_entropy(
            logits.float().permute(0, 3, 1, 2),
            safe_target,
            ignore_index=-1,
            reduction="none",
        )
        pair_weights = valid.to(dtype=torch.float32)
        if weight is not None:
            pair_weights = pair_weights * weight[safe_target.clamp_min(0)]
        return _sequence_normalized_mean(
            per_pair_loss, pair_weights, distributed_batch_reduction
        )
    if normalization not in (
        "pair",
        "per_sequence",
        "per_sequence_class_balanced",
    ):
        raise ValueError(
            "loss.multitask.normalization must be 'pair', 'per_sequence', or "
            "'per_sequence_class_balanced', "
            f"but got {normalization!r}"
        )
    if valid.any():
        local_target = target.long()[valid]
        numerator = F.cross_entropy(
            logits.float()[valid], local_target, weight=weight, reduction="sum"
        )
        denominator = (
            weight[local_target].sum()
            if weight is not None
            else torch.tensor(float(local_target.numel()), device=logits.device)
        )
    else:
        numerator = logits.float().sum() * 0.0
        denominator = torch.zeros((), dtype=torch.float32, device=logits.device)
    return distributed_weighted_mean(
        numerator, denominator, enabled=distributed_batch_reduction
    )


def compute_multitask_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    mask: torch.Tensor,
    loss_cfg: dict,
    min_sequence_separation: int,
    distributed_batch_reduction: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute contact loss plus configured auxiliary losses."""
    multitask_cfg = loss_cfg.get("multitask", {})
    losses = {
        "contact": masked_contact_loss(
            outputs["contact_logits"],
            batch["contact_map"],
            mask,
            loss_cfg,
            min_sequence_separation=min_sequence_separation,
            distributed_batch_reduction=distributed_batch_reduction,
        )
    }
    total = losses["contact"]
    if "pairedness_logits" in outputs:
        losses["pairedness"] = masked_pairedness_loss(
            outputs["pairedness_logits"],
            batch["paired_tokens"],
            mask,
            multitask_cfg,
            distributed_batch_reduction=distributed_batch_reduction,
        )
        total = (
            total
            + float(multitask_cfg.get("pairedness_weight", 0.1)) * losses["pairedness"]
        )
    if "pair_type_logits" in outputs:
        losses["pair_type"] = masked_pair_type_loss(
            outputs["pair_type_logits"],
            batch["pair_type_map"],
            mask,
            multitask_cfg,
            min_sequence_separation=min_sequence_separation,
            distributed_batch_reduction=distributed_batch_reduction,
        )
        total = (
            total
            + float(multitask_cfg.get("pair_type_weight", 0.2)) * losses["pair_type"]
        )
    losses["total"] = total
    return losses
