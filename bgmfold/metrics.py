"""Evaluation metrics for RNA secondary-structure (contact-map) prediction."""

from __future__ import annotations

import math
from typing import Dict

import torch

from bgmfold.utils import build_pair_mask

_EPS = 1e-8


def probability_threshold_logit(threshold: float) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must be within [0, 1]")
    if threshold == 0.0:
        return -math.inf
    if threshold == 1.0:
        return math.inf
    return math.log(threshold / (1.0 - threshold))


@torch.no_grad()
def confusion_counts_per_sequence(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_sequence_separation: int = 1,
) -> torch.Tensor:
    """Score every real-length LxL cell, including both directions and diagonal.

    Padding is excluded. Decoding (not scoring) disallows self-pairs. No
    symmetrization or distance filtering silently changes external matrices.
    """
    pair_mask = mask.bool().unsqueeze(1) & mask.bool().unsqueeze(2)
    preds_bool = preds.bool() & pair_mask
    target_bool = (target > 0.5) & pair_mask
    negative = pair_mask & (~target_bool)
    reduce_dims = (1, 2)
    return torch.stack(
        (
            (preds_bool & target_bool).sum(dim=reduce_dims),
            (preds_bool & (~target_bool) & pair_mask).sum(dim=reduce_dims),
            ((~preds_bool) & target_bool).sum(dim=reduce_dims),
            ((~preds_bool) & negative).sum(dim=reduce_dims),
        ),
        dim=1,
    ).to(dtype=torch.float64)


@torch.no_grad()
def confusion_counts_from_preds(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_sequence_separation: int = 1,
) -> torch.Tensor:
    """Return dataset-additive ``[tp, fp, fn, tn]`` counts for a batch."""
    return confusion_counts_per_sequence(
        preds, target, mask, min_sequence_separation=min_sequence_separation
    ).sum(dim=0)


def metrics_per_sequence_from_confusion_counts(counts: torch.Tensor) -> torch.Tensor:
    """Return ``[precision, recall, f1, mcc, accuracy]`` for each RNA."""
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("counts must have shape [B, 4]")
    tp, fp, fn, tn = counts.to(dtype=torch.float64).unbind(dim=1)
    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    accuracy = (tp + tn) / (tp + fp + fn + tn + _EPS)
    mcc_denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + _EPS)
    mcc = (tp * tn - fp * fn) / mcc_denom
    return torch.stack((precision, recall, f1, mcc, accuracy), dim=1)


def metrics_from_confusion_counts(counts: torch.Tensor) -> Dict[str, float]:
    """Compute metrics once from dataset-level ``[tp, fp, fn, tn]`` counts."""
    tp, fp, fn, tn = counts.to(dtype=torch.float64).unbind()
    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    accuracy = (tp + tn) / (tp + fp + fn + tn + _EPS)
    mcc_denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + _EPS)
    mcc = (tp * tn - fp * fn) / mcc_denom
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "mcc": mcc.item(),
        "accuracy": accuracy.item(),
    }


@torch.no_grad()
def logits_to_probs_and_preds(logits: torch.Tensor, threshold: float = 0.5):
    # Evaluation commonly runs under BF16 autocast. Keep the public metric
    # boundary in FP32 so CPU/NumPy consumers and thresholding behave
    # consistently regardless of the forward-pass precision.
    probs = torch.sigmoid(logits.float())
    preds = (probs >= threshold).float()
    return probs, preds


@torch.no_grad()
def greedy_project_contacts(
    probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    min_sequence_separation: int = 1,
    ranking_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy RNA decoding: thresholded pairs plus at most one partner per base."""

    batch_size, length, _ = probs.shape
    pair_mask = build_pair_mask(mask, min_sequence_separation=min_sequence_separation)
    preds = torch.zeros_like(probs, dtype=torch.bool)
    upper_mask = torch.triu(pair_mask, diagonal=1)
    probs = (probs.float() + probs.float().transpose(1, 2)) * 0.5
    if ranking_logits is not None:
        ranking_logits = (
            ranking_logits.double() + ranking_logits.double().transpose(1, 2)
        ) * 0.5

    for b in range(batch_size):
        # Optional continuation mode matches the cache decoder: threshold raw
        # logits in FP64, stable row-major ties; sigmoid saturation cannot reorder edges.
        rank_scores = probs[b] if ranking_logits is None else ranking_logits[b].double()
        cutoff = (
            threshold
            if ranking_logits is None
            else probability_threshold_logit(threshold)
        )
        candidate_mask = upper_mask[b] & (rank_scores >= cutoff)
        coords = candidate_mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue
        scores = rank_scores[candidate_mask]
        order = torch.argsort(
            scores, descending=True, stable=ranking_logits is not None
        )
        # Transfer the sorted coordinates once. Reading GPU scalars inside the
        # Python loop would otherwise force one CUDA synchronization per edge.
        sorted_coords = coords[order].cpu().tolist()
        used = [False] * length
        selected_i = []
        selected_j = []
        for i, j in sorted_coords:
            if used[i] or used[j]:
                continue
            used[i] = True
            used[j] = True
            selected_i.append(i)
            selected_j.append(j)
        if selected_i:
            rows = torch.tensor(selected_i, device=probs.device)
            cols = torch.tensor(selected_j, device=probs.device)
            preds[b, rows, cols] = True
            preds[b, cols, rows] = True
    return preds.float()


@torch.no_grad()
def compute_metrics_from_preds(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_sequence_separation: int = 1,
) -> Dict[str, float]:
    """Compute per-RNA metrics, then return their macro average."""

    counts = confusion_counts_per_sequence(
        preds, target, mask, min_sequence_separation=min_sequence_separation
    )
    values = metrics_per_sequence_from_confusion_counts(counts)
    names = ("precision", "recall", "f1", "mcc", "accuracy")
    return {
        name: float(values[:, index].mean().item()) for index, name in enumerate(names)
    }


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    min_sequence_separation: int = 1,
) -> Dict[str, float]:
    """Compute per-RNA Precision / Recall / F1 / MCC / Accuracy and macro-average.

    All real-length LxL cells are scored, including the diagonal.
    """
    _, preds = logits_to_probs_and_preds(logits, threshold)
    return compute_metrics_from_preds(
        preds,
        target,
        mask,
        min_sequence_separation=min_sequence_separation,
    )
