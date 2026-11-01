"""Exact full-map scoring and stable contact-logit greedy decoding (NumPy only)."""

from __future__ import annotations

import numpy as np


def metrics(counts):
    tp, fp, fn = np.asarray(counts).T
    return np.stack(
        (
            tp / np.maximum(tp + fp, 1),
            tp / np.maximum(tp + fn, 1),
            2 * tp / np.maximum(2 * tp + fp + fn, 1),
        ),
        axis=-1,
    )


def dense_counts(prediction, target):
    p = np.asarray(prediction, dtype=bool)
    g = np.asarray(target) > 0.5
    if p.shape != g.shape or p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Expected matching true-length L x L maps")
    return np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum()], dtype=np.int64)


def unordered_pairs(gt_cells):
    cells = np.asarray(gt_cells, dtype=np.int64).reshape(-1, 2)
    return np.unique(np.sort(cells, axis=1), axis=0)


def has_pseudoknot(gt_cells):
    pairs = unordered_pairs(gt_cells)
    return any(
        i < k < j < l
        for index, (i, j) in enumerate(pairs)
        for k, l in pairs[index + 1 :]
    )


def decode_curve(record, thresholds, return_pairs=False):
    """Predictions are symmetric; each GT cell is counted as actually annotated.

    Unlike an upper-triangle-only scorer, this also handles asymmetric external
    GT and diagonal GT. Self predictions are forbidden; all other distances are
    eligible. Stable ties follow row-major upper-triangle order.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or not len(thresholds)
        or np.any((thresholds <= 0) | (thresholds >= 1))
    ):
        raise ValueError("Thresholds must lie strictly between zero and one")
    if return_pairs and len(thresholds) != 1:
        raise ValueError("Pair export uses one threshold")
    n = len(record["sequence"])
    i, j = np.triu_indices(n, 1)
    scores = np.asarray(record["contact_logits_upper"], dtype=np.float64)
    if scores.shape != i.shape or not np.isfinite(scores).all():
        raise ValueError("Malformed contact logits")
    gt = np.unique(
        np.asarray(record["gt_cells"], dtype=np.int64).reshape(-1, 2), axis=0
    )
    if gt.size and (gt.min() < 0 or gt.max() >= n):
        raise ValueError("GT index outside true sequence length")
    codes = gt[:, 0] * n + gt[:, 1]
    truth = np.isin(i * n + j, codes).astype(np.int64) + np.isin(
        j * n + i, codes
    ).astype(np.int64)
    cutoffs = np.log(thresholds / (1 - thresholds))
    eligible = scores >= cutoffs.min()
    i, j, scores, truth = i[eligible], j[eligible], scores[eligible], truth[eligible]
    order = np.argsort(-scores, kind="stable")
    i, j, scores, truth = i[order], j[order], scores[order], truth[order]
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    cursor = tp = fp = 0
    selected = []
    for index in np.argsort(-cutoffs, kind="stable"):
        while cursor < len(scores) and scores[cursor] >= cutoffs[index]:
            a, b = int(i[cursor]), int(j[cursor])
            if not used[a] and not used[b]:
                used[a] = used[b] = True
                tp += int(truth[cursor])
                fp += 2 - int(truth[cursor])
                if return_pairs:
                    selected.append([a, b])
            cursor += 1
        counts[index] = [tp, fp, len(gt) - tp]
    return counts, selected


def load_record(path):
    with np.load(path, allow_pickle=False) as f:
        record = {k: f[k].copy() for k in f.files}
    for name in ("id", "sequence", "split", "dataset"):
        record[name] = str(record[name].item())
    return record


def summarize(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return dict(
        n=len(rows),
        **{
            f"per_rna_{k}": float(np.mean([r[k] for r in rows])) if rows else None
            for k in ("precision", "recall", "f1")
        },
        per_rna_f1_empty_one=float(np.mean([r["f1_empty_one"] for r in rows]))
        if rows
        else None,
        pooled_f1=2 * tp / max(2 * tp + fp + fn, 1) if rows else None,
        empty_gt_rnas=sum(r["gt_pairs"] == 0 for r in rows),
        both_empty_rnas=sum(r["tp"] + r["fp"] + r["fn"] == 0 for r in rows),
        bad_f1_le05=sum(r["gt_pairs"] > 0 and r["f1"] <= 0.5 for r in rows),
    )
