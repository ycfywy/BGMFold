"""Unrestricted partner ranking and independently distance-filtered stem recovery."""

from __future__ import annotations

import numpy as np

DEFAULT_STEM_MIN_SEPARATION = 4


def prepare_structure(
    target, min_stem_length=3, stem_min_separation=DEFAULT_STEM_MIN_SEPARATION
):
    target = np.asarray(target, dtype=bool)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Expected square GT")
    if not np.array_equal(target, target.T) or np.any(np.diag(target)):
        raise ValueError("GT must be symmetric with zero diagonal")
    if np.any(target.sum(axis=1) > 1):
        raise ValueError("At most one GT partner per nucleotide is required")
    if min_stem_length < 1:
        raise ValueError("min_stem_length must be positive")
    if stem_min_separation < 1:
        raise ValueError("stem_min_separation must be positive")
    length = len(target)
    i, j = np.triu_indices(length, k=1)
    edges = set(map(tuple, np.argwhere(np.triu(target, 1)).tolist()))
    lookup = np.full(target.shape, -1, dtype=np.int32)
    lookup[i, j] = np.arange(len(i))
    stem_candidates = (j - i) >= stem_min_separation
    stems = []
    for u, v in sorted(edges):
        if (u - 1, v + 1) in edges:
            continue
        chain = []
        while (u, v) in edges:
            chain.append(lookup[u, v])
            u, v = u + 1, v - 1
        # Extract maximal GT stems FIRST. Reject a complete stem containing
        # any forbidden edge, rather than turning its prefix into a new stem.
        if len(chain) >= min_stem_length and stem_candidates[chain].all():
            stems.append(np.asarray(chain, dtype=int))
    u, v = np.where(target)
    return dict(
        length=length,
        i=i,
        j=j,
        labels=target[i, j],
        endpoints=u,
        partners=v,
        stems=stems,
        stem_candidates=stem_candidates,
    )


def layer_metrics(scores, structure, tie_order):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != structure["labels"].shape or not np.isfinite(scores).all():
        raise ValueError("Invalid attention score vector")
    if np.asarray(tie_order).shape != scores.shape:
        raise ValueError("Invalid tie_order shape")
    length = structure["length"]
    u, v = structure["endpoints"], structure["partners"]
    partner_rank = np.nan
    if length > 2 and len(u):
        dense = np.zeros((length, length), dtype=np.float64)
        dense[structure["i"], structure["j"]] = scores
        dense[structure["j"], structure["i"]] = scores
        candidate = np.arange(length)[None, :] != u[:, None]
        real = dense[u, v, None]
        greater = ((dense[u] > real) & candidate).sum(axis=1)
        equal = ((dense[u] == real) & candidate).sum(axis=1)
        # Average tie rank r=1+greater+(equal-1)/2; n=L-1.
        partner_rank = float((1 - (greater + 0.5 * (equal - 1)) / (length - 2)).mean())
    stem_candidates = np.flatnonzero(structure["stem_candidates"])
    k = int(structure["labels"][stem_candidates].sum())
    selected = np.zeros(len(scores), dtype=bool)
    if k:
        order = np.lexsort(
            (np.asarray(tie_order)[stem_candidates], -scores[stem_candidates])
        )
        selected[stem_candidates[order[:k]]] = True
    stems = structure["stems"]
    return dict(
        partner_rank=partner_rank,
        stem_recovery_1=float(np.mean([selected[g].all() for g in stems]))
        if stems
        else np.nan,
        n_candidates=len(scores),
        n_gt_pairs=int(structure["labels"].sum()),
        n_stem_candidates=len(stem_candidates),
        n_stem_gt_pairs=k,
        n_partner_endpoints=len(u) if length > 2 else 0,
        n_stems=len(stems),
    )
