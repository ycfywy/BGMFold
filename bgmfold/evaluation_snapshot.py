"""Optional CPU prediction snapshots; no torch dependency or effect on training RNG."""

import hashlib
import os
from pathlib import Path

import numpy as np


def save_record(root, epoch, split, identity, sequence, logits, target):
    """Save true-length symmetric logits and complete GT, including diagonal GT."""
    n = len(sequence)
    logits, target = np.asarray(logits), np.asarray(target)
    if logits.shape != (n, n) or target.shape != (n, n):
        raise ValueError(f"Snapshot shape mismatch: {identity}")
    if not np.isfinite(logits).all():
        raise ValueError(f"Nonfinite snapshot: {identity}")
    # The packed decoder requires symmetric maps; never silently alter an
    # asymmetric output/GT merely to make the reporting cache fit.
    if not np.array_equal(logits, logits.T) or not np.array_equal(target, target.T):
        raise ValueError(f"Snapshot packing requires symmetric maps: {identity}")
    destination = Path(root) / f"epoch_{epoch:04d}" / "predictions" / split
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (hashlib.sha256(str(identity).encode()).hexdigest() + ".npz")
    temporary = path.with_suffix(f".partial.{os.getpid()}")
    ii, jj = np.triu_indices(n, 1)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                id=np.array(str(identity)),
                sequence=np.array(sequence),
                split=np.array(split),
                contact_logits_upper=logits[ii, jj],
                gt_pairs=np.argwhere(np.triu(target > 0.5)).astype(np.int32),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_batch(root, epoch, split, batch, logits, target):
    """Tensor conversion is per RNA to avoid an additional full GPU batch copy."""
    for index, sequence in enumerate(batch["sequences"]):
        n = len(sequence)
        save_record(
            root,
            epoch,
            split,
            batch["ids"][index],
            sequence,
            logits[index, :n, :n].detach().float().cpu().numpy(),
            target[index, :n, :n].detach().float().cpu().numpy(),
        )
