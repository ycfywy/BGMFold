"""PriFold-style, structure-preserving RNA covariation augmentation."""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np


class PriFoldCovariationAugmentation:
    """Apply PriFold's train-only covariation mutation distribution.

    A sample is selected with probability ``select``. Each labelled canonical
    base pair is then independently replaced with probability ``replace`` while
    retaining a canonical AU, CG, or GU pairing class. The transition weights
    are the values used by PriFold's official ``mode='cov'`` implementation.
    """

    _AU_TO_GU: Final[float] = 7.24 / (7.24 + 46.3)
    _CG_TO_GU: Final[float] = 7.24 / (7.24 + 25.77)
    _GU_TO_AU: Final[float] = 25.77 / (25.77 + 46.3)

    def __init__(self, select: float = 0.1, replace: float = 0.3, seed: int = 42):
        if not 0.0 <= select <= 1.0:
            raise ValueError(f"select must be in [0, 1], got {select}")
        if not 0.0 <= replace <= 1.0:
            raise ValueError(f"replace must be in [0, 1], got {replace}")
        self.select = float(select)
        self.replace = float(replace)
        self.seed = int(seed)

    def _rng(self, sample_id: str, epoch: int) -> np.random.Generator:
        payload = f"{self.seed}|{epoch}|{sample_id}".encode("utf-8")
        seed = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        )
        return np.random.default_rng(seed)

    @staticmethod
    def _normalize_base(base: str) -> str:
        return "T" if base.upper() == "U" else base.upper()

    def _replacement_pair(
        self, source_pair: str, rng: np.random.Generator
    ) -> str | None:
        if source_pair in {"AT", "TA"}:
            return "GT" if rng.random() < self._AU_TO_GU else "GC"
        if source_pair in {"CG", "GC"}:
            return "GT" if rng.random() < self._CG_TO_GU else "AT"
        if source_pair in {"GT", "TG"}:
            return "AT" if rng.random() < self._GU_TO_AU else "GC"
        return None

    def __call__(
        self, sequence: str, contact_map: np.ndarray, sample_id: str, epoch: int
    ) -> str:
        if contact_map.shape != (len(sequence), len(sequence)):
            raise ValueError(
                "contact_map shape must match sequence length: "
                f"got {contact_map.shape} for length {len(sequence)}"
            )

        rng = self._rng(sample_id, epoch)
        if rng.random() > self.select:
            return sequence

        sequence_style = (
            "T" if "T" in sequence.upper() and "U" not in sequence.upper() else "U"
        )
        source_bases = [self._normalize_base(base) for base in sequence]
        augmented = list(sequence.upper())
        row_indices, col_indices = np.triu_indices(len(sequence), k=1)
        paired = contact_map[row_indices, col_indices] > 0.5

        for row, col in zip(row_indices[paired], col_indices[paired]):
            if rng.random() >= self.replace:
                continue
            replacement = self._replacement_pair(
                source_bases[row] + source_bases[col], rng
            )
            if replacement is None:
                continue
            if rng.random() >= 0.5:
                replacement = replacement[::-1]
            augmented[row] = replacement[0].replace("T", sequence_style)
            augmented[col] = replacement[1].replace("T", sequence_style)

        return "".join(augmented)
