"""Dot-bracket <-> contact-map conversion utilities.

Supports pseudoknots via the common multi-tier bracket convention used by
bpRNA / forgi: `()`, `[]`, `{}`, `<>` and uppercase/lowercase Latin letter
pairs (`Aa`, `Bb`, ... `Zz`) as extra tiers beyond the standard four.
"""

from __future__ import annotations

import string
from typing import List, Tuple

import numpy as np

_BASIC_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
_LETTER_PAIRS = [(u, l) for u, l in zip(string.ascii_uppercase, string.ascii_lowercase)]
BRACKET_PAIRS = _BASIC_PAIRS + _LETTER_PAIRS

_OPEN2CLOSE = {o: c for o, c in BRACKET_PAIRS}
_CLOSE2OPEN = {c: o for o, c in BRACKET_PAIRS}
_UNPAIRED_CHARS = {".", "-", "_", ":", ",", "~"}
_PAIR_TYPE_TO_ID = {
    frozenset(("A", "U")): 0,
    frozenset(("C", "G")): 1,
    frozenset(("G", "U")): 2,
}
PAIR_TYPE_NAMES = ("AU", "GC", "GU", "other")


def dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Parse a (possibly pseudoknotted) dot-bracket string into index pairs."""
    stacks = {open_c: [] for open_c, _ in BRACKET_PAIRS}
    pairs: List[Tuple[int, int]] = []
    for i, c in enumerate(structure):
        if c in stacks:
            stacks[c].append(i)
        elif c in _CLOSE2OPEN:
            open_c = _CLOSE2OPEN[c]
            if stacks[open_c]:
                j = stacks[open_c].pop()
                pairs.append((j, i))
            # unmatched closing bracket: silently ignore (malformed input)
        # anything else (including '.') is treated as unpaired
    return pairs


def pairs_to_contact_map(
    pairs: List[Tuple[int, int]], length: int, dtype=np.float32
) -> np.ndarray:
    """Build a symmetric [length, length] binary contact map from base-pair indices."""
    contact_map = np.zeros((length, length), dtype=dtype)
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length:
            contact_map[i, j] = 1
            contact_map[j, i] = 1
    return contact_map


def dotbracket_to_contact_map(structure: str, dtype=np.float32) -> np.ndarray:
    """Convenience wrapper: dot-bracket string -> symmetric contact map."""
    pairs = dotbracket_to_pairs(structure)
    return pairs_to_contact_map(pairs, len(structure), dtype=dtype)


def dotbracket_to_pair_type_map(sequence: str, structure: str) -> np.ndarray:
    """Return symmetric pair-type labels, with -1 for non-pairs or unknown bases.

    Labels are 0=AU, 1=GC, 2=GU, and 3=other. The ``other`` class includes
    chemically valid but non-canonical pairs and is deliberately not treated as
    a negative contact label.
    """
    if len(sequence) != len(structure):
        raise ValueError("sequence and structure must have the same length")
    pair_type_map = np.full((len(sequence), len(sequence)), -1, dtype=np.int64)
    normalized = sequence.upper().replace("T", "U")
    for left, right in dotbracket_to_pairs(structure):
        bases = (normalized[left], normalized[right])
        if any(base not in {"A", "C", "G", "U"} for base in bases):
            continue
        pair_type = _PAIR_TYPE_TO_ID.get(frozenset(bases), 3)
        pair_type_map[left, right] = pair_type
        pair_type_map[right, left] = pair_type
    return pair_type_map
