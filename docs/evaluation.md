# Unified evaluation protocol

## Decoding

Use raw contact logits, stable descending order, and a probability cutoff converted with `log(t/(1-t))`. Accept scores >= this cutoff. Ties follow row-major upper-triangle order. Each nucleotide has at most one partner; no self-pairs; all other distances are eligible. The decoded contact map is symmetric. Auxiliary heads do not gate or reweight decoding. Crossing pairs are allowed.

## Counting and aggregation

For each RNA's true length L, compare every cell of the L×L binary prediction and GT matrices, including the diagonal. Padding is not counted. GT is not silently symmetrized. Thus each correct ordinary symmetric pair contributes 2 TP cells, while a diagonal GT cell is one FN because decoding forbids self-pairs. The public NumPy scorer also counts asymmetric external GT exactly as annotated.

```text
P = TP / max(TP + FP, 1)
R = TP / max(TP + FN, 1)
F1 = 2 TP / max(2 TP + FP + FN, 1)
```

Main F1 is the arithmetic mean of per-RNA F1, including empty-GT RNAs. It is not pooled F1 and not the harmonic mean of already-averaged P/R. `f1_empty_one` changes only the case where GT and prediction are both empty to 1. Empty GT with nonempty prediction still scores 0. Pooled F1 and both-empty counts are separate fields.

Confusion counts use matrix cells; `gt_pairs` and `predicted_pairs` use unique undirected edges. Current included data have symmetric GT with zero diagonals; full-map and upper-triangle P/R/F1 are equivalent on those maps, but the implementation retains the full-map definition.

The preserved training logger uses its existing tiny epsilon in floating-point metric ratios. Independent evaluation uses the explicit integer-count formulas above; training-log comparisons can therefore differ by a few 1e-9 without a counting-protocol difference.

## Threshold selection

Use 110 candidates: 0.01 through 0.99 in steps of 0.01, extra 0.905/0.915/.../0.995, and 0.999. Select the maximum **Val macro F1**, preferring the higher threshold on an exact tie. Lock it for Test. All Test thresholds are saved for diagnostic use, never used to select the reported operating point.

bpRNA uses its validation split. RNAStrAlign and ArchiveII share the threshold selected on RNAStrAlign Val. RNAStrAlign/ArchiveII report the full included dataset and strict length<490. The latter selects a separate threshold on Val<490, using the same inference caches. There is no implicit inference-time length truncation/filtering, and the prepared dataset itself is already <=600.

## Pseudoknots

An RNA belongs to the GT-pseudoknot cohort if GT contains crossing pairs `(i,j)` and `(k,l)` with `i<k<j<l`. Nesting alone is not a pseudoknot. For each Test, report the **complete RNA structure's F1** on this cohort at the same full-Val-selected threshold for the length group. Do not reselect a threshold on pseudoknot Test RNAs. This is not crossing-edge-only F1, pseudoknot motif matching, or a shift-tolerant score.

## Saved evidence

Each model / length group / dataset / split contains:

- `threshold_scan.csv`: 110-point P/R/F1, both-empty-as-one F1, sample counts;
- `per_rna.csv`: selected-threshold exact counts and metrics;
- `per_rna_threshold_counts.json`: every RNA's 110 TP/FP/FN triples;
- `pair_evidence.jsonl.gz`: sequence, GT directed cells, undirected GT pairs, decoded pairs, zero-based indices;
- `summary.json` and, for the RNAStrAlign/ArchiveII Tests, pseudoknot per-RNA and threshold tables.

Root `reports/manifest.json` records model completion/failure; `analysis_bundle.zip` contains reports/logs/configs. Local raw logits remain in `cache/`. `--analyze-only` reuses them without loading PyTorch/model weights. Checkpoint strict-loading detects architecture mismatches; select a matching `--model-config` instead of dropping tensors.
