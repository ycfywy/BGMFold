# Included benchmark splits

These are exact copies of the active research Parquet files; no new filtering, padding, deduplication, re-splitting or label repair was applied during release preparation. `datasets.json` records per-file counts, length ranges and source-file hashes for provenance, not runtime hash gates.

| Dataset | Split | RNAs | Source/protocol |
|---|---|---:|---|
| bpRNA-spot-0 | Train | 10,814 | current Hugging Face `multimolecule/bprna-spot-0` aligned copy |
| bpRNA-spot-0 | Val | 1,300 | same |
| bpRNA-spot-0 | Test | 1,305 | same |
| RNAStrAlign | tr / Train | 20,923 | PriFold paper-data tr, prepared at length<=600 |
| RNAStrAlign | vl / Val | 2,581 | PriFold paper-data vl, prepared at length<=600 |
| RNAStrAlign | ts / Test | 2,574 | PriFold paper-data ts, prepared at length<=600 |
| ArchiveII | Test | 3,911 | PriFold paper-data at length<=600 |

RNAStrAlign/ArchiveII are **not** the historical HF512 versions. The RNAStrAlign complete pool (`all.parquet`) is deliberately not a training input. ArchiveII is Test only. Prepared source splits and their overlap properties are retained, not redefined as a new leakage-free benchmark. RFold's RNAStrAlign records are not universally identical to these PriFold records.

Required schema: `id`, `sequence`, `secondary_structure`; additional source columns are preserved. Dot-bracket parsing supports `() [] {} <>` and A/a…Z/z pseudoknot tiers. RNA sequences and complete annotations are preserved; no minimum pair distance is imposed. Original dataset/collection ownership and terms remain applicable; this repository does not grant new dataset rights.

Ground-truth Test cohorts were recomputed directly from the included files (see `test_cohorts.json`):

| Test | Full RNAs | GT-pseudoknot RNAs | Empty-GT RNAs | Strict <490 RNAs / GT-pseudoknot RNAs |
|---|---:|---:|---:|---:|
| RNAStrAlign | 2,574 | 364 | 0 | 2,492 / 296 |
| ArchiveII | 3,911 | 1,039 | 5 | 3,845 / 997 |
| bpRNA-spot-0 | 1,305 | 129 | 1 | 1,303 / 129 |

The bpRNA entry reports the full split; its <490 count above is metadata only. Pseudoknot membership means at least one crossing GT edge pair. Empty GT alone does not imply F1=1; the alternate metric changes only both-empty cases.

Use `--data-root /path/to/data` with the same directory layout for another local copy.
