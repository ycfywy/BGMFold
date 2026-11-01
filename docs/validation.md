# Local release verification — 2026-11-01

Result: **12 CPU checks passed**. No PyTorch/model installation or GPU training/inference was performed during release preparation.

```bash
python -m unittest discover -s tests -v
```

Covered behavior:

- contact-logit greedy decoding, adjacent pairs, exact threshold boundary and stable ties;
- agreement with an independently reconstructed dense-matrix scorer, including asymmetric and diagonal GT;
- empty-prediction/empty-GT definitions and the crossing-pair pseudoknot criterion;
- both benchmark branches, all 110 thresholds, strict <490 groups and fixed Val-selected PK cohort scores;
- CPU-only cache replay through the public CLI, spawn workers and downloadable report bundles;
- C/P/T/M role configuration, 400-epoch recipes and the measured RNAStrAlign bucket table;
- completed and early-stopped roles remain stopped on suite resume;
- model-definition and loss-function AST agreement with the research source, normalizing the Python package name and empty Python-3.12-only AST fields;
- all seven copied Parquet splits retain their source bytes and counts;
- the entire source directory can be copied elsewhere; all three module CLIs and the four-role dry-run operate from that copy without parent-project imports.

Python syntax, shell syntax, unresolved-name checks and README relative links were checked as well. Runtime behavior involving actual GPU kernels, distributed collectives, weight loading and end-to-end numerical agreement still needs a smoke run in the intended CUDA environment. The original GPU runs supply the recorded recipes and data; they are not represented as new GPU tests of this reorganized directory.
