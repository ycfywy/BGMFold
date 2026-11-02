# BGMFold

RNA secondary-structure prediction using pretrained RNA representations, bidirectional gated feature modulation, and contact-derived node/edge supervision.

[Evaluation protocol](docs/evaluation.md) · [Training recipes](docs/training.md) · [Data](data/README.md)

This repository contains the source code for BGMFold. It includes the core model, **two training entry points**, **two checkpoint-evaluation entry points**, and **one attention-analysis entry point**. Model weights are supplied separately. The included seven Parquet splits are the exact benchmark files used in the current experiments. See [models/README.md](models/README.md) for the local weight layout and the distinction between pretrained encoders and trained task checkpoints.

Upstream encoders: [GB.RNA-1.6B](https://huggingface.co/genbio-ai/GB.RNA-1.6B), [RNA-FM](https://huggingface.co/multimolecule/rnafm), and [RiNALMo-Giga](https://huggingface.co/multimolecule/rinalmo-giga). Their local folders and `hf download` commands are listed in [models/pretrained/README.md](models/pretrained/README.md).

## Installation

Use Python 3.10+ and a CUDA-enabled PyTorch installation appropriate for your GPU/driver. The recorded GPU experiment environment used PyTorch `2.10.0+cu128`; dependency bounds are not a claim that every version combination has been tested.

```bash
cd BGMFold
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional: RNA-FM / RiNALMo attention analysis
pip install -e '.[attention]'
```

## Entry points

| Entry point | GPUs | Purpose |
|---|---:|---|
| `scripts/train_rnastralign.sh` | 2 | One multitask model, up to 400 epochs |
| `scripts/train_bprna_cptm.sh` | 8 | C/P/T/M trained concurrently, two GPUs per model |
| `scripts/evaluate_rnastralign.sh` | 1 | RNAStrAlign + ArchiveII, including GT-pseudoknot RNA cohorts |
| `scripts/evaluate_bprna.sh` | 1 | bpRNA-spot-0 Val/Test |
| `scripts/analyze_attention.sh` | 1 | One pretrained encoder, one or several datasets |

### Train

Supply a **complete local pretrained GB.RNA encoder directory** containing `config.json` and its weight file/shards. The encoder is pretrained; the task head/fusion is initialized for this training run.

After installation, place this directory at `models/pretrained/gbrna/`. From the BGMFold root, start either training recipe below; `--encoder-path` can also point to another existing directory.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rnastralign.sh \
  --encoder-path models/pretrained/gbrna --output outputs/rnastralign

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_bprna_cptm.sh \
  --encoder-path models/pretrained/gbrna --output outputs/bprna_cptm
```

The RNAStrAlign command starts one multitask model on two GPUs; the bpRNA command starts four independent C/P/T/M models on eight GPUs, two GPUs per model. Both train for at most 400 epochs and select `best.pt` by validation F1, with 80-epoch validation patience. Settings are in `configs/train_rnastralign_multitask.yaml` and `configs/train_bprna_cptm.yaml`.

`--dry-run` writes resolved configs and prints the torchrun commands without importing PyTorch or loading weights. Use a separate planning output directory so the actual training output remains fresh:

```bash
bash scripts/train_rnastralign.sh --encoder-path models/pretrained/gbrna \
  --output outputs/plan_rnastralign --dry-run
```

To resume an interrupted run, repeat its command with `--resume`; this restores `checkpoints/last.pt`, including the saved optimizer/scheduler:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rnastralign.sh \
  --encoder-path models/pretrained/gbrna --output outputs/rnastralign --resume
```

See [training details](docs/training.md).

Outputs are local and portable:

```text
outputs/<run>/<role>/
  checkpoints/{best.pt,last.pt,training_state.json}
  configs/training.yaml
  logs/{train.log,launcher.log,history.json,threshold_scan_epoch_*.json,...}
  visualizations/
```

Monitor training with `tail -f outputs/rnastralign/multitask/logs/train.log`, or `tail -f outputs/bprna_cptm/contact/logs/train.log` for C. Substitute `pairedness`, `pairtype` or `multitask` for the other bpRNA roles. After training, evaluate the validation-best checkpoints at the paths below. Training outputs remain under `outputs/`; `models/checkpoints/` is the local location for downloaded or collected release weights.

### Evaluate checkpoints independently

Supply a **complete task checkpoint**, rather than the original pretrained encoder. The default architecture is GB.RNA, cross-gated fusion, pair dimension 512, and the last two attention layers. The evaluator detects C/P/T/M heads from the checkpoint and loads all model tensors strictly. No separate encoder weights are needed for evaluation; the architecture configuration and vocabulary are included.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_rnastralign.sh \
  --checkpoint multitask=outputs/rnastralign/multitask/checkpoints/best.pt \
  --output outputs/eval_rnastralign

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_bprna.sh \
  --checkpoint C=outputs/bprna_cptm/contact/checkpoints/best.pt \
  --checkpoint P=outputs/bprna_cptm/pairedness/checkpoints/best.pt \
  --checkpoint T=outputs/bprna_cptm/pairtype/checkpoints/best.pt \
  --checkpoint M=outputs/bprna_cptm/multitask/checkpoints/best.pt \
  --output outputs/eval_bprna
```

Repeated checkpoints are evaluated **sequentially on one GPU**. `--batch-size` defaults to 1; `--workers` controls CPU scanning. Use `--model-config` for a different supported architecture and `--task` to assert the expected head configuration.

For each checkpoint the outputs retain:

- Val-selected results and all 110 Val/Test thresholds;
- per-RNA TP/FP/FN, precision/recall/F1, and both-empty-as-one F1;
- per-RNA counts for every threshold, plus compressed GT/predicted pair evidence;
- full-set and strict `<490` groups for RNAStrAlign/ArchiveII;
- GT-pseudoknot RNA subsets, scoring **complete RNA structures** at the same Val-selected threshold;
- `analysis_bundle.zip` containing reports/logs/configs. Raw logit caches remain under `cache/` and can be rescored with `--analyze-only`.

### Analyze encoder attention

Supply **one pretrained encoder directory**, not a downstream task checkpoint. The same encoder is loaded once for both datasets.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_attention.sh \
  --encoder-family gbrna --encoder-path models/pretrained/gbrna \
  --data-dir data/bprna-spot0 data/rnastralign.600 \
  --out outputs/attention_gbrna --max-batch-size 8
```

RNA-FM and RiNALMo use the same entry point with `--encoder-family rnafm` or `rinalmo` and a local HF/MultiMolecule directory. Partner ranking excludes self-pairs only. Stem recovery defaults to `j-i>=4`, maximum GT stems of at least three pairs, and complete-stem exclusion when any constituent pair violates this stem-only distance rule. This restriction is **not used by training loss, structure decoding or F1**.

Outputs include per-RNA/per-layer CSVs, layer summaries, bootstrap intervals, encoder metadata and dataset protocols. Add `--save-attention` to retain full directional, head-averaged per-layer attention arrays; the lightweight default retains metrics only. `--limit 20` is an explicitly partial smoke run, and `--resume` continues matching cached work. CUDA OOM splits attention batches and keeps the reduced cap.

## Code layout

```text
bgmfold/
  models/          encoder adapter, gated fusion, contact/pairedness/pair-type heads
  losses.py        contact and auxiliary losses; original DDP reduction preserved
  data/            Parquet loader, pseudoknot-aware dot-bracket parser, augmentation, buckets
  training/        shared training engine and portable two-recipe launcher
  evaluation/      strict loading, independent inference, NumPy greedy decoder and reports
  attention/       one-encoder, multi-dataset forward analysis
  vendor/rnabert/  upstream model/tokenizer implementation and vocabulary
configs/           two complete training YAMLs, task-model YAML, encoder architecture metadata
scripts/           five supported entry points
models/            local pretrained encoders and trained/downloaded task checkpoints
```

Internal package naming is `bgmfold`; checkpoint tensor names such as `rnafm.*`, fusion and task-head keys remain unchanged. Historical checkpoints with the matching architecture remain loadable. No source files outside this directory are imported.

## Verification

```bash
python -m unittest discover -s tests -v
```

CPU checks cover full-matrix counts (including diagonal/asymmetric GT), exact/tied greedy decoding, adjacent pairs, empty cases, pseudoknot cohorts, 110-point scans, `<490`, C/P/T/M configuration, source-equivalent losses, the copied data, and relocation of the entire directory. They do **not** replace a real GPU smoke run of this newly organized release. The training recipes are based on the original GPU experiments.

Upstream notices and ownership are retained; see [THIRD_PARTY.md](THIRD_PARTY.md).
