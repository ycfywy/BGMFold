# Training recipes and implementation

The two complete training YAMLs do not inherit other configurations. Both use the existing data loader, model, loss implementations, DDP normalization, learning-rate scheduler and checkpoint engine. Historical task names in saved tensor keys are preserved.

| Setting | RNAStrAlign multitask | bpRNA C/P/T/M |
|---|---|---|
| Models / GPUs | one M / 2 GPUs | C,P,T,M / 8 GPUs, 2 per model |
| Maximum epochs | 400 | 400 per model |
| Seed | 2528 | 2528 |
| Data augmentation | covariation select=0.1, replace=0.3 | same |
| New-module / encoder LR | 1e-4 / 1e-5 | same |
| Encoder head-only warmup | encoder frozen for 3 epochs | same |
| Scheduler | cosine_with_encoder_warmup; 1-epoch warmups; minimum ratio 0.1 | same |
| Precision | BF16 autocast; normal FP32 model/AdamW; TF32 off | same |
| Val / checkpoint cadence | every 20 epochs | every 20 epochs |
| Early stopping | 80 epochs without Val improvement | independent per role, same patience |
| Sampling | same length bucket across ranks, width 32 | same |
| Per-rank batch | measured table below | budget=20,000,000, power=2.6, min=1, max=8 |
| Gradient accumulation | none | none |

Global batch per model is the sum across its two ranks, not a fixed batch of 16. RNA lengths and bucket tails affect local batch sizes. No extra runtime length limit is configured.

RNAStrAlign uses the exact bucket table recorded in the completed 400-epoch run, replacing hardware calibration with an inspectable fixed configuration:

```yaml
batch_size_by_bucket:
  0: 16
  1: 16
  2: 16
  3: 16
  4: 16
  5: 16
  6: 12
  7: 11
  8: 9
  9: 7
  10: 6
  11: 5
  12: 4
  13: 4
  14: 3
  15: 3
  16: 2
  17: 2
  18: 2
```

Bucket `b` is based on `length // 32`: bucket 0 covers lengths 1–31, bucket 1 covers 32–63, etc. This is a recorded hardware-specific recipe, not a general memory guarantee for every GPU/software combination. Adjust the YAML explicitly when using different hardware.

## Losses

For valid non-diagonal positions of RNA b, with binary label y and logit z:

- `contact = sum(w * BCE(z,y)) / sum(w)`, where w=300 for a positive and 1 for a negative.
- `pairedness = mean(BCE(node_logit, paired_node_label))` over real nucleotides; no class balancing.
- `pair_type = mean(CE(type_logits, type_label))` over GT-paired, non-diagonal positions; AU/GC/GU/other, no class weights.

Each loss is normalized per RNA, then averaged over RNAs with valid supervision for that loss. A local batch with no type labels contributes zero numerator/denominator; global DDP reduction still includes other ranks with valid labels. Auxiliary coefficients are 0.1 each. C uses contact; P adds pairedness; T adds pair-type; M adds both.

`distributed_weighted_mean` compensates DDP's gradient averaging by world_size/global_valid_RNA_count. There is no extra divide by batch after the per-RNA reduction. The loss functions were moved from the research trainer without changing their function AST; the CPU check verifies this move, not actual GPU numeric execution.

## Separation of training and evaluation

The training launcher starts one continuous DDP process group per model. Validation remains inside the training engine for threshold/checkpoint selection. The historical RNAStrAlign Test monitoring schedule (every 40 epochs and best-at-end) is retained; bpRNA training has Test monitoring disabled. Neither uses Test to choose weights.

Final reusable reports, length groups, and pseudoknot cohorts are produced by the independent evaluation scripts. The orchestration is simplified, so this release preserves the training recipe rather than promising a bit-identical trajectory to every segmented historical platform run.

## Resume

Use the same output directory with `--resume`. Last checkpoints include model, optimizer, scheduler, epoch and training-state metadata. Completed/early-stopped roles should retain their existing results rather than restart. The source encoder directory is still required to construct the same model/tokenizer.
