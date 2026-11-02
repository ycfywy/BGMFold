# Model weights

## Directory layout

```text
models/
  pretrained/
    gbrna/                          # Complete upstream GB.RNA encoder directory
      config.json
      ...                           # Weight file, or index plus every shard
    rnafm/                          # multimolecule/rnafm
    rinalmo/                        # multimolecule/rinalmo-giga
  checkpoints/
    rnastralign/
      multitask/best.pt             # Trained RNAStrAlign multitask checkpoint
    bprna/
      contact/best.pt               # C
      pairedness/best.pt            # P
      pairtype/best.pt              # T
      multitask/best.pt             # M
```

`pretrained/` holds the original encoder used to initialize training or analyze attention. It must contain the complete Hugging Face-style encoder configuration and weights, not a downstream `best.pt`.

| Pretrained folder | Hugging Face model |
|---|---|
| `pretrained/gbrna/` | [genbio-ai/GB.RNA-1.6B](https://huggingface.co/genbio-ai/GB.RNA-1.6B) |
| `pretrained/rnafm/` | [multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm) |
| `pretrained/rinalmo/` | [multimolecule/rinalmo-giga](https://huggingface.co/multimolecule/rinalmo-giga) |

See [pretrained/README.md](pretrained/README.md) for the three download commands and attention-analysis examples.

`checkpoints/` holds complete task checkpoints collected from your own runs. Independent evaluation loads these checkpoints; it does not require a second copy of the original encoder weights.

### Download trained checkpoints

Trained task checkpoints are distributed as two archives, currently hosted on USTC Pan; they will migrate to Hugging Face later.

| Task | Checkpoints | Download |
|---|---|---|
| bpRNA C/P/T/M | `checkpoints/bprna/{contact,pairedness,pairtype,multitask}/best.pt` | [USTC Pan](https://pan.ustc.edu.cn/share/index/078f2892a4e14cb2b391) |
| RNAStrAlign multitask | `checkpoints/rnastralign/multitask/best.pt` | [USTC Pan](https://pan.ustc.edu.cn/share/index/cc1b94e1ab04400c8090) |

Extract each archive into `models/checkpoints/` so the `best.pt` files land under the paths shown in the layout above.

The **training scripts save to the chosen output directory**, not automatically to `models/`:

| Training run | Validation-best task checkpoints |
|---|---|
| RNAStrAlign | `outputs/rnastralign/multitask/checkpoints/best.pt` |
| bpRNA C/P/T/M | `outputs/bprna_cptm/{contact,pairedness,pairtype,multitask}/checkpoints/best.pt` |

Use `last.pt` from the same directories to resume training, including optimizer/scheduler state. Use `best.pt` for the validation-selected independent evaluation.

Example using a downloaded/collected task checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_rnastralign.sh \
  --checkpoint multitask=models/checkpoints/rnastralign/multitask/best.pt \
  --output outputs/eval_released_rnastralign
```

Run commands from the BGMFold repository root. Local weight contents are ignored by Git; directory placeholders and this README are retained. Model implementation source is in `bgmfold/models/`.

