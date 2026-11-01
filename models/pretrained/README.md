
# Pretrained encoders

Download the following three encoders from Hugging Face into their corresponding local folders. Run all commands from the **BGMFold repository root**.


| Local folder | Model | Hugging Face repository | Use |
|---|---|---|---|
| `models/pretrained/gbrna/` | GB.RNA-1.6B | [genbio-ai/GB.RNA-1.6B](https://huggingface.co/genbio-ai/GB.RNA-1.6B) | Task training and attention analysis |
| `models/pretrained/rnafm/` | RNA-FM | [multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm) | Attention analysis |
| `models/pretrained/rinalmo/` | RiNALMo-Giga | [multimolecule/rinalmo-giga](https://huggingface.co/multimolecule/rinalmo-giga) | Attention analysis |

The RNA-FM and RiNALMo repositories are MultiMolecule's Hugging Face implementations, matching this release's loading interface. For RiNALMo use the **Giga** variant linked above. The two supplied task-training recipes use GB.RNA.



## Download

Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`). Download the configuration/tokenizer files and all weight shards together:

```bash
# GB.RNA: retain the .bin weights and index; avoid downloading a second
# safetensors copy of the same model.
hf download genbio-ai/GB.RNA-1.6B \
  --local-dir models/pretrained/gbrna \
  --exclude "*safetensors*"

# RNA-FM: Hugging Face / MultiMolecule directory, including tokenizer files.
hf download multimolecule/rnafm \
  --local-dir models/pretrained/rnafm

# RiNALMo-Giga: Hugging Face / MultiMolecule directory.
hf download multimolecule/rinalmo-giga \
  --local-dir models/pretrained/rinalmo
```


```text
models/pretrained/
  gbrna/     # genbio-ai/GB.RNA-1.6B
  rnafm/     # multimolecule/rnafm
  rinalmo/   # multimolecule/rinalmo-giga
```


## Use the downloaded encoders

Train the RNAStrAlign multitask model with GB.RNA:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rnastralign.sh \
  --encoder-path models/pretrained/gbrna --output outputs/rnastralign
```

Analyze each encoder on both datasets:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_attention.sh \
  --encoder-family gbrna --encoder-path models/pretrained/gbrna \
  --data-dir data/bprna-spot0 data/rnastralign.600 \
  --out outputs/attention_gbrna

CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_attention.sh \
  --encoder-family rnafm --encoder-path models/pretrained/rnafm \
  --data-dir data/bprna-spot0 data/rnastralign.600 \
  --out outputs/attention_rnafm

CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_attention.sh \
  --encoder-family rinalmo --encoder-path models/pretrained/rinalmo \
  --data-dir data/bprna-spot0 data/rnastralign.600 \
  --out outputs/attention_rinalmo
```

RNA-FM/RiNALMo attention uses the optional dependencies: `pip install -e '.[attention]'`.

These are upstream pretrained encoders. Trained **BGMFold task checkpoints** will be released separately on Hugging Face and belong under `models/checkpoints/`; see [the model overview](../README.md).

