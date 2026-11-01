"""RNA language-model feature extraction for BGMFold.

The adapter accepts local MultiMolecule RNA checkpoints (including RNA-FM and
RiNALMo) plus the official GB.RNA-1.6B RNABert checkpoint, and exposes a stable
feature contract to downstream models:

* per-nucleotide hidden states: ``[B, L, hidden_size]``;
* stacked self-attention maps: ``[B, selected_layers * num_heads, L, L]``;
* a nucleotide validity mask: ``[B, L]``.

The downstream pair modules infer their input widths from this adapter, so a
checkpoint with a different hidden size or number of attention layers can be
selected without architecture changes. The encoder is frozen by default to
avoid retaining its transformer activations during downstream training.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class RNAEncoderFeatureExtractor(nn.Module):
    """Wrap a local single-nucleotide RNA encoder and expose 1D/2D features."""

    def __init__(
        self,
        model_path: str = "models/rnafm",
        freeze: bool = True,
        num_attn_layers: int = -1,
        gradient_checkpointing: bool = False,
        return_attentions: bool = True,
        initialize_from_config_only: bool = False,
    ):
        """
        Args:
            model_path: Local RNA checkpoint directory. MultiMolecule checkpoints
                and GB.RNA-1.6B's ``model_type=rnabert`` checkpoint are supported.
            freeze: Keep the encoder in eval mode and exclude it from gradients.
            num_attn_layers: Number of final transformer attention layers to retain;
                ``-1`` uses every layer.
            gradient_checkpointing: Recompute encoder block activations in backward
                when the encoder is trainable to reduce peak memory.
            return_attentions: Whether to request and return encoder self-attention maps.
            initialize_from_config_only: Construct the encoder architecture without
                reading pretrained encoder tensors. This is only safe when a full
                task checkpoint is loaded immediately afterwards.
        """
        super().__init__()
        checkpoint_dir = Path(model_path).expanduser()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"RNA encoder checkpoint directory does not exist: {checkpoint_dir}"
            )

        with (checkpoint_dir / "config.json").open() as config_file:
            checkpoint_config = json.load(config_file)
        self.model_type = checkpoint_config.get("model_type")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.return_attentions = bool(return_attentions)
        self.initialize_from_config_only = bool(initialize_from_config_only)
        self._uses_spaced_tokens = self.model_type == "rnabert"

        if self._uses_spaced_tokens:
            from bgmfold.vendor.rnabert.configuration_rnabert import RNABertConfig
            from bgmfold.vendor.rnabert.modeling_rnabert import RNABertModel
            from bgmfold.vendor.rnabert.tokenization_rnabert import RNABertTokenizer

            vocab_path = (
                Path(__file__).resolve().parents[1] / "vendor" / "rnabert" / "vocab.txt"
            )
            self.tokenizer = RNABertTokenizer(str(vocab_path), version="v2")
            config = RNABertConfig.from_pretrained(str(checkpoint_dir))
            self.model = (
                RNABertModel(config)
                if self.initialize_from_config_only
                else RNABertModel.from_pretrained(str(checkpoint_dir), config=config)
            )
        else:
            # Registers MultiMolecule architectures such as RNA-FM and RiNALMo with
            # Hugging Face's AutoModel/AutoTokenizer factories.
            import multimolecule  # noqa: F401
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
            self.model = (
                # Full task checkpoints still need eager attention: their 2D
                # branch consumes attention maps, which SDPA may not return.
                AutoModel.from_config(
                    AutoConfig.from_pretrained(str(checkpoint_dir)),
                    attn_implementation="eager",
                )
                if self.initialize_from_config_only
                else AutoModel.from_pretrained(
                    str(checkpoint_dir), attn_implementation="eager"
                )
            )
            config = self.model.config

        self.freeze = bool(freeze)
        self.runtime_freeze = self.freeze
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_attn_layers = self._resolve_num_attn_layers(num_attn_layers)
        self.attn_channels = self.num_attn_layers * self.num_heads
        self.attention_layer_indices = tuple(
            range(self.num_layers - self.num_attn_layers, self.num_layers)
        )

        if self.gradient_checkpointing and not freeze:
            self._enable_gradient_checkpointing()
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _resolve_num_attn_layers(self, num_attn_layers: int) -> int:
        if num_attn_layers in (-1, None):
            return self.num_layers
        num_attn_layers = int(num_attn_layers)
        if not 1 <= num_attn_layers <= self.num_layers:
            raise ValueError(
                f"num_attn_layers must be in [1, {self.num_layers}] or -1, "
                f"but got {num_attn_layers}."
            )
        return num_attn_layers

    def _enable_gradient_checkpointing(self) -> None:
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "The selected RNA encoder does not support gradient checkpointing."
            )
        self.model.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.runtime_freeze:
            # Keep a frozen encoder deterministic regardless of the outer model's
            # train()/eval() calls.
            self.model.eval()
        return self

    def set_runtime_freeze(self, frozen: bool) -> None:
        """Temporarily detach a trainable encoder for head-only warm-up."""
        if self.freeze and not frozen:
            raise RuntimeError(
                "A permanently frozen encoder cannot be runtime-unfrozen; "
                "set model.rna_encoder_freeze=false for phased training."
            )
        self.runtime_freeze = self.freeze or bool(frozen)
        if self.runtime_freeze:
            self.model.eval()
        else:
            self.model.train(self.training)

    def _validate_tokenization(
        self, encoded: dict[str, torch.Tensor], lengths: List[int]
    ) -> None:
        """Ensure the character-level RNA tokenizer preserves nucleotide alignment."""
        token_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        expected_lengths = [length + 2 for length in lengths]  # <cls>, sequence, <eos>
        if token_lengths != expected_lengths:
            raise ValueError(
                "The RNA tokenizer must produce one token per nucleotide plus "
                f"<cls>/<eos>; got token lengths {token_lengths} for sequence lengths {lengths}."
            )

    def forward(
        self, sequences: List[str], pad_to_length: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            features_1d: ``[B, L, hidden_size]``.
            attn_2d: ``[B, attn_channels, L, L]`` or ``None`` when attentions
                are disabled.
            mask: ``[B, L]`` boolean validity mask.
        """
        if not sequences:
            raise ValueError("At least one RNA sequence is required.")

        device = next(self.model.parameters()).device
        lengths = [len(sequence) for sequence in sequences]
        if any(length == 0 for length in lengths):
            raise ValueError("Empty RNA sequences are not supported.")
        max_len = max(lengths)
        if pad_to_length is not None:
            if pad_to_length < max_len:
                raise ValueError(
                    f"pad_to_length={pad_to_length} is smaller than sequence length {max_len}"
                )
            max_len = pad_to_length

        tokenizer_inputs = (
            [" ".join(sequence.upper()) for sequence in sequences]
            if self._uses_spaced_tokens
            else list(sequences)
        )
        encoded = self.tokenizer(tokenizer_inputs, return_tensors="pt", padding=True)
        self._validate_tokenization(encoded, lengths)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        model_kwargs = {"output_attentions": self.return_attentions}
        if self._uses_spaced_tokens:
            model_kwargs["attention_layer_indices"] = self.attention_layer_indices
        context = torch.no_grad() if self.runtime_freeze else nullcontext()
        with context:
            outputs = self.model(**encoded, **model_kwargs)

        hidden = outputs.last_hidden_state
        attentions = outputs.attentions
        if self.return_attentions and attentions is None:
            raise RuntimeError(
                "RNA encoder did not return attention maps with output_attentions=True."
            )

        batch_size = hidden.size(0)
        features_1d = torch.zeros(
            batch_size, max_len, self.hidden_size, device=device, dtype=hidden.dtype
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        attn_stack = None
        if self.return_attentions:
            selected_attentions = (
                attentions
                if self._uses_spaced_tokens
                else attentions[-self.num_attn_layers :]
            )
            if len(selected_attentions) != self.num_attn_layers:
                raise RuntimeError(
                    f"Expected {self.num_attn_layers} attention layers, got {len(selected_attentions)}."
                )
            attn_stack = torch.stack(selected_attentions, dim=1).flatten(1, 2)
            attn_2d = torch.zeros(
                batch_size,
                self.attn_channels,
                max_len,
                max_len,
                device=device,
                dtype=attn_stack.dtype,
            )
        else:
            attn_2d = None

        for index, length in enumerate(lengths):
            # MultiMolecule's RNA tokenizers use <cls> + one token/nucleotide + <eos>.
            nucleotide_slice = slice(1, 1 + length)
            features_1d[index, :length] = hidden[index, nucleotide_slice]
            mask[index, :length] = True
            if attn_stack is not None:
                attn_2d[index, :, :length, :length] = attn_stack[
                    index, :, nucleotide_slice, nucleotide_slice
                ]

        return features_1d, attn_2d, mask


def resolve_rna_encoder_config(cfg: dict) -> dict:
    """Resolve generic encoder settings while accepting legacy ``rnafm_*`` keys."""
    return {
        "model_path": cfg.get(
            "rna_encoder_path", cfg.get("rnafm_path", "models/rnafm")
        ),
        "freeze": cfg.get("rna_encoder_freeze", cfg.get("rnafm_freeze", True)),
        "num_attn_layers": cfg.get(
            "rna_encoder_num_attn_layers", cfg.get("rnafm_num_attn_layers", -1)
        ),
        "gradient_checkpointing": cfg.get("rna_encoder_gradient_checkpointing", False),
        "initialize_from_config_only": cfg.get(
            "rna_encoder_initialize_from_config_only", False
        ),
    }


# Backward-compatible public name for existing imports and checkpoints.
RNAFMFeatureExtractor = RNAEncoderFeatureExtractor
