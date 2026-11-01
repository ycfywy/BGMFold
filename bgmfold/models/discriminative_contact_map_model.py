"""Efficient discriminative RNA secondary-structure contact-map predictor.

The head fuses final RNA encoder token states with encoder self-attention maps
through bidirectional gated FiLM modulation and a learned interaction MLP. It
deliberately avoids the expensive explicit `[h_i, h_j, |h_i-h_j|, h_i*h_j]`
pair construction and multi-block full-resolution pair reconstruction.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from bgmfold.models.rnafm_encoder import (
    RNAEncoderFeatureExtractor,
    resolve_rna_encoder_config,
)


class CrossGatedPairFusion(nn.Module):
    """Fuse broadcast final-token features and attention-derived pair features.

    The attention pair map produces FiLM scale/shift parameters and a gate for
    token-derived pair features. Conversely, token pair features gate the
    attention path. A learned interaction MLP then combines both modulated
    paths and their Hadamard interaction into one pair representation.
    """

    def __init__(self, pair_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(pair_dim)
        self.attn_norm = nn.LayerNorm(pair_dim)
        self.attn_to_seq = nn.Linear(pair_dim, pair_dim * 3)
        self.seq_to_attn_gate = nn.Linear(pair_dim, pair_dim)
        self.fuse = nn.Sequential(
            nn.Linear(pair_dim * 3, pair_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
            nn.LayerNorm(pair_dim),
        )

    def forward(
        self, seq_pair: torch.Tensor, attn_pair: torch.Tensor, pair_mask: torch.Tensor
    ) -> torch.Tensor:
        seq = self.seq_norm(seq_pair)
        attn = self.attn_norm(attn_pair)
        scale, shift, seq_gate = self.attn_to_seq(attn).chunk(3, dim=-1)
        seq_modulated = seq * (1.0 + torch.tanh(scale)) + shift
        seq_modulated = torch.sigmoid(seq_gate) * seq_modulated
        attn_modulated = torch.sigmoid(self.seq_to_attn_gate(seq)) * attn
        fused = self.fuse(
            torch.cat(
                [seq_modulated, attn_modulated, seq_modulated * attn_modulated], dim=-1
            )
        )
        return fused * pair_mask


class SimplePairFusion(nn.Module):
    """Fuse sequence and attention pair features with a plain concatenation MLP."""

    def __init__(self, pair_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(pair_dim)
        self.attn_norm = nn.LayerNorm(pair_dim)
        self.fuse = nn.Sequential(
            nn.Linear(pair_dim * 2, pair_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
            nn.LayerNorm(pair_dim),
        )

    def forward(
        self, seq_pair: torch.Tensor, attn_pair: torch.Tensor, pair_mask: torch.Tensor
    ) -> torch.Tensor:
        fused = self.fuse(
            torch.cat([self.seq_norm(seq_pair), self.attn_norm(attn_pair)], dim=-1)
        )
        return fused * pair_mask


class DepthwisePairSmoother(nn.Module):
    """One inexpensive symmetric local refinement step on pair features."""

    def __init__(
        self, pair_dim: int, kernel_size: int = 3, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "local_smoothing_kernel_size must be a positive odd integer."
            )
        self.norm = nn.LayerNorm(pair_dim)
        self.conv = nn.Conv2d(
            pair_dim,
            pair_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=pair_dim,
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, pair_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair_mask = (
            (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).to(pair_features.dtype)
        )
        update = self.conv(self.norm(pair_features).permute(0, 3, 1, 2))
        update = self.dropout(update).permute(0, 2, 3, 1)
        pair_features = pair_features + update
        pair_features = 0.5 * (pair_features + pair_features.transpose(1, 2))
        return pair_features * pair_mask


class DiscriminativeContactMapPredictor(nn.Module):
    """Final encoder states + attention maps -> symmetric contact logits.

    Shapes:
        hidden: `[B, L, H]` -> projected token states `[B, L, C]`
        attention: `[B, A, L, L]` -> projected pair map `[B, L, L, C]`
        fusion: `[B, L, L, C]` -> logits `[B, L, L]`
    """

    def __init__(
        self,
        rna_encoder_path: str = "models/rnafm",
        rna_encoder_freeze: bool = True,
        rna_encoder_num_attn_layers: int = -1,
        rna_encoder_gradient_checkpointing: bool = False,
        rna_encoder_initialize_from_config_only: bool = False,
        pair_dim: int = 192,
        fusion_dropout: float = 0.1,
        use_local_smoothing: bool = True,
        local_smoothing_kernel_size: int = 3,
        dropout: float = 0.0,
        fusion_mode: str = "cross_gated",
        multitask: dict | None = None,
    ) -> None:
        super().__init__()
        valid_modes = {"1d_only", "2d_only", "simple_fusion", "cross_gated"}
        if fusion_mode not in valid_modes:
            raise ValueError(
                f"Unknown fusion_mode={fusion_mode!r}; expected one of {sorted(valid_modes)}."
            )
        self.fusion_mode = fusion_mode
        use_1d = fusion_mode in {"1d_only", "simple_fusion", "cross_gated"}
        use_2d = fusion_mode in {"2d_only", "simple_fusion", "cross_gated"}
        self.rnafm = RNAEncoderFeatureExtractor(
            rna_encoder_path,
            freeze=rna_encoder_freeze,
            num_attn_layers=rna_encoder_num_attn_layers,
            gradient_checkpointing=rna_encoder_gradient_checkpointing,
            return_attentions=use_2d,
            initialize_from_config_only=rna_encoder_initialize_from_config_only,
        )
        self.seq_proj = (
            nn.Sequential(
                nn.LayerNorm(self.rnafm.hidden_size),
                nn.Linear(self.rnafm.hidden_size, pair_dim),
            )
            if use_1d
            else None
        )
        self.attn_proj = (
            nn.Conv2d(self.rnafm.attn_channels, pair_dim, kernel_size=1)
            if use_2d
            else None
        )
        if fusion_mode == "cross_gated":
            self.fusion = CrossGatedPairFusion(pair_dim, dropout=fusion_dropout)
        elif fusion_mode == "simple_fusion":
            self.fusion = SimplePairFusion(pair_dim, dropout=fusion_dropout)
        else:
            self.fusion = None
        self.single_branch_norm = (
            nn.LayerNorm(pair_dim) if fusion_mode in {"1d_only", "2d_only"} else None
        )
        self.local_smoother = (
            DepthwisePairSmoother(pair_dim, local_smoothing_kernel_size, dropout)
            if use_local_smoothing
            else None
        )
        self.head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 1))

        multitask_cfg = multitask or {}
        pairedness_cfg = multitask_cfg.get("pairedness", {})
        self.multitask_enabled = bool(multitask_cfg.get("enabled", False))
        self.pairedness_enabled = self.multitask_enabled and bool(
            pairedness_cfg.get("enabled", True)
        )
        self.pairedness_source = pairedness_cfg.get("source", "sequence")
        if self.pairedness_source not in {"sequence", "pair_pool"}:
            raise ValueError(
                f"Unknown pairedness source={self.pairedness_source!r}; "
                "expected 'sequence' or 'pair_pool'."
            )
        self.pair_type_enabled = self.multitask_enabled and bool(
            multitask_cfg.get("pair_type", {}).get("enabled", True)
        )
        if (
            self.pairedness_enabled
            and self.pairedness_source == "sequence"
            and self.seq_proj is None
        ):
            raise ValueError(
                "sequence pairedness requires a fusion mode with 1D token features"
            )
        self.pairedness_head = (
            nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 1))
            if self.pairedness_enabled
            else None
        )
        self.pair_type_head = (
            nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(
                    pair_dim,
                    int(multitask_cfg.get("pair_type", {}).get("num_classes", 4)),
                ),
            )
            if self.pair_type_enabled
            else None
        )

    def forward(
        self,
        sequences: List[str],
        return_auxiliary: bool = False,
        pad_to_length: int | None = None,
        return_pair_features: bool = False,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        if return_pair_features and not return_auxiliary:
            raise ValueError("return_pair_features requires return_auxiliary=True")
        if pad_to_length is None:
            features_1d, attn_2d, mask = self.rnafm(sequences)
        else:
            features_1d, attn_2d, mask = self.rnafm(
                sequences, pad_to_length=pad_to_length
            )
        pair_mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)
        seq_features = None

        if self.fusion_mode == "1d_only":
            seq_features = self.seq_proj(features_1d)
            pair_features = 0.5 * (
                seq_features.unsqueeze(2) + seq_features.unsqueeze(1)
            )
            pair_features = self.single_branch_norm(pair_features) * pair_mask.to(
                pair_features.dtype
            )
        elif self.fusion_mode == "2d_only":
            if attn_2d is None:
                raise RuntimeError("2d_only fusion requires encoder attention maps.")
            pair_features = self.attn_proj(attn_2d).permute(0, 2, 3, 1)
            pair_features = self.single_branch_norm(pair_features) * pair_mask.to(
                pair_features.dtype
            )
        else:
            seq_features = self.seq_proj(features_1d)
            # Symmetric broadcast only; no explicit four-way pair feature construction.
            seq_pair = 0.5 * (seq_features.unsqueeze(2) + seq_features.unsqueeze(1))
            if attn_2d is None:
                raise RuntimeError(
                    f"{self.fusion_mode} fusion requires encoder attention maps."
                )
            attn_pair = self.attn_proj(attn_2d).permute(0, 2, 3, 1)
            pair_features = self.fusion(
                seq_pair, attn_pair, pair_mask.to(seq_pair.dtype)
            )

        pair_features = 0.5 * (pair_features + pair_features.transpose(1, 2))
        if self.local_smoother is not None:
            pair_features = self.local_smoother(pair_features, mask)
        logits = self.head(pair_features).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        if not return_auxiliary:
            return logits, mask

        outputs: dict[str, torch.Tensor] = {"contact_logits": logits}
        if return_pair_features:
            outputs["pair_features"] = pair_features
        if self.pairedness_head is not None:
            if self.pairedness_source == "sequence":
                if seq_features is None:
                    raise RuntimeError("sequence pairedness requires token features")
                pairedness_features = seq_features
            else:
                node_mask = pair_mask.squeeze(-1).to(pair_features.dtype)
                pooled = (pair_features * node_mask.unsqueeze(-1)).sum(dim=2)
                normalizer = node_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
                pairedness_features = pooled / normalizer
            outputs["pairedness_logits"] = self.pairedness_head(
                pairedness_features
            ).squeeze(-1)
        if self.pair_type_head is not None:
            pair_type_logits = self.pair_type_head(pair_features)
            outputs["pair_type_logits"] = 0.5 * (
                pair_type_logits + pair_type_logits.transpose(1, 2)
            )
        return outputs, mask


def build_discriminative_contact_map_model(
    cfg: dict,
) -> DiscriminativeContactMapPredictor:
    """Build the efficient direct predictor from the `model` YAML section."""
    encoder_cfg = resolve_rna_encoder_config(cfg)
    return DiscriminativeContactMapPredictor(
        rna_encoder_path=encoder_cfg["model_path"],
        rna_encoder_freeze=encoder_cfg["freeze"],
        rna_encoder_num_attn_layers=encoder_cfg["num_attn_layers"],
        rna_encoder_gradient_checkpointing=encoder_cfg["gradient_checkpointing"],
        rna_encoder_initialize_from_config_only=encoder_cfg[
            "initialize_from_config_only"
        ],
        pair_dim=cfg.get("pair_dim", 192),
        fusion_dropout=cfg.get("fusion_dropout", cfg.get("dropout", 0.1)),
        use_local_smoothing=cfg.get("use_local_smoothing", True),
        local_smoothing_kernel_size=cfg.get("local_smoothing_kernel_size", 3),
        dropout=cfg.get("dropout", 0.0),
        fusion_mode=cfg.get("fusion_mode", "cross_gated"),
        multitask=cfg.get("multitask"),
    )
