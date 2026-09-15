"""The cross-attention fusion model.

Three blocks, one per source of signal:

1. **Text encoder** — a pretrained Transformer over the ``capsanocr`` view,
   mean-pooled over the valid tokens. Frozen by default: the head has far fewer
   examples than the encoder has parameters.
2. **Physiological encoder** — polymorphic. ``flat`` runs an MLP over the
   aggregated vector; ``matrix`` runs a *shared* per-subject MLP and pools the
   subjects with attention, so the result does not depend on subject order and
   absent subjects are masked rather than averaged in.
3. **Bidirectional cross-attention fusion** — each modality re-reads the other.
   Concatenation cannot do this: the point is that a strong physiological
   response should change how an ambiguous text is read, and a blunt text
   should change how a weak response is read. Residual connections keep either
   stream usable on its own when the other carries nothing.

One independent model per subtask. The hierarchy is applied at prediction time,
never through shared weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exist2026.taxonomy import SOFT_LABEL_WIDTH, subtask_of


class SharedSubjectMLP(nn.Module):
    """Same weights for every subject: ``(B, S, F) -> (B, S, D)``."""

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128, p_drop: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, subjects, features = x.shape
        return self.net(x.reshape(batch * subjects, features)).reshape(batch, subjects, -1)


class AttentionPool(nn.Module):
    """Permutation-invariant pooling over subjects, with a validity mask."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (torch.tanh(self.proj(x)) * self.query).sum(-1)
        scores = scores.masked_fill(mask < 0.5, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        # An instance with no valid subject at all softmaxes to NaN; fall back
        # to a uniform weighting so the batch survives it.
        weights = torch.nan_to_num(weights, nan=1.0 / x.size(1))
        return (x * weights.unsqueeze(-1)).sum(1), weights


class PhysioEncoder(nn.Module):
    """``flat`` or ``matrix`` front-end, both producing ``(B, out_dim)``."""

    def __init__(
        self, kind: str, in_dim: int, hidden: int = 256, out_dim: int = 128, p_drop: float = 0.2
    ) -> None:
        super().__init__()
        if kind not in ("flat", "matrix"):
            raise ValueError(f"kind must be 'flat' or 'matrix', got {kind!r}")
        self.kind = kind
        self.out_dim = out_dim
        if kind == "flat":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.mlp = SharedSubjectMLP(in_dim, hidden, out_dim, p_drop)
            self.pool = AttentionPool(out_dim)

    def forward(self, batch: Mapping[str, torch.Tensor]):
        if self.kind == "flat":
            return self.net(batch["physio"]), None
        embedded = self.mlp(batch["physio"])
        return self.pool(embedded, batch["physio_mask"])


class CrossAttentionFusion(nn.Module):
    """Bidirectional text <-> physiology attention, concatenated."""

    def __init__(
        self, text_dim: int, physio_dim: int, hidden: int = 512, num_heads: int = 8, p_drop: float = 0.1
    ) -> None:
        super().__init__()
        if hidden % num_heads:
            raise ValueError(f"fusion_hidden={hidden} is not divisible by num_heads={num_heads}")
        self.text_proj = nn.Linear(text_dim, hidden)
        self.physio_proj = nn.Linear(physio_dim, hidden)
        self.text_to_physio = nn.MultiheadAttention(hidden, num_heads, dropout=p_drop, batch_first=True)
        self.physio_to_text = nn.MultiheadAttention(hidden, num_heads, dropout=p_drop, batch_first=True)
        self.norm_text = nn.LayerNorm(hidden)
        self.norm_physio = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p_drop)
        self.out_dim = 2 * hidden

    def forward(self, text_vector: torch.Tensor, physio_vector: torch.Tensor) -> torch.Tensor:
        text = self.text_proj(text_vector).unsqueeze(1)
        physio = self.physio_proj(physio_vector).unsqueeze(1)
        attended_text, _ = self.text_to_physio(text, physio, physio)
        attended_physio, _ = self.physio_to_text(physio, text, text)
        fused_text = self.norm_text(text + self.dropout(attended_text)).squeeze(1)
        fused_physio = self.norm_physio(physio + self.dropout(attended_physio)).squeeze(1)
        return torch.cat([fused_text, fused_physio], dim=-1)


class CrossAttentionModel(nn.Module):
    """One subtask head on top of the fused representation."""

    def __init__(
        self,
        *,
        n_out: int,
        text_model: str,
        physio_kind: str,
        physio_in: int,
        sensor_hidden: int = 256,
        sensor_out: int = 128,
        fusion_hidden: int = 512,
        num_heads: int = 8,
        p_drop: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.text_encoder = AutoModel.from_pretrained(text_model)
        if freeze_backbone:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
        self.physio = PhysioEncoder(physio_kind, physio_in, sensor_hidden, sensor_out, p_drop)
        self.fusion = CrossAttentionFusion(
            self.text_encoder.config.hidden_size, sensor_out, fusion_hidden, num_heads, p_drop
        )
        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Linear(self.fusion.out_dim, n_out)
        self.n_out = n_out

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        encoded = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_vector = self._mean_pool(encoded.last_hidden_state, batch["attention_mask"])
        physio_vector, subject_weights = self.physio(batch)
        fused = self.dropout(self.fusion(text_vector, physio_vector))
        return {"logits": self.head(fused), "subject_weights": subject_weights}


def build_model(key: str, config, hyperparameters: Mapping[str, Any], physio_in: int) -> CrossAttentionModel:
    """Instantiate the model of one subtask from a hyperparameter dict."""
    return CrossAttentionModel(
        n_out=SOFT_LABEL_WIDTH[subtask_of(key)],
        text_model=config.text_model,
        physio_kind=config.physio_kind,
        physio_in=physio_in,
        sensor_hidden=hyperparameters["sensor_hidden"],
        sensor_out=hyperparameters["sensor_out"],
        fusion_hidden=hyperparameters["fusion_hidden"],
        num_heads=hyperparameters["num_heads"],
        p_drop=hyperparameters["p_drop"],
        freeze_backbone=hyperparameters.get("freeze_backbone", True),
    )


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
