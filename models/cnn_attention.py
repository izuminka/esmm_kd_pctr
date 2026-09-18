"""CNN + attention multitask model from El-Sherief et al. (IJACSA, 2025).

Shared field embeddings feed a CNN click tower and a dense conversion tower.
Attention transfers click features into the direct impression-level conversion
head. Both outputs are logits; the conversion output is NOT conditional pCVR.
See CNN_ATTENTION.md for paper settings, assumptions, and the loss-sign correction.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _dense_tower(input_dim: int, dims: tuple, dropout: float) -> nn.Sequential:
    layers = []
    for width in dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)))
        input_dim = width
    return nn.Sequential(*layers)


class TaskAttention(nn.Module):
    """Equations 2–4: score each task representation, then pool their values.

    This is attention over two task vectors, not a transformer attention matrix:
    each vector's projected query/key dot product supplies its scalar score.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, representations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Generic multihead attention computes all query/key pairs, whereas the
        # paper scores each representation's own Q(a), K(a) before pooling V(a).
        scores = (self.query(representations) * self.key(representations)).sum(dim=-1)
        weights = torch.softmax(scores / math.sqrt(representations.shape[-1]), dim=1)
        pooled = (weights.unsqueeze(-1) * self.value(representations)).sum(dim=1)
        return pooled, weights


class CNNAttention(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: dict[str, int],
        emb_dim: int = 5,
        conv_channels: tuple = (32, 64, 32),
        kernel_size: int = 3,
        tower_dims: tuple = (128, 64, 32),
        dropout: float = 0.2,
        alpha: float = 1.0,
    ):
        super().__init__()
        if not field_vocab_sizes or emb_dim <= 0:
            raise ValueError("field_vocab_sizes must be nonempty and emb_dim positive")
        if not conv_channels or not tower_dims or any(d <= 0 for d in (*conv_channels, *tower_dims)):
            raise ValueError("conv_channels and tower_dims must contain positive widths")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd for length-preserving padding")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")

        self.field_ids = sorted(field_vocab_sizes)
        self.alpha = alpha
        self.model_params = {
            "emb_dim": emb_dim, "conv_channels": list(conv_channels),
            "kernel_size": kernel_size, "tower_dims": list(tower_dims),
            "dropout": dropout, "alpha": alpha,
        }
        self.embeddings = nn.ModuleDict()
        for field in self.field_ids:
            bag = nn.EmbeddingBag(field_vocab_sizes[field] + 1, emb_dim, mode="sum")
            # Match the existing ESMM initialization: U(-1/sqrt(K), 1/sqrt(K)).
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
            self.embeddings[field] = bag

        input_dim = len(self.field_ids) * emb_dim
        layers = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend((nn.Conv1d(in_channels, out_channels, kernel_size,
                                     stride=1, padding=kernel_size // 2),
                           nn.ReLU(), nn.Dropout(dropout)))
            in_channels = out_channels
        self.convolutions = nn.Sequential(*layers)
        self.click_tower = _dense_tower(input_dim * conv_channels[-1], tower_dims, dropout)
        self.conversion_tower = _dense_tower(input_dim, tower_dims, dropout)
        dim = tower_dims[-1]
        self.click_info = nn.Linear(dim, dim)
        self.attention = TaskAttention(dim)
        self.click_head = nn.Linear(dim, 1)
        self.conversion_head = nn.Linear(dim, 1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """EmbeddingBag input -> (click logits, direct conversion logits), each [B]."""
        pooled = []
        for field in self.field_ids:
            ids, weights, offsets = batch[field]
            pooled.append(self.embeddings[field](ids, offsets, per_sample_weights=weights))
        shared = torch.cat(pooled, dim=1)
        click = self.click_tower(self.convolutions(shared.unsqueeze(1)).flatten(start_dim=1))
        conversion = self.conversion_tower(shared)
        transferred = torch.stack((self.click_info(click), conversion), dim=1)
        attended, _ = self.attention(transferred)
        return self.click_head(click).squeeze(-1), self.conversion_head(attended).squeeze(-1)


def cnn_attention_loss(
    y: torch.Tensor,
    z: torch.Tensor,
    click_logits: torch.Tensor,
    conversion_logits: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Two entire-space BCE terms plus a soft behavioral ordering penalty.

    z is the impression-level conversion label (the data pipeline enforces z<=y).
    Equation 6 prints the reverse sign; penalizing conversion > click implements
    the ordering described in the paper's prose. This is not a hard constraint.
    """
    click_loss = F.binary_cross_entropy_with_logits(click_logits, y)
    conversion_loss = F.binary_cross_entropy_with_logits(conversion_logits, z)
    penalty = F.relu(torch.sigmoid(conversion_logits) - torch.sigmoid(click_logits)).mean()
    return click_loss + conversion_loss + alpha * penalty
