"""CNN + attention multitask model from El-Sherief et al. (IJACSA, 2025).

Shared field embeddings feed a CNN click tower and a dense conversion tower.
Attention transfers click features into the direct impression-level conversion
head. Both outputs are logits; the conversion output is NOT conditional pCVR.

Architecture version 2 makes cold-start inputs and the task gate robust:
weighted-mean pooling, a trainable ID-0 fallback for inactive fields, explicit
presence features, bounded cosine attention over normalized representations,
and a conversion residual. Version 1 remains available solely to restore old
checkpoints exactly. See CNN_ATTENTION.md for the full protocol.

Bag statistics are an opt-in extension shared with the matched DCN recipe. They
retain bounded token-count and weight-mass signals alongside either sum or mean
pooling; leaving the option disabled restores existing checkpoints exactly.
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
    each vector's projected query/key similarity supplies its scalar score.
    """

    def __init__(self, dim: int, score_mode: str = "cosine"):
        super().__init__()
        if score_mode not in ("cosine", "dot"):
            raise ValueError("score_mode must be 'cosine' or 'dot'")
        self.score_mode = score_mode
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, representations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(representations)
        key = self.key(representations)
        if self.score_mode == "cosine":
            # The old unbounded self-dot product reached score gaps above 6,000
            # and underflowed one task's weight to zero. Cosine scores stay in
            # [-1, 1], preserving a usable gradient for both task paths.
            scores = F.cosine_similarity(query, key, dim=-1)
        else:
            scores = (query * key).sum(dim=-1) / math.sqrt(representations.shape[-1])
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * self.value(representations)).sum(dim=1)
        return pooled, weights


def _bag_weight_state(
    weights: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each token's bag id, bag weight mass, and bag token count."""
    ends = torch.cat((offsets[1:], offsets.new_tensor([weights.numel()])))
    lengths = ends - offsets
    bag_ids = torch.repeat_interleave(
        torch.arange(offsets.numel(), device=offsets.device),
        lengths,
        output_size=weights.numel(),
    )
    mass = weights.new_zeros(offsets.numel())
    mass.scatter_add_(0, bag_ids, weights.abs())
    return bag_ids, mass, lengths


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
        architecture_version: int = 2,
        pooling: str = "mean",
        bag_stats: bool = False,
        use_missing_embedding: bool = True,
        use_presence_features: bool = True,
        normalize_task_representations: bool = True,
        attention_score: str = "cosine",
        conversion_residual: bool = True,
        field_dropout: dict[str, float] | None = None,
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
        if architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        if pooling not in ("sum", "mean"):
            raise ValueError("pooling must be 'sum' or 'mean'")
        if attention_score not in ("cosine", "dot"):
            raise ValueError("attention_score must be 'cosine' or 'dot'")

        # Old checkpoints predate every robustness option. Forcing their exact
        # behavior here makes version-1 restoration independent of today's
        # constructor defaults.
        if architecture_version == 1:
            pooling = "sum"
            bag_stats = False
            use_missing_embedding = False
            use_presence_features = False
            normalize_task_representations = False
            attention_score = "dot"
            conversion_residual = False
            field_dropout = {}

        self.field_ids = sorted(field_vocab_sizes)
        self.alpha = alpha
        self.architecture_version = architecture_version
        self.pooling = pooling
        self.bag_stats = bool(bag_stats)
        self.use_missing_embedding = use_missing_embedding
        self.use_presence_features = use_presence_features
        self.normalize_task_representations = normalize_task_representations
        self.conversion_residual = conversion_residual
        self.field_dropout = {
            str(field): float(rate) for field, rate in (field_dropout or {}).items()
        }
        unknown_dropout_fields = set(self.field_dropout) - set(self.field_ids)
        if unknown_dropout_fields:
            raise ValueError(f"field_dropout names unknown fields: {sorted(unknown_dropout_fields)}")
        if any(
            not math.isfinite(rate) or not 0 <= rate < 1
            for rate in self.field_dropout.values()
        ):
            raise ValueError("field_dropout rates must be finite and lie in [0, 1)")

        self.model_params = {
            "emb_dim": emb_dim,
            "conv_channels": list(conv_channels),
            "kernel_size": kernel_size,
            "tower_dims": list(tower_dims),
            "dropout": dropout,
            "alpha": alpha,
            "architecture_version": architecture_version,
            "pooling": pooling,
            "use_missing_embedding": use_missing_embedding,
            "use_presence_features": use_presence_features,
            "normalize_task_representations": normalize_task_representations,
            "attention_score": attention_score,
            "conversion_residual": conversion_residual,
            "field_dropout": self.field_dropout,
        }
        self.embeddings = nn.ModuleDict()
        if self.bag_stats:
            self.model_params["bag_stats"] = True
        for field in self.field_ids:
            bag = nn.EmbeddingBag(field_vocab_sizes[field] + 1, emb_dim, mode="sum")
            # Match the existing ESMM initialization: U(-1/sqrt(K), 1/sqrt(K)).
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
            self.embeddings[field] = bag

        per_field_dim = emb_dim + 2 * int(self.bag_stats) + int(use_presence_features)
        input_dim = len(self.field_ids) * per_field_dim
        layers = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                (
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels
        self.convolutions = nn.Sequential(*layers)
        self.click_tower = _dense_tower(input_dim * conv_channels[-1], tower_dims, dropout)
        self.conversion_tower = _dense_tower(input_dim, tower_dims, dropout)
        dim = tower_dims[-1]
        self.click_info = nn.Linear(dim, dim)
        self.attention = TaskAttention(dim, score_mode=attention_score)
        self.click_head = nn.Linear(dim, 1)
        self.conversion_head = nn.Linear(dim, 1)
        self.last_attention_weights = None

    def _pool_field(
        self,
        field: str,
        encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        forced_missing: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool one field and route inactive/dropped bags to its ID-0 vector."""
        ids, weights, offsets = encoded
        bag = self.embeddings[field]
        if self.architecture_version == 1:
            return bag(ids, offsets, per_sample_weights=weights)

        bag_ids, mass, lengths = _bag_weight_state(weights, offsets)
        active = mass > 0
        pooled_weights = weights
        if self.pooling == "mean":
            safe_mass = mass.masked_fill(~active, 1.0)
            pooled_weights = weights / safe_mass[bag_ids]
        value = bag(ids, offsets, per_sample_weights=pooled_weights)

        missing = ~active
        dropout_rate = self.field_dropout.get(field, 0.0)
        if self.training and dropout_rate:
            missing = missing | (
                torch.rand(active.shape, device=active.device) < dropout_rate
            )
        if forced_missing is not None:
            if forced_missing.shape != active.shape:
                raise ValueError(
                    f"forced-missing mask for field {field!r} has shape "
                    f"{tuple(forced_missing.shape)}, expected {tuple(active.shape)}"
                )
            missing = missing | forced_missing.to(device=active.device, dtype=torch.bool)

        if self.use_missing_embedding:
            fallback = bag.weight[0].unsqueeze(0).expand_as(value)
            value = torch.where(missing.unsqueeze(1), fallback, value)
        else:
            value = value.masked_fill(missing.unsqueeze(1), 0.0)
        if self.bag_stats:
            stats = torch.stack(
                (
                    (torch.log1p(lengths.to(weights.dtype)) / 8.0).clamp(max=1.0),
                    (torch.log1p(mass) / 8.0).clamp(max=1.0),
                ),
                dim=1,
            )
            stats = stats.masked_fill(missing.unsqueeze(1), 0.0)
            value = torch.cat((value, stats), dim=1)
        if self.use_presence_features:
            value = torch.cat(
                (value, (~missing).to(value.dtype).unsqueeze(1)), dim=1
            )
        return value

    def forward(
        self,
        batch: dict,
        forced_missing: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EmbeddingBag input -> (click logits, direct conversion logits), each [B]."""
        forced_missing = forced_missing or {}
        unknown_forced_fields = set(forced_missing) - set(self.field_ids)
        if unknown_forced_fields:
            raise ValueError(
                f"forced_missing names unknown fields: {sorted(unknown_forced_fields)}"
            )
        pooled = []
        for field in self.field_ids:
            pooled.append(
                self._pool_field(field, batch[field], forced_missing.get(field))
            )
        shared = torch.cat(pooled, dim=1)
        click = self.click_tower(
            self.convolutions(shared.unsqueeze(1)).flatten(start_dim=1)
        )
        conversion = self.conversion_tower(shared)
        transferred = torch.stack((self.click_info(click), conversion), dim=1)
        if self.normalize_task_representations:
            transferred = F.layer_norm(transferred, (transferred.shape[-1],))
        attended, weights = self.attention(transferred)
        self.last_attention_weights = weights.detach()
        if self.conversion_residual:
            attended = attended + transferred[:, 1]
        return (
            self.click_head(click).squeeze(-1),
            self.conversion_head(attended).squeeze(-1),
        )


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
    penalty = F.relu(
        torch.sigmoid(conversion_logits) - torch.sigmoid(click_logits)
    ).mean()
    return click_loss + conversion_loss + alpha * penalty
