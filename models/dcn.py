"""DCN-V2 pCTR: a separate click-only model (not ESMM's auxiliary CTR tower).

Configurable architecture used by the Ali-CCP experiments:
  - one EmbeddingBag per categorical field, width emb_dim, with configurable
    weighted-sum, weighted-mean, or sample-L2-normalized pooling
  - optional bounded log bag-count and log weight-mass features
  - num_cross_layers rank-cross_rank low-rank cross layers over the pooled input
  - a deep branch: cross_dim -> deep_hidden_dims..., ReLU + Dropout(dropout) per hidden layer
  - parallel (default): concatenate cross/deep outputs before the logit head
  - stacked: cross output -> deep network -> logit head (DCN-V2 section 3.4)
  - embeddings ~ U(-1/sqrt(K_f), 1/sqrt(K_f)); configurable He/Xavier cross
    and head initialization; ReLU deep layers use He-normal; biases are zero

Loss: single-task BCE-with-logits on the click label; no CTCVR term.

Architecture hyperparameters are passed at construction; see configs/dcn.yaml for defaults.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _normalize_weights(weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Rescale each bag's weights to sum to 1 (0 if the bag's total weight is 0).

    `offsets` gives each bag's start index into the flat `weights` tensor, as
    produced by `data.vocab.encode_batch`.
    """
    num_bags = offsets.numel()
    total_len = weights.numel()
    ends = torch.cat([offsets[1:], torch.tensor([total_len], device=offsets.device)])
    lengths = ends - offsets
    bag_ids = torch.repeat_interleave(torch.arange(num_bags, device=offsets.device), lengths)

    bag_sums = torch.zeros(num_bags, dtype=weights.dtype, device=weights.device).scatter_add_(
        0, bag_ids, weights
    )
    safe_sums = bag_sums.clone()
    safe_sums[safe_sums == 0] = 1.0  # avoid 0/0; those bags have all-zero weights anyway
    return weights / safe_sums[bag_ids]


def _normalized_weights_and_bounded_stats(
    weights: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-pooling weights and two stable bag-level numeric features.

    Raw weighted sums can be thousands of times larger for history fields than
    for singleton fields, which is unsafe inside repeated multiplicative cross
    layers.  The two bounded features retain that discarded information without
    exposing the cross network to the raw scale: log token count and log total
    non-negative weight, each mapped approximately into [0, 1].
    """
    num_bags = offsets.numel()
    total_len = weights.numel()
    ends = torch.cat([offsets[1:], offsets.new_tensor([total_len])])
    lengths = ends - offsets
    bag_ids = torch.repeat_interleave(
        torch.arange(num_bags, device=offsets.device), lengths
    )
    bag_sums = torch.zeros(
        num_bags, dtype=weights.dtype, device=weights.device
    ).scatter_add_(0, bag_ids, weights)
    safe_sums = bag_sums.masked_fill(bag_sums == 0, 1.0)
    normalized = weights / safe_sums[bag_ids]
    stats = torch.stack(
        (
            (torch.log1p(lengths.to(weights.dtype)) / 8.0).clamp(max=1.0),
            (torch.log1p(bag_sums.clamp_min(0)) / 8.0).clamp(max=1.0),
        ),
        dim=1,
    )
    return normalized, stats


def _bag_ids(weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Map each flattened per-sample weight to its bag (row) index."""
    ends = torch.cat([offsets[1:], offsets.new_tensor([weights.numel()])])
    lengths = ends - offsets
    return torch.repeat_interleave(
        torch.arange(offsets.numel(), device=offsets.device), lengths
    )


class _CrossLayer(nn.Module):
    """One low-rank DCN-V2 cross layer.

    `u0` is the model's original pooled input -- the same tensor is passed to
    every layer, unchanged. `u_l` is the running state that gets transformed
    layer by layer, starting out equal to `u0` (see `DCNv2.forward`).
    """

    def __init__(self, dim: int, rank: int, init: str = "he"):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)
        if init == "he":
            nn.init.kaiming_normal_(self.v.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(self.u.weight, nonlinearity="relu")
        elif init == "xavier":
            nn.init.xavier_normal_(self.v.weight)
            nn.init.xavier_normal_(self.u.weight)
        else:
            raise ValueError(f"cross_init must be 'he' or 'xavier', got {init!r}")
        nn.init.zeros_(self.u.bias)

    def forward(self, u0: torch.Tensor, u_l: torch.Tensor) -> torch.Tensor:
        interaction = self.u(self.v(u_l))
        return u0 * interaction + u_l


def _make_deep_branch(cross_dim: int, hidden_dims: tuple, dropout: float) -> nn.Sequential:
    """MLP that sees the same pooled input as the cross layers, side by side."""
    dims = (cross_dim,) + tuple(hidden_dims)
    layers = []
    for d_in, d_out in zip(dims, dims[1:]):
        linear = nn.Linear(d_in, d_out)
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class DCNv2(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: dict,
        emb_dim: int = 64,
        cross_rank: int = 256,
        num_cross_layers: int = 4,
        deep_hidden_dims: tuple = (2048, 1024, 512),
        dropout: float = 0.1,
        pooling: str = "mean",
        bag_stats: bool = False,
        cross_init: str = "he",
        head_init: str = "he",
        token_dropout: float = 0.0,
        structure: str = "parallel",
    ):
        super().__init__()
        if structure not in ("parallel", "stacked"):
            raise ValueError(f"structure must be 'parallel' or 'stacked', got {structure!r}")
        self.structure = structure
        if pooling not in ("mean", "sum", "l2"):
            raise ValueError(f"pooling must be 'mean', 'sum', or 'l2', got {pooling!r}")
        self.pooling = pooling
        if bag_stats and pooling != "mean":
            raise ValueError("bag_stats requires pooling='mean'")
        self.bag_stats = bag_stats
        if not 0.0 <= token_dropout < 1.0:
            raise ValueError(f"token_dropout must be in [0, 1), got {token_dropout}")
        self.token_dropout = token_dropout
        self.field_ids = sorted(field_vocab_sizes)
        cross_dim = (emb_dim + (2 if bag_stats else 0)) * len(self.field_ids)
        deep_dims = (cross_dim,) + tuple(deep_hidden_dims)

        self.embeddings = nn.ModuleDict()
        for field_id in self.field_ids:
            num_embeddings = field_vocab_sizes[field_id] + 1  # +1 for unseen-token ID 0
            bag = nn.EmbeddingBag(num_embeddings, emb_dim, mode="sum")
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
            self.embeddings[field_id] = bag

        self.cross_layers = nn.ModuleList(
            _CrossLayer(cross_dim, cross_rank, cross_init) for _ in range(num_cross_layers)
        )
        self.deep = _make_deep_branch(cross_dim, deep_hidden_dims, dropout)
        head_dim = deep_dims[-1] + (cross_dim if structure == "parallel" else 0)
        self.head = nn.Linear(head_dim, 1)
        if head_init == "he":
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        elif head_init == "xavier":
            nn.init.xavier_normal_(self.head.weight)
        else:
            raise ValueError(f"head_init must be 'he' or 'xavier', got {head_init!r}")
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        """`batch`: field_id -> (ids, weights, offsets), as from `data.vocab.encode_batch`.

        Returns the raw logit (for `dcn_loss` / `BCEWithLogitsLoss`), not the score.
        """
        pooled = []
        fields = []
        for field_id, (ids, weights, offsets) in sorted(batch.items()):
            if self.training and self.token_dropout:
                # Do not invert-rescale: this intentionally simulates the
                # zero-weight OOV entries produced by the temporal test split.
                keep = torch.rand_like(weights) >= self.token_dropout
                weights = weights * keep
            fields.append((field_id, (ids, weights, offsets)))
        l2_denominator = None
        l2_bag_ids = {}
        if self.pooling == "l2":
            num_bags = fields[0][1][2].numel()
            squared_norm = fields[0][1][1].new_zeros(num_bags)
            for field_id, (_, weights, offsets) in fields:
                bag_ids = _bag_ids(weights, offsets)
                l2_bag_ids[field_id] = bag_ids
                squared_norm.scatter_add_(0, bag_ids, weights.square())
            l2_denominator = squared_norm.sqrt().masked_fill_(squared_norm == 0, 1.0)

        for field_id, (ids, weights, offsets) in fields:
            if self.bag_stats:
                pooled_weights, stats = _normalized_weights_and_bounded_stats(weights, offsets)
            elif self.pooling == "l2":
                pooled_weights = weights / l2_denominator[l2_bag_ids[field_id]]
            else:
                pooled_weights = (_normalize_weights(weights, offsets)
                                  if self.pooling == "mean" else weights)
            field_embedding = self.embeddings[field_id](
                ids, offsets, per_sample_weights=pooled_weights)
            pooled.append(torch.cat((field_embedding, stats), dim=1)
                          if self.bag_stats else field_embedding)
        u0 = torch.cat(pooled, dim=1)

        u_l = u0
        for cross_layer in self.cross_layers:
            u_l = cross_layer(u0, u_l)

        # DCN-V2 section 3.4: either concatenate parallel branch outputs,
        # or let the deep network learn from the explicit crosses directly.
        deep_out = self.deep(u0 if self.structure == "parallel" else u_l)
        head_input = (torch.cat([u_l, deep_out], dim=1)
                      if self.structure == "parallel" else deep_out)
        logit = self.head(head_input).squeeze(-1)
        return logit


def dcn_loss(
    y: torch.Tensor,      # click label
    logit: torch.Tensor,  # the model's raw CTR logit, before the sigmoid
) -> torch.Tensor:
    """L_DCN = mean[ BCE-with-logits(y, logit) ], click-only, no class weighting."""
    return F.binary_cross_entropy_with_logits(logit, y)
