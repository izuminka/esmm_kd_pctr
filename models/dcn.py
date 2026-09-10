"""DCN-V2 pCTR: a separate click-only model (not ESMM's auxiliary CTR tower).

Architecture and init match the paper's Ali-CCP recipe:
  - one EmbeddingBag per categorical field, width emb_dim, weighted-MEAN pooling
    (weights normalized to sum to 1 per bag, 0 if the bag's total weight is 0)
  - num_cross_layers rank-cross_rank low-rank cross layers over the pooled input
  - a deep branch: cross_dim -> deep_hidden_dims..., ReLU + Dropout(dropout) per hidden layer
  - concat(cross output, deep output) -> one logit -> raw sigmoid, no calibration
  - embeddings ~ U(-1/sqrt(K_f), 1/sqrt(K_f)); every dense kernel He-normal
    (zero-mean, variance 2/fan_in), every dense bias zero

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


class _CrossLayer(nn.Module):
    """One low-rank DCN-V2 cross layer.

    `u0` is the model's original pooled input -- the same tensor is passed to
    every layer, unchanged. `u_l` is the running state that gets transformed
    layer by layer, starting out equal to `u0` (see `DCNv2.forward`).
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)
        nn.init.kaiming_normal_(self.v.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.u.weight, nonlinearity="relu")
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
    ):
        super().__init__()
        self.field_ids = sorted(field_vocab_sizes)
        cross_dim = emb_dim * len(self.field_ids)
        deep_dims = (cross_dim,) + tuple(deep_hidden_dims)

        self.embeddings = nn.ModuleDict()
        for field_id in self.field_ids:
            num_embeddings = field_vocab_sizes[field_id] + 1  # +1 for unseen-token ID 0
            bag = nn.EmbeddingBag(num_embeddings, emb_dim, mode="sum")
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
            self.embeddings[field_id] = bag

        self.cross_layers = nn.ModuleList(
            _CrossLayer(cross_dim, cross_rank) for _ in range(num_cross_layers)
        )
        self.deep = _make_deep_branch(cross_dim, deep_hidden_dims, dropout)
        self.head = nn.Linear(cross_dim + deep_dims[-1], 1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        """`batch`: field_id -> (ids, weights, offsets), as from `data.vocab.encode_batch`.

        Returns the raw logit (for `dcn_loss` / `BCEWithLogitsLoss`), not the score.
        """
        pooled = [
            self.embeddings[field_id](ids, offsets, per_sample_weights=_normalize_weights(weights, offsets))
            for field_id, (ids, weights, offsets) in sorted(batch.items())
        ]
        u0 = torch.cat(pooled, dim=1)

        u_l = u0
        for cross_layer in self.cross_layers:
            u_l = cross_layer(u0, u_l)

        deep_out = self.deep(u0)
        logit = self.head(torch.cat([u_l, deep_out], dim=1)).squeeze(-1)
        return logit


def dcn_loss(y: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """L_DCN = mean[ BCE-with-logits(y, logit) ], click-only, no class weighting."""
    return F.binary_cross_entropy_with_logits(logit, y)
