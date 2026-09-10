"""DCN-V2 pCTR: a separate click-only model (not ESMM's auxiliary CTR tower).

Architecture and init match the paper's Ali-CCP recipe:
  - one EmbeddingBag per categorical field, width 64, weighted-MEAN pooling
    (weights normalized to sum to 1 per bag, 0 if the bag's total weight is 0)
  - four rank-256 low-rank cross layers over the pooled 1472-dim input
  - a deep branch 1472 -> 2048 -> 1024 -> 512, ReLU + Dropout(0.1) per hidden layer
  - concat(cross output, deep output) -> one logit -> raw sigmoid, no calibration
  - embeddings ~ U(-1/sqrt(K_f), 1/sqrt(K_f)); every dense kernel He-normal
    (zero-mean, variance 2/fan_in), every dense bias zero

Loss: single-task BCE-with-logits on the click label; no CTCVR term.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

EMB_DIM = 64
NUM_FIELDS = 23
CROSS_DIM = EMB_DIM * NUM_FIELDS  # 1472
CROSS_RANK = 256
NUM_CROSS_LAYERS = 4
DEEP_DIMS = (CROSS_DIM, 2048, 1024, 512)
DROPOUT_P = 0.1


def _normalize_weights(weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Rescale each bag's weights to sum to 1 (0 if the bag's total weight is 0).

    `offsets` gives each bag's start index into the flat `weights` tensor, as
    produced by `data.vocab.encode_batch`.
    """
    num_bags = offsets.numel()
    total_len = weights.numel()
    ends = torch.cat([offsets[1:], torch.tensor([total_len])])
    lengths = ends - offsets
    bag_ids = torch.repeat_interleave(torch.arange(num_bags), lengths)

    bag_sums = torch.zeros(num_bags, dtype=weights.dtype).scatter_add_(0, bag_ids, weights)
    safe_sums = bag_sums.clone()
    safe_sums[safe_sums == 0] = 1.0  # avoid 0/0; those bags have all-zero weights anyway
    return weights / safe_sums[bag_ids]


class _CrossLayer(nn.Module):
    """One low-rank DCN-V2 cross layer.

    `u0` is the model's original pooled input -- the same tensor is passed to
    every layer, unchanged. `u_l` is the running state that gets transformed
    layer by layer, starting out equal to `u0` (see `DCNv2.forward`).

    Each layer: compress `u_l` to `rank` dims and back out to `dim` dims
    (`v` then `u` below -- a cheap, low-rank stand-in for a full `dim x dim`
    weight matrix), multiply that elementwise by the original `u0` (this is
    what makes it a *feature interaction*, not just another linear layer),
    then add `u_l` back as a residual connection so the layer only has to
    learn the correction on top of what it already had.
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


def _make_deep_branch() -> nn.Sequential:
    """A plain MLP (1472->2048->1024->512) that sees the same pooled input as
    the cross layers, side by side rather than stacked after them. Its output
    is concatenated with the cross layers' output right before the final
    logit."""
    layers = []
    for d_in, d_out in zip(DEEP_DIMS, DEEP_DIMS[1:]):
        linear = nn.Linear(d_in, d_out)
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU(), nn.Dropout(DROPOUT_P)])
    return nn.Sequential(*layers)


class DCNv2(nn.Module):
    def __init__(self, field_vocab_sizes: dict[str, int]):
        super().__init__()
        self.field_ids = sorted(field_vocab_sizes)
        assert len(self.field_ids) == NUM_FIELDS
        self.embeddings = nn.ModuleDict()
        for field_id in self.field_ids:
            num_embeddings = field_vocab_sizes[field_id] + 1  # +1 for unseen-token ID 0
            bag = nn.EmbeddingBag(num_embeddings, EMB_DIM, mode="sum")
            # kaiming_uniform_ with a=sqrt(5) and mode="fan_out" reads the
            # weight's dim-0 size (num_embeddings) as its fan, giving exactly
            # U(-1/sqrt(num_embeddings), 1/sqrt(num_embeddings)) -- the paper's
            # per-field embedding init rule (same one ESMM uses).
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
            self.embeddings[field_id] = bag

        self.cross_layers = nn.ModuleList(
            _CrossLayer(CROSS_DIM, CROSS_RANK) for _ in range(NUM_CROSS_LAYERS)
        )
        self.deep = _make_deep_branch()
        self.head = nn.Linear(CROSS_DIM + DEEP_DIMS[-1], 1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """`batch`: field_id -> (ids, weights, offsets), as from `data.vocab.encode_batch`.

        Returns the raw logit (for `dcn_loss` / `BCEWithLogitsLoss`), not the
        score. The paper's reported score is `q^DCN = sigmoid(logit)`, applied
        by the caller (matches the paper's logits-based loss / raw-sigmoid-score
        distinction).
        """
        pooled = [
            self.embeddings[field_id](ids, offsets, per_sample_weights=_normalize_weights(weights, offsets))
            for field_id, (ids, weights, offsets) in sorted(batch.items())
        ]
        u0 = torch.cat(pooled, dim=1)  # the fixed pooled input, passed unchanged to every cross layer

        u_l = u0  # running state, transformed layer by layer (starts out equal to u0)
        for cross_layer in self.cross_layers:
            u_l = cross_layer(u0, u_l)

        deep_out = self.deep(u0)  # deep branch also starts from u0, independently of the cross layers
        logit = self.head(torch.cat([u_l, deep_out], dim=1)).squeeze(-1)
        return logit


def dcn_loss(y: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """L_DCN = mean[ BCE-with-logits(y, logit) ], click-only, no class weighting."""
    return F.binary_cross_entropy_with_logits(logit, y)
