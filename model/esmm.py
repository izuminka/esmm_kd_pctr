"""ESMM: shared embeddings feed independent CTR and CVR towers.

Architecture and init match the paper's Ali-CCP recipe:
  - one EmbeddingBag per categorical field, width 20, raw weighted-sum pooling
  - two towers, each 460 -> 360 -> 200 -> 80 -> 2, ReLU after each hidden layer
  - q = softmax(CTR logits)[:, 1], v = softmax(CVR logits)[:, 1]
  - embeddings ~ U(-1/sqrt(K_f), 1/sqrt(K_f)); dense weight/bias ~ U(-1/sqrt(d_in), 1/sqrt(d_in))

Loss: CTR term is two-logit cross-entropy vs y; CTCVR term is BCE(y*z, q*v).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

TOWER_DIMS = (460, 360, 200, 80, 2)
EPS = 1e-7


def _init_uniform(tensor: torch.Tensor, fan_in: int) -> None:
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(tensor, -bound, bound)


def _make_tower() -> nn.Sequential:
    layers = []
    for d_in, d_out in zip(TOWER_DIMS, TOWER_DIMS[1:]):
        linear = nn.Linear(d_in, d_out)
        _init_uniform(linear.weight, d_in)
        _init_uniform(linear.bias, d_in)
        layers.append(linear)
        if d_out != TOWER_DIMS[-1]:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ESMM(nn.Module):
    def __init__(self, field_vocab_sizes: dict[str, int], emb_dim: int = 20):
        super().__init__()
        self.field_ids = sorted(field_vocab_sizes)
        self.embeddings = nn.ModuleDict()
        for field_id in self.field_ids:
            num_embeddings = field_vocab_sizes[field_id] + 1  # +1 for unseen-token ID 0
            bag = nn.EmbeddingBag(num_embeddings, emb_dim, mode="sum")
            _init_uniform(bag.weight, num_embeddings)
            self.embeddings[field_id] = bag
        self.ctr_tower = _make_tower()
        self.cvr_tower = _make_tower()

    def forward(
        self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`batch`: field_id -> (ids, per_sample_weights, offsets), as from `data.vocab.encode_batch`."""
        pooled = [
            self.embeddings[field_id](ids, offsets, per_sample_weights=weights)
            for field_id, (ids, weights, offsets) in sorted(batch.items())
        ]
        h = torch.cat(pooled, dim=1)
        q = torch.softmax(self.ctr_tower(h), dim=1)[:, 1]
        v = torch.softmax(self.cvr_tower(h), dim=1)[:, 1]
        return q, v


def esmm_loss(y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """L_ESMM = mean[ BCE(y, q) + BCE(y*z, q*v) ], eps-clipped for stability."""
    q = q.clamp(EPS, 1 - EPS)
    qv = (q * v).clamp(EPS, 1 - EPS)
    ctr_term = F.binary_cross_entropy(q, y, reduction="none")
    ctcvr_term = F.binary_cross_entropy(qv, y * z, reduction="none")
    return (ctr_term + ctcvr_term).mean()
