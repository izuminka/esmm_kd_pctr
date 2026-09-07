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


def _make_tower() -> nn.Sequential:
    # nn.Linear's own default init (kaiming_uniform_ with a=sqrt(5), plus a
    # matching bias bound) already produces exactly U(-1/sqrt(d_in),
    # 1/sqrt(d_in)) on both weight and bias -- no extra init call needed to
    # match the paper's rule.
    layers = []
    for d_in, d_out in zip(TOWER_DIMS, TOWER_DIMS[1:]):
        layers.append(nn.Linear(d_in, d_out))
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
            # kaiming_uniform_ with a=sqrt(5) and mode="fan_out" reads the
            # weight's dim-0 size (num_embeddings) as its fan, giving exactly
            # U(-1/sqrt(num_embeddings), 1/sqrt(num_embeddings)) -- the paper's
            # per-field embedding init rule, with no custom formula needed.
            nn.init.kaiming_uniform_(bag.weight, a=math.sqrt(5), mode="fan_out")
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
