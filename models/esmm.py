"""ESMM: shared embeddings feed independent CTR and CVR towers.

Architecture and init match the paper's Ali-CCP recipe:
  - one EmbeddingBag per categorical field, width emb_dim, raw weighted-sum pooling
  - two towers with configurable layer dims, ReLU after each hidden layer
  - q = softmax(CTR logits)[:, 1], v = softmax(CVR logits)[:, 1]
  - embeddings ~ U(-1/sqrt(K_f), 1/sqrt(K_f)); dense weight/bias ~ U(-1/sqrt(d_in), 1/sqrt(d_in))

Loss: CTR term is two-logit cross-entropy vs y; CTCVR term is BCE(y*z, q*v).
aux_weight scales the whole CTR term and leaves the CTCVR term's weight at one;
1 is the canonical ESMM loss.

Paper defaults: emb_dim=20, tower_dims=(460, 360, 200, 80, 2), aux_weight=1.
tower_dims[0] must equal num_fields * emb_dim.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-7

_PAPER_TOWER_DIMS = (460, 360, 200, 80, 2)


def _make_tower(tower_dims: tuple) -> nn.Sequential:
    # nn.Linear's own default init (kaiming_uniform_ with a=sqrt(5), plus a
    # matching bias bound) already produces exactly U(-1/sqrt(d_in),
    # 1/sqrt(d_in)) on both weight and bias -- no extra init call needed to
    # match the paper's rule.
    layers = []
    for d_in, d_out in zip(tower_dims, tower_dims[1:]):
        layers.append(nn.Linear(d_in, d_out))
        if d_out != tower_dims[-1]:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ESMM(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: dict[str, int],
        emb_dim: int = 20,
        tower_dims: tuple = _PAPER_TOWER_DIMS,
        aux_weight: float = 1.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.tower_dims = tuple(tower_dims)
        self.aux_weight = aux_weight
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
        self.ctr_tower = _make_tower(self.tower_dims)
        self.cvr_tower = _make_tower(self.tower_dims)

    def _pool(self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Per-field pooled embeddings concatenated in field order: the shared tower input."""
        pooled = [
            self.embeddings[field_id](ids, offsets, per_sample_weights=weights)
            for field_id, (ids, weights, offsets) in sorted(batch.items())
        ]
        return torch.cat(pooled, dim=1)

    def forward(
        self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`batch`: field_id -> (ids, per_sample_weights, offsets), as from `data.vocab.encode_batch`."""
        h = self._pool(batch)
        q = torch.softmax(self.ctr_tower(h), dim=1)[:, 1]
        v = torch.softmax(self.cvr_tower(h), dim=1)[:, 1]
        return q, v


def esmm_loss(
    y: torch.Tensor,          # click label
    z: torch.Tensor,          # conversion label; y*z is the CTCVR label
    q: torch.Tensor,          # auxiliary pCTR head
    v: torch.Tensor,          # pCVR head
    aux_weight: float = 1.0,  # weight on the CTR term; 1 is the canonical loss
) -> torch.Tensor:
    """L_ESMM = mean[ aux_weight * BCE(y, q) + BCE(y*z, q*v) ], eps-clipped for stability."""
    q = q.clamp(EPS, 1 - EPS)
    qv = (q * v).clamp(EPS, 1 - EPS)
    ctr_term = F.binary_cross_entropy(q, y, reduction="none")
    ctcvr_term = F.binary_cross_entropy(qv, y * z, reduction="none")
    return (aux_weight * ctr_term + ctcvr_term).mean()
