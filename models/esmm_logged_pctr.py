"""ESMM trained with a frozen, logged pCTR score p in its objective.

p comes from a separately trained pCTR model. It is detached and never a model
input; it enters only the loss, according to the configured objective:

  kd:      aux_weight * BCE(alpha*y + (1-alpha)*p, q) + BCE(y*z, q*v)
           Distills p into the auxiliary CTR head. For Bernoulli cross-entropy,
           alpha*BCE(y, q) + (1-alpha)*BCE(p, q) equals BCE(alpha*y + (1-alpha)*p, q)
           exactly, so the hard-click and soft-teacher terms are one soft-target
           BCE. alpha=1, aux_weight=1 recovers the ESMM loss.
  kd_tau:  Same KD loss with teacher p/(p + rho*(1-p)), on retained rows.
  product: BCE(y*z, p*v)
           Trains pCVR through the product it is served in. There is no auxiliary
           CTR head, so no CTR tower is kept and forward returns q=None.
"""

import torch
import torch.nn.functional as F

from models.esmm import _PAPER_TOWER_DIMS, EPS, ESMM
from training.downsampling import DEFAULT_NONCLICK_RATE, logged_ctr_for_space, validate_rate

OBJECTIVES = ("kd", "kd_tau", "product")


class LoggedPctrESMM(ESMM):
    def __init__(
        self,
        field_vocab_sizes: dict[str, int],
        objective: str,
        alpha: float | None = None,
        aux_weight: float | None = None,
        emb_dim: int = 20,
        tower_dims: tuple = _PAPER_TOWER_DIMS,
        nonclick_rate: float = DEFAULT_NONCLICK_RATE,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if objective in ("kd", "kd_tau"):
            if alpha is None or aux_weight is None:
                raise ValueError("KD objectives require alpha and aux_weight")
            if not 0 <= alpha <= 1:
                raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            if aux_weight < 0:
                raise ValueError(f"aux_weight must not be negative, got {aux_weight}")
        elif alpha is not None or aux_weight is not None:
            raise ValueError("objective 'product' takes no alpha or aux_weight")

        self.nonclick_rate = validate_rate(nonclick_rate)
        if objective == "kd_tau" and self.nonclick_rate == 1:
            raise ValueError("objective kd_tau requires a nonclick retention rate strictly below 1; use kd for full-rate training")
        if objective != "kd_tau" and self.nonclick_rate < 1:
            raise ValueError("nonclick_rate below one requires objective kd_tau")
        super().__init__(field_vocab_sizes, emb_dim, tower_dims)
        self.objective = objective
        self.alpha = alpha
        self.aux_weight = aux_weight
        if objective == "product":
            # Built by ESMM and only then dropped, so the embeddings and CVR tower
            # draw the same initial values as a kd model under the same seed.
            self.ctr_tower = None

    def forward(
        self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.ctr_tower is not None:
            return super().forward(batch)
        v = torch.softmax(self.cvr_tower(self._pool(batch)), dim=1)[:, 1]
        return None, v

    def loss(
        self,
        q: torch.Tensor | None,  # auxiliary pCTR head, None under the product objective
        v: torch.Tensor,         # pCVR head, the output all objectives train
        y: torch.Tensor,         # click label
        z: torch.Tensor,         # conversion label; y*z is the CTCVR label
        p: torch.Tensor,         # frozen full-population logged pCTR, detached below
        *, is_ctr_downsampled_space: bool = True,
    ) -> torch.Tensor:
        """Batch-mean loss; q is already in the requested CTR space.

        For KD-tau, align the original-space teacher to q when training in
        downsampled space. This loss never inverse-corrects the student q.
        """
        p = p.detach()
        if self.objective == "product":
            return F.binary_cross_entropy((p * v).clamp(EPS, 1 - EPS), y * z)

        if self.objective == "kd_tau":
            p = logged_ctr_for_space(
                p, self.nonclick_rate, is_ctr_downsampled_space=is_ctr_downsampled_space)
        q = q.clamp(EPS, 1 - EPS)
        qv = (q * v).clamp(EPS, 1 - EPS)
        aux = F.binary_cross_entropy(q, self.alpha * y + (1 - self.alpha) * p)
        return self.aux_weight * aux + F.binary_cross_entropy(qv, y * z)
