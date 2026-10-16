"""ESMM conversion tower trained through a frozen serving pCTR product.

This is the direct interface in Equation 7 of the paper:

    L_Direct = mean BCE(z, stopgrad(p) * v)

The architecture reuses ESMM's embeddings and pCVR tower, but it has no
auxiliary pCTR tower. The external pCTR ``p`` is a loss factor only and is
never a model input.
"""

import torch
import torch.nn.functional as F

from models.esmm import EPS, ESMM, _PAPER_TOWER_DIMS


def esmm_direct_product_loss(
    z: torch.Tensor,
    p: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Mean BCE for Equation 7, with the external pCTR detached."""
    product = (p.detach() * v).clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(product, z)


class ESMMDirectProduct(ESMM):
    """ESMM embeddings and pCVR tower without an auxiliary pCTR tower."""

    def __init__(
        self,
        field_vocab_sizes: dict[str, int],
        emb_dim: int = 20,
        tower_dims: tuple = _PAPER_TOWER_DIMS,
    ):
        # Build in canonical ESMM order before dropping the CTR tower. Under a
        # shared seed, embeddings and the pCVR tower then start identically to
        # the ESMM and KD arms used for matched comparisons.
        super().__init__(field_vocab_sizes, emb_dim, tower_dims)
        self.ctr_tower = None

    def forward(
        self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[None, torch.Tensor]:
        """Return no auxiliary pCTR and the learned pCVR score ``v``."""
        v = torch.softmax(self.cvr_tower(self._pool(batch)), dim=1)[:, 1]
        return None, v

    def loss(self, z: torch.Tensor, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return esmm_direct_product_loss(z, p, v)
