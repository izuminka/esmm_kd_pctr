"""Canonical shared-embedding ESMM with only its CTR tower replaced by DCN-V2."""

import inspect

import torch

from models.dcn import DCNv2
from models.esmm import ESMM, _PAPER_TOWER_DIMS


class ESMMDCN(ESMM):
    """Keep vanilla ESMM's embeddings, CVR tower, and external ESMM objective.

    DCN uses mean pooling and optional bag statistics over the same embedding
    tables that CVR pools by weighted sum. There is no teacher or frozen branch.
    """

    def __init__(
        self,
        field_vocab_sizes: dict[str, int],
        emb_dim: int = 20,
        tower_dims: tuple = _PAPER_TOWER_DIMS,
        aux_weight: float = 1.0,
        dcn_params: dict | None = None,
    ):
        params = dict(dcn_params or {})
        params.setdefault("emb_dim", emb_dim)
        if params["emb_dim"] != emb_dim:
            raise ValueError("DCN emb_dim must equal ESMM emb_dim to share embeddings")

        initial_rng = torch.get_rng_state()
        super().__init__(field_vocab_sizes, emb_dim, tower_dims, aux_weight)
        # Recreate standalone DCN initialization from this seed without changing
        # the RNG state left by canonical ESMM construction. Construction is on
        # CPU, so no CUDA RNG state needs to be read, initialized, or restored.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial_rng)
            self.ctr_tower = DCNv2(field_vocab_sizes, **params)
        self.ctr_tower.embeddings = self.embeddings

        arguments = inspect.signature(DCNv2).bind(field_vocab_sizes, **params)
        arguments.apply_defaults()
        self.model_params = {
            "emb_dim": self.emb_dim,
            "tower_dims": self.tower_dims,
            "aux_weight": self.aux_weight,
            "dcn_params": {
                name: value for name, value in arguments.arguments.items()
                if name != "field_vocab_sizes"
            },
        }

    def forward(
        self, batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.sigmoid(self.ctr_tower(batch))
        v = torch.softmax(self.cvr_tower(self._pool(batch)), dim=1)[:, 1]
        return q, v
