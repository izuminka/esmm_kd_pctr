"""A separate embedding rate must preserve dense updates and cover every weight."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.dcn import DCNv2
from scripts.train import load_config
from training.trainer import Trainer


def config(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/dcn_matched_stats_wd3e6.yaml")
    cfg["training"].update(gpus=0, out_dir=str(tmp_path))
    return cfg


def test_embedding_rate_changes_only_embedding_updates(tmp_path):
    cfg = config(tmp_path)
    model = DCNv2({"f": 3}, emb_dim=2, cross_rank=1, deep_hidden_dims=(3,))
    baseline = deepcopy(model)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    default = Trainer(cfg, str(tmp_path))._make_optimizer(baseline)
    cfg["training"]["embedding_lr"] = cfg["training"]["lr"] / 5
    grouped = Trainer(cfg, str(tmp_path))._make_optimizer(model)
    ids = [id(p) for g in grouped.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert set(ids) == {id(p) for p in model.parameters()}
    # The same gradient and coupled weight decay isolate the learning-rate effect.
    for m, opt in ((model, grouped), (baseline, default)):
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
    for (name, p), (_, ref) in zip(model.named_parameters(), baseline.named_parameters()):
        if name.startswith("embeddings."):
            assert torch.allclose(initial[name] - p, (initial[name] - ref) / 5, atol=1e-7)
        else:
            assert torch.equal(p, ref)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_embedding_rate_rejected(tmp_path, rate):
    cfg = config(tmp_path)
    cfg["training"]["embedding_lr"] = rate
    with pytest.raises(ValueError, match="finite and positive"):
        Trainer(cfg, str(tmp_path))
