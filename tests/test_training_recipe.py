"""Base-rate output starts, the training-rate check, and stratified batches."""

import numpy as np
import pytest
import torch

from models.dcn import DCNv2
from models.esmm import ESMM
from models.esmm_dcn import ESMMDCN
from models.esmm_direct_product import ESMMDirectProduct
from models.esmm_logged_pctr import LoggedPctrESMM
from models.output_bias import log_odds, validate_output_bias_init
from scripts.predict_encoded import build_model as restore_model
from scripts.train import check_output_bias_init, training_base_rates
from training.trainer import Trainer
from test_esmm_protocol import tiny_cfg

START = {"ctr": 0.04, "cvr": 0.005}


# --- output_bias_init ---


def _zero_last_weights(tower):
    with torch.no_grad():
        tower[-1].weight.zero_()


def test_esmm_heads_start_at_configured_rates():
    model = ESMM({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2), output_bias_init=START)
    for tower in (model.ctr_tower, model.cvr_tower):
        _zero_last_weights(tower)
    batch = {"f": (torch.tensor([1, 2]), torch.ones(2), torch.tensor([0, 1]))}
    q, v = model(batch)
    assert torch.allclose(q, torch.full((2,), START["ctr"]))
    assert torch.allclose(v, torch.full((2,), START["cvr"]))


def test_start_changes_only_final_biases_and_no_random_draws():
    torch.manual_seed(7)
    plain = ESMM({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2)).state_dict()
    torch.manual_seed(7)
    started = ESMM({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2), output_bias_init=START).state_dict()
    changed = {k for k in plain if not torch.equal(plain[k], started[k])}
    assert changed == {"ctr_tower.2.bias", "cvr_tower.2.bias"}
    assert torch.equal(started["cvr_tower.2.bias"], torch.tensor([0.0, log_odds(START["cvr"])]))


def test_dcn_head_starts_at_click_rate():
    model = DCNv2({"f": 5}, emb_dim=2, cross_rank=2, num_cross_layers=1, deep_hidden_dims=(4,),
                  output_bias_init={"ctr": START["ctr"]})
    assert model.head.bias.item() == pytest.approx(log_odds(START["ctr"]))


def test_esmm_dcn_starts_dcn_head_and_cvr_tower_and_round_trips():
    dcn = dict(cross_rank=2, num_cross_layers=1, deep_hidden_dims=[4], pooling="mean", bag_stats=True)
    model = ESMMDCN({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2), dcn_params=dcn, output_bias_init=START)
    assert model.ctr_tower.head.bias.item() == pytest.approx(log_odds(START["ctr"]))
    assert model.cvr_tower[-1].bias[1].item() == pytest.approx(log_odds(START["cvr"]))
    assert model.model_params["output_bias_init"] == START
    assert "output_bias_init" not in model.model_params["dcn_params"]
    restored = restore_model("esmm_dcn", {"f": 5}, model.model_params)
    restored.load_state_dict(model.state_dict())


def test_heads_without_a_ctr_tower_reject_a_click_start():
    with pytest.raises(ValueError, match="no CTR head"):
        ESMMDirectProduct({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2), output_bias_init=START)
    with pytest.raises(ValueError, match="no CTR head"):
        LoggedPctrESMM({"f": 5}, objective="product", emb_dim=2, tower_dims=(2, 4, 2),
                       output_bias_init=START)
    direct = ESMMDirectProduct({"f": 5}, emb_dim=2, tower_dims=(2, 4, 2),
                               output_bias_init={"cvr": START["cvr"]})
    assert direct.cvr_tower[-1].bias[1].item() == pytest.approx(log_odds(START["cvr"]))


@pytest.mark.parametrize("spec", [{}, {"ctr": 0}, {"ctr": 1}, {"ctr": True}, {"cvr": "0.1"}, {"other": 0.1}])
def test_invalid_starts_rejected(spec):
    with pytest.raises(ValueError):
        validate_output_bias_init(spec, ("ctr", "cvr"))


class _Rows:
    def __init__(self, ys, zs):
        self.ys, self.zs = torch.tensor(ys, dtype=torch.int8), torch.tensor(zs, dtype=torch.int8)
        self.device = torch.device("cpu")

    def __len__(self):
        return len(self.ys)


def test_base_rates_and_start_check():
    rates = training_base_rates(_Rows([1, 1, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]))
    assert rates == {"ctr": 3 / 8, "cvr": 1 / 3}
    check_output_bias_init({"model_params": {"output_bias_init": {"ctr": 0.375, "cvr": 0.333333}}}, rates)
    check_output_bias_init({"model_params": {}}, rates)
    with pytest.raises(ValueError, match="training rows give"):
        check_output_bias_init({"model_params": {"output_bias_init": {"ctr": 0.04}}}, rates)


def test_checkpoint_params_carry_the_start(tmp_path):
    from scripts.train import build_model
    cfg = tiny_cfg(tmp_path)
    cfg["model_params"]["output_bias_init"] = START
    model = build_model(cfg, {"f": 8})
    restored = restore_model("esmm", {"f": 8}, {"emb_dim": 2, "tower_dims": [2, 4, 2],
                                                "aux_weight": 1.0, "output_bias_init": START})
    restored.load_state_dict(model.state_dict())
    assert restored.output_bias_init == START


# --- stratified batches ---


def _trainer(tmp_path, **training):
    cfg = tiny_cfg(tmp_path)
    cfg["training"].update(training)
    return Trainer(cfg, str(tmp_path))


def test_stratified_order_is_a_permutation_with_proportional_batches(tmp_path):
    rng = np.random.default_rng(0)
    n, batch = 20_000, 1_000
    ys = (rng.random(n) < 0.2).astype(np.int8)
    zs = (ys & (rng.random(n) < 0.05)).astype(np.int8)
    rows = _Rows(ys, zs)
    trainer = _trainer(tmp_path, stratify_batches=True)
    order = trainer._stratified_order(rows, 0)
    assert torch.equal(torch.sort(order).values, torch.arange(n))
    expected = zs.sum() * batch / n
    counts = [int(rows.zs[order[s:s + batch]].sum()) for s in range(0, n, batch)]
    assert max(abs(c - expected) for c in counts) <= 2
    clicks = [int(rows.ys[order[s:s + batch]].sum()) for s in range(0, n, batch)]
    assert max(abs(c - ys.sum() * batch / n) for c in clicks) <= 2
    assert torch.equal(order, trainer._stratified_order(rows, 0))
    assert not torch.equal(order, trainer._stratified_order(rows, 1))


def test_stratification_needs_row_order(tmp_path):
    with pytest.raises(ValueError, match="shuffle_block_size"):
        _trainer(tmp_path, stratify_batches=True, shuffle_block_size=8)
