"""Unit and construction tests for the Equation 7 ESMM direct product."""

import math

import numpy as np
import pytest
import torch

from models.esmm_direct_product import ESMMDirectProduct, esmm_direct_product_loss
from models.esmm_logged_pctr import LoggedPctrESMM
from scripts.predict_encoded import build_model as restore_model
from scripts.train import build_model


TINY = {
    "field_vocab_sizes": {"1": 4, "2": 3},
    "emb_dim": 2,
    "tower_dims": (4, 3, 2),
}
Z = torch.tensor([0.0, 1.0, 0.0])
P = torch.tensor([0.2, 0.7, 0.4])
V = torch.tensor([0.6, 0.8, 0.3])


def _bce(target, score):
    return -(target * math.log(score) + (1 - target) * math.log(1 - score))


def _batch():
    return {
        "1": (
            torch.tensor([1, 2, 3]),
            torch.ones(3),
            torch.tensor([0, 1, 2]),
        ),
        "2": (
            torch.tensor([2, 1, 3]),
            torch.ones(3),
            torch.tensor([0, 1, 2]),
        ),
    }


def test_model_has_only_the_esmm_embedding_and_pcvr_paths():
    model = ESMMDirectProduct(**TINY)

    q, v = model(_batch())

    assert model.ctr_tower is None
    assert not any(name.startswith("ctr_tower") for name, _ in model.named_parameters())
    assert q is None
    assert v.shape == (3,)
    assert torch.all((0 < v) & (v < 1))


def test_equation_7_loss_matches_an_independent_row_oracle():
    expected = np.mean([
        _bce(z, p * v)
        for z, p, v in zip(Z.tolist(), P.tolist(), V.tolist())
    ])

    actual = esmm_direct_product_loss(Z, P, V)

    assert actual.item() == pytest.approx(expected, rel=1e-6)


def test_logged_pctr_is_detached_but_pcvr_receives_the_equation_7_gradient():
    p = P.double().clone().requires_grad_()
    v = V.double().clone().requires_grad_()

    esmm_direct_product_loss(Z.double(), p, v).backward()

    served = P.double() * V.double()
    expected_v_grad = (
        P.double() * (served - Z.double()) / (served * (1 - served)) / len(Z)
    )
    assert p.grad is None
    assert torch.allclose(v.grad, expected_v_grad, rtol=1e-12, atol=1e-12)


def test_click_labels_are_not_an_input_to_the_direct_objective():
    model = ESMMDirectProduct(**TINY)

    assert model.loss(Z, P, V) == esmm_direct_product_loss(Z, P, V)


def test_extreme_product_scores_are_numerically_finite():
    z = torch.tensor([0.0, 1.0, 1.0, 0.0])
    p = torch.tensor([0.0, 1.0, 0.0, 1.0])
    v = torch.tensor([1.0, 1.0, 1.0, 0.0])

    loss = esmm_direct_product_loss(z, p, v)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_gradients_reach_embeddings_and_pcvr_tower():
    model = ESMMDirectProduct(**TINY)
    _, v = model(_batch())

    model.loss(Z, P, v).backward()

    for embedding in model.embeddings.values():
        assert embedding.weight.grad is not None
        assert torch.any(embedding.weight.grad != 0)
    for parameter in model.cvr_tower.parameters():
        assert parameter.grad is not None
        assert torch.any(parameter.grad != 0)


def test_initialization_matches_the_legacy_product_arm_under_the_same_seed():
    torch.manual_seed(7)
    legacy = LoggedPctrESMM(**TINY, objective="product")
    torch.manual_seed(7)
    direct = ESMMDirectProduct(**TINY)

    legacy_state = legacy.state_dict()
    assert direct.state_dict().keys() == legacy_state.keys()
    for name, value in direct.state_dict().items():
        assert torch.equal(value, legacy_state[name]), name


def test_training_and_checkpoint_factories_build_the_standalone_model():
    params = {"emb_dim": 2, "tower_dims": [4, 3, 2]}
    cfg = {
        "model": "esmm_direct_product",
        "model_params": params,
        "training": {"seed": 9},
    }

    trained = build_model(cfg, TINY["field_vocab_sizes"])
    restored = restore_model(
        "esmm_direct_product", TINY["field_vocab_sizes"], params)

    assert isinstance(trained, ESMMDirectProduct)
    assert isinstance(restored, ESMMDirectProduct)
    restored.load_state_dict(trained.state_dict())
