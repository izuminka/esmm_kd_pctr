import math

import numpy as np
import pytest
import torch

from models.esmm import esmm_loss
from models.esmm_logged_pctr import LoggedPctrESMM


def bce(a, b):
    return -(a * math.log(b) + (1 - a) * math.log(1 - b))


# Interior values, so the hand-computed BCE needs no clipping.
Y = torch.tensor([1.0, 0.0, 1.0, 0.0])
Z = torch.tensor([1.0, 0.0, 0.0, 0.0])
Q = torch.tensor([0.8, 0.3, 0.6, 0.1])
V = torch.tensor([0.7, 0.4, 0.2, 0.5])
P = torch.tensor([0.6, 0.05, 0.4, 0.2])

# A one-field model small enough to build per test; the loss does not depend on its size.
TINY = dict(field_vocab_sizes={"f1": 5}, emb_dim=4, tower_dims=(4, 3, 2))


def tiny_batch():
    ids = torch.tensor([1, 2, 3, 4, 5])
    weights = torch.ones(5)
    offsets = torch.tensor([0, 2, 3])  # bag starts, one per row
    return {"f1": (ids, weights, offsets)}


# --- constructor ---


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(objective="unknown"),
        dict(objective="kd"),
        dict(objective="kd", alpha=0.5),
        dict(objective="kd", aux_weight=1.0),
        dict(objective="kd", alpha=1.5, aux_weight=1.0),
        dict(objective="kd", alpha=0.5, aux_weight=-1.0),
        dict(objective="product", alpha=0.5),
        dict(objective="product", aux_weight=1.0),
    ],
)
def test_constructor_rejects_inconsistent_objective_params(kwargs):
    with pytest.raises(ValueError):
        LoggedPctrESMM(**TINY, **kwargs)


def test_product_builds_no_ctr_tower():
    model = LoggedPctrESMM(**TINY, objective="product")

    assert model.ctr_tower is None
    assert not any(name.startswith("ctr_tower") for name, _ in model.named_parameters())
    q, v = model(tiny_batch())
    assert q is None
    assert v.shape == (3,)


def test_product_init_matches_kd_init_under_same_seed():
    torch.manual_seed(0)
    kd = LoggedPctrESMM(**TINY, objective="kd", alpha=0.5, aux_weight=1.0)
    torch.manual_seed(0)
    product = LoggedPctrESMM(**TINY, objective="product")

    kd_state = kd.state_dict()
    for name, tensor in product.state_dict().items():
        assert torch.equal(tensor, kd_state[name]), name


# --- loss ---


def test_kd_with_alpha_one_is_the_esmm_loss():
    model = LoggedPctrESMM(**TINY, objective="kd", alpha=1.0, aux_weight=1.0)

    assert model.loss(Q, V, Y, Z, P).item() == pytest.approx(esmm_loss(Y, Z, Q, V).item(), rel=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize("aux_weight", [1.0, 2.0])
def test_kd_loss_matches_hand_computed_two_term_form(alpha, aux_weight):
    model = LoggedPctrESMM(**TINY, objective="kd", alpha=alpha, aux_weight=aux_weight)

    expected = np.mean(
        [
            aux_weight * (alpha * bce(Y[i].item(), Q[i].item()) + (1 - alpha) * bce(P[i].item(), Q[i].item()))
            + bce((Y[i] * Z[i]).item(), (Q[i] * V[i]).item())
            for i in range(len(Y))
        ]
    )
    assert model.loss(Q, V, Y, Z, P).item() == pytest.approx(expected, rel=1e-5)


def test_product_loss_matches_hand_computed():
    model = LoggedPctrESMM(**TINY, objective="product")

    expected = np.mean([bce((Y[i] * Z[i]).item(), (P[i] * V[i]).item()) for i in range(len(Y))])
    assert model.loss(None, V, Y, Z, P).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("objective,params", [("kd", dict(alpha=0.5, aux_weight=1.0)), ("product", {})])
def test_logged_pctr_receives_no_gradient(objective, params):
    model = LoggedPctrESMM(**TINY, objective=objective, **params)
    q = Q.clone().requires_grad_() if objective == "kd" else None
    v = V.clone().requires_grad_()
    p = P.clone().requires_grad_()

    model.loss(q, v, Y, Z, p).backward()

    assert p.grad is None
    assert v.grad is not None and torch.any(v.grad != 0)
    if q is not None:
        assert q.grad is not None and torch.any(q.grad != 0)


@pytest.mark.parametrize("objective,params", [("kd", dict(alpha=0.5, aux_weight=2.0)), ("product", {})])
def test_loss_finite_at_extreme_scores(objective, params):
    model = LoggedPctrESMM(**TINY, objective=objective, **params)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    z = torch.tensor([1.0, 0.0, 0.0, 0.0])
    extremes = torch.tensor([0.0, 1.0, 1.0, 0.0])  # exact 0/1 would blow up log() without eps-clamping

    loss = model.loss(extremes if objective == "kd" else None, extremes, y, z, extremes)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


# --- gradient routing on real Ali-CCP data ---
# (real_batch fixture lives in conftest.py; the logged score is a fixed random
# draw because only its routing, not its values, is under test here)


def _logged_pctr(n: int) -> torch.Tensor:
    return torch.rand(n, generator=torch.Generator().manual_seed(0))


def _is_grad_nonzero(module) -> bool:
    return all(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())


def test_kd_gradients_reach_embeddings_and_both_towers(real_batch):
    encoded, y, z, vocab_sizes = real_batch
    model = LoggedPctrESMM(vocab_sizes, objective="kd", alpha=0.5, aux_weight=1.0)

    q, v = model(encoded)
    model.loss(q, v, y, z, _logged_pctr(len(y))).backward()

    for field_id in vocab_sizes:
        grad = model.embeddings[field_id].weight.grad
        assert grad is not None and torch.any(grad != 0)
    assert _is_grad_nonzero(model.ctr_tower)
    assert _is_grad_nonzero(model.cvr_tower)


def test_product_gradients_reach_embeddings_and_cvr_tower(real_batch):
    encoded, y, z, vocab_sizes = real_batch
    model = LoggedPctrESMM(vocab_sizes, objective="product")

    q, v = model(encoded)
    model.loss(q, v, y, z, _logged_pctr(len(y))).backward()

    for field_id in vocab_sizes:
        grad = model.embeddings[field_id].weight.grad
        assert grad is not None and torch.any(grad != 0)
    assert _is_grad_nonzero(model.cvr_tower)


@pytest.mark.parametrize("alpha,is_click_dependent", [(0.0, False), (0.5, True)])
def test_ctr_gradient_depends_on_clicks_only_when_alpha_positive(real_batch, alpha, is_click_dependent):
    encoded, y, z, vocab_sizes = real_batch
    torch.manual_seed(0)
    model = LoggedPctrESMM(vocab_sizes, objective="kd", alpha=alpha, aux_weight=1.0)
    p = _logged_pctr(len(y))
    # Flip clicks only where there is no conversion, so y*z -- the product
    # term's label -- is unchanged and any gradient change comes from the aux term.
    y_flipped = torch.where(z == 0, 1 - y, y)

    def ctr_tower_grads(clicks):
        model.zero_grad()
        q, v = model(encoded)
        model.loss(q, v, clicks, z, p).backward()
        return [param.grad.clone() for param in model.ctr_tower.parameters()]

    is_unchanged = all(
        torch.equal(a, b) for a, b in zip(ctr_tower_grads(y), ctr_tower_grads(y_flipped))
    )
    assert is_unchanged != is_click_dependent
