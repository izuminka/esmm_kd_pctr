import math

import numpy as np
import pytest
import torch
from torch import nn

from models.esmm import esmm_loss, ESMM


# --- model architecture and init ---


def test_tower_architecture_matches_paper():
    model = ESMM({"f1": 5}, emb_dim=20)
    for tower in (model.ctr_tower, model.cvr_tower):
        linears = [m for m in tower if isinstance(m, nn.Linear)]
        assert [(l.in_features, l.out_features) for l in linears] == [
            (460, 360),
            (360, 200),
            (200, 80),
            (80, 2),
        ]
        assert sum(isinstance(m, nn.ReLU) for m in tower) == 3  # after each hidden layer only
        assert isinstance(tower[-1], nn.Linear)  # no ReLU after the output layer


def test_init_bounds_match_paper():
    model = ESMM({"f1": 9, "f2": 4}, emb_dim=6)

    assert torch.all(model.embeddings["f1"].weight.abs() <= 1 / math.sqrt(10) + 1e-6)
    assert torch.all(model.embeddings["f2"].weight.abs() <= 1 / math.sqrt(5) + 1e-6)
    for tower in (model.ctr_tower, model.cvr_tower):
        for layer in tower:
            if isinstance(layer, nn.Linear):
                bound = 1 / math.sqrt(layer.in_features) + 1e-6
                assert torch.all(layer.weight.abs() <= bound)
                assert torch.all(layer.bias.abs() <= bound)


# --- esmm_loss ---


def test_loss_matches_hand_computed_bce():
    y = torch.tensor([1.0, 0.0, 1.0])
    z = torch.tensor([1.0, 0.0, 0.0])
    q = torch.tensor([0.8, 0.3, 0.6])
    v = torch.tensor([0.7, 0.4, 0.2])

    loss = esmm_loss(y, z, q, v).item()

    def bce(a, b):
        return -(a * math.log(b) + (1 - a) * math.log(1 - b))

    expected = np.mean(
        [
            bce(y[i].item(), q[i].item()) + bce((y[i] * z[i]).item(), (q[i] * v[i]).item())
            for i in range(3)
        ]
    )
    assert loss == pytest.approx(expected, rel=1e-5)


def test_loss_finite_at_extreme_predictions():
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    z = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = torch.tensor([1.0, 0.0, 1.0, 0.0])  # exact 0/1 would blow up log() without eps-clamping
    v = torch.tensor([1.0, 0.0, 0.0, 1.0])

    loss = esmm_loss(y, z, q, v)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


# --- end-to-end integration against real Ali-CCP data ---
# (real_batch fixture lives in conftest.py, shared with test_evaluate.py)


def test_forward_backward_on_real_data(real_batch):
    encoded, y, z, vocab_sizes = real_batch
    model = ESMM(vocab_sizes)

    q, v = model(encoded)

    assert q.shape == y.shape
    assert v.shape == y.shape
    assert torch.all((q >= 0) & (q <= 1))
    assert torch.all((v >= 0) & (v <= 1))

    loss = esmm_loss(y, z, q, v)
    assert torch.isfinite(loss)

    loss.backward()
    for field_id in vocab_sizes:
        grad = model.embeddings[field_id].weight.grad
        assert grad is not None and torch.any(grad != 0)
    for tower in (model.ctr_tower, model.cvr_tower):
        for module in tower:
            if hasattr(module, "weight"):
                assert module.weight.grad is not None and torch.any(module.weight.grad != 0)
