import math

import pytest
import torch
from torch import nn

from models.dcn import _CrossLayer, _normalize_weights, dcn_loss, DCNv2


# --- weight normalization (the pooling mechanic that differs from ESMM) ---


def test_normalize_weights_computes_weighted_mean_ratios():
    weights = torch.tensor([0.5, 2.0])
    offsets = torch.tensor([0])  # one bag containing both tokens

    normalized = _normalize_weights(weights, offsets)

    assert torch.allclose(normalized, torch.tensor([0.2, 0.8]))
    assert normalized.sum().item() == pytest.approx(1.0)


def test_normalize_weights_zero_total_weight_bag_is_safe():
    # both tokens in the bag are unseen (weight forced to 0 upstream by vocab.py):
    # the paper's equation defines the normalized weight as 0 in this case, not NaN.
    weights = torch.tensor([0.0, 0.0])
    offsets = torch.tensor([0])

    normalized = _normalize_weights(weights, offsets)

    assert torch.equal(normalized, torch.tensor([0.0, 0.0]))


def test_normalize_weights_handles_multiple_bags_independently():
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0])  # bag0: [1,1] bag1: [1,3]
    offsets = torch.tensor([0, 2])

    normalized = _normalize_weights(weights, offsets)

    assert torch.allclose(normalized, torch.tensor([0.5, 0.5, 0.25, 0.75]))


def test_pooling_is_weighted_mean_not_sum():
    bag = nn.EmbeddingBag(3, 2, mode="sum")
    with torch.no_grad():
        bag.weight.copy_(torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    ids = torch.tensor([1, 2])
    raw_weights = torch.tensor([0.5, 2.0])
    offsets = torch.tensor([0])

    out = bag(ids, offsets, per_sample_weights=_normalize_weights(raw_weights, offsets))

    # weighted mean: 0.2*[2,20] + 0.8*[3,30] = [2.8, 28]  (not the weighted SUM [7,70])
    assert torch.allclose(out, torch.tensor([[2.8, 28.0]]))


# --- cross layer ---


def test_cross_layer_matches_residual_formula_exactly():
    layer = _CrossLayer(dim=3, rank=2)
    with torch.no_grad():
        layer.v.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))  # V: R^{2x3}
        layer.u.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))  # U: R^{3x2}
        layer.u.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))

    u0 = torch.tensor([[1.0, 2.0, 3.0]])
    u_l = torch.tensor([[0.5, 0.5, 0.5]])

    out = layer(u0, u_l)

    v_out = u_l @ layer.v.weight.T  # V(u_l)
    u_out = v_out @ layer.u.weight.T + layer.u.bias  # U(V(u_l)) + b
    expected = u0 * u_out + u_l

    assert torch.allclose(out, expected)


# --- init ---


def test_dense_layer_variance_matches_its_own_fan_in():
    # We don't re-test kaiming_normal_ itself (that's PyTorch's job) -- this
    # checks OUR wiring calls it on the right tensor with the right implicit
    # fan_in, which would catch e.g. an accidentally swapped dimension.
    model = DCNv2({str(i): 9 for i in range(23)})

    v_weight = model.cross_layers[0].v.weight  # Linear(1472, 256) -> fan_in=1472
    assert v_weight.var().item() == pytest.approx(2 / 1472, rel=0.3)

    u_weight = model.cross_layers[0].u.weight  # Linear(256, 1472) -> fan_in=256
    assert u_weight.var().item() == pytest.approx(2 / 256, rel=0.3)

    first_deep_linear = next(m for m in model.deep if isinstance(m, nn.Linear))
    assert first_deep_linear.weight.var().item() == pytest.approx(2 / 1472, rel=0.3)


def test_embedding_init_within_paper_bounds():
    model = DCNv2({str(i): 9 for i in range(23)})

    bound = 1 / math.sqrt(10) + 1e-6  # vocab_size=9 + 1 unseen slot
    for field_id in model.field_ids:
        assert torch.all(model.embeddings[field_id].weight.abs() <= bound)


def test_dense_biases_are_zero_initialized():
    model = DCNv2({str(i): 9 for i in range(23)})

    for layer in model.cross_layers:
        assert torch.all(layer.u.bias == 0)
    for module in model.deep:
        if isinstance(module, nn.Linear):
            assert torch.all(module.bias == 0)
    assert torch.all(model.head.bias == 0)


# --- architecture ---


def test_deep_branch_architecture_matches_paper():
    model = DCNv2({str(i): 9 for i in range(23)})

    linears = [m for m in model.deep if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [
        (1472, 2048),
        (2048, 1024),
        (1024, 512),
    ]
    assert sum(isinstance(m, nn.ReLU) for m in model.deep) == 3
    dropouts = [m for m in model.deep if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 3
    assert all(d.p == pytest.approx(0.1) for d in dropouts)


def test_model_has_four_cross_layers_and_correct_head_shape():
    model = DCNv2({str(i): 9 for i in range(23)})

    assert len(model.cross_layers) == 4
    for layer in model.cross_layers:
        assert layer.v.weight.shape == (256, 1472)
        assert layer.u.weight.shape == (1472, 256)
    assert model.head.weight.shape == (1, 1984)  # 1472 (cross) + 512 (deep)


# --- loss ---


def test_dcn_loss_matches_hand_computed_bce_with_logits():
    y = torch.tensor([1.0, 0.0, 1.0])
    logit = torch.tensor([1.5, -0.5, 0.2])

    loss = dcn_loss(y, logit).item()

    def bce_with_logits(label, x):
        p = 1 / (1 + math.exp(-x))
        return -(label * math.log(p) + (1 - label) * math.log(1 - p))

    expected = sum(bce_with_logits(y[i].item(), logit[i].item()) for i in range(3)) / 3
    assert loss == pytest.approx(expected, rel=1e-5)


# --- real-data integration (reuses the shared real_batch fixture) ---


def test_forward_backward_on_real_data(real_batch):
    encoded, y, _, vocab_sizes = real_batch
    model = DCNv2(vocab_sizes)

    logit = model(encoded)
    assert logit.shape == y.shape

    loss = dcn_loss(y, logit)
    assert torch.isfinite(loss)

    loss.backward()
    for field_id in vocab_sizes:
        grad = model.embeddings[field_id].weight.grad
        assert grad is not None and torch.any(grad != 0)
    for layer in model.cross_layers:
        assert layer.v.weight.grad is not None and torch.any(layer.v.weight.grad != 0)
        assert layer.u.weight.grad is not None and torch.any(layer.u.weight.grad != 0)
    for module in model.deep:
        if hasattr(module, "weight"):
            assert module.weight.grad is not None and torch.any(module.weight.grad != 0)
    assert model.head.weight.grad is not None and torch.any(model.head.weight.grad != 0)
