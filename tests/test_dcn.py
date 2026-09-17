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


def test_model_sum_pooling_preserves_raw_weight_magnitude():
    model = DCNv2({"f": 2}, emb_dim=1, cross_rank=1, num_cross_layers=0,
                  deep_hidden_dims=(), dropout=0, pooling="sum")
    with torch.no_grad():
        model.embeddings["f"].weight.copy_(torch.tensor([[0.0], [2.0], [3.0]]))
        model.head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.head.bias.zero_()
    batch = {"f": (torch.tensor([1, 2]), torch.tensor([0.5, 2.0]), torch.tensor([0]))}
    assert torch.allclose(model(batch), torch.tensor([7.0]))


def test_invalid_pooling_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        DCNv2({"f": 2}, pooling="max")


def test_stacked_structure_sends_crosses_through_deep_network():
    model = DCNv2({"f": 1}, emb_dim=1, cross_rank=1, num_cross_layers=1,
                  deep_hidden_dims=(1,), dropout=0, structure="stacked")
    with torch.no_grad():
        model.embeddings["f"].weight.fill_(2)
        model.cross_layers[0].v.weight.fill_(1)
        model.cross_layers[0].u.weight.fill_(3)
        model.cross_layers[0].u.bias.zero_()
        model.deep[0].weight.fill_(2)
        model.deep[0].bias.zero_()
        model.head.weight.fill_(3)
        model.head.bias.zero_()
    batch = {"f": (torch.tensor([1]), torch.ones(1), torch.tensor([0]))}
    # Input 2 -> cross 2*(3*2)+2=14 -> deep 28 -> logit 84.
    logit = model(batch)
    assert logit.item() == pytest.approx(84)
    logit.sum().backward()
    assert model.cross_layers[0].u.weight.grad.item() == pytest.approx(24)


def test_parallel_structure_remains_default_checkpoint_compatible():
    torch.manual_seed(42)
    original = DCNv2({"f": 3}, emb_dim=2, cross_rank=1, deep_hidden_dims=(3,))
    torch.manual_seed(42)
    explicit = DCNv2({"f": 3}, emb_dim=2, cross_rank=1, deep_hidden_dims=(3,),
                     structure="parallel")
    explicit.load_state_dict(original.state_dict(), strict=True)
    batch = {"f": (torch.tensor([1, 2]), torch.ones(2), torch.tensor([0, 1]))}
    original.eval()
    explicit.eval()
    assert torch.equal(original(batch), explicit(batch))


def test_invalid_structure_is_rejected():
    with pytest.raises(ValueError, match="structure"):
        DCNv2({"f": 2}, structure="unknown")


def test_bag_stats_retain_bounded_count_and_weight_magnitude():
    model = DCNv2({"f": 3}, emb_dim=1, cross_rank=1, num_cross_layers=0,
                  deep_hidden_dims=(), dropout=0, pooling="mean", bag_stats=True)
    with torch.no_grad():
        model.embeddings["f"].weight.copy_(torch.tensor([[0.0], [2.0], [4.0], [8.0]]))
        # With an empty deep branch the head receives u0 twice; read only the
        # two statistics from the cross-branch copy.
        model.head.weight.copy_(torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))
        model.head.bias.zero_()
    batch = {"f": (torch.tensor([1, 2, 3]), torch.tensor([1.0, 3.0, 4.0]),
                   torch.tensor([0, 2]))}
    expected = torch.tensor([
        (torch.log1p(torch.tensor(2.0)) + torch.log1p(torch.tensor(4.0))) / 8.0,
        (torch.log1p(torch.tensor(1.0)) + torch.log1p(torch.tensor(4.0))) / 8.0,
    ])
    assert torch.allclose(model(batch), expected)


def test_bag_stats_require_mean_pooling():
    with pytest.raises(ValueError, match="bag_stats"):
        DCNv2({"f": 2}, pooling="sum", bag_stats=True)


def test_l2_pooling_normalizes_weight_mass_across_fields():
    model = DCNv2({"a": 2, "b": 1}, emb_dim=1, cross_rank=1, num_cross_layers=0,
                  deep_hidden_dims=(), dropout=0, pooling="l2")
    with torch.no_grad():
        model.embeddings["a"].weight.fill_(1.0)
        model.embeddings["b"].weight.fill_(1.0)
        # Empty deep branch repeats u0; read only its cross-branch copy.
        model.head.weight.copy_(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
        model.head.bias.zero_()
    batch = {
        "a": (torch.tensor([1, 2]), torch.tensor([3.0, 4.0]), torch.tensor([0])),
        "b": (torch.tensor([1]), torch.tensor([12.0]), torch.tensor([0])),
    }
    assert torch.allclose(model(batch), torch.tensor([(3.0 + 4.0 + 12.0) / 13.0]))


def test_xavier_cross_initialization_matches_linear_projection_shapes():
    model = DCNv2({str(i): 9 for i in range(23)}, emb_dim=20, cross_rank=128,
                  num_cross_layers=1, deep_hidden_dims=(128,),
                  cross_init="xavier", head_init="xavier")
    layer = model.cross_layers[0]
    assert layer.v.weight.var().item() == pytest.approx(2 / (460 + 128), rel=0.1)
    assert layer.u.weight.var().item() == pytest.approx(2 / (128 + 460), rel=0.1)
    assert model.head.weight.var().item() == pytest.approx(2 / (588 + 1), rel=0.15)


def test_invalid_token_dropout_is_rejected():
    with pytest.raises(ValueError, match="token_dropout"):
        DCNv2({"f": 2}, token_dropout=1.0)


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
