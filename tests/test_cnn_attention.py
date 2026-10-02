"""Architecture, cold-start handling, task transfer, and loss semantics."""

import math

import pytest
import torch
import torch.nn.functional as F

from models.cnn_attention import CNNAttention, TaskAttention, cnn_attention_loss


def _batch():
    # Includes a zero-weight unknown ID and an empty bag.
    return {
        "a": (
            torch.tensor([1, 2, 0, 3]),
            torch.tensor([1.0, 2.0, 0.0, 1.0]),
            torch.tensor([0, 2, 3]),
        ),
        "b": (
            torch.tensor([2, 1]),
            torch.tensor([0.5, 1.0]),
            torch.tensor([0, 0, 1]),
        ),
    }


def test_forward_supports_ragged_fields_and_batch_size_one():
    torch.manual_seed(42)
    model = CNNAttention({"b": 2, "a": 3}).eval()
    batch = _batch()
    click, conversion = model(batch)
    assert click.shape == conversion.shape == (3,)
    assert torch.isfinite(click).all() and torch.isfinite(conversion).all()
    reordered = model(dict(reversed(list(batch.items()))))
    torch.testing.assert_close(click, reordered[0])
    torch.testing.assert_close(conversion, reordered[1])

    single = {
        field: (ids, weights, torch.tensor([0]))
        for field, (ids, weights, _) in batch.items()
    }
    assert all(logits.shape == (1,) for logits in model(single))


def _fixed_embeddings(model):
    with torch.no_grad():
        model.embeddings["a"].weight.copy_(
            torch.tensor([[99.0, 99.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        model.embeddings["b"].weight.copy_(
            torch.tensor([[99.0, 99.0], [7.0, 8.0], [9.0, 10.0]])
        )


def test_version_one_preserves_legacy_sum_pooling_and_zero_missing_bags():
    model = CNNAttention(
        {"b": 2, "a": 3}, emb_dim=2, architecture_version=1
    )
    _fixed_embeddings(model)
    inputs = []
    handle = model.conversion_tower.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0])
    )
    model(_batch())
    handle.remove()
    torch.testing.assert_close(
        inputs[0],
        torch.tensor(
            [
                [7.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 4.5, 5.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
    )


def test_version_two_mean_pools_and_exposes_trainable_missing_state_and_presence():
    model = CNNAttention({"b": 2, "a": 3}, emb_dim=2).eval()
    _fixed_embeddings(model)
    inputs = []
    handle = model.conversion_tower.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0])
    )
    model(_batch())
    handle.remove()
    torch.testing.assert_close(
        inputs[0],
        torch.tensor(
            [
                [7.0 / 3, 10.0 / 3, 1.0, 99.0, 99.0, 0.0],
                [99.0, 99.0, 0.0, 9.0, 10.0, 1.0],
                [5.0, 6.0, 1.0, 7.0, 8.0, 1.0],
            ]
        ),
    )


def test_sum_pooling_retains_bounded_bag_stats_and_presence():
    model = CNNAttention(
        {"b": 2, "a": 3},
        emb_dim=2,
        pooling="sum",
        bag_stats=True,
        use_missing_embedding=False,
        use_presence_features=True,
    ).eval()
    _fixed_embeddings(model)

    pooled_a = model._pool_field("a", _batch()["a"], None)
    log2 = math.log(2) / 8
    torch.testing.assert_close(
        pooled_a,
        torch.tensor(
            [
                [7.0, 10.0, math.log(3) / 8, math.log(4) / 8, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 6.0, log2, log2, 1.0],
            ]
        ),
    )

    forced = torch.tensor([True, False, False])
    masked_a = model._pool_field("a", _batch()["a"], forced)
    torch.testing.assert_close(masked_a[0], torch.zeros(5))
    torch.testing.assert_close(masked_a[1:], pooled_a[1:])

def test_forced_missing_uses_id_zero_and_receives_gradient():
    torch.manual_seed(42)
    model = CNNAttention(
        {"a": 3, "b": 2},
        emb_dim=2,
        conv_channels=(2,),
        tower_dims=(4, 2),
        dropout=0,
    )
    forced = {"a": torch.tensor([True, False, False])}
    _, conversion = model(_batch(), forced_missing=forced)
    F.binary_cross_entropy_with_logits(
        conversion, torch.tensor([0.0, 0.0, 1.0])
    ).backward()
    assert model.embeddings["a"].weight.grad[0].abs().sum() > 0


def test_dot_attention_preserves_legacy_equation():
    attention = TaskAttention(2, score_mode="dot")
    with torch.no_grad():
        for layer in (attention.query, attention.key, attention.value):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    pooled, weights = attention(torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]))
    first = 1 / (1 + math.exp(3 / math.sqrt(2)))
    torch.testing.assert_close(weights, torch.tensor([[first, 1 - first]]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
    torch.testing.assert_close(pooled, torch.tensor([[first, 2 * (1 - first)]]))


def test_cosine_attention_is_bounded_and_scale_invariant():
    attention = TaskAttention(2, score_mode="cosine")
    with torch.no_grad():
        attention.query.weight.copy_(torch.eye(2))
        attention.key.weight.copy_(torch.diag(torch.tensor([1.0, -1.0])))
        attention.query.bias.zero_()
        attention.key.bias.zero_()
    first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    scaled = torch.tensor([[[10000.0, 0.0], [0.0, 0.001]]])
    first_weights = attention(first)[1]
    scaled_weights = attention(scaled)[1]
    torch.testing.assert_close(first_weights, scaled_weights)
    lower = 1 / (1 + math.exp(2))
    upper = 1 / (1 + math.exp(-2))
    assert first_weights.min().item() >= lower - 1e-6
    assert first_weights.max().item() <= upper + 1e-6


def test_conversion_loss_reaches_both_towers_and_missing_embeddings():
    torch.manual_seed(42)
    model = CNNAttention({"a": 3, "b": 2}, dropout=0)
    _, conversion = model(_batch())
    F.binary_cross_entropy_with_logits(
        conversion, torch.tensor([0.0, 0.0, 1.0])
    ).backward()
    for module in (
        model.convolutions,
        model.click_tower,
        model.click_info,
        model.conversion_tower,
        model.attention,
        model.embeddings,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert model.click_head.weight.grad is None


def test_loss_is_two_entire_space_bces_plus_correctly_directed_penalty():
    y, z = torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    q, c = torch.tensor([0.8, 0.2]), torch.tensor([0.3, 0.7])
    expected_bce = -math.log(0.8) - math.log(0.3)
    loss = cnn_attention_loss(y, z, torch.logit(q), torch.logit(c), alpha=2)
    assert loss.item() == pytest.approx(expected_bce + 2 * 0.5 / 2)
    assert cnn_attention_loss(
        y, z, torch.logit(q), torch.logit(c), alpha=0
    ).item() == pytest.approx(expected_bce)


def test_calibrator_penalizes_only_conversion_above_click():
    click = torch.logit(torch.tensor([0.8, 0.2])).requires_grad_()
    conversion = torch.logit(torch.tensor([0.3, 0.7])).requires_grad_()
    y, z = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])
    penalty = cnn_attention_loss(
        y, z, click, conversion, alpha=1
    ) - cnn_attention_loss(y, z, click, conversion, alpha=0)
    penalty.backward()
    assert penalty.item() == pytest.approx(0.25)
    assert click.grad[0] == conversion.grad[0] == 0
    assert click.grad[1] < 0 and conversion.grad[1] > 0


def test_extreme_logits_have_finite_loss_and_gradients():
    click = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    conversion = torch.tensor([1000.0, -1000.0], requires_grad=True)
    loss = cnn_attention_loss(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        click,
        conversion,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(click.grad).all() and torch.isfinite(conversion.grad).all()


@pytest.mark.parametrize(
    "params",
    [
        {"conv_channels": []},
        {"tower_dims": []},
        {"kernel_size": 2},
        {"emb_dim": 0},
        {"dropout": 1},
        {"alpha": -1},
        {"alpha": float("nan")},
        {"architecture_version": 3},
        {"pooling": "max"},
        {"attention_score": "unbounded"},
        {"field_dropout": {"missing": 0.2}},
        {"field_dropout": {"a": 1.0}},
    ],
)
def test_invalid_architecture_settings_are_rejected(params):
    with pytest.raises(ValueError):
        CNNAttention({"a": 3}, **params)
