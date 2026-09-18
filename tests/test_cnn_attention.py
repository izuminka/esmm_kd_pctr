"""Architecture, task transfer, and the paper's loss semantics."""

import math

import pytest
import torch
import torch.nn.functional as F

from models.cnn_attention import CNNAttention, TaskAttention, cnn_attention_loss


def _batch():
    # Includes a zero-weight unknown ID and an empty bag.
    return {
        "a": (torch.tensor([1, 2, 0, 3]), torch.tensor([1., 2., 0., 1.]), torch.tensor([0, 2, 3])),
        "b": (torch.tensor([2, 1]), torch.tensor([0.5, 1.]), torch.tensor([0, 0, 1])),
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

    single = {field: (ids, weights, torch.tensor([0]))
              for field, (ids, weights, _) in batch.items()}
    assert all(logits.shape == (1,) for logits in model(single))


def test_field_pooling_keeps_weights_and_empty_bags():
    model = CNNAttention({"b": 2, "a": 3}, emb_dim=2)
    with torch.no_grad():
        model.embeddings["a"].weight.copy_(torch.tensor([[99., 99.], [1., 2.], [3., 4.], [5., 6.]]))
        model.embeddings["b"].weight.copy_(torch.tensor([[99., 99.], [7., 8.], [9., 10.]]))
    inputs = []
    handle = model.conversion_tower.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    model(_batch())
    handle.remove()
    torch.testing.assert_close(inputs[0], torch.tensor([
        [7., 10., 0., 0.], [0., 0., 4.5, 5.], [5., 6., 7., 8.],
    ]))


def test_attention_normalizes_over_tasks_and_pools_projected_values():
    attention = TaskAttention(2)
    with torch.no_grad():
        for layer in (attention.query, attention.key, attention.value):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    pooled, weights = attention(torch.tensor([[[1., 0.], [0., 2.]]]))
    first = 1 / (1 + math.exp(3 / math.sqrt(2)))
    torch.testing.assert_close(weights, torch.tensor([[first, 1 - first]]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
    torch.testing.assert_close(pooled, torch.tensor([[first, 2 * (1 - first)]]))


def test_conversion_loss_reaches_click_cnn_without_click_loss_or_calibrator():
    torch.manual_seed(42)
    model = CNNAttention({"a": 3, "b": 2}, dropout=0)
    _, conversion = model(_batch())
    F.binary_cross_entropy_with_logits(conversion, torch.tensor([0., 0., 1.])).backward()
    for module in (model.convolutions, model.click_tower, model.click_info,
                   model.conversion_tower, model.attention, model.embeddings):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert model.click_head.weight.grad is None  # no click supervision in this check


def test_loss_is_two_entire_space_bces_plus_correctly_directed_penalty():
    y, z = torch.tensor([1., 0.]), torch.tensor([1., 0.])
    q, c = torch.tensor([0.8, 0.2]), torch.tensor([0.3, 0.7])
    expected_bce = -math.log(0.8) - math.log(0.3)
    loss = cnn_attention_loss(y, z, torch.logit(q), torch.logit(c), alpha=2)
    assert loss.item() == pytest.approx(expected_bce + 2 * 0.5 / 2)
    assert cnn_attention_loss(y, z, torch.logit(q), torch.logit(c), alpha=0).item() == pytest.approx(expected_bce)


def test_calibrator_penalizes_only_conversion_above_click():
    click = torch.logit(torch.tensor([0.8, 0.2])).requires_grad_()
    conversion = torch.logit(torch.tensor([0.3, 0.7])).requires_grad_()
    y, z = torch.tensor([1., 0.]), torch.tensor([0., 0.])
    penalty = (cnn_attention_loss(y, z, click, conversion, alpha=1)
               - cnn_attention_loss(y, z, click, conversion, alpha=0))
    penalty.backward()
    assert penalty.item() == pytest.approx(0.25)
    assert click.grad[0] == conversion.grad[0] == 0
    assert click.grad[1] < 0 and conversion.grad[1] > 0


def test_extreme_logits_have_finite_loss_and_gradients():
    click = torch.tensor([-1000., 1000.], requires_grad=True)
    conversion = torch.tensor([1000., -1000.], requires_grad=True)
    loss = cnn_attention_loss(torch.tensor([1., 0.]), torch.tensor([0., 0.]), click, conversion)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(click.grad).all() and torch.isfinite(conversion.grad).all()


@pytest.mark.parametrize("params", [
    {"conv_channels": []}, {"tower_dims": []}, {"kernel_size": 2},
    {"emb_dim": 0}, {"dropout": 1}, {"alpha": -1}, {"alpha": float("nan")},
])
def test_invalid_architecture_settings_are_rejected(params):
    with pytest.raises(ValueError):
        CNNAttention({"a": 3}, **params)
