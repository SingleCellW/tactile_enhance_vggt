"""Gradient and ablation checks for cached-prefix fusion into a frozen block."""

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from tactile_vggt.token_fusion import TactileTokenAdapter, TokenFusionProbe


class ToyGlobalBlock(nn.Module):
    """Differentiable token mixing with the real VGGT block's call signature."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.8)
        self.seen_pos = None
        self.input_had_grad = False

    def forward(self, tokens, pos=None):
        self.seen_pos = pos
        self.input_had_grad = tokens.requires_grad
        mixed = self.linear(tokens.mean(dim=1, keepdim=True))
        return tokens + self.dropout(torch.tanh(self.linear(tokens) + mixed))


def inputs(batch_size=3):
    torch.manual_seed(41)
    return {
        "prefix_tokens": torch.randn(batch_size, 2, 9, 16),
        "pos": torch.randint(0, 5, (batch_size, 2, 9, 2)),
        "pressure": torch.rand(batch_size, 4) + 0.1,
        "anchors": torch.randn(batch_size, 4, 3),
        "pressure_flags": torch.ones(batch_size, 4),
        "state": torch.randn(batch_size, 22),
    }


def probe(mode="internal", **kwargs):
    torch.manual_seed(17)
    return TokenFusionProbe(
        ToyGlobalBlock(16), dim=16, hidden_dim=8, feature_dim=12,
        num_sensors=4, num_heads=2, mode=mode, **kwargs,
    )


@pytest.mark.parametrize("mode", ["internal", "late", "visual", "no_xyz"])
def test_zero_gate_replays_original_frozen_block_exactly(mode):
    model = probe(mode)
    model.train()
    batch = inputs()
    original = {name: value.clone() for name, value in batch.items()}
    expected = model.frozen_global_block(
        batch["prefix_tokens"].reshape(3, 18, 16), pos=batch["pos"].reshape(3, 18, 2)
    ).reshape(3, 2, 9, 16)
    actual = model(**batch)
    torch.testing.assert_close(actual["tokens"], expected, rtol=0, atol=0)
    assert actual["prediction"].shape == (3, 3)
    assert actual["features"].shape == (3, 12)
    assert not model.frozen_global_block.training
    assert all(not parameter.requires_grad for parameter in model.frozen_global_block.parameters())
    torch.testing.assert_close(model.frozen_global_block.seen_pos, batch["pos"].reshape(3, 18, 2))
    for name, value in batch.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_touch_gradients_cross_the_actual_frozen_block():
    model = probe(initial_gate=0.2).train()
    batch = inputs()
    batch["pressure"].requires_grad_()
    batch["anchors"].requires_grad_()
    output = model(**batch)
    output["prediction"].square().sum().backward()
    assert model.frozen_global_block.input_had_grad
    for name in ("pressure", "anchors"):
        gradient = batch[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.frozen_global_block.parameters())
    for module in (model.adapter.pressure_encoder, model.adapter.xyz_encoder, model.readout, model.prediction_head):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and sum(gradient.abs().sum() for gradient in gradients) > 0


def test_adapter_preserves_special_slots_and_masks_absent_contacts():
    model = probe(initial_gate=0.3)
    batch = inputs()
    batch["pressure_flags"][1] = 0
    fused = model.adapter(
        batch["prefix_tokens"], batch["pressure"], batch["anchors"],
        pressure_flags=batch["pressure_flags"],
    )
    torch.testing.assert_close(fused[:, :, :5], batch["prefix_tokens"][:, :, :5], rtol=0, atol=0)
    torch.testing.assert_close(fused[1], batch["prefix_tokens"][1], rtol=0, atol=0)
    assert torch.isfinite(fused).all()
    assert not torch.allclose(fused[0, :, 5:], batch["prefix_tokens"][0, :, 5:])
    # Masked tactile values must not influence any output.
    changed = copy.deepcopy(batch)
    changed["pressure"][1] += 100
    changed["anchors"][1] += 100
    actual = model(**changed)
    expected = model(**batch)
    torch.testing.assert_close(actual["prediction"], expected["prediction"], rtol=0, atol=0)


def test_xyz_sensor_id_and_visual_ablations_are_isolated():
    batch = inputs()
    changed = copy.deepcopy(batch)
    changed["anchors"] = 100 * torch.randn_like(batch["anchors"])
    no_xyz = probe("no_xyz", initial_gate=0.3)
    torch.testing.assert_close(no_xyz(**batch)["tokens"], no_xyz(**changed)["tokens"], rtol=0, atol=0)
    anchored = probe(initial_gate=0.3)
    assert not torch.allclose(anchored(**batch)["tokens"], anchored(**changed)["tokens"])

    ids = torch.arange(4)
    changed_ids = ids.flip(0)
    no_ids = probe(initial_gate=0.3, use_sensor_id=False)
    torch.testing.assert_close(
        no_ids(**batch, sensor_ids=ids)["tokens"], no_ids(**batch, sensor_ids=changed_ids)["tokens"],
        rtol=0, atol=0,
    )
    assert not torch.allclose(
        anchored(**batch, sensor_ids=ids)["tokens"], anchored(**batch, sensor_ids=changed_ids)["tokens"]
    )
    visual = probe("visual", initial_gate=0.3)
    expected = visual(**batch)
    actual = visual(batch["prefix_tokens"], batch["pos"], state=batch["state"])
    torch.testing.assert_close(actual["features"], expected["features"], rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in no_xyz.parameters()) == sum(
        parameter.numel() for parameter in visual.parameters()
    )


@pytest.mark.parametrize("mode", ["internal", "late", "visual", "no_xyz", "state", "tactile"])
def test_batch_permutation_equivariance(mode):
    model = probe(mode, initial_gate=0.3).eval()
    batch = inputs()
    permutation = torch.tensor([2, 0, 1])
    expected = model(**batch)
    actual = model(**{name: value[permutation] for name, value in batch.items()})
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value[permutation])


def test_sensor_token_order_does_not_matter_when_ids_and_values_move_together():
    model = probe(initial_gate=0.3).eval()
    batch = inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    changed = {
        **batch,
        **{name: batch[name][:, permutation] for name in ("pressure", "anchors", "pressure_flags")},
    }
    expected = model(**batch, sensor_ids=torch.arange(4))
    actual = model(**changed, sensor_ids=torch.arange(4)[permutation])
    torch.testing.assert_close(actual["features"], expected["features"])


def test_rejects_invalid_cache_and_sensor_shapes():
    model = probe()
    batch = inputs()
    with pytest.raises(ValueError, match="prefix_tokens"):
        model(batch["prefix_tokens"][0], batch["pos"])
    with pytest.raises(ValueError, match="pos"):
        model(**{**batch, "pos": batch["pos"][:, :, :-1]})
    with pytest.raises(ValueError, match="pressure"):
        model(**{**batch, "pressure": batch["pressure"][:, :-1]})
    with pytest.raises(ValueError, match="anchors"):
        model(**{**batch, "anchors": None})
    with pytest.raises(ValueError, match="patch_start"):
        model(**batch, patch_start=9)
    with pytest.raises(ValueError, match="mode"):
        probe("unknown")


def test_state_baseline_skips_visual_compute_and_every_mode_observes_state():
    batch = inputs()
    changed = copy.deepcopy(batch)
    changed["prefix_tokens"] *= 100
    changed["pressure"] *= 100
    changed["anchors"] *= 100
    model = probe("state", initial_gate=0.3)
    expected = model(**batch)
    actual = model(**changed)
    assert model.frozen_global_block.seen_pos is None
    torch.testing.assert_close(actual["prediction"], expected["prediction"], rtol=0, atol=0)
    counts = set()
    for mode in ("internal", "late", "visual", "no_xyz", "state", "tactile"):
        model = probe(mode, initial_gate=0.3)
        counts.add(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        before = model(**batch)["prediction"]
        after = model(**{**batch, "state": batch["state"] + 10})["prediction"]
        assert not torch.allclose(before, after)
    assert len(counts) == 1


def test_tactile_control_ignores_rgb_values_and_skips_frozen_block():
    model = probe("tactile", initial_gate=0.3).eval()
    batch = inputs()
    changed = copy.deepcopy(batch)
    changed["prefix_tokens"] = 100 * torch.randn_like(batch["prefix_tokens"])
    changed["pos"] = torch.randint(0, 500, batch["pos"].shape)
    expected = model(**batch)
    actual = model(**changed)
    assert model.frozen_global_block.seen_pos is None
    for name in ("prediction", "features", "tokens"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(expected["tokens"][:, :, :5], torch.zeros_like(batch["prefix_tokens"][:, :, :5]))
    pressure_changed = model(**{**batch, "pressure": batch["pressure"] + 5})
    assert not torch.allclose(pressure_changed["features"], expected["features"])


def test_tactile_control_has_touch_gradients_and_no_rgb_gradient():
    model = probe("tactile", initial_gate=0.3).train()
    batch = inputs()
    for name in ("prefix_tokens", "pressure", "anchors"):
        batch[name].requires_grad_()
    model(**batch)["prediction"].square().sum().backward()
    assert batch["prefix_tokens"].grad is None
    for name in ("pressure", "anchors"):
        assert batch[name].grad is not None and batch[name].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.frozen_global_block.parameters())
