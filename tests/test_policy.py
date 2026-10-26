"""Checks for ablation isolation, ACT gradients, padding, and checkpoint replay."""

import copy

import pytest

torch = pytest.importorskip("torch")

from tactile_vggt.policy import PointACTPolicy, SpatialPointEncoder


def small_config(mode="anchored"):
    return {
        "mode": mode,
        "state_dim": 22,
        "action_dim": 22,
        "tactile_dim": 4,
        "n_visual_points": 12,
        "point_hidden_dims": [16, 24],
        "tactile_hidden_dim": 16,
        "encoder_dim": 32,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "latent_dim": 8,
        "chunk_size": 3,
        "n_action_steps": 2,
        "dropout": 0.0,
    }


def sample_batch(config, batch_size=2):
    visual_count = config["n_visual_points"]
    sensor_count = config["tactile_dim"]
    points = torch.rand(batch_size, visual_count + sensor_count, 8)
    points[..., :3] = points[..., :3] * 0.04
    points[:, :visual_count, 6:] = 0
    points[:, visual_count:, 3:6] = 0
    points[:, visual_count:, 7] = 1
    return {
        "observation.points": points,
        "observation.tactile": points[:, visual_count:, 6].clone(),
        "observation.state": torch.randn(batch_size, config["state_dim"]),
        "action": torch.randn(batch_size, config["chunk_size"], config["action_dim"]),
        "action_is_pad": torch.tensor([[False, False, True]]).expand(batch_size, -1).clone(),
    }


@pytest.mark.parametrize("mode", ["unanchored", "no_tactile"])
def test_ablation_cannot_read_tactile_coordinates_or_metadata(mode):
    torch.manual_seed(12)
    config = small_config(mode)
    encoder = SpatialPointEncoder(config).eval()
    batch = sample_batch(config)
    changed = copy.deepcopy(batch)
    # Preserve pressure only: all anchor positions, RGB and modality values change.
    changed["observation.points"][:, config["n_visual_points"]:, :6] = 100 * torch.randn(2, 4, 6)
    changed["observation.points"][:, config["n_visual_points"]:, 7] = 0
    if mode == "no_tactile":
        changed["observation.tactile"] = 100 * torch.randn(2, 4)
        changed["observation.points"][:, config["n_visual_points"]:, 6] = -100
    expected = encoder(batch["observation.points"], batch["observation.tactile"])
    actual = encoder(changed["observation.points"], changed["observation.tactile"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_modes_have_same_capacity_and_unanchored_still_uses_pressure():
    torch.manual_seed(7)
    encoders = [SpatialPointEncoder(small_config(mode)) for mode in ("anchored", "unanchored", "no_tactile")]
    assert len({sum(parameter.numel() for parameter in encoder.parameters()) for encoder in encoders}) == 1
    batch = sample_batch(small_config())
    encoder = encoders[1]
    with_pressure = encoder(batch["observation.points"], batch["observation.tactile"])
    without_pressure = encoder(batch["observation.points"], torch.zeros_like(batch["observation.tactile"]))
    assert not torch.allclose(with_pressure, without_pressure)


def test_local_pooling_uses_spatial_neighbors_and_handles_empty_region():
    encoder = SpatialPointEncoder(small_config())
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    contacts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    features = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]], requires_grad=True)
    pooled = encoder._local_visual_features(xyz, contacts, features)
    torch.testing.assert_close(pooled, torch.tensor([[[2.0, 3.0], [5.0, 7.0], [0.0, 0.0]]]))
    pooled.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))


def test_act_loss_trains_point_encoder_and_all_parameters_are_optimized():
    torch.manual_seed(3)
    config = small_config()
    policy = PointACTPolicy(config).train()
    batch = sample_batch(config)
    original = {key: value.clone() for key, value in batch.items()}
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert {"l1_loss", "kld_loss"} <= metrics.keys()
    loss.backward()
    for component in (policy.point_encoder.point_mlp, policy.point_encoder.contact_fusion, policy.act.model.action_head):
        grads = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
        assert sum(gradient.abs().sum().item() for gradient in grads) > 0
    optimized = [id(parameter) for group in policy.get_optim_params() for parameter in group["params"]]
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    assert len(optimized) == len(set(optimized)) and set(optimized) == expected
    assert batch.keys() == original.keys()
    for key in batch:
        torch.testing.assert_close(batch[key], original[key])


def test_padding_excludes_action_targets_from_validation_loss():
    config = small_config()
    policy = PointACTPolicy(config).eval()
    batch = sample_batch(config)
    changed = copy.deepcopy(batch)
    changed["action"][batch["action_is_pad"]] = 1e5
    with torch.no_grad():
        loss, _ = policy(batch)
        changed_loss, _ = policy(changed)
    torch.testing.assert_close(loss, changed_loss)


def test_checkpoint_and_action_queue_round_trip(tmp_path):
    config = small_config()
    policy = PointACTPolicy(config).eval()
    batch = sample_batch(config)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 3, 22)
    torch.testing.assert_close(policy.select_action(batch), chunk[:, 0])
    torch.testing.assert_close(policy.select_action(batch), chunk[:, 1])
    policy.reset()
    torch.testing.assert_close(policy.select_action(batch), chunk[:, 0])
    policy.save_pretrained(tmp_path)
    restored = PointACTPolicy.from_pretrained(tmp_path)
    torch.testing.assert_close(restored.predict_action_chunk(batch), chunk, rtol=0, atol=0)
    torch.testing.assert_close(restored.select_action(batch), chunk[:, 0], rtol=0, atol=0)


def test_rejects_wrong_point_shape_and_unknown_config():
    encoder = SpatialPointEncoder(small_config())
    with pytest.raises(ValueError, match="observation.points"):
        encoder(torch.zeros(2, 15, 8))
    with pytest.raises(ValueError, match="Unknown"):
        SpatialPointEncoder({"typo_dim": 128})
