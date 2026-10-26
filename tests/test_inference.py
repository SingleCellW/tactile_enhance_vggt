"""Deployment normalization and geometry-refresh behavior against real ACT."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tactile_vggt.data import CachedTactileDataset
from tactile_vggt.inference import PointACTInference
from tactile_vggt.policy import PointACTPolicy


def checkpoint_and_cache(tmp_path, **overrides):
    config = {
        "state_dim": 22, "action_dim": 22, "tactile_dim": 44, "n_visual_points": 12,
        "point_hidden_dims": [16, 24], "tactile_hidden_dim": 16, "encoder_dim": 32,
        "dim_model": 32, "n_heads": 4, "dim_feedforward": 64, "n_encoder_layers": 1,
        "n_decoder_layers": 1, "n_vae_encoder_layers": 1, "latent_dim": 8,
        "chunk_size": 3, "n_action_steps": 2, "dropout": 0.0,
    }
    config.update(overrides)
    checkpoint, cache = tmp_path / "checkpoint", tmp_path / "cache"
    cache.mkdir()
    PointACTPolicy(config).save_pretrained(checkpoint)
    stats = {
        "state_mean": np.linspace(-0.5, 0.5, 22).tolist(), "state_std": np.linspace(0.5, 1.5, 22).tolist(),
        "action_mean": np.linspace(0.2, 1.2, 22).tolist(), "action_std": np.linspace(0.2, 0.8, 22).tolist(),
        "pressure_scale": 2.3, "training_frames": 1,
    }
    (checkpoint / "normalization.json").write_text(json.dumps(stats))
    rng = np.random.default_rng(5)
    points = rng.uniform(0, 0.1, (2, 56, 8)).astype(np.float32)
    points[:, :12, 6:] = 0
    points[:, 12:, 7] = 1
    tactile = rng.uniform(-0.5, 10, (2, 44)).astype(np.float32)
    points[:, 12:, 6] = np.maximum(tactile, 0)
    values = {
        "points": points, "state": rng.normal(size=(2, 22)).astype(np.float32),
        "action": rng.normal(size=(2, 22)).astype(np.float32), "tactile": tactile,
        "episode_index": np.array([0, 1]), "frame_index": np.array([0, 0]),
        "obs_frame_indices": np.array([0, 1]),
    }
    for name, value in values.items():
        np.save(cache / f"{name}.npy", value)
    (cache / "manifest.json").write_text(json.dumps({
        "complete": True, "normalization": stats, "split": {"train": [0], "val": [1]},
    }))
    return checkpoint, cache


def raw_sample(cache):
    sample = CachedTactileDataset(cache, split="val", horizon=3, normalize=False)[0]
    return sample["observation.state"], sample["observation.tactile"], sample["observation.points"]


def test_normalization_matches_training_and_does_not_mutate_raw_inputs(tmp_path):
    checkpoint, cache = checkpoint_and_cache(tmp_path)
    adapter = PointACTInference(checkpoint)
    raw = raw_sample(cache)
    original = [value.clone() for value in raw]
    normalized = adapter.prepare_observation(*raw)
    expected = CachedTactileDataset(cache, split="val", horizon=3)[0]
    for key in normalized:
        torch.testing.assert_close(normalized[key][0], expected[key], rtol=1e-6, atol=1e-6)
    for value, original_value in zip(raw, original, strict=True):
        torch.testing.assert_close(value, original_value, rtol=0, atol=0)


def test_queue_denormalizes_each_action_once_and_reset_requires_geometry(tmp_path):
    checkpoint, cache = checkpoint_and_cache(tmp_path)
    adapter = PointACTInference(checkpoint)
    raw = raw_sample(cache)
    normalized = adapter.policy.predict_action_chunk(adapter.prepare_observation(*raw))[0]
    mean = torch.tensor(adapter.stats["action_mean"])
    std = torch.tensor(adapter.stats["action_std"])
    expected = normalized * std + mean
    assert adapter.needs_observation
    torch.testing.assert_close(adapter.select_action(*raw), expected[0])
    assert not adapter.needs_observation
    torch.testing.assert_close(adapter.select_action(), expected[1])
    assert adapter.needs_observation
    with pytest.raises(ValueError, match="requires raw_state"):
        adapter.select_action()
    adapter.reset()
    assert adapter.needs_observation
    torch.testing.assert_close(adapter.predict_action_chunk(*raw), expected)
    assert adapter.needs_observation  # Chunk inspection does not consume/prime the queue.


def test_point_builder_only_runs_on_refresh_and_receives_current_context(tmp_path):
    checkpoint, cache = checkpoint_and_cache(tmp_path)
    state, tactile, points = raw_sample(cache)
    calls = []

    def builder(rgb, current_state, current_tactile, context):
        calls.append(context["tick"])
        assert rgb == "synchronized RGB"
        torch.testing.assert_close(current_state, state)
        torch.testing.assert_close(current_tactile, tactile)
        return points

    adapter = PointACTInference(checkpoint, point_builder=builder)
    adapter.select_action(state, tactile, raw_rgb="synchronized RGB", geometry_context={"tick": 0})
    adapter.select_action()
    adapter.select_action(state, tactile, raw_rgb="synchronized RGB", geometry_context={"tick": 2})
    assert calls == [0, 2]


def test_temporal_ensemble_requires_observation_every_tick(tmp_path):
    checkpoint, cache = checkpoint_and_cache(tmp_path, n_action_steps=1, temporal_ensemble_coeff=0.01)
    adapter = PointACTInference(checkpoint)
    raw = raw_sample(cache)
    assert adapter.needs_observation
    adapter.select_action(*raw)
    assert adapter.needs_observation
    with pytest.raises(ValueError, match="requires raw_state"):
        adapter.select_action()


def test_rejects_invalid_normalization_and_raw_shapes(tmp_path):
    checkpoint, cache = checkpoint_and_cache(tmp_path)
    adapter = PointACTInference(checkpoint)
    state, tactile, points = raw_sample(cache)
    with pytest.raises(ValueError, match="raw_state must have shape"):
        adapter.prepare_observation(state[None], tactile, points)
    stats = dict(adapter.stats)
    stats["action_std"] = [0] * 22
    (checkpoint / "normalization.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="strictly positive"):
        PointACTInference(checkpoint)
