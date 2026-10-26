"""Frozen-feature row binding, unchanged ACT targets, and deployment parity."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tactile_vggt.data import CachedTactileDataset, training_statistics
from tactile_vggt.token_policy import TokenACTInference, TokenACTPolicy
from train_token_act import CachedTokenFeatureDataset, run


def small_config():
    return {
        "state_dim": 22, "action_dim": 22, "encoder_dim": 256,
        "dim_model": 32, "n_heads": 4, "dim_feedforward": 64,
        "n_encoder_layers": 1, "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1, "latent_dim": 8,
        "chunk_size": 3, "n_action_steps": 2, "dropout": 0.0,
    }


def make_cache(tmp_path):
    cache, features = tmp_path / "cache", tmp_path / "features"
    cache.mkdir()
    features.mkdir()
    rng = np.random.default_rng(19)
    values = {
        "state": rng.normal(size=(6, 22)).astype(np.float32),
        "action": rng.normal(size=(6, 22)).astype(np.float32),
        "tactile": np.ones((6, 44), dtype=np.float32),
        "episode_index": np.array([0, 0, 1, 1, 2, 2]),
        "frame_index": np.array([0, 1, 0, 1, 0, 1]),
        "obs_frame_indices": np.array([1, 3, 4, 5]),
        "points": np.zeros((4, 556, 8), dtype=np.float32),
    }
    stats = training_statistics(values["state"], values["action"], values["tactile"], values["episode_index"], [0, 2])
    for name, value in values.items():
        np.save(cache / f"{name}.npy", value)
    (cache / "manifest.json").write_text(json.dumps({
        "complete": True, "split": {"train": [0, 2], "val": [1]}, "normalization": stats,
    }))
    np.save(features / "features.npy", np.arange(4 * 256, dtype=np.float32).reshape(4, 256) / 256)
    manifest = {
        "complete": True, "feature_dim": 256, "observation_count": 4, "mode": "internal",
        "cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
        "obs_frame_indices_sha256": hashlib.sha256((cache / "obs_frame_indices.npy").read_bytes()).hexdigest(),
        "source_checkpoint": "frozen-adapter/best.pt", "source_checkpoint_sha256": "test-source-digest",
    }
    (features / "manifest.json").write_text(json.dumps(manifest))
    return cache, features, values


def test_join_uses_original_cache_row_and_keeps_action_protocol(tmp_path):
    cache, features, _ = make_cache(tmp_path)
    actual = CachedTokenFeatureDataset(cache, features, horizon=3)
    original = CachedTactileDataset(cache, horizon=3)
    np.testing.assert_array_equal(actual.observation_indices, [0, 2, 3])
    feature_values = np.load(features / "features.npy")
    for index, row in enumerate([0, 2, 3]):
        sample, reference = actual[index], original[index]
        np.testing.assert_array_equal(sample["observation.environment_state"], feature_values[row])
        for key in ("observation.state", "action", "action_is_pad"):
            torch.testing.assert_close(sample[key], reference[key], rtol=0, atol=0)
        assert not sample["observation.environment_state"].requires_grad
    assert actual[0]["action_is_pad"].tolist() == [False, True, True]
    assert len(CachedTokenFeatureDataset(cache, features, split="val")) == 1


def test_rejects_feature_cache_from_different_original_manifest(tmp_path):
    cache, features, _ = make_cache(tmp_path)
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="cache_manifest_sha256"):
        CachedTokenFeatureDataset(cache, features)


def test_cpu_training_checkpoint_and_live_feature_queue_roundtrip(tmp_path):
    try:
        # ACT is an optional runtime dependency; the current local submodule
        # needs a newer Python than some installed geometry-only torch envs.
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Compatible ACT runtime unavailable: {error}")
    cache, features, values = make_cache(tmp_path)
    config = {"policy": small_config(), "learning_rate": 1e-5, "weight_decay": 1e-4, "grad_clip": 10.0}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    run(Namespace(cache=str(cache), features_dir=str(features), output=str(output), config=str(config_path),
                  steps=1, seed=43, batch_size=2, eval_every=1, eval_batches=16, save_every=1,
                  log_every=1, cpu_threads=1, num_workers=0, device="cpu", amp="float32"))
    checkpoint = (output / "latest").resolve()
    assert (checkpoint / "checkpoint_complete.json").exists()
    assert (output / "best").resolve() == checkpoint
    assert json.loads((checkpoint / "normalization.json").read_text()) == json.loads((cache / "manifest.json").read_text())["normalization"]
    assert json.loads((checkpoint / "feature_manifest.json").read_text()) == json.loads((features / "manifest.json").read_text())
    restored = TokenACTPolicy.from_pretrained(checkpoint)
    assert not hasattr(restored, "point_encoder")
    adapter = TokenACTInference(checkpoint)
    sample = CachedTokenFeatureDataset(cache, features, split="val", horizon=3)[0]
    raw_state = values["state"][3]
    feature_row = np.load(features / "features.npy")[1]
    prepared = adapter.prepare_observation(raw_state, feature_row)
    torch.testing.assert_close(prepared["observation.state"][0], sample["observation.state"])
    torch.testing.assert_close(prepared["observation.environment_state"][0], sample["observation.environment_state"])
    normalized = restored.predict_action_chunk(prepared)[0]
    expected = adapter.denormalize_action(normalized)
    torch.testing.assert_close(adapter.predict_action_chunk(raw_state, feature_row), expected)
    torch.testing.assert_close(adapter.select_action(raw_state, feature_row), expected[0])
    assert not adapter.needs_observation
    torch.testing.assert_close(adapter.select_action(), expected[1])
    assert adapter.needs_observation
    adapter.reset()
    assert adapter.needs_observation
    with pytest.raises(ValueError, match="requires raw_state and features"):
        adapter.select_action()
