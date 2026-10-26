"""Pressure row binding, matched projection ablations, and queued ACT deployment."""

from argparse import Namespace
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tactile_vggt.data import CachedTactileDataset, training_statistics
from tactile_vggt.simple_token_policy import (
    SimpleTokenACTInference, SimpleTokenACTPolicy, make_visual_readout, visual_probe_readout,
)
from train_simple_token_act import CachedSimpleTokenDataset, run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_readout():
    torch.manual_seed(9)
    return nn.Sequential(nn.LayerNorm(1088), nn.Linear(1088, 256), nn.GELU()).state_dict()


def make_cache(tmp_path):
    cache, features = tmp_path / "cache", tmp_path / "features"
    cache.mkdir()
    features.mkdir()
    rng = np.random.default_rng(19)
    tactile = np.ones((6, 44), dtype=np.float32)
    tactile[0] = 1e7  # A training frame that is not an observation must not set p99.
    tactile[3] = 1e8  # Validation pressure must not set p99 either.
    tactile[4] = 2
    tactile[5] = -3
    values = {
        "state": rng.normal(size=(6, 22)).astype(np.float32),
        "action": rng.normal(size=(6, 22)).astype(np.float32), "tactile": tactile,
        "episode_index": np.array([0, 0, 1, 1, 2, 2]),
        "frame_index": np.array([0, 1, 0, 1, 0, 1]),
        "obs_frame_indices": np.array([1, 3, 4, 5]), "points": np.zeros((4, 556, 8), dtype=np.float32),
    }
    stats = training_statistics(values["state"], values["action"], tactile, values["episode_index"], [0, 2])
    for name, value in values.items():
        np.save(cache / f"{name}.npy", value)
    (cache / "manifest.json").write_text(json.dumps({
        "complete": True, "split": {"train": [0, 2], "val": [1]}, "normalization": stats,
        "action_semantics": "Original source action labels unchanged, including their historical offset.",
    }))
    np.save(features / "readout_inputs.npy", rng.normal(size=(4, 1088)).astype(np.float32))
    probe = tmp_path / "visual_probe.pt"
    torch.save({"kwargs": {"mode": "visual"},
                "model": {f"readout.{key}": value for key, value in source_readout().items()}}, probe)
    manifest = {
        "complete": True, "feature_dim": 1088, "observation_count": 4,
        "cache_manifest_sha256": digest(cache / "manifest.json"),
        "obs_frame_indices_sha256": digest(cache / "obs_frame_indices.npy"),
        "readout_inputs_sha256": digest(features / "readout_inputs.npy"),
        "source_checkpoint": str(probe), "source_checkpoint_sha256": digest(probe),
    }
    (features / "manifest.json").write_text(json.dumps(manifest))
    return cache, features, values, probe


def bare_policy(projection, tactile):
    # No ACT dependency is needed to verify the upstream projection/fusion path.
    policy = SimpleTokenACTPolicy.__new__(SimpleTokenACTPolicy)
    nn.Module.__init__(policy)
    policy.config = {"state_dim": 22, "encoder_dim": 300, "action_dim": 22, "chunk_size": 8}
    policy.structure = {"projection": projection, "tactile": tactile}
    policy.readout = make_visual_readout(source_readout(), projection=projection)
    return policy


def test_train_observation_pressure_scale_original_targets_and_row_binding(tmp_path):
    cache, features, _, probe = make_cache(tmp_path)
    dataset = CachedSimpleTokenDataset(cache, features, horizon=8, visual_probe=probe)
    original = CachedTactileDataset(cache, horizon=8)
    np.testing.assert_array_equal(dataset.observation_indices, [0, 2, 3])
    assert dataset.stats["simple_pressure_scale"] == pytest.approx(np.log(3))
    assert dataset.stats["simple_pressure_training_observations"] == 3
    assert dataset.stats["pressure_scale"] == original.stats["pressure_scale"]
    assert CachedSimpleTokenDataset(cache, features, split="val").stats == dataset.stats
    for index, row in enumerate([0, 2, 3]):
        actual, reference = dataset[index], original[index]
        torch.testing.assert_close(actual["observation.visual_readout_input"], torch.from_numpy(np.load(features / "readout_inputs.npy")[row]))
        for name in ("observation.state", "action", "action_is_pad"):
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
    assert dataset[0]["action_is_pad"].tolist() == [False] + [True] * 7
    assert torch.count_nonzero(dataset[2]["observation.tactile"]) == 0


def test_rejects_modified_readout_inputs_and_wrong_visual_probe(tmp_path):
    cache, features, _, probe = make_cache(tmp_path)
    wrong_probe = tmp_path / "wrong.pt"
    wrong_probe.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="source_checkpoint_sha256"):
        CachedSimpleTokenDataset(cache, features, visual_probe=wrong_probe)
    array = np.load(features / "readout_inputs.npy")
    array[0, 0] += 1
    np.save(features / "readout_inputs.npy", array)
    with pytest.raises(ValueError, match="readout_inputs_sha256"):
        CachedSimpleTokenDataset(cache, features, visual_probe=probe)


@pytest.mark.parametrize("projection", ["frozen", "joint"])
def test_projection_initialization_gradient_and_zero_pressure_isolation(projection):
    policy = bare_policy(projection, "zero")
    expected = source_readout()
    for key, value in policy.readout.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    visual = torch.randn(2, 1088, requires_grad=True)
    pressure = torch.randn(2, 44, requires_grad=True)
    batch = {"observation.state": torch.randn(2, 22), "observation.visual_readout_input": visual,
             "observation.tactile": pressure}
    actual = policy._act_batch(batch)["observation.environment_state"]
    changed = policy._act_batch({**batch, "observation.tactile": pressure * 1e9 + 1e9})["observation.environment_state"]
    assert actual.shape == (2, 300)
    torch.testing.assert_close(actual, changed, rtol=0, atol=0)
    actual.square().sum().backward()
    assert pressure.grad is None
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    if projection == "frozen":
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in policy.readout.parameters())
    else:
        assert all(parameter.grad is not None and parameter.requires_grad for parameter in policy.readout.parameters())
        assert sum(parameter.grad.abs().sum() for parameter in policy.readout.parameters()) > 0


def test_pressure_condition_supplies_exact_same_44_slots_without_extra_parameters():
    pressure_policy, zero_policy = bare_policy("joint", "pressure"), bare_policy("joint", "zero")
    batch = {"observation.state": torch.randn(2, 22), "observation.visual_readout_input": torch.randn(2, 1088),
             "observation.tactile": torch.randn(2, 44, requires_grad=True)}
    environment = pressure_policy._act_batch(batch)["observation.environment_state"]
    torch.testing.assert_close(environment[:, 256:], batch["observation.tactile"], rtol=0, atol=0)
    environment.sum().backward()
    torch.testing.assert_close(batch["observation.tactile"].grad, torch.ones(2, 44))
    assert sum(p.numel() for p in pressure_policy.parameters()) == sum(p.numel() for p in zero_policy.parameters())


class FakeQueuedPolicy:
    def __init__(self):
        self.config = {"state_dim": 22, "action_dim": 22, "n_action_steps": 8,
                       "chunk_size": 16, "temporal_ensemble_coeff": None}
        self.act = SimpleNamespace(config=SimpleNamespace(n_action_steps=8))
        self.reset()

    def reset(self):
        self.act._action_queue = deque(maxlen=self.act.config.n_action_steps)

    def select_action(self, batch):
        if not self.act._action_queue:
            assert batch["observation.visual_readout_input"].shape == (1, 1088)
            self.act._action_queue.extend(torch.full((1, 22), float(i)) for i in range(self.act.config.n_action_steps))
        return self.act._action_queue.popleft()


def test_live_pressure_normalization_and_two_step_override_resets_queue(tmp_path, monkeypatch):
    cache, features, values, probe = make_cache(tmp_path)
    dataset = CachedSimpleTokenDataset(cache, features, split="val", visual_probe=probe)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "normalization.json").write_text(json.dumps(dataset.stats))
    (checkpoint / "feature_manifest.json").write_text((features / "manifest.json").read_text())
    fake = FakeQueuedPolicy()
    fake.act._action_queue.append(torch.full((1, 22), 999.0))
    monkeypatch.setattr(SimpleTokenACTPolicy, "from_pretrained", lambda *args, **kwargs: fake)
    adapter = SimpleTokenACTInference(checkpoint, n_action_steps=2)
    assert adapter.checkpoint_n_action_steps == 8
    assert adapter.policy.config["n_action_steps"] == adapter.policy.act.config.n_action_steps == 2
    assert adapter.policy.act._action_queue.maxlen == 2 and adapter.needs_observation
    readout_input = np.load(features / "readout_inputs.npy")[1]
    prepared = adapter.prepare_observation(values["state"][3], readout_input, values["tactile"][3])
    for key in ("observation.state", "observation.visual_readout_input", "observation.tactile"):
        torch.testing.assert_close(prepared[key][0], dataset[0][key])
    action0 = adapter.select_action(values["state"][3], readout_input, values["tactile"][3])
    torch.testing.assert_close(action0, torch.tensor(dataset.stats["action_mean"], dtype=torch.float32))
    assert not adapter.needs_observation
    action1 = adapter.select_action()
    torch.testing.assert_close(action1, torch.tensor(dataset.stats["action_mean"], dtype=torch.float32) + torch.tensor(dataset.stats["action_std"], dtype=torch.float32))
    assert adapter.needs_observation
    adapter.reset()
    with pytest.raises(ValueError, match="fresh query requires"):
        adapter.select_action()
    with pytest.raises(ValueError, match="2 or 8"):
        SimpleTokenACTInference(checkpoint, n_action_steps=4)


def test_visual_probe_mode_is_checked(tmp_path):
    path = tmp_path / "probe.pt"
    torch.save({"kwargs": {"mode": "internal"}, "model": {}}, path)
    with pytest.raises(ValueError, match="visual-mode"):
        visual_probe_readout(path)


def test_actual_act_training_checkpoint_and_raw_input_roundtrip(tmp_path):
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
    except ImportError as error:
        pytest.skip(f"Compatible ACT runtime unavailable: {error}")
    cache, features, values, probe = make_cache(tmp_path)
    config = {"policy": {
        "state_dim": 22, "action_dim": 22, "encoder_dim": 256, "dim_model": 32, "n_heads": 4,
        "dim_feedforward": 64, "n_encoder_layers": 1, "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1, "latent_dim": 8, "chunk_size": 8, "n_action_steps": 8, "dropout": 0.0,
    }, "learning_rate": 1e-5, "weight_decay": 1e-4, "grad_clip": 10.0,
              "notes": "Obsolete label description must not propagate", "experiment": "obsolete"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    run(Namespace(cache=str(cache), features_dir=str(features), visual_probe=str(probe),
                  projection="joint", tactile="pressure", output=str(output), config=str(config_path),
                  steps=1, seed=43, batch_size=2, eval_every=1, eval_batches=16, save_every=1,
                  log_every=1, cpu_threads=1, num_workers=0, device="cpu", amp="float32"))
    checkpoint = (output / "best").resolve()
    assert checkpoint == (output / "latest").resolve()
    assert json.loads((checkpoint / "checkpoint_complete.json").read_text())["readout_trainable"]
    train_config = json.loads((checkpoint / "train_config.json").read_text())
    assert "notes" not in train_config
    assert train_config["action_semantics"] == json.loads((cache / "manifest.json").read_text())["action_semantics"]
    assert train_config["readout_trainable_parameters"] == 280960
    restored = SimpleTokenACTPolicy.from_pretrained(checkpoint)
    assert not hasattr(restored, "point_encoder") and restored.config["encoder_dim"] == 300
    initial_readout = visual_probe_readout(probe)
    assert any(not torch.equal(value, initial_readout[key]) for key, value in restored.readout.state_dict().items())
    adapter = SimpleTokenACTInference(checkpoint, n_action_steps=2)
    readout_input = np.load(features / "readout_inputs.npy")[1]
    prepared = adapter.prepare_observation(values["state"][3], readout_input, values["tactile"][3])
    expected = adapter.denormalize_action(restored.predict_action_chunk(prepared)[0])
    torch.testing.assert_close(adapter.predict_action_chunk(values["state"][3], readout_input, values["tactile"][3]), expected)
    torch.testing.assert_close(adapter.select_action(values["state"][3], readout_input, values["tactile"][3]), expected[0])
    assert not adapter.needs_observation
    torch.testing.assert_close(adapter.select_action(), expected[1])
    assert adapter.needs_observation
