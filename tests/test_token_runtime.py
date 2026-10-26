"""CPU parity of cached probe inputs, live features, coordinates and RGB rows."""

from argparse import Namespace
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_cache
import run_token_pilot
import token_runtime
from tactile_vggt.token_fusion import TokenFusionProbe


class ToyFrozenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.calls = 0

    def forward(self, tokens, pos=None):
        self.calls += 1
        positions = pos.sum(-1, keepdim=True).to(tokens.dtype) * 0.001
        return tokens + self.scale * torch.tanh(tokens + 0.1 * tokens.mean(1, keepdim=True) + positions)


class ToyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_blocks = nn.ModuleList([ToyFrozenBlock()])
        self.fail = False

    @staticmethod
    def prefix(images):
        b, s = images.shape[:2]
        signal = images.mean(dim=(2, 3, 4)).reshape(b, s, 1, 1)
        channel = torch.linspace(-0.5, 0.5, 1024, device=images.device).reshape(1, 1, 1, -1)
        patch = torch.arange(8, device=images.device).reshape(1, 1, 8, 1) * 0.03
        tokens = (signal + channel + patch).expand(b, s, 8, 1024).reshape(b, s * 8, 1024)
        pos = torch.arange(s * 8 * 2, device=images.device).reshape(1, s * 8, 2).expand(b, -1, -1)
        return tokens, pos

    def forward(self, images):
        if self.fail:
            raise RuntimeError("intentional prefix failure")
        tokens, pos = self.prefix(images)
        return self.global_blocks[-1](tokens, pos=pos)


@pytest.fixture
def runtime_case(tmp_path, monkeypatch):
    # Exercise the tensor path on CPU without changing production's CUDA path.
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    rng = np.random.default_rng(47)
    n = 4
    rgb = rng.integers(0, 256, (n, 2, 224, 224, 3), dtype=np.uint8)
    cameras = np.tile(np.eye(4, dtype=np.float32), (n, 2, 1, 1))
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
    cameras[:, :, :3, :3] = rotation
    cameras[:, 0, :3, 3] = [0.2, -0.3, 0.5]
    cameras[:, 1, :3, 3] = [0.4, -0.3, 0.5]
    raw = {"rgb": rgb, "state": rng.normal(size=(n, 22)).astype(np.float32),
           "pressure": rng.uniform(0, 0.01, (n, 44)).astype(np.float32),
           "anchors": rng.normal(size=(n, 44, 3)).astype(np.float32), "known_cameras": cameras}
    raw["pressure"][:, 0] = 0.0001
    data_dir, prefix_dir = tmp_path / "data", tmp_path / "prefix"
    data_dir.mkdir()
    prefix_dir.mkdir()
    arrays = {**raw, "episode": np.array([0, 0, 1, 2]), "group": np.arange(n),
              "split": np.array([0, 0, 1, 2]), "labels": rng.normal(size=(n, 3)).astype(np.float32),
              "palm": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
              "variant": np.zeros(n, dtype=np.int64), "occlusion": np.zeros((n, 2), dtype=np.float32)}
    for name, value in arrays.items():
        np.save(data_dir / f"{name}.npy", value)
    images = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).float() / 255
    tokens, pos = ToyAggregator.prefix(images)
    np.save(prefix_dir / "tokens.npy", tokens.reshape(n, 2, 8, 1024).numpy().astype(np.float16))
    np.save(prefix_dir / "positions.npy", pos[0].reshape(2, 8, 2).numpy())
    data = run_token_pilot.PilotArrays(data_dir, prefix_dir)
    # Constructor provenance is tested separately by the runtime owner. Bypass
    # model loading here to isolate preprocessing and frozen-block hook parity.
    encoder = token_runtime.LiveTokenEncoder.__new__(token_runtime.LiveTokenEncoder)
    encoder.device = "cpu"
    encoder.backbone = nn.Module()
    encoder.backbone.aggregator = ToyAggregator().eval().requires_grad_(False)
    encoder.probes, encoder.stats = {}, {}
    for mode in ("internal", "late", "visual"):
        torch.manual_seed(7)
        kwargs = {"mode": mode, "hidden_dim": 8, "num_heads": 2, "initial_gate": 0.2}
        trained = TokenFusionProbe(copy.deepcopy(encoder.backbone.aggregator.global_blocks[-1]), **kwargs).eval()
        checkpoint = {
            "kwargs": kwargs,
            "model": {key: value for key, value in trained.state_dict().items() if not key.startswith("frozen_global_block.")},
            "state_mean": data.state_mean, "state_std": data.state_std, "pressure_scale": data.pressure_scale,
            "whiten_target": True, "target_mean": data.target_mean, "target_std": data.target_std,
        }
        model = TokenFusionProbe(encoder.backbone.aggregator.global_blocks[-1], **kwargs).eval()
        model.load_state_dict(checkpoint["model"], strict=False)
        encoder.probes[mode] = model
        encoder.stats[mode] = checkpoint
    return encoder, data, raw


def test_live_features_equal_cached_training_inputs_and_hooks_are_removed(runtime_case):
    encoder, data, raw = runtime_case
    block = encoder.backbone.aggregator.global_blocks[-1]
    actual = encoder.encode(raw["rgb"], raw["state"], raw["pressure"], raw["anchors"], raw["known_cameras"])
    batch, _ = data.batch(np.arange(4), "cpu")
    for mode, model in encoder.probes.items():
        expected = model(**batch)["features"].detach().numpy()
        np.testing.assert_allclose(actual[mode], expected, rtol=1e-6, atol=1e-6)
        assert model.frozen_global_block is block
    assert not block._forward_pre_hooks
    # Three runtime probes plus the three explicit cached comparisons above;
    # the capture hook aborts before the prefix's block forward can execute.
    assert block.calls == 6
    for saved in encoder.stats.values():
        saved["target_mean"] = np.full(3, 1e6)
        saved["target_std"] = np.full(3, 1e5)
        saved["whiten_target"] = False
    again = encoder.encode(raw["rgb"], raw["state"], raw["pressure"], raw["anchors"], raw["known_cameras"])
    for mode in actual:
        np.testing.assert_array_equal(actual[mode], again[mode])


def test_live_world_to_base_transform_matches_direct_base_inputs(runtime_case):
    encoder, _, raw = runtime_case
    angle = -0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, [0.8, -0.4, 0.3]
    cameras = raw["known_cameras"][0].astype(np.float64).copy()
    cameras[:, :3, :3] = np.einsum("ij,vjk->vik", rotation, cameras[:, :3, :3])
    cameras[:, :3, 3] = cameras[:, :3, 3] @ rotation.T + transform[:3, 3]
    observation = {"base_to_world": transform, "anchors_world": raw["anchors"][0] @ rotation.T + transform[:3, 3],
                   "known_cameras_cv": cameras, "rgb": raw["rgb"][0], "state": raw["state"][0], "tactile": raw["pressure"][0]}
    actual = encoder.encode_observation(observation)
    expected = encoder.encode(raw["rgb"][0], raw["state"][0], raw["pressure"][0], raw["anchors"][0], raw["known_cameras"][0])
    for mode in expected:
        np.testing.assert_allclose(actual[mode], expected[mode][0], rtol=2e-6, atol=2e-6)


def test_hook_is_removed_when_prefix_extraction_fails(runtime_case):
    encoder, _, raw = runtime_case
    encoder.backbone.aggregator.fail = True
    with pytest.raises(RuntimeError, match="intentional prefix failure"):
        encoder.encode(raw["rgb"], raw["state"], raw["pressure"], raw["anchors"], raw["known_cameras"])
    assert not encoder.backbone.aggregator.global_blocks[-1]._forward_pre_hooks


def test_export_reads_audited_rgb_row_plus_one_in_original_cache_order(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "export"
    cache.mkdir()
    calibration = tmp_path / "cameras.json"
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, 0, 3] = 0.2
    calibration.write_text(json.dumps({"cameras": {name: {"extrinsics": pose.tolist()} for name, pose in zip(prepare_cache.CAMERAS, poses)}}))
    manifest = {"complete": True, "native_tactile_report": {"velocity_dt": 0.01}, "configuration": {
        "base_to_world": np.eye(4).tolist(), "source": {
            "camera_metadata": [{"path": str(calibration)}] * 2,
            "rgb_videos": [{"path": "left"}, {"path": "right"}],
        }}}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    arrays = {"state": np.arange(88, dtype=np.float32).reshape(4, 22), "tactile": np.zeros((4, 44), dtype=np.float32),
              "points": np.zeros((2, 556, 8), dtype=np.float32), "episode_index": np.array([0, 0, 1, 1]),
              "obs_frame_indices": np.array([0, 2]), "rgb_source_frame_indices": np.array([1, 3])}
    for name, value in arrays.items():
        np.save(cache / f"{name}.npy", value)
    reads, closed = [], []

    class Reader:
        def __init__(self, path, count):
            self.path = path
            assert count == 4

        def read(self, row):
            reads.append((self.path, row))
            return np.full((224, 224, 3), row, dtype=np.uint8)

        def close(self):
            closed.append(self.path)

    class Encoder:
        def __init__(self, *args):
            pass

        def encode(self, rgb, state, pressure, anchors, cameras):
            np.testing.assert_array_equal(state, arrays["state"][[0, 2]])
            return {"internal": np.repeat(rgb[:, 0, 0, 0, :1], 256, axis=1).astype(np.float32)}

    monkeypatch.setattr(prepare_cache, "VideoReader", Reader)
    monkeypatch.setattr(token_runtime, "LiveTokenEncoder", Encoder)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"bound-source-checkpoint")
    args = Namespace(cache=cache, output=output, repository=tmp_path, weights=tmp_path / "unused.pt",
                     checkpoints={"internal": checkpoint}, device="cpu", batch_size=4)
    token_runtime.export_features(args)
    assert reads == [("left", 1), ("right", 1), ("left", 3), ("right", 3)]
    assert closed == ["left", "right"]
    values = np.load(output / "internal" / "features.npy")
    np.testing.assert_array_equal(values[:, 0], [1, 3])
    exported = json.loads((output / "internal" / "manifest.json").read_text())
    assert exported["complete"] and exported["observation_count"] == 2
    assert exported["obs_frame_indices_sha256"] == token_runtime.sha(cache / "obs_frame_indices.npy")
