"""Matched 32-row training streams and correctly weighted object-pair losses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from test_camera_pilot import dataset, ToyBlock  # noqa: F401
from tactile_vggt.camera_pilot import CameraArrays
from run_camera_pilot import Batches, sha
import run_camera_pair_pilot as runner


def test_sampler_layout_delegates_ordinary_rows_and_matches_pair_geometry(dataset, monkeypatch):
    data = CameraArrays(dataset[0])
    original_sample = data.sample
    ordinary_calls = []

    def observed_sample(rng, view_rng, batch_size, regime):
        ids = original_sample(rng, view_rng, batch_size, regime)
        ordinary_calls.append((batch_size, regime, ids.copy()))
        return ids

    monkeypatch.setattr(data, "sample", observed_sample)
    sampler = runner.MatchedPairSampler(data, 43)
    assert sampler.ordinary_rows == 16 and sampler.pair_count == 8
    ids = sampler.sample()
    assert ids.shape == (32,) and ids.dtype == np.int64
    assert len(ordinary_calls) == 1 and ordinary_calls[0][:2] == (16, "augmented")
    np.testing.assert_array_equal(ids[:16], ordinary_calls[0][2])
    first, second = ids[16:24], ids[24:32]
    np.testing.assert_array_equal(data.group[first], data.group[second])
    np.testing.assert_array_equal(data.episode[first], data.episode[second])
    np.testing.assert_array_equal(data.camera_variant[first], data.camera_variant[second])
    np.testing.assert_allclose(data.known_cameras[first], data.known_cameras[second], rtol=0, atol=0)
    assert (data.physical_sample[first] != data.physical_sample[second]).all()
    assert (data.variant[first] == 0).all() and (data.variant[second] != 0).all()
    assert (data.split[ids] == 0).all() and (data.camera_variant[ids] <= 3).all()


def test_sampler_same_seed_replays_every_row_for_all_matched_runs(dataset):
    data = CameraArrays(dataset[0])
    samplers = [runner.MatchedPairSampler(data, 43) for _ in range(4)]
    other_seed = runner.MatchedPairSampler(data, 44)
    different = False
    # Objective/mode are deliberately absent from the sampling seed and API.
    for _ in range(10):
        reference = samplers[0].sample()
        for sampler in samplers[1:]:
            np.testing.assert_array_equal(sampler.sample(), reference)
        different |= not np.array_equal(other_seed.sample(), reference)
    assert different


def test_pair_sampling_is_uniform_over_sources_then_siblings_then_views(dataset):
    # Give train source 0 three siblings, source 1 one. Sampling the flat list
    # of all four pairs would incorrectly choose source 0 about 75% of the time.
    root = dataset[0]
    original = {name: np.load(root / f"{name}.npy") for name in CameraArrays.NAMES}
    clones = np.flatnonzero((original["group"] == 0) & (original["variant"] == 1))
    next_physical = int(original["physical_sample"].max()) + 1
    for name, values in original.items():
        extra = []
        for offset, variant in enumerate((2, 3)):
            addition = values[clones].copy()
            if name == "variant":
                addition[:] = variant
            elif name == "physical_sample":
                addition[:] = next_physical + offset
            elif name == "labels":
                addition[:, 0] += (variant - 1) * .02
            elif name == "pressure":
                addition += (variant - 1) * .02
            extra.append(addition)
        np.save(root / f"{name}.npy", np.concatenate([values, *extra]))
    data = CameraArrays(root)
    assert [sum(group == g for group, _, _ in data.training_pairs) for g in (0, 1)] == [3, 1]
    sampler = runner.MatchedPairSampler(data, 20261026)
    rows = np.concatenate([sampler.sample()[24:32] for _ in range(3000)])
    group_zero = data.group[rows] == 0
    assert .47 < group_zero.mean() < .53
    for variant in (1, 2, 3):
        assert .29 < np.mean(data.variant[rows[group_zero]] == variant) < .37
    for view in range(4):
        assert .23 < np.mean(data.camera_variant[rows] == view) < .27


@pytest.mark.parametrize("delta_weight", [0., 1.])
def test_paired_loss_values_and_gradients_use_all_centers_and_only_eight_pairs(delta_weight):
    prediction = torch.linspace(-2., 3., 96, dtype=torch.float64).reshape(32, 3).requires_grad_()
    target = torch.cos(torch.arange(96, dtype=torch.float64)).reshape(32, 3) * .7
    saved_target = target.clone()
    total, center, delta = runner.paired_losses(prediction, target, delta_weight)
    center_error = prediction.detach() - target
    pair_error = center_error[24:32] - center_error[16:24]

    def huber(error):
        return torch.where(error.abs() < 1, .5 * error.square(), error.abs() - .5)

    expected_center = huber(center_error).mean()
    expected_delta = huber(pair_error).mean()
    torch.testing.assert_close(center, expected_center, rtol=0, atol=1e-15)
    torch.testing.assert_close(delta, expected_delta, rtol=0, atol=1e-15)
    torch.testing.assert_close(total, expected_center + delta_weight * expected_delta, rtol=0, atol=1e-15)
    total.backward()
    expected_gradient = center_error.clamp(-1, 1) / 96
    pair_gradient = pair_error.clamp(-1, 1) / 24
    expected_gradient[16:24] -= delta_weight * pair_gradient
    expected_gradient[24:32] += delta_weight * pair_gradient
    torch.testing.assert_close(prediction.grad, expected_gradient, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(target, saved_target)
    assert target.grad is None


@pytest.mark.parametrize("prediction_shape,target_shape", [((31, 3), (31, 3)), ((32, 2), (32, 2)), ((32, 3), (31, 3))])
def test_paired_loss_rejects_batches_that_cannot_have_the_declared_pair_layout(prediction_shape, target_shape):
    with pytest.raises(ValueError):
        runner.paired_losses(torch.zeros(prediction_shape), torch.zeros(target_shape), 1.)


@pytest.mark.parametrize("weight", [-1., .5, 2.])
def test_paired_loss_rejects_objective_weights_outside_the_fixed_protocol(weight):
    with pytest.raises(ValueError):
        runner.paired_losses(torch.zeros(32, 3), torch.zeros(32, 3), weight)


def test_training_rejects_nonmatching_batch_size_before_creating_a_run(dataset, tmp_path):
    data = CameraArrays(dataset[0])
    batches = Batches(data, dataset[1], "cpu")
    args = SimpleNamespace(output=tmp_path / "invalid", objective="center", steps=1,
        batch_size=16, learning_rate=1e-3, eval_every=1, hidden_dim=8,
        device="cpu", resume=False, dev_only=True)
    with pytest.raises(ValueError, match="batch32"):
        runner.train_one(args, data, batches, 43, "augmented", "internal", {}, ToyBlock)
    assert not args.output.exists()


def test_dev_only_matched_objectives_have_same_initialization_and_inputs_then_resume(dataset, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = CameraArrays(dataset[0])
    initial_hashes = []
    adapter_hashes, training_forward_counts = [], []
    original_probe = runner.TokenFusionProbe

    def recording_probe(*args, **kwargs):
        model = original_probe(*args, **kwargs)
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        initial_hashes.append(digest.hexdigest())
        adapter_hashes.append(runner.initialization_digest(model))
        count = {"calls": 0}
        training_forward_counts.append(count)

        def count_training_forward(module, inputs):
            count["calls"] += int(module.training)

        model.register_forward_pre_hook(count_training_forward)
        return model

    monkeypatch.setattr(runner, "TokenFusionProbe", recording_probe)

    class ObservedBatches(Batches):
        def __init__(self):
            super().__init__(data, dataset[1], "cpu")
            self.training_stream = []
            self.test_reads = 0

        def __call__(self, ids):
            ids = np.asarray(ids)
            if (data.split[ids] == 0).all():
                self.training_stream.append(ids.copy())
            self.test_reads += int((data.split[ids] == 2).any())
            return super().__call__(ids)

    streams, markers, args_by_objective, runs, batches_by_objective = [], [], {}, {}, {}
    for objective in ("center", "delta"):
        args = SimpleNamespace(output=tmp_path / f"paired_{objective}", objective=objective, steps=2,
            batch_size=32, learning_rate=1e-3, eval_every=1, hidden_dim=8,
            displacement_weight=0., device="cpu", resume=False, dev_only=True)
        batches = ObservedBatches()
        marker = runner.train_one(args, data, batches, 43, "augmented", "internal", {}, ToyBlock)
        run = args.output / "s43/augmented/internal"
        assert marker["training_complete"] and not marker["complete"]
        assert not (run / "test_predictions.npz").exists() and batches.test_reads == 0
        assert len(batches.training_stream) == 2 and all(ids.shape == (32,) for ids in batches.training_stream)
        streams.append(np.stack(batches.training_stream))
        markers.append(marker)
        expected_stream_hash = hashlib.sha256(np.stack(batches.training_stream).astype("<i8").tobytes()).hexdigest()
        assert marker["row_stream_sha256"] == expected_stream_hash
        assert marker["first_batch_row_ids"] == batches.training_stream[0].tolist()
        assert marker["initialization_sha256"] == adapter_hashes[-1]
        selected = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        selected_stream = np.stack(batches.training_stream[:selected["step"]]).astype("<i8").tobytes()
        assert selected["row_stream_sha256_at_selected_step"] == hashlib.sha256(selected_stream).hexdigest()
        assert selected["initialization_sha256"] == marker["initialization_sha256"]
        args_by_objective[objective], runs[objective], batches_by_objective[objective] = args, run, batches
    np.testing.assert_array_equal(streams[0], streams[1])
    assert len(initial_hashes) == 2 and initial_hashes[0] == initial_hashes[1]
    assert markers[0]["initialization_sha256"] == markers[1]["initialization_sha256"]
    assert markers[0]["row_stream_sha256"] == markers[1]["row_stream_sha256"]
    assert [count["calls"] for count in training_forward_counts] == [2, 2]
    for objective in ("center", "delta"):
        args, run, batches = args_by_objective[objective], runs[objective], batches_by_objective[objective]
        best_hash, history_hash = sha(run / "best.pt"), sha(run / "history.json")
        args.resume, args.dev_only = True, False
        result = runner.train_one(args, data, batches, 43, "augmented", "internal", {}, ToyBlock)
        assert result["complete"] and set(result["domains"]) == {"nominal", "near", "strong"}
        assert (run / "test_predictions.npz").exists() and batches.test_reads > 0
        assert sha(run / "best.pt") == best_hash and sha(run / "history.json") == history_hash
        assert len(batches.training_stream) == 2
        config = json.loads((run / "config.json").read_text())
        assert config["objective"] == objective
        assert config["displacement_weight"] == (0. if objective == "center" else 1.)
    integrity = runner.paired_integrity(tmp_path)
    assert not integrity["complete"]  # Other seeds/mode were deliberately not run.
    matched = next(r for r in integrity["pairs"] if r["seed"] == 43 and r["mode"] == "internal")
    assert matched["matched"] and matched["row_stream_sha256"] == markers[0]["row_stream_sha256"]
    # Correct-looking completion markers must not conceal a mismatched stream
    # or a second objective trained with another learning rate.
    run = runs["delta"]
    marker = json.loads((run / "completion.json").read_text())
    marker["row_stream_sha256"] = "0" * 64
    runner.dump(run / "completion.json", marker)
    with pytest.raises(ValueError, match="row_stream_sha256"):
        runner.paired_integrity(tmp_path)
    marker["row_stream_sha256"] = markers[0]["row_stream_sha256"]
    config = json.loads((run / "config.json").read_text())
    config["learning_rate"] *= 2
    runner.dump(run / "config.json", config)
    marker["config_sha256"] = sha(run / "config.json")
    runner.dump(run / "completion.json", marker)
    with pytest.raises(ValueError, match="shared configuration differs"):
        runner.paired_integrity(tmp_path)
