"""CPU tests of camera geometry, leakage, pairing and train/evaluate handoff."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from tactile_vggt.camera_pilot import (CameraArrays, camera_vectors_to_base,
    evaluate_predictions, source_equal_displacement)
from run_camera_pilot import Batches, train_one, sha, convergence, verify_data_files


@pytest.fixture
def dataset(tmp_path):
    data, prefix = tmp_path / "data", tmp_path / "prefix"
    data.mkdir(); prefix.mkdir()
    arrays = {key: [] for key in CameraArrays.NAMES}
    physical = 0
    for split in range(3):
        for group_index in range(2):
            group = split * 10 + group_index
            for variant in range(2):
                for view in range(7 if split == 2 else 4):
                    angle = .04 * view
                    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                    camera = np.eye(4); camera[:3, :3] = rotation
                    camera[:3, 3] = [.1 + view * .01, .2, 1]
                    right = camera.copy(); right[:3, 3] += rotation @ [1., 0, 0]
                    palm = np.eye(4); palm[:3, 3] = [.3, 0, .1]
                    row = {"episode": split, "group": group, "split": split, "variant": variant,
                           "physical_sample": physical, "camera_variant": view,
                           "camera_domain": 0 if view == 0 else 1 if view <= 3 else 2,
                           "pressure": np.full(44, .1 + variant * .02), "state": np.full(22, group_index * .1),
                           "anchors": np.full((44, 3), .2),
                           "labels": np.array([.4 + .05 * group_index + variant * .02, .05, .12]),
                           "palm": palm, "known_cameras": np.stack([camera, right])}
                    for key in arrays:
                        arrays[key].append(row[key])
                physical += 1
    for key, value in arrays.items():
        np.save(data / (key + ".npy"), np.asarray(value))
    (data / "manifest.json").write_text(json.dumps({"complete": True, "reference_baseline_m": 1.}))
    n = len(arrays["episode"])
    np.save(prefix / "tokens.npy", np.random.default_rng(2).normal(size=(n, 2, 261, 4)).astype(np.float16))
    np.save(prefix / "positions.npy", np.zeros((2, 261, 2), dtype=np.int64))
    return data, prefix


def test_camera_relative_vector_roundtrip_and_perfect_metrics(dataset):
    data = CameraArrays(dataset[0])
    predicted = data.base_predictions(data.target[data.test_ids], data.test_ids)
    np.testing.assert_allclose(predicted, data.labels[data.test_ids], atol=1e-12)
    # A large extrinsic translation must not be added to a relative vector.
    pose = np.eye(4)[None]; pose[:, :3, 3] = 100
    np.testing.assert_allclose(camera_vectors_to_base(np.array([[1., 2, 3]]), pose, 2), [[2, 4, 6]])
    metrics, _, _ = evaluate_predictions(data, data.test_ids, data.target[data.test_ids])
    for domain in metrics["domains"].values():
        assert domain["center"]["episode_mean_mm"] < 1e-10
        assert domain["displacement"]["source_equal_metrics"]["displacement_error_mm"]["mean"] < 1e-10
        assert domain["displacement"]["source_equal_metrics"]["zero_shift_error_mm"]["mean"] == pytest.approx(20)
        if domain["cross_view"]:
            assert domain["cross_view"]["episode_mean_mm"] < 1e-10


def test_no_response_is_consistent_but_not_correct(dataset):
    data = CameraArrays(dataset[0])
    metrics, _, _ = evaluate_predictions(data, data.test_ids, np.zeros_like(data.target[data.test_ids]))
    assert metrics["domains"]["near"]["cross_view"]["episode_mean_mm"] == 0
    assert metrics["domains"]["near"]["center"]["episode_mean_mm"] > 50
    assert metrics["domains"]["near"]["displacement"]["source_equal_metrics"]["displacement_error_mm"]["mean"] == pytest.approx(20)


def test_sampling_fairness_and_train_only_shared_statistics(dataset):
    data = CameraArrays(dataset[0])
    fixed = data.sample(np.random.default_rng(43), np.random.default_rng(9), 10000, "fixed")
    aug = data.sample(np.random.default_rng(43), np.random.default_rng(9), 10000, "augmented")
    np.testing.assert_array_equal(data.physical_sample[fixed], data.physical_sample[aug])
    assert (data.camera_variant[fixed] == 0).all()
    for view in range(4):
        assert .23 < np.mean(data.camera_variant[aug] == view) < .27
    before = data.normalization
    for name in ["labels", "pressure", "state"]:
        values = np.load(dataset[0] / (name + ".npy"))
        values[data.split != 0] += 100
        np.save(dataset[0] / (name + ".npy"), values)
    assert CameraArrays(dataset[0]).normalization == before


@pytest.mark.parametrize("change", ["physical", "episode", "camera"])
def test_camera_variants_cannot_change_physics_or_split_or_pair_pose(dataset, change):
    root = dataset[0]
    if change == "physical":
        pressure = np.load(root / "pressure.npy"); pressure[1, 0] += 1
        np.save(root / "pressure.npy", pressure)
    elif change == "episode":
        episode = np.load(root / "episode.npy"); episode[episode == 1] = 0
        np.save(root / "episode.npy", episode)
    else:
        camera = np.load(root / "known_cameras.npy")
        # Different object sibling at the same source/view moves its camera.
        camera[4, :, 0, 3] += .01
        np.save(root / "known_cameras.npy", camera)
    with pytest.raises(ValueError):
        CameraArrays(root)


def test_counterfactual_weights_sources_before_episodes():
    labels = np.array([[0., 0, 0], [.02, 0, 0], [.02, 0, 0], [.02, 0, 0], [0, 0, 0], [.02, 0, 0]])
    pred = np.array([[0., 0, 0], [.01, 0, 0], [.01, 0, 0], [.01, 0, 0], [0, 0, 0], [-.01, 0, 0]])
    data = SimpleNamespace(labels=labels, group=np.array([1, 1, 1, 1, 2, 2]), episode=np.zeros(6, dtype=int),
                           object_pairs=lambda ids: [(1, 0, 1), (1, 0, 2), (1, 0, 3), (2, 4, 5)])
    result = source_equal_displacement(data, np.arange(6), pred)
    assert result["pairs"] == 4 and result["source_groups"] == 2
    assert result["source_equal_metrics"]["displacement_error_mm"]["mean"] == pytest.approx(20)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(4, 4)
    def forward(self, x, pos=None):
        return self.linear(x)


def test_development_only_then_evaluate_same_best_and_reject_changed_budget(dataset, tmp_path):
    torch.set_num_threads(1)
    data = CameraArrays(dataset[0]); batches = Batches(data, dataset[1], "cpu")
    args = SimpleNamespace(output=tmp_path / "runs", steps=2, batch_size=4, learning_rate=1e-3,
                           eval_every=1, hidden_dim=8, displacement_weight=0., device="cpu", resume=False, dev_only=True)
    marker = train_one(args, data, batches, 43, "fixed", "internal", {}, ToyBlock)
    run = args.output / "s43/fixed/internal"
    assert marker["training_complete"] and not marker["complete"]
    assert not (run / "test_predictions.npz").exists()
    checkpoint_hash, history_hash = sha(run / "best.pt"), sha(run / "history.json")
    args.resume = True; args.dev_only = False
    result = train_one(args, data, batches, 43, "fixed", "internal", {}, ToyBlock)
    assert result["complete"] and set(result["domains"]) == {"nominal", "near", "strong"}
    assert sha(run / "best.pt") == checkpoint_hash and sha(run / "history.json") == history_hash
    args.steps = 3
    with pytest.raises(ValueError, match="exact saved configuration"):
        train_one(args, data, batches, 43, "fixed", "internal", {}, ToyBlock)


def test_optional_pair_objective_is_recorded_separately(dataset, tmp_path):
    data = CameraArrays(dataset[0]); batches = Batches(data, dataset[1], "cpu")
    args = SimpleNamespace(output=tmp_path / "pair", steps=1, batch_size=2, learning_rate=1e-3,
                           eval_every=1, hidden_dim=8, displacement_weight=.5, device="cpu", resume=False, dev_only=True)
    marker = train_one(args, data, batches, 43, "augmented", "tactile", {}, ToyBlock)
    assert marker["training_complete"]
    config = json.loads((args.output / "s43/augmented/tactile/config.json").read_text())
    assert config["displacement_weight"] == .5


def test_budget_decision_reads_only_completed_development_history(tmp_path):
    for regime in ["fixed", "augmented"]:
        for mode in ["visual", "tactile", "late", "internal"]:
            p = tmp_path / "s43" / regime / mode;p.mkdir(parents=True)
            (p / "completion.json").write_text(json.dumps({"training_complete": True, "train_steps": 2000, "best_step": 2000}))
            (p / "history.json").write_text(json.dumps([{"step": s, "dev_episode_mean_mm": x} for s, x in zip([1200,1400,1600,1800,2000], [20,19,18,17,16])]))
    result = convergence(tmp_path, 2000)
    assert result["complete"] and result["extend_all_to_4000"] and result["flagged_runs"] == 8


def test_actual_arrays_are_bound_to_manifest_not_just_manifest_text(dataset):
    root = dataset[0]
    np.save(root / "rgb.npy", np.zeros((60, 2, 2, 2, 3), dtype=np.uint8))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = {p.stem: {"sha256": sha(p)} for p in root.glob("*.npy")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    verify_data_files(root)
    values = np.load(root / "labels.npy"); values[0, 0] += .01
    np.save(root / "labels.npy", values)
    with pytest.raises(ValueError, match="array changed: labels"):
        verify_data_files(root)
