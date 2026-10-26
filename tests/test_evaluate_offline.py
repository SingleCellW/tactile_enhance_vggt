"""Offline metrics must exclude padding and never expose action labels to ACT."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("evaluate_offline", Path(__file__).resolve().parents[1] / "scripts/evaluate_offline.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_padding_and_contact_subsets_use_exact_scalar_denominators():
    error = np.array([[[1, 2], [10000, 10000]], [[3, 4], [5, 6]]])
    valid = np.array([[True, False], [True, True]])
    metric = evaluation.ErrorAccumulator([2, 4])
    metric.add(error, valid)
    result = metric.result()
    assert result["observations"] == 2 and result["joint_scalar_count"] == 6
    assert result["normalized_l1"] == 3.5 and result["physical_joint_l1"] == 11
    np.testing.assert_allclose(result["per_joint_normalized_l1"], [3, 4])
    active = evaluation.ErrorAccumulator([2, 4])
    active.add(error, valid, np.array([True, False]))
    assert active.result()["normalized_l1"] == 1.5 and active.result()["physical_joint_l1"] == 5
    empty = evaluation.ErrorAccumulator([2, 4])
    empty.add(error, valid, np.zeros(2, dtype=bool))
    assert empty.result()["normalized_l1"] is None and empty.result()["observations"] == 0


def test_prediction_does_not_receive_actions_padding_or_metadata():
    class SpyPolicy:
        def predict_action_chunk(self, observations):
            assert set(observations) == set(evaluation.OBSERVATION_KEYS)
            return observations["observation.state"][:, None].expand(-1, 3, -1)
    batch = {"observation.state": torch.tensor([[1., 2.]]),
             "observation.points": torch.zeros(1, 5, 8), "observation.tactile": torch.zeros(1, 2),
             "action": torch.full((1, 3, 2), 999.), "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
             "episode_index": torch.tensor([10])}
    output = evaluation.predict_observations_only(SpyPolicy(), batch, torch.device("cpu"), "float32")
    torch.testing.assert_close(output, torch.tensor([[[1., 2.], [1., 2.], [1., 2.]]]))


def test_selects_best_saved_but_keeps_final_equal_budget():
    reports = []
    for step, error in ((100, 0.5), (5000, 0.3), (36000, 0.4)):
        reports.append({"step": step, "checkpoint": f"checkpoint/{step}",
                        "full_chunk": {"all": {"normalized_l1": error, "physical_joint_l1": error / 2},
                                       "active_contact": {"normalized_l1": error}},
                        "first_action": {"all": {"normalized_l1": error}}})
    selected = evaluation.select_checkpoints(reports, 36000)
    assert selected["best_saved"]["step"] == 5000
    assert selected["final_equal_budget"]["step"] == 36000
    assert evaluation.select_checkpoints(reports, 40000)["final_equal_budget"] is None


def test_refuses_checkpoint_from_different_cache_or_normalization(tmp_path):
    (tmp_path / "train_config.json").write_text(json.dumps({"data_manifest_sha256": "correct"}))
    (tmp_path / "normalization.json").write_text(json.dumps({"action_std": [2, 4]}))
    with pytest.raises(ValueError, match="different cache"):
        evaluation.validate_checkpoint(tmp_path, "changed", {"action_std": [2, 4]})
    with pytest.raises(ValueError, match="normalization differs"):
        evaluation.validate_checkpoint(tmp_path, "correct", {"action_std": [1, 1]})
    evaluation.validate_checkpoint(tmp_path, "correct", {"action_std": [2, 4]})
