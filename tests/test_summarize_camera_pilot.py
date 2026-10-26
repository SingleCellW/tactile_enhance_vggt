"""Independent CPU checks for camera-pilot metrics and reporting boundaries."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("camera_summary", ROOT / "scripts/summarize_camera_pilot.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def synthetic_predictions():
    arrays = {key: [] for key in ("indices", "prediction_camera", "target_camera", "prediction_base", "target_base",
        "episode", "group", "physical_sample", "variant", "camera_variant", "camera_domain", "camera_to_base", "errors_mm")}
    for physical in range(4):
        group = physical // 2
        palm = np.array([.2 * group, 0., 0.])
        target = palm + [.02 * (physical % 2) if not group else 0., .1 * group + .03 * (physical % 2) * group, .1]
        prediction = target + [.01, .02, .03]
        for view in range(7):
            angle = view * .07
            c, s = np.cos(angle), np.sin(angle)
            pose = np.eye(4)
            pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            pose[:3, 3] = [1 + .01 * view, 2, .5]
            values = {"indices": len(arrays["indices"]), "prediction_camera": (prediction-palm) @ pose[:3, :3] / 2.,
                "target_camera": (target-palm) @ pose[:3, :3] / 2., "prediction_base": prediction, "target_base": target,
                "episode": 11 + group * 7, "group": group + 100, "physical_sample": physical, "variant": physical % 2,
                "camera_variant": view, "camera_domain": 0 if not view else 1 if view <= 3 else 2,
                "camera_to_base": pose, "errors_mm": np.linalg.norm(prediction-target)*1000}
            for key, value in values.items():
                arrays[key].append(value)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    arrays["baseline_m"] = np.array(2.)
    return arrays


def test_recompute_physical_error_crossview_and_source_equal_displacement(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez(path, **synthetic_predictions())
    domains, identity, baseline = summary.audit_predictions(path, {"nominal": 4, "near": 12, "strong": 12})
    assert baseline == 2.
    for name, value in domains.items():
        assert value["center"]["episodes"] == 2
        assert value["center"]["episode_mean_mm"] == pytest.approx(np.sqrt(1400))
        assert value["displacement"]["source_groups"] == 2
        assert value["displacement"]["source_equal_metrics"]["displacement_error_mm"]["mean"] == pytest.approx(0., abs=1e-12)
        assert value["displacement"]["source_equal_metrics"]["zero_shift_error_mm"]["mean"] == pytest.approx(25.)
        if name != "nominal":
            assert value["cross_view"]["episode_mean_mm"] == pytest.approx(0.)
    assert domains["nominal"]["displacement"]["pairs"] == 2
    assert domains["near"]["displacement"]["pairs"] == 6
    assert len(identity["physical_sample"]) == 28


def test_wrong_camera_origin_or_mismatched_sibling_camera_is_rejected(tmp_path):
    path = tmp_path / "predictions.npz"
    arrays = synthetic_predictions()
    arrays["prediction_base"][0] += arrays["camera_to_base"][0, :3, 3]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="relative vectors must rotate without translation"):
        summary.audit_predictions(path, {"nominal": 4, "near": 12, "strong": 12})
    arrays = synthetic_predictions()
    arrays["camera_to_base"][7, 0, 3] += .1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="confounds different cameras"):
        summary.audit_predictions(path, {"nominal": 4, "near": 12, "strong": 12})


def test_independent_recompute_matches_actual_runner_metric_schema(tmp_path):
    from tactile_vggt.camera_pilot import CameraArrays, camera_vectors_to_base, evaluate_predictions
    arrays = synthetic_predictions()
    n = len(arrays["indices"])
    palm = np.tile(np.eye(4), (n, 1, 1))
    palm[:, :3, 3] = arrays["target_base"] - camera_vectors_to_base(arrays["target_camera"], arrays["camera_to_base"], 2.)
    data = SimpleNamespace(**{name: arrays[name] for name in ("episode", "group", "physical_sample", "variant", "camera_variant", "camera_domain")},
        labels=arrays["target_base"], camera=arrays["camera_to_base"], baseline=2., palm=palm)
    data.lookup = {(int(p), int(v)): i for i, (p, v) in enumerate(zip(data.physical_sample, data.camera_variant))}
    data.object_pairs = lambda ids: CameraArrays.object_pairs(data, ids)
    data.base_predictions = lambda predicted, ids: CameraArrays.base_predictions(data, predicted, ids)
    raw, _, _ = evaluate_predictions(data, np.arange(n), arrays["prediction_camera"])
    path = tmp_path / "predictions.npz"
    np.savez(path, **arrays)
    rebuilt, _, _ = summary.audit_predictions(path, {"nominal": 4, "near": 12, "strong": 12})
    summary.compare_recorded_domains(raw["domains"], rebuilt)


def test_seed_average_ci_keeps_original_episode_count_and_missing_is_null():
    result = summary.seed_descriptive({43: {"11": 1., "18": 11.}, 44: {"11": 2., "18": 12.}, 45: {"11": 3., "18": 13.}})
    assert result["complete"]
    assert result["mean"] == 7.
    assert result["sample_sd"] == 1.
    assert result["episodes"] == 2
    assert result["conditional_episode_ci95"] == [2., 12.]
    partial = summary.seed_descriptive({43: {"11": 1., "18": 11.}})
    assert not partial["complete"] and partial["mean"] is None and partial["conditional_episode_ci95"] is None


def test_difference_in_differences_uses_same_paired_episodes():
    runs = {}
    for seed in summary.SEEDS:
        for regime, mode, values in (("fixed", "visual", [10, 20]), ("augmented", "visual", [8, 18]),
                                      ("fixed", "internal", [9, 19]), ("augmented", "internal", [4, 14])):
            runs[(seed, regime, mode)] = {"complete": True, "domains": {"near": {"center": {"per_episode": {"11": values[0], "18": values[1]}}}}}
    result = summary.paired_contrast(runs, [("augmented", "internal", 1), ("fixed", "internal", -1),
        ("augmented", "visual", -1), ("fixed", "visual", 1)], "near", "center_mm")
    assert result["complete"]
    assert result["across_seed_descriptive"]["mean"] == -3.
    assert result["across_seed_descriptive"]["episodes"] == 2
    assert result["across_seed_descriptive"]["conditional_episode_ci95"] == [-3., -3.]
    assert all(value["ci95"] == [-3., -3.] for value in result["per_seed"].values())


def test_empty_experiment_remains_incomplete_without_invented_values(tmp_path):
    source, output = tmp_path / "source", tmp_path / "report"
    source.mkdir()
    result = summary.summarize(argparse.Namespace(input=source, output=output, budget_decision=None,
        displacement_weight=0., plots=False))
    assert not result["complete"] and result["completed_runs"] == 0
    assert len(result["runs"]) == 24
    assert result["descriptive"]["fixed"]["visual"]["near"]["center_mm"]["mean"] is None
    saved = json.loads((output / "results.json").read_text())
    assert not saved["complete"] and (output / "results.csv").is_file()
