"""CPU-only follow-up reporting checks; no trained models or remote jobs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("camera_pair_summary", ROOT / "scripts/summarize_camera_pair_pilot.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
from test_summarize_camera_pilot import synthetic_predictions


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


@pytest.fixture
def complete_toy(tmp_path, monkeypatch):
    monkeypatch.setattr(summary.base, "EXPECTED_COUNTS", {"nominal": 4, "near": 12, "strong": 12})
    source = tmp_path / "source"
    scheme = {"batch_size": 32, "ordinary_rows": 16, "object_pairs": 8,
        "forward_calls_per_optimizer_step": 1, "layout": "ordinary[0:16], first[16:24], second[24:32]", "weights": summary.WEIGHTS}
    protocol = {"conditional_followup": True, "steps": 4000, "seeds": list(summary.SEEDS),
        "modes": list(summary.MODES), "objectives": list(summary.OBJECTIVES), "batch_scheme": scheme,
        "data_manifest_sha256": "a" * 64, "prefix_manifest_sha256": "b" * 64}
    dump(source / "pair_protocol.json", protocol)
    integrity_rows = []
    for objective in summary.OBJECTIVES:
        root = source / ("paired_" + objective)
        dump(root / "normalization.json", {"source": "nominal train only"})
        dump(root / "environment.json", {"device": "cpu", "python": "3.12.13", "torch": "2.11.0", "numpy": "2.2.6",
            "cuda_version": None, "gpu": None, "cuda_visible_devices": "0", "python_executable": "/first/python", "platform": "test-linux"})
        dump(root / "summary.json", {"full_design_complete": True, "complete": True})
        for seed in summary.SEEDS:
            initial = hashlib.sha256((str(seed) + "initial").encode()).hexdigest()
            stream = hashlib.sha256((str(seed) + "rows").encode()).hexdigest()
            for mode in summary.MODES:
                directory = root / f"s{seed}/augmented/{mode}"
                config = {"seed": seed, "regime": "augmented", "mode": mode, "objective": objective,
                    "displacement_weight": summary.WEIGHTS[objective], "batch_size": 32, "steps": 4000,
                    "eval_every": 4000, "learning_rate": 1e-4, "batch_scheme": scheme, "reference_baseline_m": 2.,
                    "normalization_sha256": summary.base.sha(root / "normalization.json"),
                    "environment_sha256": summary.base.sha(root / "environment.json"),
                    **{key: protocol[key] for key in ("data_manifest_sha256", "prefix_manifest_sha256")}}
                dump(directory / "config.json", config)
                history = [{"step": 1, "dev_episode_mean_mm": 10., "row_stream_sha256": "c" * 64},
                           {"step": 4000, "dev_episode_mean_mm": 9., "row_stream_sha256": stream}]
                dump(directory / "history.json", history)
                # A hash-only CPU audit need not unpickle this toy checkpoint.
                (directory / "best.pt").write_bytes(f"toy-checkpoint:{objective}:{seed}:{mode}".encode())
                checkpoint_sha = summary.base.sha(directory / "best.pt")
                np.savez(directory / "test_predictions.npz", **synthetic_predictions())
                domains, _, _ = summary.base.audit_predictions(directory / "test_predictions.npz")
                metrics = {"complete": True, "seed": seed, "regime": "augmented", "mode": mode,
                    "objective": objective, "best_step": 4000, "train_steps": 4000, "dev_episode_mean_mm": 9.,
                    "initialization_sha256": initial, "row_stream_sha256": stream,
                    "config_sha256": summary.base.sha(directory / "config.json"), "checkpoint_sha256": checkpoint_sha,
                    "domains": domains}
                dump(directory / "metrics.json", metrics)
                marker = {"complete": True, "training_complete": True, "test_complete": True,
                    "best_step": 4000, "dev_episode_mean_mm": 9., "train_steps": 4000,
                    "first_batch_row_ids": [row % 28 for row in range(32)], "checkpoint_sha256": checkpoint_sha,
                    "initialization_sha256": initial, "row_stream_sha256": stream,
                    **{key: summary.base.sha(directory / filename) for key, filename in (
                        ("config_sha256", "config.json"), ("metrics_sha256", "metrics.json"), ("predictions_sha256", "test_predictions.npz"))}}
                dump(directory / "completion.json", marker)
                if objective == "center":
                    integrity_rows.append({"seed": seed, "mode": mode, "matched": True, "both_training_complete": True,
                        "train_steps": 4000, "initialization_sha256": initial, "row_stream_sha256": stream})
    dump(source / "paired_training_integrity.json", {"complete": True, "pairs": integrity_rows})
    return SimpleNamespace(input=source, center_root=None, delta_root=None, output=tmp_path / "report",
                           expected_steps=4000, plots=False)


def test_all_twelve_independently_audited_runs_are_complete_without_primary_twenty_four(complete_toy):
    result = summary.summarize(complete_toy)
    assert result["complete"] and result["comparability"]["complete"]
    assert result["expected_runs"] == result["completed_runs"] == 12
    comparison = result["paired_comparisons"]["delta_minus_center"]["internal"]["near"]["displacement_error_mm"]
    assert comparison["complete"]
    assert comparison["across_seed_descriptive"]["conditional_episode_ci95"] == [0., 0.]
    assert comparison["across_seed_descriptive"]["episodes"] == 2
    assert len(result["runs"]) == 12 and result["descriptive"]["center"]["visual"]["near"]["center_mm"]["mean"] == pytest.approx(np.sqrt(1400))
    assert (complete_toy.output / "results.csv").is_file()
    assert json.loads((complete_toy.output / "results.json").read_text())["complete"]


def test_last_worker_subset_summaries_do_not_override_twelve_verified_artifacts(complete_toy):
    for objective in summary.OBJECTIVES:
        root = complete_toy.input / ("paired_" + objective)
        worker_summary = {"complete": True, "full_design_complete": False,
            "requested": {"seeds": [45], "modes": list(summary.MODES), "objective": objective}, "runs": [{"seed": 45, "complete": True}]}
        dump(root / "summary.json", worker_summary)
        for seed in summary.SEEDS:
            dump(root / f"summary_seeds_{seed}_modes_visual_internal.json", {**worker_summary,
                "requested": {**worker_summary["requested"], "seeds": [seed]}})
    dump(complete_toy.input / "paired_training_integrity.json", {"complete": False, "pairs": []})
    result = summary.summarize(complete_toy)
    assert result["complete"] and result["completed_runs"] == result["expected_runs"] == 12
    assert result["problems"] == [] and "cached_runner_integrity_missing_or_partial" in result["provenance_warnings"]
    for objective in summary.OBJECTIVES:
        saved = result["metadata"][objective]
        assert saved["summary"]["requested"]["seeds"] == [45]
        assert len(saved["subset_summaries"]) == 3
        assert not saved["summary"]["full_design_complete"]
    assert all(row["checkpoint_file_sha256"] for row in result["runs"])


@pytest.mark.parametrize("change,expected_problem", [("delete", "missing_best_checkpoint"), ("corrupt", "best_checkpoint_file_hash_mismatch")])
def test_missing_or_changed_actual_best_checkpoint_prevents_completion(complete_toy, change, expected_problem):
    checkpoint = complete_toy.input / "paired_delta/s45/augmented/internal/best.pt"
    if change == "delete":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"changed after evaluation")
    result = summary.summarize(complete_toy)
    assert not result["complete"] and result["completed_runs"] == 11
    row = [row for row in result["runs"] if (row["objective"], row["seed"], row["mode"]) == ("delta", 45, "internal")][0]
    assert expected_problem in row["problems"]


def test_frozen_followup_cannot_be_reported_as_complete_at_a_different_budget(complete_toy):
    complete_toy.expected_steps = 2000
    result = summary.summarize(complete_toy)
    assert not result["complete"] and "followup_requires_predeclared_4000_step_budget" in result["problems"]


def test_actual_row_stream_and_initialization_mismatch_invalidate_objective_control(complete_toy):
    result = summary.summarize(complete_toy)
    runs = {(row["seed"], row["objective"], row["mode"]): deepcopy(row) for row in result["runs"]}
    runs[(43, "delta", "internal")]["training_integrity"]["row_stream_sha256"] = "d" * 64
    audit = summary.comparability_audit(runs)
    assert not audit["complete"]
    assert "objective_control_mismatch:row_stream_sha256" in audit["pairs"][1]["problems"]
    assert "training_inputs_differ_between_modes_or_objectives:seed43" in audit["problems"]
    runs[(43, "delta", "internal")]["training_integrity"]["initialization_sha256"] = "e" * 64
    assert "objective_control_mismatch:initialization_sha256" in summary.comparability_audit(runs)["pairs"][1]["problems"]


def test_only_objective_and_weight_may_differ_not_budget_architecture_data_or_new_fields():
    center = {"objective": "center", "displacement_weight": 0., "steps": 4000, "hidden_dim": 128,
              "data_manifest_sha256": "a" * 64, "seed": 43}
    delta = {**center, "objective": "delta", "displacement_weight": 1.}
    assert summary.config_differences(center, delta, summary.OBJECTIVE_CONFIG_KEYS) == []
    for key, value in (("steps", 2000), ("hidden_dim", 64), ("data_manifest_sha256", "b" * 64), ("unexpected_future_field", "anything")):
        assert summary.config_differences(center, {**delta, key: value}, summary.OBJECTIVE_CONFIG_KEYS) == [key]


def test_missing_test_result_never_becomes_zero_or_a_complete_design(complete_toy):
    missing = complete_toy.input / "paired_delta/s45/augmented/internal/test_predictions.npz"
    missing.unlink()
    result = summary.summarize(complete_toy)
    assert not result["complete"] and result["completed_runs"] == 11
    value = result["descriptive"]["delta"]["internal"]["near"]["center_mm"]
    assert not value["complete"] and value["mean"] is None and value["conditional_episode_ci95"] is None
    assert not result["paired_comparisons"]["delta_minus_center"]["internal"]["near"]["center_mm"]["complete"]


def test_incomplete_objective_match_suppresses_paired_inference_even_when_metrics_exist(complete_toy):
    run = complete_toy.input / "paired_delta/s43/augmented/internal"
    config = json.loads((run / "config.json").read_text())
    config["learning_rate"] = 2e-4
    dump(run / "config.json", config)
    marker = json.loads((run / "completion.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    marker["config_sha256"] = metrics["config_sha256"] = summary.base.sha(run / "config.json")
    dump(run / "metrics.json", metrics)
    marker["metrics_sha256"] = summary.base.sha(run / "metrics.json")
    dump(run / "completion.json", marker)
    result = summary.summarize(complete_toy)
    assert result["completed_runs"] == 12 and not result["complete"] and not result["comparability"]["complete"]
    contrast = result["paired_comparisons"]["delta_minus_center"]["internal"]["near"]["center_mm"]
    assert not contrast["complete"] and contrast["across_seed_descriptive"]["mean"] is None


def rebind_config_and_metrics(run):
    marker = json.loads((run / "completion.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    marker["config_sha256"] = metrics["config_sha256"] = summary.base.sha(run / "config.json")
    dump(run / "metrics.json", metrics)
    marker["metrics_sha256"] = summary.base.sha(run / "metrics.json")
    dump(run / "completion.json", marker)


def test_worker_placement_only_allowed_after_record_read_and_software_gpu_verification(complete_toy):
    root = complete_toy.input / "paired_delta"
    run = root / "s43/augmented/internal"
    environment = json.loads((root / "environment.json").read_text())
    environment.update(cuda_visible_devices="2", python_executable="/other/account/python", device="cpu")
    record = root / "environments/worker.json"
    dump(record, environment)
    config = json.loads((run / "config.json").read_text())
    config.update(environment_record="environments/worker.json", environment_sha256=summary.base.sha(record))
    dump(run / "config.json", config)
    rebind_config_and_metrics(run)
    result = summary.summarize(complete_toy)
    assert result["complete"]
    pair = result["comparability"]["pairs"][1]
    assert pair["complete"] and pair["environment_placement_differences"] == ["cuda_visible_devices", "python_executable"]
    # Same pointers/hashes do not excuse a different numerical software stack.
    environment["torch"] = "2.7.1"
    dump(record, environment)
    config["environment_sha256"] = summary.base.sha(record)
    dump(run / "config.json", config)
    rebind_config_and_metrics(run)
    result = summary.summarize(complete_toy)
    assert not result["complete"] and not result["comparability"]["complete"]
    assert "objective_control_software_os_or_gpu_mismatch" in result["comparability"]["pairs"][1]["problems"]


def test_environment_hash_and_root_containment_are_checked_not_just_config_ignored(complete_toy):
    root = complete_toy.input / "paired_delta"
    run = root / "s43/augmented/internal"
    config = json.loads((run / "config.json").read_text())
    config["environment_record"] = "../paired_center/environment.json"
    dump(run / "config.json", config)
    rebind_config_and_metrics(run)
    result = summary.summarize(complete_toy)
    matched = [row for row in result["runs"] if row["objective"] == "delta" and row["seed"] == 43 and row["mode"] == "internal"][0]
    assert "environment_record_escapes_objective_root" in matched["problems"]
    assert not result["complete"] and result["completed_runs"] == 11


def test_empty_design_is_twelve_missing_runs_and_finite_json(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    result = summary.summarize(SimpleNamespace(input=source, center_root=None, delta_root=None,
        output=tmp_path / "report", expected_steps=4000, plots=True))
    assert not result["complete"] and result["expected_runs"] == 12 and result["completed_runs"] == 0
    assert len(result["runs"]) == 12 and result["plots"] == {}
    assert "NaN" not in (tmp_path / "report/results.json").read_text()
