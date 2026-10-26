"""CPU-only selection, strict pairing, primary gate and child cleanup checks."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_token_dev_selected as dev


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def training_run(path):
    save(path / "train_config.json", {"steps": 6000, "eval_every": 1000, "policy": {}, "seed": 43,
                                      "batch_size": 64, "learning_rate": 1e-5, "weight_decay": 1e-4,
                                      "amp": "bfloat16", "data_manifest_sha256": "cache-sha"})
    save(path / "status.json", {"status": "completed", "step": 6000, "total_steps": 6000,
                               "best_val_normalized_action_l1": .2})
    records = [{"step": step, "total_steps": 6000, "val_normalized_action_l1": value}
               for step, value in ((1, None), (1000, .2), (1100, .2), (2000, .2), (3000, .3), (6000, .4))]
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    for step in (1000, 2000, 6000):
        checkpoint = path / "checkpoints" / f"{step:06d}"
        save(checkpoint / "checkpoint_complete.json", {"step": step})
        (checkpoint / "model.pt").write_bytes(f"checkpoint{step}".encode())
    (path / "best").symlink_to("checkpoints/001000", target_is_directory=True)
    return path


def test_selects_first_actual_validation_minimum_and_requires_saved_best(tmp_path):
    run = training_run(tmp_path / "run")
    selected = dev.select_checkpoint(run)
    assert selected["training_budget_steps"] == 6000 and selected["selected_step"] == 1000
    assert selected["selection_value"] == .2 and not selected["uses_test_or_rollout_outcomes"]
    assert selected["evidence_sha256"]["metrics.jsonl"] == dev.digest(run / "metrics.jsonl")
    (run / "best").unlink()
    (run / "best").symlink_to("checkpoints/002000", target_is_directory=True)
    with pytest.raises(ValueError, match="best symlink"):
        dev.select_checkpoint(run)
    (run / "best").unlink()
    (run / "best").symlink_to("checkpoints/001000", target_is_directory=True)
    (run / "checkpoints/001000/checkpoint_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        dev.select_checkpoint(run)


def test_primary_gate_recognizes_actual_stage_schema_and_never_starts_early(tmp_path):
    path = tmp_path / "pipeline_status.json"
    with pytest.raises(ValueError, match="not complete"):
        dev.wait_for_primary(path)
    save(path, {"stage": "closed_loop36", "completed": ["train_act_internal"]})
    with pytest.raises(ValueError, match="not complete"):
        dev.wait_for_primary(path)
    save(path, {"stage": "complete", "completed": ["closed_loop36"]})
    assert dev.wait_for_primary(path)["sha256"] == dev.digest(path)
    save(path, {"stage": "failed", "complete": True})
    with pytest.raises(ValueError, match="failed"):
        dev.wait_for_primary(path)


@pytest.fixture
def child_results(tmp_path):
    directories, selection, probes = {}, {}, {}
    cases, _ = dev.make_cases(1, 43)
    roles = {"act_role": "validation", "representation_role": "test"}
    for mode in dev.MODES:
        selected = dev.select_checkpoint(training_run(tmp_path / "training" / mode))
        selection[mode] = selected
        probe = tmp_path / "probes" / mode / "best.pt"
        probe.parent.mkdir(parents=True)
        probe.write_bytes(mode.encode())
        probes[mode] = probe
        directory = tmp_path / "mode_runs" / mode
        directories[mode] = directory
        plan = {"full_grid_cases": cases, "selected_case_ids": [case["case_id"] for case in cases],
                "case_limit": None, "source_split_roles": roles}
        save(directory / "case_plan.json", plan)
        config = {"cache_manifest_sha256": "cache-sha", "control_ticks": 600, "physics_dt": .01,
                  "substeps": 1, "sensor_velocity_dt": .01, "case_plan": plan,
                  "execution_horizon": {"chunk_size": 16, "n_action_steps": 8, "temporal_ensemble_coeff": None},
                  "modes": {mode: {"policy": selected["checkpoint"], "policy_sha256": selected["model_sha256"],
                                    "probe": str(probe.resolve()), "probe_sha256": dev.digest(probe)}}}
        save(directory / "eval_config.json", config)
        config_hash = dev.digest(directory / "eval_config.json")
        rows = []
        for index, case in enumerate(cases):
            row = {"complete": True, "case": case, "mode": mode, "ticks": 600, "sim_time_s": 6.,
                   "config_sha256": config_hash, "wall_seconds": 1., "encoder_refresh": {"mean_s": .01},
                   "outcome": {"strict_stable_goal": mode == "internal" or mode == "visual" and index % 2 == 0,
                               "ever_lifted": True, "lift_then_drop_outside_goal": False,
                               "final_goal_center_xy_distance_m": .1, "final_goal_region_xy_distance_m": .02}}
            case_dir = directory / mode / case["case_id"]
            save(case_dir / "result.json", row)
            np.savez_compressed(case_dir / "trajectory.npz", actions=np.zeros((600, 22)), cube_centers=np.zeros((600, 3)))
            rows.append(row)
        summary = {"complete": True, "protocol": "new36_xy_grid_live_token_act_v1", "models": {mode: dev.summarize_mode(rows)},
                   "config_sha256": config_hash, "case_plan_sha256": dev.digest(directory / "case_plan.json"),
                   "source_split_roles": roles}
        save(directory / "summary.json", summary)
    return directories, selection, probes, cases


def test_merges_three_full_paired_children_and_keeps_budget_distinct(child_results):
    directories, selection, probes, cases = child_results
    result = dev.merge_children(directories, selection, probes, "cache-sha", cases)
    assert result["complete"] and set(result["models"]) == set(dev.MODES)
    assert result["training_budget_steps"] == 6000
    assert {item["selected_step"] for item in result["selection"].values()} == {1000}
    assert result["paired_comparisons"]["internal_minus_late"]["mean_difference"] == 1.
    assert result["paired_comparisons"]["internal_minus_visual"]["mean_difference"] == .5
    assert "wall" in result["scope"] and "Posthoc" in result["scope"]


@pytest.mark.parametrize("problem", ["partial", "seed", "cache"])
def test_merge_rejects_partial_unpaired_or_wrong_source_children(child_results, problem):
    directories, selection, probes, cases = child_results
    directory = directories["visual"]
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    if problem == "partial":
        summary["models"]["visual"]["results"].pop()
    elif problem == "seed":
        summary["models"]["visual"]["results"][0]["case"]["seed"] += 1
    else:
        path = directory / "eval_config.json"
        summary = json.loads(path.read_text())
        summary["cache_manifest_sha256"] = "wrong-cache"
    save(path, summary)
    with pytest.raises(ValueError):
        dev.merge_children(directories, selection, probes, "cache-sha", cases)


def test_child_failure_terminates_and_reaps_all_owned_process_groups(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    processes, killed = [], []

    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            self.pid, self.returncode, self.waited = 100 + len(processes), None, False
            processes.append(self)

        def poll(self):
            return 1 if self.pid == 101 else None

        def wait(self, **kwargs):
            self.waited = True
            return 0

    monkeypatch.setattr(dev.subprocess, "Popen", Process)
    monkeypatch.setattr(dev.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    args = Namespace(output=tmp_path, resume=False, workers=3, child_timeout=60)
    with pytest.raises(RuntimeError, match="late failed"):
        dev.run_children(args, {mode: ["not-executed"] for mode in dev.MODES})
    assert len(processes) == 3 and all(process.waited for process in processes)
    assert killed == [(100, signal.SIGTERM), (101, signal.SIGTERM), (102, signal.SIGTERM)]
