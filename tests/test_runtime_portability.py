"""Relocated artifacts and resolved machine paths retain the same protocols."""
from argparse import Namespace
import json

import evaluate_sim
import run_simple_policy_followup
import pytest
from tactile_vggt.execution import child_environment, selected_gpus
from tactile_vggt import paths


def test_child_gpu_selection_preserves_scheduler_and_egl_mapping(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-A,GPU-allocated-B")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "7")
    assert selected_gpus("GPU-allocated-B,GPU-allocated-A") == ["GPU-allocated-B", "GPU-allocated-A"]
    environment = child_environment("GPU-allocated-B", render=True)
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-B"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "7"
    with pytest.raises(ValueError, match="Slurm"):
        child_environment("0", render=True)
    with pytest.raises(ValueError, match="distinct"):
        selected_gpus(None)


def test_worker_config_roundtrip_is_stable_for_resume(tmp_path):
    args = Namespace(**{name: tmp_path / name for name in
                        ("worker_python", "dynamic_root", "raw_root", "timestamps_root", "source_readouts")})
    config = {"worker": evaluate_sim.worker_configuration(args)}
    target = tmp_path / "eval_config.json"
    evaluate_sim.write_json(target, config)
    assert json.loads(target.read_text()) == config
    assert config["worker"]["python"] == str(args.worker_python)


def test_historical_simple_manifest_builds_new_paths_without_mutating_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("TACTILE_VGGT_ARTIFACT_ROOT", str(tmp_path / ".local"))
    manifest = {"cache": "cache/observations", "features": "cache/features", "visual_probe": "outputs/probe/best.pt",
                "training_runs": [{"mode": "frozen_zero", "directory": "train/frozen_zero"}],
                "training_steps": 6000, "seed": 43, "validation_every": 1000}
    original = json.dumps(manifest, sort_keys=True)
    args = Namespace(phase="train", output=tmp_path / ".local/runs/simple", resume=False)
    _, command = list(run_simple_policy_followup.commands(args, manifest))[0]
    for flag, expected in (("--cache", ".local/cache/observations"), ("--features-dir", ".local/cache/features"),
                           ("--visual-probe", ".local/runs/probe/best.pt")):
        assert command[command.index(flag) + 1] == str(tmp_path / expected)
    assert json.dumps(manifest, sort_keys=True) == original
