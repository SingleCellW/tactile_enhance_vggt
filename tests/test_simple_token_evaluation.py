"""CPU-only case protocol, provenance, queues, GT isolation and resume checks."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "simple_token_evaluation", Path(__file__).resolve().parents[1] / "scripts/evaluate_simple_token_act.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_declared37_support13_and_bbox_are_not_full_id():
    full, declared, selected = evaluation.make_cases(case_set="all")
    assert len(full) == len(declared) == len(selected) == 37
    assert full[0]["case_id"] == "nominal"
    assert full[0]["initial_xy"] == [.44, -.04]
    assert full[0]["initial_xy_within_train_bbox"]
    assert not full[0]["subset_flags"]["within_bbox12"]  # nominal stays separate
    assert {name: sum(case["subset_flags"][name] for case in full) for name in evaluation.SUBSETS} == {
        "nominal": 1, "grid36": 36, "within_bbox12": 12, "outside24": 24}
    _, support, _ = evaluation.make_cases(case_set="support")
    assert len(support) == 13 and support[0]["case_id"] == "nominal"
    assert all(case["subset_flags"]["nominal"] or case["subset_flags"]["within_bbox12"] for case in support)
    grid = {case["case_id"]: case for case in full if case["case_id"] != "nominal"}
    assert np.allclose(grid["grid_05"]["initial_xy"], [.405, -.005])
    assert np.allclose(grid["grid_30"]["initial_xy"], [.475, -.075])
    assert not grid["grid_30"]["initial_xy_within_train_bbox"]
    assert grid["grid_35"]["initial_xy_within_train_bbox"]
    assert [case["case_id"] for case in declared[1:5]] == ["grid_00", "grid_05", "grid_30", "grid_35"]
    _, _, smoke = evaluation.make_cases(case_set="support", case_limit=2)
    assert smoke == support[:2]
    with pytest.raises(ValueError):
        evaluation.make_cases(case_set="support", case_limit=14)
    with pytest.raises(ValueError):
        evaluation.make_cases(episode=24)


def test_raw_pressure_and_post_action_contacts_keep_distinct_ticks():
    pressure = np.zeros(44)
    pressure[3] = .02
    diagnostic = evaluation.pressure_diagnostics(pressure, 8)
    assert diagnostic["observation_tick"] == 8
    assert diagnostic["raw_pressure_active_sensors"] == 1
    assert diagnostic["raw_pressure_contact_flag"]
    counts = np.zeros(600, dtype=int)
    counts[8:11] = 2
    contact = evaluation.contact_summary(0, counts)
    assert contact["first_post_action_contact_tick"] == 9
    assert contact["post_action_contact_ticks"] == 3
    assert evaluation.contact_summary(1, counts)["first_contact_tick"] == 0
    with pytest.raises(ValueError):
        evaluation.pressure_diagnostics(np.full(44, -1.), 0)
    with pytest.raises(ValueError):
        evaluation.object_contact_count({})


def source_metadata(normalization):
    return {"complete": True, "normalization": normalization,
            "native_tactile_report": {"velocity_dt": .01, "geometry_sha256": "geometry", "native_models_sha256": "native", "object_mesh_sha256": "mesh", "source_readouts_sha256": "source"},
            "configuration": {"rgb_alignment": {"verified": True, "rgb_lag_frames": 1}},
            "action_semantics": "Original source action labels unchanged, including their historical offset.",
            "split": {"train": [0], "val": [1]}}


def worker_provenance():
    return {"mujoco_version": "3.4.0", "physics_dt": .01, "control_dt": .01, "substeps": 1, "sensor_dt": .01,
            "geometry_sha256": "geometry", "native_models_sha256": "native", "object_mesh_sha256": "mesh", "source_readouts_sha256": "source"}


def test_cache_preserves_recorded_action_semantics_and_checks_worker_hashes():
    metadata = source_metadata({})
    assert "action_label_repair" not in metadata["configuration"]
    evaluation.validate_cache(metadata)
    evaluation.validate_worker(worker_provenance(), metadata)
    missing = copy.deepcopy(metadata)
    missing.pop("action_semantics")
    with pytest.raises(ValueError, match="action_semantics"):
        evaluation.validate_cache(missing)
    changed = worker_provenance()
    changed["source_readouts_sha256"] = "another source"
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.validate_worker(changed, metadata)
    changed = worker_provenance()
    changed["sensor_dt"] = .05
    with pytest.raises(ValueError, match="sensor_dt"):
        evaluation.validate_worker(changed, metadata)


def test_subgroup_summary_does_not_mix_nominal_into_grid_rate():
    _, nominal, _ = evaluation.make_cases(case_set="nominal")
    row = {"case": nominal[0], "outcome": {"strict_stable_goal": True, "ever_lifted": True,
        "lift_then_drop_outside_goal": False, "final_goal_center_xy_distance_m": 0., "final_goal_region_xy_distance_m": 0.},
        "wall_seconds": 1., "encoder_refresh": {"mean_s": .1}}
    model = evaluation.summarize_mode([row], nominal, "nominal")
    assert model["complete"] and model["primary"]["strict_stable_goal_rate"] == 1
    assert "strict_stable_goal_rate" not in model
    assert model["subsets"]["grid36"]["cases"] == 0
    assert model["subsets"]["grid36"]["strict_stable_goal_rate"] is None
    assert model["subsets"]["grid36"]["selected_complete"]


def policy_fixture(tmp_path, modes=("frozen_zero", "frozen_pressure")):
    cache = tmp_path / "cache"
    normalization = {"state_mean": [0]*22, "state_std": [1]*22, "action_mean": [0]*22, "action_std": [1]*22}
    metadata = source_metadata(normalization)
    save(cache / "manifest.json", metadata)
    np.save(cache / "obs_frame_indices.npy", np.arange(4))
    representation = tmp_path / "representation.json"
    save(representation, {"complete": True, "split_episodes": {"train": [0], "dev": [2], "test": [1]}})
    probe = tmp_path / "probe" / "visual" / "best.pt"
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"visual probe fixture")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"VGGT weights fixture")
    policies = {}
    for mode in modes:
        checkpoint = tmp_path / "policies" / mode
        policies[mode] = checkpoint
        projection, tactile = evaluation.SIMPLE_MODES[mode]
        structure = {"projection": projection, "tactile": tactile, "readout_input_dim": 1088, "visual_dim": 256, "tactile_dim": 44}
        save(checkpoint / "checkpoint_complete.json", {"step": 1000, "structure": structure, "policy_type": "simple_token_act"})
        save(checkpoint / "config.json", {"config": {"encoder_dim": 300, "chunk_size": 16, "n_action_steps": 8, "temporal_ensemble_coeff": None}, "policy_type": "simple_token_act", "structure": structure})
        save(checkpoint / "normalization.json", {**normalization, "simple_pressure_scale": 1.})
        save(checkpoint / "feature_manifest.json", {"complete": True, "feature_dim": 1088,
            "cache_manifest_sha256": evaluation.digest(cache / "manifest.json"),
            "obs_frame_indices_sha256": evaluation.digest(cache / "obs_frame_indices.npy"), "source_checkpoint_sha256": evaluation.digest(probe)})
        save(checkpoint / "train_config.json", {"data_manifest_sha256": evaluation.digest(cache / "manifest.json"),
            "visual_probe_checkpoint_sha256": evaluation.digest(probe), "action_semantics": metadata["action_semantics"]})
        (checkpoint / "model.pt").write_bytes(b"ACT weights fixture")
    cli = ["--cache", str(cache), "--output", str(tmp_path / "evaluation"), "--device", "cpu", "--amp", "float32",
           "--weights", str(weights), "--case-set", "nominal", "--n-action-steps", "2",
           "--visual-probe-checkpoint", str(probe), "--representation-manifest", str(representation)]
    for mode, checkpoint in policies.items():
        cli.extend(["--policy-checkpoint", f"{mode}={checkpoint}"])
    return evaluation.parser().parse_args(cli), metadata, policies, probe


def test_simple_policy_source_binding_rejects_wrong_probe_and_normalization(tmp_path):
    args, metadata, policies, probe = policy_fixture(tmp_path)
    checkpoint = policies["frozen_zero"]
    kwargs = ("frozen_zero", checkpoint, probe, evaluation.digest(args.cache / "manifest.json"), metadata["normalization"],
              evaluation.digest(args.cache / "obs_frame_indices.npy"), metadata["action_semantics"])
    assert evaluation.validate_simple_policy(*kwargs)["n_action_steps"] == 8
    feature = json.loads((checkpoint / "feature_manifest.json").read_text())
    feature["source_checkpoint_sha256"] = "wrong probe"
    save(checkpoint / "feature_manifest.json", feature)
    with pytest.raises(ValueError, match="binding mismatch"):
        evaluation.validate_simple_policy(*kwargs)


def test_nominal_protocol_fake_worker_is_gt_isolated_runs600_and_resumes(tmp_path, monkeypatch):
    args, _, policies, _ = policy_fixture(tmp_path)
    resets, steps, encodes = [], [], []
    raw_pressure = np.zeros(44, np.float32)
    raw_pressure[0] = .2
    observation = {"state": np.zeros(22), "tactile": raw_pressure, "rgb": np.zeros((2,224,224,3), np.uint8),
        "anchors_world": np.zeros((44,3)), "known_cameras_cv": np.repeat(np.eye(4)[None],2,0), "base_to_world": np.eye(4),
        "cube_center": np.array([.5,-.3,.04]), "segmentation": np.ones((2,224,224))}
    class FakeClient:
        def __init__(self, *unused):
            self.tick = 0
        def request(self, command, **values):
            if command == "reset":
                self.tick = 0
                resets.append(values)
            elif command == "step":
                self.tick += 1
                steps.append(values)
            assert command in {"reset","step","observe"}
            info = {"cube_center": [.44,-.04,.035] if self.tick == 0 else [.5,-.3,.04],
                    "lifted": False, "inside_goal": self.tick > 0, "strict_stable_goal": self.tick >= 100,
                    "robot_cube_contact_count": 1 if 9 <= self.tick <= 11 else 0}
            return {"ok": True,"info":info,"sim_time": self.tick*.01,"provenance":worker_provenance()}
        def close(self):
            pass
    class FakeEncoder:
        def encode_observation(self, obs):
            assert set(obs) == set(evaluation.OBSERVATION_KEYS)
            assert "cube_center" not in obs and "segmentation" not in obs and "info" not in obs
            encodes.append(1)
            return np.zeros(1088, np.float32)
    class FakePolicy:
        def __init__(self):
            self.policy = SimpleNamespace(config={"n_action_steps": 2})
        def reset(self):
            self.calls = 0
        @property
        def needs_observation(self):
            return self.calls % 2 == 0
        def select_action(self, state=None, readout=None, pressure=None):
            if self.needs_observation:
                assert state.shape == (22,) and readout.shape == (1088,)
                np.testing.assert_array_equal(pressure, raw_pressure)  # zero-mode masking belongs inside policy
            else:
                assert state is readout is pressure is None
            self.calls += 1
            return np.zeros(22)
    monkeypatch.setattr(evaluation, "SimulationClient", FakeClient)
    monkeypatch.setattr(evaluation, "read_observation", lambda reply: observation)
    monkeypatch.setattr(evaluation, "create_runtime", lambda *unused: FakeEncoder())
    monkeypatch.setattr(evaluation, "create_policy", lambda *unused: FakePolicy())
    monkeypatch.setattr(evaluation, "implementation_hashes", lambda kind: {"fake": "fixed"})
    evaluation.run(args)
    assert len(resets) == 2 and resets[0] == resets[1]
    assert len(steps) == 1200 and len(encodes) == 600
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["complete"] and summary["selected_cases_complete"]
    assert not summary["full_grid_complete"] and not summary["full37_complete"]
    assert summary["case_set"] == summary["primary_subset"] == "nominal"
    assert summary["models"]["frozen_zero"]["primary"]["strict_stable_goal_rate"] == 1
    assert summary["paired_comparisons"]["nominal"]["frozen_pressure_minus_frozen_zero"]["mean_difference"] == 0
    record = summary["models"]["frozen_zero"]["results"][0]
    assert record["ticks"] == 600
    assert record["object_contact_scoring_only"]["first_post_action_contact_tick"] == 9
    assert record["object_contact_scoring_only"]["post_action_contact_ticks"] == 3
    with np.load(args.output / "frozen_zero/nominal/trajectory.npz") as trajectory:
        assert trajectory["robot_cube_contact_count"].shape == (600,)
        assert trajectory["refresh_ticks"][0] == 0
        assert trajectory["raw_pressure"].shape == (300,44)
    args.resume = True
    evaluation.run(args)
    assert len(resets) == 2 and len(steps) == 1200
    args.n_action_steps = 8
    with pytest.raises(ValueError, match="protocol changed"):
        evaluation.run(args)
    args.n_action_steps, args.resume, args.case_set, args.case_limit = 2, False, "all", 1
    args.output = tmp_path / "smoke"
    evaluation.run(args)
    smoke = json.loads((args.output / "summary.json").read_text())
    assert not smoke["complete"] and smoke["selected_cases_complete"]


def test_legacy_queue_override_resets_actual_config_without_changing_chunk(monkeypatch):
    import sys
    from collections import deque
    class FakeLegacy:
        def __init__(self, *unused, **kwargs):
            self.policy = SimpleNamespace(config={"chunk_size":16,"n_action_steps":8},
                act=SimpleNamespace(config=SimpleNamespace(n_action_steps=8),_action_queue=deque([1,2],maxlen=8)))
        def reset(self):
            self.policy.act._action_queue = deque(maxlen=self.policy.act.config.n_action_steps)
    monkeypatch.setitem(sys.modules, "tactile_vggt.token_policy", SimpleNamespace(TokenACTInference=FakeLegacy))
    args = SimpleNamespace(kind="legacy",device="cpu",amp="float32",n_action_steps=2)
    policy = evaluation.create_policy(args, "unused")
    assert policy.policy.config["chunk_size"] == 16
    assert policy.policy.config["n_action_steps"] == policy.policy.act.config.n_action_steps == 2
    assert policy.policy.act._action_queue.maxlen == 2 and not policy.policy.act._action_queue
