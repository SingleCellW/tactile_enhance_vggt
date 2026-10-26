import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "token_act_evaluation", Path(__file__).resolve().parents[1] / "scripts/evaluate_token_act.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_fixed36_grid_and_smoke_cases_never_depend_on_outcomes():
    full, smoke = evaluation.make_cases(1, 43, 4)
    assert len(full) == 36 and len(smoke) == 4
    assert len({case["case_id"] for case in full}) == 36
    assert {tuple(case["object_xy_offset"]) for case in smoke} == {
        (-.035, -.035), (-.035, .035), (.035, -.035), (.035, .035)}
    assert full == evaluation.make_cases(1, 43)[0]
    assert all(case["episode_id"] == 1 and case["frame"] == 0 for case in full)
    assert [case["seed"] for case in full] == [43 + int(case["case_id"].split("_")[-1]) for case in full]
    json.dumps(full, allow_nan=False)
    with pytest.raises(ValueError):
        evaluation.make_cases(case_limit=37)


def test_select_jointly_heldout_source_from_actual_manifest():
    source = {"split": {"train": [0, 2], "val": [1, 24, 35]}}
    rep = {"split_episodes": {"train": [0, 24, 35], "dev": [2], "test": [1]}}
    episode, roles = evaluation.choose_source(source, rep)
    assert episode == 1
    assert roles["representation_role"] == "test"
    assert roles["representation_test_and_act_validation_intersection"] == [1]
    episode, roles = evaluation.choose_source(source, rep, requested=24)
    assert episode == 24 and roles["representation_role"] == "train"
    with pytest.raises(ValueError, match="held out"):
        evaluation.choose_source(source, rep, requested=0)


def test_drop_requires_previous_lift_and_excludes_goal_placement():
    initial = {"cube_center": [.4, 0, .04], "lifted": False}
    final = {"strict_stable_goal": True, "inside_goal": True}
    placed = evaluation.score_trajectory(initial, final, [[.4, 0, .11], [.5, -.3, .04]],
                                         [True, True], [False, True])
    assert placed["strict_stable_goal"] and placed["ever_lifted"]
    assert not placed["lift_then_drop_outside_goal"]
    assert placed["final_goal_center_xy_distance_m"] == 0
    assert placed["final_goal_region_xy_distance_m"] == 0
    dropped = evaluation.score_trajectory(initial, {**final, "inside_goal": False},
                [[.4, 0, .04], [.4, 0, .11], [.4, 0, .03]], [False, True, True], [False]*3)
    assert dropped["lift_then_drop_outside_goal"] and dropped["first_drop_tick"] == 3
    assert dropped["final_goal_region_xy_distance_m"] == pytest.approx(.21)
    never_lifted = evaluation.score_trajectory(initial, {**final, "strict_stable_goal": False},
                         [[.4, 0, .01]], [False], [False])
    assert not never_lifted["lift_then_drop_outside_goal"]


def test_wilson_and_paired_success_intervals_use_matched_grid_cases():
    assert evaluation.wilson_interval(0, 0) is None
    assert evaluation.wilson_interval(0, 36)[0] == 0
    assert evaluation.wilson_interval(36, 36)[1] == 1
    def rows(outcomes):
        return {"results": [{"case": {"case_id": f"grid_{index:02d}"},
                             "outcome": {"strict_stable_goal": value}}
                            for index, value in enumerate(outcomes)]}
    same = evaluation.paired_comparisons({"internal": rows([True, False, True]), "visual": rows([True, False, True])})
    assert same["internal_minus_visual"]["ci95"] == [0., 0.]
    assert same["internal_minus_visual"]["group_count"] == 3
    better = evaluation.paired_comparisons({"internal": rows([True]*4), "visual": rows([False]*4)})
    assert better["internal_minus_visual"]["mean_difference"] == 1
    assert better["internal_minus_visual"]["ci95"] == [1., 1.]
    assert "higher strict success" in better["internal_minus_visual"]["direction"]
    json.dumps(better, allow_nan=False)


def test_full_protocol_with_fake_worker_keeps_cases_paired_and_resumes(tmp_path, monkeypatch):
    """Exercise orchestration without VGGT weights, native sensors, or a GPU."""
    torch = pytest.importorskip("torch")
    import token_runtime
    from tactile_vggt import token_policy

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    cache = tmp_path / "cache"
    normalization = {"state_mean": [0]*22, "state_std": [1]*22,
                     "action_mean": [0]*22, "action_std": [1]*22}
    save(cache / "manifest.json", {"complete": True, "native_tactile_report": {"velocity_dt": .01},
                                   "normalization": normalization, "split": {"train": [0], "val": [1]}})
    representation = tmp_path / "representation.json"
    save(representation, {"complete": True, "split_episodes": {"train": [0], "dev": [2], "test": [1]}})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fake weights for orchestration test")
    checkpoints, probes = {}, {}
    for mode in ("internal", "visual"):
        probes[mode] = tmp_path / "probes" / mode / "best.pt"
        probes[mode].parent.mkdir(parents=True)
        probes[mode].write_bytes(mode.encode())
        checkpoint = tmp_path / "policies" / mode
        checkpoints[mode] = checkpoint
        save(checkpoint / "checkpoint_complete.json", {"step": 1})
        save(checkpoint / "config.json", {"config": {"chunk_size": 4, "n_action_steps": 2,
                                                     "temporal_ensemble_coeff": None}})
        save(checkpoint / "normalization.json", normalization)
        save(checkpoint / "train_config.json", {"data_manifest_sha256": evaluation.digest(cache / "manifest.json")})
        save(checkpoint / "feature_manifest.json", {"complete": True, "mode": mode,
              "cache_manifest_sha256": evaluation.digest(cache / "manifest.json"),
              "source_checkpoint_sha256": evaluation.digest(probes[mode])})
        (checkpoint / "model.pt").write_bytes(b"fake ACT state")

    resets, steps, encoder_calls = [], [], []
    class FakeClient:
        def __init__(self, *args):
            self.tick = 0
        def request(self, command, **values):
            if command == "reset":
                self.tick = 0
                resets.append(values)
            elif command == "step":
                self.tick += 1
                steps.append(values)
            assert command in ("reset", "step", "observe")
            # Expose truth only in the privileged info response.
            info = {"cube_center": [.5, -.3, .04], "lifted": False, "inside_goal": True,
                    "strict_stable_goal": self.tick >= 100}
            return {"ok": True, "info": info, "sim_time": self.tick * .01,
                    "provenance": {"test": "fake"}}
        def close(self):
            pass
    observation = {"state": np.zeros(22), "tactile": np.zeros(44), "rgb": np.zeros((2, 224, 224, 3), np.uint8),
                   "anchors_world": np.zeros((44, 3)), "known_cameras_cv": np.repeat(np.eye(4)[None], 2, 0),
                   "base_to_world": np.eye(4)}
    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode_observation(self, obs):
            assert set(obs) == set(observation)
            assert "info" not in obs and "cube_center" not in obs
            encoder_calls.append(1)
            return {mode: np.zeros(256, np.float32) for mode in checkpoints}
    class FakePolicy:
        def __init__(self, *args, **kwargs):
            self.policy = type("Configuration", (), {"config": {"n_action_steps": 2}})()
        def reset(self):
            self.calls = 0
        @property
        def needs_observation(self):
            return self.calls % 2 == 0
        def select_action(self, state=None, features=None):
            if self.needs_observation:
                assert state.shape == (22,) and features.shape == (256,)
            else:
                assert state is None and features is None
            self.calls += 1
            return torch.zeros(22)
    monkeypatch.setattr(evaluation, "SimulationClient", FakeClient)
    monkeypatch.setattr(evaluation, "read_observation", lambda reply: observation)
    monkeypatch.setattr(token_runtime, "LiveTokenEncoder", FakeEncoder)
    monkeypatch.setattr(token_policy, "TokenACTInference", FakePolicy)
    cli = ["--cache", str(cache), "--output", str(tmp_path / "evaluation"), "--device", "cpu",
           "--amp", "float32", "--weights", str(weights), "--case-limit", "1",
           "--representation-manifest", str(representation)]
    for mode in checkpoints:
        cli.extend(["--probe-checkpoint", f"{mode}={probes[mode]}",
                    "--policy-checkpoint", f"{mode}={checkpoints[mode]}"])
    args = evaluation.parser().parse_args(cli)
    evaluation.run(args)
    assert resets[0] == resets[1]
    assert len(steps) == 1200  # Success at tick100 must never stop either rollout.
    assert len(encoder_calls) == 600
    result = json.loads((args.output / "summary.json").read_text())
    assert result["complete"]
    assert result["source_split_roles"]["representation_role"] == "test"
    assert result["paired_comparisons"]["internal_minus_visual"]["ci95"] == [0., 0.]
    args.resume = True
    evaluation.run(args)
    assert len(resets) == 2 and len(steps) == 1200  # Completed cases are reused.
