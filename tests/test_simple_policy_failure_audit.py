"""CPU fixtures for mutually exclusive stages, exact contacts and partial plans."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "simple_policy_failure_audit", Path(__file__).resolve().parents[1] / "scripts/audit_simple_policy_failures.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def trajectory_case(*, contact=False, lift=False, drop=False, success=False):
    centers = np.repeat([[.44, -.04, .035]], 600, axis=0)
    lifted = np.zeros(600, dtype=bool)
    inside = np.zeros(600, dtype=bool)
    contacts = np.zeros(600, dtype=np.int32)
    if contact:
        contacts[37:50] = 2
    if lift:
        lifted[99:] = True
        centers[99:, 2] = .12
    if drop:
        centers[199:, 2] = .035
    if success:
        centers[-100:] = [.5, -.3, .04]
        inside[-100:] = True
    initial = {"cube_center": [.44, -.04, .035], "lifted": False,
               "inside_goal": False, "robot_cube_contact_count": 0}
    final = {"cube_center": centers[-1].tolist(), "inside_goal": success, "strict_stable_goal": success,
             "cumulative_clipped_count": 123, "first_goal_tick": 501 if success else None,
             "stable_goal_ticks": 100 if success else 0}
    result = {"complete": True, "initial_info": initial, "final_info": final}
    result["outcome"] = audit.score_trajectory(initial, final, centers, lifted, inside)
    trajectory = {"cube_centers": centers, "lifted": lifted, "inside_goal": inside,
                  "robot_cube_contact_count": contacts, "refresh_ticks": np.arange(0, 600, 8),
                  # Always-active proxy deliberately does not establish object contact.
                  "raw_pressure": np.ones((75, 44), dtype=np.float32)}
    return result, trajectory


@pytest.mark.parametrize("settings,expected", [
    ({"success": True}, "success"),
    ({}, "no_object_contact"),
    ({"contact": True}, "contact_without_lift"),
    ({"contact": True, "lift": True, "drop": True}, "lift_then_drop_outside_goal"),
    ({"contact": True, "lift": True}, "lift_without_drop_not_stable"),
])
def test_mutually_exclusive_scored_stages_and_proxy_is_not_contact(settings, expected):
    result, trajectory = trajectory_case(**settings)
    row = audit.classify_case(result, trajectory)
    assert row["stage"] == expected
    assert row["pressure_proxy"]["ever_active"]
    assert row["cumulative_clipped_count"] == 123  # Report only, no causal label.
    if expected == "contact_without_lift":
        assert row["object_contact_scoring_only"]["first_contact_tick"] == 38
    if expected == "lift_then_drop_outside_goal":
        assert row["first_drop_tick"] == 200
        assert row["actual_height_at_least_initial_plus_5cm_ticks"] == 100
    if expected == "lift_without_drop_not_stable":
        assert row["first_lift_tick"] == 100


def test_missing_old_contact_field_is_unknown_and_outcome_mismatch_is_rejected():
    result, trajectory = trajectory_case()
    trajectory.pop("robot_cube_contact_count")
    row = audit.classify_case(result, trajectory)
    assert row["stage"] == "unknown" and row["object_contact_scoring_only"]["ever_contact"] is None
    assert row["pressure_proxy"]["ever_active"]
    result["outcome"]["ever_lifted"] = True
    with pytest.raises(ValueError, match="ever_lifted"):
        audit.classify_case(result, trajectory)


def write_evaluation(root, *, case_set="nominal", exact_contacts=True):
    entry = {"kind": "simple", "mode": "frozen_zero", "n_action_steps": 8,
             "case_set": case_set, "directory": "eval_n8/frozen_zero"}
    save(root / "experiment_manifest.json", {"seed": 43, "expected_evaluations": [entry]})
    directory = root / entry["directory"]
    config = {"kind": "simple", "case_set": case_set, "n_action_steps_override": 8,
              "control_ticks": 600, "physics_dt": .01, "requested_modes": ["frozen_zero"]}
    save(directory / "eval_config.json", config)
    names = sorted(audit.declared_ids(case_set))
    save(directory / "case_plan.json", {"declared_case_ids": names, "selected_case_ids": names, "case_limit": None})
    result, trajectory = trajectory_case(contact=True)
    result.update(mode="frozen_zero", config_sha256=audit.digest(directory / "eval_config.json"),
                  ticks=600, sim_time_s=6., case={"case_id": "nominal", "episode_id": 1,
                  "frame": 0, "seed": 43, "object_xy_offset": [0., 0.]})
    case_dir = directory / "frozen_zero/nominal"
    save(case_dir / "result.json", result)
    if not exact_contacts:
        trajectory.pop("robot_cube_contact_count")
    np.savez_compressed(case_dir / "trajectory.npz", **trajectory)
    return directory, case_dir


def test_manifest_plan_controls_completeness_and_nominal_never_fills_missing_grid(tmp_path):
    write_evaluation(tmp_path, case_set="support")
    result = audit.audit(tmp_path)
    assert not result["complete"] and result["planned_cases"] == 13 and result["completed_cases"] == 1
    value = result["evaluations"][0]
    assert value["subsets"]["nominal"]["complete"]
    assert value["subsets"]["nominal"]["stage_counts"]["contact_without_lift"] == 1
    assert value["subsets"]["within_bbox12"]["completed_cases"] == 0
    assert not value["subsets"]["within_bbox12"]["complete"]
    assert all(count is None for count in value["subsets"]["within_bbox12"]["stage_counts"].values())
    assert sum(row["status"] == "missing" for row in value["cases"]) == 12
    assert value["subsets"]["outside24"]["planned_cases"] == 0


def test_complete_nominal_and_unknown_contact_coverage_are_separate(tmp_path):
    directory, case_dir = write_evaluation(tmp_path, exact_contacts=False)
    result = audit.audit(tmp_path)
    assert result["complete"] and not result["contact_attribution_complete"]
    assert result["unknown_stage_cases"] == 1
    row = result["evaluations"][0]["cases"][0]
    assert row["stage"] == "unknown" and row["status"] == "complete"
    assert str(case_dir / "trajectory.npz") in result["sources"]
    record = json.loads((case_dir / "result.json").read_text())
    record["config_sha256"] = "another protocol"
    save(case_dir / "result.json", record)
    invalid = audit.audit(tmp_path)
    assert not invalid["complete"] and invalid["completed_cases"] == 0
    assert invalid["evaluations"][0]["cases"][0]["status"] == "invalid"
