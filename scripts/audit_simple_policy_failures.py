#!/usr/bin/env python3
"""Read-only failure-stage audit of manifest-declared, completed rollouts.

This consumes result/trajectory files without running a model or simulator.
Contact attribution uses per-tick scoring-only robot/object collision counts;
native pressure remains a separate sensor proxy. Missing cases stay missing.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_sim import digest, write_json
from evaluate_token_act import score_trajectory
from summarize_simple_policy import canonical_cases, declared_ids

STAGES = ("success", "no_object_contact", "contact_without_lift", "lift_then_drop_outside_goal",
          "lift_without_drop_not_stable", "unknown")
SUBSETS = ("nominal", "grid36", "within_bbox12", "outside24")


def nonnegative_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


def exact_object_contact(trajectory, initial_info):
    """A missing old field is unknown, never replaced by sparse trace/pressure."""
    if "robot_cube_contact_count" not in trajectory:
        return {"known": False, "ever_contact": None, "first_contact_tick": None,
                "post_action_contact_ticks": None, "reason": "missing_per_tick_robot_cube_contact_count"}
    counts = np.asarray(trajectory["robot_cube_contact_count"])
    if counts.shape != (600,) or counts.dtype.kind not in "iu" or (counts < 0).any():
        raise ValueError("robot_cube_contact_count must contain 600 nonnegative integer counts")
    initial = initial_info.get("robot_cube_contact_count")
    initial_known = nonnegative_integer(initial)
    active = np.flatnonzero(counts > 0)
    first = 0 if initial_known and initial > 0 else int(active[0] + 1) if len(active) else None
    ever = True if len(active) or initial_known and initial > 0 else False if initial_known else None
    return {"known": initial_known, "post_action_counts_known": True,
            "initial_contact_count": int(initial) if initial_known else None,
            "ever_contact": ever, "first_contact_tick": first,
            "post_action_contact_ticks": int(len(active)), "maximum_post_action_contact_count": int(counts.max()),
            "reason": None if initial_known else "missing_initial_tick0_contact_count"}


def pressure_proxy(trajectory):
    if "raw_pressure" not in trajectory or "refresh_ticks" not in trajectory:
        return {"known": False, "ever_active": None, "reason": "missing_refresh_pressure_or_ticks"}
    pressure, ticks = np.asarray(trajectory["raw_pressure"]), np.asarray(trajectory["refresh_ticks"])
    if (pressure.ndim != 2 or pressure.shape[1] != 44 or ticks.shape != (len(pressure),)
            or ticks.dtype.kind not in "iu" or not len(ticks) or (ticks < 0).any() or (ticks >= 600).any()
            or (np.diff(ticks) <= 0).any() or not np.isfinite(pressure).all()):
        raise ValueError("Malformed raw pressure44 / pre-action refresh ticks")
    active = (pressure > 5e-4).any(axis=1)
    return {"known": True, "ever_active": bool(active.any()), "active_refreshes": int(active.sum()),
            "refreshes": len(pressure), "first_active_observation_tick": int(ticks[active][0]) if active.any() else None,
            "maximum_raw_pressure": float(pressure.max()), "threshold_proxy_units": 5e-4,
            "scope": "native sensor proxy at pre-action observation ticks; not robot/object collision truth"}


def classify_case(result, trajectory):
    centers, lifted, inside = [np.asarray(trajectory[key]) for key in ("cube_centers", "lifted", "inside_goal")]
    if centers.shape != (600, 3) or lifted.shape != (600,) or inside.shape != (600,):
        raise ValueError("Completed trajectories must contain 600 cube/lift/goal rows")
    if lifted.dtype.kind != "b" or inside.dtype.kind != "b":
        raise ValueError("lifted and inside_goal must contain boolean flags")
    initial, final, outcome = result["initial_info"], result["final_info"], result["outcome"]
    checked = score_trajectory(initial, final, centers, lifted, inside)
    for key in ("strict_stable_goal", "ever_lifted", "lift_then_drop_outside_goal"):
        if not isinstance(outcome.get(key), bool) or outcome[key] != checked[key]:
            raise ValueError(f"Recorded outcome does not match full scoring trajectory: {key}")
    if outcome.get("first_drop_tick") != checked["first_drop_tick"]:
        raise ValueError("Recorded first_drop_tick differs from the original scoring definition")
    for key in ("final_goal_center_xy_distance_m", "final_goal_region_xy_distance_m"):
        if not np.isclose(outcome.get(key, np.nan), checked[key], atol=1e-10, rtol=1e-8):
            raise ValueError(f"Recorded goal distance differs from trajectory: {key}")
    contact = exact_object_contact(trajectory, initial)
    success, ever_lift, drop = [outcome[key] for key in ("strict_stable_goal", "ever_lifted", "lift_then_drop_outside_goal")]
    if success:
        stage = "success"
    elif ever_lift:
        stage = "lift_then_drop_outside_goal" if drop else "lift_without_drop_not_stable"
    elif contact["ever_contact"] is None:
        stage = "unknown"
    else:
        stage = "contact_without_lift" if contact["ever_contact"] else "no_object_contact"
    ever_goal = bool(initial.get("inside_goal", False) or inside.any())
    first_lift = 0 if initial.get("lifted", False) else int(np.flatnonzero(lifted)[0] + 1) if lifted.any() else None
    initial_height = float(initial["cube_center"][2])
    clipping = final.get("cumulative_clipped_count")
    if clipping is not None and not nonnegative_integer(clipping):
        raise ValueError("cumulative_clipped_count must be a nonnegative integer")
    return {"stage": stage, "object_contact_scoring_only": contact, "pressure_proxy": pressure_proxy(trajectory),
            "strict_stable_goal": success, "ever_lifted": ever_lift, "first_lift_tick": first_lift,
            "lift_then_drop_outside_goal": drop, "first_drop_tick": checked["first_drop_tick"],
            "actual_height_at_least_initial_plus_5cm_ticks": int((centers[:, 2] >= initial_height + .05).sum()),
            "max_cube_height_m": checked["max_cube_height_m"], "ever_inside_goal": ever_goal,
            "entered_goal_without_final_stable_success": ever_goal and not success,
            "first_goal_tick": final.get("first_goal_tick"), "final_inside_goal": bool(final["inside_goal"]),
            "final_stable_goal_ticks": final.get("stable_goal_ticks"),
            "final_goal_center_xy_distance_m": checked["final_goal_center_xy_distance_m"],
            "final_goal_region_xy_distance_m": checked["final_goal_region_xy_distance_m"],
            "cumulative_clipped_count": int(clipping) if clipping is not None else None}


def subset_counts(records, planned, subset, canonical):
    expected = {name for name in planned if canonical[name]["subsets"][subset]}
    rows = [row for row in records if row["case_id"] in expected and row["status"] == "complete"]
    count = len(rows)
    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {"planned": bool(expected), "planned_cases": len(expected), "completed_cases": count,
            "complete": count == len(expected), "case_ids": [row["case_id"] for row in rows],
            "missing_case_ids": sorted(expected - {row["case_id"] for row in rows}),
            "stage_counts": {stage: sum(row["stage"] == stage for row in rows) if count else None for stage in STAGES},
            "stage_rate_denominator_completed_cases": count,
            "exact_object_contact_cases": sum(row["object_contact_scoring_only"]["known"] for row in rows) if count else None,
            "entered_goal_without_stable_success_cases": sum(row["entered_goal_without_final_stable_success"] for row in rows) if count else None,
            "mean_cumulative_clipped_dimensions": mean("cumulative_clipped_count"),
            "mean_final_goal_center_xy_distance_m": mean("final_goal_center_xy_distance_m"),
            "mean_final_goal_region_xy_distance_m": mean("final_goal_region_xy_distance_m")}


def audit(experiment_root, manifest_path=None):
    root = Path(experiment_root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True) if manifest_path else root / "experiment_manifest.json"
    sources = {}
    def record_source(path):
        sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    def read(path):
        record_source(path)
        return json.loads(path.read_text())
    manifest = read(manifest_path)
    entries = manifest.get("expected_evaluations", [])
    if not entries:
        raise ValueError("Experiment manifest has no expected_evaluations")
    canonical, evaluations, seen = canonical_cases(), [], set()
    for entry in entries:
        key = (entry["kind"], entry["mode"], entry["n_action_steps"], entry["directory"])
        if key in seen:
            raise ValueError("Duplicate expected evaluation in experiment manifest")
        seen.add(key)
        directory = (root / entry["directory"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Evaluation directory escapes experiment root")
        planned, problems, records = declared_ids(entry["case_set"]), [], []
        config_path, plan_path = directory / "eval_config.json", directory / "case_plan.json"
        config_hash = None
        if config_path.is_file() and plan_path.is_file():
            config, plan = read(config_path), read(plan_path)
            config_hash = digest(config_path)
            horizon = config.get("n_action_steps_override") or config.get("execution_horizon", {}).get("n_action_steps")
            if (config.get("kind") != entry["kind"] or config.get("case_set") != entry["case_set"]
                    or horizon != entry["n_action_steps"] or config.get("control_ticks") != 600
                    or config.get("physics_dt") != .01 or entry["mode"] not in config.get("requested_modes", [])):
                problems.append("evaluation_protocol_differs_from_manifest")
            if (set(plan.get("declared_case_ids", [])) != planned or set(plan.get("selected_case_ids", [])) != planned
                    or len(plan.get("selected_case_ids", [])) != len(planned) or plan.get("case_limit") is not None):
                problems.append("partial_or_changed_case_plan")
        else:
            problems.append("missing_eval_config_or_case_plan")
        for name in sorted(planned):
            case_dir = directory / entry["mode"] / name
            result_path, trajectory_path = case_dir / "result.json", case_dir / "trajectory.npz"
            row = {"case_id": name, "subsets": canonical[name]["subsets"], "status": "missing", "stage": None,
                   "result": str(result_path), "trajectory": str(trajectory_path)}
            if result_path.is_file():
                try:
                    result = read(result_path)
                    if not result.get("complete"):
                        row["status"] = "incomplete"
                    elif not trajectory_path.is_file():
                        row.update(status="invalid", error="completed_result_missing_trajectory")
                    else:
                        case, expected = result.get("case", {}), canonical[name]
                        expected_seed = int(manifest.get("seed", 43)) + (int(name[5:]) if name.startswith("grid_") else 0)
                        if (config_hash is None or result.get("config_sha256") != config_hash or result.get("mode") != entry["mode"]
                                or result.get("ticks") != 600 or not np.isclose(result.get("sim_time_s", np.nan), 6., atol=1e-8, rtol=0)
                                or case.get("case_id") != name or case.get("episode_id") != 1 or case.get("frame") != 0
                                or case.get("seed") != expected_seed or np.asarray(case.get("object_xy_offset", [])).shape != (2,)
                                or not np.allclose(case["object_xy_offset"], expected["offset"], atol=1e-12, rtol=0)):
                            raise ValueError("Completed case does not match manifest protocol/config/initial condition")
                        record_source(trajectory_path)
                        with np.load(trajectory_path, allow_pickle=False) as trajectory:
                            row.update(classify_case(result, trajectory))
                        row.update(status="complete", case=case)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    row.update(status="invalid", stage=None, error=str(error))
            records.append(row)
        complete = not problems and all(row["status"] == "complete" for row in records)
        evaluations.append({**entry, "directory": str(directory), "complete": complete,
            "problems": problems, "planned_cases": len(planned),
            "completed_cases": sum(row["status"] == "complete" for row in records),
            "subsets": {subset: subset_counts(records, planned, subset, canonical) for subset in SUBSETS},
            "cases": records})
    valid = [row for value in evaluations for row in value["cases"] if row["status"] == "complete"]
    complete = all(value["complete"] for value in evaluations)
    return {"audit_version": 1, "complete": complete,
            "contact_attribution_complete": complete and all(row["object_contact_scoring_only"]["known"] for row in valid),
            "planned_cases": sum(value["planned_cases"] for value in evaluations), "completed_cases": len(valid),
            "unknown_stage_cases": sum(row["stage"] == "unknown" for row in valid),
            "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "experiment_manifest": str(manifest_path), "evaluations": evaluations, "sources": sources,
            "stage_order": list(STAGES),
            "definitions": {"success": "final strict_stable_goal; success takes precedence over earlier drops",
                "contact": "scoring-only robot_cube_contact_count after all 600 physics steps, plus initial tick0; never pressure proxy",
                "lift": "original recorded outcome.ever_lifted, verified against original score_trajectory",
                "drop": "original score_trajectory: previously lifted, cube center z<0.05m, outside goal; absolute threshold unchanged",
                "unknown": "unsuccessful and never lifted, with unavailable exact object-contact evidence",
                "complete": "every manifest-declared evaluation/case has valid completed result/trajectory; includes separately retained legacy references",
                "missing": "no stages or performance values imputed for missing/incomplete cases; counts use completed cases only",
                "scope": "observed failure phase, not causal diagnosis; nominal remains separate from grid subsets; no training or simulator calls"},
            "implementation_sha256": digest(__file__),
            "score_implementation_sha256": digest(ROOT / "scripts/evaluate_token_act.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", "--input", type=Path, default=ROOT / "outputs/simple_policy_followup_20261025")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/simple_policy_followup_20261025/failure_stages.json")
    args = parser.parse_args()
    result = audit(args.experiment_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("complete", "contact_attribution_complete", "planned_cases", "completed_cases", "unknown_stage_cases")}, allow_nan=False))


if __name__ == "__main__":
    main()
