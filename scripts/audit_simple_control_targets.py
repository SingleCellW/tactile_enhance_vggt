#!/usr/bin/env python3
"""CPU-only audit of source targets and executed control targets before contact.

Loads the actual frozen MuJoCo model to read its actuator map and control
ranges. Never constructs MjData, a renderer, a tactile model or a policy, and
never advances physics. All clipping below operates on copies for analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, machine_path, require_machine_arguments
from tactile_vggt.simulation import load_frozen_model


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clamp_targets(actions, bounds, limited):
    result = np.array(actions, dtype=np.float64, copy=True)
    result[:, limited] = np.clip(result[:, limited], bounds[limited, 0], bounds[limited, 1])
    return result


def outside_range(actions, bounds, limited):
    return ((actions < bounds[:, 0]) | (actions > bounds[:, 1])) & limited


def comparison(expert, policy, bounds, limited, action_std):
    expert_ctrl, policy_ctrl = [clamp_targets(value, bounds, limited) for value in (expert, policy)]
    raw_error = np.abs(policy - expert) / action_std
    ctrl_error = np.abs(policy_ctrl - expert_ctrl) / action_std
    expert_out, policy_out = [outside_range(value, bounds, limited) for value in (expert, policy)]
    def saturation(value):
        return (np.isclose(value, bounds[:, 0], atol=1e-9, rtol=0)
                | np.isclose(value, bounds[:, 1], atol=1e-9, rtol=0)) & limited
    expert_sat, policy_sat = saturation(expert_ctrl), saturation(policy_ctrl)
    equivalent = expert_sat & policy_sat & np.isclose(expert_ctrl, policy_ctrl, atol=1e-9, rtol=0)
    def summarize(selection):
        raw, ctrl = raw_error[:, selection], ctrl_error[:, selection]
        return {"normalized_raw_command_mae": float(raw.mean()),
                "normalized_clamped_ctrl_mae": float(ctrl.mean()),
                "raw_clamped_ctrl_mae_radian": float(np.abs(policy_ctrl[:, selection] - expert_ctrl[:, selection]).mean()),
                "normalized_mae_reduction_fraction": float(1 - ctrl.mean() / raw.mean()) if raw.mean() else None,
                "expert_out_of_range_fraction": float(expert_out[:, selection].mean()),
                "policy_out_of_range_fraction": float(policy_out[:, selection].mean()),
                "expert_saturated_fraction": float(expert_sat[:, selection].mean()),
                "policy_saturated_fraction": float(policy_sat[:, selection].mean()),
                "both_same_saturation_fraction": float(equivalent[:, selection].mean()),
                "raw_mae_contribution_from_both_same_saturation": float((raw * equivalent[:, selection]).sum() / raw.sum()) if raw.sum() else None}
    return {"aggregate": {name: summarize(selection) for name, selection in
                          (("all22", slice(None)), ("arm6", slice(0, 6)), ("hand16", slice(6, None)))},
            "per_dimension": [{"index": index, **summarize(slice(index, index + 1))} for index in range(22)]}


def first_contact(trajectory, initial):
    if initial.get("robot_cube_contact_count", 0) > 0:
        return 0
    if "robot_cube_contact_count" not in trajectory:
        return None
    values = np.asarray(trajectory["robot_cube_contact_count"])
    if values.shape != (600,) or values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError("Need 600 nonnegative scoring contact counts")
    active = np.flatnonzero(values > 0)
    return int(active[0] + 1) if len(active) else None


def audit(args):
    import mujoco  # CPU model compilation only; no Torch or GPU dependency.
    sources = {}
    def record(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
        return path
    def read(path):
        return json.loads(record(path).read_text())
    def arrays(path):
        with np.load(record(path), allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    asset_root = args.dynamic_root / "eval/assets/picklift"
    model = load_frozen_model(asset_root, mujoco)
    if mujoco.__version__ != "3.4.0" or (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError("Require the frozen MuJoCo3.4.0 PickLift model")
    xmls = {str(path.relative_to(asset_root)): digest(record(path)) for path in sorted(asset_root.rglob("*.xml"))}
    bounds, limited = np.array(model.actuator_ctrlrange), model.actuator_ctrllimited.astype(bool)
    mapping = []
    for index in range(model.nu):
        joint = int(model.actuator_trnid[index, 0])
        mapping.append({"index": index,
            "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index),
            "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), "joint_id": joint,
            "qposadr": int(model.jnt_qposadr[joint]), "dofadr": int(model.jnt_dofadr[joint]),
            "transmission_type": int(model.actuator_trntype[index]), "gear": float(model.actuator_gear[index, 0]),
            "ctrl_limited": bool(limited[index]), "ctrlrange": bounds[index].tolist(),
            "joint_limited": bool(model.jnt_limited[joint]), "joint_range": model.jnt_range[joint].tolist()})
    identity_map = all(row["qposadr"] == row["index"] == row["dofadr"] and row["transmission_type"] == 0 and row["gear"] == 1 for row in mapping)
    if not identity_map:
        raise ValueError("Actual 22D control/joint map differs from the audited identity convention")
    metadata = read(args.cache / "manifest.json")
    if not metadata.get("complete"):
        raise ValueError("Source cache is incomplete")
    cached = {name: np.load(record(args.cache / f"{name}.npy"), mmap_mode="r", allow_pickle=False)
              for name in ("action", "episode_index", "obs_frame_indices")}
    train_mask = np.isin(cached["episode_index"], metadata["split"]["train"])
    source = np.asarray(cached["action"][train_mask], dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 22 or not np.isfinite(source).all():
        raise ValueError("Need finite source action22 labels")
    source_ctrl = clamp_targets(source, bounds, limited)
    oob = outside_range(source, bounds, limited)
    raw_var, ctrl_var = source.var(0), source_ctrl.var(0)
    train_obs = cached["obs_frame_indices"][np.isin(cached["episode_index"][cached["obs_frame_indices"]], metadata["split"]["train"])]
    obs_oob = outside_range(cached["action"][train_obs], bounds, limited)
    source_rows = [{"index": index, "out_of_range_fraction": float(oob[:, index].mean()),
        "raw_min": float(source[:, index].min()), "raw_max": float(source[:, index].max()),
        "raw_variance": float(raw_var[index]), "clamped_variance": float(ctrl_var[index]),
        "clamped_to_raw_variance_ratio": float(ctrl_var[index] / raw_var[index]) if raw_var[index] else None,
        "mean_abs_clipping_radian": float(np.abs(source[:, index] - source_ctrl[:, index]).mean())}
        for index in range(22)]
    reference_result = read(args.reference / "result.json")
    reference = arrays(args.reference / "trajectory.npz")
    if not reference_result.get("complete") or not reference_result["outcome"]["strict_stable_goal"]:
        raise ValueError("Reference must be the completed successful historical nominal replay")
    qpos_sha = reference_result["initial_info"]["initialization"]["qpos_sha256"]
    policies, contact_ticks = [], []
    for mode in args.modes:
        directory = args.experiment_root / "eval_n8" / mode / mode / "nominal"
        result, trajectory = read(directory / "result.json"), arrays(directory / "trajectory.npz")
        stats = read(args.experiment_root / "train" / mode / "normalization.json")
        if (not result.get("complete") or result.get("mode") != mode or result.get("ticks") != 600
                or result["case"]["object_xy_offset"] != [0., 0.]
                or result["initial_info"]["initialization"]["qpos_sha256"] != qpos_sha):
            raise ValueError("Policy nominal reset/protocol differs from historical reference")
        for key in ("state_mean", "state_std", "action_mean", "action_std"):
            if stats[key] != metadata["normalization"][key]:
                raise ValueError("Policy state/action normalization differs from original source cache")
        if "robot_cube_contact_count" not in trajectory:
            raise ValueError("Policy precontact window requires complete per-tick object-contact counts")
        contact = first_contact(trajectory, result["initial_info"])
        if contact is not None:
            contact_ticks.append(contact)
        policies.append((mode, result, trajectory, stats, contact))
    reference_first = first_contact(reference, reference_result["initial_info"])
    if reference_first is not None:
        contact_ticks.append(reference_first)
    window = min([args.max_ticks, *[tick - 1 for tick in contact_ticks]])
    if window < 1:
        raise ValueError("No nonempty shared precontact command window")
    expert = np.asarray(reference["actions"][:window], dtype=np.float64)
    if expert.shape != (window, 22) or not np.isfinite(expert).all():
        raise ValueError("Reference commands have invalid dimensions/values")
    if not np.array_equal(reference["source_frames"][:window], np.arange(window)):
        raise ValueError("Reference commands are not original source frames0..window-1")
    comparisons = []
    for mode, result, trajectory, stats, contact in policies:
        policy = np.asarray(trajectory["actions"][:window], dtype=np.float64)
        if policy.shape != expert.shape or not np.isfinite(policy).all():
            raise ValueError("Policy commands have invalid dimensions/values")
        q, ticks = trajectory["refresh_states"], trajectory["refresh_ticks"]
        pre = q[ticks <= window]
        comparisons.append({"mode": mode, "first_exact_object_contact_tick": contact,
            **comparison(expert, policy, bounds, limited, np.asarray(stats["action_std"])),
            "observed_q": {"all_finite": bool(np.isfinite(q).all()), "all_actions_finite": bool(np.isfinite(trajectory["actions"]).all()),
                "sampling": "policy observation refresh ticks only; no expert q saved",
                "precontact_observed_ticks": ticks[ticks <= window].tolist(),
                "precontact_arm_range": [float(pre[:, :6].min()), float(pre[:, :6].max())],
                "precontact_hand_range": [float(pre[:, 6:].min()), float(pre[:, 6:].max())]},
            "pressure_precontact_refresh_max": trajectory["raw_pressure"][ticks <= window].max(axis=1).tolist(),
            "final_cumulative_clipped_count": result["final_info"]["cumulative_clipped_count"]})
    trace_path = args.reference / "trace.jsonl"
    trace = [json.loads(line) for line in record(trace_path).read_text().splitlines()]
    record(__file__)
    record(ROOT / "src/tactile_vggt/simulation.py")
    return {"audit_version": 1, "complete": True,
        "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model": {"mujoco_version": mujoco.__version__, "dimensions_nq_nv_nu": [model.nq, model.nv, model.nu],
            "asset_root": str(asset_root), "xml_sha256": xmls, "identity_22_joint_control_mapping_verified": identity_map,
            "actuators": mapping},
        "train_source_full_rate": {"rows": len(source), "training_episodes": metadata["split"]["train"],
            "action_semantics": metadata.get("action_semantics"),
            "overall_out_of_range_fraction": float(oob.mean()), "arm_out_of_range_fraction": float(oob[:, :6].mean()),
            "hand_out_of_range_fraction": float(oob[:, 6:].mean()), "any_out_of_range_row_fraction": float(oob.any(axis=1).mean()),
            "per_dimension": source_rows, "variance_definition": "population variance, ddof0, original full-rate training source labels; no file modified",
            "original_action_std_matches_source": bool(np.allclose(np.maximum(np.sqrt(raw_var), 1e-3), metadata["normalization"]["action_std"], rtol=1e-6, atol=1e-8))},
        "train_observation_start_labels": {"rows": len(train_obs), "overall_out_of_range_fraction": float(obs_oob.mean()),
            "per_dimension_out_of_range_fraction": obs_oob.mean(0).tolist(),
            "scope": "only start labels at original training observations; not chunk-weighted label frequencies"},
        "nominal_precontact_window": {"max_requested_commands": args.max_ticks, "commands": window,
            "source_command_indices_inclusive": [0, window - 1], "post_action_ticks_inclusive": [1, window],
            "initial_qpos_sha256": qpos_sha, "policy_modes": args.modes,
            "normalization": "absolute command/control difference divided by ORIGINAL train action_std; unchanged weights after analytical clamp",
            "reference_exact_first_contact_tick": reference_first,
            "reference_contact_recording": "full-rate counts unavailable in old reference trajectory; sparse trace cannot exclude transient contacts",
            "reference_early_sparse_contacts": [{"tick": row["tick"], "count": row["info"]["robot_cube_contact_count"]} for row in trace if row["tick"] <= 60],
            "reference_cube_position_span_first_window": np.ptp(reference["cube_centers"][:window], axis=0).tolist(),
            "reference_has_q_trajectory": any(key in reference for key in ("qpos", "state", "refresh_states")),
            "reference_final_cumulative_clipped_count": reference_result["final_info"]["cumulative_clipped_count"],
            "comparisons": comparisons},
        "interpretation_boundaries": [
            "Targets are analytically clamped copies using the same ranges as Simulation.step; original labels, normalization, models and rollouts are unchanged.",
            "This is a command/control imitation comparison in an early shared window, not expert q tracking error, recovered torque or a new rollout.",
            "Reference has no q trajectory and only sparse contact records; its exact first contact cannot be certified from the saved files.",
            "Source target out-of-range frequency and variance reduction describe label statistics; they do not establish a label bug or the cause of policy failure.",
            "Historical reference succeeds with its own clipped commands. Cumulative clipping counts alone cannot establish a causal failure explanation.",
            "New conditions still complete only the declared 203 rollouts. This audit neither chooses new checkpoints nor expands training."
        ], "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-root", type=Path, default=machine_path("dynamic_root"))
    parser.add_argument("--cache", type=Path, default=artifact_path("cache") / "pickplace_vggt_native_physics_dt01_stride4")
    parser.add_argument("--experiment-root", type=Path, default=artifact_path("outputs") / "simple_policy_followup_20261025")
    parser.add_argument("--reference", type=Path, default=artifact_path("outputs") / "token_pilot_20261025/command_replay_reference/nominal")
    parser.add_argument("--modes", nargs="+", default=["frozen_zero", "frozen_pressure", "joint_zero"])
    parser.add_argument("--max-ticks", type=int, default=30)
    parser.add_argument("--output", type=Path, default=artifact_path("reports") / "simple_policy_followup_20261025/control_target_audit.json")
    args = parser.parse_args()
    require_machine_arguments(args, 'dynamic_root')
    if not 1 <= args.max_ticks <= 30 or len(set(args.modes)) != len(args.modes):
        raise ValueError("Use 1..30 early ticks and unique policy modes")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "train_source_rows": result["train_source_full_rate"]["rows"],
        "window_commands": result["nominal_precontact_window"]["commands"],
        "modes": [row["mode"] for row in result["nominal_precontact_window"]["comparisons"]]}))


if __name__ == "__main__":
    main()
