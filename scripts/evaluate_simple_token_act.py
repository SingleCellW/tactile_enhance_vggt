#!/usr/bin/env python3
"""Nominal and paired XY interventions for simple/legacy token ACT policies.

GT object state stays in scoring/info/trajectory records. Model observations
are explicitly filtered. A declared nominal/support evaluation can complete
without claiming full-grid coverage; --case-limit is always a smoke run.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot/src"), str(ROOT / "scripts")]
from evaluate_sim import SimulationClient, digest, read_observation, timing_stats, write_json
from evaluate_token_act import (checkpoint_map, make_cases as make_grid_cases,
    representation_split, choose_source, score_trajectory, wilson_interval,
    validate_policy_inputs as validate_legacy_policy_inputs)
from tactile_vggt.token_metrics import paired_group_bootstrap

SIMPLE_MODES = {"frozen_zero": ("frozen", "zero"), "frozen_pressure": ("frozen", "pressure"),
                "joint_zero": ("joint", "zero"), "joint_pressure": ("joint", "pressure")}
SUBSETS = {"nominal": 1, "grid36": 36, "within_bbox12": 12, "outside24": 24}
SOURCE_XY = np.array([.44, -.04])
BBOX_MIN, BBOX_MAX = np.array([.44, -.06]), np.array([.56, .06])
OBSERVATION_KEYS = ("rgb", "state", "tactile", "anchors_world", "known_cameras_cv", "base_to_world")


def make_cases(episode=1, seed=43, case_set="all", case_limit=None):
    if episode != 1:
        raise ValueError("This declared source-XY protocol requires episode 1")
    if case_set not in {"all", "support", "nominal", "grid36"}:
        raise ValueError("Unknown case_set")
    grid, _ = make_grid_cases(episode, seed)
    nominal = {"case_id": "nominal", "episode_id": episode, "frame": 0, "seed": int(seed),
               "object_xy_offset": [0., 0.]}
    full = []
    for original in [nominal, *grid]:
        case = dict(original)
        xy = SOURCE_XY + np.asarray(case["object_xy_offset"])
        inside = bool(((xy >= BBOX_MIN - 1e-12) & (xy <= BBOX_MAX + 1e-12)).all())
        is_nominal = case["case_id"] == "nominal"
        case.update(initial_xy=xy.tolist(), initial_xy_within_train_bbox=inside,
                    subset_flags={"nominal": is_nominal, "grid36": not is_nominal,
                                  "within_bbox12": not is_nominal and inside,
                                  "outside24": not is_nominal and not inside})
        full.append(case)
    chosen = [case for case in full if case_set == "all"
              or case_set == "support" and (case["subset_flags"]["nominal"] or case["subset_flags"]["within_bbox12"])
              or case_set == "nominal" and case["subset_flags"]["nominal"]
              or case_set == "grid36" and case["subset_flags"]["grid36"]]
    if case_limit is not None:
        if not 1 <= case_limit <= len(chosen):
            raise ValueError("case-limit must be within the declared case-set size")
        selected = chosen[:case_limit]
    else:
        selected = chosen
    return full, chosen, selected


def safe_observation(observation):
    """Never forward privileged object position, segmentation or worker info."""
    return {key: observation[key] for key in OBSERVATION_KEYS}


def pressure_diagnostics(raw_pressure, observation_tick):
    pressure = np.asarray(raw_pressure)
    if pressure.shape != (44,) or not np.isfinite(pressure).all() or (pressure < 0).any():
        raise ValueError("Need 44 finite nonnegative native sensor values")
    active = pressure > 5e-4
    return {"observation_tick": int(observation_tick), "raw_pressure_max": float(pressure.max()),
            "raw_pressure_sum": float(pressure.sum()), "raw_pressure_active_sensors": int(active.sum()),
            "raw_pressure_contact_flag": bool(active.any()), "contact_threshold_proxy_units": 5e-4}


def object_contact_count(info):
    """Privileged hand-object collision count used solely for scoring."""
    value = info.get("robot_cube_contact_count")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Worker info must provide a nonnegative scoring-only cube contact count")
    return int(value)


def contact_summary(initial_count, post_action_counts):
    counts = np.asarray(post_action_counts)
    if counts.shape != (600,) or (counts < 0).any():
        raise ValueError("Need one scoring-only contact count for every post-action tick")
    active = np.flatnonzero(counts > 0)
    first_post = int(active[0] + 1) if len(active) else None
    return {"initial_contact_count": int(initial_count), "first_post_action_contact_tick": first_post,
            "first_contact_tick": 0 if initial_count > 0 else first_post,
            "post_action_contact_ticks": int(len(active)), "maximum_post_action_contact_count": int(counts.max()),
            "sampling": "scoring-only worker info after every physics step, ticks 1..600; initial count is tick0; not a raw-pressure observation"}


def summarize_subset(results, expected, selected_expected):
    count = len(results)
    success = sum(row["outcome"]["strict_stable_goal"] for row in results)
    return {"cases": count, "expected_cases": expected, "selected_expected_cases": selected_expected,
            "complete": count == expected, "selected_complete": count == selected_expected,
            "strict_stable_goal_count": success, "strict_stable_goal_rate": success / count if count else None,
            "strict_stable_goal_wilson_ci95": wilson_interval(success, count),
            "lifted_count": sum(row["outcome"]["ever_lifted"] for row in results),
            "lift_then_drop_count": sum(row["outcome"]["lift_then_drop_outside_goal"] for row in results),
            "mean_final_goal_center_xy_distance_m": float(np.mean([row["outcome"]["final_goal_center_xy_distance_m"] for row in results])) if count else None,
            "mean_final_goal_region_xy_distance_m": float(np.mean([row["outcome"]["final_goal_region_xy_distance_m"] for row in results])) if count else None,
            "wall_seconds": timing_stats([row["wall_seconds"] for row in results]),
            "encoder_refresh_mean_seconds": timing_stats([row["encoder_refresh"]["mean_s"] for row in results if row.get("encoder_refresh")]),
            "scope": "XY reset sensitivity from one held-out source, not independent source episodes; bounding-box inclusion is not full in-distribution status"}


def summarize_mode(results, planned_cases, primary_subset):
    subsets = {}
    for name, expected in SUBSETS.items():
        chosen = [row for row in results if row["case"]["subset_flags"][name]]
        selected_expected = sum(case["subset_flags"][name] for case in planned_cases)
        subsets[name] = summarize_subset(chosen, expected, selected_expected)
    return {"cases": len(results), "complete": len(results) == len(planned_cases),
            "primary_subset": primary_subset, "primary": subsets[primary_subset],
            "subsets": subsets, "results": results,
            "note": "No combined success rate over nominal plus grid observations."}


def paired_comparisons(models, kind):
    pairs = [("frozen_pressure", "frozen_zero"), ("joint_pressure", "joint_zero")] if kind == "simple" else [("internal", "visual"), ("internal", "late")]
    output = {}
    for subset in SUBSETS:
        comparisons = {}
        for first, second in pairs:
            if first not in models or second not in models:
                continue
            one = {row["case"]["case_id"]: row for row in models[first]["results"] if row["case"]["subset_flags"][subset]}
            two = {row["case"]["case_id"]: row for row in models[second]["results"] if row["case"]["subset_flags"][subset]}
            ids = sorted(one.keys() & two.keys())
            if not ids:
                continue
            for name in ids:
                if one[name]["case"] != two[name]["case"]:
                    raise ValueError("Cannot pair results from different physical initial conditions")
            difference = paired_group_bootstrap([float(one[i]["outcome"]["strict_stable_goal"]) for i in ids],
                         [float(two[i]["outcome"]["strict_stable_goal"]) for i in ids], ids)
            difference.update(direction=f"positive means {first} has higher strict success rate than {second}",
                bootstrap_unit="paired XY reset case from one original source episode",
                scope="fixed-case sensitivity, not independent episode/seed replication")
            comparisons[f"{first}_minus_{second}"] = difference
        output[subset] = comparisons
    return output


def validate_cache(metadata):
    configuration = metadata.get("configuration", {})
    if not metadata.get("complete") or metadata.get("native_tactile_report", {}).get("velocity_dt") != .01:
        raise ValueError("Require complete native physical-dt=.01 cache")
    if not isinstance(metadata.get("action_semantics"), str) or not metadata["action_semantics"].strip():
        raise ValueError("Require the original cache's recorded action_semantics; never assume or shift labels")
    alignment = configuration.get("rgb_alignment", {})
    if alignment.get("verified") is not True or alignment.get("rgb_lag_frames") != 1:
        raise ValueError("Require verified historical one-frame RGB alignment")
    if 1 not in metadata.get("split", {}).get("val", []):
        raise ValueError("Source episode 1 must be held out from ACT training")


def validate_simple_policy(mode, checkpoint, probe, cache_hash, normalization, obs_hash, action_semantics):
    if mode not in SIMPLE_MODES:
        raise ValueError(f"Unknown simple mode {mode}")
    files = {name: json.loads((checkpoint / f"{name}.json").read_text()) for name in
             ("checkpoint_complete", "config", "normalization", "feature_manifest", "train_config")}
    config, feature, training = files["config"], files["feature_manifest"], files["train_config"]
    projection, tactile = SIMPLE_MODES[mode]
    structure = config.get("structure", {})
    if config.get("policy_type") != "simple_token_act" or structure.get("projection") != projection or structure.get("tactile") != tactile:
        raise ValueError("Simple checkpoint mode/structure mismatch")
    if config.get("config", {}).get("encoder_dim") != 300:
        raise ValueError("Simple ACT must have the declared 300D condition")
    probe_hash = digest(probe)
    if (not feature.get("complete") or feature.get("cache_manifest_sha256") != cache_hash
            or feature.get("obs_frame_indices_sha256") != obs_hash
            or feature.get("source_checkpoint_sha256") != probe_hash
            or feature.get("feature_dim") != 1088):
        raise ValueError("Simple feature source/cache/probe binding mismatch")
    if training.get("data_manifest_sha256") != cache_hash or training.get("visual_probe_checkpoint_sha256") != probe_hash:
        raise ValueError("Simple policy training provenance mismatch")
    if training.get("action_semantics") != action_semantics:
        raise ValueError("Simple policy action semantics differ from the original cache")
    for name in ("state_mean", "state_std", "action_mean", "action_std"):
        if files["normalization"].get(name) != normalization.get(name):
            raise ValueError("Simple ACT state/action normalization differs from training cache")
    scale = files["normalization"].get("simple_pressure_scale")
    if not isinstance(scale, (float, int)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Simple pressure normalization must have a positive train-only scale")
    policy = config["config"]
    return {"chunk_size": policy["chunk_size"], "n_action_steps": policy["n_action_steps"],
            "temporal_ensemble_coeff": policy["temporal_ensemble_coeff"]}


def validate_worker(provenance, metadata):
    expected = {"physics_dt": .01, "control_dt": .01, "substeps": 1, "sensor_dt": .01}
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"Worker {name} differs from the audited physical protocol")
    if provenance.get("mujoco_version") != "3.4.0":
        raise ValueError("Worker must use frozen MuJoCo 3.4.0")
    native = metadata["native_tactile_report"]
    for name in ("geometry_sha256", "native_models_sha256", "object_mesh_sha256", "source_readouts_sha256"):
        if name in native and provenance.get(name) != native[name]:
            raise ValueError(f"Worker native/source hash mismatch: {name}")


def validate_reset(case, info):
    center = np.asarray(info["cube_center"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.allclose(center[:2], case["initial_xy"], atol=1e-6, rtol=0):
        raise ValueError("Actual reset XY differs from declared source XY plus offset")


def implementation_hashes(kind):
    paths = [Path(__file__), ROOT / "scripts/evaluate_token_act.py", ROOT / "scripts/evaluate_sim.py",
             ROOT / "scripts/sim_worker.py", ROOT / "src/tactile_vggt/simulation.py"]
    paths.extend([ROOT / "scripts/simple_token_runtime.py", ROOT / "src/tactile_vggt/simple_token_policy.py"] if kind == "simple" else
                 [ROOT / "scripts/token_runtime.py", ROOT / "src/tactile_vggt/token_policy.py"])
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.exists()}


def create_runtime(args, probes):
    if args.kind == "simple":
        from simple_token_runtime import SimpleVisualRuntime
        return SimpleVisualRuntime(args.repository, args.weights, args.visual_probe_checkpoint, args.device)
    from token_runtime import LiveTokenEncoder
    return LiveTokenEncoder(args.repository, args.weights, probes, device=args.device)


def create_policy(args, checkpoint):
    if args.kind == "simple":
        from tactile_vggt.simple_token_policy import SimpleTokenACTInference
        return SimpleTokenACTInference(checkpoint, device=args.device, amp=args.amp, n_action_steps=args.n_action_steps)
    from tactile_vggt.token_policy import TokenACTInference
    policy = TokenACTInference(checkpoint, device=args.device, amp=args.amp)
    if args.n_action_steps is not None:
        if args.n_action_steps > policy.policy.config["chunk_size"]:
            raise ValueError("Feedback horizon cannot exceed trained action chunk")
        policy.policy.config["n_action_steps"] = args.n_action_steps
        policy.policy.act.config.n_action_steps = args.n_action_steps
        policy.reset()  # ACT recreates its queue using the overridden config.
    return policy


def action_numpy(action):
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    result = np.asarray(action)
    if result.shape != (22,) or not np.isfinite(result).all():
        raise ValueError("Invalid physical 22D ACT action")
    return result


def run(args):
    policies = checkpoint_map(args.policy_checkpoint)
    if not policies:
        raise ValueError("At least one checkpoint is required")
    if args.kind == "simple":
        if args.visual_probe_checkpoint is None:
            raise ValueError("Simple evaluation requires --visual-probe-checkpoint")
        args.visual_probe_checkpoint = args.visual_probe_checkpoint.resolve(strict=True)
        probes = {"visual": args.visual_probe_checkpoint}
    else:
        probes = checkpoint_map(args.probe_checkpoint)
        if set(probes) != set(policies):
            raise ValueError("Legacy probe/policy modes must match")
    cache = args.cache.resolve(strict=True)
    metadata = json.loads((cache / "manifest.json").read_text())
    validate_cache(metadata)
    cache_hash = digest(cache / "manifest.json")
    weight_hash = digest(args.weights)
    expected_weights = metadata.get("configuration", {}).get("weights", {}).get("sha256")
    if expected_weights is not None and weight_hash != expected_weights:
        raise ValueError("Live VGGT weights differ from the source cache")
    representation, split_links = representation_split(probes, args.representation_manifest)
    _, roles = choose_source(metadata, representation, requested=1)
    if roles["representation_role"] != "test":
        raise ValueError("Source episode 1 must be held out from representation training")
    signatures = []
    for mode, checkpoint in policies.items():
        if args.kind == "simple":
            signature = validate_simple_policy(mode, checkpoint, args.visual_probe_checkpoint, cache_hash,
                        metadata["normalization"], digest(cache / "obs_frame_indices.npy"), metadata["action_semantics"])
        else:
            signature = validate_legacy_policy_inputs(mode, checkpoint, probes[mode], cache_hash, metadata["normalization"])
        if args.n_action_steps is not None:
            if args.n_action_steps > signature["chunk_size"]:
                raise ValueError("Feedback horizon exceeds action chunk")
            signature["n_action_steps"] = args.n_action_steps
        signatures.append(signature)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Paired modes must share one ACT execution horizon")
    full_cases, declared_cases, selected = make_cases(1, args.seed, args.case_set, args.case_limit)
    primary_subset = "nominal" if args.case_set == "nominal" else "within_bbox12" if args.case_set == "support" else "grid36"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {"case_set": args.case_set, "full37_cases": full_cases, "declared_case_ids": [case["case_id"] for case in declared_cases],
            "selected_case_ids": [case["case_id"] for case in selected], "case_limit": args.case_limit,
            "seed": args.seed, "source_split_roles": roles, "source_initial_xy": SOURCE_XY.tolist(),
            "train_xy_bbox": {"minimum": BBOX_MIN.tolist(), "maximum": BBOX_MAX.tolist()},
            "subset_note": "XY bounding-box inclusion is not full in-distribution status; nominal is separate from grid36.",
            "ordering": "nominal then original corner-first fixed grid; support filters this same sequence",
            "representation_training_links": split_links}
    config = {"protocol": "simple_token_act_37_v1", "kind": args.kind, "case_set": args.case_set,
              "primary_subset": primary_subset, "requested_modes": list(policies), "case_plan": plan,
              "cache_manifest_sha256": cache_hash, "control_ticks": 600, "physics_dt": .01, "sensor_velocity_dt": .01,
              "action_semantics": metadata["action_semantics"],
              "sim_horizon_s": 6., "n_action_steps_override": args.n_action_steps, "execution_horizon": signatures[0],
              "modes": {mode: {"policy": str(checkpoint), "policy_sha256": digest(checkpoint / "model.pt"),
                  "checkpoint_complete_sha256": digest(checkpoint / "checkpoint_complete.json"),
                  "train_config_sha256": digest(checkpoint / "train_config.json"),
                  "normalization_sha256": digest(checkpoint / "normalization.json")} for mode, checkpoint in policies.items()},
              "probes": {mode: {"path": str(path), "sha256": digest(path)} for mode, path in probes.items()},
              "weights_sha256": weight_hash, "repository": str(args.repository.resolve()), "device": args.device,
              "amp": args.amp, "trace_every": args.trace_every, "video_cases": args.video_cases,
              "checkpoint_selection": args.selection_note, "implementations": implementation_hashes(args.kind),
              "worker": {key: str(getattr(args, key)) for key in ("worker_python", "dynamic_root", "raw_root", "timestamps_root", "source_readouts")},
              "time_semantics": "Simulation pauses for inference; this is not a realtime robot benchmark.",
              "uncertainty": "One source episode and policy training seed; intervals summarize fixed XY resets only.",
              "gpu_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")}}
    config_path, plan_path = output / "eval_config.json", output / "case_plan.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config or not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Saved evaluation protocol changed; use unchanged --resume or a fresh output")
    else:
        if args.resume or plan_path.exists():
            raise ValueError("Incomplete saved protocol; use a fresh output")
        write_json(plan_path, plan)
        write_json(config_path, config)
    config_hash = digest(config_path)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        synchronize = lambda: torch.cuda.synchronize(args.device)
    else:
        synchronize = lambda: None
    worker_command = [str(args.worker_python), str(ROOT / "scripts/sim_worker.py"), "--ipc-dir", str(output / "ipc"),
        "--dynamic-root", str(args.dynamic_root), "--raw-root", str(args.raw_root), "--timestamps-root", str(args.timestamps_root),
        "--source-readouts", str(args.source_readouts), "--device", args.device, "--substeps", "1", "--sensor-dt", ".01"]
    client, models = None, {}
    started = time.monotonic()

    def save_summary():
        selected_complete = len(models) == len(policies) and all(value["cases"] == len(selected) for value in models.values())
        write_json(output / "summary.json", {"protocol": config["protocol"], "kind": args.kind,
            "complete": selected_complete and args.case_limit is None,
            "selected_cases_complete": selected_complete, "full_grid_complete": selected_complete and len([case for case in selected if case["subset_flags"]["grid36"]]) == 36,
            "full37_complete": selected_complete and len(selected) == 37,
            "case_set": args.case_set, "primary_subset": primary_subset, "requested_modes": list(policies),
            "n_action_steps_override": args.n_action_steps, "execution_horizon": signatures[0],
            "config_sha256": config_hash, "case_plan_sha256": digest(plan_path), "source_split_roles": roles,
            "models": models, "paired_comparisons": paired_comparisons(models, args.kind),
            "time_semantics": config["time_semantics"], "scope": config["uncertainty"],
            "elapsed_s": time.monotonic() - started, "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})

    try:
        encoder = create_runtime(args, probes)
        client = SimulationClient(worker_command, output / "worker.log", args.worker_timeout)
        for mode, checkpoint in policies.items():
            policy = create_policy(args, checkpoint)
            if policy.policy.config["n_action_steps"] != signatures[0]["n_action_steps"]:
                raise ValueError("Actual policy queue horizon differs from saved evaluation protocol")
            mode_output = output / mode
            mode_output.mkdir(exist_ok=True)
            results = []
            for case_number, case in enumerate(selected):
                directory = mode_output / case["case_id"]
                directory.mkdir(exist_ok=True)
                result_path = directory / "result.json"
                if args.resume and result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if (not previous.get("complete") or previous.get("config_sha256") != config_hash
                            or previous.get("case") != case or previous.get("mode") != mode
                            or previous.get("ticks") != 600 or not (directory / "trajectory.npz").is_file()):
                        raise ValueError("Saved case does not match the complete fixed protocol")
                    results.append(previous)
                    continue
                policy.reset()
                reset = {name: case[name] for name in ("episode_id", "frame", "seed", "object_xy_offset")}
                reply = client.request("reset", **reset)
                observation, initial_info = safe_observation(read_observation(reply)), reply["info"]
                provenance = reply.get("provenance", {})
                validate_worker(provenance, metadata)
                validate_reset(case, initial_info)
                current_info = initial_info
                initial_contact_count = object_contact_count(initial_info)
                actions, cubes, lifted, inside, states, refresh_ticks, raw_pressures, refresh_inputs = [], [], [], [], [], [], [], []
                object_contacts = []
                refresh_times, policy_times, step_times, observation_times, pressure_rows = [], [], [], [], []
                case_started, video = time.monotonic(), None
                if case_number < args.video_cases:
                    import cv2
                    video = cv2.VideoWriter(str(directory / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        1 / (.01 * signatures[0]["n_action_steps"]), (448, 224))
                    if not video.isOpened():
                        raise RuntimeError("Video writer failed")
                try:
                    with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                        for tick in range(600):
                            refreshed, diagnostics = policy.needs_observation, None
                            pre_contact = object_contact_count(current_info)
                            if refreshed:
                                if observation is None:
                                    begin = time.monotonic()
                                    observation = safe_observation(read_observation(client.request("observe")))
                                    observation_times.append(time.monotonic() - begin)
                                diagnostics = pressure_diagnostics(observation["tactile"], tick)
                                diagnostics["cube_contacts_scoring_only"] = pre_contact
                                pressure_rows.append(diagnostics)
                                synchronize()
                                begin = time.monotonic()
                                features = encoder.encode_observation(observation)
                                if args.kind == "legacy":
                                    features = features[mode]
                                features = np.asarray(features)
                                dimension = 1088 if args.kind == "simple" else 256
                                if features.shape != (dimension,) or not np.isfinite(features).all():
                                    raise ValueError("Runtime features have invalid dimensions or values")
                                synchronize()
                                refresh_times.append(time.monotonic() - begin)
                                states.append(observation["state"])
                                refresh_ticks.append(tick)
                                raw_pressures.append(observation["tactile"])
                                refresh_inputs.append(features)
                                if video:
                                    video.write(np.concatenate(observation["rgb"], axis=1)[..., ::-1])
                                begin = time.monotonic()
                                if args.kind == "simple":
                                    action = policy.select_action(observation["state"], features, observation["tactile"])
                                else:
                                    action = policy.select_action(observation["state"], features)
                            else:
                                begin = time.monotonic()
                                action = policy.select_action()
                            action = action_numpy(action)
                            synchronize()
                            policy_times.append(time.monotonic() - begin)
                            begin = time.monotonic()
                            reply = client.request("step", action=action.tolist(), observe=False)
                            step_times.append(time.monotonic() - begin)
                            observation, current_info = None, reply["info"]
                            actions.append(action)
                            cubes.append(current_info["cube_center"])
                            lifted.append(current_info["lifted"])
                            inside.append(current_info["inside_goal"])
                            object_contacts.append(object_contact_count(current_info))
                            if refreshed or (tick + 1) % args.trace_every == 0 or tick == 599:
                                trace.write(json.dumps({"tick": tick + 1, "sim_time": reply["sim_time"],
                                    "fresh_policy_query": refreshed, "pressure_observation": diagnostics,
                                    "cube_contact_count_post_action_scoring_only": current_info.get("robot_cube_contact_count"),
                                    "info": current_info}, allow_nan=False) + "\n")
                            if (tick + 1) % 100 == 0:
                                print(json.dumps({"mode": mode, "case": case["case_id"], "tick": tick + 1,
                                    "strict_stable_goal": current_info["strict_stable_goal"], "elapsed_s": time.monotonic() - case_started}), flush=True)
                finally:
                    if video:
                        video.release()
                if not np.isclose(reply["sim_time"], 6., rtol=0, atol=1e-8):
                    raise ValueError("Worker physical horizon differs from 600 * .01s")
                np.savez_compressed(directory / "trajectory.npz", actions=np.asarray(actions), cube_centers=np.asarray(cubes),
                    refresh_states=np.asarray(states), refresh_ticks=np.asarray(refresh_ticks), raw_pressure=np.asarray(raw_pressures),
                    readout_inputs=np.asarray(refresh_inputs), lifted=np.asarray(lifted), inside_goal=np.asarray(inside),
                    robot_cube_contact_count=np.asarray(object_contacts, dtype=np.int32))
                result = {"complete": True, "config_sha256": config_hash, "mode": mode, "case": case,
                    "ticks": 600, "sim_time_s": float(reply["sim_time"]), "wall_seconds": time.monotonic() - case_started,
                    "initial_info": initial_info, "final_info": current_info, "worker_provenance": provenance,
                    "outcome": score_trajectory(initial_info, current_info, cubes, lifted, inside),
                    "object_contact_scoring_only": contact_summary(initial_contact_count, object_contacts),
                    "encoder_refresh": timing_stats(refresh_times), "policy_query_and_queue": timing_stats(policy_times),
                    "observation_ipc_render_touch": timing_stats(observation_times), "simulation_step": timing_stats(step_times),
                    "pressure_diagnostics": {"refresh_count": len(pressure_rows),
                        "contact_refreshes": sum(row["raw_pressure_contact_flag"] for row in pressure_rows),
                        "maximum_raw_pressure": max(row["raw_pressure_max"] for row in pressure_rows),
                        "mean_raw_pressure_sum": float(np.mean([row["raw_pressure_sum"] for row in pressure_rows]))},
                    "refresh_timing_scope": "live visual encoding; excludes observation/render/touch IPC",
                    "video": str(directory / "rollout.mp4") if case_number < args.video_cases else None}
                write_json(result_path, result)
                results.append(result)
                write_json(mode_output / "summary.json", summarize_mode(results, selected, primary_subset))
            models[mode] = summarize_mode(results, selected, primary_subset)
            save_summary()
            del policy
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        if client is not None:
            client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--kind", choices=("simple", "legacy"), default="simple")
    result.add_argument("--policy-checkpoint", "--legacy-policy", dest="policy_checkpoint", required=True, action="append", help="mode=checkpoint directory")
    result.add_argument("--visual-probe-checkpoint", type=Path)
    result.add_argument("--probe-checkpoint", "--legacy-probe", dest="probe_checkpoint", action="append", default=[], help="legacy mode=best.pt")
    result.add_argument("--cache", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--representation-manifest", type=Path)
    result.add_argument("--case-set", choices=("all", "support", "nominal", "grid36"), default="all")
    result.add_argument("--case-limit", type=int, help="Smoke prefix; always marked incomplete as a final protocol")
    result.add_argument("--n-action-steps", type=int, choices=(2, 8))
    result.add_argument("--repository", type=Path, default=ROOT / "vendor/vggt")
    result.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--amp", choices=("bfloat16", "float32"), default="bfloat16")
    result.add_argument("--worker-python", type=Path, default=Path("/home/wukf/miniforge3/envs/te/bin/python"))
    result.add_argument("--dynamic-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921"))
    result.add_argument("--raw-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212"))
    result.add_argument("--timestamps-root", type=Path, default=ROOT / "inputs/raw_timestamps")
    result.add_argument("--source-readouts", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz"))
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--trace-every", type=int, default=20)
    result.add_argument("--video-cases", type=int, default=0)
    result.add_argument("--worker-timeout", type=float, default=120.)
    result.add_argument("--selection-note", default="Checkpoint chosen by training validation loss before closed-loop outcomes")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.trace_every < 1 or arguments.worker_timeout <= 0 or arguments.video_cases < 0:
        raise ValueError("Positive trace/timeout and nonnegative video count required")
    try:
        run(arguments)
    except Exception as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
