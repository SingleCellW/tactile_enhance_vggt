#!/usr/bin/env python3
"""Paired, predeclared closed-loop ACT evaluation with live token features.

Each mode sees the same fixed 6x6 grid of object XY offsets. Simulation pauses
for inference; wall latency is separate from the 600 * .01 s physical horizon.
Object truth remains in scoring/trajectory records, never encoder inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
from statistics import NormalDist
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot/src"), str(ROOT / "scripts")]
from evaluate_sim import SimulationClient, digest, read_observation, timing_stats, write_json
from tactile_vggt.token_metrics import paired_group_bootstrap


def checkpoint_map(values):
    result = {}
    for value in values:
        mode, separator, path = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", mode) or mode in result:
            raise ValueError("Checkpoint arguments must use unique safe mode=path names")
        result[mode] = Path(path).expanduser().resolve(strict=True)
    return result


def wilson_interval(successes, count, confidence=.95):
    """Wilson proportion interval; grid-case sensitivity, not training-seed CI."""
    if not isinstance(count, (int, np.integer)) or not 0 <= successes <= count or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    if not count:
        return None
    z = NormalDist().inv_cdf(.5 + confidence / 2)
    rate = successes / count
    denominator = 1 + z*z / count
    center = (rate + z*z / (2*count)) / denominator
    radius = z * np.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def make_cases(episode=1, seed=43, case_limit=None):
    """Corner smoke cases come first, by design, before any outcome is known."""
    if case_limit is not None and not 1 <= case_limit <= 36:
        raise ValueError("case-limit must lie between 1 and 36")
    values = np.linspace(-.035, .035, 6)
    order = [0, 5, 30, 35] + [i for i in range(36) if i not in (0, 5, 30, 35)]
    cases = [{"case_id": f"grid_{index:02d}", "episode_id": int(episode), "frame": 0,
              "seed": int(seed + index),
              "object_xy_offset": [float(values[index // 6]), float(values[index % 6])]}
             for index in order]
    return cases, cases[:case_limit] if case_limit is not None else cases


def representation_split(probes, explicit_manifest=None):
    """Resolve the real pilot dataset through its completed training summary."""
    manifests, links = {}, {}
    for mode, checkpoint in probes.items():
        summary_path = checkpoint.parent.parent / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if explicit_manifest is not None:
            manifest_path = Path(explicit_manifest).resolve(strict=True)
        elif summary is not None:
            manifest_path = Path(summary["args"]["data"]) / "manifest.json"
            manifest_path = manifest_path.resolve(strict=True)
        else:
            raise ValueError("Cannot establish representation split; supply --representation-manifest")
        manifest_hash = digest(manifest_path)
        if summary is not None and summary["data_manifest_sha256"] != manifest_hash:
            raise ValueError("Representation dataset differs from its training summary")
        metadata = json.loads(manifest_path.read_text())
        if not metadata.get("complete"):
            raise ValueError("Incomplete representation dataset")
        manifests[manifest_hash] = metadata
        links[mode] = {"manifest": str(manifest_path), "manifest_sha256": manifest_hash,
                       "training_summary": str(summary_path) if summary is not None else None,
                       "training_summary_sha256": digest(summary_path) if summary is not None else None}
    if len(manifests) != 1:
        raise ValueError("Paired modes must use one representation dataset/split")
    return next(iter(manifests.values())), links


def choose_source(cache_metadata, representation_metadata, requested=None):
    act_val = set(map(int, cache_metadata["split"]["val"]))
    split = representation_metadata["split_episodes"]
    common = sorted(act_val & set(map(int, split["test"])))
    episode = (1 if 1 in common else common[0]) if common else 24
    if requested is not None:
        episode = int(requested)
    if episode not in act_val:
        raise ValueError("Source episode must be held out from ACT action training")
    roles = [name for name, episodes in split.items() if episode in episodes]
    if len(roles) != 1:
        raise ValueError("Source episode must have one unambiguous representation split")
    return episode, {"act_role": "validation", "representation_role": roles[0],
                     "representation_test_and_act_validation_intersection": common,
                     "selection": "prefer episode 1 in joint held-out overlap; else smallest overlap; else episode 24",
                     "scope": "new XY interventions around one source episode, not 36 independent source episodes"}


def score_trajectory(initial_info, final_info, cube_centers, lifted_flags, goal_flags):
    centers = np.asarray(cube_centers, dtype=np.float64)
    lifted = np.asarray(lifted_flags, dtype=bool)
    inside = np.asarray(goal_flags, dtype=bool)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 1:
        raise ValueError("Need a nonempty [ticks,3] cube trajectory")
    if lifted.shape != (len(centers),) or inside.shape != lifted.shape or not np.isfinite(centers).all():
        raise ValueError("Malformed scoring trajectory")
    ever_lifted = np.maximum.accumulate(lifted | bool(initial_info.get("lifted", False)))
    previously_lifted = np.r_[bool(initial_info.get("lifted", False)), ever_lifted[:-1]]
    dropped = previously_lifted & (centers[:, 2] < .05) & ~inside
    final_xy = centers[-1, :2]
    target_xy = np.array([.5, -.3])
    # Distance to the rectangle is zero anywhere in the inherited goal region.
    outside = np.maximum(np.maximum([.4, -.39] - final_xy, final_xy - [.6, -.21]), 0)
    return {"strict_stable_goal": bool(final_info.get("strict_stable_goal", False)),
            "final_inside_goal": bool(final_info["inside_goal"]),
            "ever_lifted": bool(ever_lifted[-1]), "lift_then_drop_outside_goal": bool(dropped.any()),
            "first_drop_tick": int(np.flatnonzero(dropped)[0] + 1) if dropped.any() else None,
            "final_goal_center_xy_distance_m": float(np.linalg.norm(final_xy-target_xy)),
            "final_goal_region_xy_distance_m": float(np.linalg.norm(outside)),
            "max_cube_height_m": float(max(initial_info["cube_center"][2], centers[:, 2].max()))}


def summarize_mode(results):
    count = len(results)
    success = sum(row["outcome"]["strict_stable_goal"] for row in results)
    return {"cases": count, "strict_stable_goal_count": success,
            "strict_stable_goal_rate": success / count if count else None,
            "strict_stable_goal_wilson_ci95": wilson_interval(success, count),
            "lifted_count": sum(row["outcome"]["ever_lifted"] for row in results),
            "lift_then_drop_count": sum(row["outcome"]["lift_then_drop_outside_goal"] for row in results),
            "mean_final_goal_center_xy_distance_m": float(np.mean([row["outcome"]["final_goal_center_xy_distance_m"] for row in results])) if count else None,
            "mean_final_goal_region_xy_distance_m": float(np.mean([row["outcome"]["final_goal_region_xy_distance_m"] for row in results])) if count else None,
            "wall_seconds": timing_stats([row["wall_seconds"] for row in results]),
            "encoder_refresh_mean_seconds": timing_stats([row["encoder_refresh"]["mean_s"] for row in results]),
            "results": results}


def paired_comparisons(models):
    comparisons = {}
    if "internal" not in models:
        return comparisons
    reference = {row["case"]["case_id"]: row for row in models["internal"]["results"]}
    for mode, model in models.items():
        if mode == "internal":
            continue
        other = {row["case"]["case_id"]: row for row in model["results"]}
        ids = sorted(reference.keys() & other.keys())
        a = [float(reference[i]["outcome"]["strict_stable_goal"]) for i in ids]
        b = [float(other[i]["outcome"]["strict_stable_goal"]) for i in ids]
        difference = paired_group_bootstrap(a, b, ids, iterations=2000, seed=43)
        difference.update(direction="positive means internal has higher strict success rate",
                          bootstrap_unit="paired initial-condition case; all cases share one original source episode",
                          scope="sensitivity across this fixed grid, not independent source episodes or training seeds")
        comparisons[f"internal_minus_{mode}"] = difference
    return comparisons


def validate_policy_inputs(mode, checkpoint, probe, cache_hash, normalization):
    if not (checkpoint / "checkpoint_complete.json").is_file():
        raise ValueError("Only completed ACT checkpoints may be evaluated")
    feature = json.loads((checkpoint / "feature_manifest.json").read_text())
    if (not feature.get("complete") or feature.get("mode") != mode or
            feature.get("source_checkpoint_sha256") != digest(probe) or
            feature.get("cache_manifest_sha256") != cache_hash):
        raise ValueError(f"ACT feature source mismatch for {mode}")
    if json.loads((checkpoint / "normalization.json").read_text()) != normalization:
        raise ValueError("ACT normalization differs from the evaluated source cache")
    training = json.loads((checkpoint / "train_config.json").read_text())
    if training["data_manifest_sha256"] != cache_hash:
        raise ValueError("ACT was trained on a different cache")
    config = json.loads((checkpoint / "config.json").read_text())["config"]
    return {key: config[key] for key in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")}


def run(args):
    probes, policies = checkpoint_map(args.probe_checkpoint), checkpoint_map(args.policy_checkpoint)
    if not probes or set(probes) != set(policies):
        raise ValueError("Probe and policy checkpoint modes must match exactly")
    cache = args.cache.resolve(strict=True)
    cache_path = cache / "manifest.json"
    metadata = json.loads(cache_path.read_text())
    if not metadata.get("complete") or metadata.get("native_tactile_report", {}).get("velocity_dt") != .01:
        raise ValueError("Require completed, repaired native tactile cache with physical dt=.01")
    cache_hash = digest(cache_path)
    known_weights = metadata.get("configuration", {}).get("weights", {}).get("sha256")
    if known_weights is not None and digest(args.weights) != known_weights:
        raise ValueError("Live VGGT weights differ from the source cache")
    representation, split_links = representation_split(probes, args.representation_manifest)
    episode, roles = choose_source(metadata, representation, args.episode)
    signatures = [validate_policy_inputs(mode, policies[mode], probes[mode], cache_hash, metadata["normalization"]) for mode in policies]
    if any(item != signatures[0] for item in signatures[1:]):
        raise ValueError("Paired ACT policies must have the same execution horizon")
    all_cases, selected = make_cases(episode, args.seed, args.case_limit)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {"full_grid_cases": all_cases, "selected_case_ids": [case["case_id"] for case in selected],
            "case_limit": args.case_limit, "seed": args.seed, "source_split_roles": roles,
            "representation_training_links": split_links, "grid_span_m": [-.035, .035],
            "ordering": "four corners first, then remaining row-major grid; never selected by outcomes",
            "interpretation": "36 paired reset interventions from one source; grid cases are not independent original episodes"}
    plan_path = output / "case_plan.json"
    config = {"protocol": "new36_xy_grid_live_token_act_v1", "case_plan": plan,
              "cache_manifest_sha256": cache_hash, "control_ticks": 600, "physics_dt": .01,
              "substeps": 1, "sensor_velocity_dt": .01, "sim_horizon_s": 6.,
              "modes": {mode: {"probe": str(probes[mode]), "probe_sha256": digest(probes[mode]),
                                  "policy": str(policies[mode]), "policy_sha256": digest(policies[mode] / "model.pt")}
                        for mode in policies}, "execution_horizon": signatures[0],
              "repository": str(args.repository.resolve()), "weights_sha256": digest(args.weights),
              "device": args.device, "amp": args.amp, "trace_every": args.trace_every,
              "video_cases": args.video_cases, "checkpoint_selection": args.selection_note,
              "worker": {key: str(getattr(args, key)) for key in ("worker_python", "dynamic_root", "raw_root", "timestamps_root", "source_readouts")},
              "implementations": {str(path.relative_to(ROOT)): digest(path) for path in
                   (Path(__file__), ROOT / "scripts/token_runtime.py", ROOT / "scripts/sim_worker.py",
                    ROOT / "src/tactile_vggt/token_fusion.py", ROOT / "src/tactile_vggt/token_policy.py",
                    ROOT / "src/tactile_vggt/simulation.py")},
              "metrics": {"primary": "worker strict_stable_goal at final tick600; sustained last1s with low velocities and no robot contact",
                          "drop": "after lifting, cube center below .05m outside goal region",
                          "distance": "final XY distance to goal center (.5,-.3) and goal rectangle, metres"},
              "time_semantics": "simulation pauses for inference; wall latency reported, no hard-real-time claim",
              "uncertainty": "Wilson and paired bootstrap describe these grid cases, not independent source episodes or training-seed replication",
              "gpu_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")}}
    config_path = output / "eval_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Output protocol exists/changed; use --resume unchanged or a fresh output")
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Saved case plan differs from requested plan")
    else:
        if args.resume or plan_path.exists():
            raise ValueError("Incomplete evaluation configuration; use a fresh output")
        write_json(plan_path, plan)  # Both records precede model/worker initialization.
        write_json(config_path, config)
    config_hash = digest(config_path)

    import torch
    from token_runtime import LiveTokenEncoder
    from tactile_vggt.token_policy import TokenACTInference
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    def synchronize():
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
    worker_command = [str(args.worker_python), str(ROOT / "scripts/sim_worker.py"),
                      "--ipc-dir", str(output / "ipc"), "--dynamic-root", str(args.dynamic_root),
                      "--raw-root", str(args.raw_root), "--timestamps-root", str(args.timestamps_root),
                      "--source-readouts", str(args.source_readouts), "--device", args.device,
                      "--substeps", "1", "--sensor-dt", ".01"]
    client, models = None, {}
    started = time.monotonic()
    try:
        encoder = LiveTokenEncoder(args.repository, args.weights, probes, device=args.device)
        client = SimulationClient(worker_command, output / "worker.log", args.worker_timeout)
        for mode, checkpoint in policies.items():
            policy = TokenACTInference(checkpoint, device=args.device, amp=args.amp)
            mode_output = output / mode
            mode_output.mkdir(exist_ok=True)
            results = []
            for case_number, case in enumerate(selected):
                directory = mode_output / case["case_id"]
                directory.mkdir(exist_ok=True)
                result_path = directory / "result.json"
                if args.resume and result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if (not previous.get("complete") or previous.get("config_sha256") != config_hash or
                            previous.get("case") != case or previous.get("mode") != mode or
                            not (directory / "trajectory.npz").is_file()):
                        raise ValueError("Saved case is incomplete or belongs to a different protocol")
                    results.append(previous)
                    continue
                policy.reset()
                reply = client.request("reset", **{key: value for key, value in case.items() if key != "case_id"})
                observation, initial_info = read_observation(reply), reply["info"]
                provenance = reply.get("provenance", {})
                actions, cubes, states, state_ticks, lifted, inside = [], [], [], [], [], []
                refresh_times, policy_times, step_times, observation_times = [], [], [], []
                case_started = time.monotonic()
                video = None
                if case_number < args.video_cases:
                    import cv2
                    video = cv2.VideoWriter(str(directory / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            1 / (.01 * policy.policy.config["n_action_steps"]), (448, 224))
                    if not video.isOpened():
                        raise RuntimeError("Video writer failed")
                try:
                    with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                        for tick in range(600):
                            refreshed = policy.needs_observation
                            if refreshed:
                                if observation is None:
                                    begin = time.monotonic()
                                    observation = read_observation(client.request("observe"))
                                    observation_times.append(time.monotonic()-begin)
                                synchronize()
                                begin = time.monotonic()
                                features = encoder.encode_observation(observation)[mode]
                                synchronize()
                                refresh_times.append(time.monotonic()-begin)
                                if np.asarray(features).shape != (256,) or not np.isfinite(features).all():
                                    raise ValueError("Live representation must be a finite 256D vector")
                                if video:
                                    video.write(np.concatenate(observation["rgb"], axis=1)[..., ::-1])
                                states.append(observation["state"])
                                state_ticks.append(tick)
                                begin = time.monotonic()
                                action = policy.select_action(observation["state"], features).cpu().numpy()
                            else:
                                begin = time.monotonic()
                                action = policy.select_action().cpu().numpy()
                            synchronize()
                            policy_times.append(time.monotonic()-begin)
                            if action.shape != (22,) or not np.isfinite(action).all():
                                raise ValueError("Invalid ACT action")
                            begin = time.monotonic()
                            reply = client.request("step", action=action.tolist(), observe=False)
                            step_times.append(time.monotonic()-begin)
                            observation = None
                            info = reply["info"]
                            actions.append(action)
                            cubes.append(info["cube_center"])
                            lifted.append(info["lifted"])
                            inside.append(info["inside_goal"])
                            if refreshed or (tick+1) % args.trace_every == 0 or tick == 599:
                                trace.write(json.dumps({"tick": tick+1, "sim_time": reply["sim_time"],
                                      "fresh_policy_query": refreshed, "info": info}, allow_nan=False)+"\n")
                            if (tick+1) % 100 == 0:
                                print(json.dumps({"mode": mode, "case": case["case_id"], "tick": tick+1,
                                      "strict_stable_goal": info["strict_stable_goal"], "elapsed_s": time.monotonic()-case_started}), flush=True)
                finally:
                    if video:
                        video.release()
                if not np.isclose(reply["sim_time"], 6., rtol=0, atol=1e-8):
                    raise ValueError("Worker physical horizon differs from 600 * .01 seconds")
                np.savez_compressed(directory / "trajectory.npz", actions=np.asarray(actions), cube_centers=np.asarray(cubes),
                     refresh_states=np.asarray(states), refresh_ticks=np.asarray(state_ticks),
                     lifted=np.asarray(lifted), inside_goal=np.asarray(inside))
                result = {"complete": True, "config_sha256": config_hash, "mode": mode, "case": case,
                          "ticks": 600, "sim_time_s": float(reply["sim_time"]),
                          "wall_seconds": time.monotonic()-case_started,
                          "initial_info": initial_info, "final_info": reply["info"], "worker_provenance": provenance,
                          "outcome": score_trajectory(initial_info, reply["info"], cubes, lifted, inside),
                          "encoder_refresh": timing_stats(refresh_times), "policy_query_and_queue": timing_stats(policy_times),
                          "observation_ipc_render_touch": timing_stats(observation_times), "simulation_step": timing_stats(step_times),
                          "refresh_timing_scope": "shared live VGGT plus all configured probes; excludes worker observation/render/touch IPC",
                          "video": str(directory / "rollout.mp4") if case_number < args.video_cases else None}
                write_json(result_path, result)
                results.append(result)
                write_json(mode_output / "summary.json", summarize_mode(results))
                print(json.dumps({"event": "case_complete", "mode": mode, "case": case["case_id"], "outcome": result["outcome"]}), flush=True)
            models[mode] = summarize_mode(results)
            write_json(output / "summary.json", {"complete": len(models) == len(policies), "protocol": config["protocol"],
                 "case_plan_sha256": digest(plan_path), "config_sha256": config_hash, "source_split_roles": roles,
                 "models": models, "paired_comparisons": paired_comparisons(models),
                 "elapsed_s": time.monotonic()-started,
                 "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                 "scope": config["uncertainty"], "time_semantics": config["time_semantics"]})
            del policy
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if client is not None:
            client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--probe-checkpoint", required=True, action="append", help="mode=best.pt")
    result.add_argument("--policy-checkpoint", required=True, action="append", help="mode=ACT-checkpoint-directory")
    result.add_argument("--cache", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--representation-manifest", type=Path)
    result.add_argument("--episode", type=int, help="Default: prefer episode1 in ACT-val / representation-test overlap")
    result.add_argument("--repository", type=Path, default=ROOT / "vendor/vggt")
    result.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--amp", choices=["bfloat16", "float32"], default="bfloat16")
    result.add_argument("--worker-python", type=Path, default=Path("/home/wukf/miniforge3/envs/te/bin/python"))
    result.add_argument("--dynamic-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921"))
    result.add_argument("--raw-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212"))
    result.add_argument("--timestamps-root", type=Path, default=ROOT / "inputs/raw_timestamps")
    result.add_argument("--source-readouts", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz"))
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--case-limit", type=int, help="Predetermined prefix for smoke testing; use fresh output for all36")
    result.add_argument("--trace-every", type=int, default=20)
    result.add_argument("--video-cases", type=int, choices=[0, 1, 2], default=0)
    result.add_argument("--worker-timeout", type=float, default=120.)
    result.add_argument("--selection-note", default="Explicit checkpoints selected before closed-loop outcomes")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.trace_every < 1 or arguments.worker_timeout <= 0:
        raise ValueError("Positive trace interval and worker timeout required")
    try:
        run(arguments)
    except Exception as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
