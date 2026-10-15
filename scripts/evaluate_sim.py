#!/usr/bin/env python3
"""Paired closed-loop evaluation with live frozen VGGT and native tactile IPC.

Simulation time pauses for model inference. This measures control effectiveness,
not hard real-time deployment: wall-time latency is reported separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot/src"), str(ROOT / "scripts")]
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class SimulationClient:
    """One request at a time; native dependencies stay in their own interpreter."""
    def __init__(self, command, stderr_path, timeout=120):
        self.stderr = Path(stderr_path).open("a", buffering=1)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1, cwd=ROOT)
        self.timeout = timeout

    def request(self, command, **fields):
        if self.process.poll() is not None:
            raise RuntimeError(f"Simulation worker exited: {self.process.returncode}")
        self.process.stdin.write(json.dumps(dict(command=command, **fields), allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError(f"Simulation worker did not answer {command} within {self.timeout}s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Simulation worker closed stdout during {command}; inspect worker.log")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Non-protocol worker output: {line[:400]}") from error
        if not reply.get("ok"):
            raise RuntimeError(f"Simulation worker failed: {reply}")
        return reply

    def close(self):
        try:
            if self.process.poll() is None:
                self.request("close")
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.stderr.close()


def read_observation(reply):
    with np.load(reply["observation_path"], allow_pickle=False) as values:
        obs = {key: values[key].copy() for key in values.files}
    shapes = {"rgb": (2, 224, 224, 3), "state": (22,), "tactile": (44,),
              "anchors_world": (44, 3), "known_cameras_cv": (2, 4, 4), "base_to_world": (4, 4)}
    for name, shape in shapes.items():
        if obs[name].shape != shape or not np.isfinite(obs[name]).all():
            raise ValueError(f"Invalid synchronized observation {name}: {obs[name].shape}")
    if obs["rgb"].dtype != np.uint8:
        raise ValueError("Live RGB must use the training uint8 square crop")
    return obs


def timing_stats(values):
    if not values:
        return None
    return {"count": len(values), "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)), "p95_s": float(np.quantile(values, .95)),
            "max_s": float(np.max(values))}


def summarize_cases(results):
    n = len(results)
    return {"episodes": n,
            "final_goal_count": sum(bool(x["final_info"]["inside_goal"]) for x in results),
            "final_goal_rate": float(np.mean([x["final_info"]["inside_goal"] for x in results])) if n else None,
            "strict_stable_goal_count": sum(bool(x["final_info"].get("strict_stable_goal", False)) for x in results),
            "lifted_count": sum(bool(x["final_info"].get("lifted", False)) for x in results),
            "episodes_results": results}


def run(args):
    import torch
    from prepare_cache import FrozenVGGT
    from tactile_vggt.geometry import calibrated_similarity, unproject_depth, build_point_tensor
    from tactile_vggt.inference import PointACTInference

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cache = Path(args.cache).resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Evaluation requires a completed training cache")
    expected_sensor_dt = manifest.get("native_tactile_report", {}).get("velocity_dt")
    if expected_sensor_dt is None:
        expected_sensor_dt = .05  # v1 wall-time proxy, approximated by nominal recording interval.
    if not np.isclose(args.sensor_dt, expected_sensor_dt, rtol=0, atol=1e-12):
        raise ValueError(f"sensor-dt must match trained tactile time basis: {expected_sensor_dt}")
    episodes = manifest["split"]["val"] if args.episodes is None else [int(x) for x in args.episodes.split(",")]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Need unique evaluation episode IDs")
    if not set(episodes) <= set(manifest["split"]["val"]):
        raise ValueError("This validation-reset protocol accepts held-out episode IDs only")
    checkpoints = [Path(path).resolve(strict=True) for path in args.checkpoint]
    weight_sha = digest(args.vggt_weights)
    if weight_sha != manifest["configuration"]["weights"]["sha256"]:
        raise ValueError("Live VGGT weights differ from training geometry weights")
    names = [json.loads((p / "config.json").read_text())["config"]["mode"] + "_" + p.name for p in checkpoints]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint mode/step names collide; evaluate these runs in separate outputs")
    config = {"protocol": "heldout_initial_states_recorded_physics_v1", "episodes": episodes,
              "steps": args.max_steps, "physics_dt": .01, "substeps": args.substeps,
              "control_dt": .01 * args.substeps, "seed": args.seed,
              "sensor_velocity_dt": args.sensor_dt,
              "source_wall_recording_fps": 20,
              "cache_manifest_sha256": digest(cache / "manifest.json"),
              "checkpoints": [{"path": str(p), "model_sha256": digest(p / "model.pt")} for p in checkpoints],
              "checkpoint_selection": args.selection_note,
              "worker": {"python": args.worker_python, "dynamic_root": args.dynamic_root,
                         "raw_root": args.raw_root, "timestamps_root": args.timestamps_root,
                         "source_readouts": args.source_readouts},
              "implementation": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (Path(__file__), ROOT / "scripts/sim_worker.py", ROOT / "src/tactile_vggt/simulation.py",
                                   ROOT / "src/tactile_vggt/inference.py", ROOT / "src/tactile_vggt/geometry.py")},
              "geometry": "live synchronized stereo RGB -> frozen VGGT; calibrated first-camera frame and stereo scale; no GT depth/object pose in policy",
              "time_semantics": "simulator pauses during inference; no hard-real-time claim",
              "primary_success": "final cube center x in [.40,.60], y in [-.39,-.21], z<=.10; inherited weak metric",
              "secondary_success": "strict stable goal from worker: sustained placement, low velocities, no robot contact",
              "rgb_preprocess": "480x640 center crop 480x480 then PIL LANCZOS 224x224",
              "image_size": manifest["configuration"]["image_size"],
              "vggt_repo": str(Path(args.vggt_repo).resolve()),
              "vggt_weights": {"path": str(Path(args.vggt_weights).resolve()), "sha256": weight_sha},
              "vggt_preparation_sha256": digest(ROOT / "scripts/prepare_cache.py"),
              "n_action_steps_override": args.n_action_steps}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "eval_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Evaluation output exists or its protocol changed; use a fresh directory")
    else:
        write_json(config_path, config)
    worker_command = [args.worker_python, str(ROOT / "scripts/sim_worker.py"),
                      "--ipc-dir", str(output / "ipc"), "--dynamic-root", args.dynamic_root,
                      "--raw-root", args.raw_root, "--timestamps-root", args.timestamps_root,
                      "--source-readouts", args.source_readouts, "--device", args.device,
                      "--substeps", str(args.substeps), "--sensor-dt", str(args.sensor_dt)]
    client = None
    completed = []
    started = time.monotonic()
    try:
        model = FrozenVGGT(args.vggt_repo, args.vggt_weights, args.device, config["image_size"])
        client = SimulationClient(worker_command, output / "worker.log", args.worker_timeout)
        for checkpoint in checkpoints:
            adapter = PointACTInference(checkpoint, device=args.device, amp="bfloat16")
            trained = json.loads((checkpoint / "train_config.json").read_text())
            if trained.get("data_manifest_sha256") != config["cache_manifest_sha256"]:
                raise ValueError("Checkpoint was trained on a different data manifest")
            if adapter.stats != manifest["normalization"]:
                raise ValueError("Checkpoint normalization is not the evaluated cache normalization")
            if args.n_action_steps is not None:
                if not 1 <= args.n_action_steps <= adapter.policy.config["chunk_size"]:
                    raise ValueError("Invalid execution-horizon override")
                adapter.policy.config["n_action_steps"] = args.n_action_steps
                adapter.policy.act.config.n_action_steps = args.n_action_steps
                from collections import deque
                adapter.policy.act._action_queue = deque([], maxlen=args.n_action_steps)
            mode = adapter.policy.config["mode"]
            model_name = f"{mode}_{checkpoint.name}"
            model_output = output / model_name
            model_output.mkdir(exist_ok=True)
            results = []
            for episode in episodes:
                case = model_output / f"episode_{episode:03d}"
                case.mkdir(exist_ok=True)
                result_path = case / "result.json"
                if args.resume and result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                adapter.reset()
                reply = client.request("reset", episode_id=episode, frame=0, seed=args.seed)
                observation = read_observation(reply)
                initial_info = reply["info"]
                worker_provenance = reply.get("provenance", {})
                rows, refresh_times, policy_times, step_times = [], [], [], []
                state_rows, action_rows, cube_rows = [], [], []
                episode_started = time.monotonic()
                video = None
                if args.video:
                    import cv2
                    video = cv2.VideoWriter(str(case / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            1 / (config["control_dt"] * adapter.policy.config["n_action_steps"]), (448, 224))
                    if not video.isOpened():
                        raise RuntimeError("Video encoder could not open rollout.mp4")
                try:
                    with (case / "steps.jsonl").open("w", buffering=1) as trace:
                        for tick in range(args.max_steps):
                            refreshed = adapter.needs_observation
                            if refreshed:
                                if observation is None:
                                    observation = read_observation(client.request("observe"))
                                begin = time.monotonic()
                                rgb, depth, confidence, extrinsics, intrinsics = model.predict([observation["rgb"]])
                                scale, rotation, translation, diagnostics = calibrated_similarity(extrinsics[0], observation["known_cameras_cv"])
                                predicted = unproject_depth(depth[0], extrinsics[0], intrinsics[0])
                                world = scale * predicted @ rotation.T + translation
                                points = build_point_tensor(world, rgb[0], confidence[0], observation["anchors_world"], observation["tactile"],
                                    base_to_world=observation["base_to_world"], visual_count=adapter.policy.config["n_visual_points"],
                                    confidence_quantile=manifest["configuration"]["confidence_quantile"], seed=args.seed + tick)
                                refresh_times.append(time.monotonic() - begin)
                                if video:
                                    video.write(np.concatenate(observation["rgb"], axis=1)[..., ::-1])
                                begin = time.monotonic()
                                action = adapter.select_action(observation["state"], observation["tactile"], points).cpu().numpy()
                                policy_times.append(time.monotonic() - begin)
                                state_rows.append(observation["state"])
                            else:
                                begin = time.monotonic()
                                action = adapter.select_action().cpu().numpy()
                                policy_times.append(time.monotonic() - begin)
                            if action.shape != (22,) or not np.isfinite(action).all():
                                raise ValueError("Invalid policy action")
                            begin = time.monotonic()
                            reply = client.request("step", action=action.tolist(), observe=False)
                            step_times.append(time.monotonic() - begin)
                            observation = None
                            info = reply["info"]
                            row = {"control_tick": tick + 1, "sim_time": reply["sim_time"],
                                   "fresh_policy_query": refreshed, "info": info}
                            if refreshed:
                                row["geometry_diagnostics"] = diagnostics
                            trace.write(json.dumps(row, allow_nan=False) + "\n")
                            rows.append(row)
                            action_rows.append(action)
                            cube_rows.append(info["cube_center"])
                            if (tick + 1) % 100 == 0:
                                print(json.dumps({"model": model_name, "episode": episode, "tick": tick + 1,
                                                  "inside_goal": info["inside_goal"], "cube_center": info["cube_center"],
                                                  "elapsed_s": time.monotonic() - episode_started}), flush=True)
                finally:
                    if video:
                        video.release()
                np.savez_compressed(case / "trajectory.npz", actions=np.array(action_rows), cube_center=np.array(cube_rows),
                                    refresh_states=np.array(state_rows))
                if not np.isclose(reply["sim_time"], args.max_steps * config["control_dt"], rtol=0, atol=1e-8):
                    raise ValueError("Simulator elapsed time does not match the declared control protocol")
                result = {"model": model_name, "checkpoint": str(checkpoint), "episode": episode,
                          "completed_control_steps": len(rows), "sim_duration_s": args.max_steps * config["control_dt"],
                          "actual_sim_time_s": reply["sim_time"],
                          "wall_seconds": time.monotonic() - episode_started, "initial_info": initial_info,
                          "worker_provenance": worker_provenance,
                          "final_info": reply["info"], "geometry_refresh": timing_stats(refresh_times),
                          "policy_query_and_queue": timing_stats(policy_times), "simulation_step": timing_stats(step_times),
                          "geometry_refresh_excludes": "worker observation IPC, rendering and native tactile; full episode wall_seconds includes these",
                          "video": str(case / "rollout.mp4") if args.video else None}
                write_json(result_path, result)
                results.append(result)
                write_json(model_output / "summary.json", summarize_cases(results))
                print(json.dumps({"event": "episode_complete", **result}), flush=True)
            summary = {"model": model_name, "checkpoint": str(checkpoint), **summarize_cases(results)}
            completed.append(summary)
            write_json(output / "summary.json", {"complete": len(completed) == len(checkpoints), "models": completed,
                       "elapsed_seconds": time.monotonic() - started,
                       "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
            del adapter
            torch.cuda.empty_cache()
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, action="append")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vggt-repo", default=str(ROOT / "vendor/vggt"))
    parser.add_argument("--vggt-weights", default=str(ROOT / "weights/model.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-python", default="/home/wukf/miniforge3/envs/te/bin/python")
    parser.add_argument("--dynamic-root", default="/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921")
    parser.add_argument("--raw-root", default="/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212")
    parser.add_argument("--timestamps-root", default=str(ROOT / "inputs/raw_timestamps"))
    parser.add_argument("--source-readouts", default="/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz")
    parser.add_argument("--episodes", help="Comma-separated validation episode IDs; default all held-out five")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--sensor-dt", type=float, default=.05,
                        help="Match training tactile proxy: v1=.05 wall dt; corrected physical v2=.01")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--worker-timeout", type=float, default=120)
    parser.add_argument("--selection-note", default="explicit checkpoint; never selected by closed-loop outcome")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.substeps not in (1, 5) or args.sensor_dt <= 0:
        parser.error("Positive steps/sensor-dt and one or five physics substeps required")
    try:
        run(args)
    except Exception as error:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
