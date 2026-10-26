#!/usr/bin/env python3
"""Actor-training nominal execution sanity check, separate from held-out tests.

Preselected episodes 0/10/20/30/40 are reset to their own recorded frame0.
Historical source commands and the best frozen_zero/frozen_pressure policies
each execute 600 physical ticks. These 15 diagnostics never enter test203.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot/src"), str(ROOT / "scripts")]
from evaluate_sim import SimulationClient, digest, read_observation, timing_stats, write_json
from evaluate_token_act import checkpoint_map, score_trajectory, wilson_interval
from evaluate_token_replay import load_replay_source, validate_nominal_source
from evaluate_simple_token_act import (safe_observation, pressure_diagnostics, object_contact_count,
    contact_summary, validate_cache, validate_worker, validate_simple_policy, create_policy, action_numpy)

EPISODES = (0, 10, 20, 30, 40)
POLICY_MODES = ("frozen_zero", "frozen_pressure")
SCOPE = "Actor-training episode nominal execution diagnostic; not held-out generalization, not part of the original 203 test cases"


def make_cases(metadata, seed=43, case_limit=None):
    train = set(map(int, metadata["split"]["train"]))
    val = set(map(int, metadata["split"]["val"]))
    if not set(EPISODES) <= train or set(EPISODES) & val:
        raise ValueError("All five preselected episodes must belong to actor training only")
    cases = [{"case_id": f"train_ep{episode:03d}_nominal", "episode_id": episode,
              "frame": 0, "seed": seed + episode, "object_xy_offset": [0., 0.],
              "actor_split": "train"} for episode in EPISODES]
    if case_limit is not None and not 1 <= case_limit <= len(cases):
        raise ValueError("case-limit must be between 1 and 5")
    return cases, cases if case_limit is None else cases[:case_limit]


def summarize_models(models):
    result = {}
    for mode, rows in models.items():
        success = sum(row["outcome"]["strict_stable_goal"] for row in rows)
        count = len(rows)
        result[mode] = {"cases": count, "actor_split": "train", "excluded_from_test203": True,
            "strict_stable_goal_count": success, "strict_stable_goal_rate": success / count if count else None,
            "strict_stable_goal_wilson_ci95": wilson_interval(success, count),
            "lifted_count": sum(row["outcome"]["ever_lifted"] for row in rows),
            "lift_then_drop_count": sum(row["outcome"]["lift_then_drop_outside_goal"] for row in rows),
            "ever_object_contact_count": sum(row["object_contact_scoring_only"]["first_contact_tick"] is not None for row in rows),
            "mean_final_goal_center_xy_distance_m": float(np.mean([row["outcome"]["final_goal_center_xy_distance_m"] for row in rows])) if rows else None,
            "results": rows}
    return result


def prepare_sources(args):
    metadata = json.loads((args.cache / "manifest.json").read_text())
    validate_cache(metadata)
    full, selected = make_cases(metadata, args.seed, args.case_limit)
    sources, schedules = {}, {}
    for case in selected:
        episode = case["episode_id"]
        _, arrays, rows, frames, commands = load_replay_source(args.cache, episode)
        nominal = validate_nominal_source(SimpleNamespace(**{**vars(args), "episode": episode}), metadata, arrays, rows)
        nominal["recorded_endpoints_scoring_only"] = recorded_endpoint_audit(args.raw_root, nominal)
        sources[str(episode)] = nominal
        schedules[episode] = {"commands": commands, "source_frames": frames, "source_global_rows": rows[frames],
                              "held_last_command_ticks": max(0, 600-len(rows))}
    return metadata, full, selected, sources, schedules


def recorded_endpoint_audit(raw_root, nominal):
    """Source evidence only: demos may never have targeted this goal rectangle."""
    with Path(nominal["timestamps_path"]).open() as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Recorded source episode is empty")
    raw = Path(raw_root) / nominal["raw_episode"]
    first = raw / records[0]["prioception_path"]
    last = raw / records[-1]["prioception_path"]
    initial, terminal = np.load(first, allow_pickle=False), np.load(last, allow_pickle=False)
    if initial.shape != (29,) or terminal.shape != (29,) or not np.isfinite(initial).all() or not np.isfinite(terminal).all():
        raise ValueError("Recorded endpoints must be finite qpos29")
    xyz = terminal[22:25]
    legacy_goal = bool(.4 <= xyz[0] <= .6 and -.39 <= xyz[1] <= -.21 and xyz[2] <= .1)
    return {"initial_cube_xyz": initial[22:25].tolist(), "terminal_cube_xyz": xyz.tolist(),
            "terminal_legacy_goal_predicate": legacy_goal, "terminal_qpos_path": str(last.resolve()),
            "terminal_qpos_sha256": digest(last),
            "scope": "Recorded endpoint/source audit only, not an actor input and not evidence of strict stable release; fixed episodes are not selected by this value"}


def make_runtime(args):
    from simple_token_runtime import SimpleVisualRuntime
    return SimpleVisualRuntime(args.repository, args.weights, args.visual_probe_checkpoint, args.device)


def run(args):
    args.cache = args.cache.resolve(strict=True)
    args.visual_probe_checkpoint = args.visual_probe_checkpoint.resolve(strict=True)
    policies = checkpoint_map(args.policy_checkpoint)
    if set(policies) != set(POLICY_MODES):
        raise ValueError("Require exactly frozen_zero and frozen_pressure best checkpoints")
    metadata, full, selected, sources, schedules = prepare_sources(args)
    cache_hash = digest(args.cache / "manifest.json")
    expected_weights = metadata.get("configuration", {}).get("weights", {}).get("sha256")
    weights_hash = digest(args.weights)
    if expected_weights is not None and weights_hash != expected_weights:
        raise ValueError("Live weights differ from original cache")
    args.kind, args.n_action_steps = "simple", 8
    signatures = [validate_simple_policy(mode, path, args.visual_probe_checkpoint, cache_hash,
        metadata["normalization"], digest(args.cache / "obs_frame_indices.npy"), metadata["action_semantics"])
        for mode, path in policies.items()]
    if any(signature != signatures[0] for signature in signatures[1:]) or signatures[0]["chunk_size"] < 8:
        raise ValueError("Both policies must share the declared action-chunk protocol")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "simple_actor_train_nominal_sanity_v1", "scope": SCOPE,
        "excluded_from_test203": True, "actor_split": "train", "preselected_episodes": list(EPISODES),
        "all_cases": full, "selected_cases": selected, "case_limit": args.case_limit,
        "modes": ["replay", *POLICY_MODES], "expected_diagnostic_trials": len(selected) * 3,
        "control_ticks": 600, "physics_dt": .01, "sensor_velocity_dt": .01, "n_action_steps": 8,
        "action_semantics": metadata["action_semantics"], "replay_schedule": "source action[s] at each tick; hold final recorded command after episode end; no temporal shift",
        "sources": sources, "cache_manifest_sha256": cache_hash, "weights_sha256": weights_hash,
        "visual_probe": {"path": str(args.visual_probe_checkpoint), "sha256": digest(args.visual_probe_checkpoint)},
        "policies": {mode: {"path": str(path), "model_sha256": digest(path / "model.pt"),
                            "train_config_sha256": digest(path / "train_config.json")} for mode, path in policies.items()},
        "worker": {name: str(getattr(args, name)) for name in ("worker_python", "dynamic_root", "raw_root", "timestamps_root", "source_readouts", "device")},
        "amp": args.amp, "seed": args.seed, "trace_every": args.trace_every,
        "implementations": {str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__), ROOT / "scripts/evaluate_simple_token_act.py", ROOT / "scripts/evaluate_token_replay.py")},
        "gpu_environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")},
        "time_semantics": "simulation pauses for policy inference; replay is an open-loop protocol control, not an oracle"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Diagnostic manifest changed; use unchanged --resume or a new output")
    else:
        if args.resume:
            raise ValueError("Cannot resume without a diagnostic manifest")
        write_json(manifest_path, manifest)
    manifest_hash = digest(manifest_path)
    if args.device.startswith("cuda"):
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        synchronize = lambda: torch.cuda.synchronize(args.device)
    else:
        synchronize = lambda: None
    command = [str(args.worker_python), str(ROOT / "scripts/sim_worker.py"), "--ipc-dir", str(output / "ipc"),
        "--dynamic-root", str(args.dynamic_root), "--raw-root", str(args.raw_root), "--timestamps-root", str(args.timestamps_root),
        "--source-readouts", str(args.source_readouts), "--device", args.device, "--substeps", "1", "--sensor-dt", ".01"]
    models, encoder, client = {}, None, None
    started = time.monotonic()

    def save_summary():
        complete = len(models) == 3 and all(len(rows) == len(selected) for rows in models.values())
        write_json(output / "summary.json", {"complete": complete and args.case_limit is None,
            "selected_cases_complete": complete, "protocol": manifest["protocol"], "scope": SCOPE,
            "actor_split": "train", "excluded_from_test203": True, "manifest_sha256": manifest_hash,
            "recorded_source_endpoints_scoring_only": {episode: source.get("recorded_endpoints_scoring_only") for episode, source in sources.items()},
            "replay_interpretation": "Protocol reference, not automatically a positive control; recorded demos need not target the fixed goal rectangle. Report actual replay outcomes per episode.",
            "models": summarize_models(models), "elapsed_s": time.monotonic() - started,
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})

    try:
        client = SimulationClient(command, output / "worker.log", args.worker_timeout)
        for mode in ("replay", *POLICY_MODES):
            policy = None
            if mode != "replay":
                if encoder is None:
                    encoder = make_runtime(args)
                policy = create_policy(args, policies[mode])
            results = []
            models[mode] = results
            for case in selected:
                directory = output / mode / case["case_id"]
                directory.mkdir(parents=True, exist_ok=True)
                result_path = directory / "result.json"
                if args.resume and result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if not previous.get("complete") or previous.get("manifest_sha256") != manifest_hash or previous.get("case") != case or previous.get("mode") != mode or not (directory / "trajectory.npz").exists():
                        raise ValueError("Saved train diagnostic belongs to a different/incomplete protocol")
                    results.append(previous)
                    continue
                replay = mode == "replay"
                if policy is not None:
                    policy.reset()
                reply = client.request("reset", **{name: case[name] for name in ("episode_id", "frame", "seed", "object_xy_offset")}, observe=not replay)
                initial = reply["info"]
                initialization = initial["initialization"]
                source = sources[str(case["episode_id"])]
                if initialization["episode_id"] != case["episode_id"] or initialization["frame"] != 0 or initialization["qpos_sha256"] != source["qpos_sha256"]:
                    raise ValueError("Worker reset differs from the preflighted actor-training source")
                validate_worker(reply["provenance"], metadata)
                provenance = reply["provenance"]
                initial_contacts = object_contact_count(initial)
                observation = None if replay else safe_observation(read_observation(reply))
                actions, cubes, lifted, inside, contacts, ticks, pressures = [], [], [], [], [], [], []
                refresh_seconds, step_seconds = [], []
                case_started = time.monotonic()
                with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                    for tick in range(600):
                        diagnostic, refreshed = None, False
                        if replay:
                            action = schedules[case["episode_id"]]["commands"][tick]
                        elif policy.needs_observation:
                            refreshed = True
                            if observation is None:
                                observation = safe_observation(read_observation(client.request("observe")))
                            diagnostic = pressure_diagnostics(observation["tactile"], tick)
                            synchronize()
                            begin = time.monotonic()
                            readout = encoder.encode_observation(observation)
                            synchronize()
                            refresh_seconds.append(time.monotonic() - begin)
                            ticks.append(tick)
                            pressures.append(observation["tactile"])
                            action = policy.select_action(observation["state"], readout, observation["tactile"])
                        else:
                            action = policy.select_action()
                        action = action_numpy(action)
                        begin = time.monotonic()
                        reply = client.request("step", action=action.tolist(), observe=False)
                        step_seconds.append(time.monotonic() - begin)
                        info, observation = reply["info"], None
                        if info["control_tick"] != tick + 1:
                            raise ValueError("Worker did not advance exactly one physical command tick")
                        actions.append(action); cubes.append(info["cube_center"])
                        lifted.append(info["lifted"]); inside.append(info["inside_goal"])
                        contacts.append(object_contact_count(info))
                        if refreshed or (tick + 1) % args.trace_every == 0 or tick == 599:
                            trace.write(json.dumps({"tick": tick + 1, "sim_time": reply["sim_time"],
                                "pressure_observation": diagnostic, "info_scoring_only": info}, allow_nan=False) + "\n")
                if not np.isclose(reply["sim_time"], 6., atol=1e-8, rtol=0):
                    raise ValueError("Train diagnostic horizon differs from 600*.01s")
                extra = {"source_frames": schedules[case["episode_id"]]["source_frames"],
                         "source_global_rows": schedules[case["episode_id"]]["source_global_rows"]} if replay else {}
                np.savez_compressed(directory / "trajectory.npz", actions=np.asarray(actions), cube_centers=np.asarray(cubes),
                    lifted=np.asarray(lifted), inside_goal=np.asarray(inside), robot_cube_contact_count=np.asarray(contacts),
                    refresh_ticks=np.asarray(ticks), raw_pressure=np.asarray(pressures).reshape(-1,44), **extra)
                result = {"complete": True, "mode": mode, "case": case, "scope": SCOPE,
                    "actor_split": "train", "excluded_from_test203": True, "manifest_sha256": manifest_hash,
                    "ticks": 600, "sim_time_s": float(reply["sim_time"]), "initial_info": initial, "final_info": info,
                    "worker_provenance": provenance, "outcome": score_trajectory(initial, info, cubes, lifted, inside),
                    "object_contact_scoring_only": contact_summary(initial_contacts, contacts),
                    "encoder_refresh": timing_stats(refresh_seconds), "simulation_step": timing_stats(step_seconds),
                    "wall_seconds": time.monotonic() - case_started,
                    "held_last_command_ticks": schedules[case["episode_id"]]["held_last_command_ticks"] if replay else None}
                write_json(result_path, result)
                results.append(result)
                save_summary()
                print(json.dumps({"event":"train_diagnostic_complete","mode":mode,"episode":case["episode_id"],"outcome":result["outcome"]}), flush=True)
            if policy is not None:
                del policy
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        save_summary()
    finally:
        if client is not None:
            client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--policy-checkpoint", action="append", required=True, help="frozen_zero=bestdir and frozen_pressure=bestdir")
    result.add_argument("--visual-probe-checkpoint", type=Path, required=True)
    result.add_argument("--repository", type=Path, default=ROOT / "vendor/vggt")
    result.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    result.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    result.add_argument("--amp", choices=("bfloat16", "float32"), default="bfloat16")
    result.add_argument("--worker-python", type=Path, default=Path("/home/wukf/miniforge3/envs/te/bin/python"))
    result.add_argument("--dynamic-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921"))
    result.add_argument("--raw-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212"))
    result.add_argument("--timestamps-root", type=Path, default=ROOT / "inputs/raw_timestamps")
    result.add_argument("--source-readouts", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz"))
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--trace-every", type=int, default=20)
    result.add_argument("--worker-timeout", type=float, default=120.)
    result.add_argument("--case-limit", type=int, help="Smoke prefix of the five preselected source episodes")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.trace_every < 1 or arguments.worker_timeout <= 0:
        raise ValueError("Positive trace interval and timeout required")
    try:
        run(arguments)
    except Exception as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"error":str(error),"traceback":traceback.format_exc()})
        raise
