#!/usr/bin/env python3
"""Replay historical commands as an open-loop protocol reference.

Nominal reset plus the same 36 XY interventions used by evaluate_token_act.
This is neither an adaptive oracle nor a learned-policy test. No RGB, tactile
observation, VGGT, or ACT forward pass is requested. Every command advances one
0.01-second physics tick; after the recorded episode ends its last command is
held until tick 600. Do not interpret grid failures as oracle failures.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_sim import SimulationClient, digest, timing_stats, write_json
from evaluate_token_act import make_cases, score_trajectory


SCOPE = ("Historical full-rate commands replayed open loop from recorded frame0; "
         "nominal protocol positive control, not an adaptive grid oracle or a learned-model test")


def replay_cases(episode=1, seed=43, case_limit=None):
    grid, _ = make_cases(episode, seed)
    nominal = {"case_id": "nominal", "episode_id": int(episode), "frame": 0,
               "seed": int(seed), "object_xy_offset": [0., 0.]}
    cases = [nominal, *grid]
    if case_limit is not None and not 1 <= case_limit <= len(cases):
        raise ValueError("case-limit must lie between 1 and 37")
    return cases, cases if case_limit is None else cases[:case_limit]


def load_replay_source(cache, episode):
    cache = Path(cache).resolve(strict=True)
    metadata = json.loads((cache / "manifest.json").read_text())
    if not metadata.get("complete") or metadata.get("native_tactile_report", {}).get("velocity_dt") != .01:
        raise ValueError("Replay requires a completed physical-dt=.01 cache")
    semantics = metadata.get("action_semantics")
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("Replay requires the original cache's recorded action_semantics")
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("action", "state", "episode_index", "frame_index")}
    n = len(arrays["episode_index"])
    if arrays["action"].shape != (n, 22) or arrays["state"].shape != (n, 22) or arrays["frame_index"].shape != (n,):
        raise ValueError("Expected full-rate action/state22 and one episode/frame index per row")
    if arrays["episode_index"].dtype.kind not in "iu" or arrays["frame_index"].dtype.kind not in "iu":
        raise ValueError("Episode and frame indices must be integers")
    rows = np.flatnonzero(arrays["episode_index"] == episode)
    if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
        raise ValueError("Requested episode must occupy one contiguous original-cache segment")
    if not np.array_equal(arrays["frame_index"][rows], np.arange(len(rows))):
        raise ValueError("Episode frames must be sorted, full-rate and start at frame0")
    actions = np.array(arrays["action"][rows], dtype=np.float64, copy=True)
    if not np.isfinite(actions).all() or not np.isfinite(arrays["state"][rows]).all():
        raise ValueError("Nonfinite source actions or states")
    source_frames = np.minimum(np.arange(600), len(rows) - 1)
    return metadata, arrays, rows, source_frames, actions[source_frames]


def read_parquet_actions(paths):
    """Use the same action column and concatenation order as prepare_cache."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.concat_tables([pq.read_table(path, columns=["action", "episode_index", "frame_index"])
                              for path in paths])
    return {"action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
            "episode_index": np.asarray(table["episode_index"].to_pylist()),
            "frame_index": np.asarray(table["frame_index"].to_pylist())}


def validate_source_actions(metadata, arrays, rows):
    entries = metadata.get("configuration", {}).get("source", {}).get("source_parquet")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Original action source requires manifest source_parquet provenance")
    provenance, paths = [], []
    for entry in entries:
        path = Path(entry["path"]).resolve(strict=True)
        actual = digest(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"Original action parquet checksum mismatch: {path}")
        paths.append(path)
        provenance.append({"path": str(path), "sha256": actual})
    source = read_parquet_actions(paths)
    for name in ("action", "episode_index", "frame_index"):
        if source[name].shape != arrays[name].shape:
            raise ValueError(f"Original parquet/cache dimensions differ: {name}")
        np.testing.assert_array_equal(source[name][rows], arrays[name][rows],
                                      err_msg=f"Original parquet/cache episode mismatch: {name}")
    return {"source_parquet": provenance, "action_column": "action",
            "comparison": "selected original episode action/episode/frame exactly equal cache; no shift applied"}


def validate_nominal_source(args, metadata, arrays, rows):
    """Bind worker frame0 to the cache and raw qpos before running a simulator."""
    episodes = sorted(path for path in args.raw_root.iterdir() if (path / "prioception").exists())
    if not 0 <= args.episode < len(episodes):
        raise ValueError("Replay episode does not index the worker's sorted raw episodes")
    raw = episodes[args.episode]
    timestamps = args.timestamps_root / f"teleop_log_{raw.name}.csv"
    with timestamps.open() as stream:
        records = list(csv.DictReader(stream))
    if len(records) != len(rows) or not records:
        raise ValueError("Raw episode length differs from its full-rate cached segment")
    qpos_path = (raw / records[0]["prioception_path"]).resolve(strict=True)
    qpos = np.load(qpos_path, allow_pickle=False)
    if qpos.shape != (29,) or not np.isfinite(qpos).all():
        raise ValueError("Nominal raw frame0 must contain finite qpos29")
    np.testing.assert_array_equal(qpos[:22].astype(np.float32), arrays["state"][rows[0]],
                                  err_msg="Nominal raw frame0 differs from original cached state")
    with np.load(args.source_readouts, allow_pickle=False) as readouts:
        schema = {key: {"shape": list(readouts[key].shape), "dtype": str(readouts[key].dtype)}
                  for key in readouts.files}
        for name, source_name in (("state", "state22"), ("episode_index", "episode_index"), ("frame_index", "frame_index")):
            if source_name not in readouts.files:
                raise ValueError(f"Worker readouts lack required field: {source_name}; actual keys={readouts.files}")
            if readouts[source_name].shape != arrays[name].shape:
                raise ValueError("Worker source readouts differ from original cache dimensions")
            np.testing.assert_array_equal(readouts[source_name][rows], arrays[name][rows],
                                          err_msg=f"Worker source readout mismatch: {source_name}")
    action_source = validate_source_actions(metadata, arrays, rows)
    return {"episode_id": args.episode, "frame": 0, "raw_episode": raw.name,
            "qpos_path": str(qpos_path), "qpos_sha256": digest(qpos_path),
            "timestamps_path": str(timestamps.resolve()), "timestamps_sha256": digest(timestamps),
            "source_readouts_sha256": digest(args.source_readouts),
            "source_readout_schema": schema, "action_source": action_source,
            "cached_global_row": int(rows[0]), "original_episode_frames": len(rows)}


def replay_summary(results, selected_count, config_hash, started):
    grid = [row for row in results if row["case"]["case_id"] != "nominal"]
    count = len(grid)
    success = sum(row["outcome"]["strict_stable_goal"] for row in grid)
    nominal = next((row for row in results if row["case"]["case_id"] == "nominal"), None)
    return {"complete": len(results) == selected_count,
            "protocol": "historical_open_loop_command_replay_v1", "scope": SCOPE,
            "nominal": nominal,
            "nominal_strict_stable_goal": nominal["outcome"]["strict_stable_goal"] if nominal else None,
            "grid": {"cases": count, "strict_stable_goal_count": success,
                     "strict_stable_goal_rate": success / count if count else None,
                     "lifted_count": sum(row["outcome"]["ever_lifted"] for row in grid),
                     "lift_then_drop_count": sum(row["outcome"]["lift_then_drop_outside_goal"] for row in grid),
                     "mean_final_goal_center_xy_distance_m": float(np.mean([row["outcome"]["final_goal_center_xy_distance_m"] for row in grid])) if count else None,
                     "mean_final_goal_region_xy_distance_m": float(np.mean([row["outcome"]["final_goal_region_xy_distance_m"] for row in grid])) if count else None},
            "results": results, "config_sha256": config_hash, "elapsed_s": time.monotonic() - started,
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}


def run(args):
    if args.trace_every < 1 or args.worker_timeout <= 0:
        raise ValueError("Positive trace interval and worker timeout required")
    metadata, arrays, rows, source_frames, commands = load_replay_source(args.cache, args.episode)
    nominal_source = validate_nominal_source(args, metadata, arrays, rows)
    all_cases, selected = replay_cases(args.episode, args.seed, args.case_limit)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {"all_cases": all_cases, "selected_case_ids": [case["case_id"] for case in selected],
            "ordering": "nominal first, then the unchanged evaluate_token_act 36grid order",
            "source": nominal_source, "scope": SCOPE}
    config = {"protocol": "historical_open_loop_command_replay_v1", "scope": SCOPE, "case_plan": plan,
              "cache": str(args.cache.resolve()), "cache_manifest_sha256": digest(args.cache / "manifest.json"),
              "array_sha256": {name: digest(args.cache / f"{name}.npy")
                               for name in ("action", "state", "episode_index", "frame_index")},
              "episode_action_values_sha256": hashlib.sha256(np.asarray(arrays["action"][rows]).tobytes()).hexdigest(),
              "original_episode_frames": len(rows), "recorded_commands_used": min(600, len(rows)),
              "held_last_command_ticks": max(0, 600 - len(rows)),
              "command_schedule": "frame[tick] for tick<episode_length, then hold last; full-rate absolute22 commands",
              "action_semantics": metadata["action_semantics"],
              "control_ticks": 600, "physics_dt": .01, "substeps": 1, "sim_horizon_s": 6.,
              "worker": {name: str(getattr(args, name)) for name in
                         ("worker_python", "dynamic_root", "raw_root", "timestamps_root", "source_readouts", "device")},
              "trace_every": args.trace_every,
              "implementations": {str(path.relative_to(ROOT)): digest(path) for path in
                                  (Path(__file__), ROOT / "scripts/evaluate_token_act.py", ROOT / "scripts/sim_worker.py",
                                   ROOT / "src/tactile_vggt/simulation.py")},
              "observations": "reset and step both observe=False; no RGB/tactile/model forward",
              "interpretation": "Nominal success checks replay/protocol feasibility; shifted-grid results measure fixed historical-command robustness only"}
    config_path, plan_path = output / "eval_config.json", output / "case_plan.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config or not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing replay protocol differs; use unchanged --resume or a fresh output")
    else:
        if args.resume or plan_path.exists():
            raise ValueError("Incomplete replay configuration; use a fresh output")
        write_json(plan_path, plan)
        write_json(config_path, config)
    config_hash = digest(config_path)
    worker_command = [str(args.worker_python), str(ROOT / "scripts/sim_worker.py"),
                      "--ipc-dir", str(output / "ipc"), "--dynamic-root", str(args.dynamic_root),
                      "--raw-root", str(args.raw_root), "--timestamps-root", str(args.timestamps_root),
                      "--source-readouts", str(args.source_readouts), "--device", args.device,
                      "--substeps", "1", "--sensor-dt", ".01"]
    client, results = None, []
    nominal_center = None
    started = time.monotonic()
    try:
        client = SimulationClient(worker_command, output / "worker.log", args.worker_timeout)
        for case in selected:
            directory = output / case["case_id"]
            directory.mkdir(exist_ok=True)
            result_path = directory / "result.json"
            if args.resume and result_path.exists():
                previous = json.loads(result_path.read_text())
                if (not previous.get("complete") or previous.get("config_sha256") != config_hash or
                        previous.get("case") != case or not (directory / "trajectory.npz").is_file()):
                    raise ValueError("Saved replay case is incomplete or belongs to another protocol")
                if case["case_id"] == "nominal":
                    nominal_center = np.asarray(previous["initial_info"]["cube_center"])
                results.append(previous)
                continue
            reply = client.request("reset", observe=False, **{key: value for key, value in case.items() if key != "case_id"})
            initial = reply["info"]
            initialization = initial["initialization"]
            if (initialization["episode_id"] != args.episode or initialization["frame"] != 0 or
                    initialization["qpos_sha256"] != nominal_source["qpos_sha256"]):
                raise ValueError("Worker reset differs from the validated nominal source")
            provenance = reply.get("provenance", {})
            if provenance.get("source_readouts_sha256") != nominal_source["source_readouts_sha256"]:
                raise ValueError("Worker readout source differs from the preflight source")
            if case["case_id"] == "nominal":
                nominal_center = np.asarray(initial["cube_center"])
            else:
                expected = nominal_center + np.r_[case["object_xy_offset"], 0.]
                np.testing.assert_allclose(initial["cube_center"], expected, rtol=0, atol=1e-8,
                                           err_msg="Grid reset does not preserve the nominal source plus declared XY offset")
            cubes, lifted, inside, step_seconds = [], [], [], []
            case_started = time.monotonic()
            with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                for tick, action in enumerate(commands):
                    begin = time.monotonic()
                    reply = client.request("step", action=action.tolist(), observe=False)
                    step_seconds.append(time.monotonic() - begin)
                    info = reply["info"]
                    if info["control_tick"] != tick + 1:
                        raise ValueError("Worker did not advance exactly one control tick per command")
                    cubes.append(info["cube_center"])
                    lifted.append(info["lifted"])
                    inside.append(info["inside_goal"])
                    if tick == 0 or (tick + 1) % args.trace_every == 0 or tick == 599:
                        trace.write(json.dumps({"tick": tick + 1, "source_frame": int(source_frames[tick]),
                                               "info": info}, allow_nan=False) + "\n")
                    if (tick + 1) % 100 == 0:
                        print(json.dumps({"case": case["case_id"], "tick": tick + 1,
                                          "strict_stable_goal": info["strict_stable_goal"]}), flush=True)
            if not np.isclose(reply["sim_time"], 6., rtol=0, atol=1e-8):
                raise ValueError("Replay horizon differs from 600 * .01 seconds")
            np.savez_compressed(directory / "trajectory.npz", actions=commands, source_frames=source_frames,
                                source_global_rows=rows[source_frames], cube_centers=np.asarray(cubes),
                                lifted=np.asarray(lifted), inside_goal=np.asarray(inside))
            result = {"complete": True, "config_sha256": config_hash, "case": case, "scope": SCOPE,
                      "ticks": 600, "sim_time_s": float(reply["sim_time"]), "initial_info": initial,
                      "final_info": reply["info"], "worker_provenance": provenance,
                      "outcome": score_trajectory(initial, reply["info"], cubes, lifted, inside),
                      "wall_seconds": time.monotonic() - case_started, "simulation_step": timing_stats(step_seconds),
                      "encoder_refresh": None}
            write_json(result_path, result)
            results.append(result)
            write_json(output / "summary.json", replay_summary(results, len(selected), config_hash, started))
            print(json.dumps({"event": "replay_case_complete", "case": case["case_id"], "outcome": result["outcome"]}), flush=True)
        # Rebuild a final summary even if every completed case was resumed.
        write_json(output / "summary.json", replay_summary(results, len(selected), config_hash, started))
    finally:
        if client is not None:
            client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, default=ROOT / "outputs/token_pilot_20261025/command_replay_reference")
    result.add_argument("--episode", type=int, default=1)
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--case-limit", type=int, help="Predeclared prefix, including nominal first; omit for all37")
    result.add_argument("--device", choices=["cpu"], default="cpu", help="No VGGT/ACT or GPU model forward")
    result.add_argument("--worker-python", type=Path, default=Path("/home/wukf/miniforge3/envs/te/bin/python"))
    result.add_argument("--dynamic-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921"))
    result.add_argument("--raw-root", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212"))
    result.add_argument("--timestamps-root", type=Path, default=ROOT / "inputs/raw_timestamps")
    result.add_argument("--source-readouts", type=Path, default=Path("/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz"))
    result.add_argument("--trace-every", type=int, default=20)
    result.add_argument("--worker-timeout", type=float, default=120.)
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        run(arguments)
    except Exception as error:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
