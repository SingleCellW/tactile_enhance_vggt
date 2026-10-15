#!/usr/bin/env python3
"""Prepare or execute a paired ACT ablation suite on an explicit repaired cache.

``prepare`` writes only plans/configurations. ``execute`` is a separate command
and launches only the requested modes on explicitly selected physical GPUs.
This launcher does not generate/repair caches or alter the original run_pilot.sh.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("anchored", "unanchored", "no_tactile")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args):
    cache = Path(args.cache).resolve(strict=True)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("The repaired cache must be complete before preparing a training suite")
    native_report = manifest.get("native_tactile_report")
    if (manifest.get("historical_source_tactile_superseded") is not True
            or not isinstance(native_report, dict) or native_report.get("complete") is not True):
        raise ValueError("Refusing historical replay tactile: require historical_source_tactile_superseded=True and native_tactile_report.complete=True")
    rgb_alignment = manifest.get("configuration", {}).get("rgb_alignment")
    if (not isinstance(rgb_alignment, dict) or rgb_alignment.get("verified") is not True
            or type(rgb_alignment.get("rgb_lag_frames")) is not int
            or rgb_alignment["rgb_lag_frames"] != 1):
        raise ValueError("Refusing unaligned RGB: require configuration.rgb_alignment.verified=True and rgb_lag_frames=1")
    observation_frames = np.load(cache / "obs_frame_indices.npy", mmap_mode="r")
    episodes = np.load(cache / "episode_index.npy", mmap_mode="r")
    completed_path = cache / "completed.npy"
    if completed_path.exists() and not np.load(completed_path, mmap_mode="r").all():
        raise ValueError("Cache completion bitmap disagrees with its manifest")
    train_episodes, val_episodes = manifest["split"]["train"], manifest["split"]["val"]
    if set(train_episodes) & set(val_episodes):
        raise ValueError("Train and validation episode IDs overlap")
    selected_episodes = episodes[observation_frames]
    train_count = int(np.isin(selected_episodes, train_episodes).sum())
    val_count = int(np.isin(selected_episodes, val_episodes).sum())
    if train_count + val_count != len(observation_frames):
        raise ValueError("Some observations are missing from the shared train/validation split")
    base = json.loads(Path(args.config).read_text())
    if args.batch_size is not None:
        base["batch_size"] = args.batch_size
    if args.amp is not None:
        base["amp"] = args.amp
    batch_size = base["batch_size"]
    if batch_size < 1 or train_count < batch_size or not val_count:
        raise ValueError("Need a positive full training batch and a nonempty held-out split")
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else [base["seed"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    scaled_steps = base["reference_steps"] * train_count / base["reference_frames"]
    scaled_steps *= base["reference_batch_size"] / batch_size
    steps = args.steps if args.steps is not None else max(1000, round(scaled_steps / 1000) * 1000)
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output-root; existing suites are immutable")
    if output == cache or cache in output.parents:
        raise ValueError("Keep the suite output outside the immutable cache")
    python = Path(args.python).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir()
    (output / "logs").mkdir()
    jobs = []
    for seed in seeds:
        for mode in MODES:
            config = json.loads(json.dumps(base))
            config["seed"] = seed
            config["policy"]["mode"] = mode
            job_id = f"s{seed}_{mode}"
            config_path = output / "configs" / f"{job_id}.json"
            write_json(config_path, config)
            run_output = output / "runs" / job_id
            command = [str(python), str(ROOT / "scripts/train_act.py"),
                       "--config", str(config_path), "--cache", str(cache),
                       "--output", str(run_output), "--steps", str(steps),
                       "--device", "cuda:0", "--cpu-threads", str(args.cpu_threads)]
            jobs.append({"job_id": job_id, "mode": mode, "seed": seed,
                         "config": str(config_path), "config_sha256": digest(config_path),
                         "output": str(run_output), "log": str(output / "logs" / f"{job_id}.log"),
                         "command": command})
    implementation = [ROOT / "scripts/train_act.py", ROOT / "src/tactile_vggt/policy.py",
                      ROOT / "src/tactile_vggt/data.py", Path(__file__).resolve()]
    alignment = None
    if args.alignment_report:
        report_path = Path(args.alignment_report).resolve(strict=True)
        alignment = {"path": str(report_path), "sha256": digest(report_path),
                     "interpretation": "provenance attachment only; launcher does not certify synchronization"}
    plan = {
        "format_version": 1, "prepared_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "suite_root": str(output), "project_root": str(ROOT), "cache": str(cache),
        "manifest_sha256": digest(manifest_path),
        "cache_configuration_fingerprint": manifest.get("configuration_fingerprint"),
        "historical_source_tactile_superseded": True,
        "native_tactile_report_complete": True,
        "native_tactile_report_sha256": hashlib.sha256(json.dumps(native_report, sort_keys=True).encode()).hexdigest(),
        "rgb_alignment": rgb_alignment,
        "alignment_report": alignment, "numpy_version": np.__version__,
        "steps_per_run": steps, "batch_size": batch_size,
        "train_observations": train_count, "val_observations": val_count,
        "train_episode_ids": train_episodes, "val_episode_ids": val_episodes,
        "seeds": seeds, "modes": list(MODES),
        "budget": {"formula": "reference_steps * train_observations / reference_frames * reference_batch_size / batch_size",
                   "unrounded_scaled_steps": scaled_steps, "explicit_steps_override": args.steps is not None,
                   "chunks_processed_per_run": steps * batch_size,
                   "average_chunk_exposures_per_training_observation": steps * batch_size / train_count,
                   "reference_average_exposures": base["reference_steps"] * base["reference_batch_size"] / base["reference_frames"]},
        "paired_controls": ["same repaired cache and train/validation episode split", "same normalization",
                            "same batch order per seed", "same initial parameters per seed",
                            "same optimizer, step count, chunk horizon, evaluation schedule and precision"],
        "claim_boundary": "Compares tactile availability and spatial anchoring given VGGT geometry; does not establish VGGT superiority over RGB or oracle geometry and is not closed-loop evaluation.",
        "implementation": [{"path": str(path), "sha256": digest(path)} for path in implementation],
        "jobs": jobs,
    }
    plan_path = output / "suite_plan.json"
    write_json(plan_path, plan)
    print(json.dumps({"action": "prepared_only", "plan": str(plan_path), "steps_per_run": steps,
                      "train_observations": train_count, "val_observations": val_count,
                      "average_chunk_exposures": steps * batch_size / train_count,
                      "next": "execute --plan <suite_plan.json> --gpus 1,2 --modes anchored,unanchored"}, indent=2))


def execute(args):
    plan_path = Path(args.plan).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if plan.get("format_version") != 1:
        raise ValueError("Unsupported suite plan")
    if digest(Path(plan["cache"]) / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Cache manifest changed since suite preparation; prepare a new suite")
    for item in plan["implementation"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Implementation changed since preparation: {item['path']}")
    modes = args.modes.split(",")
    gpus = args.gpus.split(",")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError("Choose unique modes from anchored,unanchored,no_tactile")
    if not gpus or any(not gpu for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Provide unique physical GPU IDs/UUIDs")
    jobs = [job for job in plan["jobs"] if job["mode"] in modes]
    for job in jobs:
        if digest(job["config"]) != job["config_sha256"]:
            raise ValueError(f"A paired config was changed: {job['config']}")
    root = Path(plan["suite_root"])
    lock_path = root / ".execution.lock"
    lock = lock_path.open("x")
    lock.write(json.dumps({"pid": os.getpid(), "modes": modes, "gpus": gpus}) + "\n")
    lock.close()

    def worker(gpu, assigned):
        results = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        for job in assigned:
            output = Path(job["output"])
            status_path = output / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                saved_config = json.loads((output / "train_config.json").read_text())
                if (status.get("status") == "completed" and status.get("step") == plan["steps_per_run"]
                        and saved_config.get("data_manifest_sha256") == plan["manifest_sha256"]
                        and saved_config.get("seed") == job["seed"]
                        and saved_config.get("policy", {}).get("mode") == job["mode"]):
                    results.append({"job_id": job["job_id"], "status": "already_completed", "gpu": gpu})
                    continue
            if (output.exists() and any(output.iterdir())) or Path(job["log"]).exists():
                raise ValueError(f"Existing incomplete run {output}; use train_act.py --resume explicitly, preserving its planned config")
            # A dedicated worker owns each GPU and executes its assigned jobs
            # sequentially. Finishing on another GPU never starts an overlapping
            # second job on this GPU.
            with Path(job["log"]).open("x") as log:
                completed = subprocess.run(job["command"], cwd=plan["project_root"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            results.append({"job_id": job["job_id"], "status": "completed" if completed.returncode == 0 else "failed",
                            "returncode": completed.returncode, "gpu": gpu, "log": job["log"]})
            if completed.returncode:
                break
        return results

    try:
        with ThreadPoolExecutor(max_workers=min(len(gpus), len(jobs))) as pool:
            futures = [pool.submit(worker, gpu, jobs[index::len(gpus)]) for index, gpu in enumerate(gpus) if jobs[index::len(gpus)]]
            results = [item for future in futures for item in future.result()]
        write_json(root / "execution_status.json", {"modes": modes, "results": results})
        print(json.dumps(results, indent=2))
        if any(result["status"] == "failed" for result in results):
            raise RuntimeError("At least one ablation failed; inspect its preserved log and checkpoint")
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Write paired configs/plan without launching training")
    prep.add_argument("--cache", required=True, help="Explicit new complete cache with repaired native tactile")
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--config", default=str(ROOT / "configs/act_pilot.json"))
    prep.add_argument("--python", default=sys.executable)
    prep.add_argument("--seeds", help="Comma-separated paired training seeds; default is base-config seed")
    prep.add_argument("--steps", type=int)
    prep.add_argument("--batch-size", type=int)
    prep.add_argument("--amp", choices=("float32", "bfloat16"))
    prep.add_argument("--cpu-threads", type=int, default=4)
    prep.add_argument("--alignment-report", help="Optional independent synchronization audit to record by hash")
    run = commands.add_parser("execute", help="Execute selected planned modes; no automatic cache preparation")
    run.add_argument("--plan", required=True)
    run.add_argument("--gpus", required=True, help="Physical IDs/UUIDs, e.g. 1,2; one job per selected GPU")
    run.add_argument("--modes", default="anchored,unanchored", help="no_tactile remains optional until explicitly selected")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.cpu_threads < 1:
            parser.error("cpu-threads must be positive")
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
