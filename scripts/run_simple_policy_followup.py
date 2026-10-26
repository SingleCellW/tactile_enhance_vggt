#!/usr/bin/env python3
"""Run the predeclared minimal policy tests on the caller's allocated GPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, resolve_artifact_reference
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_token_dev_selected import select_checkpoint
from evaluate_sim import write_json


def commands(args, manifest):
    manifest = resolve_manifest_paths(manifest)
    root = args.output.resolve()
    if args.phase == "train":
        for run in manifest["training_runs"]:
            mode = run["mode"]
            projection, tactile = mode.split("_")
            yield mode, [sys.executable, "-u", "scripts/train_simple_token_act.py",
                "--cache", manifest["cache"], "--features-dir", manifest["features"],
                "--visual-probe", manifest["visual_probe"], "--projection", projection,
                "--tactile", tactile, "--output", str(root / run["directory"]),
                "--steps", str(manifest["training_steps"]), "--seed", str(manifest["seed"]),
                "--eval-every", str(manifest["validation_every"]), "--save-every", "1000",
                "--batch-size", "64", "--device", "cuda:0", "--num-workers", "2"]
        return
    selection = {}
    if args.phase == "eval":
        for run in manifest["training_runs"]:
            selection[run["mode"]] = select_checkpoint(root / run["directory"], manifest["training_steps"])
        selection_path = root / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Validation-selected checkpoints changed")
        write_json(selection_path, selection)
    for item in manifest["expected_evaluations"]:
        if (item["kind"] == "legacy") != (args.phase == "legacy"):
            continue
        mode = item["mode"]
        command = [sys.executable, "-u", "scripts/evaluate_simple_token_act.py", "--kind", item["kind"],
            "--cache", manifest["cache"], "--output", str(root / item["directory"]),
            "--case-set", item["case_set"], "--n-action-steps", str(item["n_action_steps"]),
            "--device", "cuda:0", "--video-cases", "1", "--representation-manifest",
            str(artifact_path("artifacts", "token_pilot_data_20261025/manifest.json"))]
        if item["kind"] == "simple":
            command += ["--policy-checkpoint", f"{mode}={selection[mode]['checkpoint']}",
                        "--visual-probe-checkpoint", manifest["visual_probe"]]
        else:
            old = select_checkpoint(artifact_path("outputs") / f"token_pilot_20261025/act_{mode}_s43")
            command += ["--policy-checkpoint", f"{mode}={old['checkpoint']}",
                "--probe-checkpoint", f"{mode}={artifact_path('runs', 'token_pilot_20261025', 's43_refined_v2', mode, 'best.pt')}"]
        if args.resume and (root / item["directory"] / "eval_config.json").exists():
            command += ["--resume"]
        yield f"{mode}_n{item['n_action_steps']}", command


def resolve_manifest_paths(manifest):
    """Relocate known artifact inputs in memory, preserving the saved protocol."""
    return {key: str(resolve_artifact_reference(value)) if key in ("cache", "features", "visual_probe") else value
            for key, value in manifest.items()}


def run(args):
    manifest = resolve_manifest_paths(json.loads((args.output / "experiment_manifest.json").read_text()))
    if args.wait_training:
        if args.phase != "eval":
            raise ValueError("--wait-training is only valid for final selected-checkpoint evaluation")
        while True:
            path = args.output / "train_driver_status.json"
            if path.exists() and json.loads(path.read_text()).get("complete"):
                break
            print(json.dumps({"event": "waiting_for_completed_training"}), flush=True)
            time.sleep(5)
    if args.phase == "train" and not (Path(manifest["features"]) / "manifest.json").exists():
        raise ValueError("Complete and audit the frozen visual export first")
    pending = list(commands(args, manifest))
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    active, completed, spawned = {}, {}, []
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    started, finished = time.monotonic(), False
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                name, command = pending.pop(0)
                log = logs / f"{args.phase}_{name}.log"
                stream = log.open("a", buffering=1)
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                spawned.append(process)
                active[name] = (process, stream, command)
                print(json.dumps({"event": "launched", "phase": args.phase, "name": name, "pid": process.pid}), flush=True)
            for name, (process, stream, command) in list(active.items()):
                result = process.poll()
                if result is not None:
                    stream.close()
                    completed[name] = {"returncode": result, "command": command, "pid": process.pid}
                    del active[name]
                    if result:
                        raise RuntimeError(f"{args.phase}/{name} failed with {result}; inspect its log")
            write_json(args.output / f"{args.phase}_driver_status.json", {
                "phase": args.phase, "complete": not pending and not active,
                "active": {name: entry[0].pid for name, entry in active.items()},
                "pending": [name for name, _ in pending], "completed": completed,
                "elapsed_seconds": time.monotonic() - started})
            if pending or active:
                time.sleep(2)
        finished = True
    finally:
        if not finished:
            for process in spawned:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in spawned:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for _, stream, _ in active.values():
            stream.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=artifact_path("outputs") / "simple_policy_followup_20261025")
    p.add_argument("--phase", choices=("train", "eval", "legacy"), required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wait-training", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three processes on the caller's allocated GPU")
    run(args)
