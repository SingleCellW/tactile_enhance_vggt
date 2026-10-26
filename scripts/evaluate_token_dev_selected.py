#!/usr/bin/env python3
"""Supplementary posthoc validation-selected ACT checkpoint sensitivity.

All ACT runs retain their completed 6000-update training budget. Checkpoints
are selected only by first minimum held-out action L1, never rollout outcomes.
The primary fixed-6000 evaluation is neither modified nor replaced. Concurrent
single-probe children share the caller's selected GPU; their wall times are not comparable with the
primary three-probe evaluation or with one another as isolated inference costs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, dependency_path, machine_path, require_machine_arguments, resolve_artifact_reference
sys.path[:0] = [str(ROOT / "scripts")]
from evaluate_sim import digest, write_json
from evaluate_token_act import checkpoint_map, make_cases, paired_comparisons, summarize_mode, validate_policy_inputs

MODES = ("internal", "late", "visual")
SCOPE = ("Posthoc dev-selected checkpoint sensitivity after completed 6000-update ACT training; "
         "selection uses held-out action L1 only, not rollout/test outcomes. Separate from the primary fixed-6000 result. "
         "Concurrent single-probe child wall times must not be compared with primary three-probe wall latency.")


def select_checkpoint(run, budget=6000):
    run = Path(run).resolve(strict=True)
    config = json.loads((run / "train_config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if config["steps"] != budget or status.get("status") != "completed" or status.get("step") != budget or status.get("total_steps") != budget:
        raise ValueError("Selection requires the completed original 6000-update training run")
    evaluation_interval = int(config["eval_every"])
    measured = []
    previous = 0
    for line_number, line in enumerate((run / "metrics.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if step <= previous or step > budget or row.get("total_steps") != budget:
            raise ValueError("Training metrics must have strictly ordered steps within the original budget")
        previous = step
        value = row.get("val_normalized_action_l1")
        # Intermediate log lines repeat the preceding validation value; only
        # actual evaluation steps are eligible checkpoint-selection evidence.
        if value is not None and (step % evaluation_interval == 0 or step == budget):
            if not math.isfinite(value):
                raise ValueError("Nonfinite validation selection metric")
            measured.append((float(value), step, line_number))
    if previous != budget or not measured:
        raise ValueError("Training metrics do not establish a completed budget and validation history")
    value, step, line_number = min(measured, key=lambda item: (item[0], item[1]))
    checkpoint = (run / "checkpoints" / f"{step:06d}").resolve(strict=True)
    marker = checkpoint / "checkpoint_complete.json"
    if json.loads(marker.read_text()).get("step") != step or not (checkpoint / "model.pt").is_file():
        raise ValueError("Selected checkpoint is not completely saved")
    best = run / "best"
    if not best.is_symlink() or best.resolve(strict=True) != checkpoint:
        raise ValueError("Original best symlink disagrees with first minimum validation checkpoint")
    if not math.isclose(status["best_val_normalized_action_l1"], value, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Training status disagrees with validation selection evidence")
    signature = {key: config.get(key) for key in
                 ("policy", "steps", "seed", "batch_size", "learning_rate", "weight_decay", "amp", "data_manifest_sha256")}
    return {"run": str(run), "checkpoint": str(checkpoint), "training_budget_steps": budget,
            "selected_step": step, "selection_metric": "val_normalized_action_l1", "selection_value": value,
            "metrics_line": line_number, "rule": "earliest actual validation step attaining the minimum L1 over completed budget",
            "uses_test_or_rollout_outcomes": False, "training_signature": signature,
            "evidence_sha256": {name: digest(run / name) for name in ("metrics.jsonl", "status.json", "train_config.json")},
            "checkpoint_complete_sha256": digest(marker), "model_sha256": digest(checkpoint / "model.pt"),
            "best_link_target": str(best.resolve(strict=True))}


def wait_for_primary(path, wait=False, timeout=43200., poll_seconds=15.):
    started = time.monotonic()
    while True:
        status = json.loads(path.read_text()) if path.exists() else None
        # The actual pipeline writer uses stage='complete', not a complete bool.
        complete = status is not None and (status.get("stage") == "complete" or status.get("complete") is True)
        if status is not None and status.get("stage") == "failed":
            raise ValueError("Primary pipeline failed; secondary GPU jobs will not start")
        if complete:
            return {"path": str(path.resolve()), "sha256": digest(path), "status": status}
        if not wait:
            raise ValueError("Primary pipeline is not complete; use --wait-primary to wait without starting children")
        if time.monotonic() - started >= timeout:
            raise TimeoutError("Timed out waiting for primary pipeline completion")
        print(json.dumps({"stage": "waiting_for_primary", "primary": status, "elapsed_s": time.monotonic() - started}), flush=True)
        time.sleep(min(poll_seconds, max(0., timeout - (time.monotonic() - started))))


def child_commands(args, runs, probes, selection):
    commands = {}
    for mode in MODES:
        commands[mode] = [str(args.python), "-u", str(ROOT / "scripts/evaluate_token_act.py"),
            "--cache", str(args.cache.resolve()), "--output", str((args.output / "mode_runs" / mode).resolve()),
            "--probe-checkpoint", f"{mode}={probes[mode]}",
            "--policy-checkpoint", f"{mode}={selection[mode]['checkpoint']}",
            "--representation-manifest", str(args.representation_manifest.resolve()),
            "--repository", str(args.repository.resolve()), "--weights", str(args.weights.resolve()),
            "--device", "cuda:0", "--amp", args.amp, "--episode", str(args.episode), "--seed", str(args.seed),
            "--worker-python", str(args.worker_python), "--dynamic-root", str(args.dynamic_root),
            "--raw-root", str(args.raw_root), "--timestamps-root", str(args.timestamps_root),
            "--source-readouts", str(args.source_readouts), "--trace-every", str(args.trace_every),
            "--video-cases", "0", "--selection-note", SCOPE]
    return commands


def merge_children(directories, selection, probes, cache_hash, expected_cases):
    models, provenance, common_roles, common_execution = {}, {}, None, None
    expected = {case["case_id"]: case for case in expected_cases}
    if len(expected) != 36 or len(expected_cases) != 36 or set(directories) != set(MODES):
        raise ValueError("Merge requires exactly three modes and 36 unique predeclared cases")
    for mode in MODES:
        directory = Path(directories[mode])
        summary_path, config_path, plan_path = [directory / name for name in ("summary.json", "eval_config.json", "case_plan.json")]
        summary, config, plan = [json.loads(path.read_text()) for path in (summary_path, config_path, plan_path)]
        config_hash = digest(config_path)
        if (not summary.get("complete") or summary.get("protocol") != "new36_xy_grid_live_token_act_v1" or
                set(summary.get("models", {})) != {mode} or summary.get("config_sha256") != config_hash or
                summary.get("case_plan_sha256") != digest(plan_path)):
            raise ValueError(f"Partial or mismatched child summary: {mode}")
        if (config.get("cache_manifest_sha256") != cache_hash or set(config.get("modes", {})) != {mode} or
                config.get("control_ticks") != 600 or config.get("physics_dt") != .01 or config.get("substeps") != 1 or
                config.get("sensor_velocity_dt") != .01 or plan.get("full_grid_cases") != expected_cases or
                plan.get("selected_case_ids") != [case["case_id"] for case in expected_cases] or
                plan.get("case_limit") is not None):
            raise ValueError(f"Child grid, cache, or physical protocol mismatch: {mode}")
        source = config["modes"][mode]
        checkpoint = Path(selection[mode]["checkpoint"]).resolve(strict=True)
        probe = Path(probes[mode]).resolve(strict=True)
        if (resolve_artifact_reference(source["policy"]).resolve(strict=True) != checkpoint or source["policy_sha256"] != selection[mode]["model_sha256"] or
                digest(checkpoint / "model.pt") != source["policy_sha256"] or
                resolve_artifact_reference(source["probe"]).resolve(strict=True) != probe or source["probe_sha256"] != digest(probe)):
            raise ValueError(f"Child probe/policy checkpoint source mismatch: {mode}")
        roles = summary.get("source_split_roles")
        execution = config.get("execution_horizon")
        if roles != plan.get("source_split_roles") or config.get("case_plan") != plan:
            raise ValueError(f"Child split or case plan records disagree: {mode}")
        if common_roles is None:
            common_roles, common_execution = roles, execution
        elif roles != common_roles or execution != common_execution:
            raise ValueError("Child source split roles or ACT execution horizons differ")
        rows = summary["models"][mode].get("results", [])
        ids = [row.get("case", {}).get("case_id") for row in rows]
        if len(rows) != 36 or len(set(ids)) != 36 or set(ids) != set(expected):
            raise ValueError(f"Missing, duplicate, or partial child cases: {mode}")
        case_hashes = {}
        for row in rows:
            case_id = row["case"]["case_id"]
            if (not row.get("complete") or row.get("mode") != mode or row["case"] != expected[case_id] or
                    row.get("ticks") != 600 or not np.isclose(row.get("sim_time_s", -1), 6., rtol=0, atol=1e-8) or
                    row.get("config_sha256") != config_hash):
                raise ValueError(f"Child case pairing or horizon mismatch: {mode}/{case_id}")
            case_directory = directory / mode / case_id
            result_path = case_directory / "result.json"
            if json.loads(result_path.read_text()) != row:
                raise ValueError("Child result file differs from its summary")
            with np.load(case_directory / "trajectory.npz", allow_pickle=False) as trajectory:
                if (trajectory["actions"].shape != (600, 22) or trajectory["cube_centers"].shape != (600, 3) or
                        not np.isfinite(trajectory["actions"]).all() or not np.isfinite(trajectory["cube_centers"]).all()):
                    raise ValueError("Child trajectory is incomplete or nonfinite")
            case_hashes[case_id] = digest(result_path)
        models[mode] = summarize_mode(rows)
        provenance[mode] = {"directory": str(directory.resolve()), "summary_sha256": digest(summary_path),
                            "eval_config_sha256": config_hash, "case_plan_sha256": digest(plan_path),
                            "case_result_sha256": case_hashes, "source": source}
    return {"complete": True, "selection": selection, "protocol": "posthoc_dev_selected36_sensitivity_v1",
            "training_budget_steps": 6000, "scope": SCOPE, "source_split_roles": common_roles,
            "models": models, "paired_comparisons": paired_comparisons(models), "child_provenance": provenance,
            "cache_manifest_sha256": cache_hash, "execution_horizon": common_execution}


def run_children(args, commands):
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    pending, active, completed, streams, spawned = list(MODES), {}, {}, [], []
    started = time.monotonic()
    finished = False
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                mode = pending.pop(0)
                command = list(commands[mode])
                if args.resume and (args.output / "mode_runs" / mode / "eval_config.json").exists():
                    command.append("--resume")
                log = args.output / "logs" / f"{mode}.log"
                stream = log.open("a" if args.resume else "x", buffering=1)
                streams.append(stream)
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                spawned.append(process)
                active[mode] = (process, command, log)
            for mode, (process, command, log) in list(active.items()):
                result = process.poll()
                if result is not None:
                    completed[mode] = {"returncode": result, "command": command, "log": str(log), "pid": process.pid}
                    del active[mode]
                    if result:
                        raise RuntimeError(f"Child {mode} failed with status {result}; inspect {log}")
            write_json(args.output / "status.json", {"stage": "running_children" if pending or active else "children_complete",
                "complete": False, "active": {mode: process.pid for mode, (process, _, _) in active.items()},
                "completed": completed, "elapsed_s": time.monotonic() - started})
            if args.child_timeout and time.monotonic() - started > args.child_timeout:
                raise TimeoutError("Secondary child evaluation timed out")
            if pending or active:
                time.sleep(2)
        finished = True
    finally:
        # Own children only. Reap all processes even on launch failure, OOM,
        # timeout or Ctrl-C; never leave independent GPU workers orphaned.
        if not finished:
            for process in spawned:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for process in spawned:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        for stream in streams:
            stream.close()
    return completed


def run(args):
    if not 1 <= args.workers <= 3 or args.primary_timeout <= 0 or not 0 < args.poll_seconds <= 60:
        raise ValueError("Need workers1..3, positive primary timeout, and poll-seconds in(0,60]")
    primary = wait_for_primary(args.primary_status, args.wait_primary, args.primary_timeout, args.poll_seconds)
    runs = checkpoint_map(args.act_run) if args.act_run else {mode: (args.input / f"act_{mode}_s43").resolve() for mode in MODES}
    probes = checkpoint_map(args.probe_checkpoint) if args.probe_checkpoint else {
        mode: (args.input / "s43_refined_v2" / mode / "best.pt").resolve(strict=True) for mode in MODES}
    if set(runs) != set(MODES) or set(probes) != set(MODES):
        raise ValueError("Exactly internal, late, and visual runs/probes are required")
    selection = {mode: select_checkpoint(runs[mode]) for mode in MODES}
    if len({json.dumps(row["training_signature"], sort_keys=True) for row in selection.values()}) != 1:
        raise ValueError("Original ACT architectures, 6000 budgets, seeds, optimizers or cache hashes differ")
    cache_hash = digest(args.cache / "manifest.json")
    metadata = json.loads((args.cache / "manifest.json").read_text())
    for mode in MODES:
        validate_policy_inputs(mode, Path(selection[mode]["checkpoint"]), probes[mode], cache_hash, metadata["normalization"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    config_path = args.output / "run_config.json"
    if args.merge_only:
        if not config_path.is_file():
            raise ValueError("Merge requires an existing saved run config")
        config = json.loads(config_path.read_text())
        validate_merge_configuration(config, selection, cache_hash, primary)
        commands = None
    else:
        commands = child_commands(args, runs, probes, selection)
        config = {"protocol": "posthoc_dev_selected36_sensitivity_v1", "scope": SCOPE, "selection": selection,
                  "primary_completion": primary, "cache_manifest_sha256": cache_hash, "workers": args.workers,
                  "gpu_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")},
                  "commands": commands, "implementation_sha256": digest(__file__)}
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text()) != config:
                raise ValueError("Secondary output exists or its protocol changed; use a fresh output")
        else:
            if args.resume:
                raise ValueError("Resume requires an existing matching secondary run config")
            write_json(config_path, config)
    child_processes = None if args.merge_only else run_children(args, commands)
    cases, _ = make_cases(args.episode, args.seed)
    directories = {mode: args.output / "mode_runs" / mode for mode in MODES}
    summary = merge_children(directories, selection, probes, cache_hash, cases)
    summary.update(primary_completion=primary, run_config_sha256=digest(config_path), child_processes=child_processes)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"stage": "complete", "complete": True})
    return summary


def validate_merge_configuration(saved, selection, cache_hash, primary):
    """Merging checks saved scientific evidence, not today's GPU or worker paths."""
    if (saved.get("protocol") != "posthoc_dev_selected36_sensitivity_v1" or saved.get("scope") != SCOPE
            or saved.get("cache_manifest_sha256") != cache_hash
            or saved.get("primary_completion", {}).get("sha256") != primary.get("sha256")
            or set(saved.get("selection", {})) != set(selection)):
        raise ValueError("Saved merge protocol or input evidence changed")
    location_fields = {"run", "checkpoint", "best_link_target"}
    for mode, selected in selection.items():
        before = {key: value for key, value in saved["selection"][mode].items() if key not in location_fields}
        after = {key: value for key, value in selected.items() if key not in location_fields}
        if before != after:
            raise ValueError(f"Saved checkpoint selection evidence changed: {mode}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    base = artifact_path("outputs") / "token_pilot_20261025"
    result.add_argument("--input", type=Path, default=base)
    result.add_argument("--output", type=Path, default=base / "dev_selected36")
    result.add_argument("--primary-status", type=Path, default=base / "pipeline_status.json")
    result.add_argument("--wait-primary", action="store_true")
    result.add_argument("--primary-timeout", type=float, default=43200.)
    result.add_argument("--poll-seconds", type=float, default=15.)
    result.add_argument("--workers", type=int, default=3)
    result.add_argument("--child-timeout", type=float, default=86400.)
    result.add_argument("--act-run", action="append", help="mode=original-training-run")
    result.add_argument("--probe-checkpoint", action="append", help="mode=representation-best.pt")
    result.add_argument("--cache", type=Path, default=artifact_path("cache") / "pickplace_vggt_native_physics_dt01_stride4")
    result.add_argument("--representation-manifest", type=Path, default=artifact_path("artifacts") / "token_pilot_data_20261025/manifest.json")
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    result.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    result.add_argument("--episode", type=int, default=1)
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--amp", choices=["float32", "bfloat16"], default="bfloat16")
    result.add_argument("--worker-python", type=Path, default=machine_path("worker_python", default=sys.executable))
    result.add_argument("--dynamic-root", type=Path, default=machine_path("dynamic_root"))
    result.add_argument("--raw-root", type=Path, default=machine_path("raw_root"))
    result.add_argument("--timestamps-root", type=Path, default=machine_path("timestamps_root", default=artifact_path("inputs") / "raw_timestamps"))
    result.add_argument("--source-readouts", type=Path, default=machine_path("source_readouts"))
    result.add_argument("--trace-every", type=int, default=20)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--merge-only", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if not args.merge_only:
        require_machine_arguments(args, 'worker_python', 'dynamic_root', 'raw_root', 'source_readouts', 'timestamps_root')
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        run(args)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", {"complete": False, "error": str(error), "scope": SCOPE})
        raise


if __name__ == "__main__":
    main()
