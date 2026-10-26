#!/usr/bin/env python3
"""Run the predeclared A40 GPU0 Stage1.b campaign with durable status.

Only development curves determine the common budget. Inference on held-out
tests starts after this decision. Each subprocess has a separate log.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    root = ROOT
    output = root / "outputs/camera_pilot_20261026"
    logs = root / "logs/camera_pilot_20261026"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    with (output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"complete": False, "pid": os.getpid(), "started_unix": time.time(), "stages": [],
                  "resource": "rimlab-server / physical A40 GPU0", "test_used_for_budget": False}
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        python = str(root / ".venv/bin/python")
        common = [python, "-u", "scripts/run_camera_pilot.py", "train", "--data", "artifacts/camera_pilot_data_20261026",
                  "--prefix", "artifacts/camera_pilot_prefix_20261026", "--batch-size", "32", "--eval-every", "200",
                  "--learning-rate", "0.0001"]

        def run(name, command):
            row = {"name": name, "command": command, "started_unix": time.time(), "complete": False}
            status["stages"].append(row)
            status["stage"] = name
            write(output / "campaign.json", status)
            with (logs / (name + ".log")).open("w") as stream:
                result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
            row.update(returncode=result.returncode, ended_unix=time.time(), complete=result.returncode == 0)
            if result.returncode:
                raise RuntimeError(f"{name} failed with exit code {result.returncode}; see its log")
            write(output / "campaign.json", status)

        try:
            pilot = output / "main_2000"
            run("seed43_development_2000", common + ["--output", str(pilot), "--steps", "2000", "--seeds", "43", "--dev-only", "--resume"])
            convergence = json.loads((pilot / "convergence.json").read_text())
            if not convergence.get("complete"):
                raise ValueError("Budget decision requires all eight completed seed43 conditions")
            steps = 4000 if convergence["extend_all_to_4000"] else 2000
            final = output / f"main_{steps}"
            status.update(selected_steps=steps, selected_output=str(final), development_decision=convergence)
            write(output / "budget_decision.json", {"selected_steps": steps, "selected_output": str(final),
                  "test_used": False, "development_decision": convergence, "decision_unix": time.time()})
            run(f"formal_three_seeds_{steps}", common + ["--output", str(final), "--steps", str(steps), "--seeds", "43", "44", "45", "--resume"])
            summary = json.loads((final / "summary.json").read_text())
            if not summary.get("full_design_complete"):
                raise ValueError("Formal 24-run matrix is incomplete")
            status.update(complete=True, stage="complete", ended_unix=time.time())
        except Exception as error:
            status.update(stage="failed", error=str(error), ended_unix=time.time())
            write(output / "campaign.json", status)
            raise
        write(output / "campaign.json", status)


if __name__ == "__main__":
    main()
