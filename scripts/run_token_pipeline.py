#!/usr/bin/env python3
"""Run the authorized single-GPU pilot stages with durable status/logs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    output = root/"outputs/token_pilot_20261025"
    logs = root/"logs/token_pilot_20261025"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", MUJOCO_EGL_DEVICE_ID="0", OMP_NUM_THREADS="4")
    python = str(root/".venv/bin/python")
    data = "artifacts/token_pilot_data_20261025"
    prefix = "artifacts/token_pilot_prefix_20261025"
    cache = "cache/pickplace_vggt_native_physics_dt01_stride4"
    completed = []

    def status(stage, **extra):
        value = {"stage": stage, "completed": completed, "pid": os.getpid(), **extra}
        temporary = output/"pipeline_status.tmp"
        temporary.write_text(json.dumps(value, indent=2)+"\n")
        temporary.replace(output/"pipeline_status.json")

    def run(stage, command):
        status(stage, command=command)
        with (logs/(stage+".log")).open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            status("failed", failed_stage=stage, returncode=result.returncode)
            raise SystemExit(result.returncode)
        completed.append(stage)
        status(stage+"_complete")

    status("waiting_for_primary_representation_run")
    deadline = time.monotonic()+1800
    while not (output/"s43_refined_v2/summary.json").exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Primary representation run did not complete")
        time.sleep(5)
    for seed in (44, 45):
        run(f"train_s{seed}_refined_v2", [python,"-u","scripts/run_token_pilot.py","train",
            "--data",data,"--prefix",prefix,"--output",str(output/f"s{seed}_refined_v2"),
            "--steps","2000","--eval-every","200","--batch-size","32",
            "--learning-rate","0.0001","--whiten-target","--seed",str(seed),
            "--modes","visual","late","internal"])
    parity = output/"runtime_cache_parity.json"
    if not parity.exists() or not json.loads(parity.read_text()).get("passed"):
        status("waiting_for_runtime_parity")
        deadline = time.monotonic()+900
        while not parity.exists():
            if time.monotonic()>deadline:
                raise TimeoutError("Runtime parity audit not supplied")
            time.sleep(5)
        if not json.loads(parity.read_text()).get("passed"):
            raise ValueError("Runtime/cache parity failed")
    modes = ("visual", "late", "internal")
    probes = {name:output/f"s43_refined_v2/{name}/best.pt" for name in modes}
    command = [python,"-u","scripts/token_runtime.py","--cache",cache,
               "--output","cache/token_features_20261025","--batch-size","4"]
    for name,path in probes.items():
        command += ["--checkpoint",f"{name}={path}"]
    run("export_original_features",command)
    for name in modes:
        run("train_act_"+name,[python,"-u","scripts/train_token_act.py","--cache",cache,
            "--features-dir",f"cache/token_features_20261025/{name}",
            "--output",str(output/f"act_{name}_s43"),"--steps","6000","--seed","43",
            "--batch-size","64","--eval-every","1000","--save-every","2000"])
    command = [python,"-u","scripts/evaluate_token_act.py","--cache",cache,
               "--output",str(output/"closed_loop36"),
               "--representation-manifest",str(root/data/"manifest.json")]
    for name,path in probes.items():
        command += ["--probe-checkpoint",f"{name}={path}",
                    "--policy-checkpoint",f"{name}={output/f'act_{name}_s43/checkpoints/006000'}"]
    run("closed_loop36",command)
    status("complete")


if __name__ == "__main__":
    main()
