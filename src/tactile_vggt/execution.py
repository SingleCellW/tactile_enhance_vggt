"""GPU selection shared by experiment launchers, without launching processes."""
from __future__ import annotations

import os


def selected_gpus(selection):
    values = [part.strip() for part in (selection or "").split(",")]
    if not all(values) or len(set(values)) != len(values):
        raise ValueError("Pass --gpus with distinct allocated GPU IDs/UUIDs")
    if os.environ.get("SLURM_JOB_ID"):
        allocated = set(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) - {""}
        if not allocated or not set(values) <= allocated:
            raise ValueError("--gpus must stay within Slurm's CUDA_VISIBLE_DEVICES allocation")
    return values


def child_environment(gpu, *, render=False):
    selected_gpus(gpu)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(name, "4")
    if render:
        environment.setdefault("MUJOCO_GL", "egl")
        # EGL may enumerate physical devices independently; honor a site's
        # explicit mapping instead of replacing it with a CUDA logical index.
        if gpu.isdigit():
            environment.setdefault("MUJOCO_EGL_DEVICE_ID", gpu)
    return environment
