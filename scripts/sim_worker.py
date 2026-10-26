#!/usr/bin/env python3
"""JSON-lines IPC around the isolated MuJoCo/native tactile runtime.

Requests: reset {episode_id, frame=0, seed=0, observe=true}, step
{action:[22], observe=true}, observe, metadata, close. Add ``command`` (or
``cmd``) to each object. stdout contains exactly one JSON response per request.
Observation NPZ files contain no object pose; privileged metrics stay in info.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ipc-dir", type=Path, required=True)
    p.add_argument("--dynamic-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--timestamps-root", type=Path, required=True)
    p.add_argument("--source-readouts", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--sensor-dt", type=float, default=.05, help="Causal tactile proxy dt; training used ~.05s wall timestamps, separately from physical steps")
    args = p.parse_args()
    args.ipc_dir.mkdir(parents=True, exist_ok=True)
    # All native model logging must stay off the machine-readable channel.
    with redirect_stdout(sys.stderr):
        import numpy as np
        from tactile_vggt.simulation import PickLiftSimulation
        simulation = PickLiftSimulation(args.dynamic_root, args.raw_root, args.timestamps_root,
                                        args.source_readouts, args.device, args.substeps, args.sensor_dt)
    observation_path = (args.ipc_dir / "response.npz").resolve()
    temporary = args.ipc_dir / "response.tmp.npz"
    try:
        for line in sys.stdin:
            started = time.monotonic()
            request = None
            try:
                request = json.loads(line)
                command = request.get("command", request.get("cmd"))
                with redirect_stdout(sys.stderr):
                    if command == "reset":
                        info = simulation.reset(request["episode_id"], request.get("frame", 0), request.get("seed", 0), request.get("object_xy_offset"))
                    elif command == "step":
                        info = simulation.step(request["action"])
                    elif command == "observe":
                        info = simulation.info()
                    elif command in ("metadata", "close"):
                        info = {}
                    else:
                        raise ValueError(f"Unsupported command: {command!r}")
                    response = {"ok": True, "command": command, "request_id": request.get("request_id"), "info": info}
                    if info:
                        response.update(control_tick=info["control_tick"], sim_time=info["sim_time"])
                    if command == "metadata" or command == "reset":
                        response["provenance"] = simulation.provenance
                    if command in ("reset", "step", "observe") and request.get("observe", True):
                        observation, diagnostics = simulation.observe()
                        np.savez(temporary, **observation)
                        temporary.replace(observation_path)
                        response.update(observation_path=str(observation_path), observation_diagnostics=diagnostics)
                response["worker_seconds"] = time.monotonic() - started
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "command": request.get("command", request.get("cmd")) if isinstance(request, dict) else None,
                            "error": str(error), "error_type": type(error).__name__,
                            "worker_seconds": time.monotonic() - started}
            print(json.dumps(response, allow_nan=False), flush=True)
            if response["ok"] and response["command"] == "close":
                break
    finally:
        with redirect_stdout(sys.stderr):
            simulation.close()


if __name__ == "__main__":
    main()
