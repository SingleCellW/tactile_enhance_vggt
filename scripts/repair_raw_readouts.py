#!/usr/bin/env python3
"""Recompute native tactile at recorded qpos, using only current/past timestamps.

This writes NEW readouts and an audit report; existing data/caches are untouched.
The object pose is used only to synthesize the simulated sensor observation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

LINK_MAP = {"palm_lower": "palm", "fingertip": "if_ds", "dip": "if_md", "pip": "if_px", "mcp_joint": "if_bs",
            "fingertip_2": "mf_ds", "dip_2": "mf_md", "pip_2": "mf_px", "mcp_joint_2": "mf_bs",
            "fingertip_3": "rf_ds", "dip_3": "rf_md", "pip_3": "rf_px", "mcp_joint_3": "rf_bs",
            "thumb_temp_base": "th_mp", "thumb_pip": "th_bs", "thumb_dip": "th_px", "thumb_fingertip": "th_ds"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--timestamps-root", type=Path, required=True)
    parser.add_argument("--source-readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100, help="0 for all frames; otherwise spread a probe across all episodes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        merge_readouts(args)
        return
    b = args.dynamic_root
    sys.path[:0] = [str(b / "eval/vendor"), str(b / "eval/sim"), str(b)]
    import mujoco
    import torch
    from tactile_sim.models import models
    from fixed44_geometry import Fixed44SaliencyProjector
    from dynamic44 import continuous_readout, recorded_anchors
    from audit_alignment import load_model
    if mujoco.__version__ != "3.4.0":
        raise RuntimeError(f"Expected frozen PickLift MuJoCo 3.4.0, got {mujoco.__version__}")
    model = load_model(b / "eval/assets/picklift", mujoco)
    data = mujoco.MjData(model)
    urdf = b / "eval/sim/tactile_sim/urdf"
    object_mesh = b / "eval/assets/rotate/meshes/0.07cube.obj"
    geometry = np.load(b / "readouts/geometry.npz", allow_pickle=False)
    bounds = json.loads((b / "readouts/geometry.json").read_text())["link_bounds"]
    canonical = np.load(b / "eval/sim/tactile_sim/tac-pts.npy", allow_pickle=False)
    projector = Fixed44SaliencyProjector(model, b / "eval/sim/tactile_sim/tac-pts.npy")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    started = time.monotonic()
    hand = models(str(urdf / "leap_hand_right_tactile.urdf"), str(urdf), str(object_mesh), args.device)
    hand.set_object(str(object_mesh)); hand.setup_models()
    for name in hand.hand_model.mesh:
        hand.hand_model.mesh[name]["surface_points"] = torch.tensor(geometry["surface_" + name], device=args.device)
    # Fixed sensor regions must retain exactly the frozen dense-field order.
    if hand.ind2link != bounds:
        raise ValueError("Native dense point order differs from frozen geometry")
    with np.load(args.source_readouts, allow_pickle=False) as source:
        state = source["state22"].copy(); episode = source["episode_index"].copy(); frame = source["frame_index"].copy()
    indices = np.arange(len(state))
    if args.max_frames and args.max_frames < len(indices):
        indices = indices[np.linspace(0, len(indices) - 1, args.max_frames, dtype=int)]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    selected_episodes = np.array_split(np.unique(episode[indices]), args.num_shards)[args.shard_index]
    indices = indices[np.isin(episode[indices], selected_episodes)]
    forces = np.empty((len(indices), 44), np.float32)
    anchors = np.empty((len(indices), 44, 3), np.float32)
    contact = np.empty_like(forces)
    body_ids = {name: model.body(name).id for name in set(LINK_MAP.values()) | {"palm", "cube"}}
    raw_episodes = sorted(p for p in args.raw_root.iterdir() if (p / "prioception").exists())
    selected_lookup = {int(global_index): row for row, global_index in enumerate(indices)}
    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=args.device).unsqueeze(0)
    diagnostics = []
    timestamps_used = []
    velocity_convention_max = 0.0
    velocity_origin_jacobian_max = 0.0
    old_errors = []
    mesh_rotation_max = 0.0
    torch.cuda.reset_peak_memory_stats(args.device)
    compute_started = time.monotonic()

    def body_state(name, object_xyzw=False):
        nonlocal velocity_convention_max, velocity_origin_jacobian_max
        bid = body_ids[name]
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
        velocity_convention_max = max(velocity_convention_max, float(np.max(np.abs(data.cvel[bid] - velocity))))
        # mjOBJ_BODY linear velocity is centered at xipos (the body COM).
        # Native all_link_trans is xpos, so shift the velocity to that origin.
        velocity[3:] += np.cross(velocity[:3], data.xpos[bid] - data.xipos[bid])
        jacp = np.empty((3, model.nv))
        jacr = np.empty((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid], bid)
        velocity_error = float(np.max(np.abs(velocity - np.r_[jacr @ data.qvel, jacp @ data.qvel])))
        velocity_origin_jacobian_max = max(velocity_origin_jacobian_max, velocity_error)
        if velocity_error > 1e-8:
            raise ValueError(f"Body-origin velocity differs from Jacobian: {name}: {velocity_error}")
        quat = data.xquat[bid][[1, 2, 3, 0]] if object_xyzw else data.xquat[bid]
        return np.r_[data.xpos[bid], quat, velocity[3:], velocity[:3]]

    for ep in np.unique(episode[indices]):
        raw = raw_episodes[int(ep)]
        csv_path = args.timestamps_root / f"teleop_log_{raw.name}.csv"
        rows = list(csv.DictReader(csv_path.open()))
        timestamps = np.array([float(row["timestamp"]) for row in rows])
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"Non-monotonic timestamps: {raw.name}")
        timestamps_used.append({"episode": int(ep), "path": str(csv_path), "sha256": sha(csv_path)})
        local_globals = indices[episode[indices] == ep]
        hand.init = False
        for global_index in local_globals:
            local_frame = int(frame[global_index])
            qpos = np.load(raw / rows[local_frame]["prioception_path"], allow_pickle=False)
            np.testing.assert_array_equal(qpos[:22].astype(np.float32), state[global_index])
            if qpos.shape != (model.nq,):
                raise ValueError((raw.name, qpos.shape, model.nq))
            data.qpos[:] = qpos
            data.qvel[:] = 0
            dt = None
            if local_frame:
                previous = np.load(raw / rows[local_frame - 1]["prioception_path"], allow_pickle=False)
                dt = float(timestamps[local_frame] - timestamps[local_frame - 1])
                mujoco.mj_differentiatePos(model, data.qvel, dt, previous, qpos)
            mujoco.mj_forward(model, data)
            qpos_before = data.qpos.copy()
            h = body_state("palm")
            o = body_state("cube", object_xyzw=True)
            links = np.stack([body_state(LINK_MAP[name]) for name in hand.ind2link])
            hand.cal_tactile_info(tensor(h), tensor(o), tensor(data.qpos[6:22]), tensor(links))
            dense_xyz = hand.current_surface_points[0].detach().cpu().numpy()
            dense_force = hand.current_forces[0].detach().cpu().numpy()
            if not np.isfinite(dense_force).all():
                raise ValueError(f"Nonfinite native force at {raw.name}/{local_frame}; no silent NaN replacement")
            observed = continuous_readout(dense_force[:, -1], geometry["weights"])
            recovered, rigid_error = recorded_anchors(dense_xyz[None], geometry["points_reference"], bounds, canonical)
            fk = projector.anchor_world(data)
            error = float(np.linalg.norm(recovered[0] - fk, axis=-1).max())
            if error > 2e-5 or rigid_error > 2e-5:
                raise ValueError(f"Native/FK anchor mismatch: {error}, rigid={rigid_error}")
            # Verify the actual transformed object vertices, not just the input quaternion.
            native_vertices = hand.object_model.object_vertices_list[0].detach().cpu().numpy()
            mesh_vertices = np.asarray(hand.object_model.object_mesh_list[0].vertices)
            cube_id = body_ids["cube"]
            expected_vertices = mesh_vertices @ data.xmat[cube_id].reshape(3, 3).T + data.xpos[cube_id]
            mesh_error = float(np.linalg.norm(native_vertices - expected_vertices, axis=-1).max())
            mesh_rotation_max = max(mesh_rotation_max, mesh_error)
            if mesh_error > 2e-6:
                raise ValueError(f"Object quaternion convention mismatch: {mesh_error}")
            np.testing.assert_array_equal(data.qpos, qpos_before)
            output_index = selected_lookup[int(global_index)]
            forces[output_index], anchors[output_index] = observed, fk
            contact[output_index] = observed > 5e-4
            diagnostics.append({"global_index": int(global_index), "episode": int(ep), "frame": local_frame,
                                "dt_s": dt, "fk_native_anchor_max_m": error, "rigid_max_m": rigid_error,
                                "object_vertices_max_m": mesh_error, "peak_force_proxy": float(observed.max()),
                                "active_sensors": int((observed > 5e-4).sum())})
            if len(diagnostics) % 100 == 0:
                print(json.dumps({"processed": len(diagnostics), "total": len(indices), "elapsed_s": time.monotonic() - compute_started}), flush=True)
    elapsed = time.monotonic() - compute_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, force44=forces, contact44=contact, anchors_world=anchors,
                        state22=state[indices], episode_index=episode[indices], frame_index=frame[indices])
    report = {"status": "PASS_NATIVE_RAW_STATE_RECOMPUTE", "complete": len(indices) == len(state), "complete_source": len(indices) == len(state),
              "shard_index": args.shard_index, "num_shards": args.num_shards, "shard_complete": True,
              "frames": len(indices), "source_frames": len(state), "compute_seconds": elapsed,
              "seconds_per_frame": elapsed / len(indices), "estimated_full_minutes": elapsed / len(indices) * len(state) / 60,
              "initialization_seconds": compute_started - started,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
              "native_fk_anchor_max_m": max(row["fk_native_anchor_max_m"] for row in diagnostics),
              "object_vertices_max_m": mesh_rotation_max, "cvel_vs_body_velocity_max_abs": velocity_convention_max,
              "body_origin_velocity_jacobian_max_abs": velocity_origin_jacobian_max,
              "velocity": "causal mj_differentiatePos(prev full qpos29,current qpos29,actual dt), first frame zero; mj_objectVelocity(body,world orientation), COM velocity shifted to xpos by omega cross (xpos-xipos); verified against xpos Jacobian at every body/frame",
              "object_quaternion": "MuJoCo WXYZ converted to native object adapter XYZW; transformed mesh verified against data.xmat",
              "force_law": "unchanged frozen native cal_force/cal_vel; continuous simulator proxy, not calibrated Newtons; legacy cross(r,omega) retained",
              "sensor_order": "32 fingertip taxels then 12 FSRs using frozen continuous readout weights",
              "observation_causality": "only current qpos and previous qpos/timestamp; no future state/action used",
              "rgb_status": "original raw capture RGB unchanged; capture/render ordering requires separate audit",
              "source_readouts_sha256": sha(args.source_readouts), "repaired_readouts_sha256": sha(args.output),
              "native_models_sha256": sha(b / "eval/sim/tactile_sim/models.py"), "geometry_sha256": sha(b / "readouts/geometry.npz"),
              "object_mesh": str(object_mesh), "object_mesh_sha256": sha(object_mesh),
              "timestamp_sources": timestamps_used, "diagnostics": diagnostics}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("timestamp_sources", "diagnostics")}, indent=2), flush=True)


def merge_readouts(args):
    reports = [json.loads(path.with_suffix(".report.json").read_text()) for path in args.merge_inputs]
    if not all(report["shard_complete"] for report in reports):
        raise ValueError("An input shard did not finish")
    source_sha = sha(args.source_readouts)
    for path, report in zip(args.merge_inputs, reports):
        if report["source_readouts_sha256"] != source_sha or report["repaired_readouts_sha256"] != sha(path):
            raise ValueError("Shard/source/report checksum mismatch")
    keys = ("force44", "contact44", "anchors_world", "state22", "episode_index", "frame_index")
    pieces = []
    for path in args.merge_inputs:
        with np.load(path, allow_pickle=False) as data:
            pieces.append({key: data[key].copy() for key in keys})
    merged = {key: np.concatenate([piece[key] for piece in pieces]) for key in keys}
    order = np.lexsort((merged["frame_index"], merged["episode_index"]))
    merged = {key: value[order] for key, value in merged.items()}
    with np.load(args.source_readouts, allow_pickle=False) as source:
        for key in ("state22", "episode_index", "frame_index"):
            np.testing.assert_array_equal(merged[key], source[key], err_msg=f"Missing, duplicate or changed rows: {key}")
    if not all(np.isfinite(merged[key]).all() for key in keys):
        raise ValueError("Nonfinite merged sensor data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    report = dict(reports[0])
    report.update(complete=True, complete_source=True, frames=len(merged["state22"]),
                  compute_seconds=sum(r["compute_seconds"] for r in reports),
                  parallel_compute_seconds=max(r["compute_seconds"] for r in reports),
                  native_fk_anchor_max_m=max(r["native_fk_anchor_max_m"] for r in reports),
                  object_vertices_max_m=max(r["object_vertices_max_m"] for r in reports),
                  cvel_vs_body_velocity_max_abs=max(r["cvel_vs_body_velocity_max_abs"] for r in reports),
                  body_origin_velocity_jacobian_max_abs=max(r["body_origin_velocity_jacobian_max_abs"] for r in reports),
                  repaired_readouts_sha256=sha(args.output),
                  merged_inputs=[{"path": str(p), "sha256": sha(p)} for p in args.merge_inputs],
                  timestamp_sources=[x for r in reports for x in r["timestamp_sources"]],
                  diagnostics=[x for r in reports for x in r["diagnostics"]])
    report.pop("shard_index", None)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("timestamp_sources", "diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
