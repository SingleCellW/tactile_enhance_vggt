#!/usr/bin/env python3
"""Build NEW synchronized vision/touch snapshots for a spatial-token pilot.

The source cache is read-only. Simulator object state is used to generate the
sensor and RGB observations and *labels*, never model input geometry. Perturbed
siblings are static counterfactual snapshots, not demonstrations or trajectories.
The native force law and physical 0.01 s causal velocity convention are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, machine_path
from tactile_vggt.geometry import world_to_base
from tactile_vggt.simulation import CAMERAS, NativeTactileAdapter, load_frozen_model, training_rgb_transform


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def split_episodes(episodes, seed, dev_count=7, test_count=8):
    """All siblings and source frames from one episode stay in one split."""
    episodes = np.unique(episodes).astype(int)
    if dev_count < 1 or test_count < 1 or dev_count + test_count >= len(episodes):
        raise ValueError("Need nonempty train/dev/test episode groups")
    ordered = np.random.default_rng(seed).permutation(episodes)
    return {"train": sorted(ordered[dev_count + test_count:].tolist()),
            "dev": sorted(ordered[:dev_count].tolist()),
            "test": sorted(ordered[dev_count:dev_count + test_count].tolist())}


def select_sources(episode, frame, obs_rows, pressure, per_episode, threshold):
    """Prefer contact-rich states, while keeping temporal spread and no-touch controls."""
    if per_episode < 1:
        raise ValueError("sources-per-episode must be positive")
    selected = []
    for ep in np.unique(episode):
        candidates = np.flatnonzero(episode[obs_rows] == ep)
        active = (pressure[obs_rows[candidates]] > threshold).any(-1)
        contact, other = candidates[active], candidates[~active]
        n_contact = min(len(contact), max(1, round(per_episode * .75)))
        n_other = min(len(other), per_episode - n_contact)
        if n_contact + n_other < min(per_episode, len(candidates)):
            n_contact = min(len(contact), per_episode - n_other)
        for pool, count in ((contact, n_contact), (other, n_other)):
            if count:
                selected.extend(pool[np.linspace(0, len(pool) - 1, count, dtype=int)].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def perturb_object(qpos, object_qadr, rng, minimum, maximum, rotation_degrees):
    """A static pose intervention; no displacement/dt velocity is invented."""
    result = np.array(qpos, dtype=np.float64, copy=True)
    direction = rng.normal(size=3)
    direction[2] *= .4
    direction /= max(np.linalg.norm(direction), 1e-12)
    shift = direction * rng.uniform(minimum, maximum)
    result[object_qadr:object_qadr + 3] += shift
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    half = .5 * np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees))
    delta = np.r_[np.cos(half), axis * np.sin(half)]
    old = result[object_qadr + 3:object_qadr + 7]
    result[object_qadr + 3:object_qadr + 7] = np.r_[delta[0] * old[0] - delta[1:] @ old[1:],
        delta[0] * old[1:] + old[0] * delta[1:] + np.cross(delta[1:], old[1:])]
    result[object_qadr + 3:object_qadr + 7] /= np.linalg.norm(result[object_qadr + 3:object_qadr + 7])
    return result, shift


def contact_regime_allowed(pressure, threshold, allow_contact_breaks=False):
    """A contact break is an actual zero-force observation, not a changed label."""
    return bool(allow_contact_breaks or np.any(np.asarray(pressure) > threshold))


class SnapshotBuilder:
    def __init__(self, args):
        import mujoco
        if mujoco.__version__ != "3.4.0":
            raise RuntimeError(f"Expected frozen MuJoCo 3.4.0, got {mujoco.__version__}")
        self.args, self.mj = args, mujoco
        self.cache = Path(args.cache).resolve()
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        configuration = self.manifest["configuration"]
        native = self.manifest.get("native_tactile_report", {})
        alignment = configuration.get("rgb_alignment", {})
        if not self.manifest.get("complete") or not native.get("complete"):
            raise ValueError("A complete repaired native cache is required")
        if native.get("velocity_dt") != .01:
            raise ValueError("Expected explicitly audited native physical velocity_dt=0.01")
        if not alignment.get("verified") or alignment.get("rgb_lag_frames") != 1:
            raise ValueError("Expected the verified historical RGB row r -> state row r-1 mapping")
        self.arrays = {name: np.load(self.cache / f"{name}.npy", mmap_mode="r") for name in
                       ("state", "tactile", "episode_index", "frame_index", "obs_frame_indices",
                        "rgb_source_frame_indices", "points")}
        self.base_to_world = np.asarray(configuration["base_to_world"], np.float64)
        self.model = load_frozen_model(Path(args.dynamic_root) / "eval/assets/picklift", mujoco)
        self.data = mujoco.MjData(self.model)
        self.native = NativeTactileAdapter(self.model, args.dynamic_root, args.device)
        native_files = {"geometry_sha256": Path(args.dynamic_root) / "readouts/geometry.npz",
                        "native_models_sha256": Path(args.dynamic_root) / "eval/sim/tactile_sim/models.py",
                        "object_mesh_sha256": self.native.object_mesh}
        for key, path in native_files.items():
            if fingerprint(path)["sha256"] != native.get(key):
                raise ValueError(f"Frozen native asset differs from the repaired cache: {key}")
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.seg_renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.cube = self.model.body("cube").id
        self.palm = self.model.body("palm").id
        self.base = self.model.body("link_base").id
        self.object_qadr = int(self.model.jnt_qposadr[self.model.joint("cube_freejoint").id])
        self.cube_geoms = np.flatnonzero(self.model.geom_bodyid == self.cube)
        self.raw = sorted(p for p in Path(args.raw_root).iterdir() if (p / "prioception").exists())
        if len(self.raw) != len(np.unique(self.arrays["episode_index"])):
            raise ValueError("Raw episode directories do not match source cache")
        self.rows, self.provenance = {}, {}
        source = configuration["source"]
        self.video_paths = [Path(item["path"]) for item in source["rgb_videos"]]

    def close(self):
        self.renderer.close()
        self.seg_renderer.close()

    def source_state(self, cache_index):
        row = int(self.arrays["obs_frame_indices"][cache_index])
        ep = int(self.arrays["episode_index"][row])
        frame = int(self.arrays["frame_index"][row])
        if ep not in self.rows:
            path = Path(self.args.timestamps_root) / f"teleop_log_{self.raw[ep].name}.csv"
            self.rows[ep] = list(csv.DictReader(path.open()))
            self.provenance[str(path.resolve())] = fingerprint(path)
        records = self.rows[ep]
        path = self.raw[ep] / records[frame]["prioception_path"]
        qpos = np.load(path, allow_pickle=False).astype(np.float64)
        if qpos.shape != (29,):
            raise ValueError("Expected complete recorded qpos29")
        np.testing.assert_array_equal(qpos[:22].astype(np.float32), self.arrays["state"][row])
        velocity = np.zeros(self.model.nv)
        if frame:
            previous_path = self.raw[ep] / records[frame - 1]["prioception_path"]
            previous = np.load(previous_path, allow_pickle=False)
            self.mj.mj_differentiatePos(self.model, velocity, .01, previous, qpos)
            self.provenance[str(previous_path.resolve())] = fingerprint(previous_path)
        self.provenance[str(path.resolve())] = fingerprint(path)
        return qpos, velocity, row, ep, frame

    def set_state(self, qpos, velocity):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = velocity
        self.data.ctrl[:] = qpos[:22]
        self.mj.mj_forward(self.model, self.data)

    def observe_native(self):
        # No no_grad/inference_mode: the frozen native SDF itself uses autograd.
        self.native.reset()
        return self.native.observe(self.data)

    def rgb(self):
        result = []
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=name)
            result.append(training_rgb_transform(self.renderer.render()))
        return np.stack(result)

    def camera_arrays(self):
        poses, intrinsic = [], []
        rotation, origin = self.base_to_world[:3, :3], self.base_to_world[:3, 3]
        for name in CAMERAS:
            cid = self.model.camera(name).id
            pose = np.eye(4)
            pose[:3, :3] = rotation.T @ self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
            pose[:3, 3] = (self.data.cam_xpos[cid] - origin) @ rotation
            focal = 112. / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
            poses.append(pose)
            intrinsic.append([[focal, 0, 111.5], [0, focal, 111.5], [0, 0, 1.]])
        return np.asarray(poses, np.float32), np.asarray(intrinsic, np.float32)

    def penetration(self):
        distances = [float(c.dist) for c in self.data.contact if np.isin(c.geom, self.cube_geoms).any()]
        return min(distances, default=0.)

    def visibility(self):
        """GT-only natural occlusion scoring; masks are never returned as inputs."""
        result = []
        groups = self.model.geom_group.copy()
        option = self.mj.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[5] = 1
        self.seg_renderer.enable_segmentation_rendering()
        try:
            for name in CAMERAS:
                self.seg_renderer.update_scene(self.data, camera=name)
                segment = self.seg_renderer.render()
                visible = int(np.isin(segment[..., 0], self.cube_geoms).sum())
                self.model.geom_group[:] = 0
                self.model.geom_group[self.cube_geoms] = 5
                self.seg_renderer.update_scene(self.data, camera=name, scene_option=option)
                segment = self.seg_renderer.render()
                amodal = int(np.isin(segment[..., 0], self.cube_geoms).sum())
                self.model.geom_group[:] = groups
                result.append(1. - min(visible / max(amodal, 1), 1.))
        finally:
            self.model.geom_group[:] = groups
            self.seg_renderer.disable_segmentation_rendering()
        return np.asarray(result, np.float32)


def audit_alignment(builder, count, output):
    """Gate generation on existing nominal sensor/RGB agreement, not GT fitting."""
    from prepare_cache import VideoReader
    a = builder.arrays
    active = (a["tactile"][a["obs_frame_indices"]] > .0005).any(-1)
    chosen = []
    for flag in (False, True):
        pool = np.flatnonzero(active == flag)
        if len(pool):
            chosen.extend(pool[np.linspace(0, len(pool) - 1, min(len(pool), max(1, count // 2)), dtype=int)].tolist())
    selected = sorted(set(chosen))
    readers = [VideoReader(path, len(a["state"])) for path in builder.video_paths]
    rows = []
    try:
        for index in selected:
            qpos, velocity, row, ep, frame = builder.source_state(index)
            builder.set_state(qpos, velocity)
            force, anchors, diagnostic = builder.observe_native()
            cached_force = np.asarray(a["tactile"][row])
            cached_anchors = a["points"][index, -44:, :3]
            base_anchors = world_to_base(anchors, builder.base_to_world)
            rgb_row = int(a["rgb_source_frame_indices"][index])
            if rgb_row != row + 1 or a["episode_index"][rgb_row] != ep:
                raise ValueError("RGB/state source mapping does not match the audited lag")
            recorded = np.stack([reader.read(rgb_row) for reader in readers])
            rendered = builder.rgb()
            rgb_mae = float(np.abs(rendered.astype(np.float32) - recorded.astype(np.float32)).mean())
            force_ok = bool(np.allclose(force, cached_force, atol=1e-4, rtol=1e-3))
            anchor_max = float(np.linalg.norm(base_anchors - cached_anchors, axis=-1).max())
            item = {"cache_index": index, "source_row": row, "rgb_source_row": rgb_row,
                    "episode": ep, "frame": frame, "force_max_abs": float(np.abs(force - cached_force).max()),
                    "force_allclose": force_ok, "anchor_max_m": anchor_max, "rgb_mae_255": rgb_mae,
                    "native_diagnostics": diagnostic}
            item["passed"] = force_ok and anchor_max < 2e-5 and rgb_mae < 7.
            rows.append(item)
    finally:
        for reader in readers:
            reader.close()
    report = {"complete": True, "passed": bool(rows) and all(row["passed"] for row in rows),
              "count": len(rows), "velocity_dt": .01, "force_atol": 1e-4, "force_rtol": 1e-3,
              "anchor_tolerance_m": 2e-5, "rgb_mae_tolerance_255": 7., "cases": rows,
              "cache_manifest": fingerprint(builder.cache / "manifest.json"),
              "implementation": fingerprint(__file__),
              "interpretation": "Nominal regenerated observations compared with repaired cache and decoded historical RGB at r+1; no sensor or geometry fitting."}
    write_json(output, report)
    if not report["passed"]:
        raise ValueError(f"Nominal alignment audit failed; inspect {output}")
    return report


def build(args):
    if args.variants < 0 or args.audit_count < 2 or args.max_attempts < 1:
        raise ValueError("Invalid generation/audit counts")
    if not 0 <= args.jitter_min <= args.jitter_max or args.jitter_max > .05:
        raise ValueError("Object translation jitter must be between zero and 5cm")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    builder = SnapshotBuilder(args)
    try:
        audit = audit_alignment(builder, args.audit_count, output / "alignment_audit.json")
        if args.audit_only:
            print(json.dumps({"audit_only": True, "passed": True, "output": str(output)}), flush=True)
            return
        a = builder.arrays
        split = split_episodes(a["episode_index"], args.seed, args.dev_episodes, args.test_episodes)
        split_lookup = {ep: code for code, name in enumerate(("train", "dev", "test")) for ep in split[name]}
        sources = select_sources(a["episode_index"], a["frame_index"], a["obs_frame_indices"],
                                 a["tactile"], args.sources_per_episode, args.contact_threshold)
        if args.max_sources and len(sources) > args.max_sources:
            sources = sources[np.linspace(0, len(sources) - 1, args.max_sources, dtype=int)]
        capacity = len(sources) * (args.variants + 1)
        schema = {"rgb": (np.uint8, (2, 224, 224, 3)), "pressure": (np.float32, (44,)),
                  "anchors": (np.float32, (44, 3)), "state": (np.float32, (22,)),
                  "labels": (np.float32, (3,)), "labels_relative_palm": (np.float32, (3,)),
                  "labels_orientation_wxyz": (np.float32, (4,)), "palm": (np.float32, (4, 4)),
                  "known_cameras": (np.float32, (2, 4, 4)), "known_intrinsics": (np.float32, (2, 3, 3)),
                  "occlusion": (np.float32, (2,)), "episode": (np.int64, ()), "group": (np.int64, ()),
                  "source_row": (np.int64, ()), "variant": (np.int16, ()), "split": (np.uint8, ())}
        arrays = {name: np.lib.format.open_memmap(output / f"{name}.partial.npy", mode="w+", dtype=dtype,
                  shape=(capacity, *shape)) for name, (dtype, shape) in schema.items()}
        rng = np.random.default_rng(args.seed)
        index, rejected, no_sibling, example_records = 0, 0, 0, []
        rotation, origin = builder.base_to_world[:3, :3], builder.base_to_world[:3, 3]
        for group, cache_index in enumerate(sources):
            qpos, velocity, row, ep, frame = builder.source_state(int(cache_index))
            builder.set_state(qpos, velocity)
            baseline_penetration = builder.penetration()
            generated = 0
            for variant in range(args.variants + 1):
                accepted = False
                for attempt in range(1 if variant == 0 else args.max_attempts):
                    pose, shift = (qpos, np.zeros(3)) if variant == 0 else perturb_object(
                        qpos, builder.object_qadr, rng, args.jitter_min, args.jitter_max, args.rotation_degrees)
                    builder.set_state(pose, velocity)
                    penetration = builder.penetration()
                    if variant and penetration < min(-args.max_penetration, baseline_penetration - .002):
                        rejected += 1
                        continue
                    pressure, anchors, _ = builder.observe_native()
                    if variant and not contact_regime_allowed(pressure, args.contact_threshold, args.allow_contact_breaks):
                        rejected += 1
                        continue
                    accepted = True
                    break
                if not accepted:
                    continue
                target = world_to_base(builder.data.xpos[builder.cube], builder.base_to_world).astype(np.float32)
                palm = np.eye(4, dtype=np.float32)
                palm[:3, :3] = rotation.T @ builder.data.xmat[builder.palm].reshape(3, 3)
                palm[:3, 3] = (builder.data.xpos[builder.palm] - origin) @ rotation
                cameras, intrinsic = builder.camera_arrays()
                orientation = np.empty(4)
                builder.mj.mju_mat2Quat(orientation, (rotation.T @ builder.data.xmat[builder.cube].reshape(3, 3)).reshape(-1))
                values = {"rgb": builder.rgb(), "pressure": pressure,
                          "anchors": world_to_base(anchors, builder.base_to_world), "state": pose[:22],
                          "labels": target, "labels_relative_palm": (target - palm[:3, 3]) @ palm[:3, :3],
                          "labels_orientation_wxyz": orientation, "palm": palm,
                          "known_cameras": cameras, "known_intrinsics": intrinsic,
                          "occlusion": builder.visibility(), "episode": ep, "group": group,
                          "source_row": row, "variant": variant, "split": split_lookup[ep]}
                for name, value in values.items():
                    arrays[name][index] = value
                example_records.append({"index": index, "group": group, "episode": ep, "frame": frame,
                                        "cache_index": int(cache_index), "source_row": row, "variant": variant,
                                        "kind": "nominal_recorded_state" if variant == 0 else "static_counterfactual",
                                        "translation_delta_world_m": shift.tolist(), "minimum_contact_distance_m": penetration})
                index += 1
                generated += 1
            no_sibling += int(generated < 2)
            if (group + 1) % 10 == 0 or group + 1 == len(sources):
                print(json.dumps({"sources": group + 1, "total_sources": len(sources), "examples": index,
                                  "rejected_candidates": rejected, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        if not index:
            raise ValueError("No generated examples")
        files = {}
        for name, array in arrays.items():
            array.flush()
            np.save(output / f"{name}.npy", array[:index])
            files[name] = fingerprint(output / f"{name}.npy")
        for path in output.glob("*.partial.npy"):
            path.unlink()
        write_json(output / "examples.json", example_records)
        write_json(output / "source_provenance.json", list(builder.provenance.values()))
        counts = {name: int((np.load(output / "split.npy") == code).sum()) for code, name in enumerate(("train", "dev", "test"))}
        metadata = {"complete": True, "format_version": 1, "purpose": "spatial representation pilot, not policy demonstrations",
                    "examples": index, "source_groups": len(sources), "split_episodes": split,
                    "split_counts": counts, "seed": args.seed, "split_codes": {"train": 0, "dev": 1, "test": 2},
                    "coordinate_frame": "robot_base", "coordinate_units": "metres",
                    "camera_convention": "OpenCV camera-to-base; known_intrinsics correspond to RGB224",
                    "input_whitelist": ["rgb", "pressure", "anchors", "state", "palm", "known_cameras", "known_intrinsics"],
                    "label_only": ["labels", "labels_relative_palm", "labels_orientation_wxyz"],
                    "scoring_only": ["occlusion"], "no_gt_depth_or_mask_inputs": True,
                    "sensor_order": "unchanged native 32 fingertip taxels then 12 FSRs",
                    "pressure_semantics": "raw frozen native force proxy; not Newtons; normalize from train split only",
                    "velocity_dt": .01, "counterfactual_velocity": "same nominal physical causal qvel; no jitter/dt velocity; independent static interventions, not trajectories",
                    "allow_contact_breaks": args.allow_contact_breaks,
                    "rgb_semantics": "fresh same-state stereo render; historical training_rgb_transform 480x640->crop480->224",
                    "rejected_candidates": rejected, "groups_without_counterfactual": no_sibling,
                    "generation": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "alignment_audit": fingerprint(output / "alignment_audit.json"), "files": files,
                    "provenance": {"source_cache_manifest": fingerprint(builder.cache / "manifest.json"),
                        "source_records": fingerprint(output / "source_provenance.json"), "examples": fingerprint(output / "examples.json"),
                        "implementation": fingerprint(__file__), "native_models": fingerprint(Path(args.dynamic_root) / "eval/sim/tactile_sim/models.py"),
                        "native_geometry": fingerprint(Path(args.dynamic_root) / "readouts/geometry.npz"),
                        "object_mesh": fingerprint(builder.native.object_mesh)},
                    "elapsed_seconds": time.monotonic() - started,
                    "limitations": ["single object mesh and fixed camera scenario", "counterfactual siblings are static snapshots, not dynamically feasible rollouts",
                        "episode-disjoint split reuses an existing historical dataset; it is a new pilot split, not an untouched external benchmark",
                        "natural occlusion comes from GT segmentation only for scoring; no masks or object poses are observation inputs"]}
        if fingerprint(builder.cache / "manifest.json")["sha256"] != audit["cache_manifest"]["sha256"]:
            raise ValueError("Source cache manifest changed during generation")
        metadata["source_cache_manifest_unchanged"] = True
        write_json(output / "metadata.json", metadata)
        write_json(output / "manifest.json", metadata)
        print(json.dumps({"complete": True, "output": str(output), "examples": index, "split_counts": counts,
                          "elapsed_s": round(metadata["elapsed_seconds"], 1)}), flush=True)
    finally:
        builder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--timestamps-root", type=Path, default=machine_path("timestamps_root", default=artifact_path("inputs") / "raw_timestamps"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--dev-episodes", type=int, default=7)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--sources-per-episode", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=0, help="0 for all selected sources; a positive value makes a spread probe")
    parser.add_argument("--variants", type=int, default=3, help="counterfactual siblings per nominal source")
    parser.add_argument("--audit-count", type=int, default=16)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--contact-threshold", type=float, default=.0005)
    parser.add_argument("--allow-contact-breaks", action="store_true", help="Retain native zero-contact counterfactuals; never synthesize or relabel pressure")
    parser.add_argument("--jitter-min", type=float, default=.01)
    parser.add_argument("--jitter-max", type=float, default=.025)
    parser.add_argument("--rotation-degrees", type=float, default=12.)
    parser.add_argument("--max-penetration", type=float, default=.006)
    parser.add_argument("--max-attempts", type=int, default=12)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
