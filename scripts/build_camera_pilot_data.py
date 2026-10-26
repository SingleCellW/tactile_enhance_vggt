#!/usr/bin/env python3
"""Camera-only Stage1.b augmentation of the frozen 775 physical snapshots.

No simulation steps, native tactile inference, VGGT, or policy inference run here.
Object/robot states and native observations are inherited from the SHA-bound
Stage1.a source. Camera transforms are calibration inputs; object poses and
segmentation are reconstruction/visibility-audit labels only.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from build_token_pilot_data import SnapshotBuilder, fingerprint, write_json
from tactile_vggt.geometry import world_to_base
from tactile_vggt.simulation import CAMERAS, load_frozen_model

DOMAIN_CODES = {"nominal": 0, "near": 1, "strong": 2}
PHYSICAL_ARRAYS = ("pressure", "anchors", "state", "labels", "labels_relative_palm",
                   "labels_orientation_wxyz", "palm", "episode", "group", "source_row", "variant", "split")


def _camera_check(cameras):
    cameras = np.asarray(cameras, dtype=np.float64)
    if cameras.shape != (2, 4, 4) or not np.isfinite(cameras).all():
        raise ValueError("Expected finite stereo camera-to-base [2,4,4]")
    if np.linalg.norm(cameras[1, :3, 3] - cameras[0, :3, 3]) < 1e-5:
        raise ValueError("Degenerate stereo baseline")
    if not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid camera homogeneous row")
    return cameras


def apply_rigid_delta(cameras, delta):
    cameras = _camera_check(cameras)
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (4, 4) or not np.isfinite(delta).all():
        raise ValueError("Expected finite rigid delta [4,4]")
    if not np.allclose(delta[:3, :3].T @ delta[:3, :3], np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(delta[:3, :3]), 1, atol=1e-7):
        raise ValueError("Rig delta must be a rotation, not a scale or reflection")
    return delta[None] @ cameras


def rigid_stereo_transform(cameras, translation, axis, angle_degrees):
    """Rotate a stereo rig about its midpoint, then translate that midpoint."""
    cameras = _camera_check(cameras)
    translation, axis = np.asarray(translation, float), np.asarray(axis, float)
    if translation.shape != (3,) or axis.shape != (3,) or not np.isfinite(np.r_[translation, axis, angle_degrees]).all():
        raise ValueError("Expected finite translation/axis/angle")
    norm = np.linalg.norm(axis)
    if norm < 1e-12 and abs(angle_degrees) > 1e-12:
        raise ValueError("Nonzero rotation needs an axis")
    axis = axis / max(norm, 1e-12)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.deg2rad(angle_degrees)
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    pivot = cameras[:, :3, 3].mean(0)
    delta = np.eye(4)
    delta[:3, :3] = rotation
    delta[:3, 3] = pivot + translation - rotation @ pivot
    return apply_rigid_delta(cameras, delta), delta


def audit_rigid_pair(before, after):
    before, after = _camera_check(before), _camera_check(after)
    baseline = lambda c: np.linalg.norm(c[1, :3, 3] - c[0, :3, 3])
    return {"baseline_error_m": float(abs(baseline(before) - baseline(after))),
            "relative_transform_max_abs": float(np.max(np.abs(
                np.linalg.inv(before[0]) @ before[1] - np.linalg.inv(after[0]) @ after[1])))}


def normalize_first_camera(points, cameras):
    cameras = _camera_check(cameras)
    baseline = np.linalg.norm(cameras[1, :3, 3] - cameras[0, :3, 3])
    return (np.asarray(points) - cameras[0, :3, 3]) @ cameras[0, :3, :3] / baseline


def relative_target_first_camera(labels, palm, cameras):
    cameras = _camera_check(cameras)
    baseline = np.linalg.norm(cameras[1, :3, 3] - cameras[0, :3, 3])
    return (np.asarray(labels) - np.asarray(palm)[:3, 3]) @ cameras[0, :3, :3] / baseline


def make_variant_plan(groups, episodes, splits, near_views=3, strong_views=3):
    groups, episodes, splits = map(np.asarray, (groups, episodes, splits))
    if groups.ndim != 1 or groups.shape != episodes.shape or groups.shape != splits.shape:
        raise ValueError("groups, episodes and splits must be aligned vectors")
    if near_views < 1 or strong_views < 1 or not np.isin(splits, [0, 1, 2]).all():
        raise ValueError("Need positive near/strong counts and canonical split codes")
    for group in np.unique(groups):
        selected = groups == group
        if len(np.unique(episodes[selected])) != 1 or len(np.unique(splits[selected])) != 1:
            raise ValueError("A physical source group cannot cross episodes or splits")
    for episode in np.unique(episodes):
        if len(np.unique(splits[episodes == episode])) != 1:
            raise ValueError("An episode cannot cross splits")
    result = []
    for physical, (group, episode, split) in enumerate(zip(groups, episodes, splits)):
        variants = [(0, 0)] + [(v, 1) for v in range(1, near_views + 1)]
        if split == 2:
            variants += [(near_views + v, 2) for v in range(1, strong_views + 1)]
        for variant, domain in variants:
            result.append({"index": len(result), "physical_sample": physical, "group": int(group),
                           "episode": int(episode), "split": int(split),
                           "camera_variant": variant, "camera_domain": domain})
    return result


def apply_camera_poses(model, data, camera_ids, cameras_base, base_to_world, mujoco):
    """Write fixed-camera local poses, including non-world parent bodies."""
    cameras = _camera_check(cameras_base)
    world = np.asarray(base_to_world) @ cameras
    flip = np.diag([1., -1., -1.])
    qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), float(data.time)
    for cid, pose in zip(camera_ids, world):
        if int(model.cam_mode[cid]) != 0:
            raise ValueError("Only fixed MuJoCo cameras are supported")
        parent = int(model.cam_bodyid[cid])
        rotation = data.xmat[parent].reshape(3, 3)
        model.cam_pos[cid] = rotation.T @ (pose[:3, 3] - data.xpos[parent])
        local_rotation = rotation.T @ pose[:3, :3] @ flip
        mujoco.mju_mat2Quat(model.cam_quat[cid], local_rotation.reshape(-1))
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    if data.time != timestamp:
        raise ValueError("Camera update advanced physical time")


class CameraSnapshotBuilder(SnapshotBuilder):
    """Reuse historical state/RGB helpers without constructing native tactile."""
    def __init__(self, args, source_manifest):
        import mujoco
        if mujoco.__version__ != "3.4.0":
            raise RuntimeError(f"Expected frozen MuJoCo 3.4.0, got {mujoco.__version__}; use the original te environment")
        self.args, self.mj = args, mujoco
        self.cache = Path(args.cache).resolve()
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        expected = source_manifest["provenance"]["source_cache_manifest"]["sha256"]
        if fingerprint(self.cache / "manifest.json")["sha256"] != expected:
            raise ValueError("Source cache manifest differs from the frozen 775-source provenance")
        self.arrays = {name: np.load(self.cache / f"{name}.npy", mmap_mode="r") for name in
                       ("state", "episode_index", "frame_index", "obs_frame_indices")}
        self.base_to_world = np.asarray(self.manifest["configuration"]["base_to_world"], np.float64)
        self.model = load_frozen_model(Path(args.dynamic_root) / "eval/assets/picklift", mujoco)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.seg_renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.cube, self.palm = self.model.body("cube").id, self.model.body("palm").id
        self.object_qadr = int(self.model.jnt_qposadr[self.model.joint("cube_freejoint").id])
        self.cube_geoms = np.flatnonzero(self.model.geom_bodyid == self.cube)
        self.camera_ids = [self.model.camera(name).id for name in CAMERAS]
        native_assets = {"native_models": Path(args.dynamic_root) / "eval/sim/tactile_sim/models.py",
                         "native_geometry": Path(args.dynamic_root) / "readouts/geometry.npz",
                         "object_mesh": Path(args.dynamic_root) / "eval/assets/rotate/meshes/0.07cube.obj"}
        self.asset_provenance = {}
        for name, path in native_assets.items():
            actual = fingerprint(path)
            if actual["sha256"] != source_manifest["provenance"][name]["sha256"]:
                raise ValueError(f"Frozen asset differs from the source 775: {name}")
            self.asset_provenance[name] = actual
        self.asset_provenance["render_assets"] = [fingerprint(p) for p in sorted(
            (Path(args.dynamic_root) / "eval/assets/picklift").rglob("*")) if p.is_file()]
        self.raw = sorted(p for p in Path(args.raw_root).iterdir() if (p / "prioception").exists())
        if len(self.raw) != len(np.unique(self.arrays["episode_index"])):
            raise ValueError("Raw episode directories do not match source cache")
        self.rows, self.provenance = {}, {}

    def reconstruct(self, record, source, physical):
        qpos, qvel, row, episode, frame = self.source_state(record["cache_index"])
        if (row, episode, frame) != (record["source_row"], record["episode"], record["frame"]):
            raise ValueError("Source record no longer matches recorded qpos")
        self.set_state(qpos, qvel)
        target_world = self.base_to_world[:3, :3] @ source["labels"][physical] + self.base_to_world[:3, 3]
        qpos[self.object_qadr:self.object_qadr + 3] += target_world - self.data.xpos[self.cube]
        rotation_base = np.empty(9)
        self.mj.mju_quat2Mat(rotation_base, source["labels_orientation_wxyz"][physical].astype(float))
        rotation_world = self.base_to_world[:3, :3] @ rotation_base.reshape(3, 3)
        self.mj.mju_mat2Quat(qpos[self.object_qadr + 3:self.object_qadr + 7], rotation_world.reshape(-1))
        self.set_state(qpos, qvel)
        actual = world_to_base(self.data.xpos[self.cube], self.base_to_world)
        palm = np.eye(4)
        palm[:3, :3] = self.base_to_world[:3, :3].T @ self.data.xmat[self.palm].reshape(3, 3)
        palm[:3, 3] = world_to_base(self.data.xpos[self.palm], self.base_to_world)
        errors = {"object_center_m": float(np.max(np.abs(actual - source["labels"][physical]))),
                  "palm_pose_max_abs": float(np.max(np.abs(palm - source["palm"][physical]))),
                  "object_rotation_max_abs": float(np.max(np.abs(rotation_world - self.data.xmat[self.cube].reshape(3, 3))))}
        if max(errors.values()) > 2e-6:
            raise ValueError(f"Cannot reconstruct source physical sample {physical}: {errors}")
        np.testing.assert_array_equal(qpos[:22].astype(np.float32), source["state"][physical])
        return qpos, qvel, errors

    def visibility_counts(self):
        """Amodal in-frame pixels gate views; natural occlusion is retained."""
        result = []
        groups = self.model.geom_group.copy()
        option = self.mj.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[5] = 1
        self.seg_renderer.enable_segmentation_rendering()
        try:
            for name in CAMERAS:
                self.seg_renderer.update_scene(self.data, camera=name)
                visible = int(np.isin(self.seg_renderer.render()[..., 0], self.cube_geoms).sum())
                self.model.geom_group[:] = 0
                self.model.geom_group[self.cube_geoms] = 5
                self.seg_renderer.update_scene(self.data, camera=name, scene_option=option)
                amodal = int(np.isin(self.seg_renderer.render()[..., 0], self.cube_geoms).sum())
                self.model.geom_group[:] = groups
                result.append((visible, amodal))
        finally:
            self.model.geom_group[:] = groups
            self.seg_renderer.disable_segmentation_rendering()
        return np.asarray(result, np.int32)


def _sample_transform(cameras, rng, minimum, maximum, angle_min, angle_max):
    direction, axis = rng.normal(size=(2, 3))
    direction /= np.linalg.norm(direction)
    axis /= np.linalg.norm(axis)
    translation = direction * rng.uniform(minimum, maximum)
    angle = float(rng.uniform(angle_min, angle_max))
    moved, delta = rigid_stereo_transform(cameras, translation, axis, angle)
    return moved, delta, {"midpoint_translation_base_m": translation.tolist(),
        "midpoint_translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_axis_base": axis.tolist(), "rotation_angle_deg": angle,
        "camera_center_displacements_m": np.linalg.norm(moved[:, :3, 3] - cameras[:, :3, 3], axis=-1).tolist()}


def build(args):
    if args.max_attempts < 1 or args.min_amodal_pixels < 1:
        raise ValueError("Positive attempts and amodal pixel gate required")
    for low, high in ((args.near_translation_min, args.near_translation_max),
                      (args.strong_translation_min, args.strong_translation_max),
                      (args.near_angle_min, args.near_angle_max), (args.strong_angle_min, args.strong_angle_max)):
        if not 0 <= low <= high:
            raise ValueError("Invalid augmentation bounds")
    source_root, output = Path(args.source_data).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    manifest_path = source_root / "manifest.json"
    source_manifest_before = fingerprint(manifest_path)
    source_manifest = json.loads(manifest_path.read_text())
    if not source_manifest.get("complete") or source_manifest.get("coordinate_frame") != "robot_base":
        raise ValueError("Complete robot-base Stage1.a source required")
    source = {}
    for name, file in source_manifest["files"].items():
        path = source_root / f"{name}.npy"
        if fingerprint(path)["sha256"] != file["sha256"]:
            raise ValueError(f"Source array checksum mismatch: {name}")
        source[name] = np.load(path, mmap_mode="r")
    records_path = source_root / "examples.json"
    if fingerprint(records_path)["sha256"] != source_manifest["provenance"]["examples"]["sha256"]:
        raise ValueError("Source example records checksum mismatch")
    records = json.loads(records_path.read_text())
    if len(records) != len(source["episode"]):
        raise ValueError("Source example/array lengths differ")
    for physical, record in enumerate(records):
        if record["index"] != physical or any(int(source[name][physical]) != record[name]
            for name in ("episode", "group", "source_row", "variant")):
            raise ValueError("Source records/array row identity mismatch")
    selected_groups = np.unique(source["group"])
    if args.max_groups:
        if args.max_groups < 3:
            raise ValueError("A smoke subset needs max-groups>=3 to include every split")
        first = [int(source["group"][np.flatnonzero(source["split"] == s)[0]]) for s in range(3)]
        spread = selected_groups[np.linspace(0, len(selected_groups) - 1, min(args.max_groups, len(selected_groups)), dtype=int)].tolist()
        selected_groups = np.array(list(dict.fromkeys(first + spread))[:args.max_groups])
    physical_ids = np.flatnonzero(np.isin(source["group"], selected_groups))
    local = {name: np.asarray(source[name][physical_ids]) for name in ("group", "episode", "split")}
    plan = make_variant_plan(local["group"], local["episode"], local["split"], args.near_views, args.strong_views)
    for record in plan:
        record["physical_sample"] = int(physical_ids[record["physical_sample"]])
    output.mkdir(parents=True)
    (output / "builder_source.py").write_text(Path(__file__).read_text())
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    builder = CameraSnapshotBuilder(args, source_manifest)
    cache_manifest_before = fingerprint(builder.cache / "manifest.json")
    arrays, examples, transform_records = {}, [], []
    audit = {"complete": False, "physical_state_unchanged": True, "native_pressure_recomputed": False,
             "camera_only_no_mj_step": True, "max_baseline_error_m": 0., "max_relative_transform_error": 0.,
             "max_camera_pose_error": 0., "max_intrinsics_error": 0., "max_reconstruction_error": {},
             "max_anchor_roundtrip_error_m": 0., "max_target_roundtrip_error_m": 0.,
             "nominal_rgb_render_mae": [], "rejected_visibility_candidates": 0}
    try:
        schema = {name: (source[name].dtype, source[name].shape[1:]) for name in PHYSICAL_ARRAYS}
        schema.update({name: (source[name].dtype, source[name].shape[1:]) for name in
                       ("rgb", "known_cameras", "known_intrinsics", "occlusion")})
        schema.update({"physical_sample": (np.int64, ()), "camera_variant": (np.int16, ()),
            "camera_domain": (np.uint8, ()), "rig_delta_base": (np.float64, (4, 4)),
            "qpos": (np.float64, (builder.model.nq,)), "qvel": (np.float64, (builder.model.nv,)),
            "visible_pixels": (np.int32, (2,)), "amodal_pixels": (np.int32, (2,))})
        for name, (dtype, shape) in schema.items():
            arrays[name] = np.lib.format.open_memmap(output / f"{name}.partial.npy", mode="w+", dtype=dtype, shape=(len(plan), *shape))
        for group_number, group in enumerate(sorted(selected_groups)):
            physicals = physical_ids[source["group"][physical_ids] == group]
            split = int(source["split"][physicals[0]])
            original_cameras = np.asarray(source["known_cameras"][physicals[0]], float)
            for physical in physicals:
                np.testing.assert_allclose(source["known_cameras"][physical], original_cameras, atol=1e-7, rtol=0)
            states = {}
            for physical in physicals:
                qpos, qvel, errors = builder.reconstruct(records[int(physical)], source, int(physical))
                states[int(physical)] = (qpos, qvel)
                for key, value in errors.items():
                    audit["max_reconstruction_error"][key] = max(audit["max_reconstruction_error"].get(key, 0.), value)
            variants = sorted({r["camera_variant"] for r in plan if r["group"] == group})
            for variant in variants:
                domain = 0 if variant == 0 else (1 if variant <= args.near_views else 2)
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, split, int(group), int(variant)]))
                accepted = False
                for attempt in range(1 if domain == 0 else args.max_attempts):
                    if domain == 0:
                        cameras, delta = original_cameras, np.eye(4)
                        change = {"midpoint_translation_base_m": [0., 0., 0.], "midpoint_translation_norm_m": 0.,
                                  "rotation_axis_base": [0., 0., 1.], "rotation_angle_deg": 0., "camera_center_displacements_m": [0., 0.]}
                    else:
                        prefix = "near" if domain == 1 else "strong"
                        cameras, delta, change = _sample_transform(original_cameras, rng,
                            getattr(args, f"{prefix}_translation_min"), getattr(args, f"{prefix}_translation_max"),
                            getattr(args, f"{prefix}_angle_min"), getattr(args, f"{prefix}_angle_max"))
                    visibility = {}
                    for physical in physicals:
                        builder.set_state(*states[int(physical)])
                        apply_camera_poses(builder.model, builder.data, builder.camera_ids, cameras, builder.base_to_world, builder.mj)
                        visibility[int(physical)] = builder.visibility_counts()
                    if all((counts[:, 1] >= args.min_amodal_pixels).all() for counts in visibility.values()):
                        accepted = True
                        break
                    audit["rejected_visibility_candidates"] += 1
                if not accepted:
                    raise ValueError(f"No stereo rig candidate with an in-frame amodal object projection for group={group}, variant={variant}; no samples were silently dropped")
                rig_audit = audit_rigid_pair(original_cameras, cameras)
                audit["max_baseline_error_m"] = max(audit["max_baseline_error_m"], rig_audit["baseline_error_m"])
                audit["max_relative_transform_error"] = max(audit["max_relative_transform_error"], rig_audit["relative_transform_max_abs"])
                transform_records.append({"group": int(group), "split": split, "camera_variant": variant,
                    "camera_domain": domain, "attempts": attempt + 1, "rig_delta_base": delta.tolist(), **change})
                for physical in physicals:
                    record = next(r for r in plan if r["physical_sample"] == physical and r["camera_variant"] == variant)
                    index = record["index"]
                    builder.set_state(*states[int(physical)])
                    before = (builder.data.qpos.copy(), builder.data.qvel.copy(), builder.data.xpos.copy(), builder.data.xmat.copy())
                    apply_camera_poses(builder.model, builder.data, builder.camera_ids, cameras, builder.base_to_world, builder.mj)
                    for actual, expected in zip((builder.data.qpos, builder.data.qvel, builder.data.xpos, builder.data.xmat), before):
                        np.testing.assert_array_equal(actual, expected)
                    actual_cameras, intrinsics = builder.camera_arrays()
                    camera_error = float(np.max(np.abs(actual_cameras - cameras)))
                    intrinsic_error = float(np.max(np.abs(intrinsics - source["known_intrinsics"][physical])))
                    audit["max_camera_pose_error"] = max(audit["max_camera_pose_error"], camera_error)
                    audit["max_intrinsics_error"] = max(audit["max_intrinsics_error"], intrinsic_error)
                    if camera_error > 2e-6 or intrinsic_error > 2e-5:
                        raise ValueError("Rendered camera/K differs from saved calibration")
                    # Audit the exact downstream first-camera/baseline convention
                    # on actual source anchors and labels, without fitting GT.
                    baseline = np.linalg.norm(cameras[1, :3, 3] - cameras[0, :3, 3])
                    normalized = normalize_first_camera(source["anchors"][physical], cameras)
                    recovered = normalized * baseline @ cameras[0, :3, :3].T + cameras[0, :3, 3]
                    relative = relative_target_first_camera(source["labels"][physical], source["palm"][physical], cameras)
                    recovered_target = relative * baseline @ cameras[0, :3, :3].T + source["palm"][physical, :3, 3]
                    anchor_error = float(np.linalg.norm(recovered - source["anchors"][physical], axis=-1).max())
                    target_error = float(np.linalg.norm(recovered_target - source["labels"][physical]))
                    audit["max_anchor_roundtrip_error_m"] = max(audit["max_anchor_roundtrip_error_m"], anchor_error)
                    audit["max_target_roundtrip_error_m"] = max(audit["max_target_roundtrip_error_m"], target_error)
                    if max(anchor_error, target_error) > 2e-6:
                        raise ValueError("First-camera normalization roundtrip failed")
                    rgb = source["rgb"][physical] if domain == 0 else builder.rgb()
                    if domain == 0 and (len(audit["nominal_rgb_render_mae"]) < args.audit_count):
                        mae = float(np.abs(builder.rgb().astype(float) - rgb.astype(float)).mean())
                        audit["nominal_rgb_render_mae"].append({"physical_sample": int(physical), "mae_255": mae})
                        if mae > args.nominal_rgb_mae_max:
                            raise ValueError(f"Nominal reconstruction RGB differs from frozen source: MAE={mae}")
                    counts = visibility[int(physical)]
                    values = {name: source[name][physical] for name in PHYSICAL_ARRAYS}
                    values.update({"rgb": rgb, "known_cameras": cameras, "known_intrinsics": intrinsics,
                        "occlusion": 1. - np.minimum(counts[:, 0] / np.maximum(counts[:, 1], 1), 1.),
                        "physical_sample": physical, "camera_variant": variant, "camera_domain": domain,
                        "rig_delta_base": delta, "qpos": states[int(physical)][0], "qvel": states[int(physical)][1],
                        "visible_pixels": counts[:, 0], "amodal_pixels": counts[:, 1]})
                    for name, value in values.items():
                        arrays[name][index] = value
                    examples.append({**record, "source_example": records[int(physical)],
                        "kind": ("nominal_camera", "near_camera", "strong_camera_test_only")[domain],
                        "visibility_passed": True, "rig_motion": change})
            print(json.dumps({"source_groups": group_number + 1, "total_groups": len(selected_groups),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        # Prove that every view retains its exact source physical observations.
        ids = np.array([r["physical_sample"] for r in plan])
        for name in PHYSICAL_ARRAYS:
            np.testing.assert_array_equal(arrays[name], source[name][ids])
        for physical in physical_ids:
            rows = np.flatnonzero(ids == physical)
            for name in ("qpos", "qvel"):
                np.testing.assert_array_equal(arrays[name][rows], np.broadcast_to(arrays[name][rows[0]], arrays[name][rows].shape))
        files = {}
        for name, array in arrays.items():
            array.flush()
            os.replace(output / f"{name}.partial.npy", output / f"{name}.npy")
            files[name] = fingerprint(output / f"{name}.npy")
        examples.sort(key=lambda r: r["index"])
        write_json(output / "examples.json", examples)
        write_json(output / "rig_transforms.json", transform_records)
        write_json(output / "source_provenance.json", list(builder.provenance.values()))
        audit.update({"complete": True, "passed": True, "physical_arrays_equal_source": list(PHYSICAL_ARRAYS),
            "native_pressure_max_difference": 0., "qpos_qvel_identical_across_camera_views": True,
            "visibility_rule": f"Both amodal object masks have >= {args.min_amodal_pixels} in-frame pixels; naturally occluded objects retained",
            "nominal_rgb_semantics": "Exact source RGB retained; fresh renders audited only", "mj_step_calls": 0,
            "reconstruction_limit": "Old object pose stored in float32 labels; reconstructed qpos agrees within audited tolerance, not a bitwise archive of old qpos"})
        write_json(output / "coordinate_audit.json", audit)
        if fingerprint(manifest_path)["sha256"] != source_manifest_before["sha256"]:
            raise ValueError("Source manifest changed during generation")
        if fingerprint(builder.cache / "manifest.json")["sha256"] != cache_manifest_before["sha256"]:
            raise ValueError("Source cache manifest changed during generation")
        baseline = np.linalg.norm(source["known_cameras"][:, 1, :3, 3] - source["known_cameras"][:, 0, :3, 3], axis=-1)
        metadata = {"complete": True, "format_version": 2, "purpose": "Stage1.b static camera-robust spatial representation pilot; not policy demonstrations",
            "examples": len(plan), "physical_examples": len(physical_ids), "source_groups": len(selected_groups),
            "full_source_coverage": not bool(args.max_groups), "split_episodes": source_manifest["split_episodes"],
            "split_codes": {"train": 0, "dev": 1, "test": 2}, "camera_domain_codes": DOMAIN_CODES,
            "camera_variant_semantics": f"0 nominal; 1..{args.near_views} near; {args.near_views + 1}..{args.near_views + args.strong_views} strong test only",
            "split_counts": {name: sum(r["split"] == i for r in plan) for i, name in enumerate(("train", "dev", "test"))},
            "domain_counts": {name: sum(r["camera_domain"] == i for r in plan) for name, i in DOMAIN_CODES.items()},
            "coordinate_frame": "robot_base", "coordinate_units": "metres",
            "camera_convention": "OpenCV camera-to-base; known_intrinsics correspond to RGB224",
            "reference_baseline_m": float(baseline.mean()), "base_to_world": builder.base_to_world.tolist(),
            "rig_augmentation": "Common SE(3) applied to both cameras about stereo midpoint; fixed baseline/relative stereo transform. Translation bounds describe midpoint motion; each camera's total displacement is separately recorded.",
            "pairing": {"cross_view": ["physical_sample"], "object_displacement": ["episode", "group", "camera_variant"],
                        "object_variant": "variant retains original object intervention; camera_variant is separate"},
            "training_protocol": "Only original train episodes; fixed uses nominal, aug uses nominal+3near (25%/75%). Dev nominal+near only; strong domain exists only in test.",
            "input_whitelist": source_manifest["input_whitelist"], "label_only": source_manifest["label_only"] + ["qpos", "qvel"],
            "scoring_only": ["occlusion", "visible_pixels", "amodal_pixels"], "no_gt_depth_or_mask_inputs": True,
            "pressure_semantics": source_manifest["pressure_semantics"], "sensor_order": source_manifest["sensor_order"],
            "velocity_dt": .01, "seed": args.seed, "files": files,
            "generation": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_manifests_unchanged": True,
            "pair_counts": {"cross_view_nominal_to_near": len(physical_ids) * args.near_views,
                "cross_view_nominal_to_strong": int((source["split"][physical_ids] == 2).sum()) * args.strong_views,
                "object_displacement_per_camera": {str(int(g)): int((source["group"][physical_ids] == g).sum()) - 1
                    for g in selected_groups}},
            "provenance": {"source_data_manifest": source_manifest_before, "source_cache_manifest": cache_manifest_before,
                "source_examples": fingerprint(records_path), "source_records": fingerprint(output / "source_provenance.json"),
                "implementation": fingerprint(__file__), "examples": fingerprint(output / "examples.json"),
                "rig_transforms": fingerprint(output / "rig_transforms.json"), "coordinate_audit": fingerprint(output / "coordinate_audit.json"),
                **builder.asset_provenance},
            "elapsed_seconds": time.monotonic() - started,
            "limitations": ["same object/physical snapshots as Stage1.a; not new dynamics or expert demonstrations",
                "reconstructed original object poses have float32 label precision; native pressure/FK observations copied exactly",
                "amodal in-frame visibility is a generation gate; naturally hidden surfaces are not reconstructed GT inputs",
                "unseen camera samples share prior source episodes and task; not an independent real-world generalization benchmark"]}
        write_json(output / "metadata.json", metadata)
        write_json(output / "manifest.json", metadata)
        print(json.dumps({"complete": True, "output": str(output), "examples": len(plan), "split_counts": metadata["split_counts"]}), flush=True)
    finally:
        builder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--timestamps-root", type=Path, default=ROOT / "inputs/raw_timestamps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--near-views", type=int, default=3)
    parser.add_argument("--strong-views", type=int, default=3)
    parser.add_argument("--near-translation-min", type=float, default=.01)
    parser.add_argument("--near-translation-max", type=float, default=.02)
    parser.add_argument("--near-angle-min", type=float, default=1.)
    parser.add_argument("--near-angle-max", type=float, default=2.)
    parser.add_argument("--strong-translation-min", type=float, default=.04)
    parser.add_argument("--strong-translation-max", type=float, default=.05)
    parser.add_argument("--strong-angle-min", type=float, default=4.)
    parser.add_argument("--strong-angle-max", type=float, default=5.)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--min-amodal-pixels", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=16)
    parser.add_argument("--nominal-rgb-mae-max", type=float, default=1.)
    parser.add_argument("--max-groups", type=int, default=0, help="Nonzero creates a visibly marked small smoke subset; 0 uses every physical source")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
