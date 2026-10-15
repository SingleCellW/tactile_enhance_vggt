#!/usr/bin/env python3
"""Small reproducible GT-depth audit; privileged geometry is ONLY for scoring.

The primary pair uses identical RGB, VGGT depth/intrinsics, and stereo-baseline
scale; only the second-camera pose differs. A third diagnostic also substitutes
known calibrated intrinsics. No GT depth enters any variant or saved cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from tactile_vggt.geometry import calibrated_similarity, unproject_depth, world_to_base


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scalar_stats(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Audit metric contains nonfinite values")
    return {"count": len(values), "mean": float(values.mean()), "p50": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def select_cases(cache, count, split, pressure_threshold):
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Audit requires a completed cache")
    obs = np.load(cache / "obs_frame_indices.npy")
    rgb_rows = np.load(cache / "rgb_source_frame_indices.npy")
    episodes = np.load(cache / "episode_index.npy", mmap_mode="r")
    frames = np.load(cache / "frame_index.npy", mmap_mode="r")
    tactile = np.load(cache / "tactile.npy", mmap_mode="r")
    active = (tactile[obs] > pressure_threshold).any(-1)
    allowed = np.ones(len(obs), bool) if split == "all" else np.isin(episodes[obs], manifest["split"][split])
    chosen = []
    for flag, requested in ((False, count // 2), (True, count - count // 2)):
        indices = np.flatnonzero(allowed & (active == flag))
        if len(indices) < requested:
            raise ValueError(f"Insufficient {flag=}: {len(indices)} < {requested}")
        chosen.extend(indices[np.linspace(0, len(indices) - 1, requested, dtype=int)])
    return [{"cache_index": int(i), "state_source_row": int(obs[i]), "rgb_source_row": int(rgb_rows[i]),
             "episode": int(episodes[obs[i]]), "frame": int(frames[obs[i]]),
             "active_contact": bool(active[i])} for i in sorted(chosen)]


def interior_mask(segmentation):
    """Exclude raster boundaries using a 3x3 same-geometry interior mask."""
    ids = segmentation[..., 0]
    types = segmentation[..., 1]
    keep = ids >= 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            keep &= (ids == np.roll(ids, (dy, dx), (-2, -1))) & (types == np.roll(types, (dy, dx), (-2, -1)))
    keep[..., 0, :] = keep[..., -1, :] = False
    keep[..., :, 0] = keep[..., :, -1] = False
    return keep


def depth_camera_points(depth, intrinsic):
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[..., 0]
    h, w = depth.shape[-2:]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack(((u - intrinsic[:, 0, 2, None, None]) * depth / intrinsic[:, 0, 0, None, None],
                     (v - intrinsic[:, 1, 2, None, None]) * depth / intrinsic[:, 1, 1, None, None], depth), -1)


def camera_to_world(points, poses):
    return np.einsum("vhwj,vij->vhwi", points, poses[:, :3, :3]) + poses[:, None, None, :3, 3]


def geometry_variants(depth, extrinsics, intrinsics, known, known_intrinsics=None):
    scale, rotation, translation, diagnostics = calibrated_similarity(extrinsics, known)
    current = scale * unproject_depth(depth, extrinsics, intrinsics) @ rotation.T + translation
    calibrated = camera_to_world(scale * depth_camera_points(depth, intrinsics), known)
    # Camera 0 fixes the global Sim3, so its points must be unchanged.
    np.testing.assert_allclose(current[0], calibrated[0], atol=2e-6, rtol=2e-6)
    variants = {"current_sim3": current, "known_extrinsics_same_depth": calibrated}
    if known_intrinsics is not None:
        variants["known_K_and_extrinsics_same_depth"] = camera_to_world(scale * depth_camera_points(depth, known_intrinsics), known)
    return variants, diagnostics


def cpu_geometry_gate(cache, threshold):
    manifest = json.loads((cache / "manifest.json").read_text())
    points = np.load(cache / "points.npy", mmap_mode="r")
    nvis = manifest["configuration"]["visual_points"]
    result = {}
    distances, actives, effective = [], [], []
    for start in range(0, len(points), 32):
        x = np.asarray(points[start:start + 32])
        d2 = ((x[:, nvis:, None, :3] - x[:, None, :nvis, :3]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * .05 ** 2)) * (d2 <= .15 ** 2)
        denom = (w * w).sum(-1)
        distances.append(np.sqrt(d2.min(-1)).reshape(-1))
        actives.append((x[:, nvis:, 6] > threshold).reshape(-1))
        effective.append(np.divide(w.sum(-1) ** 2, denom, out=np.zeros_like(denom), where=denom > 0).reshape(-1))
    d, active, ess = map(np.concatenate, (distances, actives, effective))
    for name, mask in (("all_anchors", np.ones(len(d), bool)), ("active_anchors", active)):
        result[name] = {"sampled_visual_nearest_m": scalar_stats(d[mask]),
                        "within_5cm_fraction": float((d[mask] <= .05).mean()),
                        "within_10cm_fraction": float((d[mask] <= .10).mean()),
                        "empty_local_pool_fraction": float((ess[mask] == 0).mean()),
                        "gaussian_effective_neighbors": scalar_stats(ess[mask])}
    records = [json.loads(line) for line in (cache / "geometry_diagnostics.jsonl").read_text().splitlines()]
    for key in ("second_camera_position_error_m", "second_camera_rotation_error_deg", "scale"):
        result[key] = scalar_stats([r[key] for r in records])
    result["caveat"] = "Sampled visual-point proximity/coverage is NOT GT geometry error. Active threshold uses the current cache's native force proxy."
    return result


def render_gt(spec_path, output):
    """Separate MuJoCo interpreter; no VGGT or policy imports in this worker."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from tactile_vggt.simulation import load_frozen_model, training_rgb_transform, CAMERAS
    spec = json.loads(Path(spec_path).read_text())
    model = load_frozen_model(Path(spec["dynamic_root"]) / "eval/assets/picklift", mujoco)
    data = mujoco.MjData(model)
    raw = sorted(p for p in Path(spec["raw_root"]).iterdir() if (p / "prioception").exists())
    state = np.load(Path(spec["cache"]) / "state.npy", mmap_mode="r")
    size = spec["image_size"]
    # Full-resolution cache uses 518 pixels, larger than the scene's historical
    # 480px offscreen buffer. This affects rendering allocation only.
    model.vis.global_.offwidth = max(640, size)
    model.vis.global_.offheight = max(480, size)
    renderer = mujoco.Renderer(model, height=480, width=640)
    # RGB keeps its historical multisampling context. Depth uses pixel-center
    # samples: multisample depth resolves to a sample displaced from the center,
    # producing millimetre-scale ray/depth mismatches on tilted surfaces.
    model.vis.quality.offsamples = 0
    depth_renderer = mujoco.Renderer(model, height=size, width=size)
    cube_body = model.body("cube").id
    cube_geoms = np.flatnonzero(model.geom_bodyid == cube_body)
    items, provenance = [], []
    try:
        for case in spec["cases"]:
            episode = raw[case["episode"]]
            csv_path = Path(spec["timestamps_root"]) / f"teleop_log_{episode.name}.csv"
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            qpath = episode / rows[case["frame"]]["prioception_path"]
            qpos = np.load(qpath, allow_pickle=False)
            np.testing.assert_array_equal(qpos[:22].astype(np.float32), state[case["state_source_row"]])
            mujoco.mj_resetData(model, data)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            images, depths, segments, poses, intrinsics, ray_errors = [], [], [], [], [], []
            for name in CAMERAS:
                cid = model.camera(name).id
                if np.any(model.cam_sensorsize[cid]):
                    raise ValueError("This audit's intrinsics require the frozen fovy cameras")
                pose = np.eye(4)
                pose[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
                pose[:3, 3] = data.cam_xpos[cid]
                focal = .5 * size / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
                # Pixel indices denote pixel centers. OpenGL's viewport optical
                # axis lies at (size-1)/2 in this convention, not size/2.
                intrinsic = np.array([[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1.]])
                renderer.update_scene(data, camera=name)
                images.append(training_rgb_transform(renderer.render()))
                depth_renderer.update_scene(data, camera=name)
                depth_renderer.enable_depth_rendering()
                depth = depth_renderer.render().copy()
                depth_renderer.disable_depth_rendering()
                depth_renderer.enable_segmentation_rendering()
                segmentation = depth_renderer.render().copy()
                depth_renderer.disable_segmentation_rendering()
                # Independent CPU ray intersections verify metric z-depth and
                # pixel-center convention against actual scene surfaces.
                inner = interior_mask(segmentation)
                valid = np.flatnonzero(inner & np.isfinite(depth) & (depth > 0) & (depth < 3))
                for flat in valid[np.linspace(0, len(valid) - 1, min(30, len(valid)), dtype=int)]:
                    v, u = np.unravel_index(flat, depth.shape)
                    ray_camera = np.array([(u - intrinsic[0, 2]) / focal, (v - intrinsic[1, 2]) / focal, 1.])
                    direction = pose[:3, :3] @ ray_camera
                    direction /= np.linalg.norm(direction)
                    geom_id = np.empty(1, dtype=np.int32)
                    distance = mujoco.mj_ray(model, data, pose[:3, 3].copy(), direction, None, 1, -1, geom_id)
                    if distance >= 0 and geom_id[0] == segmentation[v, u, 0]:
                        ray_errors.append(abs(distance / np.linalg.norm(ray_camera) - float(depth[v, u])))
                depths.append(depth); segments.append(segmentation); poses.append(pose); intrinsics.append(intrinsic)
            items.append({"rendered_rgb": np.stack(images), "gt_depth": np.stack(depths), "gt_segmentation": np.stack(segments),
                          "known_cameras_cv": np.stack(poses), "known_intrinsics": np.stack(intrinsics)})
            if len(ray_errors) < 10 or np.quantile(ray_errors, .95) > 1e-4:
                raise ValueError("GT depth failed independent CPU ray validation (p95 > 0.1mm or insufficient samples)")
            provenance.append({**case, "qpos_path": str(qpath), "qpos_sha256": sha(qpath),
                               "timestamp_sha256": sha(csv_path), "cpu_ray_depth_absolute_error_m": scalar_stats(ray_errors)})
    finally:
        renderer.close(); depth_renderer.close()
    np.savez_compressed(output, **{key: np.stack([item[key] for item in items]) for key in items[0]}, cube_geom_ids=cube_geoms)
    write_json(Path(output).with_suffix(".json"), {"mujoco_version": mujoco.__version__, "cases": provenance,
               "depth_multisamples": 0, "rgb_multisamples": "original scene context retained",
               "scene_sha256": sha(Path(spec["dynamic_root"]) / "eval/assets/picklift/scene.xml"),
               "renderer_source_sha256": sha(mujoco.renderer.__file__),
               "depth_semantics": "MuJoCo metric optical-axis z; direct square render at VGGT resolution, matching cropped vertical field of view", "pixel_centers": "integer indices, principal point=(size-1)/2; independently checked by mj_ray"})


def score_errors(xyz, gt_world, gt_depth, known, mask):
    camera_z = np.einsum("vhwi,vi->vhw", xyz - known[:, None, None, :3, 3], known[:, :3, 2])
    point = np.linalg.norm(xyz - gt_world, axis=-1)[mask]
    depth = np.abs(camera_z - gt_depth)[mask]
    return {"pixel_corresponded_xyz_error_m": scalar_stats(point), "optical_z_absolute_error_m": scalar_stats(depth),
            "optical_z_abs_relative_error": scalar_stats(depth / gt_depth[mask]),
            "xyz_within_1cm_fraction": float((point <= .01).mean()) if len(point) else None,
            "xyz_within_5cm_fraction": float((point <= .05).mean()) if len(point) else None}


def audit(args):
    from prepare_cache import FrozenVGGT, VideoReader
    import torch
    started = time.monotonic()
    torch.set_num_threads(4)
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((cache / "manifest.json").read_text())
    cases = select_cases(cache, args.frames, args.split, args.active_threshold)
    size = manifest["configuration"]["image_size"]
    weights_sha = sha(args.weights)
    if weights_sha != manifest["configuration"]["weights"]["sha256"]:
        raise ValueError("VGGT weights differ from the cached training geometry")
    spec = {"cache": str(cache), "cases": cases, "image_size": size, "dynamic_root": args.dynamic_root,
            "raw_root": args.raw_root, "timestamps_root": args.timestamps_root}
    write_json(output / "render_spec.json", spec)
    write_json(output / "cpu_geometry_gate.json", cpu_geometry_gate(cache, args.active_threshold))
    subprocess.run([args.worker_python, str(Path(__file__).resolve()), "--render-spec", str(output / "render_spec.json"),
                    "--render-output", str(output / "rendered_gt.npz")], check=True)
    with np.load(output / "rendered_gt.npz", allow_pickle=False) as archive:
        gt = {key: archive[key] for key in archive.files}
    rgb = gt["rendered_rgb"]
    if args.rgb_source == "recorded":
        videos = manifest["configuration"]["source"]["rgb_videos"]
        readers = [VideoReader(video["path"], manifest["frames"]) for video in videos]
        try:
            rgb = np.stack([[reader.read(case["rgb_source_row"]) for reader in readers] for case in cases])
        finally:
            for reader in readers:
                reader.close()
    np.savez_compressed(output / "rgb_inputs.npz", rgb=rgb)
    image_difference = np.abs(rgb.astype(np.float32) - gt["rendered_rgb"].astype(np.float32))
    model = FrozenVGGT(args.vggt_repo, args.weights, args.device, size)
    predictions = []
    for index in range(len(cases)):
        _, depth, confidence, extrinsics, intrinsics = model.predict([rgb[index]])
        predictions.append({"depth": depth[0], "confidence": confidence[0], "extrinsics": extrinsics[0], "intrinsics": intrinsics[0]})
        print(json.dumps({"event": "predicted", "frame": index + 1, "total": len(cases)}), flush=True)
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(output / "predictions.npz", **{key: np.stack([p[key] for p in predictions]) for key in predictions[0]})
    return score_saved(args, cases, manifest, gt, rgb, predictions, weights_sha, started)


def score_existing(args):
    """Reproduce scoring on saved predictions without running any neural model."""
    started = time.monotonic()
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    spec = json.loads((output / "render_spec.json").read_text())
    if Path(spec["cache"]).resolve() != cache:
        raise ValueError("Saved audit observations refer to a different cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(output / "rendered_gt.npz", allow_pickle=False) as archive:
        gt = {key: archive[key] for key in archive.files}
    with np.load(output / "rgb_inputs.npz", allow_pickle=False) as archive:
        rgb = archive["rgb"]
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        predictions = [{key: archive[key][i] for key in archive.files} for i in range(len(spec["cases"]))]
    return score_saved(args, spec["cases"], manifest, gt, rgb, predictions,
                       manifest["configuration"]["weights"]["sha256"], started)


def score_saved(args, cases, manifest, gt, rgb, predictions, weights_sha, started):
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    image_difference = np.abs(rgb.astype(np.float32) - gt["rendered_rgb"].astype(np.float32))
    points = np.load(cache / "points.npy", mmap_mode="r")
    base = np.asarray(manifest["configuration"]["base_to_world"])
    all_metrics, frame_reports = {}, []
    for index, (case, prediction) in enumerate(zip(cases, predictions, strict=True)):
        known, depth = gt["known_cameras_cv"][index], gt["gt_depth"][index]
        gt_world = camera_to_world(depth_camera_points(depth, gt["known_intrinsics"][index]), known)
        gt_base = world_to_base(gt_world, base)
        interior = interior_mask(gt["gt_segmentation"][index])
        workspace = (gt_base >= [-.3, -1., -.15]).all(-1) & (gt_base <= [1.2, 1., 1.2]).all(-1)
        mask = interior & workspace & np.isfinite(depth) & (depth > 0)
        confidence = prediction["confidence"]
        mask &= np.isfinite(confidence)
        threshold = np.quantile(confidence[mask], manifest["configuration"]["confidence_quantile"])
        mask &= confidence >= threshold
        cube = np.isin(gt["gt_segmentation"][index, ..., 0], gt["cube_geom_ids"])
        anchors = points[case["cache_index"], -44:, :3]
        flat = gt_base.reshape(-1, 3)
        near = np.zeros(len(flat), bool)
        for start in range(0, len(flat), 4096):
            near[start:start + 4096] = (((flat[start:start + 4096, None] - anchors[None]) ** 2).sum(-1).min(-1) <= .10 ** 2)
        near = near.reshape(depth.shape)
        variants, diagnostics = geometry_variants(prediction["depth"], prediction["extrinsics"], prediction["intrinsics"], known, gt["known_intrinsics"][index])
        frame_report = {**case, "rgb_vs_render_mae_255": float(image_difference[index].mean()), "geometry_diagnostics": diagnostics,
                        "predicted_to_known_focal_ratio_xy": (np.diagonal(prediction["intrinsics"], axis1=-2, axis2=-1)[:, :2]
                            / np.diagonal(gt["known_intrinsics"][index], axis1=-2, axis2=-1)[:, :2]).tolist(),
                        "variants": {}}
        for variant, xyz in variants.items():
            if not np.isfinite(xyz).all():
                raise ValueError("Nonfinite predicted geometry")
            frame_report["variants"][variant] = {}
            for region, region_mask in (("workspace", mask), ("cube", mask & cube), ("near_tactile_10cm", mask & near)):
                for view in ("both", "left", "right"):
                    selected = region_mask.copy()
                    if view != "both":
                        selected[1 if view == "left" else 0] = False
                    key = f"{region}/{view}"
                    result = score_errors(xyz, gt_world, depth, known, selected)
                    frame_report["variants"][variant][key] = result
                    all_metrics.setdefault(variant, {}).setdefault(key, []).append(result)
        frame_reports.append(frame_report)
    summaries = {}
    for variant, regions in all_metrics.items():
        summaries[variant] = {}
        for region, metrics in regions.items():
            summaries[variant][region] = {}
            for key in ("pixel_corresponded_xyz_error_m", "optical_z_absolute_error_m", "optical_z_abs_relative_error"):
                valid = [m[key] for m in metrics if m[key]["count"]]
                summaries[variant][region][key] = {"frames_with_pixels": len(valid), "total_pixels": sum(m["count"] for m in valid),
                    "pixel_weighted_mean": float(sum(m["mean"] * m["count"] for m in valid) / sum(m["count"] for m in valid)) if valid else None,
                    "frame_median_of_p50": float(np.median([m["p50"] for m in valid])) if valid else None,
                    "frame_median_of_p95": float(np.median([m["p95"] for m in valid])) if valid else None}
    report = {"complete": True, "audit_only": True, "gt_given_to_vggt_or_policy": False, "policy_executed": False,
              "cache": str(cache), "cache_manifest_sha256": sha(cache / "manifest.json"),
              "implementation_sha256": sha(__file__), "weights_sha256": weights_sha,
              "artifacts_sha256": {name: sha(output / name) for name in ("render_spec.json", "rendered_gt.npz", "rgb_inputs.npz", "predictions.npz")},
              "rgb_source": args.rgb_source, "rgb_vs_current_render_mae_255": scalar_stats(image_difference.mean((1, 2, 3, 4))),
              "selection": {"split": args.split, "frames": len(cases), "active_threshold_proxy_units": args.active_threshold,
                            "method": "half contact/half no-contact, evenly spaced within each stratum; diagnostic sample, not unbiased dataset estimate"},
              "shared_mask": "GT-visible fixed robot workspace, 3x3 same-geom interior, finite depth/confidence, same upper75% confidence for both variants; cube and near-FK subsets only for scoring",
              "metric": "Same-image-pixel correspondence to rendered visible GT surface: XYZ Euclidean metres and optical-axis z error. Not nearest-neighbor Chamfer; large errors cannot disappear by choosing another surface.",
              "variants": {"current_sim3": "training geometry: first known camera + predicted-to-known stereo baseline scale",
                           "known_extrinsics_same_depth": "same predicted depth, predicted K and scale; transform each view with its own known pose, isolating pose alignment",
                           "known_K_and_extrinsics_same_depth": "same predicted depth and scale; use known calibrated K and per-view pose, isolating camera-model calibration jointly"},
              "limitation": "This audit cannot establish policy success; camera-calibrated variants still inherit predicted depth and scale errors. GT depth is never used to fit any variant or written into policy caches.",
              "elapsed_seconds": time.monotonic() - started, "summary": summaries, "frames": frame_reports,
              "elapsed_seconds_scope": "CPU scoring from saved predictions" if args.score_existing else "full audit including rendering/decoding/inference/scoring",
              "render_audit": json.loads((output / "rendered_gt.json").read_text())}
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "summary": {v: {r: stats["pixel_corresponded_xyz_error_m"] for r, stats in regions.items() if r in ("workspace/both", "cube/both", "near_tactile_10cm/both")} for v, regions in summaries.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render-spec")
    p.add_argument("--render-output")
    p.add_argument("--score-existing", action="store_true", help="CPU-only: rescore the saved RGB, predictions and GT artifacts")
    p.add_argument("--cache")
    p.add_argument("--output")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--split", choices=("train", "val", "all"), default="val")
    p.add_argument("--active-threshold", type=float, default=.5)
    p.add_argument("--rgb-source", choices=("recorded", "rendered"), default="recorded")
    p.add_argument("--vggt-repo", default=str(ROOT / "vendor/vggt"))
    p.add_argument("--weights", default=str(ROOT / "weights/model.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worker-python", default="/home/wukf/miniforge3/envs/te/bin/python")
    p.add_argument("--dynamic-root", default="/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921")
    p.add_argument("--raw-root", default="/home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212")
    p.add_argument("--timestamps-root", default=str(ROOT / "inputs/raw_timestamps"))
    args = p.parse_args()
    if args.render_spec:
        render_gt(args.render_spec, args.render_output)
    else:
        if not args.cache or not args.output or args.frames < 2:
            p.error("--cache, --output and at least 2 frames required")
        score_existing(args) if args.score_existing else audit(args)
