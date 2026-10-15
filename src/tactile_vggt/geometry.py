"""Calibrated geometry for frozen VGGT. No simulator depth is consumed here."""
from __future__ import annotations

import numpy as np


def mujoco_camera_to_opencv(camera_to_world: np.ndarray) -> np.ndarray:
    """MuJoCo camera x-right/y-up/-z-forward -> OpenCV x-right/y-down/z-forward."""
    result = np.asarray(camera_to_world, dtype=np.float64).copy()
    if result.shape[-2:] != (4, 4):
        raise ValueError("Expected (...,4,4) MuJoCo camera-to-world transforms")
    result[..., :3, :3] = result[..., :3, :3] @ np.diag([1.0, -1.0, -1.0])
    return result


def invert_extrinsics(world_to_camera: np.ndarray) -> np.ndarray:
    ext = np.asarray(world_to_camera, dtype=np.float64)
    if ext.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("Expected (...,3,4) or (...,4,4) extrinsics")
    result = np.broadcast_to(np.eye(4), ext.shape[:-2] + (4, 4)).copy()
    result[..., :3, :3] = np.swapaxes(ext[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], ext[..., :3, 3])
    return result


def calibrated_similarity(predicted_extrinsics: np.ndarray, known_cameras_cv: np.ndarray):
    """Fix rotation/origin with camera 0 and metric scale with the two-camera baseline.

    Inputs are VGGT OpenCV world-to-camera and known OpenCV camera-to-world.
    The second camera's residual is diagnostic, not silently corrected with GT depth.
    Returns s,R,t for row-vector points: ``s * points @ R.T + t``.
    """
    predicted = invert_extrinsics(predicted_extrinsics)
    known = np.asarray(known_cameras_cv, dtype=np.float64)
    if predicted.shape != (2, 4, 4) or known.shape != (2, 4, 4):
        raise ValueError("Exactly two synchronized camera poses are required")
    pred_baseline = np.linalg.norm(predicted[1, :3, 3] - predicted[0, :3, 3])
    true_baseline = np.linalg.norm(known[1, :3, 3] - known[0, :3, 3])
    if not np.isfinite(pred_baseline) or pred_baseline < 1e-6 or true_baseline < 1e-6:
        raise ValueError("Degenerate predicted or calibrated camera baseline")
    scale = float(true_baseline / pred_baseline)
    rotation = known[0, :3, :3] @ predicted[0, :3, :3].T
    translation = known[0, :3, 3] - scale * rotation @ predicted[0, :3, 3]
    aligned_centres = scale * predicted[:, :3, 3] @ rotation.T + translation
    relative_rotation = known[1, :3, :3].T @ rotation @ predicted[1, :3, :3]
    diagnostics = {
        "scale": scale,
        "predicted_baseline": float(pred_baseline),
        "known_baseline_m": float(true_baseline),
        "second_camera_position_error_m": float(np.linalg.norm(aligned_centres[1] - known[1, :3, 3])),
        "second_camera_rotation_error_deg": float(np.rad2deg(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))),
    }
    return scale, rotation, translation, diagnostics


def unproject_depth(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject predicted depth using predicted OpenCV cameras into VGGT's frame."""
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError("Expected depth [views,height,width]")
    _, height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    k = np.asarray(intrinsics)
    camera = np.stack(((u - k[:, 0, 2, None, None]) * depth / k[:, 0, 0, None, None],
                       (v - k[:, 1, 2, None, None]) * depth / k[:, 1, 1, None, None], depth), axis=-1)
    c2w = invert_extrinsics(extrinsics)
    return np.einsum("vhwj,vij->vhwi", camera, c2w[:, :3, :3]) + c2w[:, None, None, :3, 3]


def world_to_base(points: np.ndarray, base_to_world: np.ndarray) -> np.ndarray:
    transform = np.asarray(base_to_world, dtype=np.float64)
    return (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]


def build_point_tensor(world_points, rgb, confidence, anchors_world, pressure, *,
                       base_to_world=None, visual_count=512, confidence_quantile=0.25,
                       workspace=((-0.3, -1.0, -0.15), (1.2, 1.0, 1.2)), seed=0):
    """Return [visual_count+44,8]: xyz(base metres), RGB[0,1], pressure, is_tactile.

    Confidence removes the lowest-confidence finite points. Workspace is fixed in
    robot-base coordinates, never derived from ground-truth object poses.
    """
    transform = np.eye(4) if base_to_world is None else base_to_world
    xyz = world_to_base(np.asarray(world_points).reshape(-1, 3), transform)
    colors = np.asarray(rgb).reshape(-1, 3).astype(np.float32)
    conf = np.asarray(confidence).reshape(-1)
    anchors = world_to_base(np.asarray(anchors_world), transform)
    pressure = np.asarray(pressure).reshape(-1)
    if anchors.shape != (44, 3) or pressure.shape != (44,):
        raise ValueError("Corrected 44 anchors and pressures must retain sensor order")
    if not np.isfinite(anchors).all() or not np.isfinite(pressure).all():
        raise ValueError("Nonfinite tactile observations")
    valid = np.isfinite(xyz).all(1) & np.isfinite(conf) & np.isfinite(colors).all(1)
    valid &= (xyz >= np.asarray(workspace[0])).all(1) & (xyz <= np.asarray(workspace[1])).all(1)
    if valid.any():
        valid &= conf >= np.quantile(conf[valid], confidence_quantile)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError("No confident VGGT points inside the fixed workspace; inspect camera alignment")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=visual_count, replace=len(indices) < visual_count)
    result = np.zeros((visual_count + 44, 8), dtype=np.float32)
    result[:visual_count, :3] = xyz[chosen]
    result[:visual_count, 3:6] = np.clip(colors[chosen], 0, 1)
    result[visual_count:, :3] = anchors
    result[visual_count:, 6] = np.maximum(pressure, 0)
    result[visual_count:, 7] = 1
    return result
