import importlib.util
from pathlib import Path

import numpy as np
from argparse import Namespace
import json
import sys

spec = importlib.util.spec_from_file_location("audit_geometry", Path(__file__).resolve().parents[1] / "scripts/audit_geometry.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_render_worker_uses_spec_without_requiring_machine_cli_paths(tmp_path, monkeypatch):
    spec_path, output = tmp_path / "spec.json", tmp_path / "render.npz"
    calls = []
    monkeypatch.setattr(audit, "render_gt", lambda spec, dest: calls.append((spec, dest)))
    monkeypatch.setattr(audit, "require_machine_arguments", lambda *args: (_ for _ in ()).throw(AssertionError("worker must use its spec")))
    monkeypatch.setattr(sys, "argv", ["audit_geometry.py", "--render-spec", str(spec_path), "--render-output", str(output)])
    audit.main()
    assert calls == [(str(spec_path), str(output))]


def test_render_spec_serializes_resolved_machine_paths(tmp_path):
    args = Namespace(dynamic_root=tmp_path / "sim", raw_root=tmp_path / "raw", timestamps_root=tmp_path / "timestamps")
    spec = audit.render_specification(args, tmp_path / "cache", [{"frame": 1}], 224)
    destination = tmp_path / "spec.json"
    audit.write_json(destination, spec)
    assert json.loads(destination.read_text()) == spec
    assert spec["dynamic_root"] == str(args.dynamic_root)


def test_known_geometry_has_zero_error_and_camera_zero_is_invariant():
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[1, 0, 3] = .2
    extrinsics = poses[:, :3].copy()
    extrinsics[:, :3, 3] *= -1
    intrinsics = np.broadcast_to(np.array([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]), (2, 3, 3))
    depth = np.ones((2, 2, 2))
    variants, _ = audit.geometry_variants(depth, extrinsics, intrinsics, poses)
    truth = audit.camera_to_world(audit.depth_camera_points(depth, intrinsics), poses)
    for xyz in variants.values():
        np.testing.assert_allclose(xyz, truth)
        result = audit.score_errors(xyz, truth, depth, poses, np.ones(depth.shape, bool))
        assert result["pixel_corresponded_xyz_error_m"]["mean"] == 0
        assert result["optical_z_absolute_error_m"]["mean"] == 0


def test_pose_correction_changes_second_camera_without_gt_depth():
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[1, 0, 3] = .2
    predicted = poses.copy()
    angle = .2
    predicted[1, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    extrinsics = np.empty((2, 3, 4))
    for i in range(2):
        extrinsics[i, :3, :3] = predicted[i, :3, :3].T
        extrinsics[i, :3, 3] = -predicted[i, :3, :3].T @ predicted[i, :3, 3]
    k = np.broadcast_to(np.eye(3), (2, 3, 3))
    variants, _ = audit.geometry_variants(np.ones((2, 3, 3)), extrinsics, k, poses)
    np.testing.assert_allclose(variants["current_sim3"][0], variants["known_extrinsics_same_depth"][0])
    assert not np.allclose(variants["current_sim3"][1], variants["known_extrinsics_same_depth"][1])


def test_metric_masks_exclude_other_surfaces_and_boundaries():
    seg = np.zeros((2, 5, 5, 2), dtype=int)
    seg[:, 2:, 2:, 0] = 1
    inside = audit.interior_mask(seg)
    assert not inside[:, 0].any() and not inside[:, -1].any()
    assert not inside[:, 2, 2].any()
    poses = np.broadcast_to(np.eye(4), (2, 4, 4))
    truth = np.ones((2, 2, 2, 3))
    predicted = truth.copy(); predicted[1] += 100
    mask = np.zeros((2, 2, 2), bool); mask[0] = True
    result = audit.score_errors(predicted, truth, np.ones((2, 2, 2)), poses, mask)
    assert result["pixel_corresponded_xyz_error_m"]["count"] == 4
    assert result["pixel_corresponded_xyz_error_m"]["mean"] == 0


def test_known_intrinsics_substitution_uses_only_same_predicted_depth():
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[1, 0, 3] = .2
    extrinsics = poses[:, :3].copy(); extrinsics[:, :3, 3] *= -1
    predicted_k = np.broadcast_to(np.diag([2., 2., 1.]), (2, 3, 3))
    known_k = np.broadcast_to(np.eye(3), (2, 3, 3))
    depth = np.full((2, 3, 3), 2.)
    variants, _ = audit.geometry_variants(depth, extrinsics, predicted_k, poses, known_k)
    expected = audit.camera_to_world(audit.depth_camera_points(depth, known_k), poses)
    np.testing.assert_allclose(variants["known_K_and_extrinsics_same_depth"], expected)
    np.testing.assert_allclose(variants["known_K_and_extrinsics_same_depth"][..., 2], 2.)
