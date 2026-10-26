"""Camera interventions preserve the physical scene and episode/pair identity."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "camera_pilot_builder", SCRIPTS / "build_camera_pilot_data.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def stereo_poses():
    cameras = np.tile(np.eye(4), (2, 1, 1))
    cosine, sine = np.cos(np.pi / 6), np.sin(np.pi / 6)
    cameras[0, :3, :3] = [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]]
    cosine, sine = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    cameras[1, :3, :3] = [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]]
    cameras[:, :3, 3] = [[0, 0, 1], [2, 0, 1]]
    return cameras


def test_rigid_stereo_rotates_about_midpoint_and_keeps_relative_calibration():
    cameras = stereo_poses()
    original = cameras.copy()
    translation = np.array([.2, -.3, .4])
    moved, delta = builder.rigid_stereo_transform(cameras, translation, [0, 0, 1], 90)
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    np.testing.assert_allclose(moved[:, :3, 3], [[1.2, -1.3, 1.4], [1.2, .7, 1.4]], atol=1e-12)
    for before, after in zip(cameras, moved):
        np.testing.assert_allclose(after[:3, :3], quarter_turn @ before[:3, :3], atol=1e-12)
    np.testing.assert_allclose(np.linalg.inv(moved[0]) @ moved[1],
                               np.linalg.inv(cameras[0]) @ cameras[1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(moved[1, :3, 3] - moved[0, :3, 3]), 2., atol=1e-12)
    np.testing.assert_allclose(builder.apply_rigid_delta(cameras, delta), moved, atol=1e-12)
    np.testing.assert_array_equal(cameras, original)
    np.testing.assert_array_equal(translation, [.2, -.3, .4])
    audit = builder.audit_rigid_pair(cameras, moved)
    assert audit["baseline_error_m"] < 1e-12
    assert audit["relative_transform_max_abs"] < 1e-12


def test_rigid_pair_audit_detects_independent_camera_miscalibration():
    cameras = stereo_poses()
    changed = cameras.copy()
    changed[1, 0, 3] += .05
    audit = builder.audit_rigid_pair(cameras, changed)
    np.testing.assert_allclose(audit["baseline_error_m"], .05, atol=1e-12)
    assert audit["relative_transform_max_abs"] > .04


def test_first_camera_coordinates_and_relative_target_use_different_origins():
    cameras = np.tile(np.eye(4), (2, 1, 1))
    cameras[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cameras[:, :3, 3] = [[1, 2, 3], [1, 4, 3]]
    points = np.array([[3., 2, 5], [1, 4, 3]])
    saved_points, saved_cameras = points.copy(), cameras.copy()
    np.testing.assert_allclose(builder.normalize_first_camera(points, cameras), [[0, -1, 1], [1, 0, 0]])
    palm = np.eye(4)
    palm[:3, 3] = [.5, .7, .9]
    labels = palm[:3, 3] + [2, 0, 2]
    target = builder.relative_target_first_camera(labels, palm, cameras)
    np.testing.assert_allclose(target, [0, -1, 1])
    translated = cameras.copy()
    translated[:, :3, 3] += [.4, -.2, .6]
    np.testing.assert_allclose(builder.relative_target_first_camera(labels, palm, translated), target)
    assert not np.allclose(builder.normalize_first_camera(points, translated),
                           builder.normalize_first_camera(points, cameras))
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(cameras, saved_cameras)


def test_variant_plan_keeps_physical_pairs_and_split_membership():
    # Physical samples 0/1 are different object variants from one source group.
    groups, episodes, splits = np.array([13, 13, 28, 99]), np.array([4, 4, 7, 9]), np.array([0, 0, 1, 2])
    plan = builder.make_variant_plan(groups, episodes, splits, near_views=2, strong_views=1)
    assert [row["index"] for row in plan] == list(range(13))
    for physical in range(4):
        rows = [row for row in plan if row["physical_sample"] == physical]
        expected_domains = [0, 1, 1] + ([2] if splits[physical] == 2 else [])
        assert [row["camera_domain"] for row in rows] == expected_domains
        assert [row["camera_variant"] for row in rows] == list(range(len(rows)))
        for row in rows:
            assert (row["group"], row["episode"], row["split"]) == (
                groups[physical], episodes[physical], splits[physical])
    assert len({(row["physical_sample"], row["camera_variant"]) for row in plan}) == len(plan)
    assert all(row["split"] == 2 for row in plan if row["camera_domain"] == 2)
    np.testing.assert_array_equal(groups, [13, 13, 28, 99])
    np.testing.assert_array_equal(episodes, [4, 4, 7, 9])
    np.testing.assert_array_equal(splits, [0, 0, 1, 2])


def test_default_variant_plan_matches_declared_pilot_counts():
    splits = np.repeat([0, 1, 2], [560, 100, 115])
    # Distinct groups here isolate expansion counts from sibling validation.
    plan = builder.make_variant_plan(np.arange(775), splits + 10, splits)
    assert len(plan) == 3445
    assert [sum(row["split"] == split for row in plan) for split in range(3)] == [2240, 400, 805]
    assert sum(row["camera_domain"] == 2 for row in plan) == 345


@pytest.mark.parametrize("episodes,splits", [([1, 2], [0, 0]), ([1, 1], [0, 2])])
def test_variant_plan_rejects_source_group_crossing_episode_or_split(episodes, splits):
    with pytest.raises(ValueError):
        builder.make_variant_plan([5, 5], episodes, splits)


def test_camera_pose_writeback_handles_parent_and_base_frames_without_physics_step():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <geom type="plane" size="2 2 .1"/>
            <body name="camera_parent" pos=".4 -.2 .6" euler="10 -15 20">
              <camera name="left" pos="-.3 -.4 .8" euler="12 20 -5"/>
              <camera name="right" pos=".4 -.3 .9" euler="-7 10 16"/>
            </body>
            <body name="object" pos="0 0 .08">
              <freejoint/>
              <geom type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[:] = [.02, -.03, -.01, .1, .2, -.1]
    mujoco.mj_forward(model, data)
    camera_ids = [model.camera(name).id for name in ("left", "right")]
    base = np.eye(4)
    cosine, sine = np.cos(-np.pi / 6), np.sin(-np.pi / 6)
    base[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    base[:3, 3] = [.2, .3, -.1]
    cv_flip = np.diag([1., -1, -1])
    cameras = []
    for cid in camera_ids:
        world = np.eye(4)
        world[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ cv_flip
        world[:3, 3] = data.cam_xpos[cid]
        cameras.append(np.linalg.inv(base) @ world)
    cameras = np.asarray(cameras)
    desired, _ = builder.rigid_stereo_transform(cameras, [.03, -.02, .01], [1, 2, -1], 13)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    time_before, constraint_forces = data.time, data.qfrc_constraint.copy()
    body_positions = data.xpos.copy()
    builder.apply_camera_poses(model, data, camera_ids, desired, base, mujoco)
    for cid, camera in zip(camera_ids, desired):
        expected_world = base @ camera
        np.testing.assert_allclose(data.cam_xpos[cid], expected_world[:3, 3], atol=1e-12)
        np.testing.assert_allclose(data.cam_xmat[cid].reshape(3, 3) @ cv_flip,
                                   expected_world[:3, :3], atol=1e-12)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.xpos, body_positions)
    np.testing.assert_array_equal(data.qfrc_constraint, constraint_forces)
    assert data.time == time_before


def test_build_exports_each_camera_view_bound_to_its_original_physical_row(tmp_path, monkeypatch):
    """Exercise real planning/writes with a fake renderer, not geometric rendering."""
    source_root, cache, output = tmp_path / "source", tmp_path / "cache", tmp_path / "output"
    source_root.mkdir()
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    n = 4
    source = {
        "pressure": np.arange(n * 44, dtype=np.float32).reshape(n, 44) / 1000,
        "anchors": np.arange(n * 44 * 3, dtype=np.float32).reshape(n, 44, 3) / 100,
        "state": np.repeat(np.array([0, 1, 1, 3], np.float32)[:, None], 22, axis=1),
        "labels": np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 10,
        "labels_relative_palm": np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 10,
        "labels_orientation_wxyz": np.tile(np.array([1., 0, 0, 0], np.float32), (n, 1)),
        "palm": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        # Deliberately differ from sorted group order to exercise indexed writes.
        "episode": np.array([7, 4, 4, 9]), "group": np.array([28, 13, 13, 99]),
        "source_row": np.array([20, 10, 10, 30]), "variant": np.array([0, 0, 1, 0]),
        "split": np.array([1, 0, 0, 2], np.uint8),
        "rgb": np.repeat(np.arange(n, dtype=np.uint8)[:, None], 2 * 224 * 224 * 3, axis=1).reshape(n, 2, 224, 224, 3),
        "known_cameras": np.repeat(stereo_poses()[None], n, axis=0).astype(np.float32),
        "known_intrinsics": np.tile(np.eye(3, dtype=np.float32), (n, 2, 1, 1)),
        "occlusion": np.zeros((n, 2), np.float32),
    }
    records = [{"index": i, **{name: int(source[name][i]) for name in ("episode", "group", "source_row", "variant")}}
               for i in range(n)]
    (source_root / "examples.json").write_text(json.dumps(records))
    files = {}
    for name, array in source.items():
        np.save(source_root / f"{name}.npy", array)
        files[name] = builder.fingerprint(source_root / f"{name}.npy")
    source_manifest = {
        "complete": True, "coordinate_frame": "robot_base", "files": files,
        "provenance": {"examples": builder.fingerprint(source_root / "examples.json"),
                       "source_cache_manifest": builder.fingerprint(cache / "manifest.json")},
        "split_episodes": {"train": [4], "dev": [7], "test": [9]},
        "input_whitelist": ["rgb", "pressure", "anchors", "state", "known_cameras", "known_intrinsics"],
        "label_only": ["labels", "labels_relative_palm", "labels_orientation_wxyz"],
        "pressure_semantics": "frozen native readout", "sensor_order": "unchanged native 44",
    }
    (source_root / "manifest.json").write_text(json.dumps(source_manifest))
    instances = []

    class FakeSnapshotBuilder:
        def __init__(self, args, manifest):
            self.cache, self.base_to_world = cache, np.eye(4)
            self.model = SimpleNamespace(nq=29, nv=28)
            self.data = SimpleNamespace(qpos=np.zeros(29), qvel=np.zeros(28),
                                        xpos=np.zeros((2, 3)), xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)))
            self.camera_ids, self.mj, self.provenance = [0, 1], None, {}
            self.asset_provenance = {}
            self.closed = False
            instances.append(self)

        def reconstruct(self, record, arrays, physical):
            qpos = np.r_[arrays["state"][physical], arrays["labels"][physical], [1., 0, 0, 0]]
            qvel = np.full(28, arrays["source_row"][physical] / 100)
            return qpos, qvel, {"object_center_m": 0.}

        def set_state(self, qpos, qvel):
            self.data.qpos[:], self.data.qvel[:] = qpos, qvel
            self.data.xpos[0] = qpos[22:25]

        def visibility_counts(self):
            return np.array([[10, 20], [8, 16]], np.int32)

        def camera_arrays(self):
            return self.data.cameras.astype(np.float32), source["known_intrinsics"][0].copy()

        def rgb(self):
            physical = int(round(self.data.qpos[22] / .3))
            return source["rgb"][physical].copy()

        def close(self):
            self.closed = True

    def fake_apply(model, data, camera_ids, cameras_base, base_to_world, mujoco):
        data.cameras = np.array(cameras_base, copy=True)

    monkeypatch.setattr(builder, "CameraSnapshotBuilder", FakeSnapshotBuilder)
    monkeypatch.setattr(builder, "apply_camera_poses", fake_apply)
    args = SimpleNamespace(source_data=source_root, cache=cache, output=output,
        seed=20261026, near_views=1, strong_views=1, max_attempts=2, min_amodal_pixels=4,
        near_translation_min=.01, near_translation_max=.02, near_angle_min=1., near_angle_max=2.,
        strong_translation_min=.04, strong_translation_max=.05, strong_angle_min=4., strong_angle_max=5.,
        max_groups=0, audit_count=4, nominal_rgb_mae_max=1.)
    builder.build(args)
    ids = np.load(output / "physical_sample.npy")
    np.testing.assert_array_equal(ids, [0, 0, 1, 1, 2, 2, 3, 3, 3])
    for name in builder.PHYSICAL_ARRAYS:
        np.testing.assert_array_equal(np.load(output / f"{name}.npy"), source[name][ids])
    qpos, qvel = np.load(output / "qpos.npy"), np.load(output / "qvel.npy")
    for physical in range(n):
        rows = np.flatnonzero(ids == physical)
        np.testing.assert_array_equal(qpos[rows], np.broadcast_to(qpos[rows[0]], qpos[rows].shape))
        np.testing.assert_array_equal(qvel[rows], np.full((len(rows), 28), source["source_row"][physical] / 100))
    examples = json.loads((output / "examples.json").read_text())
    assert [row["index"] for row in examples] == list(range(len(ids)))
    assert [row["source_example"]["index"] for row in examples] == ids.tolist()
    metadata = json.loads((output / "manifest.json").read_text())
    assert metadata["complete"] and metadata["split_counts"] == {"train": 4, "dev": 2, "test": 3}
    assert instances[0].closed
