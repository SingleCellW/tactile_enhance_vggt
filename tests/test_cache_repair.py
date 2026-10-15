"""Synthetic integrity tests; require NumPy only, no VGGT/GPU/native simulator.

Run with ``python -m unittest discover -s tests -p test_cache_repair.py -v``.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repair_cached_tactile", ROOT / "scripts/repair_cached_tactile.py")
repair_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair_module)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CacheRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "old-cache"
        self.source.mkdir()
        self.destination = self.root / "repaired-cache"
        self.readouts = self.root / "native.npz"
        self.report_path = self.root / "native.report.json"
        self.indices = np.array([0, 2, 3, 5], dtype=np.int64)
        self.visual_count = 3
        # The skipped training frame has a large value: statistics must use all
        # training frames, while held-out episodes must not affect them.
        self.state = np.repeat(np.array([0, 100, 2, 1e6, 1e6, 1e6], dtype=np.float32)[:, None], 22, axis=1)
        self.action = self.state + np.arange(22, dtype=np.float32)
        self.episodes = np.array([7, 7, 7, 19, 19, 19], dtype=np.int64)
        self.frames = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        self.points = np.random.default_rng(3).normal(size=(4, self.visual_count + 44, 8)).astype(np.float32)
        self.points[:, :self.visual_count, 6:] = 0
        self.points[0, 0, 0] = -0.0  # Value equality alone would miss this bit change.
        for key, value in {
            "state": self.state, "action": self.action,
            "episode_index": self.episodes, "frame_index": self.frames,
            "obs_frame_indices": self.indices, "points": self.points,
            "tactile": np.full((6, 44), 999, dtype=np.float32),
        }.items():
            np.save(self.source / f"{key}.npy", value)
        self.rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        self.translation = np.array([.25, -.5, 1.])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = self.rotation, self.translation
        self.manifest = {
            "complete": True,
            "configuration": {"visual_points": self.visual_count, "base_to_world": transform.tolist()},
            "split": {"train": [7], "val": [19]},
            "normalization": {"pressure_scale": 999},
            "action_semantics": "Existing next-frame labels, unchanged by repair",
        }
        self.write_manifest()
        self.diagnostics = [{"frame": int(frame), "scale": 1.25,
                             "second_camera_position_error_m": .123,
                             "anchor_nearest_visual_median_m": -1,
                             "active_anchor_count": -1}
                            for frame in self.indices]
        self.write_diagnostics()
        self.example_rgb = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
        np.savez_compressed(self.source / "example_observation.npz", rgb=self.example_rgb,
                            state=self.state[0], tactile=np.ones(44) * 999,
                            anchors_world=np.zeros((44, 3)))
        self.base_anchors = np.arange(6 * 44 * 3, dtype=np.float32).reshape(6, 44, 3) / 1024
        self.force = np.repeat(np.array([1, 9, 3, 1e6, 1e6, 1e6], dtype=np.float32)[:, None], 44, axis=1)
        self.force[0, :22] = 0  # Check active-distance diagnostics, not only all-active rows.
        self.native = {"state22": self.state.copy(), "episode_index": self.episodes.copy(),
                       "frame_index": self.frames.copy(), "force44": self.force.copy(),
                       "anchors_world": self.base_anchors @ self.rotation.T + self.translation}
        self.report = {"complete": True, "frames": 6, "source_frames": 6,
                       "sensor_order": "32 fingertip taxels then 12 FSRs"}
        self.write_native()

    def write_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def write_diagnostics(self):
        (self.source / "geometry_diagnostics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.diagnostics))

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def write_native(self):
        np.savez_compressed(self.readouts, **self.native)
        self.report["repaired_readouts_sha256"] = sha(self.readouts)
        self.write_report()

    def repair(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            repair_module.repair(self.source, self.readouts, self.report_path, self.destination, **kwargs)

    def alignment_report(self, **changes):
        report = {"verified": True, "rgb_lag_frames": 1, "evidence": {"test": "synthetic"}}
        report.update(changes)
        path = self.root / "rgb_alignment.json"
        path.write_text(json.dumps(report))
        return path

    def test_visual_bytes_and_labels_preserved_while_force_and_geometry_replaced(self):
        source_hashes = {p.name: sha(p) for p in self.source.iterdir()}
        self.repair()
        repaired = np.load(self.destination / "points.npy")
        self.assertEqual(repaired[:, :self.visual_count].tobytes(),
                         self.points[:, :self.visual_count].tobytes())
        np.testing.assert_array_equal(repaired[:, self.visual_count:, :3], self.base_anchors[self.indices])
        np.testing.assert_array_equal(repaired[:, self.visual_count:, 6], self.force[self.indices])
        np.testing.assert_array_equal(repaired[:, self.visual_count:, 3:6], 0)
        np.testing.assert_array_equal(repaired[:, self.visual_count:, 7], 1)
        np.testing.assert_array_equal(np.load(self.destination / "tactile.npy"), self.force)
        for key in ("state", "action", "episode_index", "frame_index", "obs_frame_indices"):
            self.assertEqual((self.source / f"{key}.npy").read_bytes(),
                             (self.destination / f"{key}.npy").read_bytes())
        self.assertEqual(source_hashes, {p.name: sha(p) for p in self.source.iterdir()})
        with np.load(self.destination / "example_observation.npz") as example:
            np.testing.assert_array_equal(example["rgb"], self.example_rgb)
            np.testing.assert_array_equal(example["state"], self.state[0])
            np.testing.assert_array_equal(example["tactile"], self.force[0])
            np.testing.assert_array_equal(example["anchors_world"], self.native["anchors_world"][0])
        manifest = json.loads((self.destination / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertTrue(manifest["historical_source_tactile_superseded"])
        self.assertEqual(manifest["action_semantics"], self.manifest["action_semantics"])
        provenance = manifest["configuration"]["native_tactile_repair"]
        self.assertEqual(provenance["readouts"]["sha256"], sha(self.readouts))
        self.assertEqual(provenance["source_cache_manifest"]["sha256"], source_hashes["manifest.json"])

    def test_statistics_use_full_rate_train_episodes_and_exclude_validation(self):
        self.repair()
        manifest = json.loads((self.destination / "manifest.json").read_text())
        stats = manifest["normalization"]
        self.assertEqual(manifest["split"], {"train": [7], "val": [19]})
        self.assertEqual(stats["training_frames"], 3)
        np.testing.assert_allclose(stats["state_mean"], np.full(22, 34.))
        np.testing.assert_allclose(stats["action_mean"], 34 + np.arange(22))
        self.assertAlmostEqual(stats["pressure_scale"], float(np.log1p(np.float32(9))))
        self.assertNotEqual(stats["state_mean"][0], self.state[self.indices[:2], 0].mean())

    def test_diagnostics_recomputed_in_new_base_geometry_and_camera_fields_preserved(self):
        self.repair()
        rows = [json.loads(line) for line in (self.destination / "geometry_diagnostics.jsonl").read_text().splitlines()]
        for i, row in enumerate(rows):
            frame = self.indices[i]
            distance = np.linalg.norm(self.base_anchors[frame, :, None] - self.points[i, None, :self.visual_count, :3], axis=-1).min(axis=1)
            active = self.force[frame] > .5
            self.assertEqual(row["scale"], 1.25)
            self.assertEqual(row["second_camera_position_error_m"], .123)
            self.assertEqual(row["cache_index"], i)
            self.assertEqual(row["active_anchor_count"], int(active.sum()))
            self.assertAlmostEqual(row["anchor_nearest_visual_median_m"], float(np.median(distance)))
            self.assertAlmostEqual(row["active_anchor_nearest_visual_median_m"], float(np.median(distance[active])))

    def test_incomplete_source_rejected_before_destination_created(self):
        self.manifest["complete"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Source geometry cache is incomplete"):
            self.repair()
        self.assertFalse(self.destination.exists())

    def test_incomplete_report_rejected_before_destination_created(self):
        self.report["complete"] = False
        self.write_report()
        with self.assertRaisesRegex(ValueError, "report is incomplete"):
            self.repair()
        self.assertFalse(self.destination.exists())

    def test_report_hash_required_and_must_match(self):
        for value in (None, "0" * 64):
            with self.subTest(hash=value):
                self.report.pop("repaired_readouts_sha256", None)
                if value is not None:
                    self.report["repaired_readouts_sha256"] = value
                self.write_report()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.repair()
                self.assertFalse(self.destination.exists())

    def test_report_must_cover_source_frame_count(self):
        for field in ("frames", "source_frames"):
            for value in (None, 5):
                with self.subTest(field=field, value=value):
                    self.report["frames"] = self.report["source_frames"] = 6
                    self.report[field] = value
                    self.write_report()
                    with self.assertRaisesRegex(ValueError, "every source frame"):
                        self.repair()
                    self.assertFalse(self.destination.exists())

    def test_native_episode_frame_and_state_alignment_required(self):
        for field in ("state22", "episode_index", "frame_index"):
            with self.subTest(field=field):
                old = self.native[field].copy()
                self.native[field][0] += 1
                self.write_native()
                with self.assertRaisesRegex(AssertionError, "Misaligned " + field):
                    self.repair()
                self.assertFalse(self.destination.exists())
                self.native[field] = old

    def test_invalid_forces_geometry_and_sensor_shapes_rejected(self):
        mutations = [
            ("force44", np.zeros((6, 44)), "No native tactile contacts"),
            ("force44", np.full((6, 44), -1), "negative normal force"),
            ("force44", np.full((6, 44), np.nan), "Nonfinite"),
            ("anchors_world", np.full((6, 44, 3), np.inf), "Nonfinite"),
            ("force44", np.ones((6, 43)), "44-sensor order"),
            ("anchors_world", np.ones((5, 44, 3)), "44-sensor order"),
        ]
        for field, values, error in mutations:
            with self.subTest(field=field, shape=values.shape, error=error):
                old = self.native[field]
                self.native[field] = values
                self.write_native()
                with self.assertRaisesRegex(ValueError, error):
                    self.repair()
                self.assertFalse(self.destination.exists())
                self.native[field] = old

    def test_invalid_diagnostic_never_publishes_complete_destination(self):
        self.diagnostics[0]["frame"] = 1  # This full-rate frame was not cached.
        self.write_diagnostics()
        with self.assertRaisesRegex(ValueError, "absent from cached observations"):
            self.repair()
        self.assertFalse(self.destination.exists())
        for partial in self.root.glob("repaired-cache.partial-*"):
            self.assertFalse((partial / "manifest.json").exists())

    def test_existing_destination_is_never_overwritten(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep.txt"
        sentinel.write_text("existing result")
        with self.assertRaises(FileExistsError):
            self.repair()
        self.assertEqual(sentinel.read_text(), "existing result")

    def test_one_frame_rgb_lag_pairs_current_state_force_with_correct_next_command(self):
        from tactile_vggt.data import CachedTactileDataset, action_chunk
        audit_path = self.alignment_report()
        self.repair(rgb_lag=1, rgb_alignment_report=audit_path)
        # Source RGB rows 0 and 3 are each episode's first frame. Dropping both
        # prevents row 3 from taking the preceding episode's final sensor state.
        expected_rgb = np.array([2, 5])
        expected_native = np.array([1, 4])
        np.testing.assert_array_equal(np.load(self.destination / "rgb_source_frame_indices.npy"), expected_rgb)
        np.testing.assert_array_equal(np.load(self.destination / "obs_frame_indices.npy"), expected_native)
        repaired = np.load(self.destination / "points.npy")
        self.assertEqual(repaired[:, :self.visual_count].tobytes(),
                         self.points[[1, 3], :self.visual_count].tobytes())
        np.testing.assert_array_equal(repaired[:, self.visual_count:, :3], self.base_anchors[expected_native])
        np.testing.assert_array_equal(repaired[:, self.visual_count:, 6], self.force[expected_native])
        np.testing.assert_array_equal(self.episodes[expected_rgb], self.episodes[expected_native])
        for key in ("state", "action", "episode_index", "frame_index"):
            self.assertEqual((self.source / f"{key}.npy").read_bytes(),
                             (self.destination / f"{key}.npy").read_bytes())
        # The source action[s] already denotes historical raw command[s+1].
        # Repaired observation s must keep action[s], not the RGB row's action.
        saved_actions = np.load(self.destination / "action.npy")
        chunk, pad = action_chunk(saved_actions, self.episodes, expected_native[0], horizon=4)
        np.testing.assert_array_equal(chunk, self.action[[1, 2, 2, 2]])
        np.testing.assert_array_equal(pad, [False, False, True, True])
        self.assertFalse(np.array_equal(chunk[0], self.action[expected_rgb[0]]))
        train = CachedTactileDataset(self.destination, "train")
        val = CachedTactileDataset(self.destination, "val")
        np.testing.assert_array_equal(train.observation_indices, [0])
        np.testing.assert_array_equal(val.observation_indices, [1])
        self.assertEqual(train.stats, val.stats)
        manifest = json.loads((self.destination / "manifest.json").read_text())
        alignment = manifest["configuration"]["rgb_alignment"]
        self.assertTrue(alignment["verified"])
        self.assertEqual(alignment["rgb_lag_frames"], 1)
        self.assertEqual(alignment["dropped_observations"], 2)
        self.assertEqual(alignment["source_observations"], 4)
        self.assertEqual(manifest["observations"], 2)
        self.assertEqual(alignment["report"]["sha256"], sha(audit_path))
        self.assertEqual((self.destination / "rgb_alignment_report.json").read_bytes(), audit_path.read_bytes())
        self.assertFalse((self.destination / "example_observation.npz").exists(), "Dropped RGB0 must not be used as a repaired example")
        diagnostics = [json.loads(line) for line in (self.destination / "geometry_diagnostics.jsonl").read_text().splitlines()]
        self.assertEqual([row["frame"] for row in diagnostics], expected_native.tolist())
        self.assertEqual([row["rgb_source_frame"] for row in diagnostics], expected_rgb.tolist())
        self.assertEqual([row["cache_index"] for row in diagnostics], [0, 1])

    def test_lag_repair_requires_verified_matching_render_audit(self):
        cases = [None, {"verified": False}, {"rgb_lag_frames": 0}, {"verified": "true"}]
        for changes in cases:
            with self.subTest(changes=changes):
                path = self.alignment_report(**changes) if changes is not None else None
                with self.assertRaisesRegex(ValueError, "matching verified rendering audit"):
                    self.repair(rgb_lag=1, rgb_alignment_report=path)
                self.assertFalse(self.destination.exists())

    def test_lag_repair_rejects_cross_episode_mapping_even_with_bad_local_frame_ids(self):
        # Mimic a malformed source that incorrectly calls the first frame of
        # episode 19 its second frame. The explicit episode guard must catch it.
        self.frames[3] = 1
        np.save(self.source / "frame_index.npy", self.frames)
        self.native["frame_index"] = self.frames.copy()
        self.write_native()
        with self.assertRaisesRegex(AssertionError, "episode boundary"):
            self.repair(rgb_lag=1, rgb_alignment_report=self.alignment_report())
        self.assertFalse(self.destination.exists())

    def test_lag_repair_rejects_noncontiguous_frame_numbers(self):
        self.frames[2] = 4
        np.save(self.source / "frame_index.npy", self.frames)
        self.native["frame_index"] = self.frames.copy()
        self.write_native()
        with self.assertRaisesRegex(AssertionError, "Non-contiguous"):
            self.repair(rgb_lag=1, rgb_alignment_report=self.alignment_report())
        self.assertFalse(self.destination.exists())

    def test_lag_repair_cannot_publish_empty_observation_cache(self):
        np.save(self.source / "obs_frame_indices.npy", self.indices[[0, 2]])
        np.save(self.source / "points.npy", self.points[[0, 2]])
        with self.assertRaisesRegex(ValueError, "No observations remain"):
            self.repair(rgb_lag=1, rgb_alignment_report=self.alignment_report())
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
