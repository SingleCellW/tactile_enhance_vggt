"""Integrity tests for removing the audited historical action-label shift."""
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
SPEC = importlib.util.spec_from_file_location("repair_action_labels", ROOT / "scripts/repair_action_labels.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ActionRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output, self.raw = self.root / "source", self.root / "new", self.root / "commands"
        self.source.mkdir()
        self.commands = {
            "ep7": np.array([10, 20, 90], dtype=np.float64)[:, None] + np.arange(22),
            "ep9": np.array([10000, 20000], dtype=np.float64)[:, None] + np.arange(22),
        }
        for name, commands in self.commands.items():
            directory = self.raw / name / "tacsim_dataset_processed"
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "dataset_actions.npz", data=commands)
        raw_actions = self.raw / "ep7/action"
        raw_actions.mkdir()
        for i, command in enumerate(self.commands["ep7"]):
            np.save(raw_actions / f"{i:06d}.npy", command)
        old = np.concatenate([self.commands["ep7"][[1, 2, 2]], self.commands["ep9"][[1, 1]]]).astype(np.float32)
        arrays = {
            "action": old, "episode_index": np.array([7, 7, 7, 9, 9]), "frame_index": np.array([0, 1, 2, 0, 1]),
            "points": np.arange(2 * 47 * 8, dtype=np.float32).reshape(2, 47, 8),
            "state": np.ones((5, 22), dtype=np.float32), "tactile": np.ones((5, 44), dtype=np.float32),
            "obs_frame_indices": np.array([1, 3]), "rgb_source_frame_indices": np.array([2, 4]),
        }
        for key, value in arrays.items():
            np.save(self.source / f"{key}.npy", value)
        self.manifest = {"complete": True, "configuration": {"rgb_alignment": {"rgb_lag_frames": 1}},
                         "split": {"train": [7], "val": [9]}, "normalization": {
                             "action_mean": [999] * 22, "action_std": [999] * 22,
                             "state_mean": [1.] * 22, "state_std": [.001] * 22,
                             "pressure_scale": 17., "training_frames": 3}}
        self.write_manifest()
        self.mapping_path = self.root / "episode_map.json"
        self.mapping = {"rows": [{"episode": 7, "raw_episode": "ep7", "frames": 3},
                                 {"episode": 9, "raw_episode": "ep9", "frames": 2}]}
        self.mapping_path.write_text(json.dumps(self.mapping))
        self.audit_path = self.root / "timing.json"
        self.audit = {"complete": True, "verified": True, "action_target_offset_frames": 0,
                      "substeps_per_action": 1, "velocity_dt_s": .01, "compiled_timestep_s": .01}
        self.audit_path.write_text(json.dumps(self.audit))

    def write_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def repair(self):
        with contextlib.redirect_stdout(io.StringIO()):
            module.repair(self.source, self.raw, self.mapping_path, self.output, self.audit_path)

    def test_exact_current_commands_train_only_stats_and_all_observations_preserved(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        self.repair()
        expected = np.concatenate([self.commands["ep7"], self.commands["ep9"]]).astype(np.float32)
        np.testing.assert_array_equal(np.load(self.output / "action.npy"), expected)
        for name, data in before.items():
            self.assertEqual((self.source / name).read_bytes(), data)
            if name not in ("action.npy", "manifest.json"):
                self.assertEqual((self.output / name).read_bytes(), data)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["split"], self.manifest["split"])
        self.assertEqual(manifest["configuration"]["rgb_alignment"], {"rgb_lag_frames": 1})
        np.testing.assert_allclose(manifest["normalization"]["action_mean"], 40 + np.arange(22))
        np.testing.assert_allclose(manifest["normalization"]["action_std"], self.commands["ep7"].std(0))
        for key in ("state_mean", "state_std", "pressure_scale", "training_frames"):
            self.assertEqual(manifest["normalization"][key], self.manifest["normalization"][key])
        report = json.loads((self.output / "action_label_repair_report.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["frames"], 5)
        self.assertEqual([r["command_source_kind"] for r in report["command_sources"]],
                         ["original_action_npy", "generator_unchanged_action_list_export"])
        self.assertTrue(all(r["old_next_frame_relation_exact"] for r in report["command_sources"]))
        self.assertEqual(report["new_action_sha256"], hashlib.sha256((self.output / "action.npy").read_bytes()).hexdigest())

    def test_old_next_frame_label_relation_must_match_every_joint(self):
        actions = np.load(self.source / "action.npy")
        actions[1, 21] += .25
        np.save(self.source / "action.npy", actions)
        with self.assertRaisesRegex(AssertionError, "historical next-frame"):
            self.repair()
        self.assertFalse(self.output.exists())

    def test_original_action_and_generator_export_must_match(self):
        np.save(self.raw / "ep7/action/000000.npy", self.commands["ep7"][0] + 1)
        with self.assertRaisesRegex(AssertionError, "Raw controls and generator export differ"):
            self.repair()
        self.assertFalse(self.output.exists())

    def test_partial_raw_controls_cannot_silently_fall_back_to_export(self):
        (self.raw / "ep7/action/000001.npy").unlink()
        with self.assertRaisesRegex(ValueError, "Partial raw action"):
            self.repair()
        self.assertFalse(self.output.exists())

    def test_unverified_or_inconsistent_timing_rejected(self):
        for change in ({"verified": False}, {"velocity_dt_s": .05}, {"substeps_per_action": 5}, {"action_target_offset_frames": 1}):
            with self.subTest(change=change):
                self.audit_path.write_text(json.dumps(dict(self.audit, **change)))
                with self.assertRaisesRegex(ValueError, "verified one-step"):
                    self.repair()
                self.assertFalse(self.output.exists())

    def test_duplicate_episode_map_rejected(self):
        self.mapping["rows"].append(self.mapping["rows"][0])
        self.mapping_path.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(ValueError, "Duplicate episode"):
            self.repair()
        self.assertFalse(self.output.exists())

    def test_incomplete_cache_rejected(self):
        self.manifest["complete"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.repair()
        self.assertFalse(self.output.exists())

    def test_already_repaired_labels_and_existing_destination_rejected(self):
        self.repair()
        with self.assertRaises(FileExistsError):
            self.repair()
        with self.assertRaisesRegex(ValueError, "already repaired"):
            module.repair(self.output, self.raw, self.mapping_path, self.root / "twice", self.audit_path)


if __name__ == "__main__":
    unittest.main()
