import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile_vggt.data import CachedTactileDataset, action_chunk, episode_split, training_statistics
from tactile_vggt.geometry import (build_point_tensor, calibrated_similarity,
                                   invert_extrinsics, mujoco_camera_to_opencv, unproject_depth)


class GeometryDataTests(unittest.TestCase):
    def test_mujoco_forward_and_up_convert_to_opencv(self):
        known = mujoco_camera_to_opencv(np.eye(4))
        np.testing.assert_array_equal(known[:3, :3] @ [0, 0, 1], [0, 0, -1])
        np.testing.assert_array_equal(known[:3, :3] @ [0, 1, 0], [0, -1, 0])

    def test_camera_similarity_recovers_metric_frame_without_depth_truth(self):
        # Predicted world has half metric scale and is rotated 90 degrees about z.
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([0.3, -0.2, 0.1])
        predicted = np.repeat(np.eye(4)[None], 2, axis=0)
        predicted[1, :3, 3] = [1, 0, 0]
        known = predicted.copy()
        known[:, :3, :3] = rotation
        known[:, :3, 3] = 2 * predicted[:, :3, 3] @ rotation.T + translation
        ext = invert_extrinsics(predicted)[:, :3]
        scale, recovered_r, recovered_t, diag = calibrated_similarity(ext, known)
        self.assertAlmostEqual(scale, 2)
        np.testing.assert_allclose(recovered_r, rotation)
        np.testing.assert_allclose(recovered_t, translation)
        self.assertLess(diag["second_camera_position_error_m"], 1e-10)
        self.assertLess(diag["second_camera_rotation_error_deg"], 1e-6)

    def test_unproject_and_anchor_channels(self):
        depth = np.ones((2, 2, 2))
        ext = np.repeat(np.eye(4)[None, :3], 2, 0)
        k = np.repeat(np.eye(3)[None], 2, 0)
        points = unproject_depth(depth, ext, k)
        np.testing.assert_allclose(points[0, 1, 1], [1, 1, 1])
        tensor = build_point_tensor(points, np.ones_like(points) * .25, depth,
                                    np.zeros((44, 3)), np.arange(44), seed=3)
        self.assertEqual(tensor.shape, (556, 8))
        np.testing.assert_array_equal(tensor[:512, 6:], 0)
        np.testing.assert_array_equal(tensor[512:, 7], 1)
        np.testing.assert_array_equal(tensor[512:, 6], np.arange(44))

    def test_degenerate_baseline_is_rejected(self):
        ext = np.repeat(np.eye(4)[None, :3], 2, 0)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            calibrated_similarity(ext, np.repeat(np.eye(4)[None], 2, 0))

    def test_chunk_does_not_cross_episode_or_stride_action_labels(self):
        action = np.arange(12).reshape(6, 2)
        episodes = np.array([0, 0, 0, 1, 1, 1])
        chunk, pad = action_chunk(action, episodes, 1, 4)
        np.testing.assert_array_equal(chunk, action[[1, 2, 2, 2]])
        np.testing.assert_array_equal(pad, [False, False, True, True])

    def test_split_disjoint_and_statistics_train_only(self):
        episodes = np.repeat(np.arange(10), 2)
        split = episode_split(episodes)
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertEqual(split, episode_split(episodes))
        state = np.zeros((20, 22))
        state[np.isin(episodes, split["val"])] = 1e6
        stats = training_statistics(state, state, np.zeros((20, 44)), episodes, split["train"])
        np.testing.assert_array_equal(stats["state_mean"], np.zeros(22))
        np.testing.assert_array_equal(stats["action_mean"], np.zeros(22))

    def test_memmap_dataset_normalizes_and_masks_last_frame(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {
                "state": np.arange(88, dtype=np.float32).reshape(4, 22),
                "action": np.arange(88, dtype=np.float32).reshape(4, 22),
                "tactile": np.ones((4, 44), dtype=np.float32),
                "episode_index": np.array([0, 0, 1, 1]),
                "frame_index": np.array([0, 1, 0, 1]),
                "obs_frame_indices": np.array([1, 3]),
                "points": np.zeros((2, 556, 8), dtype=np.float32),
            }
            stats = training_statistics(arrays["state"], arrays["action"], arrays["tactile"], arrays["episode_index"], [0])
            for key, value in arrays.items():
                np.save(root / f"{key}.npy", value)
            (root / "manifest.json").write_text(json.dumps({"complete": True, "split": {"train": [0], "val": [1]}, "normalization": stats}))
            train = CachedTactileDataset(root, "train", 3)
            val = CachedTactileDataset(root, "val", 3)
            self.assertEqual(train.stats, val.stats)
            self.assertEqual(len(train), 1)
            np.testing.assert_array_equal(train[0]["action_is_pad"].numpy(), [False, True, True])
            np.testing.assert_allclose(train[0]["observation.state"].numpy(), np.ones(22))
            self.assertEqual(val[0]["episode_index"], 1)


if __name__ == "__main__":
    unittest.main()
