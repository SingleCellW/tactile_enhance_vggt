import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile_vggt.token_metrics import (euclidean_errors_mm, paired_group_bootstrap,
                                       report_metrics)


class TokenMetricsTests(unittest.TestCase):
    def test_physical_units_and_full_xyz_error(self):
        prediction = np.array([[3., 4., 0.], [0., 0., 2.]])
        target = np.zeros((2, 3))
        report = report_metrics(prediction, target, [0, 1], baseline_m=[0.002, 0.003])
        np.testing.assert_allclose(euclidean_errors_mm(prediction, target, [0.002, 0.003]), [10, 6])
        self.assertEqual(report["all"]["mean_mm"], 8)
        self.assertEqual(report["all"]["median_mm"], 8)
        self.assertAlmostEqual(report["all"]["p90_mm"], 9.6)
        np.testing.assert_allclose(report["all"]["coordinate_mae_mm"], [3, 4, 3])
        # A shared scale conversion is not a fitted alignment that removes error.
        scalar = report_metrics(prediction[:1], target[:1], [0], baseline_m=0.002)
        self.assertEqual(scalar["all"]["mean_mm"], 10)
        json.dumps(report, allow_nan=False)

    def test_identical_paired_errors_have_exact_zero_ci(self):
        errors = [1., 80., 3., 7., 100.]
        result = paired_group_bootstrap(errors, errors, [0, 0, 1, 1, 2])
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result["ci95"], [0., 0.])
        self.assertEqual(result["group_count"], 3)
        self.assertIn("a is worse", result["direction"])

    def test_equal_original_group_weighting_not_frame_weighting(self):
        # Repeating the first episode 100 times must not overpower episode 2.
        errors = np.r_[np.zeros(100), 10.]
        episodes = np.r_[np.zeros(100, dtype=int), 1]
        result = paired_group_bootstrap(errors, np.zeros(101), episodes)
        self.assertEqual(result["mean_difference"], 5)
        self.assertAlmostEqual(result["row_mean_difference"], 10 / 101)
        replicated = paired_group_bootstrap([0., 10.], [0., 0.], [0, 1])
        self.assertEqual(result["ci95"], replicated["ci95"])
        self.assertEqual(result["ci95"], [0., 10.])

    def test_paired_difference_sign_and_deterministic_seed(self):
        result = paired_group_bootstrap([9., 3., 4.], [5., 2., 1.], [0, 1, 2])
        self.assertAlmostEqual(result["mean_difference"], 8 / 3)
        self.assertEqual(result, paired_group_bootstrap([9., 3., 4.], [5., 2., 1.], [0, 1, 2]))

    def test_counterfactual_deltas_detect_state_only_shortcut(self):
        target = np.array([[0., 0., 0.], [1., 2., 2.], [0., 0., 0.], [1., 2., 2.]])
        # A constant localization bias cancels in the sibling differences.
        accurate_deltas = report_metrics(target + 10, target, [0, 0, 1, 1], .01,
                                        groups=["same-state", "same-state", "same-state", "same-state"])
        self.assertEqual(accurate_deltas["counterfactual"]["pairs"], 2)
        self.assertEqual(accurate_deltas["counterfactual"]["independent_groups"], 2)
        self.assertEqual(accurate_deltas["counterfactual"]["mean_mm"], 0)
        state_only = report_metrics(np.zeros_like(target), target, [0, 0, 1, 1], .01,
                                    groups=[0, 0, 0, 0])["counterfactual"]
        self.assertEqual(state_only["mean_mm"], 30)
        self.assertEqual(state_only["mean_target_shift_mm"], 30)
        self.assertEqual(state_only["mean_predicted_shift_mm"], 0)
        self.assertEqual(state_only["group_mean_ci95_mm"], [30, 30])

    def test_same_target_pairs_not_counted_as_counterfactual_changes(self):
        target = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        report = report_metrics(target, target, [0, 0, 0], .01, groups=[1, 1, 1])
        self.assertEqual(report["counterfactual"]["pairs"], 2)
        self.assertEqual(report["counterfactual"]["excluded_same_target_pairs"], 1)
        self.assertTrue(report["counterfactual"]["insufficient_independent_groups"])

    def test_subgroup_metrics_preserve_cluster_units_and_empty_json(self):
        prediction = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        subgroups = {"clean": [True, True, False], "occluded": [False, False, True],
                     "contact": [False, True, True], "no_contact": [True, False, False],
                     "empty": [False, False, False]}
        report = report_metrics(prediction, np.zeros_like(prediction), [0, 0, 1], .001,
                                groups=[0, 1, 0], subgroups=subgroups)
        self.assertEqual(report["subgroups"]["clean"]["independent_groups"], 1)
        self.assertEqual(report["subgroups"]["clean"]["mean_mm"], 1.5)
        self.assertIsNone(report["subgroups"]["empty"]["coordinate_mae_mm"])
        self.assertIsNone(report["counterfactual"]["mean_mm"])
        json.dumps(report, allow_nan=False)
        empty = report_metrics(np.empty((0, 3)), np.empty((0, 3)), [], .1, groups=[])
        self.assertIsNone(empty["all"]["mean_mm"])
        self.assertEqual(empty["all"]["observations"], 0)
        self.assertIsNone(empty["all"]["group_mean_ci95_mm"])
        json.dumps(empty, allow_nan=False)

    def test_reject_nonfinite_invalid_scale_and_unpaired_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            report_metrics([[np.nan, 0, 0]], [[0, 0, 0]], [0], .1)
        for baseline in (0., -1., np.inf, [1., 2.]):
            with self.subTest(baseline=baseline):
                with self.assertRaises(ValueError):
                    report_metrics([[0., 0, 0]], [[0, 0, 0]], [0], baseline)
        with self.assertRaisesRegex(ValueError, "paired"):
            paired_group_bootstrap([1, 2], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_group_bootstrap([1], [np.inf], [0])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            paired_group_bootstrap([], [], [], iterations=0)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            report_metrics([[0., 0, 0]], [[0, 0, 0]], [0], .1, subgroups={"bad": [1]})
        # An exact prediction may still have an overflowing *sibling shift*;
        # reject it rather than putting Infinity into otherwise zero-error JSON.
        huge = np.array([[0., 0., 0.], [1e200, 0., 0.]])
        with self.assertRaisesRegex(ValueError, "shifts overflowed"):
            report_metrics(huge, huge, [0, 0], 1., groups=[0, 0])


if __name__ == "__main__":
    unittest.main()
