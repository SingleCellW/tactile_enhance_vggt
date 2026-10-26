"""CPU checks that command equivalence and remaining control errors stay distinct."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "simple_control_target_audit", Path(__file__).resolve().parents[1] / "scripts/audit_simple_control_targets.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_equivalent_saturated_targets_remove_only_their_control_difference():
    expert, policy = np.zeros((2, 22)), np.zeros((2, 22))
    expert[:, 6], policy[:, 6] = 4., 3.  # Different targets, identical clipped control1.
    policy[:, 0] = .5  # Genuine unsaturated arm-control difference must remain.
    before = policy.copy()
    bounds, limited = np.tile([-1., 1.], (22, 1)), np.ones(22, dtype=bool)
    stats = audit.comparison(expert, policy, bounds, limited, np.ones(22))
    hand, arm = stats["aggregate"]["hand16"], stats["aggregate"]["arm6"]
    assert hand["normalized_raw_command_mae"] == pytest.approx(1 / 16)
    assert hand["normalized_clamped_ctrl_mae"] == 0
    assert hand["raw_mae_contribution_from_both_same_saturation"] == 1
    assert arm["normalized_clamped_ctrl_mae"] == arm["normalized_raw_command_mae"] == pytest.approx(.5 / 6)
    assert stats["per_dimension"][6]["both_same_saturation_fraction"] == 1
    np.testing.assert_array_equal(policy, before)


def test_unlimited_control_dimensions_are_not_analytically_clipped():
    actions = np.zeros((2, 22))
    actions[:, :2] = 4.
    limited = np.ones(22, dtype=bool)
    limited[0] = False
    bounds = np.tile([-1., 1.], (22, 1))
    clipped = audit.clamp_targets(actions, bounds, limited)
    np.testing.assert_array_equal(clipped[:, 0], [4., 4.])
    np.testing.assert_array_equal(clipped[:, 1], [1., 1.])
    outside = audit.outside_range(actions, bounds, limited)
    assert not outside[:, 0].any() and outside[:, 1].all()
