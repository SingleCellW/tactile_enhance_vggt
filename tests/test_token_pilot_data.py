"""Dataset grouping and static interventions must not leak labels or time."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "token_pilot_builder", Path(__file__).resolve().parents[1] / "scripts/build_token_pilot_data.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_episode_splits_are_reproducible_and_disjoint():
    split = builder.split_episodes(np.repeat(np.arange(53), 4), 20261025)
    assert split == builder.split_episodes(np.arange(53), 20261025)
    assert [len(split[name]) for name in ("train", "dev", "test")] == [38, 7, 8]
    assert set(split["train"]).isdisjoint(split["dev"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["dev"]).isdisjoint(split["test"])
    assert sorted(sum(split.values(), [])) == list(range(53))


def test_static_object_intervention_preserves_hand_and_input_source():
    qpos = np.r_[np.arange(22) * .01, [.5, -.1, .2, 1., 0., 0., 0.]]
    original = qpos.copy()
    changed, shift = builder.perturb_object(qpos, 22, np.random.default_rng(12), .01, .025, 12.)
    np.testing.assert_array_equal(qpos, original)
    np.testing.assert_array_equal(changed[:22], original[:22])
    np.testing.assert_allclose(changed[22:25] - original[22:25], shift)
    assert .01 <= np.linalg.norm(shift) <= .025
    np.testing.assert_allclose(np.linalg.norm(changed[25:29]), 1., atol=1e-12)


def test_source_selection_has_no_duplicate_frames_and_retains_controls():
    episode = np.repeat(np.arange(3), 12)
    frame = np.tile(np.arange(12), 3)
    observed = np.arange(len(episode))
    pressure = np.zeros((len(episode), 44))
    pressure[frame >= 6, 0] = 1.
    selected = builder.select_sources(episode, frame, observed, pressure, 4, .0005)
    assert len(selected) == len(set(selected)) == 12
    for ep in range(3):
        subset = selected[episode[selected] == ep]
        assert len(subset) == 4
        assert (pressure[subset].max(-1) > .0005).sum() == 3
        assert (pressure[subset].max(-1) == 0).sum() == 1


def test_contact_break_flag_changes_selection_without_relabeling_sensor():
    observed = np.zeros(44, np.float32)
    original = observed.copy()
    assert not builder.contact_regime_allowed(observed, .0005)
    assert builder.contact_regime_allowed(observed, .0005, allow_contact_breaks=True)
    np.testing.assert_array_equal(observed, original)
    observed[17] = .003
    assert builder.contact_regime_allowed(observed, .0005)
