"""Metric XYZ evaluation for the tactile-token pilot.

Coordinates are normalized by a *physical* camera baseline. Multiplication by
``baseline_m`` restores metres; no truth-derived alignment or scale fit is done.
All reported errors are in millimetres. Descriptive statistics count rows, while
bootstrap estimates first average within each ORIGINAL episode/source group and
then weight those groups equally. Sibling renders and repeated frames are not
independent trials, and these metrics do not measure policy success.

``episode_index`` must identify the original independent sampling unit, not a
new ID for each rendered variant. Optional ``groups`` identifies counterfactual
siblings with identical robot/source state, shared coordinate frame and different
object positions. ``subgroups`` maps readable names to Boolean row masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np


def _vector(values, size, name):
    values = np.asarray(values)
    if values.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    return values


def _group_rows(group_ids, size):
    """Use stable, hashable IDs without coercing integers and strings together."""
    values = _vector(group_ids, size, "group_ids")
    rows = {}
    for index, value in enumerate(values):
        value = value.item() if isinstance(value, np.generic) else value
        if value is None or (isinstance(value, (float, complex)) and not np.isfinite(value)):
            raise ValueError("group_ids must contain finite, non-null identifiers")
        try:
            rows.setdefault(value, []).append(index)
        except TypeError as exc:
            raise ValueError("group_ids must contain hashable identifiers") from exc
    return rows


def _physical_coordinates(pred, target, baseline_m):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3 or target.shape != pred.shape:
        raise ValueError("pred and target must have the same shape [N,3]")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("pred and target must be finite")
    baseline = np.asarray(baseline_m, dtype=np.float64)
    if baseline.ndim == 0:
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline_m must be finite and positive")
        baseline = np.full(len(pred), baseline, dtype=np.float64)
    else:
        baseline = _vector(baseline, len(pred), "baseline_m")
        if not np.isfinite(baseline).all() or (baseline <= 0).any():
            raise ValueError("baseline_m must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        pred_mm = pred * baseline[:, None] * 1000.0
        target_mm = target * baseline[:, None] * 1000.0
    if not np.isfinite(pred_mm).all() or not np.isfinite(target_mm).all():
        raise ValueError("physical coordinates overflowed")
    return pred_mm, target_mm


def euclidean_errors_mm(pred, target, baseline_m):
    """Return one full-3D error per row after known-baseline unit conversion."""
    pred_mm, target_mm = _physical_coordinates(pred, target, baseline_m)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.linalg.norm(pred_mm - target_mm, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("physical errors overflowed")
    return errors


def _mean_bootstrap(group_means, iterations, seed):
    if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    group_means = np.asarray(group_means, dtype=np.float64)
    if not len(group_means):
        return None, None
    estimate = float(np.sum(group_means / len(group_means)))
    if len(group_means) == 1 or np.all(group_means == group_means[0]):
        return estimate, [estimate, estimate]
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    # Bound temporary memory when the held-out set contains many episodes.
    batch_size = max(1, min(iterations, 1_000_000 // len(group_means)))
    for start in range(0, iterations, batch_size):
        end = min(start + batch_size, iterations)
        indices = rng.integers(0, len(group_means), size=(end - start, len(group_means)))
        samples[start:end] = (group_means[indices] / len(group_means)).sum(axis=1)
    return estimate, np.quantile(samples, [0.025, 0.975]).tolist()


def paired_group_bootstrap(error_a, error_b, group_ids, iterations=2000, seed=43):
    """Paired 95% percentile CI for equally weighted group means of A minus B.

    Inputs must be paired scalar errors from the SAME observations and units.
    Positive difference means A is worse than B. Original groups, not frames,
    are resampled. A one-group zero-width CI is reported with an explicit flag;
    it is not evidence from independent repeated experiments.
    """
    error_a = np.asarray(error_a, dtype=np.float64)
    error_b = np.asarray(error_b, dtype=np.float64)
    if error_a.ndim != 1 or error_b.shape != error_a.shape:
        raise ValueError("error_a and error_b must be paired vectors [N]")
    if not np.isfinite(error_a).all() or not np.isfinite(error_b).all():
        raise ValueError("paired errors must be finite")
    rows = _group_rows(group_ids, len(error_a))
    with np.errstate(over="ignore", invalid="ignore"):
        difference = error_a - error_b
    if not np.isfinite(difference).all():
        raise ValueError("paired differences overflowed")
    group_means = [float(np.sum(difference[index] / len(index))) for index in rows.values()]
    estimate, interval = _mean_bootstrap(group_means, iterations, seed)
    return {
        "mean_difference": estimate,
        "ci95": interval,
        "row_mean_difference": float(np.sum(difference / len(difference))) if len(difference) else None,
        "group_count": len(rows),
        "observation_count": len(difference),
        "iterations": int(iterations),
        "seed": int(seed),
        "insufficient_independent_groups": len(rows) < 2,
        "weighting": "equal original groups after averaging paired differences within group",
        "direction": "positive means a is worse than b",
    }


def _summary(error_vectors_mm, group_ids):
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.linalg.norm(error_vectors_mm, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("physical errors overflowed")
    paired = paired_group_bootstrap(errors, np.zeros_like(errors), group_ids)
    return {
        "observations": len(errors),
        "independent_groups": paired["group_count"],
        "mean_mm": float(errors.mean()) if len(errors) else None,
        "median_mm": float(np.median(errors)) if len(errors) else None,
        "p90_mm": float(np.quantile(errors, 0.90)) if len(errors) else None,
        "coordinate_mae_mm": np.abs(error_vectors_mm).mean(axis=0).tolist() if len(errors) else None,
        "group_mean_mm": paired["mean_difference"],
        "group_mean_ci95_mm": paired["ci95"],
        "insufficient_independent_groups": paired["insufficient_independent_groups"],
    }


def _counterfactual(pred_mm, target_mm, episode_index, groups):
    # Sibling IDs are scoped to their original episode. IDs such as "frame 0"
    # can therefore be reused safely in different episodes.
    episode_rows = _group_rows(episode_index, len(pred_mm))
    group_rows = _group_rows(groups, len(pred_mm))
    source_group = np.empty(len(pred_mm), dtype=np.int64)
    for number, rows in enumerate(group_rows.values()):
        source_group[rows] = number
    delta_errors, pair_episodes, predicted_lengths, target_lengths = [], [], [], []
    sibling_groups = 0
    excluded_same_target_pairs = 0
    for episode_number, episode_indices in enumerate(episode_rows.values()):
        episode_indices = np.asarray(episode_indices)
        for sibling in np.unique(source_group[episode_indices]):
            sibling_indices = episode_indices[source_group[episode_indices] == sibling]
            if len(sibling_indices) < 2:
                continue
            added = False
            for first, second in combinations(sibling_indices, 2):
                with np.errstate(over="ignore", invalid="ignore"):
                    actual_delta = target_mm[second] - target_mm[first]
                    predicted_delta = pred_mm[second] - pred_mm[first]
                    actual_length = float(np.linalg.norm(actual_delta))
                    predicted_length = float(np.linalg.norm(predicted_delta))
                if not np.isfinite([actual_length, predicted_length]).all():
                    raise ValueError("counterfactual physical shifts overflowed")
                if actual_length <= 1e-12:
                    excluded_same_target_pairs += 1
                    continue
                delta_errors.append(predicted_delta - actual_delta)
                predicted_lengths.append(predicted_length)
                target_lengths.append(actual_length)
                pair_episodes.append(episode_number)
                added = True
            sibling_groups += int(added)
    vectors = np.asarray(delta_errors, dtype=np.float64).reshape(-1, 3)
    return {
        **_summary(vectors, np.asarray(pair_episodes, dtype=np.int64)),
        "pairs": len(vectors),
        "sibling_source_groups": sibling_groups,
        "excluded_same_target_pairs": excluded_same_target_pairs,
        "mean_predicted_shift_mm": float(np.mean(predicted_lengths)) if predicted_lengths else None,
        "mean_target_shift_mm": float(np.mean(target_lengths)) if target_lengths else None,
        "definition": "norm((prediction_j-prediction_i)-(target_j-target_i)) in shared physical coordinates",
        "sampling_note": "sibling pairs are dependent; confidence intervals resample original episodes, not pairs",
    }


def report_metrics(pred, target, episode_index, baseline_m, groups=None, subgroups=None):
    """JSON-safe metric report for normalized XYZ predictions and targets.

    ``baseline_m`` is either a positive scalar or a length-N vector of physical
    baselines. ``groups`` labels same-state counterfactual siblings within each
    original episode; it does not replace ``episode_index`` as the bootstrap
    unit. Empty sets have null statistics, never NaN. Invalid/nonfinite inputs
    are rejected rather than silently excluded. Subgroup masks may overlap.
    """
    pred_mm, target_mm = _physical_coordinates(pred, target, baseline_m)
    episode_index = _vector(episode_index, len(pred_mm), "episode_index")
    _group_rows(episode_index, len(pred_mm))
    with np.errstate(over="ignore", invalid="ignore"):
        error_vectors_mm = pred_mm - target_mm
    if not np.isfinite(error_vectors_mm).all():
        raise ValueError("physical errors overflowed")
    report = {
        "units": "mm",
        "all": _summary(error_vectors_mm, episode_index),
        "subgroups": {},
        "counterfactual": None,
        "bootstrap": {
            "unit": "original episode/source group specified by episode_index",
            "iterations": 2000,
            "seed": 43,
            "confidence": 0.95,
            "weighting": "equal original groups; mean/median/p90 are row-level descriptive statistics",
        },
        "scope": "offline spatial representation; not independent rollout trials or policy success",
    }
    if subgroups is not None:
        if not isinstance(subgroups, Mapping):
            raise ValueError("subgroups must map names to Boolean masks")
        for name, selected in subgroups.items():
            if not isinstance(name, str):
                raise ValueError("subgroup names must be strings")
            selected = _vector(selected, len(pred_mm), f"subgroups[{name!r}]")
            if selected.dtype.kind != "b":
                raise ValueError("subgroup masks must be Boolean")
            report["subgroups"][name] = _summary(error_vectors_mm[selected], episode_index[selected])
    if groups is not None:
        report["counterfactual"] = _counterfactual(pred_mm, target_mm, episode_index, groups)
    return report
