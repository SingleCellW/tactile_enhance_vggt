#!/usr/bin/env python3
"""Read-only numeric audit of saved token-pilot labels and predictions.

Primary sensitivity estimates average nominal->variant pairs within each source
group, then source groups within original episode, then episodes equally. The
bootstrap resamples the eight original test episodes. No model is loaded and
neither predictions nor datasets are changed. A rotation fit used below checks
translation-record consistency ONLY; it is never applied to model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.token_metrics import paired_group_bootstrap


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def interval(values, episodes):
    result = paired_group_bootstrap(values, np.zeros(len(values)), episodes)
    return {"mean": result["mean_difference"], "ci95": result["ci95"]}


def nominal_pairs(episode, groups, variant):
    result = []
    for ep in np.unique(episode):
        for group in np.unique(groups[episode == ep]):
            ids = np.flatnonzero((episode == ep) & (groups == group))
            nominal = ids[variant[ids] == 0]
            if len(nominal) != 1 or len(set(variant[ids].tolist())) != len(ids):
                raise ValueError(f"Invalid nominal/variant identifiers in episode {ep}, group {group}")
            for other in ids[variant[ids] != 0]:
                result.append((int(ep), int(group), int(nominal[0]), int(other)))
    return result


def numeric_audit(data, manifest, records):
    episode, groups, variants = (data[name] for name in ("episode", "group", "variant"))
    checks, diagnostics = {}, {}
    pairs = nominal_pairs(episode, groups, variants)
    for name in ("state", "palm", "known_cameras"):
        difference = [np.max(np.abs(data[name][j]-data[name][i])) for _, _, i, j in pairs]
        diagnostics[f"sibling_{name}_max_abs_difference"] = float(max(difference, default=0))
        checks[f"sibling_{name}_unchanged"] = diagnostics[f"sibling_{name}_max_abs_difference"] == 0
    checks["source_group_does_not_cross_episode"] = all(
        len(np.unique(episode[groups == group])) == 1 for group in np.unique(groups))
    labels, palm, cameras = (data[name].astype(np.float64) for name in ("labels", "palm", "known_cameras"))
    relative = np.einsum("bj,bjk->bk", labels-palm[:, :3, 3], palm[:, :3, :3])
    diagnostics["stored_palm_relative_labels_max_error_mm"] = float(np.linalg.norm(relative-data["labels_relative_palm"], axis=1).max()*1000)
    checks["stored_palm_relative_labels_consistent"] = diagnostics["stored_palm_relative_labels_max_error_mm"] < .001
    baseline = np.linalg.norm(cameras[:, 1, :3, 3]-cameras[:, 0, :3, 3], axis=1)
    checks["finite_positive_camera_baseline"] = bool(np.isfinite(baseline).all() and (baseline > 1e-5).all())
    diagnostics["baseline_min_max_m"] = [float(baseline.min()), float(baseline.max())]
    lookup = {int(item["index"]): item for item in records}
    checks["one_source_record_per_row"] = len(lookup) == len(records) == len(labels) and set(lookup) == set(range(len(labels)))
    checks["record_identifiers_match_arrays"] = all(
        item["episode"] == int(episode[i]) and item["group"] == int(groups[i]) and item["variant"] == int(variants[i])
        for i, item in lookup.items())
    shifts, label_shifts = [], []
    for _, _, i, j in pairs:
        shifts.append(np.asarray(lookup[j]["translation_delta_world_m"])-np.asarray(lookup[i]["translation_delta_world_m"]))
        label_shifts.append(labels[j]-labels[i])
    shifts, label_shifts = np.asarray(shifts), np.asarray(label_shifts)
    norm_error = np.abs(np.linalg.norm(shifts, axis=1)-np.linalg.norm(label_shifts, axis=1))*1000
    diagnostics["record_shift_vs_label_shift_max_norm_error_mm"] = float(norm_error.max())
    checks["record_translation_lengths_match_label_displacements"] = bool(norm_error.max() < .001)
    # World->base is one fixed rotation. Fit it as an audit diagnostic because
    # this local handoff contains base-frame arrays but not the source world's
    # base_to_world matrix. This fit does not affect prediction scoring.
    u, _, vt = np.linalg.svd(shifts.T @ label_shifts)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    fit_error = np.linalg.norm(shifts @ rotation-label_shifts, axis=1)*1000
    diagnostics["record_to_label_fixed_rotation_diagnostic"] = {
        "row_vector_world_to_base_rotation": rotation.tolist(),
        "max_residual_mm": float(fit_error.max()), "mean_residual_mm": float(fit_error.mean()),
        "scope": "audit-only rotation inferred from translation records and labels; never fitted to model predictions"}
    checks["all_record_translations_share_one_rigid_coordinate_rotation"] = bool(fit_error.max() < .001)
    low, high = manifest["generation"]["jitter_min"], manifest["generation"]["jitter_max"]
    lengths = np.linalg.norm(label_shifts, axis=1)
    diagnostics["nominal_to_variant_shift_min_median_max_mm"] = (np.quantile(lengths, [0, .5, 1])*1000).tolist()
    checks["label_shifts_within_declared_translation_range"] = bool((lengths >= low-1e-6).all() and (lengths <= high+1e-6).all())
    return {"passed": all(checks.values()), "checks": checks, "diagnostics": diagnostics,
            "nominal_to_variant_pairs": len(pairs)}, baseline


def sensitivity(pred_mm, target_mm, episode, groups, variants):
    """Return episode summaries after equally weighting sources within episode."""
    pairs = nominal_pairs(episode, groups, variants)
    sources = {}
    for ep, group, first, second in pairs:
        actual = target_mm[second]-target_mm[first]
        predicted = pred_mm[second]-pred_mm[first]
        actual_length, predicted_length = np.linalg.norm(actual), np.linalg.norm(predicted)
        if actual_length <= 1e-8:
            raise ValueError("A counterfactual translation is zero")
        dot = float(predicted @ actual)
        cosine = dot/(predicted_length*actual_length) if predicted_length > 1e-8 else None
        error = float(np.linalg.norm(predicted-actual))
        row = {"delta_error_mm": error, "true_shift_mm": float(actual_length),
               "predicted_shift_mm": float(predicted_length),
               "error_minus_zero_shift_mm": error-float(actual_length),
               "signed_gain_along_true_shift": dot/float(actual_length**2),
               "perpendicular_predicted_response_mm": float(np.linalg.norm(predicted-(dot/actual_length**2)*actual)),
               "magnitude_ratio": float(predicted_length/actual_length),
               "direction_cosine": cosine,
               "opposite_direction_fraction": float(cosine < 0) if cosine is not None else None,
               "nominal_absolute_error_mm": float(np.linalg.norm(pred_mm[first]-target_mm[first])),
               "variant_absolute_error_mm": float(np.linalg.norm(pred_mm[second]-target_mm[second]))}
        sources.setdefault((ep, group), []).append(row)
    source_means = {key: {name: mean(row[name] for row in rows) for name in rows[0]}
                    for key, rows in sources.items()}
    episodes = sorted(set(ep for ep, _ in sources))
    per_episode = {str(ep): {name: mean(row[name] for (source_ep, _), row in source_means.items() if source_ep == ep)
                            for name in next(iter(source_means.values()))} for ep in episodes}
    metrics = {name: mean(per_episode[str(ep)][name] for ep in episodes)
               for name in next(iter(source_means.values()))}
    metrics["delta_error_mm_ci95"] = interval([per_episode[str(ep)]["delta_error_mm"] for ep in episodes], episodes)["ci95"]
    metrics["error_minus_zero_shift_mm_ci95"] = interval([per_episode[str(ep)]["error_minus_zero_shift_mm"] for ep in episodes], episodes)["ci95"]
    # Absolute position error on ALL held-out rows, still with source then
    # episode equal weighting. This is a different estimand from old row mean.
    absolute = np.linalg.norm(pred_mm-target_mm, axis=1)
    absolute_ep = []
    for ep in np.unique(episode):
        absolute_value = mean(absolute[(episode == ep) & (groups == group)].mean()
                              for group in np.unique(groups[episode == ep]))
        absolute_ep.append(absolute_value)
        per_episode[str(ep)]["all_test_absolute_error_mm"] = absolute_value
    metrics["all_test_absolute_error_mm"] = mean(absolute_ep)
    metrics["all_test_absolute_error_mm_ci95"] = interval(absolute_ep, np.unique(episode))["ci95"]
    return {"weighting": "nominal-to-variant pairs -> equal source groups -> equal original episodes",
            "metric_scopes": {
                "displacement_and_response_metrics": "67 nominal->variant pairs from 28 sources with variants, then equal original episodes (counts below are authoritative)",
                "nominal_absolute_error_mm_and_variant_absolute_error_mm": "only sources having counterfactual variants; same subset as displacement metrics",
                "all_test_absolute_error_mm": "ALL 115 held-out observations from ALL 48 test sources, including sources without variants; source then episode equal weighting (counts below are authoritative)"},
            "counts": {"observations": len(episode), "nominal_to_variant_pairs": len(pairs),
                       "sources_with_variants": len(sources), "test_source_groups": len(np.unique(groups)),
                       "original_episodes": len(episodes),
                       "pairs_with_nonzero_predicted_shift": sum(row["direction_cosine"] is not None for rows in sources.values() for row in rows)},
            "metrics": metrics, "per_episode": per_episode}


def paired_difference(a, b, metric="delta_error_mm"):
    episodes = sorted(set(a["per_episode"]) & set(b["per_episode"]), key=int)
    result = paired_group_bootstrap([a["per_episode"][ep][metric] for ep in episodes],
                                   [b["per_episode"][ep][metric] for ep in episodes], episodes)
    return {"mean_difference_mm": result["mean_difference"], "ci95_mm": result["ci95"],
            "original_episodes": len(episodes), "direction": "negative means internal has lower displacement error"}


def run(args):
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = ("episode", "group", "variant", "state", "palm", "known_cameras", "labels", "labels_relative_palm")
    data = {name: np.load(args.data / f"{name}.npy", allow_pickle=False) for name in names}
    hashes = {name: sha(args.data / f"{name}.npy") for name in names}
    checksum_ok = all(hashes[name] == manifest["files"][name]["sha256"] for name in names)
    records_path = args.data / "examples.json"
    records = json.loads(records_path.read_text())
    labels_audit, baseline = numeric_audit(data, manifest, records)
    labels_audit["checks"]["array_checksums_match_generation_manifest"] = checksum_ok
    labels_audit["checks"]["record_checksum_matches_generation_manifest"] = sha(records_path) == manifest["provenance"]["examples"]["sha256"]
    labels_audit["passed"] = all(labels_audit["checks"].values())
    target = np.einsum("bj,bjk->bk", data["labels"]-data["palm"][:, :3, 3], data["known_cameras"][:, 0, :3, :3]) / baseline[:, None]
    runs, prediction_audits = {}, {}
    common_indices = None
    for directory in args.runs:
        summary = json.loads((directory / "summary.json").read_text())
        if summary["data_manifest_sha256"] != sha(manifest_path):
            raise ValueError("Run summary refers to a different dataset manifest")
        modes = {}
        for mode in summary["results"]:
            path = directory / mode / "test_predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                pred, saved_target, indices = saved["prediction"], saved["target"], saved["indices"]
                episode, groups, scale = saved["episode"], saved["group"], saved["baseline_m"]
            if common_indices is None:
                common_indices = indices
            audit = {"sha256": sha(path),
                     "same_test_rows_across_all_modes_and_seeds": bool(np.array_equal(indices, common_indices)),
                     "unique_test_rows": len(np.unique(indices)) == len(indices),
                     "episode_ids_match_source": bool(np.array_equal(episode, data["episode"][indices])),
                     "source_groups_match_source": bool(np.array_equal(groups, data["group"][indices])),
                     "only_declared_test_episodes": set(episode.tolist()) == set(manifest["split_episodes"]["test"]),
                     "baseline_max_error_m": float(np.max(np.abs(scale-baseline[indices]))),
                     "target_max_error_mm": float(np.max(np.linalg.norm((saved_target-target[indices])*scale[:, None]*1000, axis=1))),
                     "finite_prediction_and_target": bool(np.isfinite(pred).all() and np.isfinite(saved_target).all())}
            audit["passed"] = (all(audit[key] for key in ("same_test_rows_across_all_modes_and_seeds", "unique_test_rows", "episode_ids_match_source", "source_groups_match_source", "only_declared_test_episodes", "finite_prediction_and_target"))
                                and audit["baseline_max_error_m"] < 1e-6 and audit["target_max_error_mm"] < .001)
            prediction_audits[f"{directory.name}/{mode}"] = audit
            if not audit["passed"]:
                raise ValueError(f"Saved prediction alignment/label audit failed: {directory.name}/{mode}: {audit}")
            modes[mode] = sensitivity(pred.astype(np.float64)*scale[:, None]*1000,
                                      saved_target.astype(np.float64)*scale[:, None]*1000,
                                      episode, groups, data["variant"][indices])
        runs[directory.name] = {"training_seed": summary["args"]["seed"], "results": modes,
                                "paired_internal_minus_other": {mode: paired_difference(modes["internal"], value)
                                    for mode, value in modes.items() if mode != "internal"},
                                "paired_absolute_internal_minus_other": {mode: {
                                    **paired_difference(modes["internal"], value, "all_test_absolute_error_mm"),
                                    "direction": "negative means internal has lower absolute position error"}
                                    for mode, value in modes.items() if mode != "internal"}}
    common_modes = set.intersection(*(set(run["results"]) for run in runs.values()))
    averaged = {}
    for mode in sorted(common_modes):
        examples = [run["results"][mode] for run in runs.values()]
        episodes = list(examples[0]["per_episode"])
        metrics = list(examples[0]["per_episode"][episodes[0]])
        averaged[mode] = {"per_episode": {ep: {metric: mean(row["per_episode"][ep][metric] for row in examples)
                                                for metric in metrics} for ep in episodes}}
        averaged[mode]["metrics"] = {metric: mean(averaged[mode]["per_episode"][ep][metric] for ep in episodes) for metric in metrics}
        averaged[mode]["metrics"]["delta_error_mm_ci95"] = interval([averaged[mode]["per_episode"][ep]["delta_error_mm"] for ep in episodes], episodes)["ci95"]
        averaged[mode]["metrics"]["all_test_absolute_error_mm_ci95"] = interval([averaged[mode]["per_episode"][ep]["all_test_absolute_error_mm"] for ep in episodes], episodes)["ci95"]
    result = {"complete": True, "label_audit": labels_audit, "prediction_audits": prediction_audits,
              "all_prediction_audits_passed": all(value["passed"] for value in prediction_audits.values()),
              "runs": runs, "mean_across_same_test_episodes_and_three_training_seeds": {
                  "results": averaged,
                  "paired_internal_minus_other": {mode: paired_difference(averaged["internal"], value)
                      for mode, value in averaged.items() if mode != "internal"},
                  "paired_absolute_internal_minus_other": {mode: {
                      **paired_difference(averaged["internal"], value, "all_test_absolute_error_mm"),
                      "direction": "negative means internal has lower absolute position error"}
                      for mode, value in averaged.items() if mode != "internal"}},
              "provenance": {"data_manifest_sha256": sha(manifest_path), "records_sha256": sha(records_path),
                             "script_sha256": sha(__file__), "array_sha256": hashes},
              "interpretation": {"primary_weighting": "each original episode equally; within episode each source equally; within source each nominal->variant displacement equally",
                  "absolute_metric_scope": "all_test_absolute_error_mm uses all 48 test source groups/115 observations; nominal/variant absolute metrics use only 28 source groups/67 nominal->variant pairs",
                  "analysis_status": "post-hoc sensitivity audit; alternative absolute weighting does not replace the predeclared main comparison",
                  "bootstrap": "2000 resamples of original test episodes, seed43; three training seeds share the same test dataset",
                  "ideal_displacement_response": {"delta_error_mm": 0, "signed_gain_along_true_shift": 1, "magnitude_ratio": 1, "direction_cosine": 1},
                  "zero_displacement_response": "delta_error equals true shift; signed gain and magnitude ratio zero; direction undefined",
                  "limitations": ["No model execution/training; saved predictions only", "Single object/fixed cameras; not reconstruction or policy success",
                                   "Random rotations accompany translations; pressure may respond to orientation as well as center shift",
                                   "Counterfactual candidates retain contact, so evaluation is conditional on generation acceptance",
                                   "Only eight original test episodes; pairs and repeated training seeds do not create extra independent episodes"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "label_audit_passed": labels_audit["passed"],
                      "all_prediction_audits_passed": result["all_prediction_audits_passed"],
                      "averaged_metrics": {mode: value["metrics"] for mode, value in averaged.items()},
                      "paired": result["mean_across_same_test_episodes_and_three_training_seeds"]["paired_internal_minus_other"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/token_pilot_data_20261025")
    parser.add_argument("--runs", type=Path, nargs="+", default=[ROOT / f"outputs/token_pilot_20261025/s{seed}_refined_v2" for seed in (43, 44, 45)])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/token_pilot_20261025/counterfactual_audit.json")
    run(parser.parse_args())
