#!/usr/bin/env python3
"""CPU-only Stage1.b audit/summary; cameras, modes, seeds remain distinct.

The three trained seeds are not extra independent test episodes. Paired CIs
resample original episodes. A seed-averaged CI first averages each episode over
the SAME seeds and is explicitly conditional on those trained models.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.token_metrics import paired_group_bootstrap

SEEDS = (43, 44, 45)
REGIMES = ("fixed", "augmented")
MODES = ("visual", "tactile", "late", "internal")
DOMAINS = {0: "nominal", 1: "near", 2: "strong"}
EXPECTED_COUNTS = {"nominal": 115, "near": 345, "strong": 345}
ERROR_METRICS = ("center_mm", "displacement_error_mm", "cross_view_mm")
SCALAR_METRICS = ERROR_METRICS + ("zero_shift_error_mm", "predicted_shift_mm", "magnitude_ratio", "direction_cosine", "perpendicular_response_mm")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path, sources):
    path = Path(path)
    if not path.is_file():
        return None
    sources[str(path.resolve())] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return json.loads(path.read_text())


def episode_stats(values, episodes):
    values, episodes = np.asarray(values, float), np.asarray(episodes)
    result = paired_group_bootstrap(values, np.zeros(len(values)), episodes)
    return {"episode_mean_mm": result["mean_difference"], "ci95_mm": result["ci95"],
            "observations": len(values), "episodes": result["group_count"],
            "row_mean_mm": float(values.mean()) if len(values) else None,
            "median_mm": float(np.median(values)) if len(values) else None,
            "p90_mm": float(np.quantile(values, .9)) if len(values) else None,
            "per_episode": {str(int(ep)): float(values[episodes == ep].mean()) for ep in np.unique(episodes)}}


def source_displacement(arrays, mask):
    """Nominal-object -> variant; average views/siblings, sources, episodes."""
    ids = np.flatnonzero(mask)
    rows = []
    for group in np.unique(arrays["group"][ids]):
        selected = ids[arrays["group"][ids] == group]
        pairs = []
        for camera in np.unique(arrays["camera_variant"][selected]):
            current = selected[arrays["camera_variant"][selected] == camera]
            nominal = current[arrays["variant"][current] == 0]
            if len(nominal) != 1:
                raise ValueError("Counterfactual source/view needs one object nominal")
            first = nominal[0]
            for second in current[arrays["variant"][current] != 0]:
                target = arrays["target_base"][second] - arrays["target_base"][first]
                prediction = arrays["prediction_base"][second] - arrays["prediction_base"][first]
                target_size, pred_size = np.linalg.norm(target), np.linalg.norm(prediction)
                if target_size < 1e-8:
                    raise ValueError("Zero actual object intervention")
                along = float(prediction @ target / target_size)
                pairs.append({"displacement_error_mm": float(np.linalg.norm(prediction - target) * 1000),
                    "zero_shift_error_mm": float(target_size * 1000), "predicted_shift_mm": float(pred_size * 1000),
                    "magnitude_ratio": float(pred_size / target_size),
                    "direction_cosine": float(along / pred_size) if pred_size > 1e-8 else None,
                    "perpendicular_response_mm": float(np.linalg.norm(prediction - along * target / target_size) * 1000)})
        if pairs:
            row = {"group": int(group), "episode": int(arrays["episode"][selected[0]]), "pairs": len(pairs)}
            for key in pairs[0]:
                values = [value[key] for value in pairs if value[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            rows.append(row)
    metrics = {}
    for metric in (set(rows[0]) - {"group", "episode", "pairs"}) if rows else []:
        selected = [row for row in rows if row[metric] is not None]
        stats = episode_stats([row[metric] for row in selected], [row["episode"] for row in selected])
        metrics[metric] = {"mean": stats["episode_mean_mm"], "ci95": stats["ci95_mm"],
                           "per_episode": stats["per_episode"], "finite_sources": len(selected), "episodes": stats["episodes"]}
    return {"source_equal_metrics": metrics, "sources": rows,
            "pairs": sum(row["pairs"] for row in rows), "source_groups": len(rows),
            "weighting": "same camera; nominal-object to variant; source then original episode equally weighted"}


def audit_predictions(path, expected_counts=None):
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = ("indices", "prediction_camera", "target_camera", "prediction_base", "target_base", "episode", "group",
                "physical_sample", "variant", "camera_variant", "camera_domain", "camera_to_base", "errors_mm", "baseline_m")
    if any(name not in arrays for name in required):
        raise ValueError("Missing prediction/provenance array")
    n = len(arrays["indices"])
    if len(np.unique(arrays["indices"])) != n:
        raise ValueError("Duplicate prediction row indices")
    for key in required:
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Nonfinite prediction artifact: " + key)
    for key in ("prediction_camera", "target_camera", "prediction_base", "target_base"):
        if arrays[key].shape != (n, 3):
            raise ValueError("Invalid vector shape: " + key)
    if arrays["camera_to_base"].shape != (n, 4, 4):
        raise ValueError("Invalid saved camera shape")
    for key in ("episode", "group", "physical_sample", "variant", "camera_variant", "camera_domain"):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind not in "iu":
            raise ValueError("Invalid grouping identifier: " + key)
    if set(arrays["camera_domain"].tolist()) != set(DOMAINS):
        raise ValueError("Prediction artifact needs all three test domains")
    baseline = float(arrays["baseline_m"])
    if baseline <= 0:
        raise ValueError("Invalid physical baseline")
    rotation = arrays["camera_to_base"][:, :3, :3]
    if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("Saved camera rotation is not rigid")
    camera_error = arrays["prediction_camera"] - arrays["target_camera"]
    base_error = arrays["prediction_base"] - arrays["target_base"]
    restored = np.einsum("ni,nji->nj", camera_error * baseline, rotation)
    if not np.allclose(base_error, restored, rtol=0, atol=2e-6):
        raise ValueError("Camera/base error vectors disagree; relative vectors must rotate without translation")
    inferred_palm = arrays["target_base"] - np.einsum("ni,nji->nj", arrays["target_camera"] * baseline, rotation)
    for group in np.unique(arrays["group"]):
        ids = np.flatnonzero(arrays["group"] == group)
        if len(np.unique(arrays["episode"][ids])) != 1 or not np.allclose(inferred_palm[ids], inferred_palm[ids[0]], rtol=0, atol=2e-6):
            raise ValueError("Object siblings/camera views changed palm or episode")
        for view in np.unique(arrays["camera_variant"][ids]):
            selected = ids[arrays["camera_variant"][ids] == view]
            if not np.allclose(arrays["camera_to_base"][selected], arrays["camera_to_base"][selected[0]], rtol=0, atol=1e-6):
                raise ValueError("Object displacement confounds different cameras")
    physical_nominal = {}
    for physical in np.unique(arrays["physical_sample"]):
        ids = np.flatnonzero(arrays["physical_sample"] == physical)
        nominal = ids[arrays["camera_variant"][ids] == 0]
        if len(nominal) != 1 or len(np.unique(arrays["camera_variant"][ids])) != len(ids):
            raise ValueError("Physical sample lacks unique camera identities")
        if not np.allclose(arrays["target_base"][ids], arrays["target_base"][ids[0]], rtol=0, atol=1e-7):
            raise ValueError("Camera render changed the object target")
        physical_nominal[int(physical)] = int(nominal[0])
    errors = np.linalg.norm(base_error, axis=1) * 1000
    if not np.allclose(errors, arrays["errors_mm"], rtol=1e-6, atol=1e-5):
        raise ValueError("Saved per-row physical errors disagree")
    domains = {}
    for domain, name in DOMAINS.items():
        mask = arrays["camera_domain"] == domain
        if int(mask.sum()) != expected_counts[name]:
            raise ValueError("Unexpected test-domain row count: " + name)
        cross = []
        if domain:
            for physical in np.unique(arrays["physical_sample"][mask]):
                ids = np.flatnonzero(mask & (arrays["physical_sample"] == physical))
                nominal = physical_nominal[int(physical)]
                cross.append({"physical_sample": int(physical), "episode": int(arrays["episode"][nominal]),
                    "distance_to_nominal_mm": float(np.linalg.norm(arrays["prediction_base"][ids] - arrays["prediction_base"][nominal], axis=1).mean() * 1000)})
        domains[name] = {"center": episode_stats(errors[mask], arrays["episode"][mask]),
                         "displacement": source_displacement(arrays, mask),
                         "cross_view": {**episode_stats([row["distance_to_nominal_mm"] for row in cross], [row["episode"] for row in cross]), "rows": cross} if cross else None}
    identity = {key: arrays[key] for key in required if key not in ("prediction_camera", "prediction_base", "errors_mm")}
    return domains, identity, baseline


def compare_recorded_domains(recorded, rebuilt):
    for domain, value in rebuilt.items():
        raw = recorded.get(domain, {})
        for key in ("episode_mean_mm", "row_mean_mm", "median_mm", "p90_mm", "observations", "episodes"):
            if not np.isclose(raw.get("center", {}).get(key, np.nan), value["center"][key], atol=1e-5, rtol=1e-6):
                raise ValueError("Raw center metric differs from predictions: " + domain + "/" + key)
        for key, metric in value["displacement"]["source_equal_metrics"].items():
            old = raw.get("displacement", {}).get("source_equal_metrics", {}).get(key, {}).get("mean")
            if metric["mean"] is None:
                if old is not None:
                    raise ValueError("Undefined direction metric must remain null")
            elif old is None or not np.isclose(old, metric["mean"], atol=1e-5, rtol=1e-6):
                raise ValueError("Raw source-equal CF metric differs: " + domain + "/" + key)
        for key in ("pairs", "source_groups"):
            if raw.get("displacement", {}).get(key) != value["displacement"][key]:
                raise ValueError("Raw counterfactual identity count differs")
        if value["cross_view"] is not None and not np.isclose(raw.get("cross_view", {}).get("episode_mean_mm", np.nan), value["cross_view"]["episode_mean_mm"], atol=1e-5, rtol=1e-6):
            raise ValueError("Raw cross-view metric differs")


def metric_episodes(run, domain, metric):
    value = run.get("domains", {}).get(domain)
    if value is None:
        return {}
    if metric == "center_mm":
        return value["center"]["per_episode"]
    if metric == "cross_view_mm":
        return value["cross_view"]["per_episode"] if value["cross_view"] else {}
    return value["displacement"]["source_equal_metrics"].get(metric, {}).get("per_episode", {})


def seed_descriptive(by_seed):
    """Require three seeds; never turn 3x8 observations into 24 episodes."""
    available = sorted(seed for seed, values in by_seed.items() if values)
    result = {"complete": available == list(SEEDS), "available_seeds": available,
              "mean": None, "sample_sd": None, "seed_values": {}, "conditional_episode_ci95": None,
              "episodes": None, "scope": "descriptive seed variation; CI conditional on these models, original episodes only"}
    if not result["complete"]:
        return result
    sets = [set(by_seed[seed]) for seed in SEEDS]
    if any(value != sets[0] for value in sets):
        result.update(complete=False, problem="seed_episode_identity_mismatch")
        return result
    ids = sorted(sets[0], key=int)
    matrix = np.array([[by_seed[seed][ep] for ep in ids] for seed in SEEDS], float)
    values = matrix.mean(axis=1)
    conditional = paired_group_bootstrap(matrix.mean(axis=0), np.zeros(len(ids)), ids)
    result.update(mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                  seed_values={str(seed): float(value) for seed, value in zip(SEEDS, values)},
                  conditional_episode_ci95=conditional["ci95"], episodes=len(ids))
    return result


def paired_contrast(runs, coefficients, domain, metric):
    """Linear paired contrasts, including augmentation difference-in-differences."""
    per_seed, per_episode = {}, {}
    for seed in SEEDS:
        cells = []
        for regime, mode, coefficient in coefficients:
            run = runs.get((seed, regime, mode))
            if run is None or not run["complete"]:
                cells = []
                break
            values = metric_episodes(run, domain, metric)
            if not values:
                cells = []
                break
            cells.append((values, coefficient))
        if not cells:
            per_seed[str(seed)] = {"complete": False, "difference": None, "ci95": None}
            continue
        episodes = sorted(cells[0][0], key=int)
        if any(set(values) != set(episodes) for values, _ in cells):
            raise ValueError("Paired contrast crosses episode identities")
        difference = np.array([sum(values[ep] * coefficient for values, coefficient in cells) for ep in episodes])
        bootstrap = paired_group_bootstrap(difference, np.zeros(len(difference)), episodes)
        per_seed[str(seed)] = {"complete": True, "difference": bootstrap["mean_difference"], "ci95": bootstrap["ci95"], "episodes": len(episodes)}
        per_episode[seed] = {ep: float(value) for ep, value in zip(episodes, difference)}
    descriptive = seed_descriptive(per_episode)
    return {"complete": descriptive["complete"], "domain": domain, "metric": metric,
            "coefficients": [{"regime": r, "mode": m, "coefficient": c} for r, m, c in coefficients],
            "per_seed": per_seed, "across_seed_descriptive": descriptive,
            "direction": "negative means the first error contrast improves; difference-in-differences is fusion augmentation effect minus visual augmentation effect",
            "scope": "same original episodes; no multiplicity correction; no seed/frame pooling"}


def load_run(root, seed, regime, mode, sources, expected_weight=0.):
    directory = root / f"s{seed}" / regime / mode
    value = {"seed": seed, "regime": regime, "mode": mode, "directory": str(directory),
             "complete": False, "problems": [], "domains": {}, "config": None}
    config, marker, metrics, history = [read_json(directory / name, sources) for name in ("config.json", "completion.json", "metrics.json", "history.json")]
    value["config"] = config
    if any(item is None for item in (config, marker, metrics, history)):
        value["problems"].append("missing_config_completion_metrics_or_history")
        return value, None
    value.update(best_step=marker.get("best_step"), training_seconds=marker.get("training_seconds"),
                 peak_gpu_gb=marker.get("peak_gpu_gb"), allocated_trainable_parameters=marker.get("allocated_trainable_parameters"))
    if not marker.get("complete") or not marker.get("training_complete") or not marker.get("test_complete") or not metrics.get("complete"):
        value["problems"].append("run_not_complete")
    if (config.get("seed"), config.get("regime"), config.get("mode")) != (seed, regime, mode) or config.get("displacement_weight") != expected_weight:
        value["problems"].append("mode_seed_regime_or_objective_mismatch")
    for name, key in (("config.json", "config_sha256"), ("metrics.json", "metrics_sha256"), ("test_predictions.npz", "predictions_sha256")):
        path = directory / name
        if not path.is_file() or marker.get(key) != sha(path):
            value["problems"].append("artifact_hash_mismatch:" + name)
    if metrics.get("config_sha256") != sha(directory / "config.json") or metrics.get("checkpoint_sha256") != marker.get("checkpoint_sha256"):
        value["problems"].append("metrics_checkpoint_binding_mismatch")
    if not history or any(not np.isfinite(row.get("dev_episode_mean_mm", np.nan)) for row in history):
        value["problems"].append("invalid_development_history")
    else:
        selected = min(history, key=lambda row: (row["dev_episode_mean_mm"], row["step"]))
        if selected["step"] != marker.get("best_step") or selected["step"] != metrics.get("best_step"):
            value["problems"].append("not_the_dev_selected_checkpoint")
        if any(not np.isclose(selected["dev_episode_mean_mm"], candidate.get("dev_episode_mean_mm", np.nan), rtol=1e-6, atol=1e-5) for candidate in (marker, metrics)):
            value["problems"].append("selected_development_value_mismatch")
        expected_steps = sorted({1, *range(config["eval_every"], config["steps"] + 1, config["eval_every"])})
        if expected_steps[-1] != config["steps"]:
            expected_steps.append(config["steps"])
        if [row["step"] for row in history] != expected_steps or marker.get("train_steps") != config.get("steps"):
            value["problems"].append("incomplete_declared_training_budget")
        value["history"] = history
        value["dev_episode_mean_mm"] = selected["dev_episode_mean_mm"]
    identity = None
    if not value["problems"]:
        try:
            domains, identity, baseline = audit_predictions(directory / "test_predictions.npz")
            compare_recorded_domains(metrics["domains"], domains)
            if not np.isclose(baseline, config["reference_baseline_m"], rtol=0, atol=1e-8):
                raise ValueError("Physical baseline differs from configuration")
            value["domains"] = domains
            path = directory / "test_predictions.npz"
            sources[str(path.resolve())] = {"sha256": sha(path), "bytes": path.stat().st_size}
        except (ValueError, KeyError, OSError) as error:
            value["problems"].append(str(error))
    value["complete"] = not value["problems"]
    return value, identity


def summarize(args):
    root, sources, problems = args.input.resolve(), {}, []
    metadata = {name: read_json(root / (name + ".json"), sources) for name in ("summary", "normalization", "environment", "convergence")}
    runs, identity = {}, None
    for seed in SEEDS:
        for regime in REGIMES:
            for mode in MODES:
                value, current = load_run(root, seed, regime, mode, sources, args.displacement_weight)
                if current is not None:
                    if identity is None:
                        identity = current
                    elif any(not np.array_equal(current[key], identity[key]) for key in identity):
                        value["problems"].append("test_rows_targets_or_cameras_differ_between_runs")
                        value["complete"] = False
                runs[(seed, regime, mode)] = value
    configs = [value["config"] for value in runs.values() if value["config"] is not None]
    settings = [{key: value for key, value in config.items() if key not in ("seed", "regime", "mode")} for config in configs]
    matched = bool(settings) and all(value == settings[0] for value in settings)
    if not matched:
        problems.append("observed_run_settings_or_source_hashes_not_matched")
    for name in ("normalization", "environment"):
        if metadata[name] is None:
            problems.append("missing_root_" + name)
        for config in configs:
            if metadata[name] is not None and config.get(name + "_sha256") != sha(root / (name + ".json")):
                problems.append("root_" + name + "_hash_mismatch")
    convergence = read_json(args.budget_decision, sources) if args.budget_decision else metadata["convergence"]
    budget = configs[0].get("steps") if configs else None
    if not convergence or not convergence.get("complete"):
        problems.append("development_only_budget_decision_incomplete")
    elif budget == 2000 and convergence.get("extend_all_to_4000"):
        problems.append("development_rule_requires_4000_budget")
    elif budget == 4000 and args.budget_decision and not convergence.get("extend_all_to_4000"):
        problems.append("provided_2000_decision_did_not_require_extension")
    if budget not in (None, 2000, 4000):
        problems.append("unexpected_primary_training_budget")
    if metadata["summary"] is None or not metadata["summary"].get("full_design_complete"):
        problems.append("runner_full_design_not_complete")
    descriptive = {regime: {mode: {domain: {metric: seed_descriptive({seed: metric_episodes(runs[(seed, regime, mode)], domain, metric)
                        for seed in SEEDS if runs[(seed, regime, mode)]["complete"]})
                    for metric in SCALAR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                    for domain in DOMAINS.values()} for mode in MODES} for regime in REGIMES}
    augmentation = {mode: {domain: {metric: paired_contrast(runs, [("augmented", mode, 1), ("fixed", mode, -1)], domain, metric)
                    for metric in ERROR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                    for domain in DOMAINS.values()} for mode in MODES}
    pairs = (("late", "visual"), ("internal", "visual"), ("internal", "tactile"), ("late", "tactile"), ("internal", "late"))
    fusion = {regime: {a + "_minus_" + b: {domain: {metric: paired_contrast(runs, [(regime, a, 1), (regime, b, -1)], domain, metric)
                    for metric in ERROR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                    for domain in DOMAINS.values()} for a, b in pairs} for regime in REGIMES}
    interactions = {mode + "_minus_visual": {domain: {metric: paired_contrast(runs,
                    [("augmented", mode, 1), ("fixed", mode, -1), ("augmented", "visual", -1), ("fixed", "visual", 1)], domain, metric)
                    for metric in ERROR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                    for domain in DOMAINS.values()} for mode in ("late", "internal")}
    complete = not problems and len(runs) == 24 and all(value["complete"] for value in runs.values())
    result = {"report_version": 1, "complete": complete, "problems": sorted(set(problems)),
              "generated_utc": datetime.now(timezone.utc).isoformat(), "input_directory": str(root),
              "training_budget_steps": budget, "displacement_weight": args.displacement_weight,
              "expected_runs": 24, "completed_runs": sum(value["complete"] for value in runs.values()),
              "settings_comparability_verified": matched, "metadata": metadata, "budget_decision": convergence,
              "runs": list(runs.values()), "descriptive": descriptive,
              "paired_comparisons": {"augmentation_effect": augmentation, "fusion_contrasts": fusion, "difference_in_differences": interactions},
              "scope": {"model": "frozen VGGT, actual final global block and same adapter; camera-only primary unless nonzero weight explicitly reported",
                        "units": "base-frame millimetres, known calibration/physical baseline; no fitted prediction alignment",
                        "uncertainty": "eight original test episodes; three seeds descriptive; conditional CI averages seeds within episode first",
                        "cross_view": "consistency alone rewards constant predictions; interpret jointly with center and counterfactual accuracy",
                        "counterfactual": "nominal object to variant at same source/camera, source then episode equally weighted; zero-shift control mandatory",
                        "limitations": "same historical object/task/sources; static snapshots, no dynamics/policy/surface-reconstruction evidence"},
              "plots": {}, "sources": sources, "implementation_sha256": sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots and complete:
        result["plots"] = make_plots(result, args.output)
    elif args.plots:
        result["plots_skipped"] = "Full declared design incomplete; no final cross-seed figures fabricated"
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ("row_type", "regime", "mode", "domain", "seed", "complete", "training_budget_steps", "best_step",
              "center_mm", "center_ci95_low_mm", "center_ci95_high_mm", "center_sample_sd_mm", "p90_row_mm",
              "displacement_error_mm", "displacement_sample_sd_mm", "zero_shift_error_mm", "cross_view_mm", "cross_view_sample_sd_mm")
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs.values():
            for domain in DOMAINS.values():
                value = run["domains"].get(domain, {})
                center = value.get("center", {})
                interval = center.get("ci95_mm") or [None, None]
                writer.writerow({"row_type": "single_seed", "regime": run["regime"], "mode": run["mode"], "domain": domain,
                    "seed": run["seed"], "complete": run["complete"], "training_budget_steps": budget, "best_step": run.get("best_step"),
                    "center_mm": center.get("episode_mean_mm"), "center_ci95_low_mm": interval[0], "center_ci95_high_mm": interval[1],
                    "p90_row_mm": center.get("p90_mm"),
                    "displacement_error_mm": value.get("displacement", {}).get("source_equal_metrics", {}).get("displacement_error_mm", {}).get("mean"),
                    "zero_shift_error_mm": value.get("displacement", {}).get("source_equal_metrics", {}).get("zero_shift_error_mm", {}).get("mean"),
                    "cross_view_mm": (value.get("cross_view") or {}).get("episode_mean_mm")})
        for regime in REGIMES:
            for mode in MODES:
                for domain, values in descriptive[regime][mode].items():
                    ci = values["center_mm"]["conditional_episode_ci95"] or [None, None]
                    writer.writerow({"row_type": "three_seed_descriptive_conditional_ci", "regime": regime, "mode": mode, "domain": domain,
                        "complete": values["center_mm"]["complete"], "training_budget_steps": budget,
                        "center_mm": values["center_mm"]["mean"], "center_sample_sd_mm": values["center_mm"]["sample_sd"],
                        "center_ci95_low_mm": ci[0], "center_ci95_high_mm": ci[1],
                        "displacement_error_mm": values["displacement_error_mm"]["mean"], "displacement_sample_sd_mm": values["displacement_error_mm"]["sample_sd"],
                        "zero_shift_error_mm": values["zero_shift_error_mm"]["mean"],
                        "cross_view_mm": values.get("cross_view_mm", {}).get("mean"), "cross_view_sample_sd_mm": values.get("cross_view_mm", {}).get("sample_sd")})
    print(json.dumps({"complete": complete, "completed_runs": result["completed_runs"], "problems": result["problems"], "output": str(args.output)}, indent=2))
    return result


def make_plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fixed": "#6f8193", "augmented": "#2b826e"}
    plotted = {}
    for metric, stem, ylabel, domains in (
        ("center_mm", "center_by_camera_domain", "Center error (mm)", list(DOMAINS.values())),
        ("displacement_error_mm", "counterfactual_displacement", "Same-source displacement error (mm)", list(DOMAINS.values())),
        ("cross_view_mm", "cross_view_consistency", "Base-frame shift from nominal (mm)", ["near", "strong"])):
        fig, axes = plt.subplots(1, len(domains), figsize=(4.5 * len(domains), 4.1), squeeze=False)
        for axis, domain in zip(axes[0], domains):
            for j, regime in enumerate(REGIMES):
                x = np.arange(len(MODES)) + (j - .5) * .32
                values = [result["descriptive"][regime][mode][domain][metric] for mode in MODES]
                axis.bar(x, [v["mean"] for v in values], width=.30, yerr=[v["sample_sd"] for v in values],
                         label=regime, color=colors[regime], alpha=.78, capsize=3)
                for position, value in zip(x, values):
                    axis.scatter(np.repeat(position, 3) + [-.045, 0, .045], [value["seed_values"][str(seed)] for seed in SEEDS], color="#24372f", s=13, zorder=4)
            if metric == "displacement_error_mm":
                zero = result["descriptive"]["fixed"]["visual"][domain]["zero_shift_error_mm"]["mean"]
                axis.axhline(zero, color="#a7523d", linestyle="--", linewidth=1.2, label="zero shift control")
            axis.set_title(domain)
            axis.set_xticks(np.arange(len(MODES)), MODES, rotation=20)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.16)
            axis.set_axisbelow(True)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle("Three training seeds on the same original test episodes")
        fig.text(.5, .015, "Bars: seed mean; whiskers: sample SD across 3 seeds; dots: seeds 43/44/45. No frame/seed pooling.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .93))
        for suffix in ("png", "pdf"):
            path = output / (stem + "." + suffix)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plotted[stem + "_" + suffix] = str(path)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6), sharex=True, sharey=True)
    for j, mode in enumerate(MODES):
        for i, regime in enumerate(REGIMES):
            axis = axes[i, j]
            for run in result["runs"]:
                if run["mode"] == mode and run["regime"] == regime:
                    axis.plot([v["step"] for v in run["history"]], [v["dev_episode_mean_mm"] for v in run["history"]], label=str(run["seed"]))
            axis.set_title(regime + " / " + mode, fontsize=9)
            axis.grid(alpha=.15)
            if i == 1:
                axis.set_xlabel("Update steps")
            if j == 0:
                axis.set_ylabel("Common DEV error (mm)")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / ("development_curves." + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plotted["development_curves_" + suffix] = str(path)
    plt.close(fig)
    return plotted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/camera_pilot_20261026/main_2000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/camera_pilot_20261026")
    parser.add_argument("--budget-decision", type=Path, help="Original seed43-only 2000-step convergence.json if a 4000-step round was triggered")
    parser.add_argument("--displacement-weight", type=float, default=0., help="Keep optional objective follow-up in a separate input/output; primary is zero")
    parser.add_argument("--plots", action="store_true")
    summarize(parser.parse_args())
