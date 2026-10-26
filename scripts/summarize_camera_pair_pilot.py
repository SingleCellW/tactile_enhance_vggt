#!/usr/bin/env python3
"""Independent CPU summary of the conditional 12-run matched-pair follow-up.

This does not summarize or modify the 24-run camera-only primary. Both new
objectives use the same paired sampler; their difference isolates adding the
delta loss conditional on this sampler and the selected development checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from tactile_vggt.paths import artifact_path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_camera_pilot as base

SEEDS = base.SEEDS
MODES = ("visual", "internal")
OBJECTIVES = ("center", "delta")
WEIGHTS = {"center": 0., "delta": 1.}
OBJECTIVE_CONFIG_KEYS = {"objective", "displacement_weight"}
ENVIRONMENT_CONFIG_KEYS = {"environment_sha256", "environment_record"}
# Logical placement may differ between workers. All other recorded fields,
# including library versions, Python build, OS and GPU model, must match.
ENVIRONMENT_PLACEMENT_KEYS = {"device", "cuda_visible_devices", "python_executable"}
ENVIRONMENT_REQUIRED_KEYS = {"torch", "numpy", "python", "cuda_version", "gpu"}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DECLARED_STEPS = 4000


def digest_ok(value):
    return isinstance(value, str) and bool(HASH_PATTERN.fullmatch(value))


def config_differences(first, second, ignored=()):
    """Explicit exclusions only; unexpected source/optimizer fields cannot hide."""
    ignored = set(ignored)
    return sorted(key for key in set(first) | set(second)
                  if key not in ignored and first.get(key) != second.get(key))


def batch_scheme_problems(config):
    scheme = config.get("batch_scheme", {})
    expected = {"batch_size": 32, "ordinary_rows": 16, "object_pairs": 8,
                "forward_calls_per_optimizer_step": 1,
                "layout": "ordinary[0:16], first[16:24], second[24:32]",
                "weights": WEIGHTS}
    return ["batch_scheme_mismatch:" + key for key, value in expected.items() if scheme.get(key) != value]


def load_pair_run(root, seed, objective, mode, sources, expected_steps):
    value, identity = base.load_run(root, seed, "augmented", mode, sources, WEIGHTS[objective])
    value["objective"] = objective
    directory = root / f"s{seed}" / "augmented" / mode
    marker = base.read_json(directory / "completion.json", sources)
    metrics = base.read_json(directory / "metrics.json", sources)
    if metrics is not None:
        if (metrics.get("seed"), metrics.get("regime"), metrics.get("mode"), metrics.get("objective")) != (seed, "augmented", mode, objective):
            value["problems"].append("metrics_run_identity_mismatch")
        if metrics.get("train_steps") != expected_steps:
            value["problems"].append("metrics_training_budget_mismatch")
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file():
        value["problems"].append("missing_best_checkpoint")
    else:
        digest = base.sha(checkpoint)
        sources[str(checkpoint.resolve())] = {"sha256": digest, "bytes": checkpoint.stat().st_size}
        value["checkpoint_file_sha256"] = digest
        if marker is None or marker.get("checkpoint_sha256") != digest or metrics is None or metrics.get("checkpoint_sha256") != digest:
            value["problems"].append("best_checkpoint_file_hash_mismatch")
    config = value.get("config")
    if config is not None:
        if config.get("objective") != objective:
            value["problems"].append("objective_name_mismatch")
        if config.get("steps") != expected_steps:
            value["problems"].append("training_budget_differs_from_declared_followup")
        value["problems"].extend(batch_scheme_problems(config))
        # New runners store immutable per-worker records; the fallback only
        # supports the earlier root environment.json contract, still SHA bound.
        record = config.get("environment_record", "environment.json")
        if not isinstance(record, str) or Path(record).is_absolute():
            value["problems"].append("invalid_environment_record_path")
        else:
            path = (root / record).resolve()
            if not path.is_relative_to(root.resolve()):
                value["problems"].append("environment_record_escapes_objective_root")
            else:
                environment = base.read_json(path, sources)
                if environment is None or base.sha(path) != config.get("environment_sha256"):
                    value["problems"].append("environment_record_missing_or_sha_mismatch")
                elif not ENVIRONMENT_REQUIRED_KEYS <= set(environment):
                    value["problems"].append("environment_record_lacks_required_software_or_gpu_fields")
                else:
                    value["environment"] = environment
                    value["environment_source"] = str(path)
    if marker is not None and marker.get("training_complete"):
        integrity = {key: marker.get(key) for key in ("initialization_sha256", "row_stream_sha256", "train_steps", "first_batch_row_ids")}
        value["training_integrity"] = integrity
        if any(not digest_ok(integrity[key]) for key in ("initialization_sha256", "row_stream_sha256")):
            value["problems"].append("missing_or_invalid_initialization_or_full_row_stream_digest")
        first = integrity["first_batch_row_ids"]
        if not isinstance(first, list) or len(first) != 32 or any(type(row) is not int or row < 0 for row in first):
            value["problems"].append("invalid_first_32_row_batch")
        if metrics is not None:
            if metrics.get("objective") != objective:
                value["problems"].append("metrics_objective_mismatch")
            for key in ("initialization_sha256", "row_stream_sha256"):
                if metrics.get(key) != integrity[key]:
                    value["problems"].append("metrics_training_integrity_mismatch:" + key)
        history = value.get("history", [])
        if history:
            if history[-1].get("row_stream_sha256") != integrity["row_stream_sha256"]:
                value["problems"].append("last_history_full_row_stream_digest_mismatch")
            if any(not digest_ok(row.get("row_stream_sha256")) for row in history):
                value["problems"].append("missing_history_row_stream_digest")
    value["complete"] = not value["problems"]
    return value, identity


def comparability_audit(runs):
    """Check matched objectives and identical row streams across both modes."""
    pairs, problems = [], []
    for seed in SEEDS:
        streams = []
        for mode in MODES:
            first, second = (runs[(seed, objective, mode)] for objective in OBJECTIVES)
            row = {"seed": seed, "mode": mode, "complete": False, "problems": []}
            if not all(run.get("config") and run.get("training_integrity") and run.get("environment") for run in (first, second)):
                row["problems"].append("missing_config_training_integrity_or_verified_environment")
            else:
                row["config_differences_excluding_objective_and_verified_environment_record"] = config_differences(first["config"], second["config"], OBJECTIVE_CONFIG_KEYS | ENVIRONMENT_CONFIG_KEYS)
                if row["config_differences_excluding_objective_and_verified_environment_record"]:
                    row["problems"].append("objective_control_has_other_configuration_differences")
                differences = config_differences(first["environment"], second["environment"])
                row["environment_placement_differences"] = [key for key in differences if key in ENVIRONMENT_PLACEMENT_KEYS]
                if any(key not in ENVIRONMENT_PLACEMENT_KEYS for key in differences):
                    row["problems"].append("objective_control_software_os_or_gpu_mismatch")
                for key in ("initialization_sha256", "row_stream_sha256", "train_steps", "first_batch_row_ids"):
                    a, b = first["training_integrity"].get(key), second["training_integrity"].get(key)
                    if a is None or a != b:
                        row["problems"].append("objective_control_mismatch:" + key)
                    row[key] = a if a == b else None
                for run in (first, second):
                    streams.append(run["training_integrity"].get("row_stream_sha256"))
            row["complete"] = not row["problems"]
            pairs.append(row)
        if len(streams) == 4 and (any(not digest_ok(stream) for stream in streams) or len(set(streams)) != 1):
            problems.append(f"training_inputs_differ_between_modes_or_objectives:seed{seed}")
    configs = [run["config"] for run in runs.values() if run.get("config")]
    if configs:
        ignored = OBJECTIVE_CONFIG_KEYS | ENVIRONMENT_CONFIG_KEYS | {"seed", "mode"}
        for config in configs[1:]:
            diff = config_differences(configs[0], config, ignored)
            if diff:
                problems.append("architecture_budget_data_or_implementation_differs:" + ",".join(diff))
    else:
        problems.append("no_run_configurations_available")
    environments = [run["environment"] for run in runs.values() if run.get("environment")]
    if len(environments) != len(runs):
        problems.append("not_all_run_environments_verified")
    elif any(config_differences(environments[0], value, ENVIRONMENT_PLACEMENT_KEYS) for value in environments[1:]):
        problems.append("software_os_or_gpu_model_differs_between_seeds_or_modes")
    return {"complete": not problems and all(row["complete"] for row in pairs),
            "allowed_cross_objective_config_differences": sorted(OBJECTIVE_CONFIG_KEYS),
            "environment_record_exceptions": {"config_keys": sorted(ENVIRONMENT_CONFIG_KEYS),
                "allowed_placement_keys_after_sha_verification": sorted(ENVIRONMENT_PLACEMENT_KEYS),
                "required_identical_fields": sorted(ENVIRONMENT_REQUIRED_KEYS),
                "rule": "read and SHA-verify every referenced record; all non-placement environment fields must match"},
            "pairs": pairs, "problems": sorted(set(problems))}


def paired_contrast(runs, coefficients, domain, metric):
    value = base.paired_contrast(runs, coefficients, domain, metric)
    value["coefficients"] = [{"objective": objective, "mode": mode, "coefficient": coefficient}
                             for objective, mode, coefficient in coefficients]
    value["direction"] = "negative means reduced error; interaction is delta-loss effect on internal minus its effect on visual"
    value["scope"] = "same original episodes and three matched seeds; conditional episode CI, no multiplicity correction or frame/seed pooling"
    return value


def summarize(args):
    root = args.input.resolve()
    roots = {"center": (args.center_root or root / "paired_center").resolve(),
             "delta": (args.delta_root or root / "paired_delta").resolve()}
    sources, problems, provenance_warnings, runs, identity = {}, [], [], {}, None
    if args.expected_steps != DECLARED_STEPS:
        problems.append("followup_requires_predeclared_4000_step_budget")
    protocol = base.read_json(root / "pair_protocol.json", sources)
    runner_integrity = base.read_json(root / "paired_training_integrity.json", sources)
    metadata = {objective: {name: base.read_json(path / (name + ".json"), sources)
                           for name in ("summary", "normalization", "environment")}
                for objective, path in roots.items()}
    for objective, path in roots.items():
        # Workers write invocation-specific summaries. Preserve every saved
        # subset and the last-write summary, but derive completion from runs.
        metadata[objective]["subset_summaries"] = {
            item.name: base.read_json(item, sources) for item in sorted(path.glob("summary_seeds_*_modes_*.json"))}
        for seed in SEEDS:
            for mode in MODES:
                run, current = load_pair_run(path, seed, objective, mode, sources, args.expected_steps)
                if current is not None:
                    if identity is None:
                        identity = current
                    elif set(identity) != set(current) or any(not np.array_equal(identity[key], current[key]) for key in identity):
                        run["problems"].append("test_rows_targets_or_cameras_differ_between_runs")
                        run["complete"] = False
                runs[(seed, objective, mode)] = run
        for name in ("normalization",):
            if metadata[objective][name] is None:
                problems.append("missing_root_metadata:" + objective + "/" + name)
            else:
                digest = base.sha(path / (name + ".json"))
                for run in runs.values():
                    if run["objective"] == objective and run.get("config") and run["config"].get(name + "_sha256") != digest:
                        problems.append("root_metadata_hash_mismatch:" + objective + "/" + name)
        summary = metadata[objective]["summary"]
        if not summary or not summary.get("full_design_complete"):
            provenance_warnings.append("last_worker_summary_missing_or_subset_only:" + objective)
    audit = comparability_audit(runs)
    if not audit["complete"]:
        problems.append("matched_objective_comparability_not_verified")
    if not protocol or not protocol.get("conditional_followup"):
        problems.append("missing_conditional_followup_protocol")
    else:
        for key, expected in (("steps", args.expected_steps), ("seeds", list(SEEDS)), ("modes", list(MODES)), ("objectives", list(OBJECTIVES))):
            if protocol.get(key) != expected:
                problems.append("protocol_mismatch:" + key)
        for run in runs.values():
            config = run.get("config")
            if config is not None:
                for key in ("data_manifest_sha256", "prefix_manifest_sha256", "batch_scheme"):
                    if protocol.get(key) != config.get(key):
                        problems.append("protocol_source_or_sampling_mismatch:" + key)
                for key in ("learning_rate", "eval_every", "hidden_dim", "runner_sha256"):
                    if key in protocol and protocol[key] != config.get(key):
                        problems.append("protocol_hyperparameter_or_runner_mismatch:" + key)
    if not runner_integrity or not runner_integrity.get("complete"):
        provenance_warnings.append("cached_runner_integrity_missing_or_partial")
    elif audit["complete"]:
        recorded = {(row.get("seed"), row.get("mode")): row for row in runner_integrity.get("pairs", [])}
        for row in audit["pairs"]:
            other = recorded.get((row["seed"], row["mode"]), {})
            if not other.get("matched") or any(row.get(key) != other.get(key) for key in ("train_steps", "initialization_sha256", "row_stream_sha256")):
                provenance_warnings.append("cached_runner_integrity_disagrees_with_actual_run_artifacts")
    descriptive = {objective: {mode: {domain: {metric: base.seed_descriptive({seed: base.metric_episodes(runs[(seed, objective, mode)], domain, metric)
                        for seed in SEEDS if runs[(seed, objective, mode)]["complete"]})
                    for metric in base.SCALAR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                    for domain in base.DOMAINS.values()} for mode in MODES} for objective in OBJECTIVES}

    contrast_runs = runs if audit["complete"] else {}

    def contrasts(coefficients):
        return {domain: {metric: paired_contrast(contrast_runs, coefficients, domain, metric)
                         for metric in base.ERROR_METRICS if not (metric == "cross_view_mm" and domain == "nominal")}
                for domain in base.DOMAINS.values()}

    comparisons = {
        "delta_minus_center": {mode: contrasts([("delta", mode, 1), ("center", mode, -1)]) for mode in MODES},
        "internal_minus_visual": {objective: contrasts([(objective, "internal", 1), (objective, "visual", -1)]) for objective in OBJECTIVES},
        "difference_in_differences": contrasts([("delta", "internal", 1), ("center", "internal", -1),
                                                  ("delta", "visual", -1), ("center", "visual", 1)])}
    completed = sum(run["complete"] for run in runs.values())
    result = {"report_version": 1, "complete": completed == 12 and not problems and audit["complete"],
              "generated_utc": datetime.now(timezone.utc).isoformat(), "input_directory": str(root),
              "objective_directories": {key: str(path) for key, path in roots.items()},
              "expected_runs": 12, "completed_runs": completed, "training_budget_steps": args.expected_steps,
              "completion_basis": "all 12 actual run configs/completions/best checkpoints/histories/metrics/predictions and independent data/environment/initialization/row-stream matching; worker summaries are provenance only",
              "provenance_warnings": sorted(set(provenance_warnings)),
              "problems": sorted(set(problems)), "comparability": audit, "protocol": protocol,
              "runner_integrity": runner_integrity, "metadata": metadata, "runs": list(runs.values()),
              "descriptive": descriptive, "paired_comparisons": comparisons, "sources": sources,
              "scope": {"design": "conditional follow-up; augmented camera training only; visual/internal, three seeds, two matched objectives",
                        "comparison": "paired_delta versus NEW paired_center; primary camera-only 24-run experiment has different sampling and is not a matched objective control",
                        "sampling": "one batch32:16 ordinary rows plus8 same-camera source pairs; centre on all32, optional lambda1 delta on8 pairs; one forward per optimizer step",
                        "checkpoint": "each objective independently selects minimum common DEV centre error; the test comparison includes that predeclared selection rule",
                        "units": "base-frame millimetres and known calibration/physical baseline; no prediction alignment fitted on test",
                        "uncertainty": "eight original test episodes; source then episode equal for displacement; seed variation descriptive; conditional CI averages seeds within episode first",
                        "interpretation": "cross-view consistency alone can reward constant outputs; check centre and object-displacement response with zero-shift control together",
                        "limitations": "same historical object/task/static snapshots; no policy, dynamic reconstruction, surface-completion or unknown-calibration evidence"},
              "plots": {}, "implementation_sha256": base.sha(__file__),
              "independent_metric_helper_sha256": base.sha(base.__file__)}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots and result["complete"]:
        result["plots"] = make_plots(result, args.output)
    elif args.plots:
        result["plots_skipped"] = "Declared12-run design incomplete; no final cross-seed figures generated"
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(result, args.output / "results.csv")
    print(json.dumps({key: result[key] for key in ("complete", "completed_runs", "expected_runs", "problems")}, indent=2))
    return result


def write_csv(result, path):
    fields = ("row_type", "objective", "mode", "domain", "seed", "complete", "training_budget_steps", "best_step",
              "center_mm", "center_sample_sd_mm", "center_ci95_low_mm", "center_ci95_high_mm", "displacement_error_mm",
              "displacement_sample_sd_mm", "zero_shift_error_mm", "cross_view_mm", "cross_view_sample_sd_mm")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            for domain in base.DOMAINS.values():
                value = run["domains"].get(domain, {})
                center = value.get("center", {})
                interval = center.get("ci95_mm") or [None, None]
                metrics = value.get("displacement", {}).get("source_equal_metrics", {})
                writer.writerow({"row_type": "single_seed", "objective": run["objective"], "mode": run["mode"],
                    "domain": domain, "seed": run["seed"], "complete": run["complete"], "best_step": run.get("best_step"),
                    "training_budget_steps": result["training_budget_steps"], "center_mm": center.get("episode_mean_mm"),
                    "center_ci95_low_mm": interval[0], "center_ci95_high_mm": interval[1],
                    "displacement_error_mm": metrics.get("displacement_error_mm", {}).get("mean"),
                    "zero_shift_error_mm": metrics.get("zero_shift_error_mm", {}).get("mean"),
                    "cross_view_mm": (value.get("cross_view") or {}).get("episode_mean_mm")})
        for objective in OBJECTIVES:
            for mode in MODES:
                for domain, values in result["descriptive"][objective][mode].items():
                    interval = values["center_mm"]["conditional_episode_ci95"] or [None, None]
                    writer.writerow({"row_type": "three_seed_descriptive_conditional_ci", "objective": objective, "mode": mode,
                        "domain": domain, "complete": values["center_mm"]["complete"], "training_budget_steps": result["training_budget_steps"],
                        "center_mm": values["center_mm"]["mean"], "center_sample_sd_mm": values["center_mm"]["sample_sd"],
                        "center_ci95_low_mm": interval[0], "center_ci95_high_mm": interval[1],
                        "displacement_error_mm": values["displacement_error_mm"]["mean"], "displacement_sample_sd_mm": values["displacement_error_mm"]["sample_sd"],
                        "zero_shift_error_mm": values["zero_shift_error_mm"]["mean"],
                        "cross_view_mm": values.get("cross_view_mm", {}).get("mean"), "cross_view_sample_sd_mm": values.get("cross_view_mm", {}).get("sample_sd")})


def make_plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    colors = {"center": "#6f8193", "delta": "#2b826e"}
    for metric, stem, label, domains in (
        ("center_mm", "paired_center_error", "Centre error (mm)", list(base.DOMAINS.values())),
        ("displacement_error_mm", "paired_displacement_error", "Same-source displacement error (mm)", list(base.DOMAINS.values())),
        ("cross_view_mm", "paired_cross_view", "Base-frame shift from nominal (mm)", ["near", "strong"])):
        fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), squeeze=False)
        for axis, domain in zip(axes[0], domains):
            for j, objective in enumerate(OBJECTIVES):
                positions = np.arange(len(MODES)) + (j - .5) * .32
                values = [result["descriptive"][objective][mode][domain][metric] for mode in MODES]
                axis.bar(positions, [v["mean"] for v in values], width=.30, yerr=[v["sample_sd"] for v in values],
                         label="centre only" if objective == "center" else "centre + delta", color=colors[objective], alpha=.8, capsize=3)
                for position, value in zip(positions, values):
                    axis.scatter(position + np.array([-.04, 0, .04]), [value["seed_values"][str(seed)] for seed in SEEDS], s=14, color="#263c32", zorder=3)
            if metric == "displacement_error_mm":
                control = result["descriptive"]["center"]["visual"][domain]["zero_shift_error_mm"]["mean"]
                axis.axhline(control, linestyle="--", color="#a7523d", label="zero shift control")
            axis.set_title(domain)
            axis.set_xticks(np.arange(len(MODES)), MODES)
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=.15)
            axis.set_axisbelow(True)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle("Matched sampling: added displacement objective")
        fig.text(.5, .015, "Bars: mean over3 seeds; whiskers: sample SD; dots:43/44/45. Same original test episodes.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, .93))
        for suffix in ("png", "pdf"):
            path = output / (stem + "." + suffix)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            paths[stem + "_" + suffix] = str(path)
        plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=artifact_path('outputs', 'camera_pair_pilot_20261026'))
    parser.add_argument("--center-root", type=Path)
    parser.add_argument("--delta-root", type=Path)
    parser.add_argument("--output", type=Path, default=artifact_path('reports', 'camera_pair_pilot_20261026'))
    parser.add_argument("--expected-steps", type=int, choices=[DECLARED_STEPS], default=DECLARED_STEPS)
    parser.add_argument("--plots", action="store_true")
    arguments = parser.parse_args()
    if arguments.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    summarize(arguments)
