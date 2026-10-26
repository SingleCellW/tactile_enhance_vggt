#!/usr/bin/env python3
"""Summarize declared token-pilot runs without pooling frames across seeds.

Only s43_refined_v2 is the primary seven-mode ablation. Seeds 44/45 repeat the
visual/late/internal comparison. The initial s43 run is screening, and the
aborted s43_refined run is always excluded. Missing work stays incomplete.
This script reads results and writes a report; it never starts model inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODES = ("state", "visual", "tactile", "late", "internal", "no_xyz", "mismatch")
KEY_MODES = ("visual", "late", "internal")
RUNS = ((43, "s43_refined_v2", PRIMARY_MODES),
        (44, "s44_refined_v2", KEY_MODES),
        (45, "s45_refined_v2", KEY_MODES))


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def source_json(path, sources):
    if not path.exists():
        return None
    value = read_json(path)
    sources[str(path.resolve())] = fingerprint(path)
    return value


def metric_row(metrics):
    all_rows = metrics.get("all", {})
    counterfactual = metrics.get("counterfactual") or {}
    high = metrics.get("subgroups", {}).get("high_occlusion", {})
    contact = metrics.get("subgroups", {}).get("contact", {})
    no_contact = metrics.get("subgroups", {}).get("no_contact", {})
    zeroed = (metrics.get("test_zero_pressure") or {}).get("all", {})
    interval = all_rows.get("group_mean_ci95_mm") or [None, None]
    delta_interval = counterfactual.get("group_mean_ci95_mm") or [None, None]
    return {"observations": all_rows.get("observations"),
            "episodes": all_rows.get("independent_groups"),
            "episode_mean_mm": all_rows.get("group_mean_mm"),
            "episode_ci95_low_mm": interval[0], "episode_ci95_high_mm": interval[1],
            "row_mean_mm": all_rows.get("mean_mm"), "row_median_mm": all_rows.get("median_mm"),
            "row_p90_mm": all_rows.get("p90_mm"),
            "counterfactual_episode_mean_mm": counterfactual.get("group_mean_mm"),
            "counterfactual_ci95_low_mm": delta_interval[0],
            "counterfactual_ci95_high_mm": delta_interval[1],
            "counterfactual_pairs": counterfactual.get("pairs"),
            "high_occlusion_episode_mean_mm": high.get("group_mean_mm"),
            "contact_episode_mean_mm": contact.get("group_mean_mm"),
            "observed_no_contact_episode_mean_mm": no_contact.get("group_mean_mm"),
            "ood_zero_pressure_episode_mean_mm": zeroed.get("group_mean_mm"),
            "best_step": metrics.get("best_step"), "train_steps": metrics.get("train_steps"),
            "trainable_parameters": metrics.get("trainable_parameters")}


def primary_evidence_summary(run):
    """Expose negative controls without equating OOD sensitivity with accuracy."""
    comparisons = {}
    for key, value in run["paired_comparisons"].items():
        interval = value.get("ci95")
        comparisons[key] = {"mean_difference_mm": value.get("mean_difference"), "ci95_mm": interval,
                            "interval_includes_zero": None if not interval else interval[0] <= 0 <= interval[1],
                            "direction": "positive means internal has greater error; an interval including zero does not prove equivalence"}
    internal = (run["metrics"].get("internal", {}).get("counterfactual") or {}).get("group_mean_mm")
    deltas = {}
    for other in ("visual", "state", "late", "no_xyz"):
        value = (run["metrics"].get(other, {}).get("counterfactual") or {}).get("group_mean_mm")
        if internal is not None and value is not None:
            deltas[other] = {"internal_counterfactual_error_mm": internal, "other_counterfactual_error_mm": value,
                             "internal_minus_other_mm": internal-value,
                             "internal_has_larger_error": internal > value}
    return {"primary_complete": run["complete"], "paired_center_error_comparisons": comparisons,
            "counterfactual_shift_comparisons": deltas,
            "interpretation_limits": ["Lower absolute center error alone does not establish better spatial tactile alignment.",
                "Counterfactual shift error measures whether predictions follow same-hand-state object interventions; smaller is better. A constant within-source prediction is a relevant control.",
                "test_zero_pressure is an out-of-distribution intervention on recorded contact observations; it is not performance on the observed no_contact subgroup.",
                "Observed no_contact results and zero-pressure sensitivity are reported separately.",
                "A confidence interval containing zero leaves the direction unresolved; it does not prove equality or absence of an effect."]}


def representation_run(directory, seed, expected_modes, sources):
    summary = source_json(directory / "summary.json", sources)
    metrics, histories, conflicts = {}, {}, []
    for mode in expected_modes:
        individual = source_json(directory / mode / "metrics.json", sources)
        final = (summary or {}).get("results", {}).get(mode)
        if final is not None and individual is not None and final != individual:
            conflicts.append(f"{mode}: final summary and individual metrics disagree")
        selected = final if final is not None else individual
        if selected is not None:
            if selected.get("units") != "mm":
                conflicts.append(f"{mode}: metric units are not explicitly mm")
            metrics[mode] = selected
        history = source_json(directory / mode / "history.json", sources)
        if history is not None:
            histories[mode] = history
    missing = [mode for mode in expected_modes if mode not in metrics]
    if summary and int(summary.get("args", {}).get("seed", seed)) != seed:
        conflicts.append("run seed differs from the declared directory seed")
    comparisons = (summary or {}).get("comparisons", {})
    return {"seed": seed, "directory": str(directory.resolve()),
            "complete": summary is not None and not missing and not conflicts,
            "final_summary_present": summary is not None, "expected_modes": list(expected_modes),
            "missing_modes": missing, "conflicts": conflicts,
            "args": (summary or {}).get("args"), "splits": (summary or {}).get("splits"),
            "data_manifest_sha256": (summary or {}).get("data_manifest_sha256"),
            "prefix_manifest_sha256": (summary or {}).get("prefix_manifest_sha256"),
            "metrics": metrics, "table": {mode: metric_row(value) for mode, value in metrics.items()},
            "paired_comparisons": comparisons, "histories": histories,
            "partial_note": None if summary else "Per-mode completed metrics are provisional until the declared run summary exists."}


def cross_seed_report(runs):
    complete = all(run["complete"] for run in runs)
    problems = []
    for key in ("data_manifest_sha256", "prefix_manifest_sha256"):
        values = {run[key] for run in runs if run[key] is not None}
        if len(values) > 1:
            problems.append(f"Runs have different {key}")
    signatures = []
    for run in runs:
        if run["args"]:
            signatures.append(tuple(run["args"].get(key) for key in
                              ("steps", "batch_size", "learning_rate", "whiten_target", "hidden_dim")))
    if len(set(signatures)) > 1:
        problems.append("Run training budgets or architecture hyperparameters differ")
    available, aggregates = {}, {}
    fields = ("episode_mean_mm", "row_median_mm", "row_p90_mm", "counterfactual_episode_mean_mm")
    for mode in KEY_MODES:
        values = [{"seed": run["seed"], **run["table"][mode]}
                  for run in runs if mode in run["table"] and run["complete"]]
        available[mode] = values
        aggregate = {}
        for field in fields:
            samples = [item[field] for item in values if item[field] is not None]
            aggregate[field] = {"individual_values": [{"seed": item["seed"], "value": item[field]} for item in values],
                                "mean": mean(samples) if samples else None,
                                "sample_std": stdev(samples) if len(samples) > 1 else None,
                                "seed_count": len(samples)}
        aggregates[mode] = aggregate
    comparisons = {}
    for other in KEY_MODES:
        if other == "internal":
            continue
        key = f"internal_minus_{other}"
        values = [{"seed": run["seed"], **run["paired_comparisons"][key]}
                  for run in runs if run["complete"] and key in run["paired_comparisons"]]
        differences = [item["mean_difference"] for item in values]
        comparisons[key] = {"individual": values, "mean_difference_across_seeds_mm": mean(differences) if differences else None,
                            "sample_std_across_seeds_mm": stdev(differences) if len(differences) > 1 else None,
                            "seed_count": len(values), "direction": "positive means internal has larger error"}
    return {"complete": complete and not problems, "problems": problems,
            "expected_seeds": [43, 44, 45], "expected_modes": list(KEY_MODES),
            "individual": available, "descriptive_seed_statistics": aggregates,
            "paired_differences_by_seed": comparisons,
            "aggregation_note": "Arithmetic mean and sample SD of per-seed metrics; no pooling of observations or averaging of confidence intervals. All seeds share the same eight held-out episodes.",
            "partial_note": None if complete else "Available completed seeds are shown; this is not the completed three-seed comparison."}


def policy_report(directory, sources):
    summary = source_json(directory / "summary.json", sources)
    config = source_json(directory / "eval_config.json", sources)
    plan = source_json(directory / "case_plan.json", sources)
    expected_modes = list((config or {}).get("modes", {})) or list(KEY_MODES)
    models = dict((summary or {}).get("models", {}))
    for mode in expected_modes:
        individual = source_json(directory / mode / "summary.json", sources)
        if mode not in models and individual is not None:
            models[mode] = individual
    expected_cases = len((plan or {}).get("selected_case_ids", [])) or 36
    problems = []
    if expected_cases != 36:
        problems.append(f"Only {expected_cases} cases selected; the requested full evaluation has 36")
    if config and (config.get("control_ticks") != 600 or config.get("sensor_velocity_dt") != .01):
        problems.append("Physical horizon or native sensor dt differs from the declared protocol")
    table, training_records, training_signatures = {}, {}, []
    for mode, metrics in models.items():
        count = metrics.get("cases", 0)
        interval = metrics.get("strict_stable_goal_wilson_ci95") or [None, None]
        steps = []
        for row in metrics.get("results", []):
            if not row.get("complete") or row.get("ticks") != 600:
                problems.append(f"{mode}: incomplete case or wrong control horizon")
            steps.append(row.get("case", {}).get("case_id"))
        if len(set(steps)) != len(steps):
            problems.append(f"{mode}: repeated case IDs")
        if count != len(steps):
            problems.append(f"{mode}: case count differs from result records")
        policy = (config or {}).get("modes", {}).get(mode, {}).get("policy", "")
        original_checkpoint = Path(policy)
        checkpoint = original_checkpoint
        if not checkpoint.exists() and directory.parent.name in original_checkpoint.parts:
            position = original_checkpoint.parts.index(directory.parent.name)
            checkpoint = directory.parent.joinpath(*original_checkpoint.parts[position + 1:])
        training = source_json(checkpoint / "train_config.json", sources) if policy else None
        if training is None and policy:
            training = source_json(checkpoint.parent.parent / "train_config.json", sources)
        checkpoint_record = source_json(checkpoint / "checkpoint_complete.json", sources) if policy else None
        checkpoint_steps = (checkpoint_record or {}).get("step")
        if checkpoint_steps is None and original_checkpoint.name.isdigit():
            checkpoint_steps = int(original_checkpoint.name)
        if checkpoint_steps is not None and checkpoint_steps != 6000:
            problems.append(f"{mode}: checkpoint is {checkpoint_steps} rather than 6000 steps")
        if training is not None:
            signature = {key: training.get(key) for key in ("policy", "steps", "seed", "batch_size", "learning_rate", "weight_decay", "amp", "data_manifest_sha256")}
            training_records[mode] = signature
            training_signatures.append(json.dumps(signature, sort_keys=True))
            if training.get("steps") != 6000:
                problems.append(f"{mode}: ACT training budget is not 6000 steps")
        table[mode] = {"cases": count, "strict_successes": metrics.get("strict_stable_goal_count"),
                       "strict_success_rate": metrics.get("strict_stable_goal_rate"),
                       "success_wilson_low": interval[0], "success_wilson_high": interval[1],
                       "lifted_count": metrics.get("lifted_count"), "lift_then_drop_count": metrics.get("lift_then_drop_count"),
                       "mean_goal_center_distance_mm": None if metrics.get("mean_final_goal_center_xy_distance_m") is None else metrics["mean_final_goal_center_xy_distance_m"] * 1000,
                       "mean_goal_region_distance_mm": None if metrics.get("mean_final_goal_region_xy_distance_m") is None else metrics["mean_final_goal_region_xy_distance_m"] * 1000,
                       "wall_seconds": metrics.get("wall_seconds"), "encoder_refresh_mean_seconds": metrics.get("encoder_refresh_mean_seconds"),
                       "policy_checkpoint": policy or None, "checkpoint_steps": checkpoint_steps}
    comparability_verified = len(training_records) == len(expected_modes) and len(set(training_signatures)) == 1
    if len(set(training_signatures)) > 1:
        problems.append("ACT architecture, budget, seed, optimizer, precision or data hashes differ across modes")
    missing = [mode for mode in expected_modes if mode not in models or models[mode].get("cases") != 36]
    complete = bool(summary and summary.get("complete") and config and plan and not missing and not problems and comparability_verified)
    return {"complete": complete, "directory": str(directory.resolve()), "expected_modes": expected_modes,
            "expected_cases_per_mode": 36, "missing_or_partial_modes": missing, "problems": problems,
            "table": table, "paired_comparisons": (summary or {}).get("paired_comparisons", {}),
            "act_comparability_verified": comparability_verified,
            "act_training": training_records,
            "provenance_note": None if comparability_verified else "Need all selected checkpoints' train_config.json to verify identical ACT architecture and 6000-step training budget.",
            "protocol": (summary or {}).get("protocol", (config or {}).get("protocol")),
            "source_split_roles": (summary or {}).get("source_split_roles", (plan or {}).get("source_split_roles")),
            "case_plan": plan, "execution_horizon": (config or {}).get("execution_horizon"),
            "checkpoint_selection": (config or {}).get("checkpoint_selection"),
            "time_semantics": (summary or {}).get("time_semantics", "Simulation pauses for inference; this is not hard-real-time deployment."),
            "uncertainty_scope": (summary or {}).get("scope", "Intervals describe paired XY reset interventions around one source episode, not independent source episodes or training-seed replication."),
            "success_scope": "Inherited worker strict stable goal: center in goal region, low speed, no contact for one second. It does not require the whole object bounding box to lie in the receptacle."}


def dev_selected_report(directory, sources):
    summary = source_json(directory / "summary.json", sources)
    config = source_json(directory / "run_config.json", sources)
    table = {}
    for mode, metrics in (summary or {}).get("models", {}).items():
        selected = summary["selection"][mode]
        table[mode] = {
            "cases": metrics["cases"], "strict_successes": metrics["strict_stable_goal_count"],
            "strict_success_rate": metrics["strict_stable_goal_rate"],
            "success_wilson_ci95": metrics["strict_stable_goal_wilson_ci95"],
            "lifted_count": metrics["lifted_count"], "lift_then_drop_count": metrics["lift_then_drop_count"],
            "mean_goal_center_distance_mm": metrics["mean_final_goal_center_xy_distance_m"]*1000,
            "mean_goal_region_distance_mm": metrics["mean_final_goal_region_xy_distance_m"]*1000,
            "training_budget_steps": selected["training_budget_steps"],
            "selected_step": selected["selected_step"], "selection_value": selected["selection_value"]}
    complete = bool(summary and summary.get("complete") and config and set(table) == set(KEY_MODES)
                    and all(row["cases"] == 36 and row["training_budget_steps"] == 6000 for row in table.values()))
    return {"expected": config is not None, "complete": complete, "table": table,
            "selection": (summary or config or {}).get("selection"),
            "paired_comparisons": (summary or {}).get("paired_comparisons"),
            "scope": (summary or config or {}).get("scope"),
            "child_provenance": (summary or {}).get("child_provenance")}


def csv_rows(report):
    rows = []
    primary = report["representation"]["primary"]
    for mode, metric in primary["table"].items():
        rows.append({"section": "representation_primary", "seed": 43, "mode": mode,
                     "complete": primary["complete"], **metric})
    for run in report["representation"]["seed_runs"]:
        for mode in KEY_MODES:
            if mode in run["table"]:
                rows.append({"section": "representation_seed", "seed": run["seed"], "mode": mode,
                             "complete": run["complete"], **run["table"][mode]})
    for key, value in primary["paired_comparisons"].items():
        interval = value.get("ci95") or [None, None]
        rows.append({"section": "primary_paired_error_difference", "seed": 43, "comparison": key,
                     "complete": primary["complete"], "difference_mm": value.get("mean_difference"),
                     "difference_ci95_low_mm": interval[0], "difference_ci95_high_mm": interval[1],
                     "episodes": value.get("group_count")})
    cross = report["representation"]["cross_seed"]
    for mode, metrics in cross["descriptive_seed_statistics"].items():
        for field, value in metrics.items():
            rows.append({"section": "cross_seed_descriptive", "mode": mode, "metric": field,
                         "complete": cross["complete"], "seed_mean_mm": value["mean"],
                         "seed_sample_std_mm": value["sample_std"], "seed_count": value["seed_count"],
                         "individual_values": json.dumps(value["individual_values"], separators=(",", ":"))})
    for mode, metric in report["policy"]["table"].items():
        rows.append({"section": "policy", "mode": mode, "complete": report["policy"]["complete"], **metric})
    for key, value in report["policy"]["paired_comparisons"].items():
        interval = value.get("ci95") or [None, None]
        rows.append({"section": "policy_paired_success_difference", "comparison": key,
                     "complete": report["policy"]["complete"], "success_rate_difference": value.get("mean_difference"),
                     "difference_ci95_low": interval[0], "difference_ci95_high": interval[1],
                     "paired_cases": value.get("group_count")})
    for mode, metric in report["dev_selected_policy"]["table"].items():
        rows.append({"section": "posthoc_dev_selected_policy", "mode": mode,
                     "complete": report["dev_selected_policy"]["complete"], **metric})
    return rows


def make_plots(report, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"created": [], "reason": str(exc)}
    created = []
    primary = report["representation"]["primary"]
    colors = {"state": "#667085", "visual": "#2673c9", "tactile": "#e58b2a", "late": "#14927e",
              "internal": "#7047c5", "no_xyz": "#cf4267", "mismatch": "#986d3c"}
    if primary["histories"]:
        fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
        for mode in PRIMARY_MODES:
            history = primary["histories"].get(mode, [])
            pairs = [(row["step"], row["dev_episode_mean_mm"]) for row in history
                     if row.get("dev_episode_mean_mm") is not None and math.isfinite(row["dev_episode_mean_mm"])]
            if pairs:
                ax.plot(*zip(*pairs), label=mode, color=colors[mode], linewidth=1.6)
        ax.set(xlabel="Training update", ylabel="Dev episode-mean center error (mm)",
               title="Primary seed 43: declared ablations" + (" [incomplete]" if not primary["complete"] else ""))
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
        target = output / "dev_learning_curves.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        created.append(fingerprint(target))
    available = [(mode, primary["table"][mode]) for mode in KEY_MODES if mode in primary["table"]]
    if available:
        fig, ax = plt.subplots(figsize=(6.4, 4.7))
        fig.subplots_adjust(left=.14, right=.97, top=.86, bottom=.19)
        for number, (mode, metrics) in enumerate(available):
            value = metrics["episode_mean_mm"]
            low, high = metrics["episode_ci95_low_mm"], metrics["episode_ci95_high_mm"]
            ax.bar(number, value, color=colors[mode], width=.62)
            if low is not None and high is not None:
                ax.errorbar(number, value, yerr=[[max(0, value-low)], [max(0, high-value)]], color="#202b3a", capsize=5, fmt="none")
            ax.text(number, high + 1 if high is not None else value + 1, f"{value:.1f}", ha="center", fontsize=10)
        ax.set(xticks=range(len(available)), xticklabels=[mode for mode, _ in available],
               ylabel="Test episode-mean center error (mm)",
               title="Seed 43: 95% episode-bootstrap intervals" + (" [incomplete]" if not primary["complete"] else ""))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.text(.5, .035, "115 snapshots / 8 held-out episodes; intervals are not seed variability", ha="center", fontsize=8)
        target = output / "primary_key_comparison.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        created.append(fingerprint(target))
    return {"created": created, "scope": "Learning curves use dev only; bars use primary-seed test episode-bootstrap intervals."}


def summarize(args):
    sources = {}
    runs = [representation_run(args.input / name, seed, modes, sources) for seed, name, modes in RUNS]
    policy = policy_report(args.input / "closed_loop36", sources)
    quality = source_json(args.data_artifacts / "quality_summary.json", sources)
    data_manifest = source_json(args.data_artifacts / "manifest.json", sources)
    alignment = source_json(args.data_artifacts / "alignment_audit.json", sources)
    replay = source_json(args.input / "command_replay_reference/summary.json", sources)
    secondary = dev_selected_report(args.input / "dev_selected36", sources)
    cross = cross_seed_report(runs)
    report = {"report_version": 1, "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "complete": runs[0]["complete"] and cross["complete"] and policy["complete"]
                          and (not secondary["expected"] or secondary["complete"]),
              "representation": {"primary": runs[0], "seed_runs": runs, "cross_seed": cross,
                  "primary_evidence": primary_evidence_summary(runs[0]),
                  "metric_scope": "Object-center relative-to-palm regression in millimetres; not complete shape reconstruction.",
                  "late_scope": "Late tactile adapter prototype on VGGT features; not the previous point-cloud MLP policy baseline."},
              "policy": policy, "dev_selected_policy": secondary, "data_quality": quality,
              "command_replay_reference": None if replay is None else {
                  key: replay.get(key) for key in
                  ("complete", "protocol", "scope", "nominal_strict_stable_goal", "grid", "elapsed_s")},
              "data_manifest_summary": None if data_manifest is None else {key: data_manifest.get(key) for key in
                  ("complete", "examples", "source_groups", "split_episodes", "split_counts", "input_whitelist", "label_only", "scoring_only", "velocity_dt", "limitations")},
              "alignment_summary": None if alignment is None else {"passed": alignment.get("passed"), "frames": alignment.get("count"),
                  "maximum_force_difference": max((row["force_max_abs"] for row in alignment.get("cases", [])), default=None),
                  "maximum_anchor_difference_m": max((row["anchor_max_m"] for row in alignment.get("cases", [])), default=None)},
              "runtime_cache_parity": source_json(args.input / "runtime_cache_parity.json", sources),
              "posthoc_audits": {
                  "xyz_sensitivity": source_json(args.input / "xyz_sensitivity.json", sources),
                  "counterfactual_equal_source": source_json(args.output / "counterfactual_audit.json", sources),
                  "initial_xy_support": source_json(args.output / "initial_support_audit.json", sources),
                  "scope": "Frozen-model diagnostics after the primary run; not checkpoint selection or preregistered confirmatory comparisons."},
              "excluded_runs": {"s43": "Initial screening hyperparameters; excluded from primary and cross-seed statistics.",
                                "s43_refined": "Aborted refinement; excluded from all estimates."},
              "limitations": ["Single PickLift scenario and object mesh; no cross-task generalization test.",
                  "Counterfactual representation siblings are static snapshots, not dynamically feasible demonstrations.",
                  "Three representation training seeds share the same eight test episodes.",
                  "Closed-loop cases share one original episode; grid intervals do not establish cross-episode generalization.",
                  "ACT and representation training use distinct targets; spatial accuracy does not itself establish policy success.",
                  "Simulation pauses for inference; wall latency is not evidence of realtime robot deployment.",
                  "No claim of full geometric reconstruction improvement or research novelty is evaluated here."],
              "sources": sources, "implementation": fingerprint(__file__)}
    if args.include_screening:
        screening = representation_run(args.input / "s43", 43, PRIMARY_MODES, sources)
        screening["role"] = "screening only; excluded from primary and aggregates"
        report["screening"] = screening
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots:
        report["plots"] = make_plots(report, args.output)
    else:
        report["plots"] = {"created": [], "reason": "Not requested"}
    rows = csv_rows(report)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["section", "complete"]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"complete": report["complete"], "representation_primary_complete": runs[0]["complete"],
                      "representation_three_seed_complete": cross["complete"], "policy_complete": policy["complete"],
                      "output": str(args.output.resolve()), "csv_rows": len(rows)}, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/token_pilot_20261025")
    parser.add_argument("--data-artifacts", type=Path, default=ROOT / "artifacts/token_pilot_data_20261025")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/token_pilot_20261025")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--include-screening", action="store_true")
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
