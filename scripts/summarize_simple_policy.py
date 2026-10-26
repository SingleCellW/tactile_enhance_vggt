#!/usr/bin/env python3
"""Summarize only the manifest-declared simple-policy follow-up.

Metadata/JSON analysis only: no torch, simulator, model loading or training.
Missing runs remain incomplete with null performance values. Support13 is never
presented as a completed grid36 evaluation, and old reference runs are separate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.token_metrics import paired_group_bootstrap

MODES = ("frozen_zero", "frozen_pressure", "joint_zero", "joint_pressure")
SUBSETS = {"nominal": 1, "grid36": 36, "within_bbox12": 12, "outside24": 24}
CASE_COUNTS = {"all": 37, "support": 13, "nominal": 1, "grid36": 36}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path, sources):
    path = Path(path)
    if not path.is_file():
        return None
    sources[str(path.resolve())] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return json.loads(path.read_text())


def wilson(successes, count):
    if not count:
        return None
    z = NormalDist().inv_cdf(.975)
    p, d = successes/count, 1+z*z/count
    center = (p+z*z/(2*count))/d
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [float(max(0, center-half)), float(min(1, center+half))]


def canonical_cases():
    values = np.linspace(-.035, .035, 6)
    cases = {"nominal": {"episode_id": 1, "frame": 0, "offset": [0., 0.]}}
    for i in range(36):
        cases[f"grid_{i:02d}"] = {"episode_id": 1, "frame": 0,
                                  "offset": [float(values[i//6]), float(values[i % 6])]}
    for name, case in cases.items():
        xy = np.array([.44, -.04])+case["offset"]
        inside = bool(((xy >= np.array([.44, -.06])-1e-10) & (xy <= np.array([.56, .06])+1e-10)).all())
        case["xy"] = xy
        case["subsets"] = {"nominal": name == "nominal", "grid36": name != "nominal",
                            "within_bbox12": name != "nominal" and inside,
                            "outside24": name != "nominal" and not inside}
    return cases


def declared_ids(case_set):
    cases = canonical_cases()
    if case_set == "all":
        return set(cases)
    if case_set == "support":
        return {name for name, value in cases.items() if value["subsets"]["nominal"] or value["subsets"]["within_bbox12"]}
    if case_set in ("nominal", "grid36"):
        return {name for name, value in cases.items() if value["subsets"][case_set]}
    raise ValueError(f"Unknown declared case_set: {case_set}")


def subset_summary(records, planned_ids, subset):
    canonical = {name for name, case in canonical_cases().items() if case["subsets"][subset]}
    selected = [row for row in records if row["case"]["case_id"] in canonical]
    successes = sum(row["outcome"]["strict_stable_goal"] for row in selected)
    count = len(selected)
    expected = len(planned_ids & canonical)
    def avg(key):
        return float(np.mean([row["outcome"][key] for row in selected])) if count else None
    refresh = [row["encoder_refresh"]["mean_s"] for row in selected if row.get("encoder_refresh")]
    pressure = [row.get("pressure_diagnostics") for row in selected]
    contacts = [row.get("object_contact_scoring_only") for row in selected]
    return {"cases": count, "expected_cases": SUBSETS[subset], "selected_expected_cases": expected,
            "planned": expected > 0, "complete": {row["case"]["case_id"] for row in selected} == canonical,
            "selected_complete": {row["case"]["case_id"] for row in selected} == (planned_ids & canonical),
            "strict_successes": int(successes) if count else None,
            "strict_success_rate": float(successes/count) if count else None,
            "strict_success_wilson_ci95": wilson(successes, count),
            "lifted_count": sum(row["outcome"]["ever_lifted"] for row in selected) if count else None,
            "lift_then_drop_count": sum(row["outcome"]["lift_then_drop_outside_goal"] for row in selected) if count else None,
            "mean_goal_center_distance_mm": avg("final_goal_center_xy_distance_m")*1000 if count else None,
            "mean_goal_region_distance_mm": avg("final_goal_region_xy_distance_m")*1000 if count else None,
            "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in selected])) if count else None,
            "mean_encoder_refresh_seconds": float(np.mean(refresh)) if refresh else None,
            "cases_with_native_pressure_observed": sum(value["contact_refreshes"] > 0 for value in pressure) if count and all(value is not None for value in pressure) else None,
            "cases_with_robot_object_contact_scoring_only": sum(value["first_contact_tick"] is not None for value in contacts) if count and all(value is not None for value in contacts) else None,
            "case_ids": [row["case"]["case_id"] for row in selected]}


def training_run(root, entry, sources):
    mode, directory = entry["mode"], root / entry["directory"]
    config = read(directory / "train_config.json", sources)
    status = read(directory / "status.json", sources)
    problems = []
    if config is None or status is None:
        return {"mode": mode, "directory": str(directory), "complete": False,
                "problems": ["missing_train_config_or_status"], "selected_step": None}
    for name, expected in (("mode", mode), ("seed", 43), ("steps", 6000), ("eval_every", 1000)):
        if config.get(name) != expected:
            problems.append(f"unexpected_{name}")
    if config.get("policy", {}).get("encoder_dim") != 300:
        problems.append("ACT_input_is_not_300")
    if config.get("validation_scope") != "all original held-out observations":
        problems.append("validation_is_not_the_complete_original_heldout_split")
    batch_size, val_count = config.get("batch_size", 0), config.get("val_observations", 0)
    if not batch_size or not val_count or config.get("eval_batches", 0)*batch_size < val_count:
        problems.append("validation_batch_budget_does_not_cover_heldout_observations")
    expected_grad = 280960 if mode.startswith("joint_") else 0
    if config.get("readout_trainable_parameters") != expected_grad:
        problems.append("readout_gradient_parameter_count_mismatch")
    feature = config.get("feature_manifest", {})
    if not feature.get("complete") or feature.get("feature_dim") != 1088:
        problems.append("invalid_readout_input_manifest")
    for a, b in ((config.get("visual_probe_checkpoint_sha256"), feature.get("source_checkpoint_sha256")),
                 (config.get("data_manifest_sha256"), feature.get("cache_manifest_sha256")),
                 (config.get("readout_inputs_sha256"), feature.get("readout_inputs_sha256"))):
        if not a or a != b:
            problems.append("feature_initialization_or_dataset_binding_mismatch")
    parity = feature.get("frozen_projection_parity")
    if parity is None or not np.isfinite(parity.get("relative_l2", np.inf)) or parity["relative_l2"] > .005:
        problems.append("missing_or_failed_frozen_projection_parity")
    normalization = read(directory / "normalization.json", sources)
    if normalization is None or not np.isfinite(normalization.get("simple_pressure_scale", np.nan)) or normalization["simple_pressure_scale"] <= 0:
        problems.append("invalid_pressure_normalization")
    metrics_path = directory / "metrics.jsonl"
    validation = []
    if metrics_path.is_file():
        sources[str(metrics_path.resolve())] = {"sha256": digest(metrics_path), "bytes": metrics_path.stat().st_size}
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("validation_performed") and row.get("val_normalized_action_l1") is not None:
                validation.append({"step": int(row["step"]), "value": float(row["val_normalized_action_l1"])})
    if not validation or any(not np.isfinite(row["value"]) for row in validation):
        problems.append("missing_or_nonfinite_actual_validation_records")
        selected = None
    else:
        selected = min(validation, key=lambda row: (row["value"], row["step"]))
        if len(validation) != 6 or {row["step"] for row in validation} != set(range(1000, 6001, 1000)):
            problems.append("validation_does_not_cover_declared_budget")
        marker = read(directory / "checkpoints" / f"{selected['step']:06d}" / "checkpoint_complete.json", sources)
        if marker is None or marker.get("step") != selected["step"]:
            problems.append("selected_checkpoint_incomplete")
    if status.get("status") != "completed" or status.get("step") != 6000:
        problems.append("training_budget_not_complete")
    signature_keys = ("seed", "steps", "batch_size", "learning_rate", "weight_decay", "grad_clip", "amp", "eval_every", "eval_batches",
                      "validation_scope", "train_observations", "val_observations", "sampler_version", "action_semantics",
                      "data_manifest_sha256", "feature_manifest_sha256", "readout_inputs_sha256", "visual_probe_checkpoint_sha256", "policy")
    return {"mode": mode, "directory": str(directory), "complete": not problems, "problems": problems,
            "selected_step": selected["step"] if selected else None,
            "selection_value": selected["value"] if selected else None,
            "selection_rule": "minimum actually-performed full-validation action L1; earliest tie; never rollout outcomes",
            "training_signature": {key: config.get(key) for key in signature_keys},
            "trainable_parameters": config.get("trainable_parameters"), "total_parameters": config.get("total_parameters"),
            "readout_trainable_parameters": config.get("readout_trainable_parameters"),
            "frozen_projection_parity": parity,
            "normalization": normalization, "feature_input_scope": feature.get("input_scope"),
            "train_config_sha256": digest(directory / "train_config.json")}


def evaluate_entry(root, entry, training, sources):
    mode, directory = entry["mode"], root / entry["directory"]
    planned = declared_ids(entry["case_set"])
    summary = read(directory / "summary.json", sources)
    config = read(directory / "eval_config.json", sources)
    plan = read(directory / "case_plan.json", sources)
    result = {**entry, "directory": str(directory), "complete": False, "problems": [],
              "declared_cases": len(planned), "observed_cases": 0, "subsets": {}, "primary": None,
              "policy_sha256": None, "case_records": []}
    if summary is None or config is None or plan is None:
        result["problems"].append("missing_summary_eval_config_or_case_plan")
        return result
    if summary.get("protocol") != "simple_token_act_37_v1" or config.get("protocol") != summary.get("protocol"):
        result["problems"].append("unexpected_evaluation_protocol")
    for key in ("kind", "case_set"):
        if config.get(key) != entry[key] or summary.get(key) != entry[key]:
            result["problems"].append(f"{key}_differs_from_manifest")
    actual_steps = summary.get("n_action_steps_override") or summary.get("execution_horizon", {}).get("n_action_steps")
    if actual_steps != entry["n_action_steps"]:
        result["problems"].append("feedback_horizon_differs_from_manifest")
    if summary.get("config_sha256") != digest(directory / "eval_config.json") or summary.get("case_plan_sha256") != digest(directory / "case_plan.json"):
        result["problems"].append("evaluation_config_or_plan_sha_mismatch")
    if set(plan.get("declared_case_ids", [])) != planned or set(plan.get("selected_case_ids", [])) != planned:
        result["problems"].append("declared_or_selected_case_ids_differ_from_full_planned_case_set")
    model = summary.get("models", {}).get(mode)
    if model is None:
        result["problems"].append("requested_mode_missing_from_summary")
        return result
    rows = model.get("results", [])
    result["observed_cases"] = len(rows)
    known = canonical_cases()
    actual = [row.get("case", {}).get("case_id") for row in rows]
    if len(set(actual)) != len(actual) or set(actual) != planned:
        result["problems"].append("missing_extra_or_duplicate_case_ids")
    valid_rows = []
    for row in rows:
        name, case = row.get("case", {}).get("case_id"), row.get("case", {})
        if name not in known:
            result["problems"].append("unknown_case_id")
            continue
        reference = known[name]
        if row.get("config_sha256") != summary.get("config_sha256") or row.get("mode") != mode:
            result["problems"].append(f"case_configuration_or_mode_mismatch:{name}")
        if (case.get("episode_id") != 1 or case.get("frame") != 0 or
                not np.allclose(case.get("object_xy_offset", [np.inf]*2), reference["offset"], rtol=0, atol=1e-10) or
                case.get("subset_flags") != reference["subsets"]):
            result["problems"].append(f"incorrect_case_or_subset:{name}")
        initial = np.asarray(row.get("initial_info", {}).get("cube_center", [np.inf]*3))
        if initial.shape != (3,) or not np.isfinite(initial).all() or not np.allclose(initial[:2], reference["xy"], rtol=0, atol=1e-7):
            result["problems"].append(f"initial_GT_scoring_position_mismatch:{name}")
        if not row.get("complete") or row.get("ticks") != 600 or not np.isclose(row.get("sim_time_s", -1), 6., rtol=0, atol=1e-8):
            result["problems"].append(f"incomplete_physical_horizon:{name}")
        outcome = row.get("outcome", {})
        needed = ("strict_stable_goal", "ever_lifted", "lift_then_drop_outside_goal",
                  "final_goal_center_xy_distance_m", "final_goal_region_xy_distance_m")
        if any(key not in outcome for key in needed) or not all(np.isfinite(outcome.get(key, np.nan)) for key in needed[-2:]):
            result["problems"].append(f"invalid_outcome:{name}")
            continue
        if bool(outcome["strict_stable_goal"]) != bool(row.get("final_info", {}).get("strict_stable_goal")):
            result["problems"].append(f"success_disagrees_with_worker:{name}")
        valid_rows.append(row)
        path = directory / mode / name / "result.json"
        if path.is_file() and read(path, sources) != row:
            result["problems"].append(f"case_file_disagrees_with_summary:{name}")
    result["case_records"] = valid_rows
    result["subsets"] = {name: subset_summary(valid_rows, planned, name) for name in SUBSETS}
    primary = "nominal" if entry["case_set"] == "nominal" else "within_bbox12" if entry["case_set"] == "support" else "grid36"
    result["primary_subset"], result["primary"] = primary, result["subsets"][primary]
    for name, value in result["subsets"].items():
        old = model.get("subsets", {}).get(name, {})
        for a, b in (("cases", "cases"), ("strict_successes", "strict_stable_goal_count"), ("selected_expected_cases", "selected_expected_cases")):
            # Empty, unplanned groups contain zero observed successes in raw
            # summaries; the presentation uses null performance instead.
            expected = 0 if value[a] is None and a == "strict_successes" else value[a]
            if old.get(b) != expected:
                result["problems"].append(f"subset_summary_mismatch:{name}:{b}")
    provenance = config.get("modes", {}).get(mode, {})
    result["policy_sha256"] = provenance.get("policy_sha256")
    result["policy_checkpoint"] = provenance.get("policy")
    if entry["kind"] == "simple" and mode in training and training[mode].get("complete"):
        if provenance.get("train_config_sha256") != training[mode]["train_config_sha256"]:
            result["problems"].append("policy_training_provenance_mismatch")
        selected = training[mode]["selected_step"]
        if Path(provenance.get("policy", "")).name != f"{selected:06d}":
            result["problems"].append("evaluated_checkpoint_is_not_validation_selected")
    if not summary.get("complete") or not summary.get("selected_cases_complete") or not model.get("complete"):
        result["problems"].append("evaluation_declared_incomplete")
    result["full_grid_complete"] = bool(summary.get("full_grid_complete"))
    result["full37_complete"] = bool(summary.get("full37_complete"))
    if result["full_grid_complete"] != result["subsets"]["grid36"]["complete"] or result["full37_complete"] != (len(valid_rows) == 37):
        result["problems"].append("canonical_completion_flag_mismatch")
    result["problems"] = sorted(set(result["problems"]))
    result["complete"] = not result["problems"]
    return result


def compare(first, second, subset, label):
    one = {row["case"]["case_id"]: row for row in first["case_records"] if row["case"]["subset_flags"][subset]}
    two = {row["case"]["case_id"]: row for row in second["case_records"] if row["case"]["subset_flags"][subset]}
    expected = {name for name, case in canonical_cases().items() if case["subsets"][subset]}
    complete = first["complete"] and second["complete"] and set(one) == set(two) == expected
    result = {"comparison": label, "subset": subset, "complete": complete,
              "paired_cases": len(set(one) & set(two)), "expected_paired_cases": len(expected),
              "success_rate_difference": None, "ci95": None,
              "scope": "paired fixed resets from one source episode/ACT seed; not independent episode generalization"}
    if complete:
        ids = sorted(expected)
        if any(one[name]["case"] != two[name]["case"] for name in ids):
            result.update(complete=False, problem="physical_case_mismatch")
            return result
        diff = paired_group_bootstrap([float(one[name]["outcome"]["strict_stable_goal"]) for name in ids],
                                      [float(two[name]["outcome"]["strict_stable_goal"]) for name in ids], ids)
        result.update(success_rate_difference=diff["mean_difference"], ci95=diff["ci95"],
                      direction="positive means first condition has higher strict success rate",
                      bootstrap_unit="paired fixed XY reset case, not source episode",
                      insufficient_cases_for_case_resampling=len(ids) < 2)
    return result


def interaction(indexed, subset):
    """Difference of pressure effects, preserving one paired reset per row."""
    expected = {name for name, case in canonical_cases().items() if case["subsets"][subset]}
    rows = {}
    for mode in MODES:
        value = indexed.get((mode, 8))
        if value is None or not value["complete"]:
            return {"complete": False, "subset": subset, "difference_of_success_rate_differences": None, "ci95": None}
        rows[mode] = {row["case"]["case_id"]: row for row in value["case_records"] if row["case"]["subset_flags"][subset]}
    if any(set(value) != expected for value in rows.values()):
        return {"complete": False, "subset": subset, "difference_of_success_rate_differences": None, "ci95": None}
    ids = sorted(expected)
    if any(rows[mode][name]["case"] != rows[MODES[0]][name]["case"] for mode in MODES for name in ids):
        return {"complete": False, "subset": subset, "problem": "physical_case_mismatch",
                "difference_of_success_rate_differences": None, "ci95": None}
    success = {mode: np.asarray([float(rows[mode][name]["outcome"]["strict_stable_goal"]) for name in ids]) for mode in MODES}
    effect = (success["joint_pressure"] - success["joint_zero"]) - (success["frozen_pressure"] - success["frozen_zero"])
    diff = paired_group_bootstrap(effect, np.zeros(len(effect)), ids)
    return {"complete": True, "subset": subset, "paired_cases": len(ids),
            "difference_of_success_rate_differences": diff["mean_difference"], "ci95": diff["ci95"],
            "direction": "positive means the pressure-minus-zero effect is larger with joint readout training",
            "bootstrap_unit": "paired fixed XY reset case, not source episode",
            "insufficient_cases_for_case_resampling": len(ids) < 2,
            "scope": "secondary fixed-case sensitivity; one source episode and one ACT seed"}


def summarize(args):
    root, sources, problems = args.input.resolve(), {}, []
    manifest = read(root / "experiment_manifest.json", sources)
    if manifest is None:
        manifest = {"training_runs": [], "expected_evaluations": []}
        problems.append("missing_experiment_manifest")
    training = {entry["mode"]: training_run(root, entry, sources) for entry in manifest.get("training_runs", [])}
    if set(training) != set(MODES):
        problems.append("manifest_does_not_declare_all_four_training_conditions")
    comparable = False
    if len(training) == 4 and all(value.get("training_signature") for value in training.values()):
        signatures = [value["training_signature"] for value in training.values()]
        normalizations = [value["normalization"] for value in training.values()]
        totals = [value["total_parameters"] for value in training.values()]
        comparable = all(value == signatures[0] for value in signatures) and all(value == normalizations[0] for value in normalizations) and len(set(totals)) == 1
        base_count = training["frozen_zero"].get("trainable_parameters")
        comparable = comparable and base_count is not None and all(
            value.get("trainable_parameters") == base_count + (280960 if mode.startswith("joint_") else 0)
            for mode, value in training.items())
        if not comparable:
            problems.append("training_data_hyperparameters_initialization_or_parameter_layout_not_matched")
    evaluations = [evaluate_entry(root, entry, training, sources) for entry in manifest.get("expected_evaluations", [])]
    indexed = {(value["mode"], value["n_action_steps"]): value for value in evaluations if value["kind"] == "simple"}
    primary, auxiliary, feedback, projection_effect = {}, {}, {}, {}
    for projection in ("frozen", "joint"):
        a, b = indexed.get((projection+"_pressure", 8)), indexed.get((projection+"_zero", 8))
        if a is not None and b is not None:
            key = projection+"_pressure_minus_"+projection+"_zero"
            primary[key] = compare(a, b, "grid36", key)
            auxiliary[key] = {subset: compare(a, b, subset, key) for subset in ("nominal", "within_bbox12", "outside24")}
    for tactile in ("zero", "pressure"):
        a, b = indexed.get(("joint_"+tactile, 8)), indexed.get(("frozen_"+tactile, 8))
        if a is not None and b is not None:
            key = "joint_"+tactile+"_minus_frozen_"+tactile
            projection_effect[key] = {subset: compare(a, b, subset, key) for subset in SUBSETS}
    for mode in MODES:
        a, b = indexed.get((mode, 2)), indexed.get((mode, 8))
        if a is not None and b is not None:
            feedback[mode] = {subset: compare(a, b, subset, f"{mode}:n2_minus_n8") for subset in ("within_bbox12", "nominal")}
            same = bool(a.get("policy_sha256")) and a.get("policy_sha256") == b.get("policy_sha256")
            for value in feedback[mode].values():
                value["same_checkpoint_verified"] = same
                if not same:
                    value.update(complete=False, success_rate_difference=None, ci95=None)
    bbox_path = ROOT / "reports/token_pilot_20261025/initial_support_audit.json"
    bbox = read(bbox_path, sources)
    coverage = {"source": str(bbox_path), "scope": "initial XY bbox only; not full ID",
                "nominal_separate": True, "grid_within": 12, "grid_outside": 24}
    if bbox is not None:
        initial = {row["episode"]: row["cube_xyz"] for row in bbox["initial_states"]}
        xy = np.asarray([initial[ep][:2] for ep in bbox["train_episodes"]])
        coverage.update(train_episodes=len(xy), train_xy_min=xy.min(0).tolist(), train_xy_max=xy.max(0).tolist(),
                        source_nominal_xyz=initial[1], source_cache_manifest_sha256=bbox["source_cache_manifest_sha256"])
        if len(xy) != 48 or not np.allclose(xy.min(0), [.44, -.06]) or not np.allclose(xy.max(0), [.56, .06]):
            problems.append("source_initial_bbox_does_not_match_declared_protocol")
        for value in training.values():
            sig = value.get("training_signature", {})
            if sig and sig["data_manifest_sha256"] != bbox["source_cache_manifest_sha256"]:
                problems.append("training_source_differs_from_bbox_audit_source")
    else:
        problems.append("missing_initial_bbox_audit")
    if not evaluations:
        problems.append("no_declared_evaluations")
    complete = not problems and comparable and all(value["complete"] for value in training.values()) and all(value["complete"] for value in evaluations)
    result = {"report_version": 1, "complete": complete, "problems": sorted(set(problems)),
              "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "experiment_manifest": manifest, "training_comparability_verified": comparable, "training": training,
              "evaluations": evaluations, "primary_pressure_effect_n8_grid36": primary,
              "auxiliary_pressure_effect_n8": auxiliary, "same_checkpoint_feedback_n2_minus_n8": feedback,
              "secondary_projection_effect_n8": projection_effect,
              "secondary_pressure_projection_interaction_n8": {subset: interaction(indexed, subset) for subset in SUBSETS},
              "initial_xy_coverage": coverage,
              "rollouts": {"planned_new_simple": sum(CASE_COUNTS[value["case_set"]] for value in evaluations if value["kind"] == "simple"),
                           "actual_new_simple": sum(value["observed_cases"] for value in evaluations if value["kind"] == "simple"),
                           "planned_added_legacy": sum(CASE_COUNTS[value["case_set"]] for value in evaluations if value["kind"] == "legacy"),
                           "actual_added_legacy": sum(value["observed_cases"] for value in evaluations if value["kind"] == "legacy")},
              "scope": {"model": "VGGT/state encoder frozen; original center-pretrained readout frozen/joint; pressure44 concatenated; no new 3D alignment claim",
                        "inputs": "current RGB/state/pressure only; object GT is reset/scoring-only, never an actor feature",
                        "zero": "zero during both training and deployment; matched no-touch control, not test-time sensor deletion",
                        "feedback": "n2 support13 is nominal plus common within-bbox12; outside24/grid36 not evaluated at n2",
                        "uncertainty": "one source, one ACT seed; fixed-case sensitivity only; no multiplicity correction",
                        "time": "simulation pauses for inference; no realtime claim",
                        "references": "legacy nominal results remain separate; old grid/reference rollouts are never pooled with new conditions"},
              "sources": sources, "implementation_sha256": digest(__file__)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    fields = ["mode", "kind", "n_action_steps", "case_set", "evaluation_complete", "subset", "planned", "cases", "expected_cases",
              "selected_expected_cases", "complete", "selected_complete", "strict_successes", "strict_success_rate",
              "strict_success_wilson_ci95", "lifted_count", "lift_then_drop_count", "mean_goal_center_distance_mm", "mean_goal_region_distance_mm",
              "mean_wall_seconds", "mean_encoder_refresh_seconds", "cases_with_native_pressure_observed", "cases_with_robot_object_contact_scoring_only"]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for value in evaluations:
            subsets = value["subsets"] or {"missing": {"planned": True, "cases": 0, "complete": False, "selected_complete": False}}
            for subset, stats in subsets.items():
                row = {**{key: value[key] for key in ("mode", "kind", "n_action_steps", "case_set")},
                       "evaluation_complete": value["complete"], "subset": subset, **stats}
                if row.get("strict_success_wilson_ci95") is not None:
                    row["strict_success_wilson_ci95"] = json.dumps(row["strict_success_wilson_ci95"])
                writer.writerow(row)
    print(json.dumps({"complete": complete, "problems": result["problems"], "rollouts": result["rollouts"],
                      "missing_or_partial_training": [mode for mode, value in training.items() if not value["complete"]],
                      "missing_or_partial_evaluations": [value["directory"] for value in evaluations if not value["complete"]],
                      "output": str(args.output)}, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/simple_policy_followup_20261025")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/simple_policy_followup_20261025")
    summarize(parser.parse_args())
