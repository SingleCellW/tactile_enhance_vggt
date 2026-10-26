#!/usr/bin/env python3
"""Build portable Stage1.a portal data from existing, SHA-bound result files.

No training, simulator, model, network or plotting dependency is used. Values
and confidence intervals are copied from their audited sources; missing input
files produce explicit partial sections and null metrics, never zero results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = "reports/simple_policy_followup_20261025"
EXPERIMENT = "outputs/simple_policy_followup_20261025"
TOKEN = "reports/token_pilot_20261025"
MODES = ("frozen_zero", "frozen_pressure", "joint_zero", "joint_pressure")
SUBSETS = ("grid36", "within_bbox12", "outside24", "nominal")
LABELS = {"frozen_zero": "Frozen projection · RGB/state", "frozen_pressure": "Frozen projection · RGB/state + pressure",
          "joint_zero": "Joint projection · RGB/state", "joint_pressure": "Joint projection · RGB/state + pressure"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Sources:
    def __init__(self, root):
        self.root, self.items, self.missing = Path(root), {}, []

    def text(self, name, relative):
        path = self.root / relative
        if not path.is_file():
            self.items[name] = {"id": name, "repo_path": str(relative), "available": False, "sha256": None, "bytes": None}
            self.missing.append(str(relative))
            return None
        self.items[name] = {"id": name, "repo_path": str(relative), "available": True,
                            "sha256": digest(path), "bytes": path.stat().st_size}
        return path.read_text()

    def json(self, name, relative):
        value = self.text(name, relative)
        return json.loads(value) if value is not None else {}


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def metric(value):
    """A single metric vocabulary for current results and separate references."""
    if not value:
        return {"complete": False, "planned": None, "cases": None, "expected_cases": None,
                "successes": None, "rate": None, "ci95": None, "lifted": None, "dropped": None}
    ci = value.get("strict_success_wilson_ci95", value.get("strict_stable_goal_wilson_ci95", value.get("success_wilson_ci95")))
    if ci is None and "success_wilson_low" in value:
        ci = [value["success_wilson_low"], value["success_wilson_high"]]
    return {"complete": value.get("selected_complete", value.get("complete", True)),
            "planned": value.get("planned", True), "cases": value.get("cases"),
            "expected_cases": value.get("selected_expected_cases", value.get("expected_cases", value.get("cases"))),
            "successes": value.get("strict_successes", value.get("strict_stable_goal_count")),
            "rate": value.get("strict_success_rate", value.get("strict_stable_goal_rate")), "ci95": ci,
            "lifted": value.get("lifted_count"), "dropped": value.get("lift_then_drop_count"),
            **pick(value, ("mean_goal_center_distance_mm", "mean_goal_region_distance_mm", "mean_wall_seconds",
                "mean_encoder_refresh_seconds", "cases_with_native_pressure_observed", "cases_with_robot_object_contact_scoring_only"))}


def compact_case(row):
    return {"case": pick(row.get("case", {}), ("case_id", "episode_id", "frame", "seed", "object_xy_offset", "initial_xy", "subset_flags")),
            "outcome": row.get("outcome", {}),
            "object_contact_scoring_only": pick(row.get("object_contact_scoring_only", {}),
                ("first_contact_tick", "post_action_contact_ticks", "maximum_post_action_contact_count")),
            "pressure_diagnostics": row.get("pressure_diagnostics"), "ticks": row.get("ticks")}


def portable_paths(value, root):
    """Host filesystem paths are provenance text, never public URLs."""
    if isinstance(value, dict):
        return {key: portable_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item, root) for item in value]
    if isinstance(value, str):
        for prefix in (str(Path(root).resolve()) + "/", "/home/wukf/projects/tactile_enhance_vggt/"):
            if value.startswith(prefix):
                return value[len(prefix):]
        if value.startswith(("/home/", "/tmp/")):
            return "external-source/" + Path(value).name
    return value


def selection_evidence(source, mode, report):
    directory = f"{EXPERIMENT}/train/{mode}"
    config = source.json(f"{mode}_train_config", f"{directory}/train_config.json")
    status = source.json(f"{mode}_train_status", f"{directory}/status.json")
    text = source.text(f"{mode}_training_metrics", f"{directory}/metrics.jsonl")
    validation = []
    for line in text.splitlines() if text else []:
        row = json.loads(line)
        value = row.get("val_normalized_action_l1")
        if row.get("validation_performed") and value is not None and math.isfinite(value):
            validation.append({"step": row["step"], "normalized_action_l1": value})
    selected = min(validation, key=lambda row: (row["normalized_action_l1"], row["step"])) if validation else None
    step = report.get("selected_step")
    marker = source.json(f"{mode}_selected_checkpoint", f"{directory}/checkpoints/{step:06d}/checkpoint_complete.json") if isinstance(step, int) else {}
    expected = list(range(config.get("eval_every", 1000), config.get("steps", 0) + 1, config.get("eval_every", 1000)))
    verified = bool(selected and report.get("complete") and selected["step"] == step
                    and math.isclose(selected["normalized_action_l1"], report.get("selection_value", math.inf), abs_tol=1e-12)
                    and marker.get("step") == step and status.get("status") == "completed"
                    and status.get("step") == config.get("steps")
                    and [row["step"] for row in validation] == expected
                    and config.get("validation_scope") == "all original held-out observations")
    ids = [f"{mode}_{suffix}" for suffix in ("train_config", "train_status", "training_metrics", "selected_checkpoint")
           if f"{mode}_{suffix}" in source.items]
    return {"complete": verified, "selected_step": step, "selection_value": report.get("selection_value"),
            "training_budget_steps": config.get("steps"), "seed": config.get("seed"),
            "validation_every": config.get("eval_every"), "validation_observations": config.get("val_observations"),
            "training_observations": config.get("train_observations"), "validation_history": validation,
            "rule": report.get("selection_rule"), "uses_rollout_outcomes": False,
            "learning_rate": config.get("learning_rate"), "batch_size": config.get("batch_size"),
            "trainable_parameters": report.get("trainable_parameters"),
            "readout_trainable_parameters": report.get("readout_trainable_parameters"),
            "environment_dim": config.get("policy", {}).get("encoder_dim"),
            "frozen_projection_parity": report.get("frozen_projection_parity"), "source_ids": ids}


def build(root=ROOT):
    source = Sources(root)
    simple = source.json("simple_results", f"{SIMPLE}/results.json")
    failure = source.json("failure_stages", f"{SIMPLE}/failure_stages.json")
    completion = source.json("completion", f"{SIMPLE}/completion.json")
    control = source.json("control_target_audit", f"{SIMPLE}/control_target_audit.json")
    endpoints = source.json("source_endpoint_audit", f"{SIMPLE}/source_endpoint_audit.json")
    token = source.json("token_pilot_results", f"{TOKEN}/results.json")
    sanity = source.json("train_sanity_summary", f"{EXPERIMENT}/train_sanity/summary.json")
    sanity_plan = source.json("train_sanity_manifest", f"{EXPERIMENT}/train_sanity/manifest.json")
    plan = source.json("experiment_manifest", f"{EXPERIMENT}/experiment_manifest.json")
    main_summary = source.json("main_evaluation_summary", f"{EXPERIMENT}/eval_n8/frozen_zero/summary.json")
    problems, policy_rows = [], []
    evaluations = {(row.get("mode"), row.get("n_action_steps")): row for row in simple.get("evaluations", []) if row.get("kind") == "simple"}
    for mode in MODES:
        projection, tactile = mode.split("_", 1)
        selection = selection_evidence(source, mode, simple.get("training", {}).get(mode, {}))
        if not selection["complete"]:
            problems.append(f"selection_evidence_incomplete:{mode}")
        row = {"mode": mode, "label": LABELS[mode], "projection": projection, "tactile": tactile,
               "selection": selection, "source_ids": ["simple_results"], "n8": {}, "n2": {"not_evaluated": ["grid36", "outside24"]}}
        for steps, subsets in ((8, SUBSETS), (2, ("within_bbox12", "nominal"))):
            evaluation = evaluations.get((mode, steps), {})
            for subset in subsets:
                row[f"n{steps}"][subset] = metric(evaluation.get("subsets", {}).get(subset))
            row[f"n{steps}"]["complete"] = evaluation.get("complete", False)
            row[f"n{steps}"]["cases"] = [compact_case(case) for case in evaluation.get("case_records", [])]
        policy_rows.append(row)
    failure_rows = []
    stage_by_case = {}
    for entry in failure.get("evaluations", []):
        if entry.get("kind") != "simple":
            continue
        for subset, value in entry.get("subsets", {}).items():
            counts = value.get("stage_counts", {})
            if value.get("completed_cases") and all(count is not None for count in counts.values()) and sum(counts.values()) != value["completed_cases"]:
                problems.append(f"failure_stage_total_mismatch:{entry['mode']}:{entry['n_action_steps']}:{subset}")
        failure_rows.append({"mode": entry["mode"], "n_action_steps": entry["n_action_steps"],
            "complete": entry.get("complete", False), "subsets": entry.get("subsets", {}),
            "case_stages": [pick(row, ("case_id", "status", "stage", "first_lift_tick", "first_drop_tick", "ever_inside_goal",
                "entered_goal_without_final_stable_success", "cumulative_clipped_count")) for row in entry.get("cases", [])]})
        for row in entry.get("cases", []):
            stage_by_case[(entry["mode"], entry["n_action_steps"], row["case_id"])] = row
    flat_cases = []
    for (mode, steps), evaluation in evaluations.items():
        for row in evaluation.get("case_records", []):
            case, outcome = row.get("case", {}), row.get("outcome", {})
            stage = stage_by_case.get((mode, steps, case.get("case_id")), {})
            result_path = f"{EXPERIMENT}/eval_n{steps}/{mode}/{mode}/{case.get('case_id')}/result.json"
            path = Path(root) / result_path
            existing_sha = next((v.get("sha256") for k, v in simple.get("sources", {}).items() if k.endswith("/" + result_path)), None)
            flat_cases.append({"mode": mode, "n_action_steps": steps, "case_id": case.get("case_id"),
                "episode_id": case.get("episode_id"), "frame": case.get("frame"), "seed": case.get("seed"),
                "offset_xy": case.get("object_xy_offset"), "initial_xyz": row.get("initial_info", {}).get("cube_center"),
                "final_xyz": row.get("final_info", {}).get("cube_center"), "subsets": case.get("subset_flags"),
                "strict_success": outcome.get("strict_stable_goal"), "ever_lifted": outcome.get("ever_lifted"),
                "drop": outcome.get("lift_then_drop_outside_goal"), "first_drop_tick": outcome.get("first_drop_tick"),
                "final_goal_distance_m": outcome.get("final_goal_center_xy_distance_m"),
                "final_goal_region_distance_m": outcome.get("final_goal_region_xy_distance_m"),
                "stage": stage.get("stage"), "first_lift_tick": stage.get("first_lift_tick"),
                "pressure_contact_refreshes": (row.get("pressure_diagnostics") or {}).get("contact_refreshes"),
                "score_contact_first_tick": (row.get("object_contact_scoring_only") or {}).get("first_contact_tick"),
                "cumulative_clipped_count": row.get("final_info", {}).get("cumulative_clipped_count"),
                "ticks": row.get("ticks"), "result_repo_path": result_path,
                "result_sha256": digest(path) if path.is_file() else existing_sha,
                "source_report_sha256": source.items["simple_results"]["sha256"],
                "stage_audit_sha256": source.items["failure_stages"]["sha256"],
                "source_ids": ["simple_results", "failure_stages"]})
    if simple.get("complete") and completion.get("new_policy_main_rollouts") != len(flat_cases):
        problems.append("new_policy_case_count_differs_from_completion_audit")
    if len({(row["mode"], row["n_action_steps"], row["case_id"]) for row in flat_cases}) != len(flat_cases):
        problems.append("duplicate_new_policy_case_key")
    sanity_binding = bool(sanity and sanity_plan and sanity.get("manifest_sha256") == source.items["train_sanity_manifest"]["sha256"])
    sanity_models = [{"mode": mode, **metric(value),
        "ever_object_contact_count": value.get("ever_object_contact_count"),
        "cases_detail": [compact_case(row) for row in value.get("results", [])]}
        for mode, value in sanity.get("models", {}).items()]
    representation = token.get("representation", {})
    primary = representation.get("primary", {})
    seed_rows = [{"model_seed": run.get("seed"), "complete": run.get("complete", False),
        "modes": [{"mode": mode, **values} for mode, values in run.get("table", {}).items()],
        "paired_comparisons": run.get("paired_comparisons", {})} for run in representation.get("seed_runs", [])]
    # Model seed comes from each run. Nested bootstrap.seed is its resampling
    # RNG seed and must never be displayed as the model's training seed.
    old_fixed, old_selected = token.get("policy", {}), token.get("dev_selected_policy", {})
    old_rows = lambda section: [{"mode": mode, **metric(values),
        **pick(values, ("checkpoint_steps", "selected_step", "selection_value", "training_budget_steps"))}
        for mode, values in section.get("table", {}).items()]
    diagnostic_rows = []
    for row in control.get("nominal_precontact_window", {}).get("comparisons", []):
        diagnostic_rows.append(pick(row, ("mode", "first_exact_object_contact_tick", "aggregate", "per_dimension", "observed_q",
                                         "pressure_precontact_refresh_max", "final_cumulative_clipped_count")))
    result = {"schema_version": 1, "stage_id": "stage1a", "title": "Stage 1.a · Frozen VGGT and tactile policy pilot",
        "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "relative_data_url": "/data/stage1a.json",
        "downloads": [{"id": "metrics_json", "label": "指标 JSON", "url": "/data/stage1a-metrics.json", "mime": "application/json"},
            {"id": "metrics_csv", "label": "指标 CSV", "url": "/data/stage1a-metrics.csv", "mime": "text/csv"},
            {"id": "cases_csv", "label": "200条主策略案例 CSV", "url": "/data/stage1a-cases.csv", "mime": "text/csv"}],
        "overview": {"complete": bool(simple.get("complete") and completion.get("complete") and not problems),
            "main_rollouts": completion.get("main_rollouts"), "new_policy_rollouts": completion.get("new_policy_main_rollouts"),
            "legacy_nominal_rollouts": completion.get("added_legacy_main_rollouts"),
            "separate_train_diagnostics": completion.get("separate_actor_train_diagnostics"),
            "current_total_rollouts": (completion["main_rollouts"] + completion["separate_actor_train_diagnostics"])
                if all(isinstance(completion.get(key), int) for key in ("main_rollouts", "separate_actor_train_diagnostics")) else None,
            "historical_policy_reference_rollouts": sum(value.get("cases", 0) for section in (old_fixed, old_selected) for value in section.get("table", {}).values()) if old_fixed and old_selected else None,
            "static_probe_examples": token.get("data_manifest_summary", {}).get("examples"),
            "trajectory_files_checked": completion.get("trajectory_files_checked"),
            "missing_or_invalid_trajectories": completion.get("missing_or_invalid_trajectories"),
            "tests_passed_in_completion_audit": completion.get("tests_passed"),
            "training_comparability_verified": simple.get("training_comparability_verified", False),
            "main_results_complete": simple.get("complete", False), "source_ids": ["simple_results", "completion"]},
        "protocol": {"manifest_protocol": plan.get("protocol"), "seed": plan.get("seed"),
            "training_steps": plan.get("training_steps"), "validation_every": plan.get("validation_every"),
            "source_split_roles": main_summary.get("source_split_roles"), "actor_training_seeds": sorted({r["selection"]["seed"] for r in policy_rows if r["selection"]["seed"] is not None}),
            "main_source_episodes": [1] if main_summary else None, "control_ticks": 600, "physics_dt_seconds": .01,
            "n8_cases_per_mode": 37, "n2_cases_per_mode": 13, "n2_scope": "nominal plus the common within-bbox12; outside24 not evaluated",
            "input_dimensions": {"visual_readout_input": 1088, "visual_projection": 256, "pressure": 44, "environment": 300, "state": 22, "action": 22},
            "frozen_components": ["VGGT", "source visual probe state encoder"],
            "joint_trainable_components": ["existing visual readout", "ACT"],
            "zero_mode": "pressure44 is zero during both training and deployment; identical ACT input dimensions",
            "goal": "cube center in inherited goal region, low speed, no robot contact, stable for1s at the final horizon",
            "confidence_intervals": {"success_rates": "Wilson95%", "condition_differences": "paired fixed-case bootstrap95%; same case IDs",
                "interpretation": "one source episode and one actor training seed; fixed-case sensitivity, not independent episode generalization"},
            "initial_xy_coverage": simple.get("initial_xy_coverage"), "source_ids": ["experiment_manifest", "main_evaluation_summary", "simple_results"]},
        "methods": [{"id": r["mode"], "label": r["label"], "projection": r["projection"], "tactile": r["tactile"],
            "environment_dim": r["selection"]["environment_dim"], "readout_trainable_parameters": r["selection"]["readout_trainable_parameters"]} for r in policy_rows],
        "policy_results": {"complete": simple.get("complete", False), "rows": policy_rows, "cases": flat_cases,
            "cases_scope": "200 new-policy cases only; initial/final object coordinates and collision times are scoring-only data, never actor inputs",
            "rate_unit": "fraction, multiply by100 for percent; never pool nominal with grid36",
            "comparisons": {"pressure_n8_grid36": simple.get("primary_pressure_effect_n8_grid36", {}),
                "pressure_n8_auxiliary": simple.get("auxiliary_pressure_effect_n8", {}),
                "projection_n8": simple.get("secondary_projection_effect_n8", {}),
                "interaction_n8": simple.get("secondary_pressure_projection_interaction_n8", {}),
                "feedback_n2_minus_n8": simple.get("same_checkpoint_feedback_n2_minus_n8", {})},
            "source_ids": ["simple_results"]},
        "failure_analysis": {"complete": failure.get("complete", False), "contact_attribution_complete": failure.get("contact_attribution_complete", False),
            "stage_order": failure.get("stage_order", []), "stage_labels": {"success": "成功", "no_object_contact": "未记录物体接触", "contact_without_lift": "接触后未抬升",
                "lift_then_drop_outside_goal": "抬升后目标外掉落", "lift_without_drop_not_stable": "抬升未掉落但未稳定成功", "unknown": "证据缺失"},
            "rows": failure_rows, "definitions": failure.get("definitions", {}), "source_ids": ["failure_stages"]},
        "train_sanity": {"complete": bool(sanity.get("complete") and sanity_binding), "manifest_binding_verified": sanity_binding,
            "excluded_from_main203": sanity.get("excluded_from_test203"), "actor_split": sanity.get("actor_split"),
            "preselected_episodes": sanity_plan.get("preselected_episodes"), "planned_trials": sanity_plan.get("expected_diagnostic_trials"),
            "models": sanity_models, "recorded_source_endpoints": sanity.get("recorded_source_endpoints_scoring_only"),
            "interpretation": sanity.get("replay_interpretation"), "scope": sanity.get("scope"),
            "source_ids": ["train_sanity_summary", "train_sanity_manifest"]},
        "spatial_probe": {"complete": bool(token.get("complete") and representation.get("cross_seed", {}).get("complete")),
            "metric_scope": representation.get("metric_scope"), "data": token.get("data_manifest_summary", {}),
            "data_quality": token.get("data_quality", {}), "alignment_validation": token.get("alignment_summary"),
            "runtime_cache_parity": token.get("runtime_cache_parity"), "primary_model_seed": primary.get("seed"),
            "primary_ablation_rows": [{"mode": mode, **values} for mode, values in primary.get("table", {}).items()],
            "primary_evidence": representation.get("primary_evidence"), "seed_rows": seed_rows,
            "seed_identity_note": "model_seed is the run training seed; nested bootstrap/paired-comparison seed fields are the bootstrap RNG seed, not another model seed",
            "cross_seed_descriptive": representation.get("cross_seed", {}).get("descriptive_seed_statistics", {}),
            "seed_aggregation_note": representation.get("cross_seed", {}).get("aggregation_note"),
            "posthoc_counterfactual": pick(token.get("posthoc_audits", {}).get("counterfactual_equal_source", {}),
                ("complete", "interpretation", "mean_across_same_test_episodes_and_three_training_seeds")),
            "xyz_sensitivity": token.get("posthoc_audits", {}).get("xyz_sensitivity"),
            "limitations": token.get("limitations", []), "source_ids": ["token_pilot_results"]},
        "legacy_reference": {"environment_dim": 256, "new_environment_dim": 300,
            "cross_generation_causal_comparison_allowed": False,
            "comparison_limit": "Old ACT256 and new ACT300 have different interfaces/initialization; their before-after difference cannot isolate touch or projection training.",
            "fixed6000": {"complete": old_fixed.get("complete", False), "rows": old_rows(old_fixed), "selection": old_fixed.get("checkpoint_selection")},
            "dev_selected1000": {"complete": old_selected.get("complete", False), "rows": old_rows(old_selected), "scope": old_selected.get("scope")},
            "command_replay": token.get("command_replay_reference"),
            "nominal_added": [{"mode": row.get("mode"), "metrics": metric(row.get("subsets", {}).get("nominal"))} for row in simple.get("evaluations", []) if row.get("kind") == "legacy"],
            "source_ids": ["token_pilot_results", "simple_results"]},
        "diagnostics": {"control_targets": {"complete": control.get("complete", False), "model": control.get("model"),
                "train_source": control.get("train_source_full_rate"),
                "window": pick(control.get("nominal_precontact_window", {}), ("commands", "source_command_indices_inclusive", "post_action_ticks_inclusive", "normalization",
                    "reference_contact_recording", "reference_has_q_trajectory", "reference_final_cumulative_clipped_count")),
                "comparisons": diagnostic_rows, "interpretation_boundaries": control.get("interpretation_boundaries", []), "source_ids": ["control_target_audit"]},
            "source_endpoints": {"complete": endpoints.get("complete", False), "summary": endpoints.get("summary", {}),
                "episodes": [pick(row, ("episode", "raw_episode", "split", "frames", "initial_cube_xyz", "terminal_cube_xyz", "terminal_legacy_goal_predicate")) for row in endpoints.get("episodes", [])],
                "definitions": endpoints.get("definitions", {}), "source_ids": ["source_endpoint_audit"]}},
        "limitations": ["闭环主网格来自同一条ACT validation episode，且只有一个ACT训练seed；多个重置不等于多个独立任务或场景。",
            "空间probe的三个训练seed共享八条test episode；775是静态样本数量，包含相关的反事实兄弟样本。",
            "中心回归误差、完整形状重建和闭环策略成功是不同指标；前者改善不能直接证明后者改善。",
            "XY范围内外只是初态位置包围盒分组，不是完整ID/OOD证明。",
            "配对区间包含零时方向仍不确定，不能解释为等效；单nominal的bootstrap退化区间没有泛化意义。",
            "n2同时提高RGB、state与pressure更新频率；它不是仅改变触觉反馈的试验。",
            "仿真在推理时暂停，墙钟延迟不能证明真机实时性；未做跨任务或跨物体网格泛化。",
            "源末帧位置predicate不等于strict稳定成功；越界命令统计与clipping数量不能单独解释失败。"],
        "provenance": {"sources": list(source.items.values()), "missing_files": source.missing,
            "problems": problems, "builder_repo_path": "scripts/build_stage1a_portal_data.py", "builder_sha256": digest(__file__),
            "scope": "Portal projection of immutable source reports and checkpoint-selection evidence; source paths are provenance labels, not unverified web links."}}
    result["overview"]["data_complete"] = not source.missing and not problems
    return portable_paths(result, root)


def write_exports(result, directory):
    """Small downloads exclude raw host paths and preserve their source hashes."""
    rows = []
    source_report = next(item for item in result["provenance"]["sources"] if item["id"] == "simple_results")
    for row in result["policy_results"]["rows"]:
        for steps, subsets in ((8, SUBSETS), (2, ("within_bbox12", "nominal"))):
            for subset in subsets:
                value = row[f"n{steps}"][subset]
                ci = value.get("ci95") or [None, None]
                rows.append({"mode": row["mode"], "projection": row["projection"], "tactile": row["tactile"],
                    "n_action_steps": steps, "subset": subset, "complete": value["complete"],
                    "cases": value.get("cases"), "successes": value.get("successes"), "rate": value.get("rate"),
                    "ci95_low": ci[0], "ci95_high": ci[1], "lifted": value.get("lifted"), "dropped": value.get("dropped"),
                    "selected_step": row["selection"]["selected_step"], "training_budget_steps": row["selection"]["training_budget_steps"],
                    "source_report_repo_path": source_report["repo_path"], "source_report_sha256": source_report["sha256"]})
    payload = {"schema_version": 1, "stage_id": result["stage_id"], "complete": result["overview"]["complete"],
        "rate_unit": "fraction", "metrics": rows, "comparisons": result["policy_results"]["comparisons"],
        "limitations": result["limitations"], "provenance": result["provenance"]}
    (directory / "stage1a-metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (directory / "stage1a-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    cases = result["policy_results"]["cases"]
    with (directory / "stage1a-cases.csv").open("w", newline="") as stream:
        fields = list(cases[0]) if cases else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "portal/data/stage1a.json")
    args = parser.parse_args()
    result = build(args.repository_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    write_exports(result, args.output.parent)
    print(json.dumps({"output": str(args.output), "complete": result["overview"]["complete"],
        "data_complete": result["overview"]["data_complete"], "missing": len(result["provenance"]["missing_files"])}))


if __name__ == "__main__":
    main()
