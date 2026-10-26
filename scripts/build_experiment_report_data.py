#!/usr/bin/env python3
"""Assemble an offline HTML report's data from existing, immutable evaluation files.

This script performs no model inference, simulation, training or network access.
Run from any directory; all source locations are resolved from this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tactile_vggt.paths import artifact_path, artifact_root


ROOT = Path(__file__).resolve().parents[1]
REPORT = artifact_path('reports', 'experiment_review_20261019')
LABELS = {"anchored": "空间锚定触觉", "unanchored": "全局触觉（无显式锚定）", "no_tactile": "无触觉"}
EPISODES = [1, 24, 35, 43, 52]
SOURCE_REGISTRY: dict[str, dict] = {}


def input_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts[0] == "docs":
        return artifact_path("reports", "historical_docs", *path.parts[1:])
    if path.parts[0] in ("artifacts", "reports", "outputs", "runs", "cache"):
        return artifact_path(path.parts[0], *path.parts[1:])
    return ROOT / path


def source(path: Path | str) -> dict:
    path = Path(path)
    path = input_path(path)
    key = str(path.relative_to(artifact_root())) if path.is_relative_to(artifact_root()) else str(path)
    if key not in SOURCE_REGISTRY:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        SOURCE_REGISTRY[key] = {
            "id": key,
            "path": str(path),
            "href": os.path.relpath(path, REPORT),
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    return SOURCE_REGISTRY[key]


def read(path: Path | str) -> dict:
    source(path)
    path = Path(path)
    return json.loads(input_path(path).read_text())


def href(path: Path) -> str:
    return os.path.relpath(path, REPORT)


def failure(final: dict) -> str:
    if final["strict_stable_goal"]:
        return "success"
    if not final["lifted"]:
        return "did_not_reach_lift_threshold"
    if final.get("first_goal_tick") is None:
        return "lifted_but_never_entered_goal"
    return "entered_goal_but_no_stable_release"


def episode_data(path: Path, detailed: bool) -> dict:
    data = read(path)
    initial, final = data["initial_info"], data["final_info"]
    result = {
        "episode": data["episode"],
        "initial_xyz": initial["cube_center"],
        "strict": bool(final["strict_stable_goal"]),
        "weak": bool(final["inside_goal"]),
        "lifted": bool(final["lifted"]),
        "failure": failure(final),
        "final_xyz": final["cube_center"],
        "max_height": final["max_cube_height"],
        "first_goal_tick": final.get("first_goal_tick"),
        "stable_end_seconds": final["stable_goal_ticks"] * 0.01,
        "final_contacts": final["robot_cube_contact_count"],
        "control_steps": data["completed_control_steps"],
        "physical_seconds": data["sim_duration_s"],
        "wall_seconds": data["wall_seconds"],
        "latency": {"geometry": data.get("geometry_refresh"), "policy_query_and_queue": data.get("policy_query_and_queue")},
        "clip_fraction": final.get("cumulative_clipped_count", 0) / (data["completed_control_steps"] * 22),
        "source": source(path),
        "sources": [source(path)],
        "video": None,
        "trajectory": [],
    }
    video = path.with_name("rollout.mp4")
    if video.exists():
        result["video"] = href(video)
        result["sources"].append(source(video))
    steps = path.with_name("steps.jsonl")
    if detailed and steps.exists():
        result["sources"].append(source(steps))
        rows = [json.loads(line) for line in steps.read_text().splitlines() if line]
        assert len(rows) == data["completed_control_steps"]
        selected = rows[::5]
        if selected[-1] is not rows[-1]:
            selected.append(rows[-1])
        # Initial point and samples every 50ms; never interpolate across contact events.
        result["trajectory"] = [{"tick": 0, "t": 0.0, "x": initial["cube_center"][0], "y": initial["cube_center"][1], "z": initial["cube_center"][2], "hand_contacts": 0, "clipped_count": 0, "inside_goal": False}]
        for row in selected:
            info = row["info"]
            x, y, z = info["cube_center"]
            result["trajectory"].append({"tick": row["control_tick"], "t": row["sim_time"], "x": x, "y": y, "z": z, "hand_contacts": info["robot_cube_contact_count"], "clipped_count": info.get("clipped_count", 0), "inside_goal": info["inside_goal"]})
        result["contact_ticks"] = sum(row["info"]["robot_cube_contact_count"] > 0 for row in rows)
        result["clipped_ticks"] = sum(row["info"].get("clipped_count", 0) > 0 for row in rows)
        result["contact_intervals"] = intervals(rows, lambda r: r["info"]["robot_cube_contact_count"] > 0)
        result["lift_intervals"] = intervals(rows, lambda r: r["info"]["cube_center"][2] >= initial["cube_center"][2] + 0.05)
    return result


def intervals(rows: list, predicate) -> list:
    found, start = [], None
    for row in rows:
        if predicate(row) and start is None:
            start = row["sim_time"]
        elif not predicate(row) and start is not None:
            found.append([start, row["sim_time"]])
            start = None
    if start is not None:
        found.append([start, rows[-1]["sim_time"]])
    return found


def model_data(summary_path: Path, model: dict, detailed: bool) -> dict:
    model_name = model["model"]
    mode, step = model_name.rsplit("_", 1)
    directory = summary_path.parent / model_name
    config_path = summary_path.parent / "eval_config.json"
    config = read(config_path)
    checkpoint = next((item for item in config["checkpoints"] if item["path"] == model["checkpoint"]), {"path": model["checkpoint"]})
    per_episode = [episode_data(directory / f"episode_{episode:03d}/result.json", detailed) for episode in EPISODES]
    result = {
        "model": model_name, "mode": mode, "label": LABELS[mode], "step": int(step),
        "strict": model["strict_stable_goal_count"], "weak": model["final_goal_count"], "lifted": model["lifted_count"], "n": model["episodes"],
        "source": source(summary_path), "sources": [source(summary_path), source(config_path)],
        "checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint.get("model_sha256"),
        "cache_manifest_sha256": config.get("cache_manifest_sha256"),
        "eval_protocol": config.get("protocol"), "per_episode": per_episode,
    }
    assert result["strict"] == sum(item["strict"] for item in per_episode)
    assert result["weak"] == sum(item["weak"] for item in per_episode)
    assert result["lifted"] == sum(item["lifted"] for item in per_episode)
    assert result["n"] == len(per_episode) == 5
    return result


def stage(stage_id: str, label: str, version: str, step: int, description: str, roots: list[str], detailed: bool = False) -> dict:
    models = []
    for root in roots:
        path = input_path(Path(root) / "summary.json")
        summary = read(path)
        assert summary["complete"]
        models.extend(model_data(path, model, detailed) for model in summary["models"] if int(model["model"].rsplit("_", 1)[1]) == step)
    return {"id": stage_id, "label": label, "version": version, "step": step, "description": description, "models": models, "rollouts": sum(model["n"] for model in models)}


def build() -> dict:
    stages = [
        stage("v1_1k", "v1 · 1k", "v1", 1000, "早期开发评测；动作偏移与触觉速度时间基准尚未修正。", ["artifacts/eval_v1_paired"]),
        stage("v1_36k", "v1 · 36k", "v1", 36000, "历史锚定优势；存在时序限制，不能与v2合并计算。", ["artifacts/eval_v1_paired"]),
        stage("v2_1k_first", "v2 · 1k 首次", "v2", 1000, "修正时序后按验证L1选择的早期模型；开发集rollout。", ["artifacts/eval_v2_best_all"]),
        stage("v2_6k", "v2 · 6k", "v2", 6000, "修正当前命令标签、物理与触觉速度dt=.01s后的短训练。", ["artifacts/eval_v2_final_tactile", "artifacts/eval_v2_final_no_tactile"]),
        stage("v2_1k_repeat", "v2 · 同一1k重跑", "v2", 1000, "同一个checkpoint重跑，锚定episode24由失败变为成功。", ["artifacts/full36k_review/raw/best_saved"]),
        stage("v2_36k", "v2 · 36k 等预算", "v2", 36000, "当前主要对照：同数据、同种子、同训练步数，五个固定开发初态。", ["artifacts/full36k_review/raw/final_equal_budget"], True),
    ]
    metrics_path = "artifacts/full36k_review/metrics_audit.json"
    metrics = read(metrics_path)
    current = dict(stages[-1])
    current["paired_statistics"] = metrics["closed_loop"]["final_equal_budget"]["paired_statistics"]
    current["source"] = source(metrics_path)
    for model in current["models"]:
        audited = metrics["closed_loop"]["final_equal_budget"]["models"][model["model"]]
        model["offline"] = metrics["offline"][model["mode"]]
        model["clipping"] = audited["clipping"]
        model["clipping_magnitude"] = audited["clipping_magnitude"]
        model["failure_stage_counts"] = audited["failure_stage_counts"]
        model["latency"] = {
            "geometry_ms_mean": audited["geometry_ms_mean_over_all_refreshes"],
            "geometry_ms_median_episode_medians": audited["geometry_ms_median_episode_medians"],
            "policy_queue_ms_mean": audited["mixed_policy_query_queue_ms_mean"],
            "wall_seconds_mean": audited["wall_seconds_per_episode"]["mean"],
        }
        model["sources"].append(source(metrics_path))

    geometry_path = "artifacts/geometry_gt_audit_20261016_r2/report.json"
    geometry = read(geometry_path)
    geometry_rows = []
    for variant, label in [("current_sim3", "当前Sim3对齐"), ("known_extrinsics_same_depth", "已知外参、同预测深度"), ("known_K_and_extrinsics_same_depth", "已知内外参、同预测深度")]:
        data = geometry["summary"][variant]
        geometry_rows.append({"variant": variant, "label": label, **{key: 100 * data[subset]["pixel_corresponded_xyz_error_m"]["pixel_weighted_mean"] for key, subset in [("workspace_cm", "workspace/both"), ("cube_cm", "cube/both"), ("near_tactile_cm", "near_tactile_10cm/both")]}})

    repeat_path = "docs/repeatability_audit.json"
    repeat = read(repeat_path)
    bounds_path = "artifacts/full36k_review/action_bounds_audit.json"
    bounds = read(bounds_path)
    bounds_check = bounds["raw_vs_explicit_clipped_cpu_one_step_check"]
    train = bounds["statistics"]["train"]["full_rate_rows"]
    timing_path = "docs/physics_transition_audit.json"
    timing = read(timing_path)
    repair_path = "docs/native_tactile_repair_result.json"
    repair = read(repair_path)
    training_path = "artifacts/training_v2/runs/s43_anchored/train_config.json"
    training = read(training_path)
    offline_path = "artifacts/full36k_review/raw/offline.json"
    offline = read(offline_path)
    curves = {}
    for run in offline["runs"]:
        checkpoints = run["checkpoints"]
        mode = checkpoints[0]["mode"]
        curves[mode] = [{"step": ck["step"], "full_chunk_l1": ck["full_chunk"]["all"]["normalized_l1"], "executed_prefix_l1": ck["executed_prefix"]["all"]["normalized_l1"], "first_action_l1": ck["first_action"]["all"]["normalized_l1"]} for ck in checkpoints]
    current["offline_curves"] = curves
    current["offline_source"] = source(offline_path)
    status_path = "artifacts/full36k_review/raw/status.json"
    status = read(status_path)
    report = {
        "meta": {
            "title": "Tactile × VGGT · 仿真策略实验复核", "report_date": "2026-10-19", "data_snapshot": "截至2026-10-16已完成结果；本报告不运行新测试", "timezone": "Asia/Shanghai",
            "total_rollouts": sum(item["rollouts"] for item in stages), "unique_initial_states": len(EPISODES), "training_seeds": [43],
            "episodes": EPISODES, "status": status["status"], "auto_eval_completed_utc": status["updated_at"], "auto_eval_completed_local": "2026-10-16 04:32:46 +08:00", "status_source": source(status_path),
            "warning": "80次rollout重复使用五个开发初态，包含模型阶段与重跑，不是80个独立测试场景。历史v1与v2不可合并估计效果。",
            "signal_note": "触觉是固定原生proxy，非校准牛顿力。当前留档只有MuJoCo接触计数与物体轨迹，未保存每次refresh的44路压力序列。",
            "training": {"seed": training["seed"], "batch_size": training["batch_size"], "learning_rate": training["learning_rate"], "amp": training["amp"], "train_observations": training["train_observations"], "val_observations": training["val_observations"], "chunk_size": training["policy"]["chunk_size"], "n_action_steps": training["policy"]["n_action_steps"], "kl_weight": training["policy"]["kl_weight"], "frozen_vggt": training["frozen_vggt"], "source": source(training_path)},
            "metric_definitions": {"strict": "最终位于目标区，线速度<0.05m/s、角速度<0.5rad/s、无机器人接触连续至少1个物理秒。", "weak": "仅最终中心x∈[.4,.6]、y∈[-.39,-.21]、z≤.1；不检查完整入框或稳定释放。", "lifted": "曾达到初始物体中心高度+5cm，不等于持续稳定抓取。", "offline": "515个验证观测，unpadded动作标量加权normalized L1；不是独立闭环测试。", "time": "每动作10ms，预测16步并执行8步；600步=6物理秒。推理时暂停物理，不代表实时控制。"},
        },
        "stages": stages, "current": current,
        "audits": {
            "geometry": {"source": source(geometry_path), "rows": geometry_rows, "frames": geometry["selection"]["frames"], "selection": geometry["selection"]["method"], "metric": geometry["metric"], "limitation": geometry["limitation"], "gt_given_to_policy": geometry["gt_given_to_vggt_or_policy"]},
            "repeatability": {"source": source(repeat_path), "confirmed_findings": repeat["confirmed_findings"], "hypotheses": repeat["unproven_hypotheses"], "provenance_gaps": repeat["provenance_gaps"], "cases": [{"case": case["case"], "strict_success": case["strict_success"], "lifted": case["lifted"], "initial_equal": case["initial_info_equal"], "first_8_actions_max_abs_difference_rad": case["trajectory_differences"]["actions"]["max_abs_difference_first_8_rows"]} for case in repeat["cases"]]},
            "action_bounds": {"source": source(bounds_path), "train_rows": train["rows"], "train_hand_oob_fraction": train["hand"]["out_of_bounds_scalar_fraction"], "train_all_oob_fraction": train["all"]["out_of_bounds_scalar_fraction"], "train_rows_any_oob_fraction": train["all"]["row_fraction_any_out_of_bounds"], "original_commands_equal": bounds["equality_with_original_commands"]["all_20212_rows_equal_after_original_float32_cast"], "one_step_states": bounds_check["states"], "all_next_qpos_equal": bounds_check["all_next_qpos_equal"], "all_next_qvel_equal": bounds_check["all_next_qvel_equal"], "all_actuator_force_equal": bounds_check["all_actuator_force_equal"], "notes": bounds["notes"]},
            "timing": {"source": source(timing_path), "conclusion": timing["conclusion"], "action_target_offset_frames": timing["action_target_offset_frames"], "velocity_dt_s": timing["velocity_dt_s"], "substeps_per_action": timing["substeps_per_action"]},
            "repair": {"source": source(repair_path), "summary": repair},
        },
    }
    assert report["meta"]["total_rollouts"] == 80
    assert len(stages) == 6
    assert sum(len(stage["models"]) for stage in stages) == 16
    assert [(m["strict"], m["lifted"]) for m in current["models"]] == [(3, 5), (3, 4), (2, 3)]
    # Expose all consulted files with content hashes. No model files are read here;
    # model hashes above are those recorded by the original evaluation.
    report["sources"] = sorted(SOURCE_REGISTRY.values(), key=lambda item: item["id"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "data.json")
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {report['meta']['total_rollouts']} rollouts, five unique initial states, {len(report['sources'])} hashed sources")


if __name__ == "__main__":
    main()
