"""Portable report projection, null partial results, and real selection evidence."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "stage1a_portal_data", Path(__file__).resolve().parents[1] / "scripts/build_stage1a_portal_data.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_partial_sources_leave_metrics_null_and_exports_have_only_real_subsets(tmp_path):
    source = tmp_path / builder.SIMPLE / "results.json"
    save(source, {"complete": False, "evaluations": [{"kind": "simple", "mode": "frozen_zero", "n_action_steps": 8,
        "subsets": {"grid36": {"cases": 36, "selected_expected_cases": 36, "selected_complete": True,
            "strict_successes": 8, "strict_success_rate": 8/36, "strict_success_wilson_ci95": [.11, .38]}},
        "case_records": [{"case": {"case_id": "nominal", "episode_id": 1, "object_xy_offset": [0,0],
                                  "subset_flags": {"nominal": True}},
            "outcome": {"strict_stable_goal": False}, "pressure_diagnostics": None}]}]})
    data = builder.build(tmp_path)
    assert not data["overview"]["complete"] and not data["overview"]["data_complete"]
    first = data["policy_results"]["rows"][0]
    assert first["n8"]["grid36"]["successes"] == 8
    assert first["n8"]["grid36"]["ci95"] == [.11, .38]  # Copied, not recomputed.
    assert first["n2"]["within_bbox12"]["successes"] is None
    assert "outside24" not in first["n2"] and first["n2"]["not_evaluated"] == ["grid36", "outside24"]
    assert data["policy_results"]["cases"][0]["pressure_contact_refreshes"] is None
    assert data["policy_results"]["cases"][0]["stage"] is None
    provenance = next(row for row in data["provenance"]["sources"] if row["id"] == "simple_results")
    assert provenance["sha256"] == builder.digest(source)
    builder.write_exports(data, tmp_path)
    exported = json.loads((tmp_path / "stage1a-metrics.json").read_text())
    assert len(exported["metrics"]) == 24
    assert not any(row["n_action_steps"] == 2 and row["subset"] in ("grid36", "outside24") for row in exported["metrics"])


def test_checkpoint_selection_requires_actual_metrics_marker_and_completed_budget(tmp_path):
    mode = "frozen_zero"
    directory = tmp_path / builder.EXPERIMENT / "train" / mode
    config = {"steps": 6000, "seed": 43, "eval_every": 1000,
              "validation_scope": "all original held-out observations", "policy": {"encoder_dim": 300}}
    save(directory / "train_config.json", config)
    save(directory / "status.json", {"status": "completed", "step": 6000})
    (directory / "metrics.jsonl").write_text("\n".join(json.dumps({"step": step, "validation_performed": True,
        "val_normalized_action_l1": .3 if step in (1000,2000) else .5}) for step in range(1000,6001,1000)))
    marker = directory / "checkpoints/001000/checkpoint_complete.json"
    save(marker, {"step": 1000})
    report = {"complete": True, "selected_step": 1000, "selection_value": .3}
    evidence = builder.selection_evidence(builder.Sources(tmp_path), mode, report)
    assert evidence["complete"] and evidence["selected_step"] == 1000
    assert len(evidence["validation_history"]) == 6 and not evidence["uses_rollout_outcomes"]
    save(marker, {"step": 2000})
    assert not builder.selection_evidence(builder.Sources(tmp_path), mode, report)["complete"]


def test_public_paths_keep_root_relative_urls_without_host_filesystem_leaks(tmp_path):
    data = {"download": "/data/stage1a.json", "local": str(tmp_path / "reports/result.json"),
            "remote": "/home/wukf/projects/tactile_enhance_vggt/outputs/checkpoint.json",
            "external": "/home/wukf/raw/episode/qpos.npy"}
    public = builder.portable_paths(data, tmp_path)
    assert public == {"download": "/data/stage1a.json", "local": "reports/result.json",
                      "remote": "outputs/checkpoint.json", "external": "external-source/qpos.npy"}
