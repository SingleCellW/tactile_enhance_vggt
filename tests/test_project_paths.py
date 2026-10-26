"""Path portability, explicit precedence and historical evidence compatibility."""
from argparse import Namespace
import json
from pathlib import Path

import pytest

from tactile_vggt import paths


def test_wheel_defaults_to_working_directory_and_explicit_root_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("TACTILE_VGGT_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "venv/lib/python3.12/site-packages/tactile_vggt/paths.py"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    assert paths._project_root() == workspace
    monkeypatch.setenv("TACTILE_VGGT_PROJECT_ROOT", str(tmp_path / "explicit"))
    assert paths._project_root() == tmp_path / "explicit"


def test_artifacts_and_dependencies_follow_configured_roots_without_creating_them(tmp_path, monkeypatch):
    monkeypatch.setenv("TACTILE_VGGT_ARTIFACT_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("TACTILE_VGGT_DEPENDENCY_ROOT", str(tmp_path / "shared"))
    assert paths.artifact_path("outputs", "run1") == tmp_path / "scratch/runs/run1"
    assert paths.artifact_path("inputs", "raw") == tmp_path / "scratch/data/raw"
    assert paths.dependency_path("lerobot") == tmp_path / "shared/lerobot"
    assert not (tmp_path / "scratch").exists()
    with pytest.raises(ValueError):
        paths.dependency_path("../other-project")


def test_machine_precedence_and_relative_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    config = tmp_path / "machine/config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"raw_root": "../raw", "dynamic_root": None}))
    monkeypatch.setenv("TACTILE_VGGT_MACHINE_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)
    assert paths.machine_path("raw_root").resolve() == tmp_path / "raw"
    monkeypatch.setenv("TACTILE_VGGT_RAW_ROOT", str(tmp_path / "override"))
    assert paths.machine_path("raw_root") == tmp_path / "override"
    assert paths.machine_path("dynamic_root") is None
    with pytest.raises(ValueError, match="TACTILE_VGGT_DYNAMIC_ROOT"):
        paths.machine_path("dynamic_root", required=True)


def test_cli_override_and_python_symlink_are_preserved(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(__import__("sys").executable))
    monkeypatch.setenv("TACTILE_VGGT_WORKER_PYTHON", str(python))
    assert paths.machine_path("worker_python") == python
    explicit = tmp_path / "actual"
    explicit.mkdir()
    args = Namespace(raw_root=explicit)
    monkeypatch.setenv("TACTILE_VGGT_RAW_ROOT", str(tmp_path / "missing"))
    assert paths.require_machine_arguments(args, "raw_root").raw_root == explicit


def test_old_artifact_reference_is_resolved_without_rewriting_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("TACTILE_VGGT_ARTIFACT_ROOT", str(tmp_path / ".local"))
    assert paths.resolve_artifact_reference("artifacts/prefix/manifest.json") == tmp_path / ".local/artifacts/prefix/manifest.json"
    existing = tmp_path / "artifacts/prefix/manifest.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("historical")
    assert paths.resolve_artifact_reference(existing) == existing
    foreign = Path("/another/project/artifacts/prefix/manifest.json")
    assert paths.resolve_artifact_reference(foreign) == foreign
    assert existing.read_text() == "historical"
