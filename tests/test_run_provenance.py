"""A run is tied to actual sources, including dirty revisions and resumes."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess

from tactile_vggt.provenance import file_sha256, write_run_manifest


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_run_records_commit_dirty_sources_and_input_hashes_without_overwriting(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src/model.py"
    source.write_text("version = 1\n")
    git(repo, "init")
    git(repo, "add", "src")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"split": [1, 2]}')
    args = Namespace(seed=43, data=data)
    output = tmp_path / "run"
    first = write_run_manifest(output, args, root=repo)
    original = first.read_bytes()
    clean = json.loads(original)
    assert clean["git"]["commit"] == git(repo, "rev-parse", "HEAD")
    assert clean["git"]["dirty"] is False
    assert clean["inputs"]["data"]["sha256"] == file_sha256(data / "manifest.json")
    source.write_text("version = 2\n")
    second = write_run_manifest(output, args, root=repo)
    dirty = json.loads(second.read_text())
    assert dirty["git"]["dirty"] is True
    assert dirty["source_sha256"]["src/model.py"] != clean["source_sha256"]["src/model.py"]
    assert dirty["configuration_sha256"] == clean["configuration_sha256"]
    assert first != second and first.read_bytes() == original
    assert not list(output.rglob("*.py"))


def test_missing_git_or_inputs_are_explicitly_unknown(tmp_path):
    repo = tmp_path / "source-distribution"
    repo.mkdir()
    record = json.loads(write_run_manifest(tmp_path / "run", {"weights": str(tmp_path / "missing.pt")}, root=repo).read_text())
    assert record["git"] == {"available": False, "commit": None, "dirty": None}
    assert record["inputs"]["weights"]["sha256"] is None


def test_explicit_backbone_and_resume_checkpoint_are_recorded(tmp_path):
    repo = tmp_path / "alternate-vggt"
    repo.mkdir()
    git(repo, "init")
    (repo / "model.py").write_text("alternate backbone")
    git(repo, "add", "model.py")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"checkpoint-content")
    record = json.loads(write_run_manifest(tmp_path / "run", {"repository": repo, "resume": checkpoint}, root=repo).read_text())
    assert record["dependencies"]["vggt"]["path"] == str(repo)
    assert record["dependencies"]["vggt"]["commit"] == git(repo, "rev-parse", "HEAD")
    assert record["inputs"]["resume"]["files"][str(checkpoint / "model.pt")] == file_sha256(checkpoint / "model.pt")
