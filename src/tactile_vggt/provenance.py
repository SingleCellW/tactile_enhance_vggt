"""Record immutable run invocations without copying implementations into outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .paths import PROJECT_ROOT, dependency_path


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def source_hashes(paths):
    """Hash live sources; historical output snapshots remain untouched."""
    return {Path(path).name: file_sha256(path) for path in paths}


def _git(root, *arguments):
    if not Path(root).is_dir():
        return None
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *arguments],
                            capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def git_state(root):
    root = Path(root)
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.decode().strip()).resolve() != root.resolve():
        return {"commit": None, "dirty": None, "available": False}
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    diff = _git(root, "diff", "--binary", "HEAD")
    return {"commit": commit.decode().strip() if commit else None,
            "dirty": bool(status) if status is not None else None,
            "available": commit is not None,
            "status": status.decode().splitlines() if status is not None else None,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_run_manifest(output, configuration, *, root=PROJECT_ROOT):
    """Append a content-addressed invocation record; never rewrite past evidence.

    A dirty checkout is recorded honestly, including hashes of the present
    sources. A later commit cannot be inferred as a historical run's revision.
    """
    output, root = Path(output), Path(root)
    config = _jsonable(vars(configuration) if hasattr(configuration, "__dict__") else configuration)
    encoded_config = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    inputs = {}
    for key in ("cache", "data", "prefix", "features_dir", "config", "weights", "vggt_weights"):
        value = config.get(key) if isinstance(config, dict) else None
        if not value or not isinstance(value, str):
            continue
        path = Path(value)
        candidate = path / "manifest.json" if path.is_dir() else path
        inputs[key] = {"path": str(candidate), "sha256": file_sha256(candidate) if candidate.is_file() else None}
    resume = config.get("resume") if isinstance(config, dict) else None
    if isinstance(resume, str) and resume:
        checkpoint = Path(resume)
        candidates = [checkpoint / name for name in ("model.pt", "training_state.pt", "config.json", "checkpoint_complete.json")] if checkpoint.is_dir() else [checkpoint]
        inputs["resume"] = {"path": str(checkpoint), "files": {
            str(path): file_sha256(path) if path.is_file() else None for path in candidates}}
    versions = {}
    for package in ("numpy", "torch", "torchvision", "lerobot", "mujoco"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = {}
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            sources[str(path.relative_to(root))] = file_sha256(path)
    actual_vggt = (config.get("repository") or config.get("vggt_repo")) if isinstance(config, dict) else None
    dependencies = {"lerobot": dependency_path("lerobot"), "vggt": Path(actual_vggt) if actual_vggt else dependency_path("vggt")}
    record = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "git": git_state(root), "source_sha256": sources,
              "dependencies": {name: {"path": str(path.absolute()), **git_state(path)} for name, path in dependencies.items()},
              "configuration": config, "configuration_sha256": hashlib.sha256(encoded_config).hexdigest(),
              "inputs": inputs, "argv": list(sys.argv), "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(), "packages": versions,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = output / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(payload.encode()).hexdigest() + ".json")
    with path.open("x") as stream:
        stream.write(payload)
    return path
