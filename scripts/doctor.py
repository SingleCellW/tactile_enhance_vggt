#!/usr/bin/env python3
"""Read-only prerequisite checks; a passing core check is not a GPU experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootstrap_dependencies import read_lock, verify_checkout
from tactile_vggt.paths import artifact_path, dependency_path, machine_path


def check_environment(profile="core", *, require_cuda=False, check_weights=False):
    lock = read_lock(ROOT / "dependencies.lock.json")
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"name": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    def minimum_python(version):
        if sys.version_info < version:
            raise RuntimeError(f"Requires Python {version[0]}.{version[1]}+; found {sys.version.split()[0]}")
        return sys.version.split()[0]

    def module_version(name):
        module = importlib.import_module(name)
        return f"{getattr(module, '__version__', 'imported')} at {getattr(module, '__file__', '?')}"

    def source(name):
        path = dependency_path(name)
        if not path.is_symlink():
            raise RuntimeError(f"Missing dependency link {path}; run scripts/bootstrap_dependencies.py")
        return verify_checkout(path, lock["git"][name]["commit"])

    def lerobot_import():
        expected = dependency_path("lerobot").resolve() / "src/lerobot"
        module = importlib.import_module("lerobot")
        actual = Path(module.__file__).resolve()
        if not actual.is_relative_to(expected):
            raise RuntimeError(f"Imported {actual}, expected locked checkout {expected}; install -e external/lerobot")
        # Verify the imports actually needed by the project's ACT adapter.
        importlib.import_module("lerobot.policies.act.configuration_act")
        importlib.import_module("lerobot.policies.act.modeling_act")
        version = metadata.version("lerobot")
        if version != lock["git"]["lerobot"]["distribution_version"]:
            raise RuntimeError(f"LeRobot metadata is {version}, expected {lock['git']['lerobot']['distribution_version']}")
        return f"{version} at {actual}"

    def configured(name, *, directory=False):
        path = machine_path(name, required=True)
        if not (path.is_dir() if directory else path.is_file()):
            raise RuntimeError(f"Configured {name} does not exist: {path}")
        return path

    def dynamic_assets():
        root = configured("dynamic_root", directory=True)
        spec = lock["local_assets"]["dynamic_root"]
        missing = [name for name in spec["required_paths"] if not (root / name).exists()]
        import_roots = (root / "eval/vendor", root / "eval/sim", root)
        missing.extend(name for name in spec.get("required_modules", [])
                       if not any((parent / name).is_file() for parent in import_roots))
        if missing:
            raise RuntimeError(f"Missing native tactile assets under {root}: {', '.join(missing)}")
        return root

    def worker_environment():
        python = configured("worker_python")
        result = subprocess.run([str(python), "-B", "-c",
            "import json, mujoco, torch; print(json.dumps({'mujoco': mujoco.__version__, 'torch': torch.__version__}))"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        versions = json.loads(result.stdout)
        if versions["mujoco"] != "3.4.0":
            raise RuntimeError(f"Expected frozen worker MuJoCo 3.4.0, found {versions['mujoco']}")
        return versions

    def weights():
        path = machine_path("vggt_weights", default=artifact_path("weights", "model.pt"))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != lock["weights"]["vggt_1b"]["sha256"]:
            raise RuntimeError(f"VGGT weight SHA256 mismatch: {path}")
        return path

    check("python", lambda: minimum_python((3, 10)))
    check("numpy", lambda: module_version("numpy"))
    if profile in {"train", "full"}:
        check("training_python", lambda: minimum_python((3, 12)))
        check("lerobot_source", lambda: source("lerobot"))
        check("torch", lambda: module_version("torch"))
        check("torchvision", lambda: module_version("torchvision"))
        check("lerobot_import", lerobot_import)
    if profile == "full":
        check("vggt_source", lambda: source("vggt"))
    if profile in {"simulation", "full"}:
        check("dynamic_assets", dynamic_assets)
        check("raw_root", lambda: configured("raw_root", directory=True))
        check("timestamps_root", lambda: configured("timestamps_root", directory=True))
        check("source_readouts", lambda: configured("source_readouts"))
        check("worker_environment", worker_environment)
    if check_weights or profile == "full":
        check("vggt_weights_sha256", weights)
    if require_cuda:
        def cuda():
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA device is available in this process")
            return torch.cuda.get_device_name(0)
        check("cuda", cuda)
    return {"profile": profile, "ok": all(item["ok"] for item in checks), "checks": checks,
            "scope": "Prerequisites only; does not execute training, VGGT inference or simulator rollouts"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "train", "simulation", "full"), default="core")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-weights", action="store_true", help="Read and verify the full VGGT checkpoint SHA256")
    args = parser.parse_args(argv)
    result = check_environment(args.profile, require_cuda=args.require_cuda, check_weights=args.check_weights)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
