#!/usr/bin/env python3
"""Build by default. SSH publishing occurs only with an explicit --deploy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

from tactile_vggt.paths import artifact_path

try:
    from .common import public_files, sha256, validate_bundle, validate_links
    from .config import load_config, require_remote, SERVICE
except ImportError:
    from common import public_files, sha256, validate_bundle, validate_links
    from config import load_config, require_remote, SERVICE

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def entries(root):
    return [{"path": rel, "size": path.stat().st_size, "sha256": sha256(path)} for rel, path in public_files(root)]


def build(source=None, bundles=None, config=None):
    config = load_config(config)
    source = Path(source or artifact_path("site"))
    if source.is_symlink():
        raise ValueError("Portal source cannot be a symlink")
    source = source.resolve(strict=True)
    if not (source / "index.html").is_file():
        raise ValueError("portal/index.html is required before building")
    source_files = public_files(source)
    release = "portal-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = Path(bundles or artifact_path("deploy", "bundles")) / release
    public = bundle / "public"
    public.mkdir(parents=True)
    for rel, path in source_files:
        destination = public / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    runtime = bundle / "runtime"
    runtime.mkdir()
    for name in ("serve.py", "README.md"):
        shutil.copyfile(HERE / name, runtime / name)
    unit = (HERE / SERVICE).read_text().replace("__BIND__", config["bind"]).replace("__PORT__", str(config["port"]))
    (runtime / SERVICE).write_text(unit)
    manifest = {"format_version": 1, "release": release, "url": config["url"], "remote": config["remote"],
                "remote_directory": config["remote_directory"], "service": SERVICE, "bind": config["bind"], "port": config["port"],
                "source_directory": str(source), "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": entries(public), "runtime_files": entries(runtime),
                "links": validate_links(public)}
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    validate_bundle(bundle)
    return bundle, manifest


def remote(phase, release, token, previous=None, manifest_sha256=None, config=None):
    config = config or load_config()
    require_remote(config)
    args = ["/usr/bin/python3", "-", phase, release, token]
    if previous is not None:
        args.append(previous)
    if manifest_sha256 is not None:
        args.append(manifest_sha256)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config["remote"],
                             " ".join(shlex.quote(arg) for arg in args)],
                            input="DEPLOY_CONFIG = " + repr(config) + "\n" + (HERE / "remote_install.py").read_text(), text=True,
                            capture_output=True, check=True, timeout=90)
    return json.loads(result.stdout)


def deploy(bundle, config=None):
    config = load_config(config)
    bundle = Path(bundle).resolve(strict=True)
    manifest, _ = validate_bundle(bundle)
    expected = {key: config[key] for key in ("remote", "remote_directory", "service", "url", "bind", "port")}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Bundle is not scoped to this independent portal")
    require_remote(config)
    release = manifest["release"]
    if not re.fullmatch(r"portal-[A-Za-z0-9_-]{1,90}", release):
        raise ValueError("Invalid release identifier")
    token, prepared = uuid.uuid4().hex, False
    try:
        preparation = remote("prepare", release, token, config=config)
        prepared = True
        if preparation["incoming"] != config["remote_directory"] + "/.incoming/" + release:
            raise ValueError("Unexpected upload destination")
        subprocess.run(["rsync", "-rt", "--chmod=D755,F644", str(bundle / "public"),
                        str(bundle / "runtime"), str(bundle / "manifest.json"),
                        config["remote"] + ":" + preparation["incoming"] + "/"], check=True)
        receipt = remote("activate", release, token, preparation["previous_current"] or "-", sha256(bundle / "manifest.json"), config=config)
        receipt["manifest_sha256"] = sha256(bundle / "manifest.json")
        receipt["bundle"] = str(bundle)
        artifact_path("deploy").mkdir(parents=True, exist_ok=True)
        (artifact_path("deploy", "deployment.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        if prepared:
            remote("unlock", release, token, config=config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--build-only", action="store_true", help="Default; creates a local, verified snapshot only")
    operation.add_argument("--deploy", action="store_true", help="Explicitly publish this portal's own service")
    parser.add_argument("--bundle", type=Path, help="Validate/reuse a previously built snapshot")
    parser.add_argument("--source", type=Path, help="Explicit generated site directory (default: artifact site)")
    parser.add_argument("--bundles", type=Path, help="Local bundle output directory")
    parser.add_argument("--config", type=Path, help="Local machine configuration; required for deployment")
    args = parser.parse_args(argv)
    if args.bundle:
        bundle, manifest = args.bundle.resolve(strict=True), validate_bundle(args.bundle)[0]
    else:
        bundle, manifest = build(args.source, args.bundles, args.config)
    result = deploy(bundle, args.config) if args.deploy else {"deployed": False, "bundle": str(bundle),
        "release": manifest["release"], "files": len(manifest["files"]),
        "bytes": sum(entry["size"] for entry in manifest["files"]), "links": manifest["links"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
