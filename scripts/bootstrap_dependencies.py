#!/usr/bin/env python3
"""Link clean, pinned external checkouts without copying them into this repository.

No network access occurs unless --fetch is supplied. Existing directories,
links, and dependency worktrees are never overwritten or reset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    result = subprocess.run(
        ["git", "-C", str(path), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock.get("format_version") != 1 or not isinstance(lock.get("git"), dict):
        raise ValueError("Unsupported dependency lock format")
    for name, spec in lock["git"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"Unsafe dependency name: {name!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", spec.get("commit", "")):
            raise ValueError(f"{name}: expected a full 40-character commit")
        if not spec.get("url", "").startswith("https://"):
            raise ValueError(f"{name}: expected an HTTPS source URL")
    return lock


def verify_checkout(path, expected_commit):
    path = Path(path).resolve(strict=True)
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path:
        raise ValueError(f"Dependency must be a checkout root: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError(f"{path}: expected {expected_commit}, found {actual}")
    if git(path, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Dependency checkout is dirty: {path}; preserve changes separately")
    return path


def fetch_checkout(spec, target):
    """Create a new checkout atomically; never mutate an existing checkout."""
    target = Path(target)
    if target.exists() or target.is_symlink():
        return verify_checkout(target, spec["commit"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bootstrap-", dir=target.parent))
    try:
        git(staging, "init", "--quiet")
        git(staging, "remote", "add", "origin", spec["url"])
        git(staging, "fetch", "--depth=1", "origin", spec["commit"])
        git(staging, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        verify_checkout(staging, spec["commit"])
        # mkdir above is harmless; a concurrent checkout is never replaced.
        if target.exists() or target.is_symlink():
            return verify_checkout(target, spec["commit"])
        staging.rename(target)
        return target.resolve()
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def ensure_dependency(name, spec, external_root, shared_root, *, local=None, fetch=False, check=False):
    link = Path(external_root) / name
    if link.is_symlink():
        actual = verify_checkout(link, spec["commit"])
        if local is not None and actual != Path(local).resolve():
            raise ValueError(f"{link} already points elsewhere; refusing to replace it")
        return {"name": name, "commit": spec["commit"], "path": str(actual), "status": "verified"}
    if link.exists():
        raise ValueError(f"{link} is not a symlink; refusing to replace existing content")
    if check:
        raise ValueError(f"Missing dependency link {link}; run bootstrap with --local or --fetch")
    target = Path(local).expanduser() if local is not None else Path(shared_root) / name / spec["commit"]
    if not target.exists():
        if local is not None or not fetch:
            raise ValueError(f"Missing {name} checkout {target}; supply --local {name}=PATH or --fetch")
        target = fetch_checkout(spec, target)
    target = verify_checkout(target, spec["commit"])
    link.parent.mkdir(parents=True, exist_ok=True)
    # Symlink creation fails if another process created the destination meanwhile.
    link.symlink_to(os.path.relpath(target, link.parent.resolve()), target_is_directory=True)
    return {"name": name, "commit": spec["commit"], "path": str(target), "status": "linked"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "dependencies.lock.json")
    parser.add_argument("--external-root", type=Path, default=ROOT / "external")
    default_shared = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "tactile-vggt/deps"
    parser.add_argument("--shared-root", type=Path, default=Path(os.environ.get("TACTILE_VGGT_DEPS_ROOT", default_shared)))
    parser.add_argument("--dependency", action="append", help="Limit to a named dependency; repeatable")
    parser.add_argument("--local", action="append", default=[], metavar="NAME=PATH")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Only verify links, commits and clean worktrees")
    mode.add_argument("--fetch", action="store_true", help="Fetch missing pinned sources into the shared pool")
    args = parser.parse_args(argv)
    try:
        lock = read_lock(args.lock)
        local = {}
        for entry in args.local:
            name, separator, path = entry.partition("=")
            if not separator or not path or name not in lock["git"] or name in local:
                raise ValueError(f"Invalid or duplicate --local argument: {entry}")
            local[name] = path
        names = args.dependency or list(lock["git"])
        if set(names) - set(lock["git"]):
            raise ValueError(f"Unknown dependencies: {sorted(set(names) - set(lock['git']))}")
        if set(local) - set(names):
            raise ValueError("A --local dependency was excluded by --dependency")
        results = [ensure_dependency(name, lock["git"][name], args.external_root, args.shared_root,
                                    local=local.get(name), fetch=args.fetch, check=args.check) for name in names]
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"Dependency setup failed: {exc}\n{exc.stderr or ''}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dependency setup failed: {exc}\n")
    print(json.dumps({"dependencies": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
