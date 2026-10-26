"""Fixed-scope SSH deployment transaction; executed remotely over stdin.

This never invokes another service, sudo, firewall changes or loginctl enable.
"""
import hashlib
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

_CONFIG = globals().get("DEPLOY_CONFIG", {})
ROOT = Path(_CONFIG.get("remote_directory") or Path.home() / "tactile-vggt-report-host")
SERVICE = "tactile-vggt-report.service"
UNIT = Path.home() / ".config/systemd/user" / SERVICE
HOST, PORT = _CONFIG.get("bind", "127.0.0.1"), _CONFIG.get("port", 8766)
MARKER = ROOT / ".portal-host.json"
LOCK = ROOT / ".deployment-lock"


def checked_name(value):
    if not re.fullmatch(r"portal-[A-Za-z0-9_-]{1,90}", value):
        raise ValueError("Invalid release identifier")
    return value


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def scoped_root():
    if Path.home() != ROOT.parent or os.geteuid() == 0:
        raise ValueError("Deploy only as the dedicated host account")
    if ROOT.is_symlink() or ROOT.resolve() != ROOT:
        raise ValueError("Hosting directory cannot be a symlink")
    if ROOT.exists() and (not MARKER.is_file() or json.loads(MARKER.read_text()) != {"portal": "tactile-vggt", "version": 1}):
        raise ValueError("Refusing to claim an existing unmarked directory")
    if UNIT.is_symlink() or UNIT.exists() and ("%h/tactile-vggt-report-host/serve.py" not in UNIT.read_text() or f"--port {PORT}" not in UNIT.read_text()):
        raise ValueError("Refusing to replace a unit outside this portal")


def current():
    path = ROOT / "current"
    if not path.is_symlink():
        if path.exists():
            raise ValueError("current must be the portal release symlink")
        return None
    value = os.readlink(path)
    if not value.startswith("releases/") or value != "releases/" + checked_name(value[9:]):
        raise ValueError("Unexpected current target")
    return value


def assert_lock(token):
    if not re.fullmatch(r"[0-9a-f]{32}", token) or (LOCK / "owner").read_text() != token:
        raise ValueError("Deployment lock not owned by this transaction")


def port_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", PORT))


def prepare(release, token):
    scoped_root()
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Invalid lock token")
    if not ROOT.exists() and UNIT.exists():
        raise ValueError("Unit already exists without its owned hosting directory")
    previous = current() if ROOT.exists() else None
    if previous is None:
        port_free()
    if command("loginctl", "show-user", getpass.getuser(), "-p", "Linger", "--value") != "yes":
        raise ValueError("Persistent user services require existing Linger=yes; not changed by this tool")
    ROOT.mkdir(mode=0o755, exist_ok=True)
    if not MARKER.exists():
        MARKER.write_text(json.dumps({"portal": "tactile-vggt", "version": 1}) + "\n")
    LOCK.mkdir()
    (LOCK / "owner").write_text(token)
    try:
        for name in ("releases", ".incoming", "backups"):
            path = ROOT / name
            if path.is_symlink():
                raise ValueError("Unexpected directory symlink")
            path.mkdir(exist_ok=True)
        if (ROOT / "releases" / release).exists():
            raise ValueError("Release already exists; build a new snapshot")
        incoming = ROOT / ".incoming" / release
        incoming.mkdir()
        return {"prepared": True, "previous_current": previous, "incoming": str(incoming)}
    except Exception:
        (LOCK / "owner").unlink()
        LOCK.rmdir()
        raise


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_uploaded(incoming, release, expected_sha256):
    if hash_file(incoming / "manifest.json") != expected_sha256:
        raise ValueError("Uploaded manifest SHA mismatch")
    manifest = json.loads((incoming / "manifest.json").read_text())
    if (manifest.get("release") != release or manifest.get("remote_directory") != str(ROOT) or
            manifest.get("service") != SERVICE or manifest.get("bind") != HOST or manifest.get("port") != PORT):
        raise ValueError("Uploaded manifest scope mismatch")
    for name, entries in (("public", manifest["files"]), ("runtime", manifest["runtime_files"])):
        parent, declared = incoming / name, {}
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("Uploaded root must be a real directory")
        for entry in entries:
            value = entry["path"]
            if not value or value.startswith("/") or "\\" in value or any(p.startswith(".") for p in value.split("/")) or value in declared:
                raise ValueError("Invalid uploaded manifest path")
            declared[value] = entry
        actual = {}
        for path in parent.rglob("*"):
            if path.is_symlink() or not path.is_file() and not path.is_dir():
                raise ValueError("Uploaded snapshots cannot contain symlinks/special files")
            if path.is_file():
                actual[str(path.relative_to(parent))] = path
        if set(actual) != set(declared):
            raise ValueError("Uploaded file set mismatch")
        for value, path in actual.items():
            if path.stat().st_size != declared[value]["size"] or hash_file(path) != declared[value]["sha256"]:
                raise ValueError("Uploaded file size/SHA mismatch: " + value)
    if set(entry["path"] for entry in manifest["runtime_files"]) != {"serve.py", SERVICE, "README.md"}:
        raise ValueError("Unexpected runtime file set")
    return manifest


def atomic_copy(source, destination, token):
    temporary = destination.with_name(destination.name + ".next-" + token)
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Unexpected temporary destination")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o644)
    os.replace(temporary, destination)


def switch_current(value, token):
    temporary = ROOT / ("current.next-" + token)
    os.symlink(value, temporary)
    os.replace(temporary, ROOT / "current")


def wait_healthy(manifest, timeout=5.0):
    """Read this release's registry through HTTP; active alone can race exit."""
    expected = next((entry for entry in manifest["files"] if entry["path"] == "data/reports.json"), None)
    if expected is None:
        raise ValueError("Published data/reports.json is required for activation health")
    opener, deadline, last_error = build_opener(ProxyHandler({})), time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            remaining = max(.01, deadline - time.monotonic())
            subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE],
                           check=True, capture_output=True, timeout=min(1.0, remaining))
            remaining = max(.01, deadline - time.monotonic())
            request = Request("http://%s:%d/data/reports.json" % (HOST, PORT), headers={"Cache-Control": "no-cache"})
            with opener.open(request, timeout=min(1.0, remaining)) as response:
                content = response.read(expected["size"] + 1)
                if response.status != 200 or len(content) != expected["size"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
                    raise ValueError("Live registry differs from the selected snapshot SHA/size")
            return {"url": "http://%s:%d/data/reports.json" % (HOST, PORT),
                    "sha256": expected["sha256"], "unit_active": True}
        except Exception as error:
            last_error = error
            if time.monotonic() < deadline:
                time.sleep(min(.2, max(0, deadline - time.monotonic())))
    raise RuntimeError("Portal did not become healthy within %.1fs: %s" % (timeout, last_error))


def activate(release, token, expected_current, expected_sha256):
    scoped_root()
    assert_lock(token)
    previous = current()
    if (previous or "-") != expected_current:
        raise ValueError("current changed after preparation")
    incoming = ROOT / ".incoming" / release
    manifest = verify_uploaded(incoming, release, expected_sha256)
    unit_text = (incoming / "runtime" / SERVICE).read_text()
    expected_exec = f"ExecStart=/usr/bin/python3 -u %h/tactile-vggt-report-host/serve.py --bind {HOST} --port {PORT} --directory %h/tactile-vggt-report-host/current/public"
    if expected_exec not in unit_text.splitlines() or sum(line.startswith("ExecStart=") for line in unit_text.splitlines()) != 1:
        raise ValueError("Unit ExecStart is outside this portal")
    if previous is None:
        port_free()
    backup = ROOT / "backups" / release
    backup.mkdir()
    (backup / "previous-current.json").write_text(json.dumps(previous) + "\n")
    old_serve, old_unit = (ROOT / "serve.py").is_file(), UNIT.is_file()
    if old_serve:
        shutil.copyfile(ROOT / "serve.py", backup / "serve.py")
    if old_unit:
        shutil.copyfile(UNIT, backup / "unit.service")
    incoming.rename(ROOT / "releases" / release)
    runtime = ROOT / "releases" / release / "runtime"
    UNIT.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_copy(runtime / "serve.py", ROOT / "serve.py", token)
        atomic_copy(runtime / SERVICE, UNIT, token)
        switch_current("releases/" + release, token)
        command("systemctl", "--user", "daemon-reload")
        command("systemctl", "--user", "enable", "--now", SERVICE)
        command("systemctl", "--user", "restart", SERVICE)
        command("systemctl", "--user", "is-active", "--quiet", SERVICE)
        health = wait_healthy(manifest)
    except Exception:
        if previous:
            switch_current(previous, token)
        elif (ROOT / "current").is_symlink():
            (ROOT / "current").unlink()
        if old_serve:
            atomic_copy(backup / "serve.py", ROOT / "serve.py", token)
        elif (ROOT / "serve.py").is_file():
            (ROOT / "serve.py").unlink()
        if old_unit:
            atomic_copy(backup / "unit.service", UNIT, token)
            command("systemctl", "--user", "daemon-reload")
            command("systemctl", "--user", "restart", SERVICE)
        else:
            subprocess.run(["systemctl", "--user", "disable", "--now", SERVICE], check=False, capture_output=True)
            UNIT.unlink(missing_ok=True)
            command("systemctl", "--user", "daemon-reload")
        raise
    return {"deployed": True, "release": release, "previous_current": previous,
            "url": manifest["url"], "service": SERVICE, "uploaded_sha256_verified": True,
            "activation_health": health}


def main():
    phase, release, token = sys.argv[1:4]
    checked_name(release)
    if phase == "prepare":
        result = prepare(release, token)
    elif phase == "activate":
        result = activate(release, token, sys.argv[4], sys.argv[5])
    elif phase == "unlock":
        scoped_root()
        assert_lock(token)
        (LOCK / "owner").unlink()
        LOCK.rmdir()
        result = {"unlocked": True}
    else:
        raise ValueError("Unknown deployment phase")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
