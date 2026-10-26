"""CPU-only publication safeguards; no socket, SSH, rsync or service actions."""
import io
import json
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from portal_host import common, deploy, remote_install, serve, verify


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "site"
    files = {
        "index.html": '<link href="assets/site.css" rel="stylesheet"><script src="/assets/app.js"></script><a href="reports/stage1a/">Report</a>',
        "assets/site.css": 'body{background-image:url("../images/a.svg")}',
        "assets/app.js": 'fetch("/data/report.json");',
        "images/a.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
        "reports/stage1a/index.html": '<a href="../../">Home</a><video src="media/clip.mp4" poster="../../images/a.svg"></video>',
        "data/report.json": json.dumps({"video": "/reports/stage1a/media/clip.mp4", "provenance": {"path": "/home/user/private/result.json", "url": "/home/user/private/data.json"}}),
        "reports/stage1a/media/clip.mp4": "0123456789" * 20,
    }
    # Poster resolves from reports/stage1a/ up twice to the public images path.
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_build_links_hashes_immutable_and_default_never_deploys(source, tmp_path, monkeypatch):
    bundle, manifest = deploy.build(source, tmp_path / "bundles")
    assert manifest["links"]["internal_references_checked"] >= 8
    assert common.validate_bundle(bundle)[0] == manifest
    original = (bundle / "public/index.html").read_bytes()
    (source / "index.html").write_text("new draft")
    assert (bundle / "public/index.html").read_bytes() == original
    monkeypatch.setattr(deploy, "remote", lambda *args: pytest.fail("Default must never SSH"))
    assert deploy.main(["--bundle", str(bundle)])["deployed"] is False


def test_broken_resource_symlinks_and_tampering_rejected(source, tmp_path):
    bundle, _ = deploy.build(source, tmp_path / "bundles")
    (bundle / "runtime/serve.py").write_text("changed runtime")
    with pytest.raises(ValueError, match="Runtime size/SHA"):
        common.validate_bundle(bundle)
    (source / "images/a.svg").unlink()
    with pytest.raises(FileNotFoundError, match="Broken published resource"):
        deploy.build(source, tmp_path / "broken")
    (source / "images/a.svg").symlink_to(tmp_path / "outside-secret")
    with pytest.raises(ValueError, match="Symlinks"):
        deploy.build(source, tmp_path / "symlink")


def test_wrong_deployment_scope_never_connects(source, tmp_path, monkeypatch):
    bundle, manifest = deploy.build(source, tmp_path / "bundles")
    manifest["remote_directory"] = "/home/wukf/some-other-project"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(deploy, "remote", lambda *args: pytest.fail("Wrong scope must fail before SSH"))
    with pytest.raises(ValueError, match="not scoped"):
        deploy.deploy(bundle)


def test_legacy_catalog_explicit_document_base_without_skipping_links(source, tmp_path):
    report = source / "reports/experiment_review_20261019"
    (report / "assets").mkdir(parents=True)
    (report / "assets/example.mp4").write_bytes(b"legacy video")
    (report / "assets/media_manifest.json").write_text(json.dumps({"videos": [{"video": "assets/example.mp4"}]}))
    deploy.build(source, tmp_path / "valid")
    (report / "assets/example.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="media_manifest.json"):
        deploy.build(source, tmp_path / "invalid")


def test_valid_document_relative_json_is_not_rejected_by_root_fallback(source, tmp_path):
    artifact = source / "artifacts/example.json"
    artifact.parent.mkdir()
    artifact.write_text("{}")
    (source / "reports/stage1a/data.json").write_text(json.dumps({"href": "../../artifacts/example.json"}))
    deploy.build(source, tmp_path / "valid")
    (source / "index.html").write_text('<a href="../secret.json">invalid root escape</a>')
    with pytest.raises(ValueError, match="Invalid published resource in index.html"):
        deploy.build(source, tmp_path / "invalid")


class FakeHandler(serve.ReportHandler):
    def __init__(self, root, path, range_header=None):
        self.directory, self.path = str(root), path
        self.headers = {} if range_header is None else {"Range": range_header}
        self.out_headers, self.status, self.wfile = {}, None, io.BytesIO()

    def send_response(self, status, message=None):
        self.status = int(status)

    def send_header(self, name, value):
        self.out_headers[name] = value

    def send_error(self, status, message=None, explain=None):
        self.status = int(status)

    def end_headers(self):
        pass

    def run(self):
        handle = self.send_head()
        if handle:
            with handle:
                self.copyfile(handle, self.wfile)
        return self


@pytest.mark.parametrize("header,expected", [("bytes=2-5", b"2345"), ("bytes=-3", b"789"), ("bytes=8-", b"89")])
def test_single_ranges_stream_exact_bytes(tmp_path, header, expected):
    (tmp_path / "clip.mp4").write_bytes(b"0123456789")
    response = FakeHandler(tmp_path, "/clip.mp4", header).run()
    assert response.status == 206
    assert response.wfile.getvalue() == expected
    assert response.out_headers["Accept-Ranges"] == "bytes"
    assert int(response.out_headers["Content-Length"]) == len(expected)


def test_private_escape_directory_and_invalid_ranges_rejected(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"0123456789")
    (tmp_path / "assets").mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-secret")
    outside.write_text("not public")
    (tmp_path / "escape").symlink_to(outside)
    for path in ("/.git/config", "/%2e%2e/secret", "/escape", "/assets/", "/%00"):
        assert FakeHandler(tmp_path, path).run().status == 404
    (tmp_path / "assets/index.html").symlink_to(outside)
    assert FakeHandler(tmp_path, "/assets/").run().status == 404
    for value in ("bytes=10-", "bytes=2-1", "bytes=-0", "bytes=0-1,3-4"):
        assert FakeHandler(tmp_path, "/clip.mp4", value).run().status == 416


class Response(io.BytesIO):
    def __init__(self, status, headers, content):
        super().__init__(content)
        self.status, self.headers = status, headers


def test_full_http_verifier_without_network(source, tmp_path, monkeypatch):
    bundle, manifest = deploy.build(source, tmp_path / "bundles")
    class Opener:
        def open(self, request, timeout):
            path = urlsplit(request.full_url).path
            handler = FakeHandler(bundle / "public", path, request.headers.get("Range")).run()
            if handler.status >= 400:
                raise HTTPError(request.full_url, handler.status, "expected refusal", handler.out_headers, None)
            return Response(handler.status, handler.out_headers, b"" if request.method == "HEAD" else handler.wfile.getvalue())
    monkeypatch.setattr(verify, "build_opener", lambda *args: Opener())
    report = verify.verify_http(manifest, "http://unused/")
    assert report["status"] == "passed"
    assert report["assets_sha256_checked"] == len(manifest["files"])
    assert report["range_requests_checked"] == 3
    assert report["directory_listing_rejection_checked"]


@pytest.fixture
def remote_env(tmp_path, monkeypatch):
    home, commands = tmp_path / "home", []
    home.mkdir()
    root = home / "tactile-vggt-report-host"
    monkeypatch.setattr(remote_install, "ROOT", root)
    monkeypatch.setattr(remote_install, "UNIT", home / ".config/systemd/user/tactile-vggt-report.service")
    monkeypatch.setattr(remote_install, "MARKER", root / ".portal-host.json")
    monkeypatch.setattr(remote_install, "LOCK", root / ".deployment-lock")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(remote_install.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(remote_install, "port_free", lambda: commands.append(("check_port_free",)))
    def command(*args):
        commands.append(args)
        return "yes" if args[0] == "loginctl" else ""
    monkeypatch.setattr(remote_install, "command", command)
    monkeypatch.setattr(remote_install, "wait_healthy", lambda manifest: {"unit_active": True, "mock": True})
    return root, commands


def upload_fixture(bundle, root, incoming):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["remote_directory"] = str(root)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    for name in ("public", "runtime"):
        shutil.copytree(bundle / name, Path(incoming) / name)
    shutil.copyfile(bundle / "manifest.json", Path(incoming) / "manifest.json")
    return common.sha256(bundle / "manifest.json")


def unlock_fixture(token):
    remote_install.assert_lock(token)
    (remote_install.LOCK / "owner").unlink()
    remote_install.LOCK.rmdir()


def test_remote_first_install_verifies_upload_then_updates_only_own_service(source, tmp_path, remote_env):
    root, commands = remote_env
    bundle, manifest = deploy.build(source, tmp_path / "bundles")
    token, release = "a" * 32, manifest["release"]
    preparation = remote_install.prepare(release, token)
    assert preparation["previous_current"] is None and commands[0] == ("check_port_free",)
    sha = upload_fixture(bundle, root, preparation["incoming"])
    with pytest.raises(ValueError, match="manifest SHA"):
        remote_install.activate(release, token, "-", "0" * 64)
    assert not (root / "current").exists()
    result = remote_install.activate(release, token, "-", sha)
    assert result["uploaded_sha256_verified"]
    assert (root / "current").resolve() == root / "releases" / release
    service_commands = [cmd for cmd in commands if cmd[0] == "systemctl" and "daemon-reload" not in cmd]
    assert service_commands and all(cmd[-1] == "tactile-vggt-report.service" for cmd in service_commands)
    unlock_fixture(token)


def test_remote_occupied_port_stops_before_directory_creation(remote_env, monkeypatch):
    root, _ = remote_env
    def busy():
        raise OSError("port in use")
    monkeypatch.setattr(remote_install, "port_free", busy)
    with pytest.raises(OSError, match="port in use"):
        remote_install.prepare("portal-test", "b" * 32)
    assert not root.exists()


def test_remote_failed_update_restores_previous_release(source, tmp_path, remote_env, monkeypatch):
    root, commands = remote_env
    first, first_manifest = deploy.build(source, tmp_path / "bundles")
    token = "c" * 32
    prepared = remote_install.prepare(first_manifest["release"], token)
    sha = upload_fixture(first, root, prepared["incoming"])
    remote_install.activate(first_manifest["release"], token, "-", sha)
    unlock_fixture(token)
    second, second_manifest = deploy.build(source, tmp_path / "bundles")
    prepared = remote_install.prepare(second_manifest["release"], token)
    sha = upload_fixture(second, root, prepared["incoming"])
    original, fail_once = remote_install.command, [True]
    def fail_restart(*args):
        if args[:3] == ("systemctl", "--user", "restart") and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("simulated new-service activation failure")
        return original(*args)
    monkeypatch.setattr(remote_install, "command", fail_restart)
    with pytest.raises(RuntimeError, match="simulated"):
        remote_install.activate(second_manifest["release"], token, prepared["previous_current"], sha)
    assert (root / "current").resolve() == root / "releases" / first_manifest["release"]
    assert (root / "releases" / first_manifest["release"]).is_dir()
    assert (root / "backups" / second_manifest["release"] / "unit.service").is_file()
    unlock_fixture(token)


def test_http_health_failure_restores_previous_release(source, tmp_path, remote_env, monkeypatch):
    root, _ = remote_env
    first, manifest = deploy.build(source, tmp_path / "bundles")
    token = "d" * 32
    prepared = remote_install.prepare(manifest["release"], token)
    sha = upload_fixture(first, root, prepared["incoming"])
    remote_install.activate(manifest["release"], token, "-", sha)
    unlock_fixture(token)
    second, next_manifest = deploy.build(source, tmp_path / "bundles")
    prepared = remote_install.prepare(next_manifest["release"], token)
    sha = upload_fixture(second, root, prepared["incoming"])
    def bad_health(manifest):
        raise RuntimeError("HTTP registry hash does not match")
    monkeypatch.setattr(remote_install, "wait_healthy", bad_health)
    with pytest.raises(RuntimeError, match="HTTP registry hash"):
        remote_install.activate(next_manifest["release"], token, prepared["previous_current"], sha)
    assert (root / "current").resolve() == root / "releases" / manifest["release"]
    assert remote_install.UNIT.read_bytes() == (root / "backups" / next_manifest["release"] / "unit.service").read_bytes()
    unlock_fixture(token)
