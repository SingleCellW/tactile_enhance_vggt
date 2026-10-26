"""A clean source checkout builds a catalog without private experiment data."""
import importlib.util
import json
from pathlib import Path

import pytest

from portal_host import build_site, common, config, deploy


def test_clean_catalog_build_is_local_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("TACTILE_VGGT_ARTIFACT_ROOT", str(tmp_path))
    site = build_site.build_site()
    assert site == tmp_path / "deploy/site-preview"
    assert json.loads((site / "data/reports.json").read_text()) == {"reports": []}
    assert common.validate_links(site)["internal_references_checked"] == 3
    bundle, manifest = deploy.build(site)
    assert bundle.parent == tmp_path / "deploy/bundles"
    assert manifest["remote"] is None
    assert common.validate_bundle(bundle)[0] == manifest
    (site / "index.html").write_text("preserve this draft")
    with pytest.raises(FileExistsError, match="existing reports are preserved"):
        build_site.build_site(site)
    assert (site / "index.html").read_text() == "preserve this draft"


def test_deployment_requires_explicit_machine_and_matches_bundle(tmp_path, monkeypatch):
    site = build_site.build_site(tmp_path / "site")
    machine = tmp_path / "portal.local.json"
    machine.write_text(json.dumps({"remote_host": "test-host", "remote_home": "/home/test",
                                   "bind": "127.0.0.1", "port": 8899}))
    bundle, manifest = deploy.build(site, tmp_path / "bundles", machine)
    assert manifest["remote_directory"] == "/home/test/tactile-vggt-report-host"
    assert "--bind 127.0.0.1 --port 8899" in (bundle / "runtime/tactile-vggt-report.service").read_text()
    monkeypatch.setattr(deploy, "remote", lambda *a, **k: pytest.fail("Mismatched config must never connect"))
    with pytest.raises(ValueError, match="not scoped"):
        deploy.deploy(bundle)
    with pytest.raises(ValueError, match="Deployment requires"):
        config.require_remote(config.load_config())


def test_catalog_can_reuse_explicit_report_artifacts_without_changing_them(tmp_path):
    old = tmp_path / "old-site"
    (old / "reports/example").mkdir(parents=True)
    (old / "data").mkdir()
    (old / "index.html").write_text("old home remains unchanged")
    (old / "reports/example/index.html").write_text("original experiment report")
    (old / "data/reports.json").write_text(json.dumps({"reports": [{"title": "Example", "url": "/reports/example/"}]}))
    new = build_site.build_site(tmp_path / "new-site", report_source=old)
    assert (old / "index.html").read_text() == "old home remains unchanged"
    assert (new / "reports/example/index.html").read_bytes() == (old / "reports/example/index.html").read_bytes()
    common.validate_links(new)
    with pytest.raises(ValueError, match="outside its report input"):
        build_site.build_site(old / "child", report_source=old)


def test_remote_config_cannot_change_owned_directory_or_service(tmp_path):
    machine = tmp_path / "portal.local.json"
    machine.write_text(json.dumps({"remote_directory": "/home/test/unrelated", "service": "other.service"}))
    with pytest.raises(ValueError, match="Unknown portal configuration"):
        config.load_config(machine)


def test_historical_run_relocation_keeps_recorded_hash_binding(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("stage1b_report", Path(__file__).resolve().parents[1] / "scripts/build_stage1b_report.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setenv("TACTILE_VGGT_ARTIFACT_ROOT", str(tmp_path))
    old = "/home/old/projects/tactile_enhance_vggt/outputs/campaign/s43/fixed/visual"
    directory = tmp_path / "runs/campaign/s43/fixed/visual"
    directory.mkdir(parents=True)
    names = ("config.json", "completion.json", "history.json", "metrics.json", "test_predictions.npz")
    sources = {}
    for name in names:
        path = directory / name
        path.write_bytes(b"original evidence")
        sources[old + "/" + name] = {"sha256": common.sha256(path), "bytes": path.stat().st_size}
    data = {"runs": [{"directory": old, "seed": 43, "regime": "fixed", "mode": "visual"}], "sources": sources}
    builder.prediction_archive(data, None, tmp_path / "predictions.zip")
    (directory / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="Recorded source size/SHA mismatch"):
        builder.prediction_archive(data, None, tmp_path / "invalid.zip")
