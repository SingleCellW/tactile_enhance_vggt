from types import SimpleNamespace

import doctor


def test_core_check_does_not_require_private_assets_or_external_sources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Core checks must not require project assets")
    monkeypatch.setattr(doctor, "machine_path", forbidden)
    monkeypatch.setattr(doctor, "dependency_path", forbidden)
    result = doctor.check_environment("core")
    assert result["ok"]
    assert [check["name"] for check in result["checks"]] == ["python", "numpy"]


def test_training_check_rejects_an_unrelated_installed_lerobot(tmp_path, monkeypatch):
    checkout = tmp_path / "shared"
    checkout.mkdir()
    link = tmp_path / "lerobot"
    link.symlink_to(checkout, target_is_directory=True)
    monkeypatch.setattr(doctor, "dependency_path", lambda name: link)
    monkeypatch.setattr(doctor, "verify_checkout", lambda *args: checkout)
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name:
        SimpleNamespace(__version__="test", __file__=str(tmp_path / "unrelated/site-packages/lerobot/__init__.py")))
    result = doctor.check_environment("train")
    item = next(item for item in result["checks"] if item["name"] == "lerobot_import")
    assert not item["ok"]
    assert "expected locked checkout" in item["detail"]
    assert not result["ok"]


def test_missing_simulation_settings_are_all_reported(monkeypatch):
    def missing(name, **kwargs):
        raise ValueError(f"Missing {name}")
    monkeypatch.setattr(doctor, "machine_path", missing)
    result = doctor.check_environment("simulation")
    failures = {item["name"] for item in result["checks"] if not item["ok"]}
    assert failures == {"dynamic_assets", "raw_root", "timestamps_root", "source_readouts", "worker_environment"}
    assert not result["ok"]
