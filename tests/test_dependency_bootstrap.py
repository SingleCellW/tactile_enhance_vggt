"""Dependency setup must never silently change or replace a user's checkout."""
import json
import pytest

import bootstrap_dependencies as bootstrap


def checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bootstrap.git(source, "init", "--quiet")
    (source / "library.py").write_text("value = 1\n")
    bootstrap.git(source, "add", "library.py")
    bootstrap.git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "initial")
    return source, bootstrap.git(source, "rev-parse", "HEAD")


def test_local_link_is_repeatable_and_check_is_read_only(tmp_path):
    source, commit = checkout(tmp_path)
    spec = {"commit": commit, "url": "https://example.invalid/repo.git"}
    external, shared = tmp_path / "external", tmp_path / "shared"
    result = bootstrap.ensure_dependency("demo", spec, external, shared, local=source)
    assert result["status"] == "linked"
    link = external / "demo"
    assert link.is_symlink() and link.resolve() == source
    before = link.lstat().st_mtime_ns
    assert bootstrap.ensure_dependency("demo", spec, external, shared, check=True)["status"] == "verified"
    assert link.lstat().st_mtime_ns == before
    assert not shared.exists()


def test_wrong_commit_or_dirty_tree_never_gets_linked(tmp_path):
    source, commit = checkout(tmp_path)
    external = tmp_path / "external"
    with pytest.raises(ValueError, match="expected"):
        bootstrap.ensure_dependency("demo", {"commit": "0" * 40}, external, tmp_path, local=source)
    (source / "untracked.py").write_text("uncommitted work\n")
    with pytest.raises(ValueError, match="dirty"):
        bootstrap.ensure_dependency("demo", {"commit": commit}, external, tmp_path, local=source)
    assert not external.exists()
    assert (source / "untracked.py").read_text() == "uncommitted work\n"


def test_existing_directory_or_other_link_is_never_replaced(tmp_path):
    source, commit = checkout(tmp_path)
    external = tmp_path / "external"
    (external / "demo").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing"):
        bootstrap.ensure_dependency("demo", {"commit": commit}, external, tmp_path, local=source)
    (external / "demo").rmdir()
    (external / "demo").symlink_to(source)
    with pytest.raises(ValueError, match="elsewhere"):
        bootstrap.ensure_dependency("demo", {"commit": commit}, external, tmp_path, local=tmp_path / "other")
    assert (external / "demo").resolve() == source


def test_missing_dependency_check_and_setup_do_not_fetch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network fetch")
    monkeypatch.setattr(bootstrap, "fetch_checkout", forbidden)
    external, shared = tmp_path / "external", tmp_path / "shared"
    spec = {"commit": "1" * 40, "url": "https://example.invalid/repo.git"}
    for check in (True, False):
        with pytest.raises(ValueError, match="Missing"):
            bootstrap.ensure_dependency("demo", spec, external, shared, check=check)
    assert not external.exists() and not shared.exists()


def test_subdirectory_cannot_masquerade_as_checkout_root(tmp_path):
    source, commit = checkout(tmp_path)
    child = source / "subdir"
    child.mkdir()
    with pytest.raises(ValueError, match="checkout root"):
        bootstrap.verify_checkout(child, commit)


def test_lock_rejects_path_traversal_and_unpinned_revision(tmp_path):
    path = tmp_path / "lock.json"
    for name, commit in (("../escape", "1" * 40), ("demo", "main")):
        path.write_text(json.dumps({"format_version": 1, "git": {name: {"commit": commit, "url": "https://example.invalid/repo.git"}}}))
        with pytest.raises(ValueError):
            bootstrap.read_lock(path)
