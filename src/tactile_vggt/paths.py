"""Portable locations for dependencies, machine inputs and generated artifacts.

Importing this module never creates directories or requires private data. CLI
arguments take precedence by using these helpers only for their defaults.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

def _project_root():
    explicit = os.environ.get("TACTILE_VGGT_PROJECT_ROOT")
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser().absolute()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src/tactile_vggt").is_dir():
        return checkout
    # A wheel contains modules, not the research checkout. Never place results
    # inside site-packages: an explicit root or the caller's cwd owns artifacts.
    return Path.cwd()


PROJECT_ROOT = _project_root()


def _absolute(value, base=None):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT if base is None else base) / path


def artifact_root():
    """Return the generated-file root, without creating it."""
    return _absolute(os.environ.get("TACTILE_VGGT_ARTIFACT_ROOT", PROJECT_ROOT / ".local"))


def artifact_path(kind, *parts):
    """Join a named artifact category; historical outputs map to runs."""
    aliases = {"outputs": "runs", "inputs": "data"}
    category = aliases.get(kind, kind)
    if category not in {"runs", "data", "cache", "artifacts", "reports", "weights",
                        "logs", "site", "deploy", "tmp", "qa"}:
        raise ValueError(f"Unknown artifact category: {kind}")
    return artifact_root().joinpath(category, *parts)


def dependency_path(name):
    """Return a local link to a version-pinned shared checkout."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Dependency name must be a single path component")
    return _absolute(os.environ.get("TACTILE_VGGT_DEPENDENCY_ROOT", PROJECT_ROOT / "external")) / name


def resolve_artifact_reference(value):
    """Read an old project-relative artifact reference without rewriting it.

    Existing paths always win. Only this checkout's old category paths are
    translated; paths belonging to another machine/project are never guessed.
    """
    path = Path(value).expanduser()
    original = path if path.is_absolute() else PROJECT_ROOT / path
    if original.exists():
        return original
    try:
        relative = original.relative_to(PROJECT_ROOT)
    except ValueError:
        return original
    if relative.parts and relative.parts[0] in {"outputs", "cache", "artifacts", "reports", "weights", "inputs", "logs"}:
        return artifact_path(relative.parts[0], *relative.parts[1:])
    return original


def machine_config_path():
    return _absolute(os.environ.get("TACTILE_VGGT_MACHINE_CONFIG",
                                    PROJECT_ROOT / "configs/machines/local.json"))


def machine_path(name, default=None, *, required=False):
    """Resolve environment > machine JSON > default, without checking existence.

    Relative JSON paths are relative to the JSON file, so the same configuration
    can be used from a shell, a scheduler and an IDE regardless of working dir.
    Empty/null settings count as absent. Parsing errors are never suppressed.
    """
    variable = "TACTILE_VGGT_" + name.upper()
    value = os.environ.get(variable)
    if value:
        return _absolute(value)
    config_file = machine_config_path()
    if config_file.is_file():
        config = json.loads(config_file.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Machine configuration must be a JSON object: {config_file}")
        value = config.get(name)
        if value:
            return _absolute(value, config_file.parent)
    if default is not None:
        return _absolute(default)
    if required:
        raise ValueError(f"Missing {name}: pass --{name.replace('_', '-')}, set {variable}, "
                         f"or configure {config_file}")
    return None


def require_machine_arguments(args, *names):
    """Resolve and validate simulator inputs just before executing a CLI."""
    for name in names:
        value = getattr(args, name, None)
        path = Path(value) if value else machine_path(name, required=True)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}; pass --{name.replace('_', '-')} "
                                    f"or set TACTILE_VGGT_{name.upper()}")
        setattr(args, name, path)
    return args
