"""Explicit machine configuration for the portal's single owned service."""
import ipaddress
import json
import os
from pathlib import Path
import re

SERVICE = "tactile-vggt-report.service"
DIRECTORY = "tactile-vggt-report-host"


def load_config(path=None):
    selected = path or os.environ.get("TACTILE_VGGT_PORTAL_CONFIG")
    config = json.loads(Path(selected).expanduser().read_text()) if selected else {}
    allowed = {"remote_host", "remote_home", "bind", "port", "url"}
    if set(config) - allowed:
        raise ValueError("Unknown portal configuration fields: " + str(sorted(set(config) - allowed)))
    host, home = config.get("remote_host"), config.get("remote_home")
    if host is not None and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
        raise ValueError("remote_host must be an SSH hostname or alias")
    if home is not None:
        value = Path(home)
        if not value.is_absolute() or value == Path("/") or ".." in value.parts or any(c.isspace() for c in home):
            raise ValueError("remote_home must be an absolute account home without whitespace")
    bind, port = config.get("bind", "127.0.0.1"), config.get("port", 8766)
    ipaddress.IPv4Address(bind)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("port must be an unprivileged TCP port")
    url = config.get("url", f"http://{bind}:{port}/")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("url must be an HTTP(S) URL")
    return {"remote": host, "remote_directory": str(Path(home) / DIRECTORY) if home else None,
            "remote_home": home, "service": SERVICE, "bind": bind, "port": port, "url": url}


def require_remote(config):
    if not config["remote"] or not config["remote_home"]:
        raise ValueError("Deployment requires --config or TACTILE_VGGT_PORTAL_CONFIG with remote_host and remote_home")
