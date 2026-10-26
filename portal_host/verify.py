#!/usr/bin/env python3
"""Validate a snapshot, every HTTP file SHA/length, and media seeking ranges."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

try:
    from .common import sha256, validate_bundle
    from .config import load_config, require_remote
except ImportError:
    from common import sha256, validate_bundle
    from config import load_config, require_remote

HERE = Path(__file__).resolve().parent


def verify_http(manifest, url):
    base = url.rstrip("/") + "/"
    opener = build_opener(ProxyHandler({}))
    def request(path, method="GET", headers=None, encoded=False):
        return opener.open(Request(base + quote(path, safe="/%" if encoded else "/"), method=method, headers=headers or {}), timeout=30)
    hashed, ranged = 0, 0
    for entry in manifest["files"]:
        with request(entry["path"], method="HEAD") as response:
            if response.status != 200 or int(response.headers["Content-Length"]) != entry["size"]:
                raise ValueError("HTTP HEAD size/status mismatch: " + entry["path"])
        value, size, leading, trailing = hashlib.sha256(), 0, b"", b""
        with request(entry["path"]) as response:
            if response.status != 200:
                raise ValueError("HTTP GET status mismatch: " + entry["path"])
            for block in iter(lambda: response.read(1024 * 1024), b""):
                if not size:
                    leading = block[:64]
                trailing = (trailing + block)[-8:]
                value.update(block)
                size += len(block)
        if size != entry["size"] or value.hexdigest() != entry["sha256"]:
            raise ValueError("HTTP GET size/SHA mismatch: " + entry["path"])
        hashed += 1
        if Path(entry["path"]).suffix.lower() in (".mp4", ".webm", ".zip", ".pdf") and size:
            length = min(64, size)
            with request(entry["path"], headers={"Range": "bytes=0-%d" % (length - 1)}) as response:
                if (response.status != 206 or response.read() != leading[:length] or
                        response.headers.get("Content-Range") != "bytes 0-%d/%d" % (length - 1, size)):
                    raise ValueError("HTTP leading range mismatch: " + entry["path"])
            length = min(8, size)
            with request(entry["path"], headers={"Range": "bytes=-%d" % length}) as response:
                if (response.status != 206 or response.read() != trailing[-length:] or
                        response.headers.get("Content-Range") != "bytes %d-%d/%d" % (size - length, size - 1, size)):
                    raise ValueError("HTTP suffix range mismatch: " + entry["path"])
            try:
                with request(entry["path"], headers={"Range": "bytes=%d-" % size}):
                    raise ValueError("Unsatisfiable byte range was accepted")
            except HTTPError as error:
                if error.code != 416:
                    raise
            ranged += 3
    paths = [".git/config", "%2e%2e/.ssh/id_rsa", ".deployment-lock/owner", "__unlisted_directory__/"]
    names = {entry["path"] for entry in manifest["files"]}
    no_index = sorted({str(Path(name).parent) + "/" for name in names if str(Path(name).parent) != "." and str(Path(name).parent / "index.html") not in names})
    if no_index:
        paths.append(no_index[0])
    for path in paths:
        try:
            with request(path, encoded=True):
                raise ValueError("Private/absent path exposed: " + path)
        except HTTPError as error:
            if error.code != 404:
                raise
    return {"status": "passed", "url": base, "assets_head_checked": hashed,
            "assets_sha256_checked": hashed, "range_requests_checked": ranged,
            "bytes_sha256_checked": sum(entry["size"] for entry in manifest["files"]),
            "private_or_unlisted_paths_rejected": len(paths), "directory_listing_rejection_checked": bool(no_index)}


def via_ssh(manifest, url, config=None):
    config = config or load_config()
    require_remote(config)
    # Only stdlib definitions run remotely. The manifest travels over stdin,
    # never through shell interpolation or a very large command-line argument.
    import inspect
    program = ("import hashlib,json,sys\nfrom pathlib import Path\n"
               "from urllib.error import HTTPError\nfrom urllib.parse import quote\n"
               "from urllib.request import ProxyHandler,Request,build_opener\n" +
               inspect.getsource(verify_http) +
               "\nvalue=json.load(sys.stdin)\nprint(json.dumps(verify_http(value['manifest'],value['url'])))\n")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config["remote"],
                             "/usr/bin/python3 -c " + shlex.quote(program)],
                            input=json.dumps({"manifest": manifest, "url": url}),
                            capture_output=True, text=True, check=True, timeout=1200)
    return json.loads(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--url", help="Default: the selected bundle URL")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--local-only", action="store_true", help="No HTTP, SSH or service actions")
    parser.add_argument("--via-ssh", action="store_true", help="Read HTTP through the host account when campus routing requires SSH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest, links = validate_bundle(args.bundle)
    args.url = args.url or manifest["url"]
    if args.local_only and args.via_ssh:
        parser.error("--local-only and --via-ssh are mutually exclusive")
    result = {"status": "passed_local_only" if args.local_only else "pending_http",
              "release": manifest["release"], "manifest_sha256": sha256(args.bundle / "manifest.json"),
              "local_assets_checked": len(manifest["files"]), "links": links,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    if not args.local_only:
        result.update(via_ssh(manifest, args.url, load_config(args.config)) if args.via_ssh else verify_http(manifest, args.url))
        result["http_check_location"] = "SSH host account" if args.via_ssh else "current machine"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
