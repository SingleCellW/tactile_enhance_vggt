#!/usr/bin/env python3
"""Range-download a public HF weight, verifying size and the published LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    path = Path(args.output)
    head = httpx.head(args.url, follow_redirects=False, timeout=30)
    head.raise_for_status() if head.status_code >= 400 else None
    size = int(head.headers["x-linked-size"])
    expected = head.headers["x-linked-etag"].strip('"')
    if len(expected) != 64:
        raise ValueError("No published LFS SHA256")
    parts = path.with_name(path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = (size + args.workers - 1) // args.workers

    def fetch(index):
        start = index * chunk_size
        end = min(size - 1, start + chunk_size - 1)
        dst = parts / f"{index:03d}"
        if dst.exists() and dst.stat().st_size == end - start + 1:
            return index
        for attempt in range(4):
            try:
                with httpx.stream("GET", args.url, headers={"Range": f"bytes={start}-{end}"},
                                  follow_redirects=True, timeout=120) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{size}":
                        raise ValueError(f"Invalid range response for part {index}")
                    with dst.open("wb") as target:
                        for block in response.iter_bytes(1024 * 1024):
                            target.write(block)
                if dst.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range response")
                return index
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    print(json.dumps({"size": size, "sha256": expected, "workers": args.workers}), flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for done in as_completed([pool.submit(fetch, i) for i in range(args.workers)]):
            print(json.dumps({"part_complete": done.result()}), flush=True)
    temporary = path.with_suffix(path.suffix + ".verified.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as target:
        for index in range(args.workers):
            with (parts / f"{index:03d}").open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
                    target.write(block)
    if digest.hexdigest() != expected or temporary.stat().st_size != size:
        raise ValueError("Published SHA256/size verification failed")
    temporary.replace(path)
    for part in parts.iterdir():
        part.unlink()
    parts.rmdir()
    path.with_suffix(".provenance.json").write_text(json.dumps({
        "url": args.url, "size": size, "sha256": expected,
        "repository_commit": head.headers.get("x-repo-commit")}, indent=2) + "\n")
    print(json.dumps({"verified": str(path), "sha256": expected}), flush=True)


if __name__ == "__main__":
    main()
