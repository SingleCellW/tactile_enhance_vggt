#!/usr/bin/env python3
"""Static snapshots with single-byte-range support; no automatic file listing."""
import argparse
import functools
import mimetypes
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


class ReportHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def list_directory(self, path):
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def send_head(self):
        self.byte_range = None
        url = urlsplit(self.path)
        root = Path(self.directory).resolve()
        decoded = unquote(url.path)
        if ("\x00" in decoded or "\\" in decoded or
                any(part.startswith(".") for part in decoded.split("/") if part)):
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        path = (root / decoded.lstrip("/")).resolve()
        if path != root and root not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        if path.is_dir():
            if not url.path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", url.path + "/" + ("?" + url.query if url.query else ""))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            path = (path / "index.html").resolve()
            if path != root and root not in path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        try:
            handle = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        size, start = path.stat().st_size, 0
        end = size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            try:
                if not match or not any(match.groups()):
                    raise ValueError()
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    raise ValueError()
            except ValueError:
                handle.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            self.byte_range = (start, end)
            handle.seek(start)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in (".py", ".rst", ".md"):
            content_type = "text/plain; charset=utf-8"
        self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if requested:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        return handle

    def copyfile(self, source, outputfile):
        remaining = self.byte_range[1] - self.byte_range[0] + 1 if self.byte_range else None
        try:
            while remaining is None or remaining > 0:
                data = source.read(min(65536, remaining) if remaining is not None else 65536)
                if not data:
                    break
                outputfile.write(data)
                if remaining is not None:
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.bind, args.port), functools.partial(ReportHandler, directory=args.directory)) as server:
        print("Serving http://%s:%s/" % (args.bind, args.port), flush=True)
        server.serve_forever()
