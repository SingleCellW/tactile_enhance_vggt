"""Local-only snapshot and resource-link validation. Uses the Python stdlib."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

ASSET = re.compile(r"[\"'`]([^\"'`<>\s]+\.(?:html?|css|m?js|json|csv|tsv|png|jpe?g|svg|webp|gif|mp4|webm|pdf|zip|woff2?)(?:[?#][^\"'`<>\s]*)?)[\"'`]", re.I)
CSS_URL = re.compile(r"url\(\s*[\"']?([^\s\)\"']+)|@import\s+[\"']([^\"']+)", re.I)
JS_URL = re.compile(r"(?:fetch|import|URL|Worker)\s*\(\s*[\"'`]([^\"'`]+)[\"'`]|(?:from|import)\s+[\"']([^\"']+)[\"']")
JSON_URL_KEYS = {"href", "src", "poster", "url", "public_path", "download", "video", "image"}
WEB_PREFIXES = ("/reports/", "/data/", "/assets/", "/media/", "/images/", "/downloads/")
# This archived catalog is consumed by its report document. Its existing
# assets/... URLs intentionally use the document base, not the JSON directory.
JSON_DOCUMENT_BASES = {
    "reports/experiment_review_20261019/assets/media_manifest.json": Path("reports/experiment_review_20261019"),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or "\x00" in value or
            any(part.startswith(".") for part in value.split("/"))):
        raise ValueError("Unsafe snapshot path: " + str(value))
    return path


def public_files(public):
    public = Path(public)
    if public.is_symlink() or not public.is_dir():
        raise ValueError("Public root must be a real directory")
    result = []
    for path in sorted(public.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are never published: " + str(path))
        relative = str(path.relative_to(public))
        safe_relative(relative)
        if path.is_file():
            result.append((relative, path))
        elif not path.is_dir():
            raise ValueError("Only regular files/directories can be published: " + str(path))
    return result


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src", "poster") and value:
                self.links.append(value)
            elif key == "srcset" and value and not value.startswith("data:"):
                self.links.extend(item.strip().split()[0] for item in value.split(",") if item.strip())
            elif key == "style" and value:
                self.links.extend(a or b for a, b in CSS_URL.findall(value))


def local_target(public, parent, link):
    """Resolve browser-style root/relative URLs while prohibiting root escape."""
    url = urlsplit(link)
    if url.scheme or url.netloc or not url.path or "${" in link:
        return None
    value = unquote(url.path)
    if "\x00" in value or "\\" in value:
        raise ValueError("Malformed local URL: " + link)
    root = Path(public).resolve()
    path = (root / value.lstrip("/") if value.startswith("/") else root / parent / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError("URL escapes public root: " + link)
    relative = path.relative_to(root)
    if any(part.startswith(".") for part in relative.parts):
        raise ValueError("Hidden resource URL: " + link)
    return path / "index.html" if path.is_dir() else path


def json_urls(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in JSON_URL_KEYS and isinstance(item, str):
                # Provenance can contain filesystem paths and run URLs. Only
                # explicit web URLs/relative asset paths are browser resources.
                if not item.startswith("/") or item.startswith(WEB_PREFIXES) or item == "/":
                    yield item
            else:
                yield from json_urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_urls(item)


def validate_links(public):
    files = public_files(public)
    public = Path(public).resolve()
    references, html_contexts = [], {}
    for rel, path in files:
        if path.suffix.lower() in (".html", ".htm"):
            parser = Links()
            parser.feed(path.read_text(encoding="utf-8"))
            for link in parser.links:
                references.append((rel, [Path(rel).parent], link))
                try:
                    target = local_target(public, Path(rel).parent, link)
                except ValueError:
                    target = None  # The validation pass adds source context.
                if target is not None and target.suffix in (".js", ".mjs"):
                    html_contexts.setdefault(target, set()).add(Path(rel).parent)
    for rel, path in files:
        suffix, text = path.suffix.lower(), None
        if suffix in (".css", ".js", ".mjs", ".json"):
            text = path.read_text(encoding="utf-8")
        if suffix == ".css":
            references.extend((rel, [Path(rel).parent], a or b) for a, b in CSS_URL.findall(text))
        elif suffix in (".js", ".mjs"):
            contexts = [Path(rel).parent, *html_contexts.get(path.resolve(), ()), Path()]
            links = {a or b for a, b in JS_URL.findall(text)} | set(ASSET.findall(text))
            references.extend((rel, contexts, link) for link in sorted(links) if not link.startswith(("/home/", "/tmp/")))
        elif suffix == ".json":
            parents = [JSON_DOCUMENT_BASES[rel]] if rel in JSON_DOCUMENT_BASES else [Path(rel).parent, Path()]
            references.extend((rel, parents, link) for link in json_urls(json.loads(text)))
    checked, skipped = 0, 0
    for rel, parents, link in references:
        targets, errors = [], []
        for parent in parents:
            try:
                targets.append(local_target(public, parent, link))
            except ValueError as error:
                errors.append(str(error))
        if not targets:
            raise ValueError("Invalid published resource in %s: %s (%s)" % (rel, link, "; ".join(errors)))
        if all(target is None for target in targets):
            skipped += 1
            continue
        if not any(target is not None and target.is_file() for target in targets):
            raise FileNotFoundError("Broken published resource in %s: %s" % (rel, link))
        checked += 1
    return {"internal_references_checked": checked, "external_or_dynamic_references_skipped": skipped,
            "scope": "static HTML/CSS/JS and explicit JSON URL fields; dynamic URL behavior requires browser verification"}


def validate_bundle(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    actual = dict(public_files(bundle / "public"))
    if not (bundle / "public/index.html").is_file():
        raise ValueError("Portal index.html is required")
    declared = {}
    for entry in manifest["files"]:
        name = entry["path"]
        safe_relative(name)
        if name in declared:
            raise ValueError("Duplicate manifest file: " + name)
        declared[name] = entry
    if set(actual) != set(declared):
        raise ValueError("Manifest file set differs from snapshot")
    for name, path in actual.items():
        if path.stat().st_size != declared[name]["size"] or sha256(path) != declared[name]["sha256"]:
            raise ValueError("Snapshot size/SHA mismatch: " + name)
    runtime = dict(public_files(bundle / "runtime"))
    declared_runtime = {entry["path"]: entry for entry in manifest["runtime_files"]}
    if (set(runtime) != {"serve.py", "tactile-vggt-report.service", "README.md"} or
            set(runtime) != set(declared_runtime) or len(declared_runtime) != len(manifest["runtime_files"])):
        raise ValueError("Runtime file set mismatch")
    for name, path in runtime.items():
        entry = declared_runtime[name]
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError("Runtime size/SHA mismatch: " + name)
    return manifest, validate_links(bundle / "public")
