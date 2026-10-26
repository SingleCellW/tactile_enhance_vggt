"""Build the maintained catalog into a fresh local artifact directory.

Existing reports are never removed or overwritten. Use a new output directory
to preview a catalog; pass an explicit reports registry to include real reports.
"""
import argparse
import json
from pathlib import Path
import shutil

from tactile_vggt.paths import artifact_path

try:
    from .common import public_files, validate_links
except ImportError:
    from common import public_files, validate_links

SOURCE = Path(__file__).resolve().parents[1] / "portal"


def build_site(output=None, registry=None, report_source=None):
    output = Path(output or artifact_path("deploy", "site-preview"))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new site output directory; existing reports are preserved: " + str(output))
    prepared = []
    if report_source is not None:
        report_source = Path(report_source)
        prepared = public_files(report_source)
        if report_source.resolve() in output.resolve().parents:
            raise ValueError("Site output must be outside its report input tree")
    output.mkdir(parents=True)
    for relative, path in prepared:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(SOURCE / "templates/index.html", output / "index.html")
    (output / "assets").mkdir(exist_ok=True)
    # A catalog-only preview does not require historic report scripts/data.
    for name in ("catalog.css", "catalog.js", "portal.css", "stage1b.css", "stage1b.js") if prepared else ("catalog.css", "catalog.js"):
        shutil.copyfile(SOURCE / "assets" / name, output / "assets" / name)
    registry = registry or (output / "data/reports.json" if (output / "data/reports.json").is_file() else None)
    data = json.loads(Path(registry).read_text()) if registry else {"reports": []}
    (output / "data").mkdir(exist_ok=True)
    (output / "data/reports.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    public_files(output)
    validate_links(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=artifact_path("deploy", "site-preview"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--report-source", type=Path, help="Explicit generated site whose reports and evidence are copied; source remains unchanged")
    args = parser.parse_args(argv)
    print(build_site(args.output, args.registry, args.report_source))


if __name__ == "__main__":
    main()
