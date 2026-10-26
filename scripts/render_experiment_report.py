#!/usr/bin/env python3
"""Assemble the offline HTML review from verified, existing experiment records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile_vggt.paths import artifact_path

ROOT = Path(__file__).resolve().parents[1]
REPORT = artifact_path('reports', 'experiment_review_20261019')


def embedded(path: Path) -> str:
    value = json.loads(path.read_text())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT, help="Artifact directory containing original template and audited JSON")
    parser.add_argument("--output", type=Path, help="Default: <report>/index.html; use a fresh path for a preview")
    args = parser.parse_args(argv)
    template = (args.report / "report.template.html").read_text()
    replacements = {
        "__REPORT_DATA__": embedded(args.report / "data.json"),
        "__LEGACY_DATA__": embedded(args.report / "legacy.json"),
        "__MEDIA_DATA__": embedded(args.report / "assets/media_manifest.json"),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Expected one template marker: {marker}")
        template = template.replace(marker, value)
    output = args.output or args.report / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template)
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
