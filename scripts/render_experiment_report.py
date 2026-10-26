#!/usr/bin/env python3
"""Assemble the offline HTML review from verified, existing experiment records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiment_review_20261019"


def embedded(path: Path) -> str:
    value = json.loads(path.read_text())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def main() -> None:
    template = (REPORT / "report.template.html").read_text()
    replacements = {
        "__REPORT_DATA__": embedded(REPORT / "data.json"),
        "__LEGACY_DATA__": embedded(REPORT / "legacy.json"),
        "__MEDIA_DATA__": embedded(REPORT / "assets/media_manifest.json"),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Expected one template marker: {marker}")
        template = template.replace(marker, value)
    output = REPORT / "index.html"
    output.write_text(template)
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
