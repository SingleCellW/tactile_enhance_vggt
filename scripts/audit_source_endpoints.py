#!/usr/bin/env python3
"""CPU-only recorded source endpoint audit; no policy, simulator or filtering.

The terminal predicate uses the historical cube-center goal rectangle. A single
qpos endpoint cannot establish low velocity, release, sustained stability, or
strict rollout success. All source episodes are retained unchanged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, machine_path, require_machine_arguments


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def legacy_goal(xyz):
    return bool(.4 <= xyz[0] <= .6 and -.39 <= xyz[1] <= -.21 and xyz[2] <= .1)


def audit(args):
    sources = {}
    def source(path):
        path = Path(path).resolve(strict=True)
        sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
        return path
    metadata = json.loads(source(args.cache / "manifest.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("Original cache must be complete")
    episodes = np.load(source(args.cache / "episode_index.npy"), allow_pickle=False)
    frames = np.load(source(args.cache / "frame_index.npy"), allow_pickle=False)
    raw_episodes = sorted(path for path in args.raw_root.iterdir() if (path / "prioception").exists())
    train, val = [set(map(int, metadata["split"][name])) for name in ("train", "val")]
    if (len(raw_episodes) != 53 or len(train) != 48 or len(val) != 5 or train & val
            or train | val != set(range(53)) or set(map(int, np.unique(episodes))) != train | val
            or frames.shape != episodes.shape):
        raise ValueError("Require the original53-episode source with disjoint48/5 split")
    rows = []
    for episode, raw in enumerate(raw_episodes):
        csv_path = source(args.timestamps_root / f"teleop_log_{raw.name}.csv")
        with csv_path.open() as stream:
            records = list(csv.DictReader(stream))
        indices = np.flatnonzero(episodes == episode)
        if (not records or len(records) != len(indices)
                or not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
                or not np.array_equal(frames[indices], np.arange(len(records)))):
            raise ValueError(f"Raw CSV/cache episode length or ordering differs: {episode}")
        qpaths = [source(raw / records[index]["prioception_path"]) for index in (0, -1)]
        endpoints = [np.load(path, allow_pickle=False) for path in qpaths]
        if any(q.shape != (29,) or not np.isfinite(q).all() for q in endpoints):
            raise ValueError(f"Need finite first/last qpos29: episode{episode}")
        initial, terminal = [q[22:25] for q in endpoints]
        rows.append({"episode": episode, "raw_episode": raw.name, "split": "train" if episode in train else "val",
            "frames": len(records), "first_frame_index": 0, "terminal_frame_index": len(records) - 1,
            "initial_cube_xyz": initial.tolist(), "terminal_cube_xyz": terminal.tolist(),
            "initial_legacy_goal_predicate": legacy_goal(initial), "terminal_legacy_goal_predicate": legacy_goal(terminal),
            "timestamps_path": str(csv_path), "timestamps_sha256": digest(csv_path),
            "initial_qpos_path": str(qpaths[0]), "initial_qpos_sha256": digest(qpaths[0]),
            "terminal_qpos_path": str(qpaths[1]), "terminal_qpos_sha256": digest(qpaths[1])})
    summaries = {}
    for name in ("train", "val", "all"):
        selected = [row for row in rows if name == "all" or row["split"] == name]
        lengths = [row["frames"] for row in selected]
        goal_count = sum(row["terminal_legacy_goal_predicate"] for row in selected)
        summaries[name] = {"episodes": len(selected), "frames": sum(lengths),
            "terminal_legacy_goal_predicate_count": goal_count,
            "terminal_legacy_goal_predicate_fraction": goal_count / len(selected),
            "initial_legacy_goal_predicate_count": sum(row["initial_legacy_goal_predicate"] for row in selected),
            "terminal_predicate_false_episodes": [row["episode"] for row in selected if not row["terminal_legacy_goal_predicate"]],
            "episode_length_frames": {"minimum": min(lengths), "maximum": max(lengths),
                "mean": float(np.mean(lengths)), "median": float(np.median(lengths))}}
    source(__file__)
    return {"audit_version": 1, "complete": True,
        "generated_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "cache": str(args.cache.resolve()), "cache_manifest_sha256": digest(args.cache / "manifest.json"),
        "raw_root": str(args.raw_root.resolve()), "timestamps_root": str(args.timestamps_root.resolve()),
        "split_episodes": {"train": sorted(train), "val": sorted(val)}, "summary": summaries,
        "episodes": rows, "sources": sources,
        "definitions": {"episode_mapping": "same as worker: sorted raw directories containing prioception; checked against full-rate cache episode/frame indices and CSV lengths",
            "cube_xyz": "recorded raw qpos29[22:25], original cube free-joint translation",
            "legacy_terminal_predicate": "0.4<=x<=0.6 and -0.39<=y<=-0.21 and z<=0.1, evaluated only at the final recorded frame",
            "strict_success": "not evaluated; endpoints do not establish velocity, release, robot contact or one-second stability",
            "length": "CSV source record count, validated against full-rate cache frame count; not interpreted as wall-time duration",
            "purpose": "source quality/goal-coverage evidence for a possible future data review; no episodes filtered, labels changed, extra training or simulation run",
            "completion_scope": "all original53 source endpoints audited; does not signal completion of the separate policy evaluation"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=artifact_path("cache") / "pickplace_vggt_native_physics_dt01_stride4")
    parser.add_argument("--raw-root", type=Path, default=machine_path("raw_root"))
    parser.add_argument("--timestamps-root", type=Path, default=machine_path("timestamps_root", default=machine_path("timestamps_root", default=artifact_path("inputs") / "raw_timestamps")))
    parser.add_argument("--output", type=Path, default=artifact_path("reports") / "simple_policy_followup_20261025/source_endpoint_audit.json")
    args = parser.parse_args()
    require_machine_arguments(args, 'raw_root', 'timestamps_root')
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
