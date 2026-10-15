#!/usr/bin/env python3
"""Create a separate cache with synchronized native tactile, retaining VGGT pixels.

Requires independently regenerated full-rate readouts. Never relabel old forces
by moving their anchors: both force and position are replaced together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.data import training_statistics
from tactile_vggt.geometry import world_to_base


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def repair(source, readouts, report, destination, rgb_lag=0, rgb_alignment_report=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new destination: {destination}")
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Source geometry cache is incomplete")
    audit = json.loads(Path(report).read_text())
    if not audit.get("complete"):
        raise ValueError("Native tactile regeneration report is incomplete")
    readouts_fingerprint = fingerprint(readouts)
    if audit.get("repaired_readouts_sha256") != readouts_fingerprint["sha256"]:
        raise ValueError("Native tactile report does not match the supplied readouts SHA256")
    if rgb_lag not in (0, 1):
        raise ValueError("Only explicitly audited RGB lag 0 or 1 is supported")
    alignment = None
    if rgb_alignment_report is not None:
        alignment = json.loads(Path(rgb_alignment_report).read_text())
    if rgb_lag and not (alignment and alignment.get("verified") is True
                        and alignment.get("rgb_lag_frames") == rgb_lag):
        raise ValueError("RGB lag correction requires a matching verified rendering audit")
    arrays = {key: np.load(source / f"{key}.npy", mmap_mode="r") for key in
              ("state", "action", "episode_index", "frame_index", "obs_frame_indices")}
    n = len(arrays["state"])
    if audit.get("frames") != n or audit.get("source_frames") != n:
        raise ValueError("Native tactile report does not cover every source frame")
    with np.load(readouts, allow_pickle=False) as native:
        for key, source_key in (("state22", "state"), ("episode_index", "episode_index"),
                                ("frame_index", "frame_index")):
            np.testing.assert_array_equal(native[key], arrays[source_key], err_msg=f"Misaligned {key}")
        force, anchors = native["force44"].copy(), native["anchors_world"].copy()
    if force.shape != (n, 44) or anchors.shape != (n, 44, 3):
        raise ValueError("Native tactile must cover every source frame in 44-sensor order")
    if not np.isfinite(force).all() or not np.isfinite(anchors).all() or (force < 0).any():
        raise ValueError("Nonfinite geometry/force or negative normal force")
    if not np.any(force > 0):
        raise ValueError("No native tactile contacts: inspect simulator reconstruction")
    transform = np.asarray(manifest["configuration"]["base_to_world"])
    base_anchors = world_to_base(anchors, transform)
    original = np.load(source / "points.npy", mmap_mode="r")
    visual_count = manifest["configuration"]["visual_points"]
    original_indices = arrays["obs_frame_indices"]
    if original.shape != (len(original_indices), visual_count + 44, 8):
        raise ValueError("Unexpected point cache shape")
    keep = arrays["frame_index"][original_indices] >= rgb_lag
    rgb_indices = original_indices[keep]
    indices = rgb_indices - rgb_lag
    if not len(indices):
        raise ValueError("No observations remain after RGB alignment")
    np.testing.assert_array_equal(arrays["episode_index"][rgb_indices], arrays["episode_index"][indices],
                                  err_msg="RGB lag correction crosses an episode boundary")
    np.testing.assert_array_equal(arrays["frame_index"][rgb_indices] - rgb_lag,
                                  arrays["frame_index"][indices], err_msg="Non-contiguous frame alignment")
    staging = destination.with_name(destination.name + f".partial-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    for key in arrays.keys() - {"obs_frame_indices"}:
        shutil.copyfile(source / f"{key}.npy", staging / f"{key}.npy")
    np.save(staging / "obs_frame_indices.npy", indices)
    np.save(staging / "rgb_source_frame_indices.npy", rgb_indices)
    np.save(staging / "tactile.npy", force.astype(np.float32))
    points = np.lib.format.open_memmap(staging / "points.npy", mode="w+", dtype=np.float32,
                                      shape=(len(indices), visual_count + 44, 8))
    points[:] = original[keep]
    points[:, visual_count:, :] = 0
    points[:, visual_count:, :3] = base_anchors[indices]
    points[:, visual_count:, 6] = force[indices]
    points[:, visual_count:, 7] = 1
    points.flush()
    np.testing.assert_array_equal(points[:, :visual_count], original[keep, :visual_count],
                                  err_msg="Repair changed visual geometry")
    np.save(staging / "completed.npy", np.ones(len(indices), dtype=bool))
    # Preserve camera diagnostics but recompute all tactile-distance diagnostics.
    with (staging / "geometry_diagnostics.jsonl").open("w") as out:
        for line in (source / "geometry_diagnostics.jsonl").read_text().splitlines():
            diagnostic = json.loads(line)
            rgb_frame = int(diagnostic["frame"])
            old_index = int(np.searchsorted(original_indices, rgb_frame))
            if old_index >= len(original_indices) or original_indices[old_index] != rgb_frame:
                raise ValueError("Diagnostic frame is absent from cached observations")
            if not keep[old_index]:
                continue
            frame = rgb_frame - rgb_lag
            cache_index = int(np.searchsorted(indices, frame))
            sample = points[cache_index]
            distance = np.linalg.norm(sample[visual_count:, None, :3]
                                      - sample[None, :visual_count, :3], axis=-1).min(axis=1)
            active = force[frame] > 0.5
            diagnostic.update(cache_index=cache_index, frame=frame, rgb_source_frame=rgb_frame,
                              anchor_nearest_visual_median_m=float(np.median(distance)),
                              active_anchor_count=int(active.sum()),
                              active_anchor_nearest_visual_median_m=float(np.median(distance[active]))
                              if active.any() else None)
            out.write(json.dumps(diagnostic) + "\n")
    example_source = source / "example_observation.npz"
    if not example_source.exists():
        example_source = source / "shard-000" / "example_observation.npz"
    if example_source.exists() and rgb_lag == 0:
        with np.load(example_source, allow_pickle=False) as data:
            example = {key: data[key].copy() for key in data.files}
        first = int(indices[0])
        np.testing.assert_array_equal(example["state"], arrays["state"][first])
        example.update(tactile=force[first], anchors_world=anchors[first], source_frame=np.array(first))
        np.savez_compressed(staging / "example_observation.npz", **example)
    manifest["configuration"]["native_tactile_repair"] = {
        "source_cache_manifest": fingerprint(source / "manifest.json"),
        "readouts": readouts_fingerprint, "report": fingerprint(report),
        "implementation": fingerprint(__file__),
        "visual_points_bitwise_unchanged": True,
    }
    manifest["configuration"]["rgb_alignment"] = {
        "rgb_lag_frames": rgb_lag,
        "report": fingerprint(rgb_alignment_report) if alignment else None,
        "verified": bool(alignment and alignment.get("verified") is True),
        "mapping": "RGB source row r -> native state/force/action-label row r - rgb_lag_frames",
        "dropped_observations": int((~keep).sum()),
        "source_observations": len(original_indices),
    }
    manifest.update(observations=len(indices), total_observations=len(indices),
                    completed_observations=len(indices))
    if alignment:
        shutil.copyfile(rgb_alignment_report, staging / "rgb_alignment_report.json")
    manifest["configuration_fingerprint"] = hashlib.sha256(
        json.dumps(manifest["configuration"], sort_keys=True).encode()).hexdigest()
    manifest["normalization"] = training_statistics(arrays["state"], arrays["action"], force,
                                                    arrays["episode_index"], manifest["split"]["train"])
    manifest["tactile_semantics"] = (
        "Native tactile model recomputed at complete recorded qpos29; causal timestamp-based "
        "MuJoCo generalized velocity; synchronized mj_forward geometry/velocity; "
        "32 corrected-order taxels then 12 FSRs. See native_tactile_report.json.")
    manifest["historical_source_tactile_superseded"] = True
    if rgb_lag:
        manifest["action_semantics"] = (
            "Full-rate source labels unchanged. For corrected state row s and RGB row s+lag, "
            "the target is source action[s], historically raw command[s+1]; episode-local padding.")
    manifest["native_tactile_report"] = audit
    shutil.copyfile(report, staging / "native_tactile_report.json")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    staging.rename(destination)
    print(json.dumps({"complete": True, "output": str(destination), "frames": n,
                      "observations": len(indices), "active_frames": int((force > 0.5).any(1).sum())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--readouts", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rgb-lag", required=True, type=int, choices=(0, 1))
    parser.add_argument("--rgb-alignment-report", required=True, type=Path)
    args = parser.parse_args()
    repair(args.source, args.readouts, args.report, args.output, args.rgb_lag, args.rgb_alignment_report)
