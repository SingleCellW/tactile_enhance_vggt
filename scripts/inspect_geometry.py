#!/usr/bin/env python3
"""Read-only geometry QA for a complete cache or an in-progress cache shard.

Reads only observations marked completed in a snapshot of the completion bitmap.
The output contains coordinate integrity statistics and calibrated second-camera
residuals. Tactile-to-sampled-visual nearest distances are proximity diagnostics,
NOT ground-truth geometry errors: visibility, point sampling and sensor placement
all affect them. No ground-truth depth or object pose is loaded or inferred.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


class CoordinateStatistics:
    def __init__(self):
        self.points = 0
        self.finite_points = 0
        self.nonfinite_values = 0
        self.minimum = np.full(3, np.inf)
        self.maximum = np.full(3, -np.inf)
        self.total = np.zeros(3)
        self.total_sq = np.zeros(3)

    def update(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        self.points += len(xyz)
        finite = np.isfinite(xyz)
        self.nonfinite_values += int((~finite).sum())
        xyz = xyz[finite.all(axis=1)]
        if not len(xyz):
            return
        self.finite_points += len(xyz)
        self.minimum = np.minimum(self.minimum, xyz.min(axis=0))
        self.maximum = np.maximum(self.maximum, xyz.max(axis=0))
        self.total += xyz.sum(axis=0)
        self.total_sq += np.square(xyz).sum(axis=0)

    def result(self):
        result = {
            "point_count": self.points,
            "finite_point_count": self.finite_points,
            "nonfinite_coordinate_values": self.nonfinite_values,
            "axis_order": ["x", "y", "z"],
        }
        if self.finite_points:
            mean = self.total / self.finite_points
            variance = np.maximum(self.total_sq / self.finite_points - mean ** 2, 0)
            result.update(min_m=self.minimum.tolist(), max_m=self.maximum.tolist(),
                          mean_m=mean.tolist(), std_m=np.sqrt(variance).tolist())
        return result


def scalar_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = {"count": len(values), "finite_count": len(finite), "nonfinite_count": len(values) - len(finite)}
    if len(finite):
        result.update(min=float(finite.min()), median=float(np.median(finite)),
                      p95=float(np.percentile(finite, 95)), max=float(finite.max()), mean=float(finite.mean()))
    return result


def cache_roots(path):
    path = Path(path).resolve()
    if (path / "manifest.json").is_file():
        return [path]
    shards = sorted(directory for directory in path.glob("shard-*") if (directory / "manifest.json").is_file())
    if shards:
        return shards
    raise FileNotFoundError(f"No cache manifest or initialized shard manifests in {path}")


def read_completed(root):
    manifest = json.loads((root / "manifest.json").read_text())
    points = np.load(root / "points.npy", mmap_mode="r")
    observation_frames = np.load(root / "obs_frame_indices.npy", mmap_mode="r")
    if points.ndim != 3 or points.shape[-1] != 8 or len(observation_frames) != len(points):
        raise ValueError(f"Invalid point tensor/observation index shape in {root}")
    if (root / "completed.npy").exists():
        # Point writes are flushed before each flag is set. A private snapshot
        # prevents this report's observation set changing while preparation runs.
        completed = np.array(np.load(root / "completed.npy", mmap_mode="r"), copy=True)
        if completed.shape != (len(points),) or completed.dtype != np.bool_:
            raise ValueError(f"Invalid completion bitmap in {root}")
        if manifest.get("complete") and not completed.all():
            raise ValueError(f"Complete manifest disagrees with completion bitmap in {root}")
    elif manifest.get("complete"):
        completed = np.ones(len(points), dtype=bool)
    else:
        raise ValueError(f"An in-progress cache requires a completion bitmap: {root}")
    return manifest, points, observation_frames, np.flatnonzero(completed)


def diagnostic_records(root, included_frames):
    path = root / "geometry_diagnostics.jsonl"
    if not path.exists():
        return [], {"present": False, "invalid_lines": 0, "duplicate_completed_records": 0}
    records = {}
    invalid, duplicates = 0, 0
    # Reading a file while its last record is being appended can expose a partial
    # final line. Report it and exclude it, rather than treating it as data.
    for line in path.read_bytes().splitlines():
        try:
            record = json.loads(line)
            frame = int(record["frame"])
        except (ValueError, KeyError, TypeError):
            invalid += 1
            continue
        if frame in included_frames:
            duplicates += int(frame in records)
            records[frame] = record
    return list(records.values()), {
        "present": True, "invalid_lines": invalid,
        "duplicate_completed_records": duplicates,
    }


def render_overview(examples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    columns = min(2, len(examples))
    rows = (len(examples) + columns - 1) // columns
    figure = plt.figure(figsize=(7 * columns, 5.8 * rows), layout="constrained")
    all_xyz = np.concatenate([item["points"][:, :3] for item in examples])
    all_xyz = all_xyz[np.isfinite(all_xyz).all(axis=1)]
    if not len(all_xyz):
        raise ValueError("Cannot render: no finite XYZ in selected observations")
    minimum, maximum = all_xyz.min(axis=0), all_xyz.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float((maximum - minimum).max()) / 2, 0.01) * 1.08
    pressure_values = np.concatenate([item["points"][item["visual_count"]:, 6] for item in examples])
    pressure_values = pressure_values[np.isfinite(pressure_values)]
    maximum_pressure = max(float(np.log1p(np.maximum(pressure_values, 0)).max()), 1e-6) if len(pressure_values) else 1.0
    pressure_norm = Normalize(vmin=0, vmax=maximum_pressure)
    axes, color_artist = [], None
    for index, item in enumerate(examples):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        axes.append(axis)
        points, visual_count = item["points"], item["visual_count"]
        visual, anchors = points[:visual_count], points[visual_count:]
        valid = np.isfinite(visual[:, :6]).all(axis=1)
        visual = visual[valid]
        if len(visual):
            axis.scatter(*visual[:, :3].T, c=np.clip(visual[:, 3:6], 0, 1), s=8, alpha=0.75,
                         depthshade=False, label="VGGT visual points")
        anchors = anchors[np.isfinite(anchors[:, [0, 1, 2, 6]]).all(axis=1)]
        if len(anchors):
            log_pressure = np.log1p(np.maximum(anchors[:, 6], 0))
            color_artist = axis.scatter(*anchors[:, :3].T, c=log_pressure, cmap="plasma", norm=pressure_norm,
                                       s=30 + 75 * log_pressure / maximum_pressure,
                                       edgecolors="black", linewidths=0.6, depthshade=False,
                                       label="Tactile sensor anchors")
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                 xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[1] - radius, center[1] + radius),
                 zlim=(center[2] - radius, center[2] + radius))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=25, azim=-65)
        axis.set_title(f"Episode {item['episode']} · frame {item['frame_index']}\nSource row {item['source_frame']}", fontsize=11)
        axis.legend(loc="upper left", fontsize=8)
    if color_artist is not None:
        figure.colorbar(color_artist, ax=axes, shrink=0.62, pad=0.03, label="log(1 + raw tactile pressure)")
    figure.suptitle("Frozen VGGT visual geometry and tactile anchors\nRobot-base metres · calibrated origin/scale · no ground-truth depth", fontsize=14)
    figure.supxlabel("Proximity to sampled visible points is not a ground-truth geometry error.", fontsize=10)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_matching_source_rgb(examples, output):
    """Match by state row while labeling a separately recorded RGB source row."""
    first = examples[0]
    path = Path(first["cache"]) / "example_observation.npz"
    if not path.exists():
        return {"included": False, "reason": "No example RGB file for the first plotted observation"}
    with np.load(path, allow_pickle=False) as archive:
        if "source_frame" not in archive:
            return {"included": False, "reason": "Legacy example has no explicit source_frame; no correspondence assumed"}
        source_frame = int(archive["source_frame"].item())
        if source_frame != first["source_frame"]:
            return {"included": False, "reason": "Example state source_frame differs from the first plotted observation"}
        # Repaired lagged examples distinguish the RGB row from the state row.
        # Older zero-lag examples store only source_frame, shared by both.
        rgb_source_frame = int(archive["rgb_source_frame"].item()) if "rgb_source_frame" in archive else source_frame
        rgb = archive["rgb"].copy()
    if rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3:
        return {"included": False, "reason": "Example RGB shape is not two RGB views"}
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.2), layout="constrained")
    for view, axis in enumerate(axes):
        axis.imshow(rgb[view])
        axis.set_title(("left_cam", "right_cam")[view])
        axis.axis("off")
    figure.suptitle(f"Source RGB · episode {first['episode']} / state frame {first['frame_index']}\n"
                   f"State row {source_frame} · RGB row {rgb_source_frame}")
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return {"included": True, "source_frame": source_frame,
            "source_frame_semantics": "state row used by the plotted observation",
            "state_source_frame": source_frame, "rgb_source_frame": rgb_source_frame,
            "rgb_lag_frames": rgb_source_frame - source_frame,
            "path": str(output), "source_npz": str(path)}


def inspect(args):
    roots = cache_roots(args.cache)
    output = Path(args.output_dir).resolve()
    for root in roots:
        if output == root or root in output.parents:
            raise ValueError("Use an output directory outside the cache/shard to keep preparation artifacts unchanged")
    visual_xyz, anchor_xyz = CoordinateStatistics(), CoordinateStatistics()
    nonfinite_channels = np.zeros(8, dtype=np.int64)
    all_distances, active_distances, raw_pressures = [], [], []
    examples, cache_info, all_diagnostics, record_info = [], [], [], []
    entries, planned_count = [], 0
    for root in roots:
        manifest, points, observation_frames, completed = read_completed(root)
        visual_count = int(manifest["configuration"]["visual_points"])
        if not 0 < visual_count < points.shape[1]:
            raise ValueError(f"Invalid visual point count in {root}")
        selected = completed
        if args.max_observations:
            selected = selected[:max(args.max_observations - planned_count, 0)]
        entries.append((root, manifest, points, observation_frames, completed, selected, visual_count))
        planned_count += len(selected)
    if not planned_count:
        raise ValueError("No completed observations available yet")
    plot_indices = set(np.linspace(0, planned_count - 1, min(args.plot_observations, planned_count), dtype=int).tolist())
    active_example = None
    selected_count = 0
    for root, manifest, points, observation_frames, completed, selected, visual_count in entries:
        episodes = np.load(root / "episode_index.npy", mmap_mode="r")
        frames = np.load(root / "frame_index.npy", mmap_mode="r")
        cache_info.append({
            "path": str(root), "manifest_complete": bool(manifest.get("complete")),
            "total_observations": len(points), "completed_at_snapshot": len(completed),
            "inspected_observations": len(selected), "visual_points_per_observation": visual_count,
            "tactile_points_per_observation": points.shape[1] - visual_count,
            "configuration_fingerprint": manifest.get("configuration_fingerprint"),
        })
        included_frames = set()
        for cache_index in selected:
            sample = np.array(points[cache_index], dtype=np.float64, copy=True)
            source_frame = int(observation_frames[cache_index])
            included_frames.add(source_frame)
            nonfinite_channels += (~np.isfinite(sample)).sum(axis=0)
            visual, anchors = sample[:visual_count], sample[visual_count:]
            visual_xyz.update(visual[:, :3])
            anchor_xyz.update(anchors[:, :3])
            raw_pressures.extend(anchors[:, 6].tolist())
            finite_visual = visual[np.isfinite(visual[:, :3]).all(axis=1), :3]
            finite_anchor = np.isfinite(anchors[:, :3]).all(axis=1)
            distances = np.full(len(anchors), np.nan)
            if len(finite_visual) and finite_anchor.any():
                delta = anchors[finite_anchor, None, :3] - finite_visual[None, :, :]
                distances[finite_anchor] = np.sqrt(np.square(delta).sum(axis=-1)).min(axis=1)
            active = np.isfinite(anchors[:, 6]) & (anchors[:, 6] > args.active_pressure_threshold)
            all_distances.extend(distances.tolist())
            active_distances.extend(distances[active].tolist())
            new_active_example = active.any() and (active_example is None or int(active.sum()) > active_example["active_anchor_count"])
            if selected_count in plot_indices or new_active_example:
                example = {
                    "points": sample, "visual_count": visual_count, "cache": str(root),
                    "cache_index": int(cache_index), "source_frame": source_frame,
                    "inspection_index": selected_count,
                    "episode": int(episodes[source_frame]), "frame_index": int(frames[source_frame]),
                    "active_anchor_count": int(active.sum()),
                    "nearest_visual_distance_m": scalar_statistics(distances),
                    "active_nearest_visual_distance_m": scalar_statistics(distances[active]),
                }
                if selected_count in plot_indices:
                    examples.append(example)
                if new_active_example:
                    active_example = example
            selected_count += 1
        records, info = diagnostic_records(root, included_frames)
        all_diagnostics.extend(records)
        record_info.append({"cache": str(root), **info, "included_completed_records": len(records)})
    # Spread views over the inspected sequence, retaining the first observation
    # when >1 panel is requested. If the evenly spaced views missed all contact,
    # replace the last one with a completed observation containing active touch.
    if active_example is not None and not any(item["active_anchor_count"] for item in examples):
        examples[-1] = active_example
    examples.sort(key=lambda item: item["inspection_index"])

    camera_keys = (
        "second_camera_position_error_m", "second_camera_rotation_error_deg",
        "scale", "predicted_baseline", "known_baseline_m",
    )
    camera_metrics = {}
    for key in camera_keys:
        values = [record[key] for record in all_diagnostics if isinstance(record.get(key), (float, int))]
        camera_metrics[key] = scalar_statistics(values)
    report = {
        "created_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "read-only spatial representation QA",
        "coordinate_frame": "robot_base", "coordinate_units": "metres",
        "ground_truth_geometry_evaluated": False,
        "cache_snapshots": cache_info, "inspected_observations": selected_count,
        "visual_xyz": visual_xyz.result(), "tactile_anchor_xyz": anchor_xyz.result(),
        "nonfinite_values_by_channel": dict(zip(["x", "y", "z", "r", "g", "b", "pressure", "is_tactile"], nonfinite_channels.tolist(), strict=True)),
        "raw_tactile_pressure": scalar_statistics(raw_pressures),
        "second_camera_diagnostics": {
            "interpretation": "Residual against known camera calibration after first-camera origin/orientation and stereo-baseline scale alignment; not object/depth accuracy",
            "metrics": camera_metrics, "logs": record_info,
        },
        "tactile_to_sampled_visual_proximity": {
            "is_ground_truth_geometry_error": False,
            "interpretation": "Nearest sampled visible point distance only; affected by occlusion, sampling density and sensor anchor placement. No correspondence or contact-surface truth is known.",
            "units": "metres", "active_raw_pressure_threshold": args.active_pressure_threshold,
            "all_anchors": scalar_statistics(all_distances),
            "active_anchors": scalar_statistics(active_distances),
        },
        "plotted_observations": [{key: value for key, value in item.items() if key != "points"} for item in examples],
        "plot_selection": "Spread across completed inspected observations; retain the first when using multiple panels and include active contact when available. One-panel output prioritizes contact.",
    }
    output.mkdir(parents=True, exist_ok=True)
    png_path, json_path = output / "geometry_overview.png", output / "geometry_summary.json"
    render_overview(examples, png_path)
    report["overview_png"] = str(png_path)
    report["source_rgb"] = render_matching_source_rgb(examples, output / "source_rgb_example.png")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"observations": selected_count, "summary": str(json_path), "overview": str(png_path)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Merged cache, individual shard, or parent of initialized shards")
    parser.add_argument("--output-dir", required=True, help="Report directory outside the cache/shard")
    parser.add_argument("--plot-observations", type=int, default=4)
    parser.add_argument("--max-observations", type=int, default=0, help="Limit statistics to first N completed observations; 0 means all")
    parser.add_argument("--active-pressure-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 1 <= args.plot_observations <= 4 or args.max_observations < 0 or not np.isfinite(args.active_pressure_threshold):
        parser.error("plot-observations must be 1..4; max-observations must be nonnegative; pressure threshold must be finite")
    inspect(args)


if __name__ == "__main__":
    main()
