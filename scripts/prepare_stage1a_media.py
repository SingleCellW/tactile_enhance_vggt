#!/usr/bin/env python3
"""Package existing test recordings for Stage1.a; never run or re-render a test.

The six current clips are fixed nominal cases, selected independently of their
outcomes. Two explicitly historical successful ep24 clips illustrate the older
36k policy only. Source MP4/result/trajectory files remain unchanged. Downloads
and all video work use ordinary CPU processes, with no simulator/model imports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from tactile_vggt.paths import artifact_path
import shlex
import shutil
import struct
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
URL_ROOT = "/reports/stage1a/media"
REMOTE_ROOT = None


def selections():
    rows = []
    for n, modes in [(8, ["frozen_zero", "frozen_pressure", "joint_zero", "joint_pressure"]),
                     (2, ["frozen_zero", "frozen_pressure"])]:
        for mode in modes:
            rows.append({"id": f"{mode}_n{n}_nominal", "mode": mode, "n_action_steps": n,
                         "role": "current_nominal", "case_id": "nominal", "episode_id": 1,
                         "directory": f"outputs/simple_policy_followup_20261025/eval_n{n}/{mode}/{mode}/nominal"})
    for mode in ["anchored", "unanchored"]:
        rows.append({"id": f"historical_{mode}_36k_ep024", "mode": mode, "n_action_steps": 8,
                     "role": "historical_reference", "case_id": "episode_024", "episode_id": 24,
                     "directory": f"outputs/eval_v2_full36k_auto/final_equal_budget/{mode}_036000/episode_024"})
    return rows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def download_sources(host, remote_root, cache, rows):
    """Fetch a fixed allowlist and verify bytes against remote SHA after fetch."""
    files = [f"{row['directory']}/{name}" for row in rows
             for name in ["rollout.mp4", "result.json", "trajectory.npz"]]
    files += [str(Path(row["directory"]).parents[1] / "eval_config.json")
              for row in rows if row["role"] == "current_nominal"]
    files = sorted(set(files))
    cache.mkdir(parents=True, exist_ok=True)
    command = "tar -C " + shlex.quote(remote_root) + " -cf - " + " ".join(map(shlex.quote, files))
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["ssh", host, command], stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            members = tar.getmembers()
            if {m.name for m in members} != set(files) or any(not m.isfile() for m in members):
                raise ValueError("Remote archive differs from the fixed regular-file allowlist")
            for member in members:
                destination = cache / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    command = "cd " + shlex.quote(remote_root) + " && sha256sum -- " + " ".join(map(shlex.quote, files))
    remote_hashes = {}
    for line in run(["ssh", host, command]).stdout.splitlines():
        digest, path = line.split(None, 1)
        remote_hashes[path.strip()] = digest
    if set(remote_hashes) != set(files):
        raise ValueError("Incomplete remote SHA manifest")
    for path, expected in remote_hashes.items():
        if sha256(cache / path) != expected:
            raise ValueError(f"Source changed or download corrupted: {path}")
    (cache / "verified_remote_hashes.json").write_text(json.dumps({
        "host": host, "remote_root": remote_root, "files": remote_hashes,
        "verified_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


def probe(binary, path):
    result = json.loads(run([binary, "-v", "error", "-count_frames", "-show_streams",
                             "-show_format", "-of", "json", str(path)]).stdout)
    video = [s for s in result["streams"] if s["codec_type"] == "video"]
    if len(video) != 1 or any(s["codec_type"] == "audio" for s in result["streams"]):
        raise ValueError(f"Expected one silent video stream: {path}")
    stream = video[0]
    timestamps = json.loads(run([binary, "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]).stdout)
    pts = [float(frame["best_effort_timestamp_time"]) for frame in timestamps["frames"]]
    frames = int(stream["nb_read_frames"])
    if frames != len(pts):
        raise ValueError("Frame count / timestamp mismatch")
    return {"codec": stream["codec_name"], "pixel_format": stream.get("pix_fmt"),
            "width": stream["width"], "height": stream["height"], "frames": frames,
            "fps": float(Fraction(stream["avg_frame_rate"])), "fps_fraction": stream["avg_frame_rate"],
            "duration_seconds": float(result["format"]["duration"]), "frame_timestamps_seconds": pts}


def full_decode(ffmpeg, path):
    run([ffmpeg, "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(path),
         "-map", "0:v:0", "-an", "-f", "null", "-"])


def is_faststart(path):
    boxes = {}
    size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() + 8 <= size:
            offset = stream.tell()
            length, kind = struct.unpack(">I4s", stream.read(8))
            if length == 1:
                length = struct.unpack(">Q", stream.read(8))[0]
            elif length == 0:
                length = size - offset
            if length < 8:
                raise ValueError("Invalid MP4 box")
            boxes.setdefault(kind, offset)
            stream.seek(offset + length)
    return b"moov" in boxes and b"mdat" in boxes and boxes[b"moov"] < boxes[b"mdat"]


def validate_timing(source, asset, n_action_steps):
    fps = 1 / (.01 * n_action_steps)
    if source["frames"] != 600 // n_action_steps or abs(source["fps"] - fps) > 1e-6:
        raise ValueError("Source video does not match the recorded policy refresh cadence")
    if abs(source["duration_seconds"] - 6) > 1e-3 or abs(asset["duration_seconds"] - 6) > 1e-3:
        raise ValueError("Source/output must retain the complete six simulated seconds")
    if any(source[key] != asset[key] for key in ["width", "height", "frames", "fps"]):
        raise ValueError("Transcode changed frame geometry, count, or rate")
    error = max(abs(a - b) for a, b in zip(source["frame_timestamps_seconds"], asset["frame_timestamps_seconds"]))
    expected = [i * .01 * n_action_steps for i in range(source["frames"])]
    if error > 1e-5 or max(abs(a - b) for a, b in zip(source["frame_timestamps_seconds"], expected)) > 1e-5:
        raise ValueError("Transcode/source PTS no longer match the simulated-time sampling")
    return error


def normalized_outcome(result, row):
    if row["role"] == "current_nominal":
        if not result.get("complete") or result["case"]["case_id"] != "nominal" or result["ticks"] != 600:
            raise ValueError("Current clip must bind to a complete nominal test")
        if result["mode"] != row["mode"] or result["case"]["episode_id"] != 1:
            raise ValueError("Current result / selection mismatch")
        return dict(result["outcome"])
    final = result["final_info"]
    if result["episode"] != 24 or result["completed_control_steps"] != 600 or not final["strict_stable_goal"]:
        raise ValueError("Historical ep24 must have verified full-horizon strict success")
    xyz = final["cube_center"]
    return {"strict_stable_goal": True, "final_inside_goal": bool(final["inside_goal"]),
            "ever_lifted": bool(final["lifted"]), "lift_then_drop_outside_goal": None,
            "first_drop_tick": None, "max_cube_height_m": final["max_cube_height"],
            "final_goal_center_xy_distance_m": math.hypot(xyz[0] - .5, xyz[1] + .3),
            "scope": "Historical strict-success predicate from original result; drop was not scored in this result schema."}


def prepare(args):
    rows = selections()
    cache, output = args.source_cache.resolve(), args.output.resolve()
    if args.download:
        download_sources(args.remote_host, args.remote_root, cache, rows)
    verification = json.loads((cache / "verified_remote_hashes.json").read_text())
    if verification["host"] != args.remote_host or verification["remote_root"] != args.remote_root:
        raise ValueError("Source cache provenance differs from requested origin")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    videos = []
    for row in rows:
        directory = cache / row["directory"]
        source_files = {name: directory / name for name in ["rollout.mp4", "result.json", "trajectory.npz"]}
        for name, path in source_files.items():
            if sha256(path) != verification["files"][f"{row['directory']}/{name}"]:
                raise ValueError(f"Source cache changed: {path}")
        result = json.loads(source_files["result.json"].read_text())
        outcome = normalized_outcome(result, row)
        checkpoint = None
        if row["role"] == "current_nominal":
            configuration_path = directory.parents[1] / "eval_config.json"
            configuration_relative = str(configuration_path.relative_to(cache))
            if sha256(configuration_path) != verification["files"][configuration_relative] or sha256(configuration_path) != result["config_sha256"]:
                raise ValueError("Result / eval_config SHA mismatch")
            configuration = json.loads(configuration_path.read_text())
            if configuration["execution_horizon"]["n_action_steps"] != row["n_action_steps"]:
                raise ValueError("Configuration / policy refresh cadence mismatch")
            checkpoint = dict(configuration["modes"][row["mode"]])
            checkpoint["path_relative"] = str(Path(checkpoint.pop("policy")).relative_to(args.remote_root))
        remote_video = f"{args.remote_root}/{row['directory']}/rollout.mp4"
        if result.get("video") != remote_video:
            raise ValueError("Original result does not identify this source MP4")
        source_probe = probe(args.ffprobe, source_files["rollout.mp4"])
        full_decode(args.ffmpeg, source_files["rollout.mp4"])
        asset, poster = output / f"{row['id']}.mp4", output / f"{row['id']}.png"
        run([args.ffmpeg, "-v", "error", "-y", "-i", str(source_files["rollout.mp4"]),
             "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-profile:v", "main", "-vsync", "0", "-movflags", "+faststart", str(asset)])
        asset_probe = probe(args.ffprobe, asset)
        full_decode(args.ffmpeg, asset)
        timestamp_error = validate_timing(source_probe, asset_probe, row["n_action_steps"])
        if asset_probe["codec"] != "h264" or not is_faststart(asset):
            raise ValueError("Web asset must be H264 with moov before mdat")
        run([args.ffmpeg, "-v", "error", "-y", "-ss", "2.0", "-i", str(asset), "-frames:v", "1", str(poster)])
        evidence = output / "evidence" / f"{row['id']}.snapshot.json"
        evidence.write_text(json.dumps({"original_result_path_relative": f"{row['directory']}/result.json",
            "original_result_sha256": sha256(source_files["result.json"]),
            "note": "Public scoring excerpt; the SHA above binds the unmodified original result, not this excerpt.",
            "case_id": row["case_id"], "mode": row["mode"], "role": row["role"],
            "outcome": outcome, "initial_cube_center": result["initial_info"]["cube_center"],
            "final_cube_center": result["final_info"]["cube_center"],
            "strict_stability_rule": result["final_info"]["strict_stability_rule"],
            "object_contact_scoring_only": result.get("object_contact_scoring_only")}, indent=2, ensure_ascii=False) + "\n")
        initialization = {key: value for key, value in result["initial_info"]["initialization"].items() if not key.endswith("_path")}
        source = {"source_kind": "original_saved_test_video",
                  "rendered_again": False, "verified_at": verification["verified_at"],
                  "video": {"path_relative": f"{row['directory']}/rollout.mp4", "sha256": sha256(source_files["rollout.mp4"])},
                  "result": {"path_relative": f"{row['directory']}/result.json", "sha256": sha256(source_files["result.json"]),
                             "snapshot_url": f"{URL_ROOT}/evidence/{evidence.name}", "snapshot_sha256": sha256(evidence)},
                  "trajectory": {"path_relative": f"{row['directory']}/trajectory.npz", "sha256": sha256(source_files["trajectory.npz"])},
                  "configuration_sha256": result.get("config_sha256"),
                  "checkpoint": checkpoint, "initialization": initialization,
                  "worker_provenance": result["worker_provenance"], "source_video_probe": source_probe}
        if row["role"] == "current_nominal":
            if outcome["lift_then_drop_outside_goal"]:
                stage = "抬升后掉落，未到目标区域"
            elif result["object_contact_scoring_only"]["post_action_contact_ticks"] > 0 and not outcome["ever_lifted"]:
                stage = "发生物体接触，但未达到抬升阈值"
            else:
                stage = "未达到严格成功条件"
            caption = (f"本轮 {row['mode']}，n_action_steps={row['n_action_steps']}，固定 nominal 初态：{stage}。"
                       "这是已保存的 nominal 失败视频，不代表完整 36 点网格结果；并非按最好表现挑选。")
            title = f"{row['mode']} · n{row['n_action_steps']} · nominal"
        else:
            caption = (f"历史 {row['mode']} 36k checkpoint / episode 24：原测试的严格成功案例。"
                       "来自历史 5 来源评测，训练预算、表示结构与本轮不同，只作历史参考，不计入本轮 36 点网格统计；明确按成功案例展示。")
            title = f"历史 {row['mode']} · 36k · ep24 · 严格成功"
        timing = {"timebase": "simulation_seconds", "simulation_seconds": 6.0,
                  "source_fps": source_probe["fps"], "source_frames": source_probe["frames"],
                  "capture_interval_seconds": .01 * row["n_action_steps"],
                  "first_capture_sim_seconds": 0.0, "last_capture_sim_seconds": source_probe["frame_timestamps_seconds"][-1],
                  "sync_start_seconds": 0.0, "sync_end_seconds": 6.0, "playback_rate": 1.0,
                  "note": "每次策略刷新保存左右 RGB224 拼接帧；保持原帧时间。评分在600tick后，末视频帧早于最终评分。暂停推理的墙钟耗时未进入视频。"}
        asset_metadata = {key: value for key, value in asset_probe.items() if key != "frame_timestamps_seconds"}
        asset_metadata.update(sha256=sha256(asset), bytes=asset.stat().st_size, faststart=True,
                              complete_decode_passed=True, source_complete_decode_passed=True,
                              max_frame_timestamp_change_seconds=timestamp_error,
                              processing="CPU H264 transcode of original saved frames only; no interpolation, new rollout, or re-render",
                              poster_sim_seconds=2.0, poster_sha256=sha256(poster))
        videos.append({**{key: value for key, value in row.items() if key != "directory"},
                       "title": title, "caption": caption, "src": f"{URL_ROOT}/{asset.name}",
                       "poster": f"{URL_ROOT}/{poster.name}", "outcome": outcome,
                       "contact": result.get("object_contact_scoring_only"), "timing": timing,
                       "source": source, "asset": asset_metadata})
        print(f"{row['id']}: {asset_probe['frames']} frames @ {asset_probe['fps']}fps; 6s; strict={outcome['strict_stable_goal']}", flush=True)
    by_id = {video["id"]: video for video in videos}
    for mode in ["frozen_zero", "frozen_pressure"]:
        pair = [by_id[f"{mode}_n{n}_nominal"] for n in [8, 2]]
        if pair[0]["source"]["checkpoint"] != pair[1]["source"]["checkpoint"] or pair[0]["source"]["initialization"] != pair[1]["source"]["initialization"]:
            raise ValueError("Synchronized pair differs in checkpoint or physical initialization")
    comparisons = [{"id": f"{mode}_n8_vs_n2_nominal", "title": f"{mode}：相同 nominal 初态，n8 与 n2",
                    "video_ids": [f"{mode}_n8_nominal", f"{mode}_n2_nominal"],
                    "timebase": "simulation_seconds", "start_seconds": 0, "end_seconds": 6,
                    "sync_rule": "将两播放器 currentTime 设为同一模拟秒；速度1x。保留 n8=12.5fps / n2=50fps，不以帧序号对齐。",
                    "same_initial_state": True, "same_checkpoint_within_pair": True}
                   for mode in ["frozen_zero", "frozen_pressure"]]
    catalog = {"version": 1, "complete": True, "created_at": datetime.now(timezone.utc).isoformat(),
               "selection_policy": {"current": "Fixed nominal clips: all four n8 modes plus frozen_zero/frozen_pressure n2, independent of outcomes.",
                                    "historical": "Two explicitly successful historical ep24 examples (anchored and unanchored 36k); illustrative success selection, not an unbiased sample."},
               "limitations": ["六段本轮视频均为固定 nominal；不用于估计36网格成功率。",
                               "两段严格成功来自历史36k/5来源评测，只作历史参考，不是本轮成功。",
                               "视频是原测试保存帧的CPU转码，未重跑、未从轨迹重渲染；2秒海报固定选取。",
                               "视频按模拟时间播放；推理期间模拟暂停，因此不能用播放速度推断现实控制吞吐或实时性。",
                               "左右图像为策略所见RGB224拼接；评分来自独立result/trajectory，GT仅用于评分。"],
               "videos": videos, "comparisons": comparisons}
    temporary = output / "videos.json.tmp"
    temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output / "videos.json")
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch only the eight fixed source recordings and evidence via authorized SSH")
    parser.add_argument("--remote-host", help="SSH alias; required only with --download")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--source-cache", type=Path, default=artifact_path("cache", "stage1a_media_sources"))
    parser.add_argument("--output", type=Path, default=artifact_path('site', 'reports/stage1a/media'))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    args = parser.parse_args()
    if args.download and (not args.remote_host or not args.remote_root):
        parser.error("--download requires explicit --remote-host and --remote-root")
    prepare(args)


if __name__ == "__main__":
    main()
