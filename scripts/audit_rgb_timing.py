#!/usr/bin/env python3
"""Compare recorded RGB with nearby full-qpos renders; never modify input data.

Uses the frozen MuJoCo 3.4 runtime, with its matching Python 3.10 binary, and
software AV1 decoding by exact sequential frame ordinal. Output is audit JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_video_frames(path, indices):
    import av
    wanted = set(indices)
    result = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        if stream.codec_context.codec.id == av.Codec("av1", "r").id:
            codec = next(c for c in ("libdav1d", "libaom-av1") if c in av.codecs_available)
        decoder = av.CodecContext.create(codec, "r")
        decoder.extradata = stream.codec_context.extradata
        decoder.thread_count = 2
        ordinal = 0
        for packet in container.demux(stream):
            decoded = decoder.decode(packet) if packet.size else decoder.decode(None)
            for frame in decoded:
                if ordinal in wanted:
                    result[ordinal] = frame.to_ndarray(format="rgb24")
                ordinal += 1
                if len(result) == len(wanted):
                    return result
    raise ValueError("Source video ended before requested frame ordinals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-png-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.campaign / "eval/physics34"))
    import cv2
    import mujoco
    from PIL import Image
    from audit_alignment import load_model

    assets = args.campaign / "eval/assets/picklift"
    model = load_model(assets, mujoco)
    data = mujoco.MjData(model)
    source_rows = json.loads((args.campaign / "audit/source_picklift.json").read_text())["rows"]
    lags = [-2, -1, 0, 1, 2]
    rows = []
    raw_hashes = {}

    def compare(renderer, raw, qpos, frame, camera, episode, source_type):
        errors = []
        for lag in lags:
            data.qpos[:] = qpos[frame + lag]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            if source_type == "lerobot_video":
                # Exact converter operation: 640x480 -> center 480 square ->
                # Pillow LANCZOS 224 square, then source AV1 compression.
                image = np.asarray(Image.fromarray(image).crop((80, 0, 560, 480)).resize((224, 224), Image.Resampling.LANCZOS))
            errors.append(float(np.abs(raw.astype(np.float64) - image).mean()))
        rows.append({"source_type": source_type, "episode": episode, "frame": frame,
                     "camera": camera, "offset_mae_0_to_255": errors,
                     "best_qpos_offset": lags[int(np.argmin(errors))]})

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        raw_episode = args.raw_png_episode
        paths = sorted((raw_episode / "prioception").glob("*.npy"))
        qpos = np.stack([np.load(p) for p in paths])
        for frame in [25, 75, 125, 150, 175, 225, 275, 325, 375, 425]:
            for offset in lags:
                raw_hashes[str(paths[frame + offset])] = sha(paths[frame + offset])
            for camera, folder in (("left_cam", "cam_left_rgb"), ("right_cam", "cam_right_rgb")):
                path = raw_episode / folder / f"{frame:06d}.png"
                raw_hashes[str(path)] = sha(path)
                image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                compare(renderer, image, qpos, frame, camera, raw_episode.name, "original_png")

        # Distinct early and later demonstrations check that the lag was not
        # peculiar to the only complete original-PNG episode preserved here.
        selected = [0, 2, 26]
        starts = np.cumsum([0] + [row["frames"] for row in source_rows[:-1]])
        wanted = [int(starts[ep]) + f for ep in selected for f in (75, 150, 225)]
        for camera in ("left_cam", "right_cam"):
            video = args.source / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            decoded = selected_video_frames(video, wanted)
            raw_hashes[str(video)] = sha(video)
            for ep in selected:
                episode_name = source_rows[ep]["raw_episode"]
                paths = sorted((args.raw_root / episode_name / "prioception").glob("*.npy"))
                qpos = np.stack([np.load(p) for p in paths])
                for frame in (75, 150, 225):
                    for offset in lags:
                        raw_hashes[str(paths[frame + offset])] = sha(paths[frame + offset])
                    compare(renderer, decoded[int(starts[ep]) + frame], qpos, frame, camera, episode_name, "lerobot_video")

    grouped = {}
    for kind in ("original_png", "lerobot_video"):
        selected_rows = [row for row in rows if row["source_type"] == kind]
        grouped[kind] = {"comparisons": len(selected_rows),
                         "all_best_offset_minus_one": all(row["best_qpos_offset"] == -1 for row in selected_rows),
                         "median_offset_mae_0_to_255": np.median([row["offset_mae_0_to_255"] for row in selected_rows], axis=0).tolist()}
    verified = all(group["all_best_offset_minus_one"] for group in grouped.values())
    report = {"verified": verified, "rgb_lag_frames": 1,
              "interpretation": "RGB at recorded index t depicts full recorded qpos[t-1]. Reindex visual observations to t-1 within the same episode; discard each episode's first RGB observation. No future target is used to repair sensor geometry.",
              "scope": "Sampled pixel/render evidence across four episodes, not every source frame or a recovered original recorder implementation.",
              "evidence": {"mujoco_version": mujoco.__version__, "raw_png_episode": str(args.raw_png_episode),
                           "source": str(args.source), "qpos_offsets": lags, "summary": grouped, "comparisons": rows,
                           "scene_root": str(assets), "scene_sha256": sha(assets / "scene.xml"),
                           "asset_sha256": {str(p.relative_to(assets)): sha(p) for p in sorted(assets.rglob("*")) if p.is_file()},
                           "input_sha256": raw_hashes}}
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)
    if not verified:
        raise SystemExit("Not all sampled views support a common one-frame RGB lag")


if __name__ == "__main__":
    main()
