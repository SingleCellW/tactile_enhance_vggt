#!/usr/bin/env python3
"""Cache synchronized RGB pairs with real frozen VGGT and corrected44 tactile.

Run each --shard-index with the same --num-shards/arguments, then --merge-shards.
Shards are independent directories; interruptions resume from a per-frame bitmap.
Simulator depth and object poses are never loaded into this preparation path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile_vggt.data import episode_split, training_statistics
from tactile_vggt.geometry import (build_point_tensor, calibrated_similarity,
                                   mujoco_camera_to_opencv, unproject_depth)

CAMERAS = ("left_cam", "right_cam")
NUMERIC_ARRAYS = ("state", "action", "tactile", "episode_index", "frame_index")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_info(path, digest=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        result["sha256"] = h.hexdigest()
    return result


def weight_info(path):
    """Reuse the downloader's verified digest when present, otherwise hash weights."""
    path = Path(path)
    provenance_path = path.with_suffix(".provenance.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        digest = provenance.get("sha256", provenance.get("weights_sha256", ""))
        if isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdefABCDEF" for c in digest):
            return dict(file_info(path), sha256=digest.lower(), verified_download_provenance=file_info(provenance_path, digest=True))
    return file_info(path, digest=True)


def load_source(args):
    import pyarrow.parquet as pq
    parquet_files = sorted(Path(args.source).glob("data/chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No source parquet in {args.source}")
    # The selected historical source is contiguous, one row per same-frame RGB pair.
    import pyarrow as pa
    table = pa.concat_tables([pq.read_table(p) for p in parquet_files])
    column = lambda name: np.asarray(table[name].to_pylist())
    episode, frame = column("episode_index"), column("frame_index")
    state, action = column("observation.state").astype(np.float32), column("action").astype(np.float32)
    with np.load(args.readouts, allow_pickle=False) as readouts:
        for key, actual in (("episode_index", episode), ("frame_index", frame), ("state22", state)):
            np.testing.assert_array_equal(readouts[key], actual, err_msg=f"Readout/source mismatch: {key}")
        tactile = readouts["force44"].copy()
        anchors = readouts["anchors_world"].copy()
    n = len(state)
    if state.shape != (n, 22) or action.shape != (n, 22) or tactile.shape != (n, 44) or anchors.shape != (n, 44, 3):
        raise ValueError("Unexpected source dimensions")
    if not all(np.isfinite(a).all() for a in (state, action, tactile, anchors)):
        raise ValueError("Source contains nonfinite values")
    for ep in np.unique(episode):
        np.testing.assert_array_equal(frame[episode == ep], np.arange(np.count_nonzero(episode == ep)))
    roots = sorted(p for p in Path(args.raw_root).iterdir() if (p / "tacsim_dataset_processed/metadata.json").exists())
    if len(roots) != len(np.unique(episode)):
        raise ValueError("Raw episode directories do not match source episode count")
    calibrations = {}
    calibration_files = []
    for ep, root in zip(np.unique(episode), roots):
        meta_path = root / "tacsim_dataset_processed/metadata.json"
        meta = json.loads(meta_path.read_text())
        if meta["total_frames"] != np.count_nonzero(episode == ep):
            raise ValueError(f"Raw/source frame count differs at {root.name}")
        camera_mj = np.array([meta["cameras"][name]["extrinsics"] for name in CAMERAS])
        calibrations[int(ep)] = mujoco_camera_to_opencv(camera_mj)
        calibration_files.append(file_info(meta_path, digest=True))
    videos = []
    for name in CAMERAS:
        paths = sorted(Path(args.source).glob(f"videos/observation.images.{name}/chunk-*/*.mp4"))
        if len(paths) != 1:
            raise ValueError("This pilot loader requires one contiguous source video per camera")
        videos.append(paths[0])
    arrays = dict(state=state, action=action, tactile=tactile, episode_index=episode, frame_index=frame)
    provenance = {"readouts": file_info(args.readouts, digest=True),
                  "source_parquet": [file_info(p, digest=True) for p in parquet_files],
                  "rgb_videos": [file_info(p, digest=True) for p in videos], "camera_metadata": calibration_files}
    return arrays, anchors, calibrations, videos, provenance


class VideoReader:
    """Software AV1 decoder; exact decoded-frame ordinals, never timestamp seeking.

    The source videos concatenate one frame for each parquet row. Sequential
    demux/decode avoids keyframe seek rounding and does not resample frame rate.
    Only the current RGB frame is retained, even when skipping to a later shard.
    """
    def __init__(self, path, expected_frames):
        import av
        self.av, self.path, self.expected_frames = av, str(path), int(expected_frames)
        self.container = None
        self._open()

    def _open(self):
        if self.container is not None:
            self.container.close()
        self.container = self.av.open(self.path, mode="r")
        stream = self.container.streams.video[0]
        if stream.frames and stream.frames != self.expected_frames:
            self.close()
            raise ValueError(f"RGB/source frame mismatch: {self.path}: {stream.frames} != {self.expected_frames}")
        # Explicit codec context avoids selecting an AV1 hardware-only decoder.
        is_av1 = stream.codec_context.codec.id == self.av.Codec("av1", "r").id
        if is_av1:
            codec = next((c for c in ("libdav1d", "libaom-av1") if c in self.av.codecs_available), None)
            if codec is None:
                self.close()
                raise RuntimeError("PyAV needs a software AV1 decoder (libdav1d or libaom-av1)")
        else:
            codec = stream.codec_context.name
        decoder = self.av.CodecContext.create(codec, "r")
        decoder.extradata = stream.codec_context.extradata
        decoder.thread_count = 2
        self.codec_name = codec

        def frames():
            for packet in self.container.demux(stream):
                if packet.size:
                    yield from decoder.decode(packet)
            yield from decoder.decode(None)

        self.frames = iter(frames())
        self.position = 0
        self.current = self._next_rgb()  # Validate decoding before loading 5 GB model.

    def _next_rgb(self):
        try:
            frame = next(self.frames)
        except StopIteration as error:
            raise RuntimeError(f"RGB stream ended before frame {self.position}: {self.path}") from error
        return frame.to_ndarray(format="rgb24")

    def read(self, index):
        index = int(index)
        if not 0 <= index < self.expected_frames:
            raise IndexError(index)
        if index < self.position:
            self._open()
        while self.position < index:
            self.position += 1
            self.current = self._next_rgb()
        return self.current.copy()

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None


class FrozenVGGT:
    def __init__(self, repository, weights, device, image_size):
        import torch
        sys.path.insert(0, str(Path(repository).resolve()))
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.torch, self.pose_decoder = torch, pose_encoding_to_extri_intri
        self.device, self.image_size = device, image_size
        self.model = VGGT(enable_point=False, enable_track=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        incompatible = self.model.load_state_dict(state, strict=False)
        unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(("point_head.", "track_head."))]
        if incompatible.missing_keys or unexpected:
            raise ValueError(f"Incomplete VGGT weights: {incompatible.missing_keys}, {unexpected}")
        del state
        self.model.to(device).eval().requires_grad_(False)
        self.dtype = torch.bfloat16 if not device.startswith("cuda") or torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    def predict(self, rgb_pairs):
        import cv2
        torch = self.torch
        prepared = []
        for pair in rgb_pairs:
            views = []
            for rgb in pair:
                # Historical RGB source is square centre-cropped. Keep its full FOV.
                if rgb.shape[0] != rgb.shape[1]:
                    raise ValueError("Expected the historical square RGB source, avoiding hidden crop changes")
                views.append(cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR))
            prepared.append(views)
        rgb = np.asarray(prepared, dtype=np.float32) / 255.0
        images = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", dtype=self.dtype, enabled=self.device.startswith("cuda")):
            output = self.model(images)
        extrinsics, intrinsics = self.pose_decoder(output["pose_enc"].float(), (self.image_size, self.image_size))
        return rgb, output["depth"].float().cpu().numpy(), output["depth_conf"].float().cpu().numpy(), extrinsics.cpu().numpy(), intrinsics.cpu().numpy()


def prepare(args):
    arrays, anchors, calibrations, videos, provenance = load_source(args)
    split = episode_split(arrays["episode_index"], args.seed, args.val_fraction)
    observations = np.flatnonzero(arrays["frame_index"] % args.obs_stride == 0)
    if args.max_frames and len(observations) > args.max_frames:
        observations = observations[np.linspace(0, len(observations) - 1, args.max_frames, dtype=int)]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    total_observations = len(observations)
    observations = np.array_split(observations, args.num_shards)[args.shard_index]
    if not len(observations):
        raise ValueError("Empty shard")
    root = Path(args.output)
    if args.num_shards > 1:
        root /= f"shard-{args.shard_index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    base_transform = np.eye(4) if args.base_transform is None else np.asarray(json.loads(Path(args.base_transform).read_text()))
    if base_transform.shape != (4, 4):
        raise ValueError("base-transform must contain a 4x4 base-to-world transform")
    try:
        vggt_commit = subprocess.check_output(["git", "-C", str(args.vggt_repo), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        vggt_commit = "unavailable; source files hashed individually"
    configuration = {"obs_stride": args.obs_stride, "max_frames": args.max_frames, "image_size": args.image_size,
                     "visual_points": args.visual_points, "seed": args.seed, "val_fraction": args.val_fraction,
                     "base_to_world": base_transform.tolist(), "num_shards": args.num_shards,
                     "confidence_quantile": args.confidence_quantile, "source": provenance,
                     "weights": weight_info(args.weights), "vggt_commit": vggt_commit,
                     "implementation": [file_info(p, digest=True) for p in (
                         Path(__file__), Path(__file__).resolve().parents[1] / "src/tactile_vggt/geometry.py",
                         Path(args.vggt_repo) / "vggt/models/vggt.py", Path(args.vggt_repo) / "vggt/utils/pose_enc.py")]}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration_fingerprint"] != fingerprint:
            raise ValueError("Existing cache configuration differs; use a new output directory")
    else:
        for name, values in arrays.items():
            np.save(root / f"{name}.npy", values)
        np.save(root / "obs_frame_indices.npy", observations)
        np.lib.format.open_memmap(root / "points.npy", mode="w+", dtype=np.float32,
                                  shape=(len(observations), args.visual_points + 44, 8)).flush()
        np.lib.format.open_memmap(root / "completed.npy", mode="w+", dtype=bool, shape=(len(observations),)).flush()
        manifest = {"version": 1, "complete": False, "configuration": configuration,
                    "configuration_fingerprint": fingerprint, "split": split,
                    "normalization": training_statistics(arrays["state"], arrays["action"], arrays["tactile"], arrays["episode_index"], split["train"]),
                    "frames": len(arrays["state"]), "observations": len(observations), "total_observations": total_observations,
                    "shard_index": args.shard_index, "num_shards": args.num_shards,
                    "camera_order": list(CAMERAS), "point_channels": ["x", "y", "z", "r", "g", "b", "pressure", "is_tactile"],
                    "coordinate_frame": "robot_base", "base_transform_note": "scene.xml link_base is a root body with identity pose; override for other scenes",
                    "action_semantics": "source parquet labels unchanged; historical converter uses raw action[t+1], repeats last; 22 joint position targets",
                    "tactile_semantics": "corrected44 force44 and anchors_world, 32 taxels then 12 FSRs",
                    "geometry": "VGGT predicted depth/cameras; first known camera fixes orientation/origin, known stereo baseline fixes scale; no GT depth or object pose",
                    "demonstrations": "all source episodes, including failed demonstrations; no success filtering",
                    "source_fps": json.loads((Path(args.source) / "meta/info.json").read_text())["fps"]}
        write_json(manifest_path, manifest)
    points = np.load(root / "points.npy", mmap_mode="r+")
    completed = np.load(root / "completed.npy", mmap_mode="r+")
    if points.shape != (len(observations), args.visual_points + 44, 8) or completed.shape != (len(observations),) or completed.dtype != np.bool_:
        raise ValueError("Cache tensor/bitmap shape or bitmap dtype does not match this shard")
    np.testing.assert_array_equal(np.load(root / "obs_frame_indices.npy"), observations, err_msg="Resume frame selection changed")
    if manifest.get("complete"):
        if not completed.all():
            raise ValueError("Manifest says complete but frame bitmap contains incomplete frames")
        print(f"Already complete: {root}", flush=True)
        return
    remaining = np.flatnonzero(~completed)
    if len(remaining):
        readers = []
        try:
            for path in videos:
                readers.append(VideoReader(path, len(arrays["state"])))
        except Exception:
            for reader in readers:
                reader.close()
            raise
        started = time.monotonic()
        try:
            manifest["rgb_decoder"] = {"backend": "PyAV", "codecs": [reader.codec_name for reader in readers],
                                       "indexing": "sequential decoded-frame ordinal; no timestamp seek or fps conversion"}
            write_json(manifest_path, manifest)
            model = FrozenVGGT(args.vggt_repo, args.weights, args.device, args.image_size)
            with (root / "geometry_diagnostics.jsonl").open("a", buffering=1) as log:
                for offset in range(0, len(remaining), args.batch_size):
                    cache_indices = remaining[offset:offset + args.batch_size]
                    frame_indices = observations[cache_indices]
                    pairs = [[reader.read(int(frame)) for reader in readers] for frame in frame_indices]
                    example_path = root / "example_observation.npz"
                    if not example_path.exists():
                        example_frame = int(frame_indices[0])
                        np.savez_compressed(example_path, rgb=np.asarray(pairs[0]),
                                            state=arrays["state"][example_frame], tactile=arrays["tactile"][example_frame],
                                            anchors_world=anchors[example_frame],
                                            known_cameras_cv=calibrations[int(arrays["episode_index"][example_frame])],
                                            base_to_world=base_transform)
                    rgb, depth, confidence, extrinsics, intrinsics = model.predict(pairs)
                    for b, (cache_index, frame) in enumerate(zip(cache_indices, frame_indices)):
                        ep = int(arrays["episode_index"][frame])
                        scale, rotation, translation, diagnostics = calibrated_similarity(extrinsics[b], calibrations[ep])
                        predicted = unproject_depth(depth[b], extrinsics[b], intrinsics[b])
                        world = scale * predicted @ rotation.T + translation
                        points[cache_index] = build_point_tensor(world, rgb[b], confidence[b], anchors[frame], arrays["tactile"][frame],
                                                               base_to_world=base_transform, visual_count=args.visual_points,
                                                               confidence_quantile=args.confidence_quantile, seed=args.seed + int(frame))
                        # Diagnostic only: never move contacts onto the predicted surface.
                        sample = points[cache_index]
                        distance = np.linalg.norm(sample[-44:, None, :3] - sample[None, :args.visual_points, :3], axis=-1).min(axis=1)
                        active = arrays["tactile"][frame] > 0.5
                        diagnostics["anchor_nearest_visual_median_m"] = float(np.median(distance))
                        diagnostics["active_anchor_count"] = int(active.sum())
                        diagnostics["active_anchor_nearest_visual_median_m"] = float(np.median(distance[active])) if active.any() else None
                        # Persist tensor before marking it valid. Restart may harmlessly redo this frame.
                        points.flush()
                        diagnostics.update(cache_index=int(cache_index), frame=int(frame), episode=ep)
                        log.write(json.dumps(diagnostics) + "\n")
                        completed[cache_index] = True
                        completed.flush()
                    count = min(offset + args.batch_size, len(remaining))
                    print(json.dumps({"shard": args.shard_index, "processed": count, "remaining_total": len(remaining), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        finally:
            for reader in readers:
                reader.close()
    manifest["complete"] = bool(completed.all())
    manifest["completed_observations"] = int(completed.sum())
    write_json(manifest_path, manifest)


def merge_shards(output):
    root = Path(output)
    shards = sorted(root.glob("shard-*"))
    manifests = [json.loads((p / "manifest.json").read_text()) for p in shards]
    if not manifests or len(shards) != manifests[0]["num_shards"]:
        raise ValueError("Missing shards")
    if not all(m["complete"] and m["configuration_fingerprint"] == manifests[0]["configuration_fingerprint"] for m in manifests):
        raise ValueError("Incomplete or incompatible shards")
    indices = np.concatenate([np.load(p / "obs_frame_indices.npy") for p in shards])
    if len(indices) != len(np.unique(indices)) or len(indices) != manifests[0]["total_observations"]:
        raise ValueError("Duplicate or missing observation frames")
    order = np.argsort(indices)
    np.save(root / "obs_frame_indices.npy", indices[order])
    for name in NUMERIC_ARRAYS:
        first = np.load(shards[0] / f"{name}.npy", mmap_mode="r")
        for shard in shards[1:]:
            np.testing.assert_array_equal(first, np.load(shard / f"{name}.npy", mmap_mode="r"))
        np.save(root / f"{name}.npy", first)
    shape = np.load(shards[0] / "points.npy", mmap_mode="r").shape[1:]
    points = np.lib.format.open_memmap(root / "points.npy", mode="w+", dtype=np.float32, shape=(len(indices),) + shape)
    destination = np.argsort(order)
    offset = 0
    with (root / "geometry_diagnostics.jsonl").open("w") as log:
        for shard in shards:
            source = np.load(shard / "points.npy", mmap_mode="r")
            points[destination[offset:offset + len(source)]] = source
            offset += len(source)
            log.write((shard / "geometry_diagnostics.jsonl").read_text())
    points.flush()
    manifest = dict(manifests[0], complete=True, observations=len(indices), completed_observations=len(indices), merged_shards=[str(p) for p in shards])
    manifest.pop("shard_index", None)
    write_json(root / "manifest.json", manifest)
    print(f"Merged {len(shards)} shards: {len(indices)} observations -> {root}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--readouts", type=Path)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vggt-repo", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--obs-stride", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0, help="Cap observation count only; retain all full-rate numeric labels")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--visual-points", type=int, default=512)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--confidence-quantile", type=float, default=0.25)
    p.add_argument("--base-transform", type=Path)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-shards", action="store_true")
    args = p.parse_args()
    if args.merge_shards:
        merge_shards(args.output)
    else:
        for key in ("source", "readouts", "raw_root", "vggt_repo", "weights"):
            if getattr(args, key) is None:
                p.error(f"--{key.replace('_', '-')} is required")
        if args.obs_stride < 1 or args.batch_size < 1 or args.image_size % 14 or not 0 < args.val_fraction < 1:
            p.error("positive stride/batch, image size multiple of 14 and 0<val-fraction<1 required")
        prepare(args)


if __name__ == "__main__":
    main()
