"""Live current-snapshot VGGT features; no object labels enter this path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, dependency_path
sys.path[:0] = [str(ROOT / "scripts")]
from tactile_vggt.experiment_utils import dump, sha
from tactile_vggt.token_fusion import TokenFusionProbe


from tactile_vggt.runtime import LiveTokenEncoder, _Captured


def export_features(args):
    from prepare_cache import VideoReader, CAMERAS
    from tactile_vggt.geometry import mujoco_camera_to_opencv
    cache = Path(args.cache)
    metadata = json.loads((cache/"manifest.json").read_text())
    if not metadata.get("complete") or metadata.get("native_tactile_report", {}).get("velocity_dt") != .01:
        raise ValueError("Export requires completed physical dt=.01 cache")
    arrays = {name: np.load(cache/f"{name}.npy", mmap_mode="r") for name in
              ("state", "tactile", "points", "episode_index", "obs_frame_indices", "rgb_source_frame_indices")}
    source = metadata["configuration"]["source"]
    calibrations = {}
    for ep, info in zip(np.unique(arrays["episode_index"]), source["camera_metadata"]):
        cal = json.loads(Path(info["path"]).read_text())
        poses = mujoco_camera_to_opencv(np.array([cal["cameras"][name]["extrinsics"] for name in CAMERAS]))
        base = np.asarray(metadata["configuration"]["base_to_world"])
        poses[:, :3, :3] = np.einsum("ij,vjk->vik", base[:3, :3].T, poses[:, :3, :3])
        poses[:, :3, 3] = (poses[:, :3, 3]-base[:3, 3]) @ base[:3, :3]
        calibrations[int(ep)] = poses
    encoder = LiveTokenEncoder(args.repository, args.weights, args.checkpoints, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    n = len(arrays["obs_frame_indices"])
    outputs = {}
    for name in args.checkpoints:
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        if (directory/"manifest.json").exists():
            raise FileExistsError("Use a fresh feature export directory")
        outputs[name] = np.lib.format.open_memmap(directory/"features.npy", mode="w+", dtype=np.float32, shape=(n, 256))
    readers = [VideoReader(item["path"], len(arrays["state"])) for item in source["rgb_videos"]]
    started = time.monotonic()
    try:
        for start in range(0, n, args.batch_size):
            indices = np.arange(start, min(start+args.batch_size, n))
            rows = arrays["obs_frame_indices"][indices]
            rgb = np.stack([np.stack([reader.read(int(row)) for reader in readers]) for row in arrays["rgb_source_frame_indices"][indices]])
            if not np.array_equal(arrays["rgb_source_frame_indices"][indices], rows+1):
                raise ValueError("Historical RGB alignment changed")
            cameras = np.stack([calibrations[int(arrays["episode_index"][row])] for row in rows])
            features = encoder.encode(rgb, arrays["state"][rows], arrays["tactile"][rows], arrays["points"][indices, -44:, :3], cameras)
            for name in outputs:
                outputs[name][indices] = features[name]
            if start % 200 == 0:
                for value in outputs.values():
                    value.flush()
                print(json.dumps({"export": int(indices[-1]+1), "total": n, "seconds": time.monotonic()-started}), flush=True)
    finally:
        for reader in readers:
            reader.close()
    for name, output in outputs.items():
        output.flush()
        dump(args.output/name/"manifest.json", {"complete": True, "feature_dim": 256,
            "observation_count": n, "cache_manifest_sha256": sha(cache/"manifest.json"),
            "obs_frame_indices_sha256": sha(cache/"obs_frame_indices.npy"),
            "features_sha256": sha(args.output/name/"features.npy"), "mode": name,
            "source_checkpoint": str(args.checkpoints[name]), "source_checkpoint_sha256": sha(args.checkpoints[name]),
            "seconds": time.monotonic()-started, "input_scope": "Current RGB/state/touch/FK/camera only; no object GT"})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="mode=best.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    p.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    args.checkpoints = {k:Path(v) for k,v in (value.split("=", 1) for value in args.checkpoint)}
    torch.set_num_threads(4)
    export_features(args)
