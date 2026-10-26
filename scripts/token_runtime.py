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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_token_pilot import load_vggt, dump, sha
from tactile_vggt.token_fusion import TokenFusionProbe


class _Captured(Exception):
    pass


class LiveTokenEncoder:
    def __init__(self, repository, weights, checkpoints, device="cuda:0"):
        self.device = device
        weights_digest = sha(weights)
        for path in checkpoints.values():
            path = Path(path).resolve()
            summary_path = path.parent.parent / "summary.json"
            if not summary_path.exists():
                raise ValueError(f"Require completed probe-run summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            prefix = Path(summary["args"]["prefix"])
            prefix = prefix if prefix.is_absolute() else ROOT / prefix
            manifest = prefix / "manifest.json"
            if sha(manifest) != summary["prefix_manifest_sha256"]:
                raise ValueError("Probe prefix manifest changed since training")
            source = json.loads(manifest.read_text())
            if not source.get("complete") or source["image_size"] != 224 or source["weights_sha256"] != weights_digest:
                raise ValueError("Live VGGT weights or image size differ from representation training")
        self.backbone = load_vggt(Path(repository), Path(weights), device)
        self.probes, self.stats = {}, {}
        for name, path in checkpoints.items():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = TokenFusionProbe(self.backbone.aggregator.global_blocks[-1], **saved["kwargs"])
            incompatible = model.load_state_dict(saved["model"], strict=False)
            if incompatible.unexpected_keys or any(not k.startswith("frozen_global_block.") for k in incompatible.missing_keys):
                raise ValueError("Probe checkpoint does not match its recorded architecture")
            self.probes[name] = model.to(device).eval().requires_grad_(False)
            self.stats[name] = saved

    @torch.inference_mode()
    def encode(self, rgb, state, pressure, anchors_base, cameras_base):
        rgb, state, pressure = map(np.asarray, (rgb, state, pressure))
        anchors_base, cameras_base = map(np.asarray, (anchors_base, cameras_base))
        if rgb.ndim == 4:
            rgb, state, pressure = rgb[None], state[None], pressure[None]
            anchors_base, cameras_base = anchors_base[None], cameras_base[None]
        b, s = rgb.shape[:2]
        if rgb.shape[2:] != (224, 224, 3) or s != 2 or state.shape != (b, 22) or pressure.shape != (b, 44):
            raise ValueError("Live token pilot requires synchronized RGB224 stereo/state22/pressure44")
        captured = {}

        def hook(module, inputs, kwargs):
            captured["prefix"] = inputs[0]
            captured["pos"] = kwargs.get("pos", inputs[1] if len(inputs) > 1 else None)
            raise _Captured

        handle = self.backbone.aggregator.global_blocks[-1].register_forward_pre_hook(hook, with_kwargs=True)
        images = torch.as_tensor(rgb, device=self.device).permute(0, 1, 4, 2, 3).float()/255
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    self.backbone.aggregator(images)
                except _Captured:
                    pass
        finally:
            handle.remove()
        prefix = captured["prefix"].reshape(b, s, -1, 1024)
        # Match the frozen prefix cache quantization used in probe training.
        prefix = prefix.to(torch.float16).float()
        pos = captured["pos"].reshape(b, s, -1, 2)
        baseline = np.linalg.norm(cameras_base[:, 1, :3, 3]-cameras_base[:, 0, :3, 3], axis=-1)
        if (baseline < 1e-5).any():
            raise ValueError("Degenerate camera baseline")
        anchors = np.einsum("bnj,bjk->bnk", anchors_base-cameras_base[:, 0, None, :3, 3],
                            cameras_base[:, 0, :3, :3])/baseline[:, None, None]
        flags = torch.as_tensor(pressure > 5e-4, device=self.device)
        result = {}
        for name, probe in self.probes.items():
            stats = self.stats[name]
            normalized_state = (state-np.asarray(stats["state_mean"]))/np.asarray(stats["state_std"])
            normalized_pressure = np.log1p(np.maximum(pressure, 0))/stats["pressure_scale"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = probe(prefix, pos,
                    torch.as_tensor(normalized_pressure, device=self.device, dtype=torch.float32),
                    torch.as_tensor(anchors, device=self.device, dtype=torch.float32),
                    pressure_flags=flags,
                    state=torch.as_tensor(normalized_state, device=self.device, dtype=torch.float32))
            result[name] = output["features"].float().cpu().numpy()
        return result

    def encode_observation(self, observation):
        transform = observation["base_to_world"]
        rotation, origin = transform[:3, :3], transform[:3, 3]
        anchors = (observation["anchors_world"]-origin) @ rotation
        cameras = observation["known_cameras_cv"].copy()
        cameras[:, :3, :3] = np.einsum("ij,vjk->vik", rotation.T, cameras[:, :3, :3])
        cameras[:, :3, 3] = (cameras[:, :3, 3]-origin) @ rotation
        return {name: x[0] for name, x in self.encode(observation["rgb"], observation["state"],
                     observation["tactile"], anchors, cameras).items()}


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
    p.add_argument("--repository", type=Path, default=ROOT/"vendor/vggt")
    p.add_argument("--weights", type=Path, default=ROOT/"weights/model.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    args.checkpoints = {k:Path(v) for k,v in (value.split("=", 1) for value in args.checkpoint)}
    torch.set_num_threads(4)
    export_features(args)
