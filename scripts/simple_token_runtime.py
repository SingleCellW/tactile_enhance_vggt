"""Frozen RGB/state readout inputs for the minimal ACT follow-up.

No pressure, anchor, camera pose, object label, or simulator truth is read by
this visual path. The existing visual probe supplies only its frozen state
encoder; its small projection is subsequently frozen or action-trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from token_runtime import LiveTokenEncoder, _Captured
from run_token_pilot import dump, sha


class SimpleVisualRuntime:
    def __init__(self, repository, weights, visual_probe_checkpoint, device="cuda:0"):
        self.base = LiveTokenEncoder(repository, weights, {"visual": visual_probe_checkpoint}, device)
        if self.base.probes["visual"].mode != "visual":
            raise ValueError("The source checkpoint must be the touch-free visual probe")
        self.device = device

    @torch.inference_mode()
    def encode(self, rgb, state):
        rgb, state = np.asarray(rgb), np.asarray(state)
        if rgb.ndim == 4:
            rgb, state = rgb[None], state[None]
        b, views = rgb.shape[:2]
        if rgb.shape[2:] != (224, 224, 3) or views != 2 or state.shape != (b, 22):
            raise ValueError("Require synchronized RGB224 stereo and state22")
        captured = {}

        def capture(module, inputs, kwargs):
            captured["tokens"] = inputs[0]
            captured["pos"] = kwargs.get("pos", inputs[1] if len(inputs) > 1 else None)
            raise _Captured

        block = self.base.backbone.aggregator.global_blocks[-1]
        handle = block.register_forward_pre_hook(capture, with_kwargs=True)
        images = torch.as_tensor(rgb, device=self.device).permute(0, 1, 4, 2, 3).float() / 255
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                try:
                    self.base.backbone.aggregator(images)
                except _Captured:
                    pass
        finally:
            handle.remove()
        # Match the original frozen-prefix cache, including its FP16 roundtrip.
        prefix = captured["tokens"].to(torch.float16).float()
        pos = captured["pos"]
        saved = self.base.stats["visual"]
        normalized = (state - np.asarray(saved["state_mean"])) / np.asarray(saved["state_std"])
        probe = self.base.probes["visual"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            tokens = block(prefix, pos=pos).reshape(b, views, -1, 1024)
            pooled = tokens[:, :, 5:].mean(dim=(1, 2))
            state_features = probe.state_encoder(torch.as_tensor(normalized, device=self.device, dtype=torch.float32))
            combined = torch.cat((pooled.to(state_features.dtype), state_features), dim=-1)
        return combined.float().cpu().numpy()

    def encode_observation(self, observation):
        return self.encode(observation["rgb"], observation["state"])[0]


def export_inputs(args):
    from prepare_cache import VideoReader
    cache, output = Path(args.cache), Path(args.output)
    metadata = json.loads((cache / "manifest.json").read_text())
    if not metadata.get("complete") or metadata.get("native_tactile_report", {}).get("velocity_dt") != .01:
        raise ValueError("Require the completed physical dt=.01 cache")
    if (output / "manifest.json").exists() or (output / "readout_inputs.npy").exists():
        raise FileExistsError("Use a fresh feature export directory")
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in
              ("state", "obs_frame_indices", "rgb_source_frame_indices")}
    if not np.array_equal(arrays["rgb_source_frame_indices"], arrays["obs_frame_indices"] + 1):
        raise ValueError("Audited source RGB row mapping changed")
    runtime = SimpleVisualRuntime(args.repository, args.weights, args.visual_probe, args.device)
    n = len(arrays["obs_frame_indices"])
    output.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(output / "readout_inputs.npy", mode="w+", dtype=np.float32, shape=(n, 1088))
    readers = [VideoReader(item["path"], len(arrays["state"])) for item in metadata["configuration"]["source"]["rgb_videos"]]
    reference = np.load(args.reference_features, mmap_mode="r") if args.reference_features else None
    if reference is not None and reference.shape != (n, 256):
        raise ValueError("Reference feature shape changed")
    max_error, squared_error, reference_squared, count = 0., 0., 0., 0
    started = time.monotonic()
    try:
        for start in range(0, n, args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, n))
            rows = arrays["obs_frame_indices"][indices]
            rgb = np.stack([np.stack([reader.read(int(row)) for reader in readers])
                            for row in arrays["rgb_source_frame_indices"][indices]])
            values[indices] = runtime.encode(rgb, arrays["state"][rows])
            if reference is not None:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                    projected = runtime.base.probes["visual"].readout(torch.as_tensor(np.asarray(values[indices]), device=args.device)).float().cpu().numpy()
                diff = projected.astype(np.float64) - reference[indices]
                max_error = max(max_error, float(np.abs(diff).max()))
                squared_error += float((diff ** 2).sum())
                reference_squared += float((reference[indices].astype(np.float64) ** 2).sum())
                count += diff.size
            if start % 200 == 0:
                values.flush()
                print(json.dumps({"export": int(indices[-1] + 1), "total": n, "seconds": time.monotonic() - started}), flush=True)
    finally:
        for reader in readers:
            reader.close()
    values.flush()
    parity = {"reference": str(args.reference_features), "max_abs": max_error,
              "rmse": (squared_error / max(count, 1)) ** .5,
              "relative_l2": (squared_error / max(reference_squared, 1e-12)) ** .5} if reference is not None else None
    # BF16 kernels may differ by a few ULPs; a material discrepancy blocks use.
    if parity is not None and parity["relative_l2"] > .005:
        raise ValueError(f"Frozen readout parity failed: {parity}")
    dump(output / "manifest.json", {"complete": True, "feature_dim": 1088, "observation_count": n,
        "cache_manifest_sha256": sha(cache / "manifest.json"),
        "obs_frame_indices_sha256": sha(cache / "obs_frame_indices.npy"),
        "readout_inputs_sha256": sha(output / "readout_inputs.npy"),
        "source_checkpoint": str(args.visual_probe), "source_checkpoint_sha256": sha(args.visual_probe),
        "weights_sha256": sha(args.weights), "frozen_projection_parity": parity,
        "input_scope": "Current RGB/state only; no touch/anchors/cameras/object GT", "seconds": time.monotonic() - started})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visual-probe", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=ROOT / "vendor/vggt")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    p.add_argument("--reference-features", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    export_inputs(args)
