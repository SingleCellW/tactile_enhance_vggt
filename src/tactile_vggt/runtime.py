"""Reusable live encoders for current stereo RGB and tactile observations.

These classes never read object labels. Dataset export remains a CLI concern.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .paths import PROJECT_ROOT as ROOT, resolve_artifact_reference
from .experiment_utils import load_vggt, sha
from .token_fusion import TokenFusionProbe

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
            prefix = resolve_artifact_reference(prefix)
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
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
