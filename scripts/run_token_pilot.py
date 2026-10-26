#!/usr/bin/env python3
"""Small, episode-held-out VGGT internal tactile adapter experiment.

The frozen visual prefix is cached BEFORE a real pretrained global block. GT
object positions are supervision only; neither these labels nor segmentation
enter the encoder. Training selection uses dev episodes, never the test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, dependency_path
from tactile_vggt.provenance import write_run_manifest


from tactile_vggt.experiment_utils import dump, sha, seed_all, load_vggt


def cache_prefix(args):
    args.prefix.mkdir(parents=True, exist_ok=True)
    metadata_path = args.prefix / "manifest.json"
    if metadata_path.exists():
        m = json.loads(metadata_path.read_text())
        if m.get("complete"):
            validate_prefix(args)
            print("Prefix already complete", flush=True)
            return
    rgb = np.load(args.data / "rgb.npy", mmap_mode="r")
    model = load_vggt(args.repository, args.weights, args.device)
    layer = len(model.aggregator.global_blocks) - 1
    torch.save(model.aggregator.global_blocks[layer].cpu().state_dict(), args.prefix / "global_block.pt")
    model.aggregator.global_blocks[layer].to(args.device)
    captured = {}

    class PrefixCaptured(Exception):
        pass

    def hook(module, inputs, kwargs):
        captured["tokens"] = inputs[0].detach()
        captured["pos"] = kwargs.get("pos", inputs[1] if len(inputs) > 1 else None)
        raise PrefixCaptured

    handle = model.aggregator.global_blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
    tokens_file = None
    started = time.monotonic()
    size = args.image_size
    for start in range(0, len(rgb), args.extract_batch):
        images = torch.from_numpy(np.array(rgb[start:start + args.extract_batch])).permute(0, 1, 4, 2, 3).to(args.device).float() / 255
        if images.shape[-2:] != (size, size):
            b, s, c, h, w = images.shape
            images = torch.nn.functional.interpolate(images.reshape(b*s, c, h, w), (size, size), mode="bilinear", align_corners=False).reshape(b, s, c, size, size)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                model.aggregator(images)
            except PrefixCaptured:
                pass
        token = captured.pop("tokens")
        b, s = images.shape[:2]
        token = token.reshape(b, s, -1, token.shape[-1]).float().cpu().numpy().astype(np.float16)
        pos = captured.pop("pos")
        if tokens_file is None:
            shape = (len(rgb), *token.shape[1:])
            tokens_file = np.lib.format.open_memmap(args.prefix / "tokens.npy", mode="w+", dtype=np.float16, shape=shape)
            np.save(args.prefix / "positions.npy", pos[0].reshape(s, -1, 2).cpu().numpy())
        tokens_file[start:start + b] = token
        if start % max(args.extract_batch, 100) == 0:
            tokens_file.flush()
            print(json.dumps({"extract": start + b, "total": len(rgb), "seconds": time.monotonic()-started}), flush=True)
    tokens_file.flush()
    handle.remove()
    dump(metadata_path, {"complete": True, "count": len(rgb), "shape": list(tokens_file.shape),
         "layer": layer, "image_size": size, "patch_start": model.aggregator.patch_start_idx,
         "data_manifest_sha256": sha(args.data / "manifest.json"), "weights_sha256": sha(args.weights),
         "block_sha256": sha(args.prefix / "global_block.pt"), "seconds": time.monotonic()-started,
         "description": "Actual pretrained VGGT activation immediately before its final global attention block"})


def validate_prefix(args):
    metadata = json.loads((args.prefix / "manifest.json").read_text())
    source = json.loads((args.data / "manifest.json").read_text())
    if not metadata.get("complete") or not source.get("complete"):
        raise ValueError("Data and prefix must both be complete")
    if metadata["data_manifest_sha256"] != sha(args.data / "manifest.json"):
        raise ValueError("Prefix belongs to different source data")
    if metadata["weights_sha256"] != sha(args.weights):
        raise ValueError("Prefix was extracted with different pretrained weights")
    if metadata["block_sha256"] != sha(args.prefix / "global_block.pt"):
        raise ValueError("Frozen block changed after extraction")
    if metadata["image_size"] != args.image_size:
        raise ValueError("Prefix image size does not match requested image size")
    rgb = np.load(args.data / "rgb.npy", mmap_mode="r")
    tokens = np.load(args.prefix / "tokens.npy", mmap_mode="r")
    if len(rgb) != metadata["count"] or list(tokens.shape) != metadata["shape"] or len(tokens) != len(rgb):
        raise ValueError("Source and prefix shapes/counts disagree")
    return metadata


def make_block(repository, prefix, device):
    sys.path.insert(0, str(repository))
    from vggt.layers.block import Block
    from vggt.layers.rope import RotaryPositionEmbedding2D
    block = Block(dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True, proj_bias=True,
                  ffn_bias=True, init_values=.01, qk_norm=True, rope=RotaryPositionEmbedding2D(frequency=100))
    block.load_state_dict(torch.load(prefix / "global_block.pt", map_location="cpu", weights_only=True))
    return block.to(device).eval().requires_grad_(False)


class PilotArrays:
    def __init__(self, data, prefix):
        self.tokens = np.load(prefix / "tokens.npy", mmap_mode="r")
        self.device_tokens = None
        self.pos = np.load(prefix / "positions.npy")
        self.episode = np.load(data / "episode.npy")
        self.groups = np.load(data / "group.npy")
        self.split = np.load(data / "split.npy")
        self.pressure = np.load(data / "pressure.npy")
        self.flags = (self.pressure > 5e-4).astype(np.float32)
        self.state = np.load(data / "state.npy")
        cameras = np.load(data / "known_cameras.npy")
        self.baseline = np.linalg.norm(cameras[:, 1, :3, 3] - cameras[:, 0, :3, 3], axis=-1)
        if (self.baseline < 1e-5).any():
            raise ValueError("Degenerate physical baseline")
        origin, rotation = cameras[:, 0, :3, 3], cameras[:, 0, :3, :3]
        anchors = np.load(data / "anchors.npy")
        self.anchors = np.einsum("bnj,bjk->bnk", anchors-origin[:, None], rotation) / self.baseline[:, None, None]
        labels = np.load(data / "labels.npy")
        palm = np.load(data / "palm.npy")[:, :3, 3]
        self.target = np.einsum("bj,bjk->bk", labels-palm, rotation) / self.baseline[:, None]
        self.variant = np.load(data / "variant.npy")
        self.occlusion = np.load(data / "occlusion.npy")
        train = self.split == 0
        self.target_mean = self.target[train].mean(0)
        self.target_std = np.maximum(self.target[train].std(0), 1e-3)
        self.state_mean = self.state[train].mean(0)
        self.state_std = np.maximum(self.state[train].std(0), 1e-3)
        self.pressure_scale = float(max(np.quantile(np.log1p(np.maximum(self.pressure[train], 0)), .99), 1e-3))
        self.state = (self.state-self.state_mean)/self.state_std
        self.pressure = np.log1p(np.maximum(self.pressure, 0))/self.pressure_scale

    def batch(self, indices, device, mismatch=False, zero_pressure=False, rng=None):
        pressure = self.pressure[indices].copy()
        anchors = self.anchors[indices].copy()
        flags = self.flags[indices].copy()
        if mismatch:
            rng = np.random.default_rng(123) if rng is None else rng
            for row in range(len(indices)):
                perm = rng.permutation(44)
                pressure[row], flags[row] = pressure[row, perm], flags[row, perm]
        if zero_pressure:
            pressure.fill(0)
            flags.fill(0)
        tensor = lambda x: torch.as_tensor(np.array(x), device=device, dtype=torch.float32)
        tokens = (tensor(self.tokens[indices]) if self.device_tokens is None else
                  self.device_tokens.index_select(0, torch.as_tensor(indices, device=device)).float())
        return {"prefix_tokens": tokens,
                "pos": torch.as_tensor(self.pos, device=device).unsqueeze(0).expand(len(indices), -1, -1, -1),
                "pressure": tensor(pressure), "anchors": tensor(anchors),
                "pressure_flags": tensor(flags), "state": tensor(self.state[indices])}, tensor(self.target[indices])


@torch.no_grad()
def predict(model, data, indices, args, **kwargs):
    model.eval()
    predictions = []
    for start in range(0, len(indices), args.batch_size):
        batch, target = data.batch(indices[start:start+args.batch_size], args.device, **kwargs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**batch)
        predicted = result["prediction"].float().cpu().numpy()
        if args.whiten_target:
            predicted = predicted * data.target_std + data.target_mean
        predictions.append(predicted)
    return np.concatenate(predictions)


def train(args):
    from tactile_vggt.token_fusion import TokenFusionProbe
    from tactile_vggt.token_metrics import report_metrics, paired_group_bootstrap, euclidean_errors_mm
    args.output.mkdir(parents=True, exist_ok=True)
    validate_prefix(args)
    data = PilotArrays(args.data, args.prefix)
    # The 775-example prefix is <1 GB in fp16. Keep it on the reserved A40 to
    # avoid repeatedly transferring large pageable CPU arrays every SGD step.
    data.device_tokens = torch.from_numpy(np.array(data.tokens)).to(args.device)
    write_run_manifest(args.output, args)
    train_ids, dev_ids, test_ids = [np.flatnonzero(data.split == value) for value in range(3)]
    if min(map(len, (train_ids, dev_ids, test_ids))) == 0:
        raise ValueError("Empty split")
    all_results, predictions = {}, {}
    for mode in args.modes:
        seed_all(args.seed)
        block = make_block(args.repository, args.prefix, args.device)
        kwargs = {"mode": "internal" if mode in ("no_xyz", "mismatch") else mode,
                  "use_xyz": mode != "no_xyz", "hidden_dim": args.hidden_dim,
                  "feature_dim": 256, "initial_gate": .01}
        model = TokenFusionProbe(block, **kwargs).to(args.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        perturb_rng = np.random.default_rng(args.seed + 100_000)
        run = args.output / mode
        run.mkdir(exist_ok=True)
        best, best_step = float("inf"), 0
        started = time.monotonic()
        history = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, args.steps+1):
            model.train()
            ids = rng.choice(train_ids, args.batch_size, replace=True)
            batch, target = data.batch(ids, args.device, mismatch=mode == "mismatch", rng=perturb_rng)
            if args.whiten_target:
                target = (target - torch.as_tensor(data.target_mean, device=args.device)) / torch.as_tensor(data.target_std, device=args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**batch)
                loss = torch.nn.functional.smooth_l1_loss(result["prediction"].float(), target, beta=1.0 if args.whiten_target else .02)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at {mode}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                pred = predict(model, data, dev_ids, args, mismatch=mode == "mismatch")
                errors = np.linalg.norm(pred-data.target[dev_ids], axis=-1)*data.baseline[dev_ids]*1000
                dev = float(np.mean([errors[data.episode[dev_ids] == ep].mean() for ep in np.unique(data.episode[dev_ids])]))
                record = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad), "dev_episode_mean_mm": dev,
                          "seconds": time.monotonic()-started}
                history.append(record)
                print(json.dumps({"mode": mode, **record}), flush=True)
                if dev < best:
                    best, best_step = dev, step
                    torch.save({"model": {k:v.cpu() for k,v in model.state_dict().items() if not k.startswith("frozen_global_block.")},
                                "kwargs": kwargs, "step": step, "dev_episode_mean_mm": dev,
                                "state_mean": data.state_mean, "state_std": data.state_std, "pressure_scale": data.pressure_scale,
                                "whiten_target": args.whiten_target, "target_mean": data.target_mean, "target_std": data.target_std}, run / "best.pt")
                dump(run / "history.json", history)
        checkpoint = torch.load(run / "best.pt", map_location=args.device, weights_only=False)
        result = model.load_state_dict(checkpoint["model"], strict=False)
        # Only frozen block state is intentionally excluded from the small checkpoint.
        if result.unexpected_keys or any(not k.startswith("frozen_global_block.") for k in result.missing_keys):
            raise ValueError((result.missing_keys, result.unexpected_keys))
        pred = predict(model, data, test_ids, args, mismatch=mode == "mismatch")
        predictions[mode] = pred
        masks = {"nominal": data.variant[test_ids] == 0, "counterfactual": data.variant[test_ids] != 0,
                 "contact": data.flags[test_ids].any(-1), "no_contact": ~data.flags[test_ids].any(-1).astype(bool),
                 "high_occlusion": np.mean(data.occlusion[test_ids], axis=-1) >= .5,
                 "low_occlusion": np.mean(data.occlusion[test_ids], axis=-1) < .5}
        metrics = report_metrics(pred, data.target[test_ids], data.episode[test_ids], data.baseline[test_ids],
                                 groups=data.groups[test_ids], subgroups=masks)
        metrics.update({"best_step": best_step, "dev_episode_mean_mm": best, "train_steps": args.steps,
                        "seconds": time.monotonic()-started, "peak_gpu_gb": torch.cuda.max_memory_allocated()/1e9,
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        if mode in ("internal", "late", "no_xyz", "tactile"):
            perturbed = predict(model, data, test_ids, args, mismatch=True)
            metrics["test_mismatched_pressure"] = report_metrics(perturbed, data.target[test_ids], data.episode[test_ids], data.baseline[test_ids])
            without = predict(model, data, test_ids, args, zero_pressure=True)
            metrics["test_zero_pressure"] = report_metrics(without, data.target[test_ids], data.episode[test_ids], data.baseline[test_ids])
        all_results[mode] = metrics
        np.savez(run / "test_predictions.npz", prediction=pred, target=data.target[test_ids], episode=data.episode[test_ids],
                 group=data.groups[test_ids], baseline_m=data.baseline[test_ids], indices=test_ids)
        dump(run / "metrics.json", metrics)
        del model, block, optimizer
        torch.cuda.empty_cache()
    comparisons = {}
    for name, pred in predictions.items():
        if name != "internal" and "internal" in predictions:
            comparisons["internal_minus_"+name] = paired_group_bootstrap(
                euclidean_errors_mm(predictions["internal"], data.target[test_ids], data.baseline[test_ids]),
                euclidean_errors_mm(pred, data.target[test_ids], data.baseline[test_ids]), data.episode[test_ids])
    dump(args.output / "summary.json", {"args": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         "splits": {"train": len(train_ids), "dev": len(dev_ids), "test": len(test_ids)},
         "data_manifest_sha256": sha(args.data / "manifest.json"), "prefix_manifest_sha256": sha(args.prefix / "manifest.json"),
         "implementation_sha256": sha(__file__),
         "results": all_results, "comparisons": comparisons,
         "scope": "Object-center relative-to-palm regression, not full reconstruction or policy success"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["extract", "train"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    p.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    p.add_argument("--output", type=Path, default=artifact_path("outputs") / "token_pilot_20261025")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--extract-batch", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--whiten-target", action="store_true", help="Standardize regression target using train episodes only")
    p.add_argument("--modes", nargs="+", default=["state", "visual", "tactile", "late", "internal", "no_xyz", "mismatch"])
    args = p.parse_args()
    if not args.device.startswith("cuda"):
        p.error("This pilot runner requires CUDA; CPU correctness is covered by unit tests")
    torch.set_num_threads(4)
    if args.command == "extract":
        cache_prefix(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
