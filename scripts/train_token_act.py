#!/usr/bin/env python3
"""Short ACT feasibility training on frozen 256D token-representation caches.

Rows of features.npy must match the ORIGINAL cache's obs_frame_indices.npy
exactly. Original episode splits, train-only normalization, full-rate action
labels and episode-safe padding are reused without any relabeling.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot" / "src")]

import numpy as np
import torch
from torch.utils.data import DataLoader

from tactile_vggt.data import CachedTactileDataset, action_chunk
from tactile_vggt.token_policy import TokenACTPolicy
from train_act import ResumeBatchSampler, amp_context, jsonable, move_batch, validate, write_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class CachedTokenFeatureDataset(CachedTactileDataset):
    """Join frozen feature rows to original observations, never by split order."""

    def __init__(self, cache_dir, features_dir, split="train", horizon=16):
        super().__init__(cache_dir, split=split, horizon=horizon, normalize=True)
        self.features_dir = Path(features_dir)
        self.feature_manifest = json.loads((self.features_dir / "manifest.json").read_text())
        expected = {
            "complete": True,
            "cache_manifest_sha256": digest(self.cache_dir / "manifest.json"),
            "obs_frame_indices_sha256": digest(self.cache_dir / "obs_frame_indices.npy"),
            "observation_count": len(self.obs_frame_indices),
            "feature_dim": 256,
        }
        for name, value in expected.items():
            if self.feature_manifest.get(name) != value:
                raise ValueError(f"Feature manifest does not match original cache: {name}")
        self.features = np.load(self.features_dir / "features.npy", mmap_mode="r", allow_pickle=False)
        if self.features.shape != (len(self.obs_frame_indices), 256):
            raise ValueError("features.npy must contain [original observation count, 256] in original row order")
        if self.features.dtype.kind != "f" or not np.isfinite(self.features).all():
            raise ValueError("features.npy must contain finite floating-point values")
        recorded_digest = self.feature_manifest.get("features_sha256")
        if recorded_digest is not None and recorded_digest != digest(self.features_dir / "features.npy"):
            raise ValueError("features.npy checksum differs from its manifest")

    def __getitem__(self, index):
        cache_index = int(self.observation_indices[index])
        frame = int(self.obs_frame_indices[cache_index])
        state = np.array(self.state[frame], dtype=np.float32, copy=True)
        action, padding = action_chunk(self.action, self.episode_index, frame, self.horizon)
        state = (state - np.asarray(self.stats["state_mean"], dtype=np.float32)) / np.asarray(self.stats["state_std"], dtype=np.float32)
        action = (action - np.asarray(self.stats["action_mean"], dtype=np.float32)) / np.asarray(self.stats["action_std"], dtype=np.float32)
        return {
            "observation.environment_state": torch.from_numpy(np.array(self.features[cache_index], dtype=np.float32, copy=True)),
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "action_is_pad": torch.from_numpy(padding),
            "episode_index": int(self.episode_index[frame]),
            "frame_index": int(self.frame_index[frame]),
        }


def update_link(output, name, destination):
    temporary = output / f".{name}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(destination.relative_to(output), target_is_directory=True)
    temporary.replace(output / name)


def save_checkpoint(policy, optimizer, output, step, config, dataset, best_val):
    destination = output / "checkpoints" / f"{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    staging = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(staging)
    for name, value in (("normalization", dataset.stats), ("data_manifest", dataset.metadata),
                        ("feature_manifest", dataset.feature_manifest), ("train_config", config)):
        write_json(staging / f"{name}.json", value)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "best_val": best_val,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               staging / "training_state.pt")
    write_json(staging / "checkpoint_complete.json", {"step": step, "frozen_representation": True})
    staging.rename(destination)
    update_link(output, "latest", destination)
    return destination


def run(args):
    for name in ("steps", "seed", "batch_size", "eval_every", "eval_batches", "save_every", "log_every", "cpu_threads"):
        if getattr(args, name) < (0 if name == "seed" else 1):
            raise ValueError(f"--{name.replace('_', '-')} has an invalid value")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("train_config.json", "status.json", "metrics.jsonl", "checkpoints")):
        raise ValueError(f"Output already contains a run; choose a fresh directory: {output}")
    source = json.loads(Path(args.config).read_text())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.amp == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is unsupported on the selected CUDA device")
    policy_config = dict(source["policy"])
    policy_config.update(encoder_dim=256, device=str(device))
    train = CachedTokenFeatureDataset(args.cache, args.features_dir, "train", policy_config["chunk_size"])
    val = CachedTokenFeatureDataset(args.cache, args.features_dir, "val", policy_config["chunk_size"])
    if len(train) < args.batch_size or not len(val):
        raise ValueError("Need a full original training batch and held-out validation observations")
    if jsonable(train.stats) != jsonable(val.stats):
        raise ValueError("Train and validation normalization must match")
    policy = TokenACTPolicy(policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=source["learning_rate"], betas=(0.9, 0.999),
                                  weight_decay=source["weight_decay"], fused=device.type == "cuda")
    config = {**source, "policy": policy.config, "steps": args.steps, "seed": args.seed,
              "batch_size": args.batch_size, "amp": args.amp, "eval_every": args.eval_every,
              "eval_batches": args.eval_batches, "save_every": args.save_every,
              "num_workers": args.num_workers, "cache": str(Path(args.cache).resolve()),
              "features_dir": str(Path(args.features_dir).resolve()), "torch": torch.__version__,
              "train_observations": len(train), "val_observations": len(val),
              "data_manifest_sha256": digest(Path(args.cache) / "manifest.json"),
              "feature_manifest_sha256": digest(Path(args.features_dir) / "manifest.json"),
              "features_sha256": digest(Path(args.features_dir) / "features.npy"),
              "feature_manifest": train.feature_manifest,
              "scope": "frozen 256D representation -> train ACT head only; unchanged original action labels/split",
              "argv": sys.argv, "sampler_version": 1,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    write_json(output / "train_config.json", config)
    write_json(output / "normalization.json", train.stats)
    write_json(output / "feature_manifest.json", train.feature_manifest)
    kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
              "persistent_workers": args.num_workers > 0}
    sampler = ResumeBatchSampler(len(train), args.batch_size, args.seed)
    loader = DataLoader(train, batch_sampler=sampler,
                        generator=torch.Generator().manual_seed(args.seed + 100_000), **kwargs)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 200_000), **kwargs)
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    print(json.dumps({"event": "started", "trainable_parameters": trainable,
                      "train_observations": len(train), "val_observations": len(val),
                      "feature_mode": train.feature_manifest.get("mode"), "steps": args.steps}), flush=True)
    iterator, durations = iter(loader), deque(maxlen=200)
    best_val, last_val = float("inf"), None
    started = time.monotonic()
    policy.train()
    with (output / "metrics.jsonl").open("x", buffering=1) as metrics:
        for step in range(1, args.steps + 1):
            tick = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.next_epoch()
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp, device):
                loss, parts = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), source["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.monotonic() - tick)
            improved = False
            if step % args.eval_every == 0 or step == args.steps:
                last_val = validate(policy, val_loader, device, args.amp, args.eval_batches)
                improved = last_val < best_val
                best_val = min(best_val, last_val)
            if step % args.save_every == 0 or step == args.steps or improved:
                checkpoint = save_checkpoint(policy, optimizer, output, step, config, train, best_val)
                if improved:
                    update_link(output, "best", checkpoint)
                print(json.dumps({"event": "checkpoint", "step": step, "path": str(checkpoint)}), flush=True)
            if step <= 3 or step % args.log_every == 0 or step % args.eval_every == 0 or step == args.steps:
                seconds_per_step = float(np.mean(durations))
                status = {"status": "completed" if step == args.steps else "running", "step": step,
                          "total_steps": args.steps, "loss": loss.item(), **parts, "grad_norm": float(gradient),
                          "val_normalized_action_l1": last_val, "best_val_normalized_action_l1": best_val if last_val is not None else None,
                          "seconds_per_step": seconds_per_step,
                          "eta_seconds_compute_only": (args.steps - step) * seconds_per_step,
                          "elapsed_seconds": time.monotonic() - started, "trainable_parameters": trainable,
                          "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                          "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                          "pid": os.getpid()}
                metrics.write(json.dumps(status, allow_nan=False) + "\n")
                write_json(output / "status.json", status)
                print(json.dumps(status, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/act_physics_aligned_v2.json"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
