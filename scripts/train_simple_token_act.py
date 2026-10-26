#!/usr/bin/env python3
"""Matched 2x2 ACT study: frozen/joint visual projection x zero/pressure44.

The source VGGT tokens and original state encoder stay frozen and cached.
Original observation rows, episode splits, state/action statistics and full-rate
source action labels are reused unchanged. Only the original visual readout and
ACT can receive gradients, according to the selected projection condition.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime
import json
import math
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
from tactile_vggt.simple_token_policy import SimpleTokenACTPolicy, READOUT_INPUT_DIM
from train_act import ResumeBatchSampler, amp_context, jsonable, move_batch, validate, write_json
from train_token_act import digest, update_link


class CachedSimpleTokenDataset(CachedTactileDataset):
    """Readout inputs joined by original observation row, with train-only pressure scale."""

    def __init__(self, cache_dir, features_dir, split="train", horizon=16, *, visual_probe=None):
        super().__init__(cache_dir, split=split, horizon=horizon, normalize=True)
        self.features_dir = Path(features_dir)
        self.feature_manifest = json.loads((self.features_dir / "manifest.json").read_text())
        expected = {
            "complete": True,
            "cache_manifest_sha256": digest(self.cache_dir / "manifest.json"),
            "obs_frame_indices_sha256": digest(self.cache_dir / "obs_frame_indices.npy"),
            "observation_count": len(self.obs_frame_indices),
            "feature_dim": READOUT_INPUT_DIM,
            "readout_inputs_sha256": digest(self.features_dir / "readout_inputs.npy"),
        }
        for name, value in expected.items():
            if self.feature_manifest.get(name) != value:
                raise ValueError(f"Feature manifest does not match original cache/readout inputs: {name}")
        if not isinstance(self.feature_manifest.get("source_checkpoint_sha256"), str):
            raise ValueError("Feature manifest must bind source_checkpoint_sha256")
        if visual_probe is not None and self.feature_manifest["source_checkpoint_sha256"] != digest(visual_probe):
            raise ValueError("Feature source_checkpoint_sha256 does not match --visual-probe")
        self.readout_inputs = np.load(self.features_dir / "readout_inputs.npy", mmap_mode="r", allow_pickle=False)
        if self.readout_inputs.shape != (len(self.obs_frame_indices), READOUT_INPUT_DIM):
            raise ValueError("readout_inputs.npy must contain [original observation count, 1088]")
        if self.readout_inputs.dtype.kind != "f" or not np.isfinite(self.readout_inputs).all():
            raise ValueError("readout_inputs.npy must contain finite floating-point values")
        train_rows = np.isin(self.episode_index[self.obs_frame_indices], self.metadata["split"]["train"])
        raw = np.asarray(self.tactile[self.obs_frame_indices[train_rows]], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 44 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("Original training observations must contain finite pressure44")
        self.stats = copy.deepcopy(self.stats)
        self.stats["simple_pressure_scale"] = float(max(np.quantile(np.log1p(np.maximum(raw, 0)), 0.99), 1e-3))
        self.stats["simple_pressure_training_observations"] = int(train_rows.sum())
        self.stats["simple_pressure_normalization"] = "log1p(max(raw,0))/p99; scalar p99 over pressure44 at original ACT training observation rows; floor=1e-3"

    def __getitem__(self, index):
        cache_index = int(self.observation_indices[index])
        frame = int(self.obs_frame_indices[cache_index])
        state = np.array(self.state[frame], dtype=np.float32, copy=True)
        pressure = np.array(self.tactile[frame], dtype=np.float32, copy=True)
        if not np.isfinite(pressure).all():
            raise ValueError("Observation contains nonfinite pressure")
        actions, padding = action_chunk(self.action, self.episode_index, frame, self.horizon)
        state = (state - np.asarray(self.stats["state_mean"], dtype=np.float32)) / np.asarray(self.stats["state_std"], dtype=np.float32)
        actions = (actions - np.asarray(self.stats["action_mean"], dtype=np.float32)) / np.asarray(self.stats["action_std"], dtype=np.float32)
        pressure = np.log1p(np.maximum(pressure, 0)) / self.stats["simple_pressure_scale"]
        return {
            "observation.visual_readout_input": torch.from_numpy(np.array(self.readout_inputs[cache_index], dtype=np.float32, copy=True)),
            "observation.tactile": torch.from_numpy(pressure.astype(np.float32)),
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "action": torch.from_numpy(actions.astype(np.float32)),
            "action_is_pad": torch.from_numpy(padding),
            "episode_index": int(self.episode_index[frame]), "frame_index": int(self.frame_index[frame]),
        }


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
    write_json(staging / "checkpoint_complete.json", {
        "step": step, "policy_type": "simple_token_act", "structure": policy.structure,
        "upstream_vggt_frozen": True, "source_state_encoder_frozen": True,
        "readout_trainable": policy.structure["projection"] == "joint", "sampler_version": 1,
    })
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
    policy_config.update(encoder_dim=300, device=str(device))
    train = CachedSimpleTokenDataset(args.cache, args.features_dir, "train", policy_config["chunk_size"], visual_probe=args.visual_probe)
    val = CachedSimpleTokenDataset(args.cache, args.features_dir, "val", policy_config["chunk_size"], visual_probe=args.visual_probe)
    if len(train) < args.batch_size or not len(val):
        raise ValueError("Need a full original training batch and held-out validation observations")
    if args.eval_batches < math.ceil(len(val) / args.batch_size):
        raise ValueError("--eval-batches must cover the complete original validation split")
    if jsonable(train.stats) != jsonable(val.stats):
        raise ValueError("Train and validation normalization must match")
    policy = SimpleTokenACTPolicy.from_visual_probe(policy_config, args.visual_probe,
                                                   projection=args.projection, tactile=args.tactile).to(device)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=source["learning_rate"], betas=(0.9, 0.999),
                                  weight_decay=source["weight_decay"], fused=device.type == "cuda")
    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    readout_trainable = sum(parameter.numel() for parameter in policy.readout.parameters() if parameter.requires_grad)
    # Do not inherit historical experiment names/notes describing a different
    # label interpretation. The source cache's actual semantics are recorded.
    config = {key: value for key, value in source.items() if key not in {"notes", "experiment"}}
    config.update({"experiment": "simple_token_act_2x2", "policy": policy.config, "structure": policy.structure,
              "projection": args.projection, "tactile": args.tactile, "mode": f"{args.projection}_{args.tactile}",
              "steps": args.steps, "seed": args.seed, "batch_size": args.batch_size, "amp": args.amp,
              "eval_every": args.eval_every, "eval_batches": args.eval_batches, "validation_scope": "all original held-out observations",
              "save_every": args.save_every, "num_workers": args.num_workers,
              "cache": str(Path(args.cache).resolve()), "features_dir": str(Path(args.features_dir).resolve()),
              "visual_probe": str(Path(args.visual_probe).resolve()), "torch": torch.__version__,
              "visual_probe_checkpoint_sha256": digest(args.visual_probe),
              "train_observations": len(train), "val_observations": len(val),
              "data_manifest_sha256": digest(Path(args.cache) / "manifest.json"),
              "feature_manifest_sha256": digest(Path(args.features_dir) / "manifest.json"),
              "readout_inputs_sha256": digest(Path(args.features_dir) / "readout_inputs.npy"),
              "feature_manifest": train.feature_manifest, "trainable_parameters": trainable,
              "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
              "readout_trainable_parameters": readout_trainable,
              "action_semantics": train.metadata.get("action_semantics"),
              "scope": "frozen VGGT and source state encoder; original visual projection frozen/joint with ACT; direct zero/pressure44 concatenation; unchanged source action labels and episode split",
              "argv": sys.argv, "sampler_version": 1,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    write_json(output / "train_config.json", config)
    write_json(output / "normalization.json", train.stats)
    write_json(output / "feature_manifest.json", train.feature_manifest)
    kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0}
    sampler = ResumeBatchSampler(len(train), args.batch_size, args.seed)
    loader = DataLoader(train, batch_sampler=sampler, generator=torch.Generator().manual_seed(args.seed + 100_000), **kwargs)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                           generator=torch.Generator().manual_seed(args.seed + 200_000), **kwargs)
    print(json.dumps({"event": "started", "mode": config["mode"], "trainable_parameters": trainable,
                      "readout_trainable_parameters": readout_trainable, "train_observations": len(train),
                      "val_observations": len(val), "steps": args.steps}), flush=True)
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
            gradient = torch.nn.utils.clip_grad_norm_(trainable_parameters, source["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.monotonic() - tick)
            improved, validation_performed = False, False
            if step % args.eval_every == 0 or step == args.steps:
                last_val = validate(policy, val_loader, device, args.amp, args.eval_batches)
                improved, validation_performed = last_val < best_val, True
                best_val = min(best_val, last_val)
            if step % args.save_every == 0 or step == args.steps or improved:
                checkpoint = save_checkpoint(policy, optimizer, output, step, config, train, best_val)
                if improved:
                    update_link(output, "best", checkpoint)
                print(json.dumps({"event": "checkpoint", "step": step, "path": str(checkpoint)}), flush=True)
            if step <= 3 or step % args.log_every == 0 or validation_performed or step == args.steps:
                seconds_per_step = float(np.mean(durations))
                status = {"status": "completed" if step == args.steps else "running", "step": step,
                          "total_steps": args.steps, "loss": loss.item(), **parts, "grad_norm": float(gradient),
                          "validation_performed": validation_performed,
                          "val_normalized_action_l1": last_val, "best_val_normalized_action_l1": best_val if last_val is not None else None,
                          "seconds_per_step": seconds_per_step, "eta_seconds_compute_only": (args.steps - step) * seconds_per_step,
                          "elapsed_seconds": time.monotonic() - started, "trainable_parameters": trainable,
                          "readout_trainable_parameters": readout_trainable, "mode": config["mode"],
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
    parser.add_argument("--visual-probe", required=True)
    parser.add_argument("--projection", choices=("frozen", "joint"), required=True)
    parser.add_argument("--tactile", choices=("zero", "pressure"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/act_physics_aligned_v2.json"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1000)
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
