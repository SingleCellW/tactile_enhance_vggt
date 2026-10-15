#!/usr/bin/env python3
"""Train the pinned LeRobot ACT on a reproducible frozen-VGGT cache."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timedelta
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

from tactile_vggt.data import CachedTactileDataset
from tactile_vggt.policy import PointACTPolicy


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def amp_context(amp, device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bfloat16" and device.type == "cuda" else nullcontext()


class ResumeBatchSampler:
    """Address a shuffled epoch by seed and completed optimizer steps.

    A private RNG keeps both dropout and VAE sampling independent of epoch
    boundaries. Cached observations have no stochastic worker transforms.
    """

    def __init__(self, size, batch_size, seed, completed_steps=0):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.batches_per_epoch = size // batch_size
        self.epoch, self.start_batch = divmod(completed_steps, self.batches_per_epoch)

    def __iter__(self):
        indices = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        for batch in range(self.start_batch, self.batches_per_epoch):
            yield indices[batch * self.batch_size:(batch + 1) * self.batch_size]

    def __len__(self):
        return self.batches_per_epoch - self.start_batch

    def next_epoch(self):
        self.epoch += 1
        self.start_batch = 0


@torch.inference_mode()
def validate(policy, loader, device, amp, max_batches):
    policy.eval()
    error, count = 0.0, 0
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, device)
        with amp_context(amp, device):
            pred = policy.predict_action_chunk(batch)
        if pred.shape != batch["action"].shape or not torch.isfinite(pred).all():
            raise ValueError("Validation predicted nonfinite actions or a wrong action shape")
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        error += ((pred.float() - batch["action"]).abs() * valid).sum().item()
        count += valid.sum().item() * pred.shape[-1]
    policy.train()
    if count == 0:
        raise ValueError("Validation did not contain any unpadded action targets")
    return error / count


def save_checkpoint(policy, optimizer, output, step, config, dataset, best_val):
    dest = output / "checkpoints" / f"{step:06d}"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {dest}")
    staging = dest.with_name(f".{dest.name}.partial-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(staging)
    write_json(staging / "normalization.json", dataset.stats)
    write_json(staging / "train_config.json", config)
    write_json(staging / "data_manifest.json", dataset.metadata)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "best_val": best_val,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()},
               staging / "training_state.pt")
    write_json(staging / "checkpoint_complete.json", {"step": step, "sampler_version": 1})
    staging.rename(dest)
    link = output / "latest"
    tmp = output / ".latest.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(dest.relative_to(output), target_is_directory=True)
    tmp.replace(link)
    return dest


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in
                               ("status.json", "train_config.json", "checkpoints", "metrics.jsonl")):
        raise ValueError(f"Output already has a run; use a fresh directory or --resume: {output}")
    config = json.loads(Path(args.config).read_text())
    if args.mode:
        config["policy"]["mode"] = args.mode
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.amp:
        config["amp"] = args.amp
    for key in ("batch_size", "log_every", "save_every", "eval_every", "eval_batches"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if config["num_workers"] < 0 or args.cpu_threads < 1:
        raise ValueError("num_workers must be nonnegative and cpu_threads must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if config["amp"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 is unsupported; choose --amp float32 explicitly")
    horizon = config["policy"]["chunk_size"]
    train = CachedTactileDataset(args.cache, split="train", horizon=horizon)
    val = CachedTactileDataset(args.cache, split="val", horizon=horizon)
    if jsonable(train.stats) != jsonable(val.stats):
        raise ValueError("Train and validation must share the same train-only normalization")
    if len(train) < config["batch_size"] or not len(val):
        raise ValueError("Need at least a full training batch and an episode-held-out validation set")
    scaled = config["reference_steps"] * len(train) / config["reference_frames"]
    scaled *= config["reference_batch_size"] / config["batch_size"]
    steps = args.steps or max(1000, round(scaled / 1000) * 1000)
    config.update({"steps": steps, "cache": str(Path(args.cache).resolve()),
                   "train_observations": len(train), "val_observations": len(val),
                   "budget_rule": "explicit --steps override" if args.steps is not None else
                                  "reference_steps * train_observations/reference_frames * reference_batch/batch",
                   "argv": sys.argv, "torch": torch.__version__,
                   "lerobot_source": str(ROOT / "lerobot"), "frozen_vggt": True,
                   "sampler_version": 1,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "data_manifest_sha256": hashlib.sha256((Path(args.cache) / "manifest.json").read_bytes()).hexdigest()})
    workers = config["num_workers"]
    kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda",
              "persistent_workers": workers > 0}
    if workers:
        kwargs["prefetch_factor"] = 3
    policy = PointACTPolicy(config["policy"]).to(device)
    config["policy"] = policy.config
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"],
                                  betas=(0.9, 0.999), weight_decay=config["weight_decay"],
                                  fused=device.type == "cuda")
    start_step, best_val = 0, float("inf")
    if args.resume:
        previous = json.loads((Path(args.resume) / "train_config.json").read_text())
        immutable = ("policy", "batch_size", "seed", "amp", "learning_rate", "weight_decay", "grad_clip",
                     "data_manifest_sha256", "train_observations", "val_observations", "sampler_version")
        changed = [key for key in immutable if previous.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Resume changes the training contract: {changed}")
        if not (Path(args.resume) / "checkpoint_complete.json").exists():
            raise ValueError("Resume requires a completely saved checkpoint")
        previous_stats = json.loads((Path(args.resume) / "normalization.json").read_text())
        if previous_stats != jsonable(train.stats):
            raise ValueError("Resume normalization differs from the saved checkpoint")
        policy = PointACTPolicy.from_pretrained(args.resume, map_location=device).to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"], fused=device.type == "cuda")
        state = torch.load(Path(args.resume) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_val = state["step"], state["best_val"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if steps <= start_step:
            raise ValueError(f"Requested {steps} steps, but checkpoint is already at {start_step}")
        if (output / "status.json").exists():
            recorded = json.loads((output / "status.json").read_text())
            if recorded.get("step", 0) > start_step:
                raise ValueError("Output contains progress after this checkpoint; resume into a fresh directory")
    write_json(output / "train_config.json", config)
    write_json(output / "normalization.json", train.stats)
    (output / "failure.json").unlink(missing_ok=True)
    sampler = ResumeBatchSampler(len(train), config["batch_size"], seed, start_step)
    # DataLoader worker seeding must not consume the policy's global RNG.
    loader = DataLoader(train, batch_sampler=sampler,
                        generator=torch.Generator().manual_seed(seed + 100_000), **kwargs)
    val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False,
                            generator=torch.Generator().manual_seed(seed + 200_000), **kwargs)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(json.dumps({"event": "started", "steps": steps, "train_samples": len(train),
                      "val_samples": len(val), "trainable_parameters": n_params,
                      "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}), flush=True)
    policy.train()
    iterator = iter(loader)
    durations = deque(maxlen=300)
    started = time.monotonic()
    last_val = None
    validation_seconds, checkpoint_seconds = [], []
    evaluation_steps = set(range(config["eval_every"], steps + 1, config["eval_every"])) | {steps}
    checkpoint_steps = {s for s in ({100, 1000, steps}
                                   | set(range(config["save_every"], steps + 1, config["save_every"]))) if s <= steps}
    metrics_file = (output / "metrics.jsonl").open("a", buffering=1)
    for step in range(start_step + 1, steps + 1):
        tick = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            sampler.next_epoch()
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(config["amp"], device):
            loss, parts = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}: {parts}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.monotonic() - tick
        if step > start_step + 20:
            durations.append(duration)
        if step in evaluation_steps:
            overhead_start = time.monotonic()
            last_val = validate(policy, val_loader, device, config["amp"], config["eval_batches"])
            best_val = min(best_val, last_val)
            validation_seconds.append(time.monotonic() - overhead_start)
        if step in checkpoint_steps:
            overhead_start = time.monotonic()
            ckpt = save_checkpoint(policy, optimizer, output, step, config, train, best_val)
            checkpoint_seconds.append(time.monotonic() - overhead_start)
            print(json.dumps({"event": "checkpoint", "step": step, "path": str(ckpt)}), flush=True)
        if step <= 3 or step % config["log_every"] == 0 or step == steps:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            sec_per_step = float(np.mean(durations)) if durations else duration
            remaining = (steps - step) * sec_per_step
            remaining_evals = sum(s > step for s in evaluation_steps)
            remaining_saves = sum(s > step for s in checkpoint_steps)
            overhead_ready = bool((not remaining_evals or validation_seconds)
                                  and (not remaining_saves or checkpoint_seconds))
            wall_remaining = (remaining + remaining_evals * float(np.mean(validation_seconds or [0]))
                              + remaining_saves * float(np.mean(checkpoint_seconds or [0])))
            status = {"status": "completed" if step == steps else "running", "step": step, "total_steps": steps,
                      "loss": loss.item(), "grad_norm": float(grad_norm), **parts,
                      "val_normalized_action_l1": last_val, "seconds_per_step": sec_per_step,
                      "eta_seconds_compute_only": remaining,
                      "eta_seconds_estimated": wall_remaining if overhead_ready else None,
                      "eta_asia_shanghai": (now + timedelta(seconds=wall_remaining)).isoformat() if overhead_ready else None,
                      "eta_basis": "measured training, validation and checkpoint durations" if overhead_ready else "collecting overhead measurements",
                      "updated_at": now.isoformat(), "elapsed_seconds": time.monotonic() - started,
                      "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                      "trainable_parameters": n_params, "pid": os.getpid()}
            metrics_file.write(json.dumps(status, allow_nan=False) + "\n")
            write_json(output / "status.json", status)
            print(json.dumps(status, allow_nan=False), flush=True)
    metrics_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/act_pilot.json"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["anchored", "unanchored", "no_tactile"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", choices=["bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / "failure.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
