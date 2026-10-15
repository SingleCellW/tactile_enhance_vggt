#!/usr/bin/env python3
"""Evaluate every saved checkpoint on the same complete held-out observation set.

Predictions use observations only and the inference-time ACT latent (zero), never
the training posterior or action labels. Metrics are imitation errors, not task
success. Selecting best_saved on this validation set does not create an unbiased
test score; the final checkpoint is retained as the equal-training-budget control.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lerobot" / "src")]

import numpy as np
import torch
from torch.utils.data import DataLoader

from tactile_vggt.data import CachedTactileDataset
from tactile_vggt.policy import PointACTPolicy

OBSERVATION_KEYS = ("observation.state", "observation.points", "observation.tactile")


class ErrorAccumulator:
    def __init__(self, action_std):
        self.action_std = np.asarray(action_std, dtype=np.float64)
        self.normalized_sum = np.zeros_like(self.action_std)
        self.physical_sum = np.zeros_like(self.action_std)
        self.observations = 0
        self.action_steps = 0

    def add(self, absolute_error, valid, selected=None):
        error = np.asarray(absolute_error, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        if error.ndim != 3 or valid.shape != error.shape[:2] or error.shape[-1] != len(self.action_std):
            raise ValueError("Expected errors [B,H,A] and valid mask [B,H]")
        selected = np.ones(len(error), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
        if selected.shape != (len(error),):
            raise ValueError("Observation selection mask must have shape [B]")
        mask = valid & selected[:, None]
        self.observations += int(selected.sum())
        self.action_steps += int(mask.sum())
        weighted = np.where(mask[..., None], error, 0)
        self.normalized_sum += weighted.sum(axis=(0, 1))
        self.physical_sum += (weighted * self.action_std).sum(axis=(0, 1))

    def result(self):
        result = {"observations": self.observations, "unpadded_action_steps": self.action_steps,
                  "joint_scalar_count": self.action_steps * len(self.action_std)}
        if not self.action_steps:
            return {**result, "normalized_l1": None, "physical_joint_l1": None,
                    "per_joint_normalized_l1": None, "per_joint_physical_l1": None}
        normalized = self.normalized_sum / self.action_steps
        physical = self.physical_sum / self.action_steps
        return {**result, "normalized_l1": float(normalized.mean()), "physical_joint_l1": float(physical.mean()),
                "per_joint_normalized_l1": normalized.tolist(), "per_joint_physical_l1": physical.tolist()}


@torch.inference_mode()
def predict_observations_only(policy, batch, device, amp):
    observations = {key: batch[key].to(device, non_blocking=True) for key in OBSERVATION_KEYS}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bfloat16" and device.type == "cuda" else nullcontext()
    with context:
        predictions = policy.predict_action_chunk(observations)
    return predictions.float().cpu()


def checkpoint_candidates(run):
    run = Path(run).resolve(strict=True)
    candidates, skipped = [], []
    for path in sorted((run / "checkpoints").iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        needed = ("checkpoint_complete.json", "config.json", "model.pt", "normalization.json", "train_config.json")
        if not all((path / name).is_file() for name in needed):
            skipped.append(str(path))
            continue
        marker = json.loads((path / "checkpoint_complete.json").read_text())
        if marker.get("step") != int(path.name):
            raise ValueError(f"Checkpoint step marker disagrees with directory: {path}")
        candidates.append(path)
    if not candidates:
        raise ValueError(f"No complete saved checkpoints found under {run}")
    return sorted(candidates, key=lambda path: int(path.name)), skipped


def validate_checkpoint(checkpoint, cache_manifest_hash, stats):
    training = json.loads((checkpoint / "train_config.json").read_text())
    if training.get("data_manifest_sha256") != cache_manifest_hash:
        raise ValueError(f"Checkpoint was trained on a different cache manifest: {checkpoint}")
    if json.loads((checkpoint / "normalization.json").read_text()) != stats:
        raise ValueError(f"Checkpoint normalization differs from the evaluation cache: {checkpoint}")
    return training


def select_checkpoints(reports, expected_final_step):
    best = min(reports, key=lambda report: (report["full_chunk"]["all"]["normalized_l1"], report["step"]))
    final = next((report for report in reports if report["step"] == expected_final_step), None)
    def compact(report):
        if report is None:
            return None
        return {"checkpoint": report["checkpoint"], "step": report["step"],
                "normalized_l1": report["full_chunk"]["all"]["normalized_l1"],
                "physical_joint_l1": report["full_chunk"]["all"]["physical_joint_l1"],
                "active_contact_normalized_l1": report["full_chunk"]["active_contact"]["normalized_l1"],
                "first_action_normalized_l1": report["first_action"]["all"]["normalized_l1"]}
    return {"selection_metric": "full_chunk.all.normalized_l1, minimum over complete saved checkpoints",
            "best_saved": compact(best), "final_equal_budget": compact(final),
            "latest_saved": compact(max(reports, key=lambda report: report["step"]))}


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, dataset, device, amp, batch_size, active_threshold, workers):
    started = time.monotonic()
    policy = PointACTPolicy.from_pretrained(checkpoint, map_location=device)
    horizon = policy.config["chunk_size"]
    if horizon != dataset.horizon:
        raise ValueError("All checkpoints must use the same action horizon for this comparison")
    execute_steps = policy.config["n_action_steps"]
    action_std = np.asarray(dataset.stats["action_std"], dtype=np.float64)
    if len(action_std) != policy.config["action_dim"]:
        raise ValueError("Action statistics dimension mismatch")
    def groups():
        return {name: ErrorAccumulator(action_std) for name in ("all", "active_contact", "inactive_contact")}
    accumulators = {"full_chunk": groups(), "executed_prefix": groups(), "first_action": groups()}
    per_episode = {int(episode): groups() for episode in dataset.metadata["split"]["val"]}
    observation_rows = np.asarray(dataset.obs_frame_indices[dataset.observation_indices])
    raw_active = (np.asarray(dataset.tactile[observation_rows]) > active_threshold).any(axis=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    offset = 0
    for batch in loader:
        prediction = predict_observations_only(policy, batch, device, amp)
        target = batch["action"]
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError(f"Nonfinite or malformed predictions from {checkpoint}")
        error = (prediction - target).abs().numpy()
        valid = (~batch["action_is_pad"]).numpy()
        active = raw_active[offset:offset + len(error)]
        episodes = batch["episode_index"].numpy()
        selections = {"all": np.ones(len(error), dtype=bool), "active_contact": active, "inactive_contact": ~active}
        for name, length in (("full_chunk", horizon), ("executed_prefix", execute_steps), ("first_action", 1)):
            for subset, selected in selections.items():
                accumulators[name][subset].add(error[:, :length], valid[:, :length], selected)
        for episode in np.unique(episodes):
            episode_mask = episodes == episode
            for subset, selected in selections.items():
                per_episode[int(episode)][subset].add(error, valid, episode_mask & selected)
        offset += len(error)
    if offset != len(dataset):
        raise AssertionError("Evaluation did not cover the full held-out observation set")
    report = {"checkpoint": str(checkpoint), "step": int(checkpoint.name), "mode": policy.config["mode"],
              "chunk_size": horizon, "executed_prefix_length": execute_steps,
              "evaluated_observations": offset, "elapsed_seconds_including_load": time.monotonic() - started,
              **{name: {subset: metric.result() for subset, metric in group.items()} for name, group in accumulators.items()},
              "per_episode_full_chunk": {str(episode): {subset: metric.result() for subset, metric in group.items()}
                                         for episode, group in per_episode.items()}}
    report["episode_macro_normalized_l1"] = float(np.mean([
        group["all"]["normalized_l1"] for group in report["per_episode_full_chunk"].values()
        if group["all"]["normalized_l1"] is not None]))
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if args.amp == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 evaluation requires a supporting CUDA device")
    cache = Path(args.cache).resolve(strict=True)
    manifest_hash = hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
    candidates = [(Path(run_dir).resolve(strict=True), *checkpoint_candidates(run_dir)) for run_dir in args.runs]
    first_config = json.loads((candidates[0][1][0] / "config.json").read_text())["config"]
    dataset = CachedTactileDataset(cache, split="val", horizon=first_config["chunk_size"])
    if args.expected_observations is not None and len(dataset) != args.expected_observations:
        raise ValueError(f"Expected {args.expected_observations} held-out observations, found {len(dataset)}")
    results = {
        "evaluation_type": "offline imitation on the entire validation split", "closed_loop_evaluated": False,
        "created_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "cache": str(cache), "cache_manifest_sha256": manifest_hash,
        "observations": len(dataset), "episode_ids": dataset.metadata["split"]["val"],
        "device": str(device), "amp": args.amp, "batch_size": args.batch_size,
        "inference_inputs": list(OBSERVATION_KEYS), "action_labels_given_to_policy": False,
        "active_contact_mask": {"definition": "any raw tactile sensor > threshold at the current observation state row; select the whole observation's unpadded target chunk",
                                "raw_pressure_threshold": args.active_pressure_threshold,
                                "units": "native tactile proxy units, not calibrated Newtons", "uses_future_contact": False},
        "metric_definition": "sum absolute joint error / number of unpadded action-step joint scalars; per-episode metrics use the same weighting",
        "physical_joint_units": "original action target units, in original 22-joint order",
        "selection_caveat": "best_saved is selected on these validation errors; it is not an independent test result. Final checkpoints remain the equal-step comparison.",
        "data_caveats": list(args.data_caveat or []),
        "runs": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for run_dir, checkpoints, skipped in candidates:
        reports, expected_final_step = [], None
        for checkpoint in checkpoints:
            training = validate_checkpoint(checkpoint, manifest_hash, dataset.stats)
            expected_final_step = training["steps"]
            report = evaluate_checkpoint(checkpoint, dataset, device, args.amp, args.batch_size,
                                         args.active_pressure_threshold, args.num_workers)
            reports.append(report)
            print(json.dumps({"run": run_dir.name, "step": report["step"],
                              "normalized_l1": report["full_chunk"]["all"]["normalized_l1"],
                              "physical_joint_l1": report["full_chunk"]["all"]["physical_joint_l1"],
                              "active_normalized_l1": report["full_chunk"]["active_contact"]["normalized_l1"]}), flush=True)
        results["runs"].append({"run": str(run_dir), "skipped_incomplete_checkpoints": skipped,
                                "checkpoints": reports, **select_checkpoints(reports, expected_final_step)})
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
    print(json.dumps({"output": str(output.resolve()), "selection": [{"run": run["run"], "best_saved": run["best_saved"],
                       "final_equal_budget": run["final_equal_budget"]} for run in results["runs"]]}, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--active-pressure-threshold", type=float, default=0.5)
    parser.add_argument("--expected-observations", type=int)
    parser.add_argument("--data-caveat", action="append", help="Repeatable known data limitation recorded with the numerical report")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.num_workers < 0 or not np.isfinite(args.active_pressure_threshold) or args.active_pressure_threshold < 0:
        parser.error("Invalid batch/thread counts or raw-pressure threshold")
    run(args)


if __name__ == "__main__":
    main()
