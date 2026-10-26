#!/usr/bin/env python3
"""Benchmark saved policy latency using held-out observations, not task success.

The default cached-geometry benchmark excludes VGGT and camera acquisition.
Optional --geometry-input is an NPZ containing rgb uint8[2,H,W,3], state[22],
tactile[44], anchors_world[44,3], known_cameras_cv[2,4,4], and optional
base_to_world[4,4]. RGB must already have the training square crop. This optional
path separately times frozen VGGT plus calibrated point construction, and a full
geometry/policy refresh. It still excludes camera capture and simulator control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import dependency_path
sys.path[:0] = [str(dependency_path("lerobot") / "src")]

import numpy as np
import torch

from tactile_vggt.data import CachedTactileDataset
from tactile_vggt.inference import PointACTInference


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, device, iterations):
    durations = []
    for index in range(iterations):
        synchronize(device)
        started = time.perf_counter()
        call(index)
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    values = np.asarray(durations)
    return {
        "iterations": iterations,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def raw_args(sample):
    return sample["observation.state"], sample["observation.tactile"], sample["observation.points"]


def geometry_benchmark(args, adapter, metadata):
    from prepare_cache import FrozenVGGT
    from tactile_vggt.geometry import build_point_tensor, calibrated_similarity, unproject_depth

    with np.load(args.geometry_input, allow_pickle=False) as archive:
        example = {key: archive[key].copy() for key in archive.files}
    required = {"rgb", "state", "tactile", "anchors_world", "known_cameras_cv"}
    if not required <= example.keys():
        raise ValueError(f"geometry-input is missing {sorted(required - example.keys())}")
    rgb = example["rgb"]
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3:
        raise ValueError("geometry-input rgb must be uint8 [2,H,W,3]")
    config = metadata["configuration"]
    model = FrozenVGGT(args.vggt_repo, args.vggt_weights, str(adapter.device), config["image_size"])

    def builder(rgb, state, tactile, context):
        colors, depth, confidence, extrinsics, intrinsics = model.predict([rgb])
        scale, rotation, translation, _ = calibrated_similarity(extrinsics[0], context["known_cameras_cv"])
        predicted = unproject_depth(depth[0], extrinsics[0], intrinsics[0])
        world = scale * predicted @ rotation.T + translation
        return build_point_tensor(
            world, colors[0], confidence[0], context["anchors_world"], tactile,
            base_to_world=context.get("base_to_world", np.asarray(config["base_to_world"])),
            visual_count=adapter.policy.config["n_visual_points"],
            confidence_quantile=config["confidence_quantile"], seed=config["seed"],
        )

    adapter.point_builder = builder
    for _ in range(args.geometry_warmup):
        adapter.predict_action_chunk(example["state"], example["tactile"], raw_rgb=rgb, geometry_context=example)
    geometry = measure(
        lambda _: builder(rgb, example["state"], example["tactile"], example),
        adapter.device, args.geometry_iterations,
    )
    combined = measure(
        lambda _: adapter.predict_action_chunk(example["state"], example["tactile"], raw_rgb=rgb, geometry_context=example),
        adapter.device, args.geometry_iterations,
    )
    return {
        "geometry_refresh": geometry,
        "geometry_and_policy_refresh": combined,
        "geometry_input": str(Path(args.geometry_input).resolve()),
        "geometry_example_split": "not asserted; latency only, not an accuracy evaluation",
        "uses_known_camera_calibration": True,
        "geometry_includes": "RGB resize/transfer, VGGT, CPU readback, calibrated alignment, point sampling",
        "excludes": "weights loading, image acquisition/decoding, robot/simulator control",
    }


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.cpu_threads)
    adapter = PointACTInference(args.checkpoint, device=args.device, amp=args.amp)
    dataset = CachedTactileDataset(args.cache, split="val", horizon=adapter.policy.config["chunk_size"], normalize=False)
    if dataset.stats != adapter.stats:
        raise ValueError("Checkpoint normalization does not match this cache; refuse an inconsistent benchmark")
    sample_indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    samples = [dataset[int(index)] for index in sample_indices]
    normalized = [adapter.prepare_observation(*raw_args(sample)) for sample in samples]

    def policy_only(index):
        with adapter.autocast_context():
            return adapter.policy.predict_action_chunk(normalized[index % len(samples)])

    def cached_refresh(index):
        return adapter.predict_action_chunk(*raw_args(samples[index % len(samples)]))

    for index in range(args.warmup):
        cached_refresh(index)
    result = {
        "benchmark_type": "offline_latency",
        "closed_loop_evaluated": False,
        "checkpoint": str(adapter.checkpoint),
        "cache": str(Path(args.cache).resolve()),
        "split": "val",
        "normalization_verified": True,
        "device": str(adapter.device),
        "device_name": torch.cuda.get_device_name(adapter.device) if adapter.device.type == "cuda" else "CPU",
        "amp": args.amp,
        "warmup_iterations": args.warmup,
        "cpu_threads": args.cpu_threads,
        "policy_config": adapter.policy.config,
        "heldout_samples": len(samples),
        "fresh_chunk_policy_only": measure(policy_only, adapter.device, args.iterations),
        "fresh_chunk_cached_geometry": measure(cached_refresh, adapter.device, args.iterations),
        "cached_geometry_excludes": "VGGT, camera input, disk I/O, robot/simulator control",
        "policy_only_excludes": "observation normalization/transfer and action denormalization",
    }

    def control_step(index):
        if adapter.needs_observation:
            return adapter.select_action(*raw_args(samples[index % len(samples)]))
        return adapter.select_action()

    adapter.reset()
    result["control_ticks_cached_geometry"] = measure(control_step, adapter.device, args.iterations)
    if adapter.policy.config["n_action_steps"] > 1 and adapter.policy.config["temporal_ensemble_coeff"] is None:
        durations = []
        for index in range(args.iterations):
            if adapter.needs_observation:
                adapter.select_action(*raw_args(samples[index % len(samples)]))
            durations.append(measure(lambda _: adapter.select_action(), adapter.device, 1)["mean_ms"])
        values = np.asarray(durations)
        result["queued_action_only"] = {
            "iterations": len(values), "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()), "max_ms": float(values.max()),
        }
    if args.geometry_input:
        result["optional_geometry"] = geometry_benchmark(args, adapter, dataset.metadata)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--geometry-input")
    parser.add_argument("--vggt-repo")
    parser.add_argument("--vggt-weights")
    parser.add_argument("--geometry-warmup", type=int, default=2)
    parser.add_argument("--geometry-iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.iterations, args.samples, args.cpu_threads, args.geometry_iterations) < 1 or min(args.warmup, args.geometry_warmup) < 0:
        parser.error("iteration/sample/thread counts must be positive; warmup counts must be nonnegative")
    geometry_arguments = (args.geometry_input, args.vggt_repo, args.vggt_weights)
    if any(geometry_arguments) and not all(geometry_arguments):
        parser.error("--geometry-input requires --vggt-repo and --vggt-weights together")
    run(args)


if __name__ == "__main__":
    main()
