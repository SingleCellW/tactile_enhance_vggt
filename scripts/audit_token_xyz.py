#!/usr/bin/env python3
"""Post-hoc frozen-model XYZ sensitivity; not a training-selection experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from run_token_pilot import PilotArrays, make_block, dump, sha
from tactile_vggt.token_fusion import TokenFusionProbe
from tactile_vggt.token_metrics import euclidean_errors_mm, paired_group_bootstrap, report_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=Path("vendor/vggt"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = PilotArrays(args.data, args.prefix)
    ids = np.flatnonzero(data.split == 2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TokenFusionProbe(make_block(args.repository, args.prefix, "cuda:0"), **saved["kwargs"])
    incompatible = model.load_state_dict(saved["model"], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("frozen_global_block.") for k in incompatible.missing_keys):
        raise ValueError("Checkpoint architecture mismatch")
    model.to("cuda:0").eval()
    original = data.anchors.copy()
    rng = np.random.default_rng(20261025)
    permuted = original.copy()
    for row in range(len(original)):
        permuted[row] = original[row, rng.permutation(44)]
    noise = rng.normal(size=original.shape).astype(np.float32)
    conditions = {"correct": original, "shuffle_xyz_only": permuted,
                  "independent_xyz_noise_sd5mm": original + noise*.005/data.baseline[:, None, None],
                  "independent_xyz_noise_sd15mm": original + noise*.015/data.baseline[:, None, None],
                  "common_camera_x_bias15mm": original + np.array([.015, 0, 0])/data.baseline[:, None, None]}
    predictions = {}
    # Match the original runner's @no_grad evaluation path exactly. Inference
    # tensors can select different BF16 attention kernels on this PyTorch build.
    with torch.no_grad():
        for name, anchors in conditions.items():
            data.anchors = anchors
            values = []
            for start in range(0, len(ids), 32):
                batch, _ = data.batch(ids[start:start+32], "cuda:0")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(**batch)["prediction"].float().cpu().numpy()
                if saved["whiten_target"]:
                    predicted = predicted*np.asarray(saved["target_std"])+np.asarray(saved["target_mean"])
                values.append(predicted)
            predictions[name] = np.concatenate(values)
    recorded = np.load(args.checkpoint.parent/"test_predictions.npz")
    if not np.array_equal(recorded["indices"], ids):
        raise ValueError("Recorded test rows changed")
    parity = float(np.max(np.abs(recorded["prediction"]-predictions["correct"])))
    if parity > 1e-6:
        raise ValueError(f"Frozen correct-input prediction differs from recorded result: {parity}")
    errors = {k: euclidean_errors_mm(v, data.target[ids], data.baseline[ids]) for k, v in predictions.items()}
    result = {name: {"metrics": report_metrics(pred, data.target[ids], data.episode[ids], data.baseline[ids]),
                     "minus_correct": paired_group_bootstrap(errors[name], errors["correct"], data.episode[ids])}
              for name, pred in predictions.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, {"complete": True, "checkpoint_sha256": sha(args.checkpoint), "seed": 20261025,
                       "prediction_cache_max_abs": parity, "conditions": result,
                       "input_intervention": "XYZ only; same RGB prefix, pressure, flags, sensor IDs and state",
                       "scope": "Post-hoc sensitivity audit of seed43; no training or checkpoint selection. Shuffling is out of distribution, not calibration accuracy."})


if __name__ == "__main__":
    main()
