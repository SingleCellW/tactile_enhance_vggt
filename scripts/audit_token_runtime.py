#!/usr/bin/env python3
"""Compare actual online VGGT features to the frozen training-prefix cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from run_token_pilot import PilotArrays, dump, sha
from token_runtime import LiveTokenEncoder, ROOT

from tactile_vggt.paths import artifact_path, dependency_path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    p.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(4)
    data = PilotArrays(args.data, args.prefix)
    # One nominal and one perturbed contact from each held-out episode.
    ids = []
    variant = np.load(args.data/"variant.npy")
    for ep in np.unique(data.episode[data.split == 2]):
        for perturbed in (False, True):
            candidates = np.flatnonzero((data.episode == ep) & (data.flags.any(1)) & ((variant > 0) == perturbed))
            if len(candidates):
                ids.append(int(candidates[0]))
    raw = {name: np.load(args.data/(name+".npy"), mmap_mode="r") for name in
           ("rgb", "state", "pressure", "anchors", "known_cameras")}
    checkpoints = {mode: args.run/mode/"best.pt" for mode in ("visual", "late", "internal")}
    started = time.monotonic()
    encoder = LiveTokenEncoder(args.repository, args.weights, checkpoints, args.device)
    expected, actual = {k: [] for k in checkpoints}, {k: [] for k in checkpoints}
    with torch.inference_mode():
        for start in range(0, len(ids), 4):
            chosen = np.asarray(ids[start:start+4])
            online = encoder.encode(*(raw[name][chosen] for name in raw))
            batch, _ = data.batch(chosen, args.device)
            for mode, probe in encoder.probes.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cached = probe(**batch)["features"].float().cpu().numpy()
                expected[mode].append(cached)
                actual[mode].append(online[mode])
    results = {}
    for mode in checkpoints:
        a, b = np.concatenate(actual[mode]), np.concatenate(expected[mode])
        diff = np.abs(a-b)
        relative_l2 = np.linalg.norm(a-b, axis=1)/np.maximum(np.linalg.norm(b, axis=1), 1e-8)
        results[mode] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                         "max_relative_l2": float(relative_l2.max()),
                         "passed": bool(np.isfinite(a).all() and relative_l2.max() < .01),
                         "checkpoint_sha256": sha(checkpoints[mode])}
    dump(args.output, {"passed": all(r["passed"] for r in results.values()), "results": results,
                       "sample_ids": ids, "seconds": time.monotonic()-started,
                       "criterion": "finite features and relative L2 < 1%; BF16 batch kernels can differ",
                       "scope": "Actual official VGGT online RGB versus frozen fp16 prefix; no target input"})
    if not all(r["passed"] for r in results.values()):
        raise SystemExit("Online/cache feature parity failed")


if __name__ == "__main__":
    main()
