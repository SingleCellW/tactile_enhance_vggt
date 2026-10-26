#!/usr/bin/env python3
"""Stage1.b camera-only augmentation around the frozen real VGGT global23.

Separate outputs preserve Stage1.a. --dev-only permits a development-only
budget decision; a later --resume scores the saved best without retraining.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile_vggt.paths import artifact_path, dependency_path
from tactile_vggt.provenance import write_run_manifest, source_hashes
sys.path[:0] = [str(ROOT / "scripts")]
from tactile_vggt.camera_pilot import CameraArrays, evaluate_predictions, episode_summary
from tactile_vggt.token_fusion import TokenFusionProbe
from tactile_vggt.token_metrics import paired_group_bootstrap
from run_token_pilot import cache_prefix, validate_prefix, make_block, sha, seed_all, dump

MODES = ("visual", "tactile", "late", "internal")
REGIMES = ("fixed", "augmented")


def verify_data_files(data):
    """Bind actual arrays, including RGB, to the signed-by-hash manifest text."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if not manifest.get("complete") or not manifest.get("files"):
        raise ValueError("Complete camera data and per-array hashes required")
    required = set(CameraArrays.NAMES) | {"rgb"}
    if not required <= set(manifest["files"]):
        raise ValueError("Camera manifest omits required input/scoring arrays")
    for name, metadata in manifest["files"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("Unsafe camera array name")
        path = data / (name + ".npy")
        if path.is_symlink() or not path.is_file() or sha(path) != metadata.get("sha256"):
            raise ValueError("Camera data array changed: " + name)
    return manifest


def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()


def extract(args):
    # The earlier extractor hooks the ACTUAL final pretrained global block and
    # exits before it runs. Only a new Stage1.b prefix path may be supplied.
    if args.prefix.resolve() == (artifact_path("artifacts") / "token_pilot_prefix_20261025").resolve():
        raise ValueError("Refusing to alter the Stage1.a prefix")
    verify_data_files(args.data)
    cache_prefix(args)
    path = args.prefix / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata.update(camera_pilot_format_version=1,
                    tokens_sha256=sha(args.prefix / "tokens.npy"),
                    positions_sha256=sha(args.prefix / "positions.npy"))
    dump(path, metadata)


def validate(args):
    verify_data_files(args.data)
    metadata = validate_prefix(args)
    for key, name in [("tokens_sha256", "tokens.npy"), ("positions_sha256", "positions.npy")]:
        if metadata.get(key) != sha(args.prefix / name):
            raise ValueError("Missing/changed camera-prefix array hash: " + name)
    if metadata.get("layer") != 23 or metadata.get("patch_start") != 5:
        raise ValueError("Expected official VGGT global23 / five special tokens")
    return metadata


def snapshot_sources(output):
    """Compatibility name: retain source hashes, without creating copies."""
    paths = [Path(__file__), ROOT / "scripts/run_token_pilot.py",
             ROOT / "src/tactile_vggt/camera_pilot.py", ROOT / "src/tactile_vggt/token_fusion.py",
             ROOT / "src/tactile_vggt/token_metrics.py", ROOT / "src/tactile_vggt/experiment_utils.py"]
    return source_hashes(paths)


class Batches:
    def __init__(self, data, prefix, device, preload_gb=8):
        self.data, self.device = data, device
        self.tokens = np.load(prefix / "tokens.npy", mmap_mode="r")
        self.positions = np.load(prefix / "positions.npy")
        if len(self.tokens) != len(data.episode) or self.tokens.shape[1:3] != (2, 261):
            raise ValueError("Camera rows and prefix do not match stereo RGB224")
        self.device_tokens = None
        if device.startswith("cuda") and self.tokens.nbytes <= preload_gb * 1e9:
            free, _ = torch.cuda.mem_get_info(device)
            if self.tokens.nbytes < free * .5:
                self.device_tokens = torch.as_tensor(np.array(self.tokens), device=device)

    def __call__(self, ids):
        d, device = self.data, self.device
        ids = np.asarray(ids, dtype=int)
        tensor = lambda x: torch.as_tensor(np.array(x), device=device, dtype=torch.float32)
        tokens = (tensor(self.tokens[ids]) if self.device_tokens is None else
                  self.device_tokens.index_select(0, torch.as_tensor(ids, device=device)).float())
        inputs = {"prefix_tokens": tokens,
                  "pos": torch.as_tensor(self.positions, device=device).unsqueeze(0).expand(len(ids), -1, -1, -1),
                  "pressure": tensor(d.pressure_input[ids]), "pressure_flags": tensor(d.flags[ids]),
                  "anchors": tensor(d.anchor_input[ids]), "state": tensor(d.state_input[ids])}
        target = tensor((d.target[ids] - d.target_mean) / d.target_std)
        return inputs, target


@torch.no_grad()
def predict(model, batches, ids, args):
    model.eval()
    values = []
    for start in range(0, len(ids), args.batch_size):
        inputs, _ = batches(ids[start:start + args.batch_size])
        with autocast(args):
            prediction = model(**inputs)["prediction"]
        values.append(prediction.float().cpu().numpy())
    result = np.concatenate(values) * batches.data.target_std + batches.data.target_mean
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite spatial predictions")
    return result


def small_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("frozen_global_block.")}


def restore(model, checkpoint):
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.unexpected_keys or any(not key.startswith("frozen_global_block.") for key in result.missing_keys):
        raise ValueError("Adapter checkpoint layout differs")


def run_signature(args, data, seed, regime, mode, provenance):
    return {"format_version": 1, "seed": seed, "regime": regime, "mode": mode,
            "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "eval_every": args.eval_every, "weight_decay": 1e-4, "clip_norm": 1.,
            "hidden_dim": args.hidden_dim, "feature_dim": 256, "initial_gate": .01,
            "displacement_weight": args.displacement_weight, "target_whitening": True,
            "normalization": "shared nominal camera TRAIN only",
            "sampling": "same physical-row RNG across regimes; separate uniform 0..3 camera RNG for augmented",
            "dev_selection": "minimum episode-equal base-frame centre error over common DEV nominal+3near; earliest tie",
            "test_selection": "never used for checkpoint or development budget decision",
            "reference_baseline_m": data.baseline, "amp": "bfloat16" if args.device.startswith("cuda") else "float32",
            **provenance}


def train_one(args, data, batches, seed, regime, mode, provenance, block_factory=None):
    run = args.output / f"s{seed}" / regime / mode
    signature = run_signature(args, data, seed, regime, mode, provenance)
    config_path, marker_path = run / "config.json", run / "completion.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != signature:
            raise ValueError(f"Existing run needs --resume and the exact saved configuration: {run}")
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if not marker.get("training_complete"):
            raise ValueError(f"Partial training is not resumed; preserve it and use a new output: {run}")
        if marker.get("checkpoint_sha256") != sha(run / "best.pt") or marker.get("config_sha256") != sha(config_path):
            raise ValueError("Completed checkpoint/config changed")
        if marker.get("complete"):
            for key, name in [("metrics_sha256", "metrics.json"), ("predictions_sha256", "test_predictions.npz")]:
                if marker.get(key) != sha(run / name):
                    raise ValueError("Completed evaluation changed")
            return json.loads((run / "metrics.json").read_text())
        if args.dev_only:
            return marker
        trained = True
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Refusing an untracked existing output directory")
        run.mkdir(parents=True, exist_ok=True)
        dump(config_path, signature)
        dump(marker_path, {"complete": False, "training_complete": False})
        trained = False
    seed_all(seed)
    block = block_factory() if block_factory else make_block(args.repository, args.prefix, args.device)
    kwargs = {"mode": mode, "dim": int(batches.tokens.shape[-1]), "hidden_dim": args.hidden_dim,
              "feature_dim": 256, "initial_gate": .01}
    model = TokenFusionProbe(block, **kwargs).to(args.device)
    started = time.monotonic()
    if not trained:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
        rng, view_rng = np.random.default_rng(seed), np.random.default_rng(seed + 100_000)
        pair_rng, pair_view_rng = np.random.default_rng(seed + 200_000), np.random.default_rng(seed + 300_000)
        pair_groups = {}
        for group, first, second in data.training_pairs:
            pair_groups.setdefault(group, []).append((first, second))
        if args.displacement_weight and not pair_groups:
            raise ValueError("Displacement objective requires train object siblings")
        history, best, best_step = [], float("inf"), 0
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        for step in range(1, args.steps + 1):
            model.train()
            ids = data.sample(rng, view_rng, args.batch_size, regime)
            inputs, target = batches(ids)
            with autocast(args):
                result = model(**inputs)["prediction"].float()
                center_loss = torch.nn.functional.smooth_l1_loss(result, target, beta=1.)
                displacement_loss = center_loss.new_zeros(())
                if args.displacement_weight:
                    first_ids, second_ids = [], []
                    for _ in range(args.batch_size):
                        group = pair_rng.choice(list(pair_groups))
                        first, second = pair_groups[group][pair_rng.integers(len(pair_groups[group]))]
                        view = int(pair_view_rng.integers(4)) if regime == "augmented" else 0
                        first_ids.append(data.lookup[(int(data.physical_sample[first]), view)])
                        second_ids.append(data.lookup[(int(data.physical_sample[second]), view)])
                    pair_inputs, pair_target = batches(first_ids + second_ids)
                    pair_prediction = model(**pair_inputs)["prediction"].float()
                    n = len(first_ids)
                    displacement_loss = torch.nn.functional.smooth_l1_loss(pair_prediction[n:] - pair_prediction[:n], pair_target[n:] - pair_target[:n], beta=1.)
                loss = center_loss + args.displacement_weight * displacement_loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss {seed}/{regime}/{mode}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                pred = predict(model, batches, data.dev_ids, args)
                base = data.base_predictions(pred, data.dev_ids)
                errors = np.linalg.norm(base - data.labels[data.dev_ids], axis=1) * 1000
                dev_metric = episode_summary(errors, data.episode[data.dev_ids])["episode_mean_mm"]
                dev_domains = {name: episode_summary(errors[data.camera_domain[data.dev_ids] == domain], data.episode[data.dev_ids][data.camera_domain[data.dev_ids] == domain])["episode_mean_mm"]
                               for domain, name in [(0, "nominal"), (1, "near")]}
                record = {"step": step, "loss": float(loss.detach()), "center_loss": float(center_loss.detach()),
                          "displacement_loss": float(displacement_loss.detach()), "grad_norm": float(gradient),
                          "dev_episode_mean_mm": dev_metric, "dev_domains": dev_domains,
                          "seconds": time.monotonic() - started}
                history.append(record)
                print(json.dumps({"seed": seed, "regime": regime, "mode": mode, **record}), flush=True)
                if dev_metric < best:
                    best, best_step = dev_metric, step
                    torch.save({"model": small_state(model), "kwargs": kwargs, "step": step,
                                "dev_episode_mean_mm": dev_metric, "normalization": data.normalization,
                                "config": signature}, run / "best.pt")
                dump(run / "history.json", history)
        del optimizer
        marker = {"complete": False, "training_complete": True, "test_complete": False,
                  "best_step": best_step, "dev_episode_mean_mm": best, "train_steps": args.steps,
                  "training_seconds": time.monotonic() - started,
                  "allocated_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "peak_gpu_gb": torch.cuda.max_memory_allocated(args.device) / 1e9 if args.device.startswith("cuda") else 0,
                  "checkpoint_sha256": sha(run / "best.pt"), "config_sha256": sha(config_path)}
        dump(marker_path, marker)
    if args.dev_only:
        del model, block
        return marker
    checkpoint = torch.load(run / "best.pt", map_location=args.device, weights_only=False)
    restore(model, checkpoint)
    pred = predict(model, batches, data.test_ids, args)
    metrics, predicted_base, errors = evaluate_predictions(data, data.test_ids, pred)
    ids = data.test_ids
    np.savez_compressed(run / "test_predictions.npz", indices=ids, prediction_camera=pred,
        target_camera=data.target[ids], prediction_base=predicted_base, target_base=data.labels[ids],
        episode=data.episode[ids], group=data.group[ids], physical_sample=data.physical_sample[ids],
        variant=data.variant[ids], camera_variant=data.camera_variant[ids], camera_domain=data.camera_domain[ids],
        camera_to_base=data.camera[ids], errors_mm=errors, baseline_m=data.baseline)
    metrics.update(complete=True, seed=seed, regime=regime, mode=mode,
                   best_step=checkpoint["step"], dev_episode_mean_mm=checkpoint["dev_episode_mean_mm"],
                   train_steps=args.steps, config_sha256=sha(config_path), checkpoint_sha256=sha(run / "best.pt"))
    dump(run / "metrics.json", metrics)
    marker.update(complete=True, test_complete=True, metrics_sha256=sha(run / "metrics.json"),
                  predictions_sha256=sha(run / "test_predictions.npz"))
    dump(marker_path, marker)
    del model, block
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def convergence(output, steps):
    rows = []
    for regime in REGIMES:
        for mode in MODES:
            path = output / "s43" / regime / mode
            if not (path / "history.json").is_file() or not (path / "completion.json").is_file():
                continue
            marker, history = json.loads((path / "completion.json").read_text()), json.loads((path / "history.json").read_text())
            if not marker.get("training_complete") or marker.get("train_steps") != steps:
                continue
            recent = history[-5:]
            gain = (recent[0]["dev_episode_mean_mm"] - recent[-1]["dev_episode_mean_mm"]) / recent[0]["dev_episode_mean_mm"] if len(recent) == 5 and recent[0]["dev_episode_mean_mm"] > 0 else 0.
            late = marker["best_step"] >= steps - 400
            rows.append({"regime": regime, "mode": mode, "best_step": marker["best_step"], "late_best": late,
                         "last_five_relative_improvement": gain, "flag": bool(late and gain > .05)})
    complete = len(rows) == 8
    result = {"complete": complete, "steps": steps, "runs": rows,
              "flagged_runs": sum(r["flag"] for r in rows),
              "extend_all_to_4000": bool(complete and sum(r["flag"] for r in rows) >= 4 and steps == 2000),
              "rule": "each run: best within final400 AND first-to-last of final5 dev records >5% improvement; at least4 of8 seed43 runs; test not read"}
    dump(output / "convergence.json", result)
    return result


def train(args):
    validate(args)
    data = CameraArrays(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args.output, args)
    norm_path = args.output / "normalization.json"
    if norm_path.exists() and json.loads(norm_path.read_text()) != data.normalization:
        raise ValueError("Output normalization differs from nominal-training statistics")
    dump(norm_path, data.normalization)
    environment = {"python": sys.version, "python_executable": sys.executable,
                   "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
                   "cuda_version": torch.version.cuda, "device": args.device,
                   "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    env_path = args.output / "environment.json"
    if not env_path.exists():
        dump(env_path, environment)
    elif json.loads(env_path.read_text()) != environment:
        raise ValueError("Resume environment differs; use a new output and report this change")
    provenance = {"data_manifest_sha256": sha(args.data / "manifest.json"),
                  "prefix_manifest_sha256": sha(args.prefix / "manifest.json"),
                  "normalization_sha256": sha(norm_path), "environment_sha256": sha(env_path),
                  "runner_sha256": sha(__file__), "helper_sha256": sha(ROOT / "src/tactile_vggt/camera_pilot.py"),
                  "token_fusion_sha256": sha(ROOT / "src/tactile_vggt/token_fusion.py"),
                  "legacy_runner_dependency_sha256": sha(ROOT / "scripts/run_token_pilot.py"),
                  "source_hashes": snapshot_sources(args.output)}
    batches = Batches(data, args.prefix, args.device, args.preload_gb)
    runs = []
    for seed in args.seeds:
        for regime in args.regimes:
            for mode in args.modes:
                if args.command == "evaluate" and not (args.output / f"s{seed}" / regime / mode / "completion.json").is_file():
                    raise ValueError("evaluate requires previously completed training")
                args.resume = args.resume or args.command == "evaluate"
                result = train_one(args, data, batches, seed, regime, mode, provenance)
                runs.append({"seed": seed, "regime": regime, "mode": mode,
                             "training_complete": bool(result.get("training_complete", result.get("complete"))),
                             "complete": bool(result.get("complete")),
                             "metrics": f"s{seed}/{regime}/{mode}/metrics.json" if result.get("complete") else None})
                convergence(args.output, args.steps)
    dump(args.output / "summary.json", {"complete": all(r["complete"] for r in runs),
         "training_complete": all(r["training_complete"] for r in runs), "runs": runs,
         "requested": {"seeds": args.seeds, "regimes": args.regimes, "modes": args.modes, "steps": args.steps},
         "full_design_complete": (set(args.seeds) == {43, 44, 45} and set(args.regimes) == set(REGIMES)
             and set(args.modes) == set(MODES) and args.steps in {2000, 4000}
             and args.displacement_weight == 0 and bool(data.manifest.get("full_source_coverage"))
             and len(runs) == 24 and all(r["complete"] for r in runs)),
         "data_counts": {"train_rows": int((data.split == 0).sum()), "dev_rows": int((data.split == 1).sum()),
                         "test_rows": int((data.split == 2).sum()), "train_physical": len(data.train_physical)},
         "provenance": provenance, "scope": "Camera-only data intervention; all previous outputs preserved; spatial representation, not policy/reconstruction success"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "train", "evaluate"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    parser.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    parser.add_argument("--output", type=Path, default=artifact_path("outputs") / "camera_pilot_20261026")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--extract-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--preload-gb", type=float, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dev-only", action="store_true", help="No test inference; later --resume evaluates the same saved best")
    parser.add_argument("--displacement-weight", type=float, default=0., help="Optional separate follow-up objective, never enabled in camera-only primary")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.learning_rate <= 0 or args.displacement_weight < 0:
        parser.error("Positive training settings and nonnegative displacement weight required")
    if args.command == "evaluate" and args.dev_only:
        parser.error("evaluate cannot be dev-only")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes) or len(set(args.regimes)) != len(args.regimes):
        parser.error("Duplicate requested seed/mode/regime")
    torch.set_num_threads(4)
    extract(args) if args.command == "extract" else train(args)


if __name__ == "__main__":
    main()
