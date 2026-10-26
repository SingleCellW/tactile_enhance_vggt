#!/usr/bin/env python3
"""Conditional matched-batch centre versus centre+displacement follow-up.

This separate runner never changes the frozen camera-only primary experiment.
Both objectives see exactly the same 32 input rows per step and initialize from
the same seed. A single forward serves centre supervision on all rows and the
optional eight-pair delta loss. No GPU job is started by importing this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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
from run_camera_pilot import Batches, autocast, predict, restore, small_state, validate
from run_token_pilot import make_block, seed_all, sha

MODES = ("visual", "internal")
OBJECTIVES = ("center", "delta")
BATCH_SCHEME = {"batch_size": 32, "ordinary_rows": 16, "object_pairs": 8,
    "layout": "ordinary[0:16], first[16:24], second[24:32]",
    "ordinary_sampling": "uniform original TRAIN physical samples, uniform camera 0..3; includes no-contact samples",
    "pair_sampling": "uniform TRAIN source group with siblings, uniform sibling within source, shared uniform camera 0..3",
    "forward_calls_per_optimizer_step": 1,
    "center_loss": "SmoothL1(beta=1), mean over all32 whitened target XYZ rows",
    "delta_loss": "SmoothL1(beta=1), mean of (second-first) on8 pairs in the same camera frame",
    "weights": {"center": 0., "delta": 1.},
    "sampling_match": "all RNGs depend on seed only; neither objective nor mode changes input sampling"}


def dump(path, value):
    """Atomic records permit independently scheduled seeds without torn JSON."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class MatchedPairSampler:
    ordinary_rows = 16
    pair_count = 8

    def __init__(self, data, seed):
        self.data = data
        self.ordinary_rng = np.random.default_rng(seed)
        self.ordinary_view_rng = np.random.default_rng(seed + 100_000)
        self.pair_rng = np.random.default_rng(seed + 200_000)
        self.pair_view_rng = np.random.default_rng(seed + 300_000)
        self.pairs = {}
        for group, first, second in data.training_pairs:
            if data.split[first] != 0 or data.split[second] != 0:
                raise ValueError("Paired supervision may use TRAIN only")
            if data.group[first] != data.group[second] or data.variant[first] != 0 or data.variant[second] == 0:
                raise ValueError("Expected same-source object nominal-to-variant pair")
            self.pairs.setdefault(int(group), []).append((int(first), int(second)))
        self.groups = np.asarray(sorted(self.pairs), dtype=np.int64)
        if not len(self.groups):
            raise ValueError("Both objectives require TRAIN source groups with object siblings")

    def sample(self):
        ordinary = self.data.sample(self.ordinary_rng, self.ordinary_view_rng, self.ordinary_rows, "augmented")
        first, second = [], []
        for _ in range(self.pair_count):
            group = int(self.groups[self.pair_rng.integers(len(self.groups))])
            candidates = self.pairs[group]
            a, b = candidates[self.pair_rng.integers(len(candidates))]
            view = int(self.pair_view_rng.integers(4))
            first.append(self.data.lookup[(int(self.data.physical_sample[a]), view)])
            second.append(self.data.lookup[(int(self.data.physical_sample[b]), view)])
        return np.asarray([*ordinary, *first, *second], dtype=np.int64)


def paired_losses(prediction, target, delta_weight):
    if prediction.shape != (32, 3) or target.shape != prediction.shape:
        raise ValueError("Matched pair losses require exactly32 XYZ rows")
    if delta_weight not in (0., 1.):
        raise ValueError("Predeclared objective weights are exactly0 or1")
    center = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.)
    delta = torch.nn.functional.smooth_l1_loss(
        prediction[24:32] - prediction[16:24], target[24:32] - target[16:24], beta=1.)
    return center + delta_weight * delta, center, delta


def initialization_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(small_state(model).items()):
        value = tensor.contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def snapshot_sources(output):
    """Compatibility name: retain source hashes, without creating copies."""
    paths = [Path(__file__), ROOT / "scripts/run_camera_pilot.py", ROOT / "scripts/run_token_pilot.py",
             ROOT / "src/tactile_vggt/camera_pilot.py", ROOT / "src/tactile_vggt/token_fusion.py",
             ROOT / "src/tactile_vggt/token_metrics.py", ROOT / "src/tactile_vggt/experiment_utils.py"]
    return source_hashes(paths)


def run_signature(args, data, seed, regime, mode, provenance):
    if args.batch_size != 32 or regime != "augmented" or args.objective not in OBJECTIVES or mode not in MODES:
        raise ValueError("Matched pair design requires batch32, augmented, visual/internal, center/delta")
    return {"format_version": 1, "seed": seed, "regime": regime, "mode": mode,
        "objective": args.objective, "steps": args.steps, "batch_size": 32,
        "batch_scheme": BATCH_SCHEME, "displacement_weight": 0. if args.objective == "center" else 1.,
        "learning_rate": args.learning_rate, "eval_every": args.eval_every, "weight_decay": 1e-4, "clip_norm": 1.,
        "hidden_dim": args.hidden_dim, "feature_dim": 256, "initial_gate": .01,
        "normalization": "shared nominal camera TRAIN only", "target_whitening": True,
        "dev_selection": "minimum episode-equal base-frame centre error over common DEV nominal+3near; earliest tie",
        "test_selection": "test is scored only after selecting the saved best; never used to pick checkpoint",
        "reference_baseline_m": data.baseline, "amp": "bfloat16" if args.device.startswith("cuda") else "float32",
        "scope": "Conditional matched-sampling objective intervention; compare paired_center with paired_delta, not directly with primary sampling",
        **provenance}


def train_one(args, data, batches, seed, regime, mode, provenance, block_factory=None):
    signature = run_signature(args, data, seed, regime, mode, provenance)
    run = args.output / f"s{seed}" / regime / mode
    config_path, marker_path = run / "config.json", run / "completion.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != signature:
            raise ValueError(f"Existing run needs --resume and the exact saved configuration: {run}")
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if not marker.get("training_complete"):
            raise ValueError("Partial training is not resumed; preserve it and use a new output")
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
    initial_sha = initialization_digest(model)
    started = time.monotonic()
    if not trained:
        sampler = MatchedPairSampler(data, seed)
        stream = hashlib.sha256()
        first_batch = None
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
        history, best, best_step = [], float("inf"), 0
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        for step in range(1, args.steps + 1):
            model.train()
            ids = sampler.sample()
            stream.update(ids.astype("<i8").tobytes())
            if first_batch is None:
                first_batch = ids.tolist()
            inputs, target = batches(ids)
            with autocast(args):
                prediction = model(**inputs)["prediction"].float()
                loss, center_loss, delta_loss = paired_losses(prediction, target, signature["displacement_weight"])
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss {seed}/{args.objective}/{mode}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                pred = predict(model, batches, data.dev_ids, args)
                base = data.base_predictions(pred, data.dev_ids)
                errors = np.linalg.norm(base - data.labels[data.dev_ids], axis=1) * 1000
                dev = episode_summary(errors, data.episode[data.dev_ids])["episode_mean_mm"]
                domains = {name: episode_summary(errors[data.camera_domain[data.dev_ids] == domain],
                    data.episode[data.dev_ids][data.camera_domain[data.dev_ids] == domain])["episode_mean_mm"]
                    for domain, name in [(0, "nominal"), (1, "near")]}
                record = {"step": step, "loss": float(loss.detach()), "center_loss": float(center_loss.detach()),
                    "displacement_loss": float(delta_loss.detach()), "grad_norm": float(gradient),
                    "dev_episode_mean_mm": dev, "dev_domains": domains, "row_stream_sha256": stream.hexdigest(),
                    "seconds": time.monotonic() - started}
                history.append(record)
                print(json.dumps({"seed": seed, "objective": args.objective, "mode": mode, **record}), flush=True)
                if dev < best:
                    best, best_step = dev, step
                    torch.save({"model": small_state(model), "kwargs": kwargs, "step": step,
                        "dev_episode_mean_mm": dev, "normalization": data.normalization, "config": signature,
                        "initialization_sha256": initial_sha, "row_stream_sha256_at_selected_step": stream.hexdigest()}, run / "best.pt")
                dump(run / "history.json", history)
        del optimizer
        marker = {"complete": False, "training_complete": True, "test_complete": False,
            "best_step": best_step, "dev_episode_mean_mm": best, "train_steps": args.steps,
            "training_seconds": time.monotonic() - started, "initialization_sha256": initial_sha,
            "row_stream_sha256": stream.hexdigest(), "row_stream_encoding": "concatenated step-order little-endian int64 row IDs",
            "first_batch_row_ids": first_batch,
            "allocated_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_gpu_gb": torch.cuda.max_memory_allocated(args.device) / 1e9 if args.device.startswith("cuda") else 0,
            "checkpoint_sha256": sha(run / "best.pt"), "config_sha256": sha(config_path)}
        dump(marker_path, marker)
    elif initial_sha != marker.get("initialization_sha256"):
        raise ValueError("Same-seed model initialization differs from completed training")
    if args.dev_only:
        del model, block
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return marker
    checkpoint = torch.load(run / "best.pt", map_location=args.device, weights_only=False)
    restore(model, checkpoint)
    ids = data.test_ids
    prediction = predict(model, batches, ids, args)
    metrics, predicted_base, errors = evaluate_predictions(data, ids, prediction)
    np.savez_compressed(run / "test_predictions.npz", indices=ids, prediction_camera=prediction,
        target_camera=data.target[ids], prediction_base=predicted_base, target_base=data.labels[ids],
        episode=data.episode[ids], group=data.group[ids], physical_sample=data.physical_sample[ids],
        variant=data.variant[ids], camera_variant=data.camera_variant[ids], camera_domain=data.camera_domain[ids],
        camera_to_base=data.camera[ids], errors_mm=errors, baseline_m=data.baseline)
    metrics.update(complete=True, seed=seed, regime=regime, mode=mode, objective=args.objective,
        best_step=checkpoint["step"], dev_episode_mean_mm=checkpoint["dev_episode_mean_mm"], train_steps=args.steps,
        config_sha256=sha(config_path), checkpoint_sha256=sha(run / "best.pt"),
        initialization_sha256=marker["initialization_sha256"], row_stream_sha256=marker["row_stream_sha256"])
    dump(run / "metrics.json", metrics)
    marker.update(complete=True, test_complete=True, metrics_sha256=sha(run / "metrics.json"),
                  predictions_sha256=sha(run / "test_predictions.npz"))
    dump(marker_path, marker)
    del model, block
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def paired_integrity(output):
    rows = []
    for seed in (43, 44, 45):
        for mode in MODES:
            markers = []
            for objective in OBJECTIVES:
                path = Path(output) / f"paired_{objective}" / f"s{seed}" / "augmented" / mode / "completion.json"
                markers.append(json.loads(path.read_text()) if path.exists() else {})
            ready = all(m.get("training_complete") for m in markers)
            row = {"seed": seed, "mode": mode, "both_training_complete": ready}
            if ready:
                for key in ("train_steps", "initialization_sha256", "row_stream_sha256"):
                    if markers[0].get(key) != markers[1].get(key) or markers[0].get(key) is None:
                        raise ValueError(f"Unmatched centre/delta experiment: {seed}/{mode}/{key}")
                    row[key] = markers[0][key]
                configurations = []
                for objective, marker in zip(OBJECTIVES, markers):
                    run = Path(output) / f"paired_{objective}" / f"s{seed}" / "augmented" / mode
                    if sha(run / "config.json") != marker.get("config_sha256"):
                        raise ValueError("Paired run configuration changed after completion")
                    config = json.loads((run / "config.json").read_text())
                    # Device placement is recorded independently. Scientific
                    # inputs, hyperparameters and source versions must match.
                    config = {k: v for k, v in config.items() if k not in
                              {"objective", "displacement_weight", "environment_sha256", "environment_record"}}
                    configurations.append(config)
                if configurations[0] != configurations[1]:
                    raise ValueError(f"Centre/delta shared configuration differs: {seed}/{mode}")
                row["matched"] = True
            rows.append(row)
    result = {"complete": all(r["both_training_complete"] for r in rows), "pairs": rows,
              "scope": "Input row stream, optimizer budget and model initialization match within each seed/mode"}
    dump(Path(output) / "paired_training_integrity.json", result)
    return result


def train(args):
    validate(args)
    data = CameraArrays(args.data)
    root_output = args.output
    root_output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(root_output, args)
    protocol = {"format_version": 1, "conditional_followup": True, "batch_scheme": BATCH_SCHEME,
        "steps": args.steps, "modes": list(MODES), "objectives": list(OBJECTIVES), "seeds": [43, 44, 45],
        "learning_rate": args.learning_rate, "eval_every": args.eval_every, "hidden_dim": args.hidden_dim,
        "image_size": args.image_size, "runner_sha256": sha(__file__),
        "data_manifest_sha256": sha(args.data / "manifest.json"), "prefix_manifest_sha256": sha(args.prefix / "manifest.json"),
        "scope": "Separate conditional follow-up. Matching centre-only control is retrained with the same paired sampler; original primary is unchanged."}
    protocol_path = root_output / "pair_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Pair protocol changed; use a new output root")
    dump(protocol_path, protocol)
    batches = Batches(data, args.prefix, args.device, args.preload_gb)
    for objective in args.objectives:
        run_args = argparse.Namespace(**vars(args))
        run_args.objective, run_args.output = objective, root_output / f"paired_{objective}"
        run_args.output.mkdir(exist_ok=True)
        norm_path = run_args.output / "normalization.json"
        if norm_path.exists() and json.loads(norm_path.read_text()) != data.normalization:
            raise ValueError("Output normalization changed")
        dump(norm_path, data.normalization)
        environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "cuda_version": torch.version.cuda,
            "device": args.device, "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        # Independently scheduled seeds can occupy different physical GPUs.
        # Keep immutable per-environment records, rather than a shared mutable
        # root environment.json that would reject otherwise matched workers.
        environments = run_args.output / "environments"
        environments.mkdir(exist_ok=True)
        environment_key = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
        env_path = environments / f"{environment_key}.json"
        if env_path.exists() and json.loads(env_path.read_text()) != environment:
            raise ValueError("Recorded environment changed")
        dump(env_path, environment)
        provenance = {"data_manifest_sha256": sha(args.data / "manifest.json"),
            "prefix_manifest_sha256": sha(args.prefix / "manifest.json"), "normalization_sha256": sha(norm_path),
            "environment_sha256": sha(env_path), "runner_sha256": sha(__file__),
            "environment_record": str(env_path.relative_to(run_args.output)),
            "helper_sha256": sha(ROOT / "src/tactile_vggt/camera_pilot.py"),
            "token_fusion_sha256": sha(ROOT / "src/tactile_vggt/token_fusion.py"),
            "source_hashes": snapshot_sources(run_args.output)}
        runs = []
        for seed in args.seeds:
            for mode in args.modes:
                if args.command == "evaluate" and not (run_args.output / f"s{seed}" / "augmented" / mode / "completion.json").exists():
                    raise ValueError("evaluate requires previously completed training")
                run_args.resume = args.resume or args.command == "evaluate"
                result = train_one(run_args, data, batches, seed, "augmented", mode, provenance)
                runs.append({"seed": seed, "regime": "augmented", "mode": mode, "objective": objective,
                    "training_complete": bool(result.get("training_complete", result.get("complete"))),
                    "complete": bool(result.get("complete")), "metrics": f"s{seed}/augmented/{mode}/metrics.json" if result.get("complete") else None})
                paired_integrity(root_output)
        summary = {"complete": all(r["complete"] for r in runs), "training_complete": all(r["training_complete"] for r in runs),
            "runs": runs, "requested": {"seeds": args.seeds, "modes": args.modes, "objective": objective, "steps": args.steps},
            "full_design_complete": (set(args.seeds) == {43, 44, 45} and set(args.modes) == set(MODES)
                and args.steps == 4000 and bool(data.manifest.get("full_source_coverage")) and len(runs) == 6 and all(r["complete"] for r in runs)),
            "provenance": provenance, "scope": protocol["scope"]}
        dump(run_args.output / ("summary_seeds_" + "_".join(map(str, args.seeds)) + "_modes_" + "_".join(args.modes) + ".json"), summary)
        dump(run_args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=dependency_path("vggt"))
    parser.add_argument("--weights", type=Path, default=artifact_path("weights") / "model.pt")
    parser.add_argument("--output", type=Path, default=artifact_path("outputs") / "camera_pair_pilot_20261026")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--preload-gb", type=float, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.batch_size != 32 or args.learning_rate <= 0:
        parser.error("Positive budget/evaluation settings and batch-size exactly32 required")
    if any(len(set(values)) != len(values) for values in (args.seeds, args.modes, args.objectives)):
        parser.error("Duplicate seed/mode/objective")
    if args.command == "evaluate" and args.dev_only:
        parser.error("evaluate cannot be dev-only")
    torch.set_num_threads(4)
    train(args)


if __name__ == "__main__":
    main()
