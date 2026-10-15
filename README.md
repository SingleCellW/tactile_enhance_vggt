# Frozen VGGT + tactile ACT

Simulation pilot for spatial tactile anchoring in the existing pick-and-place task.
VGGT is **frozen and run offline once**. A small spatial point encoder and the
current LeRobot ACT are trained. This repository does not modify upstream LeRobot.

## Pinned sources

- LeRobot submodule: `297774fca4e3ad0aa34e7425fbf937111637e77b` (0.6.2, Python 3.12).
- VGGT: `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- Official VGGT-1B `model.pt` SHA256:
  `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
- ACT reference: remote `rgbs_act_backbone_dynamic_20260923/configs/pickplace_act_rgbs_r3m18_dyn44_s43.json`.

## Experiment contract

- Same-time left/right RGB pairs; no future images or whole-episode reconstruction.
  Rendering audit establishes that stored RGB row `r` depicts recorded state
  row `r-1`; repaired observations explicitly correct this historical lag.
- VGGT predicted camera/depth heads, BF16 inference, no gradients. No GT depth.
- Known first-camera orientation/origin and known stereo baseline fix the global
  coordinate and scale ambiguity. This is a **calibrated** experiment.
- 512 sampled visual points and 44 corrected tactile anchors. Channels are
  `[x, y, z, r, g, b, pressure, is_tactile]`; XYZ stays in base-frame metres.
- Shared point MLP plus local visual neighborhoods around tactile anchors feed
  an ACT condition. This is a lightweight PointNet-style encoder, not PointNet++.
- `anchored` and `unanchored` use the same parameter allocation, visual points,
  training split, and global tactile readings. The latter does not read tactile
  positions or activate the spatial contact branch. `no_tactile` is also supported.
- 22 joint-position observations and 22 joint-position target actions.
- All 53 historical episodes are retained to preserve the source training set,
  including trajectories with failed success labels. Do not describe all 53 as
  successful demonstrations. Split by original episode (seed 43, 10% validation).
- Pilot geometry observations use stride 4. Action labels remain the full 20 Hz
  source stream, including its historical next-frame command convention.
- Normalization uses training episodes only. Action chunks never cross episodes.

## Historical data alignment

The inherited dataset pairs original recorded RGB/state with tactile generated
by a later controller replay. Its tactile anchors are not consistently aligned
to the recorded hand: an audit found a 26.2 mm 90th-percentile error among active
sensor/frame pairs, with a 115.7 mm maximum. Those inherited tactile observations
must not be used to assess spatial anchoring.

The current pilot recomputes native tactile at the complete recorded 29D simulator
state, using causal generalized velocities from consecutive timestamps within each
episode. Object pose is used only to synthesize tactile, never as a policy input.
Both force and anchors are replaced together. `repair_cached_tactile.py` creates a
new cache, verifies frame/state correspondence, preserves retained VGGT visual points
bit for bit, and recomputes train-only pressure normalization. Its manifest records
the regeneration report and hashes. The paired launcher requires this repaired cache.
Each trajectory's first cached RGB is dropped because its previous recorded
state is unavailable. Remaining RGB row `r` is paired with state/force and the
existing source action label at `r-1`; action arrays retain their original rate.
See [native regeneration](docs/native_tactile_repair.md) for the force-proxy
assumptions and [paired protocol](docs/ablation_protocol.md) for comparison limits.

## Training

`configs/act_pilot.json` retains the reference ACT shape, LR, KL weight and chunk
settings: batch 64, AdamW 1e-5, weight decay 1e-4, gradient clip 10, chunk 16,
execute 8, hidden 512, encoder 4, decoder 1, KL weight 10, no LR scheduler.
BF16, TF32, fused AdamW, memory-mapped cached geometry, pinned transfers and
persistent workers avoid repeated visual inference and video decoding.

The initial step budget preserves approximate reference sample exposure:

`160000 * train_observations / 20212 * 64 / batch_size`, rounded to 1000 steps.

This is an initial budget, not evidence of convergence. The held-out action L1
is an imitation diagnostic, **not a closed-loop task success rate**.

On rimlab-server, the isolated `.venv` inherits the existing Python 3.12 CUDA
PyTorch installation without modifying it. Project-local packages pin compatible
NumPy and torchvision CUDA wheels. Source and dependencies are separate from
the old tactile project.

After creating and checking the synchronized tactile cache, submit the paired pilot:

```bash
nohup bash scripts/run_pilot.sh > logs/pilot_supervisor.log 2>&1 < /dev/null &
```

The supervisor uses physical GPU 1/2 and trains anchored/unanchored ACT against
the same completed `cache/pickplace_vggt_native_sync_stride4` cache.
It refuses duplicate campaign names and insufficient free GPU memory. Inspect
the process IDs and logs under `logs/`, and `outputs/*/status.json` for loss,
step, memory, throughput and Shanghai-time ETA. Checkpoints are written atomically
at 100, 1000, every 5000, and the final step; `latest` points to a complete save.

Direct single-run or resume commands:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_act.py \
  --cache cache/pickplace_vggt_native_sync_stride4 --output outputs/example --mode anchored
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_act.py \
  --cache cache/pickplace_vggt_native_sync_stride4 --output outputs/resumed --mode anchored \
  --resume outputs/example/checkpoints/005000
```

## Inference

`PointACTInference` loads a fixed checkpoint and its normalization, accepts raw
robot state / tactile / calibrated point observations, and returns physical
joint targets. `needs_observation` avoids recomputing VGGT while the action queue
is nonempty; `reset()` clears it. A supplied point-builder callback can create
points from current RGB and calibration at each refresh.

Simulator deployment must use the same native tactile adapter and sensor ordering
as the repaired training readouts. The inherited replay-generated forces are not
interchangeable with the repaired observations. Geometry refresh currently runs
synchronously when the action queue empties; this adapter alone does not establish
continuous 20 Hz closed-loop operation. Async refresh/prefetch requires separate
control-loop validation.

`scripts/benchmark_inference.py` separately measures fresh ACT chunks, normalized
point-input processing, queued actions, and optionally true RGB→VGGT→ACT refreshes:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/benchmark_inference.py \
  --checkpoint outputs/example/latest --cache cache/pickplace_vggt_native_sync_stride4 \
  --device cuda:0 --amp bfloat16 --output outputs/example/latency.json \
  --geometry-input cache/pickplace_vggt_native_sync_stride4/example_observation.npz \
  --vggt-repo vendor/vggt --vggt-weights weights/model.pt
```

Full simulator control-loop evaluation and task success comparisons are a later
step; cached-geometry latency must not be presented as full-system latency.

## Verification and synchronization

```bash
PYTHONPATH=src:lerobot/src .venv/bin/python -m pytest tests -q
```

Tests cover coordinate/scale transforms, modality isolation, gradient flow,
episode padding, train-only statistics, checkpoint round trips, deployment
normalization and action queues. Cache manifests record source and model hashes,
episode splits and per-frame geometry diagnostics. Git tracks source/configuration
and the LeRobot submodule; weights, datasets, caches and outputs stay out of Git.
