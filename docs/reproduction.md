# Reproduction workflow

Start with the [tested installation](../README.md#install-and-verify). Run from
the repository root in a clean environment without another project's `PYTHONPATH`.
The CPU lock is for regression checks; use matching CUDA wheels and record the
resolved environment for GPU training.

## Inputs and machine settings

Copy `configs/machines/example.json` to ignored `configs/machines/local.json`
and supply actual paths. CLI paths override defaults; machine defaults resolve
environment variable `TACTILE_VGGT_<NAME>` before local JSON. Relative JSON paths
resolve against the JSON file's directory. `TACTILE_VGGT_MACHINE_CONFIG` selects
a different config file.

The native worker needs `worker_python`, `dynamic_root`, `raw_root`,
`timestamps_root` and `source_readouts`. The dependency lock lists expected
historical asset paths; `python scripts/doctor.py --profile simulation` checks
prerequisites. This does not prove a rollout works. See
[issue #2](https://github.com/SingleCellW/tactile_enhance_vggt/issues/2).

Fetch VGGT source with `python scripts/bootstrap_dependencies.py --fetch`.
Place official weights at `.local/weights/model.pt`, or pass an explicit path.
The expected SHA256 and URL are in `dependencies.lock.json`;
`scripts/download_weights.py --help` describes the downloader.
`python scripts/doctor.py --profile full --check-weights --require-cuda` checks a
fully provisioned machine. It is not a benchmark.

Set `TACTILE_VGGT_ARTIFACT_ROOT` to relocate generated content. Examples below use
the default `.local/`. Input names must refer to audited existing data; these
commands do not create the input datasets.

## Existing point baseline

```bash
python scripts/train_act.py \
  --cache .local/cache/pickplace_vggt_native_physics_dt01_stride4 \
  --config configs/act_physics_aligned_v2.json \
  --mode anchored --output .local/runs/point-anchored-s43 --seed 43
```

`scripts/run_ablation_suite.py prepare --help` describes how to create a paired
plan. Its `execute` command launches the declared runs after validating repaired
data, RGB alignment and source/config hashes. `scripts/run_pilot.sh` requires
explicitly allocated GPUs; it does not select a lab GPU by hostname.

## Spatial token and camera experiments

`build_token_pilot_data.py` creates native spatial observations;
`run_token_pilot.py` extracts frozen VGGT prefixes and trains spatial controls.
`build_camera_pilot_data.py` renders camera variants of identical physical states.
`run_camera_pilot.py` provides `extract`, `train` and `evaluate`.

```bash
python scripts/run_camera_pilot.py train \
  --data .local/artifacts/camera_pilot_data_20261026 \
  --prefix .local/artifacts/camera_pilot_prefix_20261026 \
  --output .local/runs/camera-reproduction \
  --seeds 43 44 45 --steps 4000 --batch-size 32
```

Historical input directories must exist and pass their manifests. Extraction is
separate from training. Keep fixed/augmented regimes, modalities, selection
rules and budgets matched. The paired-displacement runner is a separate objective,
not a hidden change to the camera-only comparison.

`train_token_act.py` and `train_simple_token_act.py` require matching feature
manifests and repaired action caches. The simple command also requires explicit
`--visual-probe`, `--projection` and `--tactile`. Use the corresponding evaluator
with the same timing contract. Static object variants cannot inherit unrelated
demonstration actions. See [data contracts](data_contract.md).

## Provenance and cluster execution

Training commands append records under `<output>/provenance/`: commit, dirty
state, source/dependency hashes, resolved config, input manifests, supplied weight
hashes, command, interpreter and package versions. Missing revision or input
information remains explicit. Existing checkpoint/signature checks govern resume.

Commit and push code before formal experiments; record the commit actually
checked out on the worker. Use a new run ID for a changed protocol, and retain the
full environment with the run. Dirty-source hashes document a diagnostic run but
do not make an unpublished checkout a published reproduction.

`scripts/hpc_stage1b_pair.sh` preserves scheduler GPU allocation. Supply account,
partition, resource and output options to `sbatch` for your cluster. Do not copy
another user's account or home directory into the launcher.

```bash
mkdir -p .local/logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --gres=gpu:1 \
  --output=.local/logs/stage1b-%j.out --error=.local/logs/stage1b-%j.err \
  scripts/hpc_stage1b_pair.sh 43 4000
```

The scheduler opens output files before the script starts; provide these paths
at submission time. Scheduler/account values above are placeholders.

## Reports

Summarizers write under the artifact reports directory. Stage 1.a/1.b builders
read matching result inputs, not guessed numbers. Historical document inputs are
in `.local/reports/historical_docs/`. Archive reports separately and link them
from issues with run/commit IDs. [Portal tooling](../portal_host/README.md)
describes shell builds, report builds and explicit publication. A Git push never
deploys the campus site.
