# Tactile Enhance VGGT

Research code for combining frozen VGGT visual representations with tactile
observations and ACT policies in a calibrated PickLift simulation. The current
source includes the point-cloud baseline, internal/late token fusion, simpler
policy controls, and Stage 1.b camera augmentation and displacement experiments.

This is a research baseline. Historical experiments do **not** yet establish a
consistent internal-fusion advantage, reliable spatial displacement response or
improved policy success. See [research status and open issues](docs/research_status.md).

## Install and verify

Use Linux and **Python 3.12** for the pinned LeRobot ACT implementation. The
following installs the tested CPU environment; it does not download training
data or VGGT weights and does not launch training.

```bash
git clone https://github.com/SingleCellW/tactile_enhance_vggt.git
cd tactile_enhance_vggt
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
python scripts/bootstrap_dependencies.py --dependency lerobot --fetch
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r envs/cpu-py312.lock.txt
python -m pip install --no-deps --no-build-isolation -e '.[train,report,dev]' -e external/lerobot
python -m pip check
python scripts/doctor.py --profile train
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 python -m pytest tests -q
```

While the maintenance PR is open, check out `chore/repository-standardization`
after cloning; `main` receives this layout only when the PR is merged.
[Environment details](envs/README.md) distinguish core, CPU verification, CUDA
training and the separate simulation worker environment.

For the visual backbone, run `python scripts/bootstrap_dependencies.py --fetch`.
This creates ignored `external/lerobot` and `external/vggt` links into a shared
pool, at the commits in [dependencies.lock.json](dependencies.lock.json).
Reuse an existing checkout with `--local NAME=/path/to/checkout`; check all links
with `--check`. The tool refuses wrong revisions, dirty sources and replacement
of existing directories. No third-party source is vendored into this repository.

## Implementations and entry points

| Purpose | Entry point |
|---|---|
| Point-cloud ACT baseline | `scripts/train_act.py`, `scripts/evaluate_sim.py` |
| Spatial token fusion | `scripts/build_token_pilot_data.py`, `scripts/run_token_pilot.py` |
| Token-conditioned ACT | `scripts/train_token_act.py`, `scripts/evaluate_token_act.py` |
| Simpler policy control | `scripts/train_simple_token_act.py`, `scripts/evaluate_simple_token_act.py` |
| Camera augmentation | `scripts/build_camera_pilot_data.py`, `scripts/run_camera_pilot.py` |
| Paired displacement objective | `scripts/run_camera_pair_pilot.py` |
| Report generation | `scripts/summarize_*.py`, `scripts/build_stage1a_portal_data.py`, `scripts/build_stage1b_report.py` |

Use `--help` for each command. [Architecture](docs/architecture.md),
[data contracts](docs/data_contract.md), and [reproduction](docs/reproduction.md)
describe inputs, coordinate conventions, matched controls and provenance.
Synthetic CPU regression tests do not require private demonstrations. Native
simulation needs historical tactile code and assets tracked by
[issue #2](https://github.com/SingleCellW/tactile_enhance_vggt/issues/2).

## Source, data and reports

Maintain code once under `src/tactile_vggt/`. Version it with commits and select
baselines by configuration. Source, tests, configurations, reusable portal
templates and maintained documentation are tracked. Generated results are not.

The default artifact root is `.local/`; set `TACTILE_VGGT_ARTIFACT_ROOT` to use
another disk. It contains `data/`, `weights/`, `cache/`, `artifacts/`, `runs/`,
`logs/`, `reports/`, `site/`, `deploy/`, `qa/` and `tmp/`. Machine configuration
is ignored under `configs/machines/local.json`. See the full
[layout and migration map](docs/repository_layout.md).

Full historical reports remain in the local artifact area and the existing
[Stage 1.a](http://10.15.89.229:8766/reports/stage1a/) and
[Stage 1.b](http://10.15.89.229:8766/reports/stage1b/) campus-network pages.
They are not included in a public clone. The maintained portal shell can be built
without those results:

```bash
python -m portal_host.build_site
python -m portal_host.deploy --build-only --source .local/deploy/site-preview
```

These commands do not publish or start a service. See
[portal instructions](portal_host/README.md) for local validation and explicit
deployment configuration. Important results need a separate backup; being ignored
by Git does not make them disposable.

## Maintenance

See [validation](docs/validation.md) for checks actually performed and their limits.
The [CI workflow template](ci/README.md) is ready, but online activation awaits
workflow permission ([#6](https://github.com/SingleCellW/tactile_enhance_vggt/issues/6)).
Report problems using [issues](https://github.com/SingleCellW/tactile_enhance_vggt/issues)
with a source commit, reproduction command, evidence and acceptance criteria.
Follow [CONTRIBUTING.md](CONTRIBUTING.md): focused local commits, push the same
history, PR to `main`, and keep unresolved scientific issues open.

Please cite this repository via [CITATION.cff](CITATION.cff) and credit upstream
[VGGT](https://github.com/facebookresearch/vggt) and
[LeRobot](https://github.com/huggingface/lerobot) separately. Third-party source,
weights and historical robot/data assets retain their own terms;
see [third-party notes](docs/third_party.md).
