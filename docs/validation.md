# Maintenance validation

Validation date: 2026-10-26. These checks cover repository portability, preserved
algorithm contracts and report regeneration. They do not claim newly trained
models, new policy success measurements or live campus deployment.

## Reproduce the checks

Install the Linux Python 3.12 CPU environment from [envs](../envs/README.md),
including the locked LeRobot source and MuJoCo 3.4.0. Then run:

```bash
python -m pip check
python scripts/bootstrap_dependencies.py --dependency lerobot --check
python scripts/doctor.py --profile train
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  python -m pytest tests -q -ra
python -m portal_host.build_site --output .local/qa/clean-shell
python -m portal_host.deploy --build-only --source .local/qa/clean-shell
python scripts/check_repository.py
git diff --check
```

Use a fresh shell output directory on each build. The builder deliberately refuses
to overwrite previous evidence. The prepared CI template runs the same CPU checks
and clean portal build, then verifies source-tree cleanliness. It is not active:
GitHub rejected workflow upload because the current token lacks `workflow` scope.
See [activation instructions](../ci/README.md) and issue #6.

## Completed checks

- Fixed Python 3.12.10 environment: actual imports verified for LeRobot 0.6.2 at
  `297774fca4e3ad0aa34e7425fbf937111637e77b`, CPU Torch 2.9.1 and torchvision
  0.24.1. Installed dependency compatibility passed.
- Shared LeRobot and VGGT checkouts passed commit/clean-worktree checks. The
  latter is fixed at `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- A clean clone at `a3f61f8`, with a fresh Python 3.12 environment installed from
  the 67-package CPU lock and shared pinned LeRobot source, passed **243 tests,
  zero failures and zero skips**. No historical data, outputs, machine config or
  project environment were copied into that clone. Coverage includes ACT gradients,
  checkpoint round trips, cached/live feature parity, coordinate/data isolation,
  camera sampling/training fixtures, native MuJoCo synthetic state preservation,
  relocation without rewriting evidence, configuration and scheduler boundaries.
- Seventeen representative training/evaluation/data/dependency CLIs accepted
  `--help` in the isolated environment without private inputs or starting jobs.
- Wheel build and an independent wheel-install import smoke passed. Artifacts
  defaulted to the caller's directory rather than site-packages. The wheel contains
  package source and metadata, excluding external repositories and run outputs.
- All 2,597 moved project artifact/document files were checked before/after using
  SHA256. All 374 files in the preserved portal were separately verified unchanged.
- Stage 1.a data rebuilt with complete inputs and no missing evidence. The legacy
  80-rollout report rebuilt with 196 input hashes; Stage 1.b main and paired reports
  rebuilt into fresh QA outputs. These regenerate existing results, not new runs.
- A complete candidate site/bundle validated 376 public files and 861 internal
  references, plus all recorded hashes. The data-free shell also built and passed
  static link/hash checks. No SSH publication or browser interaction check occurred.
- The clean clone also passed dependency compatibility, active ACT source checks,
  data-free portal build and the 153-file source-boundary check. Its Git working
  tree remained clean after all checks. Later changes to this document only record
  those results; they do not alter the tested implementation.

Raw test logs, JUnit output, build receipts and migration manifests live under
`.local/maintenance/`, `.local/qa/` and `.local/migration/`; they are not tracked.
This maintained document records the scope, rather than copying generated reports.

## Limits

The CPU suite includes synthetic MuJoCo state checks, not the historical native
tactile worker or full PickLift rollouts. The latter require the external inputs
in [issue #2](https://github.com/SingleCellW/tactile_enhance_vggt/issues/2).
VGGT weight inference, CUDA training, real-time control and new policy experiments
were not run during maintenance. Scientific limitations remain in
[issue #3](https://github.com/SingleCellW/tactile_enhance_vggt/issues/3).
Portal rollback service-state recovery is tracked separately in
[issue #4](https://github.com/SingleCellW/tactile_enhance_vggt/issues/4).

Local checks are not a claim of GitHub Actions success. No online run is claimed;
activation and verification of an actual run remain tracked in issue #6.
