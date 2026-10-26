# Repository and artifact ownership

The Git root is `tactile_enhance_vggt/`. `src/tactile_vggt/` is the single maintained
implementation. `scripts/` contains command entry points; shared runtime and
experiment helpers belong in the package. Distinct point, token and simple
policy baselines remain available. Configuration selects a method; Git selects
a source version.

| Location | Responsibility | Tracked |
|---|---|---|
| `src/`, `scripts/`, `tests/` | Implementation, commands and regression checks | Yes |
| `configs/`, `envs/`, dependency lock | Portable configuration and dependency versions | Yes |
| `docs/`, README | Maintained contracts and reproduction instructions | Yes |
| `portal/templates/`, `portal/assets/`, `portal_host/` | Site source and publication tools | Yes |
| `external/` | Local symlinks into shared pinned checkouts | No |
| `.local/data/`, `.local/weights/` | Input data and model checkpoints | No |
| `.local/cache/`, `.local/artifacts/` | Derived features and intermediate evidence | No |
| `.local/runs/`, `.local/logs/` | Training/evaluation outputs | No |
| `.local/reports/`, `.local/site/` | Full reports, figures, videos, generated site | No |
| `.local/deploy/`, `.local/qa/`, `.local/tmp/` | Bundles, receipts and temporary files | No |

Set `TACTILE_VGGT_ARTIFACT_ROOT` to relocate generated artifacts. Machine input
paths are configured by CLI, environment or ignored `configs/machines/local.json`;
see [reproduction](reproduction.md). Shared dependency locations can differ on
every machine. Their commits, not those absolute paths, are portable.

During the 2026-10-26 maintenance, the existing `outputs`, `cache`, `artifacts`
and `reports` directories were moved into their corresponding artifact categories.
Existing dated documents and JSON evidence were preserved, byte for byte, in
`.local/reports/historical_docs/`. Old report-site source downloads remain inside
the preserved `.local/site/`; they are historical evidence, not editable copies.
Migration inventories live in `.local/migration/`. Do not rewrite historical
manifests to make them look as if they ran on the new source commit.

The formerly uncommitted source was first recorded in commit `fd8f65d`. This is
a recovery baseline, not proof of the exact source used by every old run. Earlier
published commits remain in Git history. Ordinary historical reports are removed
from the current tree without force-pushing or rewriting that history.

Generated reports are reproducible outputs of the report scripts, not tracked
documents. A public code clone can build the generic portal shell. Rebuilding
historical scientific reports also requires the preserved run evidence. The
existing campus report service is a separate deployment, not updated by a Git
push or this repository-maintenance PR.

Keep important runs and reports backed up separately. Only disposable caches and
unneeded build bundles are cleanup candidates; no automatic deletion of historical
results is configured. Run `python scripts/check_repository.py` before a PR.
