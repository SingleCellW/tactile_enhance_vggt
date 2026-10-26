# Contributing

Original project contributions use [Apache-2.0](LICENSE). Preserve applicable
third-party attribution and document external material's separate terms; the
project license does not relicense downloaded code, models or assets.

Use one source tree under `src/tactile_vggt`. Keep distinct research baselines and
select them through configuration. Do not maintain dated or `final_v2` copies of
the same implementation. Git commits record code versions; run IDs record experiments.

Before work, synchronize the target branch and inspect the working tree. Use a
focused branch, commit reviewable changes, push that same history, and open a PR
against `main`. Do not edit an independent copy on GitHub or a training machine.
Training machines must record the exact commit they execute. When networking is
unavailable, use a commit-derived archive and record its checksum.

Open or update an issue for a reproducible defect, unmet acceptance criterion or
research question. Include priority, source commit, reproduction command,
environment, evidence location/hash and explicit acceptance checkboxes. Link the
fixing PR. Use `Closes #N` only when all criteria are delivered; a PR in review
does not mean the issue is already resolved. Keep partial fixes and scientific
uncertainties open, with a dated progress update. Search existing issues first.

Run the checks in [validation](docs/validation.md). Add a focused regression test
when fixing a behavioral bug. A CPU unit test, a GPU smoke test and a completed
robot rollout establish different things; report them separately, including
skips and unavailable inputs. Do not relabel historical results as new measurements.

Track code, experiment configuration, tests, reusable templates and maintained
documentation. Keep raw data, weights, caches, checkpoints, logs, report results,
media and deployment bundles under the ignored artifact root. Local machine
configuration belongs in `configs/machines/local.json` (or `*.local.json`).
Review an explicit staged file list before committing. Never stage the workspace
wholesale. Do not rewrite shared history to remove ordinary historical reports.

Use `feat:`, `fix:`, `refactor:`, `test:`, `build:`, `docs:` or `chore:` commit
subjects where useful. Describe the final behavior and validation in the PR;
retain negative findings and link unresolved issues. A tag marks a reviewed
milestone after merge, not an unverified working directory.
