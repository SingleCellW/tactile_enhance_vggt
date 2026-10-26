# Repository maintenance

- Keep one implementation in `src/tactile_vggt`; preserve scientifically distinct
  baselines and choose them with configuration. Use commits, not dated source copies.
- Read `README.md`, `docs/data_contract.md` and the relevant tests before changing
  experiment behavior. Do not silently change timing, coordinate, split, action,
  normalization or checkpoint-selection contracts while reorganizing code.
- Use `external/` links and `dependencies.lock.json` for third-party sources.
  Never switch or modify a shared pinned checkout under another running job.
- Put generated data, runs, reports, site builds and receipts under the ignored
  artifact root. Preserve historical evidence and hashes; never assign a new
  commit retroactively to an old run. Machine configuration is local and ignored.
- Use the documented Python 3.12 environment and actual pinned LeRobot for full
  CPU checks. Report tests, skipped prerequisites and GPU/native-simulation scope
  separately. Run `scripts/check_repository.py` before a delivery commit.
- Record reproducible unresolved problems in GitHub issues with priority,
  evidence, reproduction and acceptance criteria. Link the fixing PR, and keep
  partial fixes/research uncertainties open. Follow the user's current task scope;
  issue tracking does not authorize new long-running experiments or deployments.
- Preserve the existing Git history. Review explicitly staged paths, commit and
  push the same source history, and prepare the requested PR. Do not merge it or
  deploy the report service unless the user requests those actions.
