# Portal sources

This directory contains the maintained catalog template and reusable CSS/JS.
Generated pages, metrics, media, downloaded evidence and copied source snapshots
belong under the configured artifact root, not in Git.

After installing the project, build a clean catalog without experiment data:

```bash
python -m portal_host.build_site
```

The default output is `.local/deploy/site-preview/`. Existing destinations are
refused so a preview cannot replace an archived report. Choose a fresh `--output`
for another build.

To use existing generated reports, provide the artifact site explicitly:

```bash
python -m portal_host.build_site \
  --report-source .local/site \
  --output .local/deploy/site-candidate
python -m portal_host.deploy --build-only --source .local/deploy/site-candidate
```

The catalog reads `data/reports.json`; entries use `title`, `summary`, and `url`.
`--registry` accepts an explicitly prepared replacement registry. Internal URLs
must have matching files in the supplied report source. Source files remain
unchanged; the builder copies regular files and rejects symlinks.

The maintenance migration preserved the former complete website at `.local/site`
and recorded its original file hashes in `.local/migration/portal-paths.json`.
That site includes historical source snapshots as evidence, not maintained code.
Existing reports and their original manifests are not retroactively attributed
to the maintenance commit.

Report generators remain in `scripts/`. Their defaults use the artifact root:
`build_stage1a_portal_data.py` projects existing results into JSON;
`build_stage1b_report.py` renders audited results and a reviewed narrative;
`render_experiment_report.py` needs the historical template and result files.
A clean Git clone intentionally contains none of those experiment results.
