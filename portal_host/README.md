# Report portal tools

These tools build, validate and optionally deploy this project's static report
portal. Building and local verification do not start services or connect to SSH.
Generated bundles, deployment receipts and QA output belong under the artifact
root (`.local/` by default; override `TACTILE_VGGT_ARTIFACT_ROOT`).

## Local build and verification

Install the project first, then run from the repository root:

```bash
python -m portal_host.build_site
python -m portal_host.deploy --build-only --source .local/deploy/site-preview
python -m portal_host.verify --bundle .local/deploy/bundles/<release> --local-only
```

For a complete archived report site use `--source .local/site`, or build a new
catalog around it with `build_site --report-source .local/site --output <new-path>`.
The new catalog refuses an existing output directory. Report generation remains
separate from publication and must use the corresponding original input files.

Each bundle has immutable `public/`, `runtime/`, and `manifest.json`. The manifest
records every public/runtime file's size and SHA256. Validation rejects symlinks,
hidden paths, special files, URL escapes, missing resources and modified payloads.
Static HTML/CSS/JS and explicit JSON URL fields are checked; browser interactions
still need manual or browser-based QA before publication.

## Explicit deployment configuration

Copy `portal_host/config.example.json` to an ignored local file such as
`configs/machines/portal.local.json` and set:

- `remote_host`: the SSH alias of the target account.
- `remote_home`: that account's absolute home directory.
- `bind`, `port`, `url`: the server bind address, unprivileged TCP port and public URL.

Pass the same file while building and deploying, or set
`TACTILE_VGGT_PORTAL_CONFIG`. Without a remote host/home, the tools can build local
previews but refuse deployment. Configuration cannot select another service or
arbitrary remote directory: the service is `tactile-vggt-report.service` and the
owned directory is `<remote_home>/tactile-vggt-report-host`.

```bash
python -m portal_host.deploy --build-only --source .local/site \
  --config configs/machines/portal.local.json
# Explicitly publishes the selected, verified bundle:
python -m portal_host.deploy --bundle .local/deploy/bundles/<release> --deploy \
  --config configs/machines/portal.local.json
python -m portal_host.verify --bundle .local/deploy/bundles/<release> --via-ssh \
  --config configs/machines/portal.local.json \
  --output .local/qa/verification.json
```

The bundle's destination must exactly match the active configuration before SSH
is attempted. Remote checks retain account/home ownership, the existing portal
marker, dedicated service identity, port availability, and existing `Linger=yes`.
They never enable linger, change firewall rules, use sudo, or edit another service.

Upload goes to `.incoming/<release>`, verifies hashes, then atomically switches
`current` to `releases/<release>`. Activation requires both service readiness and
an HTTP hash match for `data/reports.json`. A failed activation attempts to restore
the prior release/runtime/unit. HTTP verification checks all bytes and media Range
responses; `--via-ssh` validates reachability from the SSH host, not a user's browser.

## Retention and recovery

Historical migration bundles are preserved under `.local/deploy/bundles` and their
manifests remain unchanged. New builds do not delete old snapshots. Back up important
reports independently of Git. Local previews and expendable caches may be cleaned
separately; do not treat the only copy of an experiment result as a cache.

Remote rollback uses `backups/<release>/previous-current.json`, the saved runtime
and unit, and the old `releases/<release>` target. Do not edit `current/public` in
place or remove locks while a deployment is active. The rollback mechanism preserves
contents, but does not capture an intentionally disabled/stopped prior unit state.
