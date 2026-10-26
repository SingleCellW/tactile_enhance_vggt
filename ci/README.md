# GitHub Actions activation

`github-actions.yml` is the complete CPU validation workflow, currently an
**inactive template**. GitHub rejected the maintenance push with the configured
token because it lacks `workflow` scope. No token scope was changed.

The code PR is delivered independently of that permission. Once an owner uses
an authorized workflow-capable credential/session, activate it with:

```bash
mkdir -p .github/workflows
cp ci/github-actions.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m 'ci: activate verified CPU repository checks'
git push
```

Check the actual run for the pushed commit. Local test success is not an online
CI result. Track activation and the resulting run in
[issue #6](https://github.com/SingleCellW/tactile_enhance_vggt/issues/6).
