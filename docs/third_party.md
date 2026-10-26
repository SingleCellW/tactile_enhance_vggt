# Third-party sources and assets

The repository does not currently grant an explicit license for its original
code. Owner selection is tracked in
[issue #5](https://github.com/SingleCellW/tactile_enhance_vggt/issues/5).
Maintenance preserves the existing authorization state; public visibility alone
is not presented as an open-source license.

The source lock records URLs and full commit IDs for LeRobot and VGGT.
`external/` contains ignored local symlinks. Bootstrap verifies revision and a
clean worktree. Use separate pinned checkouts for changed dependencies; do not
switch a shared checkout underneath another running experiment.

LeRobot source, VGGT source and VGGT-1B weights retain their respective upstream
terms. A checksum establishes identity, not permission to redistribute a model.
Retain upstream copyright/license notices for any permitted derivative work.

Historical tactile code, robot URDF/mesh assets and demonstrations are not bundled
or automatically downloaded. Their distribution scope and independent retrieval
remain tracked in [issue #2](https://github.com/SingleCellW/tactile_enhance_vggt/issues/2).

Any license selected for this repository's original code does not relicense
dependencies, model weights, datasets or external assets.
