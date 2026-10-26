# Third-party sources and assets

The owner selected **Apache-2.0** for this project's original code on 2026-10-26.
The standard license is in [LICENSE](../LICENSE), with attribution in
[NOTICE](../NOTICE). Package and citation metadata use the same SPDX identifier.
The decision and delivery are tracked in
[issue #5](https://github.com/SingleCellW/tactile_enhance_vggt/issues/5).

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

The Apache-2.0 license for this repository's original code does not relicense
dependencies, model weights, datasets or external assets.
