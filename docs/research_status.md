# Research status and open work

The recovered source includes the point-cloud baseline, Stage 1.a token fusion
and simpler policy controls, and Stage 1.b camera augmentation plus paired
displacement objectives. Repository maintenance does not rerun or reinterpret
those historical experiments.

Historical Stage 1.b evidence supports camera generalization improvements, but
does not establish reliable object-displacement response, a consistent internal
fusion advantage over late fusion, or new policy benefits. Contact-boundary data
and modality controls remain required before expanding policy training.

| Issue | Scope | Status / remaining acceptance |
|---|---|---|
| [#1](https://github.com/SingleCellW/tactile_enhance_vggt/issues/1) | Repository maintenance | Completed in merged [PR #7](https://github.com/SingleCellW/tactile_enhance_vggt/pull/7) |
| [#2](https://github.com/SingleCellW/tactile_enhance_vggt/issues/2) | Historical simulator/data/asset reproduction | Versioned retrieval, hashes, distribution scope and native-worker verification |
| [#3](https://github.com/SingleCellW/tactile_enhance_vggt/issues/3) | Contact-boundary and spatial-response evidence | Matched controls meeting declared spatial acceptance criteria |
| [#4](https://github.com/SingleCellW/tactile_enhance_vggt/issues/4) | Portal rollback service state | Restore the prior enabled/active state and verify failure paths |
| [#5](https://github.com/SingleCellW/tactile_enhance_vggt/issues/5) | Original-source license | Apache-2.0 selected; license text and consistent metadata delivered for review |
| [#6](https://github.com/SingleCellW/tactile_enhance_vggt/issues/6) | Online CI activation | Workflow permission and an inspected GitHub Actions run |

The existing [Stage 1.a](http://10.15.89.229:8766/reports/stage1a/) and
[Stage 1.b](http://10.15.89.229:8766/reports/stage1b/) reports require the campus
network. Complete local reports, media and raw metrics remain in the ignored
artifact area. A public clone does not include those outputs.

Future issues should carry source/run identifiers, reproducible evidence and
explicit acceptance criteria. A partial implementation or a passing CPU suite
must not close a scientific question or an unavailable asset requirement.
