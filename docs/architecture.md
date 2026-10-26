# Implemented research paths

All paths use synchronized observations at one time step. They do not use future
images or whole-episode reconstruction as policy inputs.

| Path | Implementation | Role |
|---|---|---|
| Frozen VGGT points → point encoder → ACT | `geometry.py`, `policy.py`, `inference.py` | Early calibrated spatial anchoring baseline |
| Frozen VGGT prefix → tactile adapter around final global block → representation → ACT | `token_fusion.py`, `token_policy.py`, `runtime.py` | Internal token fusion, late fusion and modality controls |
| Visual projection + pressure/state → ACT | `simple_token_policy.py`, `runtime.py` | Simpler policy control, sharing the ACT implementation |
| Same physical state under changed stereo poses | `camera_pilot.py`, camera experiment commands | Spatial generalization and paired-displacement experiments |

The point path uses calibrated base-frame metres and sampled visual points.
The token spatial experiments express anchors and supervision in a compatible
first-camera frame with shared stereo-baseline normalization. These are distinct
input contracts; a refactor must not silently substitute one for the other.

The token adapter and readout train while the VGGT backbone remains frozen.
Spatial prediction of object-centre-relative-to-palm is an auxiliary task, not
full-surface reconstruction. Policy supervision and action queues use the ACT
implementation in the pinned LeRobot checkout. VGGT weights and extracted prefix
features are external inputs with source/hash records.

The simulator runs in a separate process/environment. `simulation.py` implements
the native tactile adapter and PickLift worker contract; it depends on historical
MuJoCo assets and readout code. A core import does not launch the simulator.

Runtime configuration and evidence are shared concerns: `paths.py` resolves
artifact/dependency/machine locations, and `provenance.py` records the executed
source state. Report generators consume run outputs, while portal source and
deployment tools remain separate from generated pages and bundles.

The repository is a research baseline. Existing negative results and unsupported
capabilities are listed in [research status](research_status.md); passing unit
tests does not establish real-robot performance or a tactile policy advantage.
