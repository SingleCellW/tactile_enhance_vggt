# Simulation worker contract

Run `scripts/sim_worker.py` using the frozen native environment
`/home/wukf/miniforge3/envs/te/bin/python`. The policy and VGGT remain in the
new project's `.venv`; only JSON-lines and an atomic NPZ file cross the process
boundary. Native library logs go to stderr. The worker stays alive across
episode resets and sends one JSON response for every request.

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  /home/wukf/miniforge3/envs/te/bin/python -u scripts/sim_worker.py \
  --ipc-dir artifacts/eval_ipc \
  --dynamic-root /home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921 \
  --raw-root /home/wukf/projects/tactile_enhance/rgbs_corrected44_20260906/raw_numeric/Mujoco_Picklift_random_cube_260212 \
  --timestamps-root inputs/raw_timestamps \
  --source-readouts /home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921/readouts/picklift.npz \
  --device cuda:0 --substeps 1 --sensor-dt .05
```

`MUJOCO_EGL_DEVICE_ID` selects the physical EGL device; `cuda:0` selects the first
device exposed through `CUDA_VISIBLE_DEVICES`. The caller controls these values.

The primary requests are:

```json
{"command":"reset","episode_id":1,"frame":0,"seed":43}
{"command":"step","action":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"observe":false}
{"command":"observe"}
{"command":"metadata"}
{"command":"close"}
```

The zero action above illustrates the schema; it is not a recommended control
command. Actual actions contain 22 absolute actuator position targets. Limited
actuators are clipped to their scene `ctrlrange`; counts and dimensions are
reported. `reset` uses complete raw qpos29, zero velocity at frame zero and
`ctrl=qpos[:22]`, with no extra settling step. Optional `object_xy_offset` is an
explicit initialization intervention. Nonzero reset frames are supported for
fixture tests, using only that frame and its predecessor.

Successful responses contain `ok`, `command`, `info`, `control_tick`, `sim_time`
and elapsed `worker_seconds`. Reset also includes native runtime provenance.
Requested observations add `observation_path` and numerical consistency checks.
Read and copy `response.npz` before sending the next command. Its arrays are:

| Key | Shape | Meaning |
| --- | --- | --- |
| `rgb` | 2 × 224 × 224 × 3 | uint8, left then right |
| `state`, `state22` | 22 | Robot joint positions; aliases |
| `tactile`, `force44` | 44 | Continuous native proxy; aliases |
| `anchors_world` | 44 × 3 | Current FK anchors |
| `known_cameras_cv` | 2 × 4 × 4 | OpenCV camera-to-world |
| `base_to_world` | 4 × 4 | Current fixed robot-base transform |

Object pose appears only in evaluation `info`, never in policy observation
arrays. Current RGB is rendered after `mj_forward`; it therefore has no
historical recording lag. Rendering matches the source converter: 640 × 480
RGB, a centered 480 × 480 crop, then PIL LANCZOS resize to 224 × 224.

Physics and tactile velocity time are explicit and separate. The audited raw
state transitions match one `.01`-second physics step per action. The first
trained data version used wall-time differences around `.05` seconds for its
tactile velocity proxy; use `--sensor-dt .05` to approximate that version's
input distribution. The corrected physical-cadence data uses
`--sensor-dt .01`. Both modes compute sensor velocity from the immediately
preceding control tick, including ticks where observation/rendering was skipped.
They never overwrite the physical simulator's instantaneous qvel.

Legacy success is the cube center inside x `[.4,.6]`, y `[-.39,-.21]`, z `<=.1`
at the caller's final horizon. The worker never ends an episode early on this
condition. Supplemental strict success requires the cube to stay inside, with
linear speed below `.05 m/s`, angular speed below `.5 rad/s`, and no robot-cube
contacts for one physical second. This requires 100 ticks at one substep, or
20 ticks at five substeps. Lift and first-goal time are separate diagnostics.

The native force law retains the historical `cross(r, omega)` convention and
is not a calibrated physical force. WXYZ/XYZW object conversion, COM-to-origin
velocity conversion, full-frame FK and mesh checks match the data repair.

Verified smoke results are under `artifacts/sim_worker_smoke` on the server:
the native force and anchors at episode 1/frame 125 exactly matched the v1
repaired cache; five control ticks advanced `.05` physical seconds with one
substep; a separate five-substep check exactly matched manual MuJoCo stepping.
Repeated observation did not alter physics state, and skipped observations did
not change the causal sensor velocity interval.
