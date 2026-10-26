# Data, geometry and evaluation contracts

- Preserve source-episode identity. Sibling object perturbations and camera views
  remain in the same train/dev/test split. Repeated renderings are not independent
  demonstrations, and training seeds do not add independent test trajectories.
- Historical stored RGB row `r` depicts recorded state row `r-1`. Repaired caches
  explicitly map to the corresponding state, native tactile and source action
  labels. Drop the first unavailable observation within each episode; never cross
  an episode boundary. Do not reuse replay-generated force with corrected anchors.
- Reconstruct native tactile from the recorded full 29D simulator state. Keep
  `state22`, episode/frame indices, sensor order and source action conventions
  bound to the cache manifest. Normalization uses training data only.
- The 44 channels comprise 32 fingertip taxels and 12 FSR channels. Values are
  inherited continuous simulator proxies, not calibrated force in Newtons.
  Quaternion ordering, body-COM to origin velocity and FK/mesh correspondence are
  checked separately. Changing the force law requires a new data version.
- Policies receive 22 robot joint observations and predict 22 joint targets.
  Object pose is permitted for simulation synthesis, supervision and scoring,
  not as a policy observation. Action chunks never cross episodes.
- Static object counterfactuals support spatial supervision; they cannot simply
  inherit the original action labels. Camera-only rerendering can preserve action
  labels when the physical trajectory is unchanged.
- The simulator fixes MuJoCo at 3.4.0 and physics timestep at 0.01 seconds. Sensor
  velocity timing is explicit and independent; a 0.05-second historical proxy and
  the corrected 0.01-second proxy are different data contracts. Match checkpoint,
  cache and runtime settings instead of silently changing a default.
- A worker observation includes two `224×224` RGB images, state22, force44,
  anchors, actual camera matrices and the robot-base transform. Rendering does
  not advance physics. Preserve reset, skipped-observation and action-queue
  behavior when refactoring runtime code.

Compare methods with matched source split, observations, action protocol, seeds,
update/sample budget and validation checkpoint selection. Report parameter
allocation separately from active compute. Do not select on the test set.

For spatial experiments, score in a declared common coordinate system. Report
centre error alongside displacement response and the zero-displacement control;
cross-view consistency alone can reward a constant wrong prediction. Aggregate
paired results by source episode before uncertainty estimation.

For policy experiments, action L1 is an imitation diagnostic, not a task success
rate. Report lifting, holding, dropping, placing/releasing and strict stable-goal
success separately. Cached-point latency, fresh ACT chunk latency and complete
RGB→VGGT→ACT refresh latency are different measurements. The current synchronous
refresh adapter does not prove continuous real-time 20 Hz operation.

Detailed historical protocols and audits are preserved under
`.local/reports/historical_docs/`; the old point protocol is also available in
Git history at `f463b64`. New runs record their own resolved configuration and
input fingerprints. Historical result values are not copied into this contract.
