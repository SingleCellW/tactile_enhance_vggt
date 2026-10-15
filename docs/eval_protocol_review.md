# Simulation evaluation and training review

Read-only review of the frozen evaluation sources and completed seed-43 runs, 2026-10-16. No GPU jobs were started by this review.

## Important correction to the earlier timing assumption

The dataset's 20 Hz video/wall-clock cadence does **not** mean that each recorded command advances 0.05 simulated seconds. The frozen model compiles to Euler, timestep 0.01 s; the historical evaluator advances one physics step per command.

The CPU transition audit in `physics_transition_audit.json` compares 172 sampled transitions across episodes 0, 1, 26, and 52. It initializes recorded full qpos[i], reconstructs causal velocity with candidate dt, applies candidate command[i] or command[i+1], and advances one or five physics steps. All four episodes favor dt=0.01 s, one step, command[i]. Mean arm errors are 4.1–6.0 microradians; mean robot errors including the hand are 0.00017–0.00035 rad. This is strong evidence for the timing/command mapping, not exact reconstruction of hidden solver state.

This corrects two earlier assumptions: five substeps were not justified by video FPS, and the converter's label command[i+1] is ahead of the command that produces the demonstrated next physical state. It also implies that the first native-tactile cache's wall-clock velocity denominator is a sensor proxy rather than physical simulated velocity. These limitations apply equally to both completed pilot models.

`action_source_audit.json` separately checks all 20,212×22 historical labels: each is exactly exported command[min(i+1,last)] after the historical float32 cast. The full original action directory available for episode 1 also matches its 449×22 exported commands exactly. The generator appends `action_list[i]` unchanged at `dataset_generation.py:433` and exports it at line 437. Other episodes use that generator export; the report does not claim original action files were independently available for all episodes.

## Frozen evaluator evidence and interfaces

Remote campaign root: `/home/wukf/projects/tactile_enhance/rgbs_dynamic44_20260921`.

- `eval_dynamic44.py` selects `eval/sim/imitation_learning/evaluation/sim/base/lerobot_eval_policy_rc_260408_universal.py` for PickLift, normally 121 trials and 600 commands. It freezes MuJoCo 3.4.0, scene assets, the 7 cm object mesh, and continuous44 geometry. The old wrapper's native tactile adapter is not interchangeable with the newly corrected adapter.
- `lerobot_eval_policy_rc_260408_universal.py:955–966` resets keyframe 0, applies a cube XY offset, forwards state, and resets the policy's action queue. At lines 1095–1096 it writes an absolute 22D actuator target and calls `mj_step` once. Therefore the old 600-command horizon is 6 simulated seconds.
- `lerobot_eval_policy_rc_260317.py:54–57` defines an 11×11 grid: each axis has offsets -0.08…0.08 m in increments of 0.016 m. The order is X-major. The first N trials form an edge strip, not a representative random subset.
- `lerobot_eval_policy_rc_260317.py:557–611` finds `cube_freejoint`, adds offsets to the current reset pose, preserves Z and quaternion, and zeroes cube velocity. It catches reset errors instead of failing; new code should fail explicitly if reset cannot be verified.
- `lerobot_eval_policy_rc_260317.py:614–634` calls a trial successful only from final cube center: X in [0.40,0.60], Y in [-0.39,-0.21], Z≤0.10 m. It does not require a lift, release, low speed, full cube containment, or persistence.
- `eval/assets/picklift/scene.xml` visibly says RK4/0.002, but its included `xarm_with_leaphand.xml:4` overrides it to Euler/0.01. Verify compiled settings, not just the outer XML. The latter also defines the 22 position-target actuators and their ranges.

## Recommended paired rollout protocol

1. Start with the five existing validation episodes [1,24,35,43,52], initialized from their recorded complete qpos29 and zero velocity at frame 0. Use identical resets, seed, point-sampling rules, action execution count, horizon, clipping, and observation preprocessing for both methods. These are development rollouts, not an untouched independent test set, because the same episodes select checkpoints offline.
2. Their raw episode names are `0211_204543`, `0211_211128`, `0211_212940`, `0211_214023`, and `0211_215840`. Initial XY are respectively (0.44,-0.04), (0.50,-0.02), (0.52,0), (0.54,0.02), and (0.56,0.06). Initial cube Z≈0.03489 m, quaternion≈identity. Recorded robot state is already settled (arm joint 3≈-0.708, versus keyframe -0.7854), so replacing it with home introduces an avoidable distribution change.
3. Advance **one 0.01 s physics step per predicted command**, 600 commands maximum. Reset ACT queues per trial; execute the saved configuration's first 8 actions from each 16-action prediction and refresh VGGT/observations when the queue empties. Wall-clock inference duration must not advance physics implicitly. Report simulated seconds and measured wall-clock throughput separately.
4. Render current, freshly forwarded physics data and compute current state/tactile/anchors from the same geometry. Do not reproduce the old raw RGB delay at inference: training cache repair already pairs RGB[r] with physical state[r−1]. Preserve the converter's 640×480→center-crop 480 square→Pillow LANCZOS 224 preprocessing before VGGT resize to 518.
5. For already-trained v1 models, explicitly label sensor_dt≈0.05 as a legacy wall-time proxy, while physical substeps remain 1. For v2, regenerate native tactile with causal qpos difference divided by 0.01 s, repair targets to command[s], recompute train-only statistics, and use sensor_dt=0.01 online. Never change sensor convention silently between training and evaluation.
6. Keep the legacy final-center predicate as a clearly named compatibility metric, evaluated after `mj_forward`. Also report final cube XYZ, maximum height, first entry time, contact/release state, final speed, action clipping fraction, and whether the placed/released/low-speed condition persists for one physical second. A stricter release/stability metric must have its thresholds fixed before comparing methods and be reported separately from the legacy rate. The raw demonstrations satisfy the weak final-center rule in 52/53 episodes; this alone does not establish successful lifting/release.
7. Once the loop passes, use a predeclared paired grid or seeded list covering the demonstrated XY range (X 0.44–0.56, Y ±0.06). The old ±0.08 grid extends beyond that range and should be identified as a broader generalization test. Do not choose checkpoints or the best variant from this final rollout set. Report paired per-start outcomes and uncertainty; five trials or one training seed cannot establish a small advantage.

Privileged object state is appropriate for simulator mechanics and success metrics; keep it out of policy inputs. Policies should receive only synchronized RGB-derived VGGT points, state22, and the specified tactile features. Hash checkpoints, normalization, source cache manifest, scene, native tactile adapter, and rollout protocol in results.

## Completed checkpoint selection and generalization diagnosis

Both completed runs have 36,000 updates, batch 64, and the same 4,505 training / 515 validation observations. Validation episodes are [1,24,35,43,52]. The configured 16 validation batches cover all 515 observations at batch 64.

| Model | Earliest/best logged validation | Final validation normalized action L1 |
|---|---:|---:|
| Anchored | step 2000: 0.3921167293 | 0.4140279418 |
| Unanchored | step 2000: 0.3916947267 | 0.4158890799 |

The actual saved checkpoints are 100, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, and 36000. Step 2000 was not saved. `metrics.jsonl` carries the last validation result into intermediate status rows: the value logged at step 5000 is a measurement of step 4000, not of checkpoint 5000.

Re-evaluate every saved checkpoint using deterministic deployment inference on the full validation set. Select each method's best **available** checkpoint by the same predeclared metric, and also report the matched 36k endpoint. Do not infer checkpoint scores from carried-forward status fields, and do not select by test-rollout outcomes. Preserve all selected checkpoint hashes.

Training L1 uses the CVAE posterior conditioned on action labels; validation uses the deployment latent. Their raw gap is not a clean train-versus-validation comparison. Diagnose with the same `predict_action_chunk` path on fixed train and validation subsets, separately reporting first-action, executed-prefix-8, full-chunk-16, physical-unit and normalized error, per episode and contact/noncontact slices. Worsening validation after early training suggests an early-stopping opportunity, but the timing/label repairs should precede conclusions about VGGT effectiveness.

The final 0.00186 normalized-L1 difference is a single-seed pilot result, not evidence that anchoring improves closed-loop success. Anchored versus unanchored also tests tactile spatial association with VGGT present in both methods; it does not by itself establish VGGT's value relative to RGB-only or no-VGGT alternatives.

## Action repair utility

`scripts/repair_action_labels.py` requires a completed cache, the source command root, `audit/source_picklift.json`, and the verified `physics_transition_audit.json`. It verifies the full old next-frame relation, creates a separate cache with action[s]=command[s], recomputes only train-episode action normalization, and verifies observation arrays remain byte-identical. It rejects partial/mismatched controls, episode-map problems, unsupported timing evidence, repeat application, and existing destinations. Native tactile timing is inherited from its input and must be repaired separately first.

Eight synthetic tests in `tests/test_action_repair.py` pass, including complete 22D label alignment, raw/export identity, train-only statistics, unchanged visual/tactile/alignment arrays, and rejection paths.

## Independent action-spacing audit and v2 short-run plan

Observation stride and action spacing were checked independently in the implemented dataset, actual remote arrays, the ACT action queue, and the historical LeRobot loader. The stride-4 cache does **not** downsample action chunks.

- `src/tactile_vggt/data.py::CachedTactileDataset.__getitem__` maps one cached point observation to its full-rate source frame using `obs_frame_indices`. It then calls `action_chunk` on the full action array. `action_chunk` uses `arange(start,start+horizon)`, not a stride-four range, and repeat-pads only after the same episode ends.
- Real cache shapes are state/action `[20212,22]` versus points `[5020,556,8]`. The first twelve source observation indices are `[3,7,11,15,19,23,27,31,35,39,43,47]`. Actual sampled chunks at starts 3 and 7 equal source actions `[3..18]` and `[7..22]` exactly. A chunk at source row 20209 uses 20209,20210,20211 followed by masked repeats of 20211; it does not cross episodes.
- The historical `ActShallowConfig` inherits `ACTConfig.action_delta_indices=range(chunk_size)`. Its dataset factory divides these indices by dataset FPS, and `get_delta_indices` multiplies by the same FPS and rounds. Consequently a nominal delta of 1/20 s maps back to **one source row**, not five physics steps or four cached observation rows. The old query function clamps those adjacent rows to episode boundaries. The current project does not use that timestamp loader for cached action targets at all.
- `ACTPolicy.select_action` fills its queue with the first `n_action_steps=8` predictions and pops one on each call. It does not interpolate, multiply time indices, or repeat each action four times. With the corrected v2 labels, a prediction from state row s therefore produces commands s through s+15, and the first eight commands should each advance one 0.01 s physics tick. A full predicted horizon spans 0.16 simulated seconds; refreshing after eight executed commands spans 0.08 simulated seconds. Training observation starts are separated by 0.04 simulated seconds. These are different quantities.

For the next diagnostic stage, use all three modes (`anchored`, `unanchored`, `no_tactile`) for **6,000 updates each**, seed 43, batch 64, LR 1e-5, weight decay 1e-4, BF16, chunk 16, execution prefix 8, unchanged stride-4 observation selection and episode split. Set `eval_every=save_every=500`, `eval_batches=16`, and use the complete 515-observation validation set. This provides about 85.24 chunk exposures per training observation, compared with 511.43 at 36k; describe it as a short diagnostic budget rather than the previous reference budget. Physical timestep corrections do not require rescaling the optimizer learning rate.

Matching save and evaluation intervals avoids losing an early validation optimum. With the existing extra step-100 checkpoint, 500-step saving produces 13 checkpoints per run; at the observed approximately 474 MiB per complete checkpoint, three runs occupy roughly 18 GiB. No scheduler is currently used; keeping constant LR avoids adding an optimizer change while diagnosing the data correction.

Before launch, require both `configuration.action_label_repair.action_target_offset_frames==0` and `native_tactile_report.velocity_dt==0.01`, alongside the existing verified RGB-lag and complete-cache guards. The existing ablation launcher only checks native completion and RGB alignment, so these v2 checks must be made explicitly to avoid accidentally reusing the v1 cache. All three jobs must use the same manifest hash, normalization, initialization seed, sampler order and update budget. If the experiment is later extended for a matched-budget claim, extend all compared modes to the same budget and select checkpoints by the same full-validation criterion; do not choose by final closed-loop outcomes.

The practical v2 preflight is to inspect several real normalized/denormalized chunks against recorded command[s:s+16], including a moving/contact segment and an episode end; assert `action_step_ticks=1`, `observation_stride_ticks=4`, and simulator substeps=1 in protocol metadata. The action repair's all-frame source checks and existing boundary tests already cover the underlying indexing invariant. No new GPU or training jobs were submitted for this review.
