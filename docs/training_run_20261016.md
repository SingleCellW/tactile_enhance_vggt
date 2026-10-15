# ACT 正式训练启动记录（2026-10-16）

本记录是北京时间 02:43:40 的启动快照，任务在后台继续运行。预计完成时间会随服务器负载变化。

## 已提交的任务

服务器：`rimlab-server`。项目：`/home/wukf/projects/tactile_enhance_vggt`。
监督进程 PID：4051306。源码分支：`codex/frozen-vggt-act-pilot`；训练代码快照提交：`21bc091`。

| 模式 | A40 物理卡 | PID | 已完成步数 / 总步数 | 稳态秒/步 | 预计完成（北京时间） |
|---|---|---|---|---|---|
| anchored：触觉位置与局部视觉融合 | 1 | 4051323 | 2500 / 36000 | 0.0351 | 2026-10-16 03:03:25 |
| unanchored：保留全局触觉，不使用触觉位置 | 2 | 4051324 | 2700 / 36000 | 0.0330 | 2026-10-16 03:02:07 |

时间估算包含实测训练、验证和 checkpoint 保存开销；建议预留到 **03:10**。
两组 loss 从约 67 降至约 0.4，梯度和预测均为有限值。第 100、1000 步 checkpoint 已完整保存，第 2000 步 held-out 验证已通过，无 failure.json。
首次验证的归一化动作 L1：anchored 0.3921；unanchored 0.3917。这只是早期训练诊断，不能说明任务成功率或锚定收益。

## 配置与数据

- VGGT 全部冻结，离线生成双目视觉几何；训练空间融合编码器与当前 LeRobot ACT，共 40,764,182 个可训练参数。
- batch 64，AdamW 1e-5，weight decay 1e-4，clip 10，chunk 16，执行 8 步，KL 10，seed 43；BF16、TF32、fused AdamW。
- 参考 160k-step / 20,212-frame 配置，按实际训练观测数缩放至两组各 36k steps。
- 53 条历史轨迹（含失败轨迹）、20,212 帧完整动作标签；几何观测 stride 4。
- 原始 RGB 滞后一帧：RGB[r] → state/force/action-label[r-1]。每条轨迹首个观测丢弃，最终 5,020 个观测：训练 4,505、验证 515，按 episode 划分。
- 触觉在原始完整 qpos29 上重新计算；对象姿态只用于模拟传感器生成，不进入策略。保留原生力律代理，数值未经物理力标定。
- 修复缓存：`cache/pickplace_vggt_native_sync_stride4`。
- 缓存 manifest SHA256：`3d02b3dbd75132e3a9372b40367fbb5f4a079d882f50e5528c7e8eb3cd4c22fb`。
- 原生触觉 NPZ SHA256：`7135035e388d97f3227962f31456f8a2cbd989cf688ef6ce5c66ab4f0fe03d47`。
- 相关验证：36 tests、19 subtests 通过；另有全量 FK、对象姿态、速度 Jacobian、数据有限值与时序抽查。

## 运行文件

以下均相对远端项目目录：

- `logs/pilot_native_sync_supervisor.log`
- `logs/act_vggt_native_sync_s43_anchored.log`
- `logs/act_vggt_native_sync_s43_unanchored.log`
- `outputs/act_vggt_native_sync_s43_anchored/status.json`
- `outputs/act_vggt_native_sync_s43_unanchored/status.json`
- 各输出目录的 `latest/` 指向完整 checkpoint；每 5000 步以及最终保存。
- `artifacts/native_sync_geometry_check/` 保存几何 QA 图及统计。

较早的 `integration_*`、`batch64_performance_check` 和旧 `act_vggt_s43` 缓存监督过程属于兼容性/性能检查，不属于本正式对照实验。

## 推理与后续评估

已整理 checkpoint 加载、归一化、动作队列、reset 和端到端测速接口。动作队列避免每个控制步重复执行 VGGT。
当前几何刷新仍同步执行；先前并发负载下 VGGT 刷新约 0.5–0.6 秒，不能据此宣称可持续 20 Hz 控制。后续需要控制循环中的异步刷新或频率策略验证。

本轮比较“给定 VGGT 几何，触觉空间锚定是否有帮助”。证明 VGGT 相对原始 RGB 策略的价值，还需 RGB+tactile 基线、真值几何对照和闭环任务成功率。当前尚未完成闭环 eval。

