# 2026-10-16：闭环评估与物理时序修正

所有时间为 Asia/Shanghai。本次使用 rimlab-server 的 A40 1、2 号卡；VGGT 保持冻结，训练只更新点云/触觉编码器和 ACT。

## 评估口径

- 5 个 validation episode 初态 `[1,24,35,43,52]`，相同完整 qpos29、相同随机种子 43，逐场景配对。
- 每次动作推进一次 0.01 秒 MuJoCo 物理步，共 600 次动作、6 秒模拟时间。
- ACT chunk=16，每次执行前 8 个动作后刷新观测。训练观测 stride4 不改变动作序列的逐帧间隔。
- 当前双目 RGB 经训练时相同的裁剪/缩放进入冻结 VGGT；关节状态、44 路触觉、FK 触觉位置与 RGB 同步。物体真值只用于触觉仿真和评分。
- `final_goal` 为旧项目的最终物体中心判据；`strict_stable_goal` 另要求连续 1 秒处于目标区、线速度 <0.05 m/s、角速度 <0.5 rad/s、无手/物接触。二者均在固定 horizon 结束时评分。
- “曾抬起”要求物体中心比初始高度至少升高 5cm。
- 模拟时间在推理期间暂停。这是控制有效性评估，不是实时控制频率测试。
- 这些初态来自用于 checkpoint 选择的 validation episodes，因此属于开发集闭环评估，不能视作独立测试集。

## 原 v1 模型结果

全量 515 个验证观测上重评所有实际保存的 checkpoint，各组最低归一化动作 L1 都出现在 step 1000。与此同时保留 step 36000 的相同训练预算对照。

| 模式 / checkpoint | 全 chunk 验证 L1 | 最终进入目标区 | 稳定放置并释放 | 曾抬起 |
| --- | ---: | ---: | ---: | ---: |
| anchored / 1000 | 0.382814 | 2/5 | 1/5 | 3/5 |
| unanchored / 1000 | 0.382064 | 1/5 | 1/5 | 2/5 |
| anchored / 36000 | 0.414028 | 4/5 | 4/5 | 5/5 |
| unanchored / 36000 | 0.415889 | 1/5 | 1/5 | 1/5 |

离线 L1 的排序与任务成功率不一致，不能把验证 L1 后期上升直接解释成闭环策略变差。36k 对照出现空间锚定的正向信号，但这里只有一个 seed 和 5 个场景，且 v1 存在以下时序限制。

数值摘要：`docs/eval_v1_results.json`。完整报告、20 段视频、逐步轨迹和模型/数据哈希在远端 `outputs/eval_v1_paired/`；报告和视频也已同步到本地 `artifacts/eval_v1_paired/`。

## 本次发现并修复的时序问题

1. 源视频/CSV 的约 20 Hz 是墙钟采样频率；相邻记录状态实际推进一条命令、0.01 秒模拟时间。4 个 episode 的 172 个物理转移支持这一结论。
2. 历史转换器的标签 `action[s]` 实际等于原始 `command[s+1]`。全部 20,212×22 标签逐元素核验一致；新缓存改成推进 `qpos[s]→qpos[s+1]` 的 `command[s]`。
3. v1 原生触觉用墙钟约 0.05 秒重建速度。v2 重建全部 20,212 帧触觉，改用物理 0.01 秒，并独立保存原始 wall timestamps。v1 闭环仍使用约 0.05 秒触觉代理以匹配其训练输入，不能和 v2 混用。
4. 原来已确认的 RGB 一帧滞后继续单独修正：RGB[r] 对应状态/触觉[r−1]。v2 从原始几何缓存创建新目录，只应用一次此映射。

新缓存：`cache/pickplace_vggt_physics_aligned_v2_stride4`。包含 53 episodes、20,212 条完整动作、5,020 个观测起点，train/val 为 4,505/515；相机点云、划分和模型结构保持不变。

新增工具包括 `evaluate_offline.py`、`evaluate_sim.py`、`sim_worker.py`、`repair_action_labels.py` 和 `audit_geometry.py`。保留旧缓存和原模型。仿真 worker 使用独立原生触觉 Python 环境，避免与新版 LeRobot 的依赖冲突。

## v2 下一步实验

三组 `anchored` / `unanchored` / `no_tactile` 使用完全相同的数据、初始化、batch 顺序、优化器和 6000 步预算。保留 seed43、batch64、LR1e-5、BF16、chunk16/execute8，每 500 步完整验证并保存。此次是短程诊断，不与 v1 的 36k 模型宣称等预算优劣；同时评估验证集选出的 checkpoint 和统一 6000 步终点。

训练目录：`outputs/v2_physics_aligned_s43_r2/`。首次启动暴露 launcher 对 Python 符号链接调用 `resolve()` 导致脱离虚拟环境，已改为保留 `.venv/bin/python` 入口并补回归测试；失败日志保留在不带 `_r2` 的目录，未产生有效优化步骤。

三组短程训练已完成。6000 步终点的完整配对结果为：

| 模式 | 最终进入目标区 | 稳定放置并释放 | 曾抬起 |
| --- | ---: | ---: | ---: |
| anchored | 0/5 | 0/5 | 2/5 |
| unanchored | 1/5 | 1/5 | 1/5 |
| no_tactile | 0/5 | 0/5 | 2/5 |

验证集最佳 checkpoint 三组都为 step1000，稳定放置分别为 anchored 0/5、unanchored 1/5、no_tactile 1/5。短程共 30 次、加上旧版 20 次，共完成 50 次闭环测试。完整数值摘要见 `docs/eval_v2_short_results.json`；原始报告和视频保存在本地 `artifacts/eval_v2_*` 与远端 `outputs/eval_v2_*`。

短程修正数据实验尚未复现旧版 36k 的锚定收益。这里既改变了数据时序、又缩短了相对 v1 的训练预算，不能将二者混为一个因果结论；三组 v2 内部仍是相同预算比较。

## 已执行的完整预算续训与自动评估

三个模式均从各自完整的 6000 步 checkpoint 恢复模型、优化器、随机数状态和 batch sampler，继续到总计 36000 步。早期 checkpoint 通过只读使用的符号链接保留，未重写原实验。继续保持同一数据划分、seed43、batch64、LR1e-5、BF16 和模型结构；后半程每 1000 步验证、每 2000 步保存，以控制磁盘用量。

- 配置：`configs/act_physics_aligned_v2_full.json`。
- 远端训练根目录：`/home/wukf/projects/tactile_enhance_vggt/outputs/v2_physics_aligned_full36k_s43/`。
- 三个训练 PID：anchored=4093377、unanchored=4093378、no_tactile=4093379；物理 GPU 分别为 1、2、1。
- 续训已实际超过 11000 步，损失有限，峰值训练显存约 0.8GiB/进程；不是只写了启动计划。
- 一次性评估依赖任务 PID=4095204，输出目录 `outputs/eval_v2_full36k_auto/`。`status.json` 已验证处于 `waiting`，绑定上述三个训练任务。

交付前最后检查为 **2026-10-16 04:11 北京时间**：三组分别已到 18800 / 19500 / 18700 步，均正常运行。结合实测训练、验证和存档开销，训练预计 **04:21–04:25** 完成；后续完整离线评分和双卡闭环评估预计 **04:35–04:40** 结束。以上是共享 GPU 条件下的估计，最新状态以远端文件为准。

`scripts/evaluate_when_ready.py` 在三个训练状态都完成 36000 步后，先重评所有早期及后期 saved checkpoints，再将 validation-best / 36000-step endpoint 两组闭环评估分配给 A40 1、2。相同 checkpoint 只执行一次，`selection.json` 保存两种选择角色的映射。脚本固定数据 manifest 与评估实现哈希，并在训练失败、进程死亡、超时或评估错误时记录失败；它不会停止训练进程。这是当前实验的一次性任务依赖，不是周期调度。

截至本次提交，完整 36k 的新结果仍在计算，不能把 v1 的 4/5 当作修正后的成绩。完整新结果将写入自动评估目录中的 `offline.json`、`selection.json` 及 `best_saved/`、`final_equal_budget/`。

## 验证

最终测试集通过 **61 tests + 27 subtests**。另外在真实 MuJoCo/native tactile 环境验证了单步/多步因果推进、重复观测不改变物理速度、独立的物理与传感器速度基准，以及 v2 接触帧的在线 force44/anchors 与缓存逐元素一致。闭环视频已核验真实抓取和释放行为。

## 几何诊断的解释边界

独立深度审计使用 16 个分层抽样帧（8 接触、8 非接触），GT 深度只用于评分，不进入模型或策略缓存。比较同 RGB、同 VGGT 预测深度和同尺度下的三种相机处理；不能把几何误差改善直接当作策略成功率提升。

本轮三组策略均继续使用原 Sim3 几何，避免同时修改时序和几何处理。已知相机内参/外参的几何变体作为后续独立对照，具体审计见 `docs/geometry_audit_20261016.md`。

空间锚定与未锚定、无触觉的比较都仍然使用 VGGT。若要回答“VGGT 本身是否胜过不用 VGGT”，还需要同数据、同 ACT 的 RGB+tactile 基线，以及更多独立初态和训练种子。
