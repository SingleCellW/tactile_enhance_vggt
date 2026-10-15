# 修复数据后的配对实验

先完成 raw qpos29 与因果时间步重建的 native tactile，并完成独立同步审查，再使用新的完整 cache 路径。缓存生成完成只说明文件写完，不证明 RGB、力和 anchor 的时序正确。不要将旧 replay force 移到新 FK 位置后称为同步数据，也不要复用旧缓存的 normalization。

渲染审计进一步确认，原始 `RGB[t]` 近像素级匹配 `qpos[t-1]`。修复保留视觉第 `t` 帧的 VGGT 点云，把 `obs_frame_indices` 映射到 `t-1`，使机器人状态、native force、anchor 和已有动作标签取自同一状态行；动作标签继续沿用历史转换器的定义，不再额外平移。每个 episode 的首个观测被丢弃，避免跨 episode 对齐。

`scripts/run_ablation_suite.py` 独立于原有 `run_pilot.sh`，不会自动准备缓存或启动旧路径的任务。`prepare` 强制要求 manifest 的 `historical_source_tactile_superseded=true`、`native_tactile_report.complete=true`，以及 `configuration.rgb_alignment.verified=true`、`rgb_lag_frames=1`，拒绝旧 replay tactile 或未完成一帧 RGB 对齐的缓存；它只生成三种模式的配置和计划，显式执行 `execute` 才训练。

```bash
.venv/bin/python scripts/run_ablation_suite.py prepare \
  --cache /absolute/path/to/repaired_native_cache \
  --output-root /absolute/path/to/new_ablation_suite \
  --seeds 43

# 初轮只执行两个模式；物理 GPU 1/2 各自只运行一个任务。
.venv/bin/python scripts/run_ablation_suite.py execute \
  --plan /absolute/path/to/new_ablation_suite/suite_plan.json \
  --gpus 1,2 --modes anchored,unanchored

# 有余量时，再执行已经配好的无触觉对照。
.venv/bin/python scripts/run_ablation_suite.py execute \
  --plan /absolute/path/to/new_ablation_suite/suite_plan.json \
  --gpus 1 --modes no_tactile
```

可通过 `prepare --alignment-report <audit.json>` 保存独立审查报告的哈希。脚本只记录来源，不解释报告为“同步认证”。配置、实现或缓存 manifest 在计划生成后变化时，执行会拒绝继续；使用新实验目录重新生成计划。原运行结果和日志不会被覆盖；中断续训用对应配置调用 `train_act.py --resume`。

## 对照含义和公平性

| 模式 | 共享的视觉几何 | 44 维全局触觉 | 触觉位置与局部视觉融合 |
|---|---|---|---|
| anchored | 同一份 VGGT 点云 | 有 | 有 |
| unanchored | 同一份 VGGT 点云 | 有 | 无 |
| no_tactile | 同一份 VGGT 点云 | 零输入 | 无 |

每个 seed 共用缓存、episode split、训练统计、观测顺序、初始化、batch、优化器、更新步数、动作 chunk、验证频率和精度。点编码器没有随机采样或 dropout，模式分支不会改变 ACT 的随机数调用顺序。CUDA 算子仍可能存在数值差异，因此相同 seed 不等于跨硬件逐位相同。

三种模式实例化的参数量相同，但实际激活的分支和计算量不同。不要把它们描述成严格相同有效容量或相同 FLOPs。`anchored` 同时提供已知传感器空间布局和局部关联，因此后续可增加“保留位置但打乱压力对应”的消融，区分布局先验和真实触觉对应。

**anchored/unanchored 只能回答：给定 VGGT 几何时，触觉空间锚定是否有帮助。它们不能证明 VGGT 优于 RGB 或真实几何。** 后续仍需同示范、同视角、同控制接口的 RGB + tactile／RGB-S 基线，以及相同点编码器使用真实几何的 oracle 对照，最终进行闭环成功率评估。

## 训练预算

默认继承参考 ACT：batch 64、AdamW 1e-5、chunk 16、执行 8、hidden 512、encoder 4、decoder 1、VAE encoder 4、KL 10。初轮预算为：

`160000 × 训练缓存观测数 / 20212 × 64 / batch_size`，取整到 1000 步。

这保持每个训练观测约 `160000 × 64 / 20212 ≈ 507` 次平均曝光；它并不保持总更新数或总数据量。一帧 RGB 对齐并丢弃每个 episode 首观测后，预计保留 5020 个 stride 4 观测：4505 个训练观测、515 个验证观测。预算仍取整为 36k 步，实际约 511 次训练观测曝光；最终由新缓存的实际观测数重新计算。三种模式必须使用同一预算，不能单独给较差模式追加训练后仍称为配对比较。

这个预算是起点，不是收敛证明。训练中检查 held-out action L1、训练/验证差距和触觉消融差异；action L1 不代表闭环任务成功率。多 seed 扩展应继续复用同一数据 split，仅改变训练初始化和批次顺序。

## 吞吐与部署边界

batch 64 时局部关联包含 `64 × 44 × 512 = 1,441,792` 个距离，FP32 三维差值临时张量约 16.5 MiB。局部邻域矩阵乘约 0.369G MAC，contact 融合 MLP 约 0.369G MAC，视觉点 MLP 约 1.36G MAC。它没有引入全点云二次 self-attention 或额外 CUDA 扩展；ACT 的 VAE 编码器是更主要的训练计算部分。整个默认配置前向粗估约 30G MAC，实际性能仍取决于 GPU、kernel、数据读取和同步开销。

A40 GPU 2 的一次 batch 64、anchored、200 步性能检查实测稳态约 **0.0358 s/step**，即约 **28 steps/s、1788 observations/s**，PyTorch 最大已分配显存约 **0.811 GiB**。该检查使用存在触觉错位的旧 5073 观测缓存，仅用于计算性能测量，不能作为模型效果实验。按此步时，36k 步纯训练计算约 21.5 分钟；正式修复缓存后的运行仍需重新测量，并计入验证和 checkpoint 保存后再报告 ETA。不要把离线 VGGT 缓存生成速度算作策略训练速度。

推理时分别报告新 chunk、缓存动作和 RGB→VGGT→ACT 的刷新延迟。20 Hz 控制、执行 8 步意味着名义上每 0.4 s 需要一段新动作；仅测平均刷新时间不够，还需验证刷新能否及时完成以及同步执行造成的停顿。当前 adapter 会在队列耗尽后同步请求几何，没有实现异步几何/动作预取，因此尚未证明持续 20 Hz 闭环控制。
