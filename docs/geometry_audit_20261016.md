本次审计支持保持 v2 的原有视觉点云，让 v2 三组实验只修复动作对齐与物理时间基准。仅用已知双目外参替换 VGGT 相机姿态没有改善本次几何误差；同时使用已知内参和外参则有改善，可作为后续 v3 的独立对照。本轮没有修改任何训练缓存、预处理或 encoder，也没有把 GT 深度送入 VGGT 或 policy。

审计使用 held-out 五个 episode 中的 16 帧，按当前 v1 原始触觉代理值任一传感器 >0.5 分为 8 帧接触、8 帧非接触，再在各层内按序列位置分散选取。这是一个分层诊断样本，**不是全数据误差的无偏估计**。所有变体使用相同的原始训练视频 RGB、同一个冻结 VGGT 的预测深度、同一双目基线尺度；没有用 GT 深度拟合尺度或模型参数。

下表是固定评分像素上的 **pixel-correspondence XYZ error**：预测点与该输入像素对应的 GT 可见表面点之间的欧氏距离，按像素数加权取均值，单位 cm。它不是 Chamfer，也不通过寻找邻近表面来降低误差。

| 几何处理 | 固定工作空间 | 方块可见表面 | 触觉锚点 10cm 邻域 |
|---|---:|---:|---:|
| 当前 Sim3：第一相机定姿态、双目基线定尺度 | 8.090 | 4.752 | 4.234 |
| 相同预测 depth/K/scale，换已知双目外参 | 8.314 | 5.341 | 4.675 |
| 相同预测 depth/scale，换已知 K 和双目外参 | 6.163 | 3.538 | 3.037 |

已知 K+外参变体在此样本的方块和触觉邻域误差分别下降约 25.5% 和 28.3%，但局部仍有约 3cm 的对应点误差。VGGT 预测焦距平均比已知相机大约 16–17%，说明仅修正外参不足以检验完整相机标定的价值。以上改善不等于 policy 成功率改善，后续训练和闭环仍需独立验证。

评分只使用 GT 可见工作空间内、3×3 同一 geom 内部的像素，并使用对所有变体相同的置信度筛选。方块分割和 GT 深度仅构造离线评分区域；10cm 邻域由已有 FK 触觉锚点定义，不限定为当时有压力的传感器。完整报告另外包含左右视角、逐帧、光轴深度绝对误差和相对误差。原视频与当前状态重新渲染的 RGB 平均绝对差为 2.40/255，保留了录制编码与重渲染差异这一限制。

GT 深度在 MuJoCo 3.4.0 中以实际 VGGT 分辨率 518×518 直接渲染，保持原相机垂直视场；原始 RGB 仍按历史 480×640 中央裁剪后缩放至 224×224，再由 VGGT 放大。深度采用光轴 z，内参使用整数像素中心的 `(size-1)/2` 约定。GT 深度独立关闭 MSAA，RGB 保留原场景采样；默认 MSAA 深度曾引入毫米级偏差，原结果仅保留在远端 `*_msaa.*` 诊断文件，不是最终结果。关闭深度 MSAA 后，16 帧独立 CPU `mj_ray` 检查的最大误差为 **0.034mm**；每帧 p95 必须小于 0.1mm 才允许评分。原始 qpos 文件、时间索引、渲染器源码、输入/预测档案均记录校验值。

另对完整 5020 帧缓存进行了 CPU 覆盖检查。第二相机标定残差中位数为 11.19cm、p95 为 13.74cm，旋转残差中位数为 8.25°；这些是相机残差，**不能当作表面点误差**。全部接触 sensor-observations 中，88.8% 在 5cm 内有采样视觉点，100% 在 10cm 内有点，局部 Gaussian pool 的有效邻居数中位数约 29.5。因此未观察到空间融合因邻域完全为空而普遍失效。采样点邻近距离和覆盖率也不是 GT 几何精度。

最终文件位于项目的 `artifacts/geometry_gt_audit_20261016_r2/`，本地和远端均已同步：

- `report.json`：三种几何的完整评分、逐帧结果、精度验证及输入校验值。
- `cpu_geometry_gate.json`：完整缓存的相机残差与邻域覆盖诊断。
- `render_spec.json`、`rendered_gt.json`：固定采样帧和独立 GT 渲染验证。
- `rgb_inputs.npz`、`predictions.npz`、`rendered_gt.npz`：可重复评分的实际输入、VGGT 输出和 GT；仅供审计，不是训练缓存。

在远端项目 `/home/wukf/projects/tactile_enhance_vggt` 中，以下命令只在 CPU 重复评分，不加载 VGGT、policy 或 native tactile：

```bash
.venv/bin/python scripts/audit_geometry.py \
  --cache cache/pickplace_vggt_native_sync_stride4 \
  --output artifacts/geometry_gt_audit_20261016_r2 \
  --score-existing --split val --rgb-source recorded
```

完整重跑需要使用一个新输出目录，并设置 GPU2 的 CUDA 与 EGL 设备：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
  .venv/bin/python scripts/audit_geometry.py \
  --cache cache/pickplace_vggt_native_sync_stride4 \
  --output artifacts/geometry_gt_audit_reproduction \
  --frames 16 --split val --rgb-source recorded --device cuda:0
```

四个数值测试通过，覆盖已知相机零误差、第一相机 Sim3 不变量、评分掩码，以及只替换 K/外参而保留预测深度。最终冻结 SHA256：

```text
scripts/audit_geometry.py
d609cdfee4b14e3ebbfda21fb63529c9122c3a229b86faf9bedf7628f96a2bf5
tests/test_audit_geometry.py
2b86444d7ca7fcd9b18c81cb044b2a8fe6ff2d2cd2a6afc10f58cfd5a572617b
report.json
aedbc0a0088e58e2c78f4eb1b1ac962b36480414aaa82b9b9ab17a3285b06b5f
```

`--score-existing` 会重新写报告，并改变耗时字段及报告文件 SHA256；脚本校验值、输入档案校验值与数值结果应保持一致。
