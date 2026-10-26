#!/usr/bin/env python3
"""Publish the audited camera experiment as a short report in the existing portal.

Numbers come from the independently rebuilt summary. Interpretive sentences
come from a separately reviewed narrative.json, never generated from missing
results. This script writes local static files; portal_host handles deployment.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from tactile_vggt.paths import artifact_path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODES = ("visual", "tactile", "late", "internal")
LABELS = {"visual": "RGB + 关节", "tactile": "触觉 + 关节", "late": "Late 融合", "internal": "Internal 融合"}
DOMAINS = {"nominal": "原视角", "near": "未见轻扰动", "strong": "较强扰动"}


def read(path):
    return json.loads(Path(path).read_text())


def fmt(number, digits=2):
    return "—" if number is None else f"{number:.{digits}f}"


def summary_cell(value):
    return fmt(value["mean"]) + " ± " + fmt(value["sample_sd"])


def table(headers, rows, caption=""):
    return '<div class="table-scroll"><table class="b-table"><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>' + (f'<p class="table-note">{escape(caption)}</p>' if caption else '')


def metric_table(data, metric, domains=("nominal", "near", "strong")):
    rows = []
    for mode in MODES:
        for regime in ("fixed", "augmented"):
            rows.append([LABELS[mode], "固定训练" if regime == "fixed" else "相机增强"] +
                        [summary_cell(data["descriptive"][regime][mode][domain][metric]) for domain in domains])
    return table(["模型", "训练数据"] + [DOMAINS[v] + " ↓" for v in domains], rows,
                 "单位 mm；三个训练 seed 的均值 ± sample SD。每个 seed 先按原始 episode 等权汇总。")


def interval_cell(value):
    ci = value["conditional_episode_ci95"]
    return f"{fmt(value['mean'])} [{fmt(ci[0])}, {fmt(ci[1])}]"


def section(key, eyebrow, title, body):
    return f'<section id="{key}" class="report-section"><div class="section-heading"><div><div class="eyebrow">{eyebrow}</div><h2>{title}</h2></div></div>{body}</section>'


def recorded_run_path(path):
    """Resolve the former project/outputs layout without rewriting evidence."""
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    if path.is_absolute():
        try:
            index = parts.index("tactile_enhance_vggt")
        except ValueError:
            return path
        parts = parts[index + 1:]
    if parts and parts[0] in ("outputs", "runs"):
        return artifact_path("runs", *parts[1:])
    return path


def prediction_archive(main_data, pair_data, destination):
    manifest = {"scope": "Per-observation predictions and run records; model weights and RGB are not included", "files": {}}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for phase, data in (("main", main_data), ("paired", pair_data)):
            if data is None:
                continue
            for run in data["runs"]:
                prefix = f"{phase}/s{run['seed']}/{run.get('objective', run.get('regime'))}/{run['mode']}"
                for name in ("config.json", "completion.json", "history.json", "metrics.json", "test_predictions.npz"):
                    recorded = Path(run["directory"]) / name
                    source = recorded_run_path(recorded)
                    payload = source.read_bytes()
                    expected = data.get("sources", {}).get(str(recorded))
                    if expected and (expected["sha256"] != hashlib.sha256(payload).hexdigest() or
                                     expected["bytes"] != len(payload)):
                        raise ValueError("Recorded source size/SHA mismatch: " + str(recorded))
                    key = prefix + "/" + name
                    manifest["files"][key] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                    archive.writestr(key, payload)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Prediction archive failed CRC verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=artifact_path('reports', 'camera_pilot_20261026'))
    parser.add_argument("--output", type=Path, default=artifact_path('site', 'reports/stage1b'))
    parser.add_argument("--pair-report", type=Path)
    args = parser.parse_args()
    d, narrative = read(args.report / "results.json"), read(args.report / "narrative.json")
    if not d.get("complete") or not narrative.get("reviewed"):
        raise ValueError("Complete audited results and reviewed narrative are required")
    audit, media = read(args.report / "data_audit.json"), read(args.report / "media/manifest.json")
    pair = read(args.pair_report / "results.json") if args.pair_report else None
    if pair and not pair.get("complete"):
        raise ValueError("The conditional follow-up must be independently audited and complete")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("data", "figures", "media", "evidence"):
        (output / directory).mkdir(exist_ok=True)
    prediction_archive(d, pair, output / "data" / "predictions_and_training_records.zip")
    for name in ("results.json", "results.csv", "narrative.json", "data_audit.json", "prefix_parity.json", "source_binding_audit.json"):
        path = args.report / name
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, output / "data" / name)
    for path in args.report.glob("failure_audit*.json"):
        shutil.copy2(path, output / "data" / path.name)
    for path in (args.report / "media").iterdir():
        if path.is_file():
            shutil.copy2(path, output / "media" / path.name)
    for path in args.report.glob("*.png"):
        shutil.copy2(path, output / "figures" / path.name)
    for path in args.report.glob("*.pdf"):
        shutil.copy2(path, output / "figures" / path.name)
    if pair:
        for name in ("results.json", "results.csv"):
            shutil.copy2(args.pair_report / name, output / "data" / ("paired_" + name))
        for pattern in ("*.png", "*.pdf"):
            for path in args.pair_report.glob(pattern):
                shutil.copy2(path, output / "figures" / path.name)
    sources = [("docs/stage1b_protocol_20261026.md", "训练前协议"),
               ("docs/stage1b_data_audit_20261026.md", "独立数据审计"),
               ("docs/stage1b_input_audit_20261026.md", "输入与目标泄漏检查"),
               ("docs/stage1b_failure_audit_20261026.md", "接触切换失败案例审计"),
               ("scripts/build_camera_pilot_data.py", "相机生成代码"),
               ("scripts/run_camera_pilot.py", "训练与评估代码"),
               ("src/tactile_vggt/camera_pilot.py", "坐标与统计实现"),
               ("scripts/summarize_camera_pilot.py", "独立结果复算"),
               ("src/tactile_vggt/token_fusion.py", "沿用的小型融合模块"),
               ("artifacts/camera_pilot_data_20261026/manifest.json", "数据生成清单"),
               ("artifacts/camera_pilot_data_20261026/coordinate_audit.json", "生成时坐标审计")]
    if pair:
        sources.extend([("scripts/run_camera_pair_pilot.py", "同输入位移监督补测"),
                        ("scripts/summarize_camera_pair_pilot.py", "补测独立复算"),
                        ("docs/stage1b_followup_20261026.md", "补测触发与运行记录")])
    source_links = []
    for relative, label in sources:
        parts = Path(relative).parts
        source = (artifact_path("reports", "historical_docs", *parts[1:]) if parts[0] == "docs" else
                  artifact_path("artifacts", *parts[1:]) if parts[0] == "artifacts" else ROOT / relative)
        destination = output / "evidence" / source.name
        shutil.copy2(source, destination)
        source_links.append(f'<a href="evidence/{escape(destination.name)}">{escape(label)} ↗</a>')

    diagram = '''<div class="pipeline-wrap"><svg viewBox="0 0 1040 310" role="img" aria-label="同步左右RGB经过冻结VGGT前缀；触觉编码为44个token，在最后global block前或后融合；池化并结合关节状态，预测物体相对手掌中心；按每个观测的相机标定转回机器人坐标评分。" class="b-pipeline">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#77978b"/></marker></defs>
<g fill="#fff" stroke="#cbdcd1"><rect x="20" y="35" width="175" height="65" rx="8"/><rect x="245" y="35" width="220" height="65" rx="8"/><rect x="515" y="35" width="225" height="65" rx="8"/><rect x="790" y="35" width="230" height="65" rx="8"/><rect x="245" y="150" width="220" height="65" rx="8"/><rect x="515" y="150" width="225" height="65" rx="8"/><rect x="790" y="150" width="230" height="65" rx="8"/></g>
<g fill="#203b36" font-family="sans-serif" text-anchor="middle" font-size="17"><text x="107" y="61">同步双目 RGB</text><text x="107" y="84" font-size="12">一个时刻 · 一个视角版本</text><text x="355" y="61">冻结 VGGT 前缀</text><text x="355" y="84" font-size="12">真实预训练 patch tokens</text><text x="627" y="61">最后一个 global block</text><text x="627" y="84" font-size="12">权重冻结 · 输入梯度可通过</text><text x="905" y="61">池化 + 关节状态</text><text x="905" y="84" font-size="12">训练读出与中心回归头</text><text x="355" y="176">44 个触觉 token</text><text x="355" y="199" font-size="12">压力 / ID / 已知位置</text><text x="627" y="176">小型触觉 adapter</text><text x="627" y="199" font-size="12">internal 在前；late 在后</text><text x="905" y="176">物体相对手掌中心</text><text x="905" y="199" font-size="12">归一化第一相机坐标</text></g>
<g stroke="#77978b" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M195 67H240"/><path d="M465 67H510"/><path d="M740 67H785"/><path d="M465 182H510"/><path d="M627 150V105"/><path d="M905 100V145"/></g>
<text x="520" y="266" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#176a5c">本轮改变：同一物理状态换相机视角 → 更新坐标表达 → 在机器人坐标系统一评分</text>
<text x="520" y="291" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6b7d75">visual 省略触觉；tactile 控制组使用零视觉 token，并跳过 VGGT global block</text></svg></div>'''

    motivation = '<p>Stage 1.a 的触觉融合降低了部分中心定位误差，但物体真实移动时，预测位移仍不可靠；775 个空间样本中的相机外参也完全固定。</p><div class="b-question">保持网络不变，增加标定一致的视角，能否让空间预测更可靠，并增强 RGB 与触觉的共同使用？</div><p>判断依据包括定位精度、未见视角表现、跨视角一致性和物体位移响应。触觉单模态对照用于检查收益是否主要来自接触与手型先验。</p>'
    implementation = diagram + table(["控制项", "本轮实际设置"], [
        ["物理状态", "318 个原状态 + 457 个静态物体变体 = 775；没有新增操作示范"],
        ["视角数据", "每状态原视角 + 3 个 near；test 再加 3 个 strong；共 3,445 行"],
        ["near / strong", "双目中点平移 1–2 / 4–5 cm，共同旋转 1–2 / 4–5°"],
        ["划分", "38 / 7 / 8 条来源 episode；2,240 / 400 / 805 行观测"],
        ["模式与预算", f"fixed / augmented × visual / tactile / late / internal × 3 seeds；每组 {d['training_budget_steps']:,} 步"],
        ["保持可比", "相同基础样本抽样；共享 train-only 归一化；同一 dev 选点规则"],
        ["训练范围", "更新 adapter / 状态编码 / 读出与回归头；VGGT 全部冻结"],
        ["坐标来源", "实际双目基线统一归一化；传感器位置来自FK和已知相机标定"],
        ["评分", "用已知逐样本外参转回机器人基座；无真值拟合尺度或 Sim3"]],
        "2 cm / 5 cm 指双目中点平移，旋转还会移动每台相机；实际单相机总位移上界约 4.43 / 10.85 cm。")
    implementation += '<p>VGGT 在这里提供冻结视觉特征，未调用其相机或深度预测头来恢复米制场景。所有模式共用 22 维机器人关节输入；物体自由位姿不进入 encoder。</p>'
    implementation += f'<details><summary>查看数据检查、训练步数选择与完整曲线</summary><div class="details-body"><p>以前的物体变体重新计算过 native 压力；本轮仅相机变化，压力与 12 类原物理数据逐行一致，qpos / qvel 跨视角一致。保存后坐标往返误差小于 1e-7 m；未推进物理时间。训练入口校验所有源数组 SHA。</p><p>{escape(narrative["budget_explanation"])}</p><img class="b-figure" loading="lazy" src="figures/development_curves.png" alt="全部八组、三个seed的开发集定位误差曲线"><p><a href="evidence/stage1b_protocol_20261026.md">训练前协议</a> · <a href="evidence/stage1b_data_audit_20261026.md">完整审计</a></p></div></details>'

    results = '<p>先看定位，再看它是否正确响应物体移动。固定与增强训练在同一批测试物理状态上评估。</p>' + metric_table(d, "center_mm")
    results += '<figure class="b-plot"><img class="b-figure" loading="lazy" src="figures/center_by_camera_domain.png" alt="原视角、未见轻扰动、较强扰动的定位误差，固定与增强训练对照"><figcaption>柱：三 seed 均值；误差线：训练 seed 的 sample SD；散点：各 seed。</figcaption></figure>'
    contrasts = []
    for mode in MODES:
        values = d["paired_comparisons"]["augmentation_effect"][mode]
        contrasts.append([LABELS[mode]] + [interval_cell(values[domain]["center_mm"]["across_seed_descriptive"]) for domain in DOMAINS])
    results += '<details><summary>查看增强效果的配对区间与三个 seed 明细</summary><div class="details-body">' + table(["增强 − 固定（mm）", "原视角", "未见轻扰动", "较强扰动"], contrasts,
        "负值表示增强后误差更低；方括号是对这三个已训练模型条件化的 95% episode 配对区间。只有 8 条原始测试 episode，未做多重比较校正。")
    seed_rows = [[r["seed"], r["regime"], r["mode"], r["best_step"]] + [fmt(r["domains"][dom]["center"]["episode_mean_mm"]) for dom in DOMAINS] for r in d["runs"]]
    results += table(["seed", "数据", "模式", "选中步数", "原视角", "轻扰动", "较强扰动"], seed_rows) + '</div></details>'

    mechanism = '<h3>物体位移：手部状态相同，物体位置改变时预测有没有跟着变化？</h3><p>这里比较静态物体干预前后的预测差值，未评估连续运动轨迹。</p>' + metric_table(d, "displacement_error_mm")
    zero = d["descriptive"]["fixed"]["visual"]["nominal"]["zero_shift_error_mm"]["mean"]
    mechanism += f'<p class="b-callout">零位移预测对照：<strong>{fmt(zero)} mm</strong>。这个对照始终预测“没有移动”；融合模型需要超出这种静态先验，才更支持可靠的位移响应。</p><img class="b-figure" loading="lazy" src="figures/counterfactual_displacement.png" alt="同来源同相机的物体位移误差，与零位移控制比较">'
    mechanism += '<details><summary>跨视角一致性、方向与幅度，以及融合相对视觉的变化</summary><div class="details-body">' + metric_table(d, "cross_view_mm", ("near", "strong"))
    mechanism += '<p>一致性是同一物理状态换视角后，预测在基座坐标系的变化。它越小不一定越好：常量预测也很稳定，因此需与上方准确性一起看。</p><img class="b-figure" loading="lazy" src="figures/cross_view_consistency.png" alt="同一物理状态跨视角预测的基座坐标变化">'
    interaction_rows = []
    for mode in ("late", "internal"):
        for domain in DOMAINS:
            interaction_rows.append([LABELS[mode], DOMAINS[domain], interval_cell(d["paired_comparisons"]["difference_in_differences"][mode + "_minus_visual"][domain]["center_mm"]["across_seed_descriptive"])])
    mechanism += table(["融合模式", "测试视角", "融合增强收益 − 视觉增强收益（mm）"], interaction_rows,
        "负值表示融合获得了更多定位收益；单靠这项也不能证明三维对应。完整方向、幅度、P90 和逐 seed 数据可下载。") + '</div></details>'

    followup = ''
    if pair:
        followup = '<h3>补测：增加位移监督是否有帮助？</h3><p>主实验触发了预先约定的诊断：中心误差下降，但位移响应仍弱。因此额外训练 12 个模型，只比较 RGB + 关节与 Internal。新两组使用完全相同的初始化和每步输入：16 个普通观测 + 8 对同相机的物体变体；一组只监督中心，另一组增加权重为 1 的位移损失。每组仍为 4,000 步。</p>'
        pair_rows = []
        for mode in ("visual", "internal"):
            for objective in ("center", "delta"):
                pair_rows.append([LABELS[mode], "仅中心" if objective == "center" else "中心 + 位移"] +
                                 [summary_cell(pair["descriptive"][objective][mode][domain]["displacement_error_mm"]) for domain in DOMAINS])
        followup += table(["模型", "训练目标", "原视角 ↓", "未见轻扰动 ↓", "较强扰动 ↓"], pair_rows,
                          "位移误差，mm；三个 seed 均值 ± SD。仅在这两组新对照之间解释损失作用；与主实验的采样方式不同。")
        followup += '<img class="b-figure" loading="lazy" src="figures/paired_displacement_error.png" alt="相同初始化和输入批次，增加位移监督的结果">'
        followup += f'<p class="b-callout">{escape(narrative["pair_finding"])}</p>'
        followup += '<details><summary>查看补测的中心误差、配对区间与来源</summary><div class="details-body">'
        rows = []
        for mode in ("visual", "internal"):
            for objective in ("center", "delta"):
                rows.append([LABELS[mode], "仅中心" if objective == "center" else "中心 + 位移"] +
                            [summary_cell(pair["descriptive"][objective][mode][domain]["center_mm"]) for domain in DOMAINS])
        followup += table(["模型", "训练目标", "原视角中心 ↓", "轻扰动中心 ↓", "较强扰动中心 ↓"], rows)
        rows = [[LABELS[mode]] + [interval_cell(pair["paired_comparisons"]["delta_minus_center"][mode][domain]["displacement_error_mm"]["across_seed_descriptive"]) for domain in DOMAINS] for mode in ("visual", "internal")]
        followup += table(["增加位移损失的误差变化", "原视角", "轻扰动", "较强扰动"], rows,
                          "负值表示误差降低；95% 条件 episode 配对区间。补测复用历史测试集，是诊断性结果，仍需要新数据确认。")
        followup += '<p>各 seed 的两组模型逐一核验完整训练输入流与初始化 SHA；同一对照使用相同实际运行环境。权重和选点均未根据补测测试结果调参。</p><p><a href="data/paired_results.json">补测完整 JSON</a> · <a href="data/paired_results.csv">补测 CSV</a> · <a href="evidence/stage1b_followup_20261026.md">补测触发与计算环境</a></p></div></details>'
        mechanism += followup

    conclusion = '<div class="b-findings">' + ''.join(f'<article><span>0{i+1}</span><div><h3>{escape(v["title"])}</h3><p>{escape(v["text"])}</p></div></article>' for i, v in enumerate(narrative["findings"])) + '</div>'
    conclusion += f'<p class="b-callout">{escape(narrative["policy_status"])}</p>'
    next_body = table(["优先级", "下一步", "要回答什么"], narrative["next_steps"])
    next_body += f'<p>{escape(narrative["recommendation"])}</p>'
    videos = '<p>三例按测试集的接触与遮挡程度预先选取，未按预测好坏挑选。每段依次展示原视角、三个 near 和三个 strong；每张画面停留 1.5 秒。</p><div class="b-videos">'
    video_labels = {"no_contact": "无接触参考", "contact_visible": "接触 · 较少遮挡", "contact_occluded": "接触 · 较重遮挡"}
    for item in media["videos"]:
        videos += f'<figure><video controls preload="metadata" playsinline poster="media/{escape(item["poster"])}" src="media/{escape(item["file"])}"></video><figcaption><strong>{video_labels[item["id"]]}</strong> · physical sample {item["physical_sample"]} / episode {item["episode"]}<br>同一物理状态，触觉不变；这是视角展示，未执行策略 rollout。 <a href="media/{escape(item["comparison"])}">查看并排图片 ↗</a></figcaption></figure>'
    videos += '</div>'
    evidence = '<div class="b-downloads"><a href="data/results.json">完整结果 JSON ↗</a><a href="data/results.csv">结果表 CSV ↗</a><a href="data/predictions_and_training_records.zip">逐样本预测与训练记录 ZIP ↗</a><a href="data/data_audit.json">数据审计 JSON ↗</a><a href="data/source_binding_audit.json">预测与源数据绑定核验 ↗</a><a href="data/prefix_parity.json">原视角特征复现检查 ↗</a><a href="media/manifest.json">视频选例与来源 ↗</a>' + ''.join(source_links) + '</div>'
    evidence += '<p class="table-note">主实验在 rimlab-server 的 A40 slot 0 运行；12 组补测在 HPC 的三张 A40 并行运行。共 77 项本地 CPU 检查通过；全部 36 个正式模型的预测已独立复算。主任务与集群作业均已结束并释放 GPU。</p>'
    evidence += '<p class="table-note">所有数值由逐样本预测独立复算；缺少运行或来源不匹配时，报告构建会失败。数据仍是单任务、单方块和原历史 episode，重复渲染不增加独立操作样本。本轮中心回归指标不等同完整几何重建或策略成功率。</p>'
    evidence += '<details><summary>这些结果还不能说明什么</summary><div class="details-body"><p>融合比 RGB 对照同时多出压力与传感器锚点 XYZ，后者还包含已知相机外参线索。因此二者差值不能唯一归因为压力；Internal/Late 的输入更匹配。将来需保留几何通路的 geometry-only 训练对照。</p><p>触觉是仿真 native 力代理，未标定成真实牛顿值。457 个物体变体保留原 qvel，并筛选有接触、穿透较小的状态；没有模拟后续动力学或重新达到力学平衡，也未覆盖完整的接触断开与恢复分布。GT mask 用于视角有效性筛选与分析，未作为模型输入。</p><p>已知外参和运动学提供坐标关系；没有学习未知相机标定。中心回归与位移监督的改善，不能直接证明隐藏表面补全、真实接触因果或正确局部三维对应。补测继续使用同一历史测试集，需要独立新场景确认。</p></div></details>'
    sections = [section("motivation", "01 / MOTIVATION", "为什么先改变视角数据", motivation),
                section("implementation", "02 / IMPLEMENTATION", "只改数据，保留小型融合结构", implementation),
                section("results", "03 / RESULTS", "定位精度：增强带来什么变化", results),
                section("mechanism", "04 / SPATIAL RESPONSE", "空间响应：位置更准，还是只更稳定", mechanism),
                section("conclusion", "05 / CONCLUSION", "这一轮能够支持的结论", conclusion),
                section("next", "06 / NEXT", "下一阶段建议", next_body),
                section("videos", "07 / OBSERVATIONS", "实际输入变化", videos),
                section("evidence", "08 / EVIDENCE", "复核与下载", evidence)]
    navigation = ''.join(f'<a class="nav-link" href="#{key}"><span>{i+1:02d}</span> {label}</a>' for i, (key, label) in enumerate([
        ("motivation", "实验动机"), ("implementation", "实现与数据"), ("results", "定位结果"), ("mechanism", "空间响应"), ("conclusion", "结论"), ("next", "下一步"), ("videos", "视角视频"), ("evidence", "数据与证据")]))
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 1.b · 相机增强与触觉融合</title><meta name="description" content="冻结VGGT与现有小型触觉adapter，相机视角增强的完整对照、空间响应和下一步建议。"><link rel="stylesheet" href="/assets/portal.css"><link rel="stylesheet" href="/assets/stage1b.css"><script src="/assets/stage1b.js" defer></script></head><body class="stage1b"><a class="skip" href="#main">跳到正文</a><aside class="sidebar" id="sidebar"><a class="brand" href="/"><span class="brand-mark">V<span>×</span>T</span><span>TACTILE × VGGT<small>实验与证据门户</small></span></a><div class="nav-label">STAGE 1.b</div><nav aria-label="报告导航">{navigation}</nav><div class="nav-label">HISTORY</div><a class="nav-link" href="/reports/stage1a/">↗ Stage 1.a 完整报告</a><a class="nav-link" href="/">← 返回全部阶段</a></aside><div class="workspace"><header class="topbar"><button class="menu-toggle" aria-label="展开导航" aria-expanded="false" aria-controls="sidebar">☰</button><div class="breadcrumb">研究工作台 <span>/</span> Stage 1.b</div><a class="quiet-link" href="#evidence">下载证据 ↗</a></header><main id="main"><section class="report-hero"><div class="eyebrow">STAGE 1.b / CAMERA AUGMENTATION / 2026.10.26</div><div class="report-title-row"><h1>换一个视角，<br>触觉融合是否更可靠？</h1><span class="badge">空间实验完成</span></div><p class="lead">{escape(narrative["lead"])}</p><div class="decision-panel"><div class="eyebrow">CURRENT FINDING</div><h2>{escape(narrative["verdict"])}</h2><p>{escape(narrative["verdict_detail"])}</p></div><div class="meta-row"><span>原 775 个物理快照</span><span>8 条测试来源 episode</span><span>VGGT 权重冻结</span><span>逐样本预测已复算</span></div></section><section class="overview-metrics report-metrics" aria-label="本轮规模"><div><strong>3,445</strong><span>同步双目观测</span></div><div><strong>{"24 + 12" if pair else "24"}</strong><span>{"主实验 + 条件补测模型" if pair else "正式模型训练"}</span></div><div><strong>{d['training_budget_steps']:,}</strong><span>每组统一步数</span></div><div><strong>3</strong><span>训练 seeds / 视角视频</span></div></section>{''.join(sections)}<footer class="footer"><span>Tactile × VGGT · Stage 1.b</span><a href="/reports/stage1a/">回顾 Stage 1.a ↗</a><a href="/">所有实验报告 ↗</a></footer></main></div></body></html>'''
    (output / "index.html").write_text(html)
    print(json.dumps({"output": str(output), "complete": True, "runs": d["completed_runs"]}))


if __name__ == "__main__":
    main()
