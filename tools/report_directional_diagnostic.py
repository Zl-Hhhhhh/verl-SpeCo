"""Audit completed diagnostic scalars and render a report without loading NPU models.

Run with the host Python, which already provides NumPy and Matplotlib.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "paired_panel_manifest.json").read_text())
    stats = json.loads((root / "target_diagnostics.json").read_text())
    candidates = read_lines(root / "candidate_scores.jsonl")
    selected = read_lines(root / "selected_target_metrics.jsonl")
    memory = read_lines(root / "memory_usage.jsonl")
    key = lambda row: (row["sample_id"], row["anchor"])
    candidate_by_key = {key(row): row for row in candidates}
    assert len(candidate_by_key) == len(candidates) == manifest["candidate_count"]
    assert len({key(row) for row in selected}) == len(selected) == manifest["selected_count"]
    assert {key(row) for row in selected} == {key(row) for row in candidates if row["selected"]}
    fields = ("surplus", "deficit", "m", "current_tv", "actor_tv", "actor_entropy",
              "absolute_position", "trace_accepted_count", "trace_draft_count")
    for row in candidates:
        assert all(row.get(field) is not None and np.isfinite(row[field]) for field in fields)
        assert 0 <= row["m"] <= min(row["current_tv"], row["actor_tv"]) + 1e-5
        assert abs(row["m"] - min(row["surplus"], row["deficit"])) < 1e-5
        assert 0 <= row["trace_accepted_count"] <= row["trace_draft_count"]
    for row in selected:
        assert abs(row["m"] - candidate_by_key[key(row)]["m"]) < 1e-5
        assert [row[field] for field in ("old_actor_step", "new_actor_step", "feature_actor_step", "trace_actor_step")] == [0, 5, 5, 0]
        assert len(row["prefix_sha256"]) == 64
        assert np.isfinite(row["feature_drift_tv"])
    assert manifest["actor_reference_steps"] == [0, 5] and manifest["drafter_optimizer_steps"] == 0
    assert manifest["training_validation_overlap"] == 0
    assert stats["target_probability_checks"]["passed"]
    for check in stats["target_probability_checks"]["results"].values():
        assert check["checked_positions"] == len(selected)
        assert check["minimum_probability"] >= 0
        assert max(check["max_normalization_error"], check["max_tv_improvement_error"]) <= 1e-5

    log = re.sub(r"\x1b\[[0-9;]*m", "", (root / "execution.log").read_text())
    steps = []
    for line in log.splitlines():
        if re.search(r"\bstep:[1-5] -", line):
            values = {name: float(value) for name, value in re.findall(
                r"([\w/]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)", line)}
            steps.append({name: values[name] for name in (
                "step", "timing_s/gen", "timing_s/update_actor", "timing_s/step",
                "response_length/clip_ratio", "drafter/spec_decode/mean_acceptance_length")})
    assert [step["step"] for step in steps] == [1, 2, 3, 4, 5]
    assert (root / "status.txt").read_text().strip() == "complete"
    checks = stats["target_probability_checks"]["results"]
    audit = {
        "passed": True, "candidate_positions": len(candidates), "selected_positions": len(selected),
        "candidate_positions_per_response": dict(Counter(Counter(row["sample_id"] for row in candidates).values())),
        "selected_positions_per_response": dict(Counter(Counter(row["sample_id"] for row in selected).values())),
        "distinct_selected_prefix_hashes": len({row["prefix_sha256"] for row in selected}),
        "selected_mass_fraction": sum(row["m"] for row in selected) / sum(row["m"] for row in candidates),
        "minimum_available_gib": min(row["available_gib"] for row in memory),
        "initial_swap_used_gib": memory[0]["swap_used_gib"],
        "maximum_swap_used_gib": max(row["swap_used_gib"] for row in memory),
        "maximum_hbm_mib": max(device["used_mib"] for row in memory for device in row["hbm"].values()),
        "memory_guard_triggered": (root / "memory_guard_triggered.json").exists(),
        "actor_steps": steps,
        "greedy_counts": {},
    }
    for eta in (.5, 1):
        for name in ("direction", "all_mismatch"):
            prefix = f"eta{eta:g}_{name}"
            gains = np.array([row[prefix + "_greedy_p_gain"] for row in selected])
            audit["greedy_counts"][prefix] = {
                "changed": sum(row[prefix + "_top1_changed"] for row in selected),
                "beneficial": int((gains > 0).sum()), "harmful": int((gains < 0).sum()),
                "unchanged_probability": int((gains == 0).sum()), "mean_gain": float(gains.mean()),
            }
    (root / "result_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    quantiles = np.linspace(0, 100, 1001)
    colors = ("#2166ac", "#d6604d")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, field, label in zip(axes, ("m", "m_over_tv", "actor_entropy"),
                               ("Repair mass m", "m / TV(p+, q0)", "Actor entropy (nats)")):
        for rows, color, title in zip((candidates, selected), colors, ("All candidates (8192)", "Selected top-m (2048)")):
            values = np.array([row[field] for row in rows if row[field] is not None])
            ax.plot(quantiles, np.percentile(values, quantiles), label=title, color=color)
        ax.set(xlabel="Percentile", ylabel=label, xlim=(0, 100))
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(root / "candidate_selected_quantiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, field, label in zip(axes, ("absolute_position", "current_tv"),
                               ("Absolute token position (prompt included)", "TV(p+, q0)")):
        for rows, color, title in zip((candidates, selected), colors, ("All candidates", "Selected top-m")):
            ax.scatter([row[field] for row in rows], [row["m"] for row in rows],
                       s=4, alpha=.25, color=color, label=title, rasterized=True)
        ax.set(xlabel=label, ylabel="Repair mass m")
        ax.grid(alpha=.25)
    axes[0].legend(markerscale=3)
    fig.savefig(root / "position_and_mismatch.png", dpi=180)
    plt.close(fig)

    # Use recorded timestamps, since sampling includes npu-smi execution time.
    from datetime import datetime
    parse_time = lambda value: datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    start = parse_time(memory[0]["time"])
    elapsed = [(parse_time(row["time"]) - start).total_seconds() / 60 for row in memory]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    axes[0].plot(elapsed, [row["available_gib"] for row in memory], label="Host available")
    axes[0].plot(elapsed, [row["swap_used_gib"] for row in memory], label="Host swap used")
    axes[0].axhline(32, color="red", ls="--", label="Stop threshold: 32 GiB")
    axes[0].set(ylabel="GiB", ylim=(0, None))
    axes[0].legend(ncol=3, fontsize=8)
    for device_id in range(8, 16):
        axes[1].plot(elapsed, [row["hbm"].get(str(device_id), {}).get("used_mib", np.nan) / 1024 for row in memory], label=str(device_id), lw=.8)
    axes[1].set(xlabel="Minutes since experiment start", ylabel="HBM used (GiB)", ylim=(0, 64))
    axes[1].legend(title="Logical device", ncol=8, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(root / "memory_timeline.png", dpi=180)
    plt.close(fig)

    all_c = stats["cohorts"]["all_candidates"]
    sel_c = stats["cohorts"]["selected_top_m"]
    lines = [
        "# 定向概率修复：第 1 项实验结果（2026-10-15）", "",
        "actor0→5 配对诊断已完成，drafter 优化器更新为 0。新增失配信号超过本轮重复前向噪声；",
        "定向目标没有显示出优于等质量普通失配目标的 greedy 提议收益。本轮不能证明训练收益或在线吞吐收益。", "",
        "## 运行与覆盖范围", "",
        "- 分支 `experiment`，起点提交 `13624b06f8b43c75e556423a2ac12d22209d6994`；执行时包含本地兼容和诊断补丁。",
        "- 原始 Qwen3-8B + dspark_qwen3_8b_block7；seed=17，窗口 512。",
        "- 仅物理卡 4–7（逻辑设备 8–15），TP/SP=4，2 个数据副本；只持久化 SP leader 0、4 的结果。",
        "- 每副本 token 预算由 16,384 调整为 32,768，以维持全局预算；采集 80 条响应，16 条作内部 holdout，64 条进入候选池。",
        "- 64 条响应 × 128 候选 = 8,192；每响应 top-m 32 个，共 2,048 个不同选中位置；实际位置训练次数为 0。",
        "- 64 是去重响应 sample_id 数，不能直接当成 64 道独立题目。5 个 actor 批次共 317 个去重训练 prompt；验证集 30 个，交集为 0。",
        "- 全部候选保存 S/U/m、TV(p+,q0)、TV(p-,p+)、actor 熵、绝对位置、原 trace 接受/提议数。",
        "- 完整分布和修复目标数值检查覆盖选中的 2,048 个位置；先检查 512 个，再检查全部选中位置。其余 6,144 个候选没有保存完整概率，不能声称其目标已逐位置验证。", "",
        "## 修复质量与分母", "",
        "| 指标 | 全部候选 | 选中 top-m |", "| --- | ---: | ---: |",
        f"| 位置分母 | {len(candidates):,} | {len(selected):,} |",
        f"| m 均值 | {stats['candidate']['m']['mean']:.8f} | {stats['selected']['m']['mean']:.8f} |",
        f"| TV(p+,q0) 均值 | {stats['candidate']['current_tv']['mean']:.6f} | {stats['selected']['current_tv']['mean']:.6f} |",
        f"| TV(p-,p+) 均值 | {stats['candidate']['actor_tv']['mean']:.6f} | {stats['selected']['actor_tv']['mean']:.6f} |",
        f"| mean(m/TV) | {all_c['m_over_tv']['mean']:.4%} | {sel_c['m_over_tv']['mean']:.4%} |",
        f"| sum(m)/sum(TV) | {all_c['sum_m_over_sum_tv']:.4%} | {sel_c['sum_m_over_sum_tv']:.4%} |",
        f"| m > 1e-6 | {all_c['above_noise_count']}/{len(candidates)} ({all_c['above_noise_fraction']:.2%}) | {sel_c['above_noise_count']}/{len(selected)} (100%) |",
        f"| S>0 且 U>0 | {all_c['two_sided_count']}/{len(candidates)} | {sel_c['two_sided_count']}/{len(selected)} |",
        f"| m>噪声，条件为两侧非空 | {all_c['above_noise_given_two_sided_count']}/{all_c['two_sided_count']} ({all_c['above_noise_given_two_sided_fraction']:.2%}) | 2048/2048 (100%) |",
        f"| 原 trace 首个提议接受 | {all_c['trace_first_proposal_accepted_count']}/{len(candidates)} ({all_c['trace_first_proposal_accepted_fraction']:.2%}) | {sel_c['trace_first_proposal_accepted_count']}/{len(selected)} ({sel_c['trace_first_proposal_accepted_fraction']:.2%}) |", "",
        f"两组均无 TV=0 的位置。选中 25% 的候选保留了 {audit['selected_mass_fraction']:.2%} 的 m 总量。",
        "mean(m/TV) 是逐位置比值的均值，sum(m)/sum(TV) 是质量汇总比值；两者分母不同。", "",
        "| m/TV 分位数 | 全部候选 | 选中 top-m |", "| --- | ---: | ---: |",
    ]
    for percentile in ("p10", "p50", "p90", "p99"):
        lines.append(f"| {percentile} | {all_c['m_over_tv'][percentile]:.4%} | {sel_c['m_over_tv'][percentile]:.4%} |")
    lines += ["", "## 两类目标与 greedy 提议", "",
              "以下分母均为 2,048 个选中位置。Δa_target 是固定 p+ 对新 top-1 token 的概率变化，属于局部目标诊断。", "",
              "| η | 目标 | top-1 翻转 | 有利位置 | 不利位置 | 平均 Δa_target | 平均 TV 改善 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for eta in (.5, 1):
        for name in ("direction", "all_mismatch"):
            prefix = f"eta{eta:g}_{name}"
            value = audit["greedy_counts"][prefix]
            lines.append(f"| {eta:g} | {'r_dir' if name == 'direction' else 'r_all'} | {value['changed']} ({value['changed']/len(selected):.2%}) | {value['beneficial']} | {value['harmful']} | {value['mean_gain']:.8f} | {stats[prefix]['tv_improvement']['mean']:.8f} |")
    lines += ["", "两类目标都精确改善 ηm 的软分布 TV，但 r_dir 在 η=1 时平均 Δa_target 比 r_all 低约 0.000628。",
              "绝大多数位置没有改变 greedy top-1；逐位置 token 和 margin 已保存在 selected_target_metrics 中。未运行修复损失训练，不能把表中 Δa_target 当成实测接受率或神经网络更新后的收益。", "",
              "## 正确性与特征漂移", "",
              f"- 四种目标的最小概率均非负；最大归一化误差 {max(value['max_normalization_error'] for value in checks.values()):.3g}，最大 TV 改善误差 {max(value['max_tv_improvement_error'] for value in checks.values()):.3g}，均小于预设容差 1e-5。",
              "- 重复 q 前向的概率最大差及伪 m 差均为 0；噪声阈值为 max(1e-6, 10×p99(伪 m 差))=1e-6；p-=p+ 空变化 m 最大为 0。",
              "- 此噪声测试重复的是 drafter 前向；没有单独估计相同 actor 权重重复采集特征的噪声。",
              f"- q(F-)/q(F+) 覆盖 {stats['feature_drift_denominator']:,} 个相同 prefix 位置；固定同一个冻结 drafter 和 actor5 输出头，只替换 actor0/actor5 特征。",
              f"- 特征漂移 TV 均值 {stats['selected_feature_drift_tv']['mean']:.6f}，中位数 {stats['selected_feature_drift_tv']['p50']:.6f}，p90 {stats['selected_feature_drift_tv']['p90']:.6f}，最大 {stats['selected_feature_drift_tv']['max']:.6f}。不能把特征漂移全部解释为 actor 输出分布变化或训练效果。", "",
              "## 内存与运行成本", "",
              f"- {len(memory)} 个约 10 秒间隔采样；主机最低可用内存 {audit['minimum_available_gib']:.2f} GiB，32 GiB 停止保护未触发。",
              f"- 后四张卡逻辑设备的最高 HBM 占用 {audit['maximum_hbm_mib']/1024:.2f}/64 GiB。主机 Swap 从 {audit['initial_swap_used_gib']:.2f} GiB 起，峰值 {audit['maximum_swap_used_gib']:.2f} GiB；Swap 不是本轮开始时为零。",
              f"- actor0→5 主循环约 68 分 25 秒；配对处理 {manifest['pair_seconds']:.2f} 秒。没有无诊断运行的配对计时，不能把全部耗时当成上线增量。",
              "- vLLM 休眠使用普通 CPU 内存同步备份，绕过本机锁页分配失败；未修改已有 Python 包。外部环境准备和两次失败启动详情见 RUN.md。",
              "- 收尾核查：本次 run ID 的进程已全部退出；irqbalance 服务处于 active。", "",
              "## 产物与下一步", "",
              "原始产物目录：`outputs/directional_repair_diagnostic_20261015/halfnode_r1/`。", "",
              "- paired_panel_manifest.json：actor 版本、数据哈希、预算、SP leader、去重与隔离记录。",
              "- candidate_scores.parquet / .jsonl：全部候选逐位置指标；selected_target_metrics.parquet / .jsonl：选中位置的目标、greedy、特征漂移指标。",
              "- paired_rank*.pt：选中位置的 p-/p+/q(F+)/重复 q/q(F-) 完整词表 FP32 分布，可重新构造修复目标。",
              "- target_diagnostics.json、result_audit.json：分母、分位数、数学检查与运行审计。",
              "- candidate_selected_quantiles.png、position_and_mismatch.png、memory_timeline.png：分位数、位置与内存曲线。",
              "- 原默认 Parquet 编码在本机回读出现崩溃/选择标志读取错误，离线重分析禁用字典编码，并将 Parquet 的 selected 字段存为整数 0/1，JSONL 保留布尔值；逐字段回读验证见 artifact_readback.log。", "",
              "本轮已有超过重复 q 噪声的修复质量，并出现少量有利 greedy 提议变化；没有证据表明定向目标优于普通失配目标。后续若继续，应按计划先做第 3 项接入验证，再做固定同位置的四臂小规模可学习性对照。当前授权的第 1 项在上述覆盖范围内完成，第 2、3 项未启动。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
