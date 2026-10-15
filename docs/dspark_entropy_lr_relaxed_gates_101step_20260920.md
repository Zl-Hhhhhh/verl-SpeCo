# Entropy LR relaxed-gate 101-step run (2026-09-20)

状态：已完成 101/101，exit 0；UTC 2026-09-20 21:56:08 完成归档。2026-09-21 已完成结果审计。

结果：20 次候选全部通过预检、全部验证回滚；LR 始终为 `1e-5`。320 份 K3 dump 的
480 次同 SP 组 rank 对比精确一致，最终 62 个导出 tensor 与原始 drafter 完全相同。
没有复现之前的块状置零；本轮没有保留任何 drafter 参数更新。

另发现 vLLM Ray worker 新进程重载后探针 `max_tokens=512` 未传入 sampling 参数的
CPU 可复现问题；本轮历史请求长度未持久化，不能视为已经核实的 512-token 短验证。
验证总耗时 2.687 小时，约占训练循环 18.63%。

详细统计、比较边界和建议见
[运行分析报告](../outputs/drafter_analysis/dspark_dapo2000_online_entropy_lr_relaxed_gates_101step_20260920_r1/analysis/REPORT.md)。

## 变更

本轮从原始 Qwen3-8B actor 和 `dspark_qwen3_8b_block7` drafter 开始，不恢复旧训练状态。
相对 2026-09-18 的 entropy LR 规则只删除以下三个门槛：

- 候选不再要求平均接受长度至少提升 0.05；仍要求配对筛选区间下界大于 0、低熵失配下降，
  且预计净时间收益为正。
- rejected K3 saturation 不再阻止 LR 增长；该指标继续记录。
- certainty EMA 不再有 70% 激活门槛。LR ceiling 改为
  `base + (maximum - base) * clip(certainty_ema, 0, 1)`，即从 0% 到 100% 连续映射。

保留的 LR 增长条件包括：候选通过上述配对验证，当前低熵失配严格大于 10%。每次通过后
LR 乘 1.1，并受 certainty ceiling 和全局上限约束。明确退化仍将 LR 减半并受下限约束。

## 运行配置

- 实验 ID：`dspark_dapo2000_online_entropy_lr_relaxed_gates_101step_20260920_r1`
- managed run：`outputs/managed_runs/dspark_20260920_entropy_lr_relaxed_gates_101step_r1`
- W&B run ID：`03bab215aac9`
- 日志：
  `outputs/validation_logs/2026-09-20/dspark_vllm/dspark_dapo2000_online_entropy_lr_relaxed_gates_101step_20260920_r1.log`
- 101 actor steps，DAPO seed-17 固定 2000 条训练数据，原 test 集，batch 64 x 5 responses。
- 16 NPU，rollout TP=4，actor/drafter SP=4；actor 参数与 optimizer offload 均开启。
- 每 5 actor steps 收集并执行 5 个 drafter optimizer steps；候选 alpha 为 `[0, 1]`。
- drafter LR base `1e-5`、floor `5e-6`、maximum `1.5e-5`，EMA beta `0.3`。
- asymmetric K3 全部关闭；保留 `log(p/q)` 截断和 K3 loss 上限。
- 每 5 步记录 verification entropy 与 16-rank K3 样本；每 10 步保存完整 checkpoint。
- actor checkpoint 最多保留 1 份；成功后归档最终 drafter 并清理大型训练 checkpoint。

相关回归测试：

```text
86 passed, 17 warnings
```

在线 checkpoint 存储生命周期修复已先通过 15-step 压力回归；本轮仍需在每个保存后检查
K3 跨 rank 一致性，不能只以进程未退出作为完整性依据。
