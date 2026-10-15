# 拒绝流 asymmetric K3：普通在线采样，101步

此为已停止的首个asymmetric实验记录。随后用户要求取消负侧log-ratio截断并重新启动；
当前启动脚本默认实验已更新，见 `dspark_rejected_negative_unclamped_20260917.md`。

用户要求：开启 `dspark_rejected_asymmetric_k3_enabled`，使用原普通在线训练配置，
不切换低熵首次拒绝anchor采样，继续保存熵及概率诊断。

## 配置

- 原始actor：`/workspace/models/Qwen3-8B`；原始drafter：`/workspace/models/dspark_qwen3_8b_block7`。
- DAPO unique2000 seed17，batch64、rollout n5、prompt512/response8192、16 NPU、TP/SP4。
- 101个actor步，禁止自动续训；独立实验目录。
- 每5个actor步收集并训练，5个drafter optimizer step、每周期4批、每卡batch4，
  收集样本上限20/replica，窗口2048，原32-anchor预算及均匀哈希采样。
- drafter LR固定 `1e-5`；不启用动态LR。
- `dspark_rejected_asymmetric_k3_enabled=true`；accepted流该开关为false；
  总开关仍false，拒绝流显式覆盖为true。正尾切线阈值2.0。
- `dspark_low_entropy_anchor_fraction=0.0`、`dspark_certainty_share_threshold=0.0`。
- 不加入首次拒绝CE，保留原流权重和位置衰减。
- 保留发布后平均接受长度下降0.4的终止保护；候选gate维持false。

## 记录

逐步保存actor verify entropy gzip，K3概率样本每5个optimizer step保存一次，每rank
每流最多256个token。保存梯度、固定留出loss、接受率、训练/发布/整步耗时等现有指标。
不启用完整hidden-state诊断，以保持和上一轮相同的记录开销口径。

开关开启后负尾保留原K3，正尾线性化，不再使用K3值上限10的硬截断；但计算前的
log-ratio ±20截断仍存在。`legacy_clamp_exceed_ratio`应解读为“按旧规则会被截断的比例”，
不能继续据此判断新损失是否零梯度。后续应结合raw logp-logq、实际loss、linearized字段分析。

上一轮在第95轮训练前出现状态异常并于第96步保护终止；本轮未宣称该异常已修复。
本轮只运行用户指定的损失开关对照，不改动生产训练实现。

## 启动

```bash
docker exec speco-linzihan-workspace bash /workspace/verl-SpeCo/tools/run_dspark_rejected_asymmetric_101step.sh
```

实验名：`dspark_dapo2000_online_rejected_asym_k3_entropy_101step_20260917`。

托管运行：`dspark_20260917T161940Z_3234646`，W&B run：`d56322d74728`。
启动前DSpark backend测试70项通过，shell语法和dry-run配置检查通过。

## 对两层截断的补充检查

用上一轮低熵采样实验45–90步的K3转储，按每个SP组仅保留rank%4==0去重：
共2257个首次拒绝token，其中1409（62.43%）触发旧K3硬截断，971（43.02%）
的原始logp-logq超出±20，438（19.41%）触发旧K3截断但仍处于(-20,20)。
后一部分可由asymmetric开关恢复该项梯度；超出±20的部分仍受前置截断限制。
这是旧运行概率上的抽样诊断，不是新实验结果，也不是全量token比例。
