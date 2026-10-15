# Actor 确定性触发的在线 drafter 实验

用户选择：从原始 Qwen3-8B 重新训练 101 步，与
`dspark_dapo2000_offline_k3_entropy_101step_20260916_r2` 对照。
初始 drafter 为 `dspark_qwen3_8b_block7`，DAPO unique 2000 seed17，
batch64、每题5个回答、长度512/8192、16 NPU、TP/SP4；actor lr=1e-6。

## 策略

- 前期为普通在线 drafter 训练；每5个 actor 步收集并训练，沿用5训练步、
  每周期4批、每卡batch4、lr=1e-5及现有32-anchor预算。
- 每次收集时，在整个 rollout batch 的已到达验证位置统计 `H<0.01` 比例。
  包括已接受前缀及首次拒绝，不包括未验证后缀。
- 连续两次收集比例达到70%，才启用低熵优先采样；低于70%立即恢复普通采样，
  连续计数归零。上次固定 drafter 运行在约第40步达到70%，新运行按实测触发。
- 启用时将最多一半现有 anchor 预算保留给低熵首次拒绝，剩余名额按原确定性
  哈希排序补足；不足时回填，不重复 anchor、不增加预算、不改变 crop。
- 固定评估留出数据始终使用原采样。K3及其截断保持不变，不额外加入CE。
  K3截断可能导致极端失配位置没有梯度，这是本轮需要检查的限制。
- 切换由当次收集数据携带到训练端；若将来启用跨周期重放，旧样本仍携带其
  收集时策略标记。当前运行使用现有逐周期清理逻辑。

## 记录与解释

- 与基线一样保存每步 actor verify entropy（预计数GB）。
- 每5个 drafter optimizer step、每rank、每个accepted/rejected流，最多保存256个
  K3概率样本；包括actor/drafter logprob、原始K3、位置和验证计数。
  不启用隐藏状态诊断转储。
- 日志记录 certainty gate、measurement_step、候选/目标/实际选中/普通采样参考数量。
  W&B记录 `drafter/certainty/*`，非收集步显示最近一次测量及其原始步数。
- 接受长度、生成吞吐、actor奖励、整步耗时、drafter训练和发布开销使用现有指标。
  额外采集和训练开销应计入实际端到端性能，不能只比较生成耗时。
- 该运行与固定 drafter 基线比较只能评估整套在线方案；要单独归因于阈值采样，
  仍需等预算普通在线组。相同seed也不保证更换drafter后actor轨迹完全相同。
- 当前不是可恢复的门控状态实验：脚本禁止自动续训；若后续恢复checkpoint，
  需要保存/恢复门控计数或明确重新预热两个周期。

## 启动

```bash
docker exec speco-linzihan-workspace bash /workspace/verl-SpeCo/tools/run_dspark_low_entropy_101step.sh
```

默认后台托管运行，独立experiment目录；保留结束checkpoint并沿用已有保留策略。
仍启用既有发布后接受长度骤降保护，不自动改变损失或重试失败训练。

## 本次运行记录

- 实验：`dspark_dapo2000_online_lowentropy_k3_101step_20260917_r2`
- 托管运行：`dspark_20260917T050543Z_957043`
- W&B run：`e29fb3a22ae7`
- 初次启动在数据准备阶段主动停止，原因是检查发现 worker 字段白名单未传递
  门控标记；修复并补充 True/False 传递测试后重新启动。初次启动没有训练步。
- 验证：原backend、采样、门控、runtime合同联合139项通过；补充worker传递及
  丢失标记保护后，采样和门控专项16项通过；shell语法及diff空白检查通过。
