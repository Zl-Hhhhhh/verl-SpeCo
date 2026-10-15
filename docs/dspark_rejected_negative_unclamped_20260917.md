# 拒绝流 asymmetric K3：取消负侧 log-ratio 截断

用户要求终止正在初始化/首批生成的 asymmetric 对照，取消拒绝流负侧截断后重新启动。
旧托管任务 `dspark_20260917T161940Z_3234646` 已停止，NPU进程确认清空；其日志和输出保留。

## 修改

`DSparkTrainingModel._compute_rejected_sampled_k3` 在
`dspark_rejected_asymmetric_k3_enabled=True` 时使用：

```python
delta = (log_p - log_q).clamp(max=rejected_log_ratio_clamp)
```

不再将负尾限制为-20；正侧上限仍为20，正尾切线阈值仍为2。
关闭拒绝流asymmetric时继续使用原双侧截断；接受流保持原样。
负尾K3对delta的导数趋于-1，对log(q)的导数趋于+1，因此能恢复压低错误候选概率
的梯度，而不会出现该局部导数随负尾绝对值指数增长的情况。参数梯度范数裁剪仍为1。

## 实验配置

- 新实验：`dspark_dapo2000_online_rejected_asym_neg_unclamped_101step_20260917`。
- 从原始Qwen3-8B与dspark_qwen3_8b_block7启动，101个actor步，禁用resume。
- 同上次DAPO unique2000 seed17、batch64、n5、512/8192、16 NPU、TP/SP4。
- 普通均匀anchor采样，低熵优先比例0，确定性切换阈值0；drafter固定LR 1e-5。
- 每5个actor步训练5次、4批/周期、每卡batch4；accepted asymmetric=false，rejected=true。
- 逐步记录actor验证熵；每5个optimizer step、每rank每流最多256个K3概率样本。
- 保留原发布后接受长度下降0.4的保护和checkpoint保存策略。

旧K3转储字段 `clamped` / `legacy_clamp_exceed_ratio` 表示按旧K3上限10会截断的样本，
不是本轮实际零梯度标记。`target_logprobs-draft_logprobs` 可用于重建未截断的原始delta。
现有元数据 `k3_log_ratio_clamp=20` 在本轮拒绝流asymmetric模式下仅表示正侧上限。

此前第96步的状态异常尚未定位，本轮没有宣称已修复该异常。

## 启动

```bash
docker exec speco-linzihan-workspace bash /workspace/verl-SpeCo/tools/run_dspark_rejected_asymmetric_101step.sh
```

回归测试覆盖极端负尾-30/-1000的有限纠错梯度、关闭asymmetric时原行为、正尾线性化和+20上限。

本次验证：DSpark backend测试76项通过，shell语法/diff空白/dry-run通过。
托管运行：`dspark_20260917T163422Z_3302504`；W&B：`4bff7eafd81f`。
启动时间：2026-09-17 16:34:22 UTC。运行时配置校验通过。

## OOM 后重启（r2）

上述任务在第5步完成首次drafter的5次更新后，恢复vLLM KV cache时遇到
`aclrtMallocPhysical ... 207001 (Out of Memory)`，退出状态1。
checkpoint目录仅有W&B元数据，没有可恢复的训练权重。

按用户要求，启动脚本将actor `fsdp_config.param_offload` 改为 `true`，
新实验名为 `dspark_dapo2000_online_rejected_asym_neg_unclamped_101step_20260917_r2`。
从原始actor/drafter重新运行101步；其他训练配置、损失修改与熵记录设置沿用。
旧任务日志、概率转储和熵记录保留。

r2已通过shell语法、diff空白和dry-run检查，最终actor参数offload覆盖值为true。
新托管任务：`dspark_20260918T030746Z_82084`，启动时间2026-09-18 03:07:46 UTC。
按用户此前无需等待训练确认的要求，后台启动后交还会话；未声称已通过第5步切换。
