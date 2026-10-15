# response-K3 + 0.1 首拒 CE：实验重点总结

截至 2026-10-07，覆盖 9 月 21 日切换损失以来的已完成实验，包括 10 月 3 日补齐的控制实验。只保留影响配置选择的结果。

**核心判断：这套损失能改善局部对齐，是目前适合保留的实验基线；但尚未证明在线更新能跨数据集、跨训练阶段稳定带来端到端净加速。后续重点应是“成熟 drafter 何时值得更新、更新哪些样本”，而不是继续堆叠 loss 或增加更新强度。**

## 基线及统计口径

基线为 `mean(response-K3) + 0.1 × mean(首拒 replacement-token CE)`，两项独立归一化，rejected-K3 关闭。response 流包含接受位置及首拒位置的 actor 替代 token；“首拒”不限定为 block 第一个位置。

常用在线配置：每 5 个 actor step 更新一次 drafter，每次 5 个 optimizer step，LR=`1e-5`，训练窗口 2048，每个数据并行副本采集上限 16384 token。更新频率、更新步数、窗口和总采集预算是四个不同参数。

接受长度包含额外输出的 1 个 token，不是接受率；吞吐按总输出 token／总生成时间计算。探针中的 `quality` 仅指短输出设置，不代表答案正确率。

## 主要实验、结果与结论

| 实验 | 关键结果 | 支持的结论 |
|---|---|---|
| CE 配方筛选：冻结 actor50/100，CE=0.05/0.1/0.2、关闭 rejected-K3 | actor100 更新 20 步后，CE=0.05/0.1 的接受长度分别增加 0.0263/0.0247；0.2 仅增加 0.0058。早期 actor50 存在 teacher 状态跳变，不能据其结果排序。[1] | CE 提供有效纠正信号，但系数越大并非越好；0.1 是折中基线，不是已证实的全局最优值。 |
| DAPO 完整 CE 训练、K3-only 50 步及 CE 50 步复跑 | 同取前 50 步，CE 对 K3-only：接受长度 3.825 对 3.606，生成吞吐 +6.09%；CE 复跑为 3.761、6267 token/s，仍高于 K3-only 的 5626。可是首轮 CE 全 101 步对固定初始 drafter 的 offline 基线，整步累计时间仅少 0.71%，生成吞吐反而低 1.95%。[2][3] | CE 在 DAPO 早中期有正向证据，复跑没有推翻方向；完整在线轨迹有分叉，不能把差值全部归因于 CE，也不能外推为全程稳定收益。 |
| 额外加入独立 0.1 rejected-K3：首拒起前 1/2/3/4 位，再完整训练前 3 位方案 | 冻结实验中，各范围相对 CE 基线的额外接受收益区间均跨零。前 3 位完整训练相比 CE：接受长度 −3.47%、吞吐 −3.45%、整步耗时 +12.00%；末期首拒 K3 抽样饱和率仍约 66%。[4] | 缩短 rejected 范围没有解决首拒饱和，也未证明额外收益；不作为默认项。loss 变大或分量相同不等于梯度贡献相同。 |
| accepted-FKL 替换 response-K3；交叉开关 CE 与全拒绝分支独立 K3，再直接比较 K3+CE／FKL+CE | actor50/100、5/10/20 步及短／长探针的优劣反复变化。直接比较中，FKL 在 actor100 的 5 步负载下略有接受收益，10 步未保持；没有一致吞吐优势。[5] | 现有证据不足以替换 response-K3，或再加入全分支 rejected-K3。 |
| CE=0.1/0.2 × LR=5e-6/1e-5 × 5/10/15/20 步；完整 10 步更新与 32K/5 步训练 | 冻结扫描的最优深度随 actor、CE 和探针变化。完整 10 步运行中 CE0.2 吞吐高于 CE0.1，但没有隔离轨迹因素。固定同一 actor 复测已保存 drafter，5步/16K、5步/32K、10步/16K 的负载接受长度为 3.778、3.765、3.731；32K 无明确优势，10步较差。[3][6] | 保留 5步/16K 的保守基线；不因训练 loss 下降就增加更新步数或 LR。32K 降复用有效，但不等于提速。 |
| TACO2000 完整训练；actor50/100 × 初始／成熟 drafter 的更新深度扫描 | 完整训练验证分数 0.1568→0.2808，但接受长度 3.804→3.633。初始 drafter 追加训练可改善接受；成熟 drafter 收益弱，actor100/16K 追加20步的配对接受变化 −0.0422，95%区间 [−0.0573,−0.0260]。[7] | actor 任务能力提升与 drafter 接受改善是不同目标；冷启动适配所需步数不能直接套到成熟 drafter。 |
| TACO：同份 rollout 的嵌套 16K/32K 子集，0/3/5 步，多轮新数据 | 六轮平均独立训练响应 33.7→51.7，5步重复抽取比例 57.9%→35.4%；所有单轮接受收益区间跨零。actor100/seed47 零更新即生成异常，未并入正常效果均值。[8] | 多样性确实增加，尚未转化为可重复接受收益；3步、5步和不更新之间仍需按净收益选择。 |
| WildSci／GPQA：online 对 offline、固定 actor 复测、同前缀机制验证及加权 CE 完整训练 | 跨轨迹 GPQA 末期接受长度曾高 +0.167；固定同一 actor 后变为 −0.0264，无熵测速仅 +0.93%且配对方向不一致。固定前缀下，GPQA 的 KL(actor‖drafter) 3.896→3.226、同前缀接受率 +0.69个百分点，说明局部对齐确有改善。低熵严重失配 CE 权重加倍后，未证明净效率或 GPQA 收益。[9] | “学到更好的分布”不保证完整解码更快。加权 CE 暂不替代普通 CE；其完整训练起点已有差异，不能把全程差值当因果效果。 |
| MATH500／GSM8K 全集：三个 actor checkpoint，各比较初始／配套训练后 drafter，共12组 | 六对接受长度变化仅 −0.0269～+0.0366；准确率差 −0.40～+0.68个百分点，配对检验未见明确升降。单次吞吐 −5.69%～+20.06%，与接受变化不一致。[10] | 未证明稳定的跨任务加速或准确率变化；单次大幅 TPS 提升不能直接归功于 drafter，也不能据此证明严格无损。 |
| DAPO：16K 内位置／低熵动态采样完整训练，随后同数据三 seed 控制及成熟 drafter 连续三轮更新 | 完整训练第50步启用分层，全程耗时较历史基线 +7.92%，启用前已落后。修正为副本内全池位置分层后，初始 drafter 三 seed 负载接受增量均正（+0.006～+0.038）。成熟 drafter 累积15步后，三轮平均“分层−均匀”的题均接受增量：短输出 +0.0361，负载仅 +0.0070，后者95%区间 [−0.0132,+0.0280]；吞吐方向不稳定。[10][11] | 位置＋低熵失配是值得保留的小规模候选，尚未证明动态门控或成熟阶段持续更新的净加速，不直接替换在线默认策略。 |

## 最重要的认识与下一步

1. **熵是诊断信号，不是更新收益的替代指标。** DAPO 与 WildSci 都出现 actor 越来越确定、低熵位置条件失配却上升的现象。固定初始 drafter 的早／晚阶段，`H<0.01` 的接受率分别由 92.45%→86.87%、93.74%→81.84%。因此应关注“低熵但提议错误”的具体位置，不能仅凭低熵就升 LR。[9]
2. **成熟阶段的边际收益是当前瓶颈。** 初始 drafter 能学习，成熟 drafter 继续拟合旧小池通常收益很小。降低重复抽取、降低 loss、增加接受长度，均不自动等于扣除采集／训练／发布成本后的净收益。优先检验少量更新或跳过更新，而非默认加到10/20步。
3. **评测可信度决定能得出多强的结论。** 同 actor 的 target-only greedy 重复生成仍有48/64条不同，说明输出分叉不只来自 drafter；存在异常 teacher 跳变的结果不作为主要选型证据，也不能全部归为普通噪声。最新连续三轮通过精确恢复模型与 Adam 补齐，但轮间 source 状态漂移根因未明，因此仍属受限证据。[11]

建议继续以 **response-K3 + 0.1首拒CE、LR=1e-5、每5个actor步更新5步、2048窗口／16K预算、无rejected-K3** 为对照。下一项决策实验应只比较成熟 drafter 的“不更新／均匀更新／位置＋低熵更新”，共享新 rollout、轮换评测顺序，并以**扣除更新成本后的净吞吐、长输出接受长度和任务质量**共同决策。暂不扩大 CE、LR、训练步数或采集预算。

## 证据索引

[1]: ../outputs/frozen_actor_ce_replace_rejected_20260921/r1/ANALYSIS_20260921.md
[2]: ../outputs/drafter_analysis/dspark_dapo2000_online_response_k3_first_ce01_101step_20260921_r2/analysis_20260922/REPORT.md
[3]: ../outputs/drafter_analysis/dspark_dapo2000_online_response_k3_first_ce01_5draft_32ktok_101step_20260925_r1/analysis/report.md
[4]: ../outputs/drafter_analysis/dspark_dapo2000_online_response_k3_first_ce01_rejected_prefix3_101step_20260922_r2/analysis/REPORT.md
[5]: ../outputs/frozen_actor_objective_compare_20260926/r1/actor100/REPORT.md
[6]: ../outputs/frozen_drafter_compare_actor100_20260925_r4/report.md
[7]: ../outputs/taco2000_frozen_depth_20260927/COMPLETED_ANALYSIS_20260928.md
[8]: ../outputs/taco2000_nested_caps_20260928/ANALYSIS_20260929.md
[9]: ../outputs/drafter_analysis/wildsci_online_offline_comparison/REPORT.md
[10]: ../outputs/drafter_analysis/dspark_dapo2000_online_ce01_16k_position_entropy_gate_101step_20260930_r1/analysis_20261002/REPORT.md
[11]: ../outputs/experiment_setup/dapo_sampling_multiround_20261003/README.md

补充原始对照：[K3-only与CE](../outputs/drafter_analysis/dspark_dapo2000_online_response_k3_only_50step_20260922_r1/analysis_vs_ce/REPORT.md)；[前1–4位rejected-K3](../outputs/frozen_actor_rejected_prefix_20260922/r1/ANALYSIS_20260922.md)；[CE/LR/深度扫描](../outputs/frozen_actor_ce_lr_steps_20260923/r1/actor100/REPORT.md)；[FKL交叉实验](../outputs/frozen_actor_fkl_factorial_20260926/r1/actor100/REPORT.md)；[TACO完整训练](../outputs/experiment_setup/taco2000_frozen_depth_20260927/ONLINE_RUN_ANALYSIS.md)；[WildSci固定actor](../outputs/drafter_analysis/wildsci_fixed_actor100_comparison/REPORT.md)；[固定前缀](../outputs/drafter_analysis/wildsci_decode_probe/REPORT.md)；[加权CE](../outputs/drafter_analysis/wildsci_weighted_ce_20260929/REPORT.md)；[三seed及target-only补齐](../outputs/dapo_sampling_controls_20261002/analysis_sampling_20261003/COMPLETION.md)。
