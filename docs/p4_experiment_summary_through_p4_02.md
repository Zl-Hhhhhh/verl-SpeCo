# P4 系列阶段分析：截至 P4_02（2026-09-08）

目前应继续把原始 P4 作为参照。P4_01 的“首拒绝 ×2、后续 ×0.4、增加样本及更新步数”组合没有取得收益；P4_02 的学生/老师 top-16 并集确实补足了候选覆盖，但尚未提高接受长度。下一步更值得验证的是首拒绝处正确替换 token 的直接正向监督，而不是继续扩大候选数或叠加 rejected 权重。

本文分析已有日志和 dump，没有改动训练配置或启动新实验。

## 1. 实验与数据完整性

三组均完成 101 个 actor step、20 轮 drafter 更新，管理进程退出码均为 0。逐文件检查全部 2,880 个 K3 log-probability dump；同一 replica/update 的四份 SP 数据内容一致，去重后为 720 份，包含 1,296,858 条 token 训练观测和 140,023 对可匹配的首拒绝正确/错误 token 观测。没有 dump 或 batch preparation 失败。

这些是训练观测，不是同样数量的独立样本或留出集。首轮有少量样本复用；同一 response 的多个 anchor 也不是独立题目。

| 配置 | P4 | P4_01 | P4_02 |
| --- | --- | --- | --- |
| W&B run ID | `0fe556fc2ba1` | `5ddaa04ec4e6` | `5cd6af91301e` |
| 开始日期 | 09-05 | 09-06 | 09-07 |
| Response / rejected loss | 学生 top-16 reverse KL | 学生 top-16 reverse KL | 学生 top-16 ∪ 老师 top-16 reverse KL |
| 首拒绝 / 后续 rejected 乘数 | 1 / 1 | 2 / 0.4 | 1 / 1 |
| 每 replica 每轮采集 | 40 | 64 | 40 |
| 全局每轮实际采集 | 80 | 128 | 80 |
| 每轮 optimizer steps | 5 | 8 | 5 |
| 总 optimizer steps | 100 | 160 | 100 |
| 原始 dump / 去重 dump | 800 / 200 | 1280 / 320 | 800 / 200 |
| 额外固定轨迹导出 | step 60 / 80 / 100 | 无 | 无 |

共同设置：DAPO，Qwen3-8B + 官方 DSpark block-7 冷启动，8 个逻辑设备、两个 replica，TP/SP 均为 4；每 5 个 actor step 训练 drafter，LR 固定为 1e-5；每 replica optimizer batch 8，全局 16；每 response 32 个 anchors；response/rejected 基础权重 1/0.5，rejected 绝对 block 位置衰减 0.8；asymmetric K3 和动态流权重关闭。101 step、双 best 保存、K3 dump 均启用。

三组都是原来的 full attention：actor prefix 可见，draft block 内双向可见。训练仍裁剪 512 行 feature（另有一行对齐），rollout 读取完整请求 prefix；这不是 rollout SWA-512。实际被选 anchor 的平均局部 context 约 254 tokens，不能把“full”理解成训练每个 anchor 都看到几千 token。

P4_01 的采集上限实际达到了 128，并非参数没有生效。首轮每 replica 留出 8 条内部评估样本后，训练独立 response 数为 32/56/32；之后各轮分别为 40/64/40，轮内样本复用率为 0。因此这次结果也不能归因于“仍然只在反复训练少量 response”。独立 response 数不等于独立题目数。

原始配置和启动记录见 [P4 实验记录](p4_top16_full_run_2026-09-05.md)。

## 2. Rollout 结果

以下统一统计 actor step 1–100，step 101 单列，避免终止步影响分段比较。接受长度是逐 step 指标的算术平均，包含 bonus token。生成吞吐为总有效 response tokens / 总 `timing_s/gen`；有效输出量由每步平均 response 长度 ×320 重建，存在浮点取整误差。

| 指标，step 1–100 | P4 | P4_01 | P4_02 |
| --- | ---: | ---: | ---: |
| 平均接受长度 | 3.717 | 3.510 | 3.697 |
| 相对 P4 | — | -5.57% | -0.55% |
| 生成吞吐，输出 token/s | 3805 | 3614 | 3458 |
| 相对 P4 | — | -5.03% | -9.13% |
| 平均生成时间，秒/step | 450.2 | 508.4 | 510.3 |
| 平均总时间，秒/step | 636.3 | 713.5 | 717.7 |
| 100 step 总时间，小时 | 17.68 | 19.82 | 19.94 |
| 平均 response 长度 | 5354 | 5742 | 5515 |
| 输出截断比例 | 29.65% | 35.43% | 31.75% |
| 训练 reward，[-1, 1] | 0.260 | 0.181 | 0.238 |
| W&B `perf/throughput` | 347.7 | 331.6 | 318.9 |

W&B `perf/throughput` 包含 prompt+response tokens，并按总 step 时间和逻辑设备数归一，不能与上表输出 token/s 混用。训练 reward 也不是独立 held-out 正确率。

| Actor steps | P4 接受长度 | P4_01 | P4_02 |
| --- | ---: | ---: | ---: |
| 1–20 | 3.408 | 3.387 | 3.414 |
| 21–40 | 3.632 | 3.595 | 3.579 |
| 41–60 | 3.868 | 3.519 | 3.820 |
| 61–80 | 3.836 | 3.555 | 3.858 |
| 81–100 | 3.841 | 3.494 | 3.812 |
| 101，单步 | 3.823 | 3.626 | 3.817 |

P4 在约 40–60 步后进入平台。P4_01 后半段持续低于 P4，增加到 160 次 optimizer 更新没有带来预期收益。P4_02 在 60–80 步接近 P4，但没有形成持续领先；最后 20 步也没有继续上升。

P4_01 同时改变了流内权重、采集量、更新次数。这是该组合的负面结果，不能单独推导“增加独立样本有害”或“首拒绝加权必然有害”。P4_02 与 P4 的接受长度差只有约 0.5%，单次在线 run 不足以证明并集本身会降低质量。

![Rollout 与实际 top-k loss](../outputs/drafter_analysis/p4_comparison_2026-09-08/p4_rollout_and_losses.png)

### 吞吐回退不能全部归因于 loss

首次 drafter 更新发生在 step 5。在前 4 步、drafter 尚未发生这些策略更新时，三组接受长度分别为 3.330 / 3.335 / 3.342，平均 response 长度也都约 6900，但生成吞吐已经分别为 3623 / 3181 / 2932 token/s。

P4_01 在这个阶段已经比 P4 慢 12.2%，P4_02 慢 19.1%。P4_01 没有新增 teacher top-k 并集计算，说明存在未控制的运行时差异；仅凭总耗时无法定位到具体硬件或算子原因。P4_02 的并集采集确实新增 teacher top-k 和元数据工作，但它贡献了多少时间，需要同样硬件、actor、请求下的 profile 才能分离。

三组平均每个 actor step 中，drafter 阶段仅为 3.65 / 4.63 / 3.95 秒，生成阶段为 450 / 508 / 510 秒。P4_01 多出的直接训练时间解释不了多出的约 77 秒总 step 时间。生成仍占总 step 约 71%，另有输出变长导致的 actor 更新、old-logprob 成本。

## 3. 实际训练 loss 和 teacher 覆盖

| Run | 总 loss：step 5 → 50 → 100 | Response top-k：5 → 100 | Rejected top-k：5 → 100 |
| --- | --- | --- | --- |
| P4 | 5.729 → 6.873 → 7.946 | 5.314 → 6.681 | 7.140 → 13.660 |
| P4_01 | 5.691 → 7.452 → 8.165 | 5.412 → 6.801 | 6.588 → 13.601 |
| P4_02 | 6.674 → 8.416 → 8.962 | 6.429 → 7.347 | 7.433 → 15.175 |

top-k loss 上升属实，rejected 流最明显。但训练样本、actor 分布都在变；不同支持集和流内权重的 loss 数值也不能直接横向排名。比如 P4 的 response 实际 target token teacher 概率从约 0.865 升至 0.978，teacher 越来越尖锐：student 在错误候选上的剩余概率可能产生更大的 reverse KL，同时 target top-1 和 CE 仍然改善。

因此，不能单凭 loss 上升认定数值发散，也不能据此把上升全部解释为正常现象。结合 rollout 平台和下面的首拒绝分析，更准确的描述是：一般 response 预测在改善，关键失败位置仍存在较大分布差异。

| Step 100 候选概率覆盖 | P4 | P4_01 | P4_02 |
| --- | ---: | ---: | ---: |
| Response teacher mass | 96.70% | 95.53% | ≈100% |
| Rejected teacher mass | 87.86% | 85.24% | ≈100% |
| Rejected student mass | 94.81% | 93.68% | 96.11% |
| Response / rejected 平均去重候选数 | 16 / 16 | 16 / 16 | 26.26 / 26.78 |

P4_02 已经解决了绝大部分“teacher 高概率 token 不在候选集”问题。少量 teacher mass 为 1.000016 属浮点误差；并集重复 token 已在实现中排除。覆盖补足但接受长度不升，使“继续扩大 k”优先级下降，并提示候选覆盖之外的目标形式、训练分布和上下文问题值得检查。

三组实际优化的是支持集内重新归一的 `KL(q || p)`，见 [top-k 实现](../verl_speco/trainer/dspark_topk.py)。虽然仍导出 K3/clamp 诊断，K3 clamp 没有用于这三组的主 top-k objective。不能用 dump 中较高的 `clamped` 比例证明当前 top-k 梯度被截断。

## 4. Drafter 的训练步 × 位置矩阵

已整理全部 20 轮 ×7 个位置，以及各轮内部 optimizer step 的矩阵。图中位置 1–7 对应 CSV 的位置 0–6。

![逐轮逐位置 top-1 和 CE 矩阵](../outputs/drafter_analysis/p4_comparison_2026-09-08/p4_training_position_matrices.png)

| Step 100 response-path top-1 | 位置 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P4 | 82.3% | 75.5% | 74.1% | 74.8% | 69.3% | 69.2% | 67.0% |
| P4_01 | 78.4% | 73.2% | 71.5% | 69.5% | 68.4% | 64.9% | 65.2% |
| P4_02 | 79.7% | 75.7% | 73.3% | 71.3% | 68.3% | 68.1% | 60.3% |

| Step 100 response-path CE | 位置 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P4 | 0.964 | 1.410 | 1.654 | 1.520 | 2.026 | 1.992 | 2.200 |
| P4_01 | 1.184 | 1.591 | 1.906 | 1.934 | 2.185 | 2.310 | 2.436 |
| P4_02 | 1.104 | 1.486 | 1.561 | 1.996 | 1.983 | 1.989 | 2.630 |

三组从早期到后期的各位置 top-1 大体都提高，不能描述为整体不学习。P4_01 的不足也不是仅集中在后几个位置；P4_02 的第 7 位置从 step 75 的约 63.3% 到 step 100 的 60.3%，并集没有稳定改善尾部。

解释这张矩阵时必须区分三点：

1. `dspark_loss_per_position` 是 CE/NLL 诊断，即使 CE 权重为 0 也会计算；它不是逐位置 top-k loss。
2. top-1 统计 response 流中的已接受位置和首拒绝处正确替换位置，使用对应 response 的 previous token。它不是 rollout 的“到达位置 j 后条件接受率”；后面位置的样本还存在到达条件带来的选择偏差。
3. 每个 optimizer step 使用的 response/anchor 会变，轮内第 1→5/8 步曲线不能当成同一批样本的 pre/post 效果。当前 dump 不包含所有 top-k 候选分布或分流梯度，不能恢复真实逐位置 top-k KL 和梯度贡献矩阵。

## 5. 首拒绝配对分析：更值得解决的剩余问题

把同一 dump 中同一 sample/anchor/位置的“response 正确替换 token”和“首个错误 draft token”配对。首拒绝前的 draft prefix 已被接受，所以这个位置的两者可以用于直接比较正确/错误 token 的排序。下表采用不加权均值，避免 P4_01 权重变化本身改变统计口径。

| Step 100 首拒绝观测 | P4 | P4_01 | P4_02 |
| --- | ---: | ---: | ---: |
| 匹配 token 对数 | 1860 | 3392 | 2042 |
| Teacher P(正确替换) | 0.925 | 0.929 | 0.919 |
| Teacher P(错误 proposal)，配对样本 | 0.031 | 0.026 | 0.034 |
| Drafter Q(正确替换) | 0.295 | 0.280 | 0.288 |
| Drafter Q(错误 proposal)，配对样本 | 0.406 | 0.408 | 0.421 |
| Q(正确) > Q(错误) 的比例 | 40.5% | 38.1% | 37.3% |
| 正确替换 Q < 0.1 的比例 | 55.9% | 58.9% | 58.7% |
| 首拒绝占 rejected 流内权重 | 29.8% | 67.4% | 29.5% |

这些是记录过首拒绝的困难训练位置，不能解释成“整体生成有六成预测错误”。它们也不是训练后在统一固定集上的评估，跨 run 差异不能当成严格因果效果。

尽管如此，这组证据没有支持原先的预期：P4_01 首拒绝加权已经按设置生效，首拒绝的流内权重占比从约 30% 提至 67%，但正确替换概率、正确/错误排序没有优于 P4；P4_02 把 teacher 覆盖补到近 100%，也没有解决这个问题。

P4_01 是对首拒绝位置整项 top-k 分布损失加权，不是只对单个错误 token 加惩罚。简单增加该项权重，并不等于直接增强正确替换 token 的梯度。

![首拒绝匹配 token 对](../outputs/drafter_analysis/p4_comparison_2026-09-08/p4_first_rejection_pairs.png)

## 6. 保存的 best 与现有评估限制

| Run | best_acceptance drafter step / 当时 AL | best_throughput drafter step / 当时输出 token/s |
| --- | --- | --- |
| P4 | 85 / 3.9447 | 85 / 4247.5 |
| P4_01 | 25 / 3.6924 | 25 / 4212.4 |
| P4_02 | 70 / 3.9076 | 80 / 3887.9 |

这些选择器按发布后的下一轮 rollout 评价：对应 rollout 为 86、26、71/81。checkpoint 的 step 是 drafter 发布关联的 actor step，不是累计 optimizer step。原始 verify 次数、accepted tokens、有效输出量和计时均保存在 [best_checkpoints.json](../outputs/drafter_analysis/p4_comparison_2026-09-08/best_checkpoints.json)。

三组 best 使用不同 actor snapshot 和不同请求，不能直接据此判断哪个 checkpoint 在统一 actor 下最好。这里只核对保存结果，没有重测或改动已有 best。

内部 `fixed_cohort` 的 source step 一直为 5，实际唯一 response 为全局 16；日志 post_samples=64 包含四路 SP 重复计数。它长期复用早期的 actor features/trace，而当前 lm_head 会更新，因此不能作为当前 actor 上的新鲜 holdout。末几轮其 loss 改善变小，仅说明在这份内部数据上的边际变化小，不能单独证明过拟合或指导早停。

P4 另有 step 60/80/100 的固定轨迹，各 128 条 response、64 个 prompt，可作为统一比较的输入。应在选定 actor（例如已保留的 P4 actor 99）上重新计算 teacher/features；不同时间的原始 teacher trace 不能直接当作同一 teacher。它们来源于 DAPO 训练流，不是独立任务测试集。P4_01/P4_02 没有额外导出同规模轨迹，不能伪造相应的完整 rollout 条件接受矩阵。

## 7. 下一步优先级

### 优先一：以 P4 为底，单独测试首拒绝正确替换的 CE 辅助项

建议保持 full、学生 top-16、40 response/replica、每轮 5 optimizer steps、每 5 actor steps 更新、LR 1e-5、基础权重 1/0.5 和 decay 0.8，只增加首拒绝正确替换 token 的小权重 CE。一个明确的起点是：

```text
L_new = L_P4 + 0.1 * mean_first_rejection[-log Q(correct_replacement)]
```

这是待验证方案，不是由当前数据已经证明有效的方案。新增项应有独立开关，默认关闭。先观察 40–50 个 actor step 的同 actor/同批样本 pre/post 指标，有收益再做完整 101 step。

理由是支持集 reverse KL 对低 Q、较高 P 的候选，其 logit 梯度仍带有支持集内 Q 的因子；加入 teacher 候选并不保证这些 token 获得足够大的正向梯度。CE 直接对正确 token 施加 `Q-1` 形式的 logit 梯度，且不要求它先进入学生 top-16。该推断与首拒绝矩阵一致，但因果效果仍需实验。

当前已有的 `DSPARK_CE_LOSS_ALPHA=0.1` 作用于整个 response 流（已接受位置 + 首拒绝正确替换），不是 first-rejection-only。如果优先使用现有开关，它可以作为低实现成本的独立实验，但必须按全 response CE 解释。首拒绝专用 CE 需要加可选 mask。

`DSPARK_ACCEPTED_FKL_ALPHA` 目前是“目标 token / 其余词表”的二元 forward KL，不是全词表 forward KL，也不是首拒绝专用项；此外它先把 log Q 截到 -10，极低 Q 部分仍存在截断梯度的问题。末轮首拒绝正确替换中，P4/P4_01/P4_02 分别有 13.9%/14.4%/13.1% 的原始 log Q 低于 -10。因此这次优先推荐 CE，而不是直接开启现有 FKL 并假定其能解决低概率正确 token 的学习。这里讨论的是可选 FKL 的限制，不代表当前 top-k 主损失发生了该截断。

第二种可单独消融的目标是匹配首拒绝 token 对的 `softplus(log Q(wrong)-log Q(correct))`，可按 teacher 对这对 token 的偏好强度筛选。不要在第一轮同时加入 CE、pairwise、动态权重，避免无法判断哪项起效。

### 优先二：让评估直接衡量当前 actor 下的更新收益

用少量当前 actor 的独立 response，在同一批 teacher/features 下做更新前后对照，报告首拒绝 Q(correct)、Q(wrong)、正确/错误排序、各位置条件接受率和 KL；同时报告 response、首拒绝、suffix 的损失与梯度范数。

若希望先确认现有 checkpoint 的收益，可用同一个 actor 99 和同一批输入比较官方冷启动、P4 best85、P4_01 best25、P4_02 best70。吞吐测试应分开统计是否采集训练用 top-k 元数据。这里只建议该比较，没有运行，也没有改变双 best 保存机制。

### 优先三：单独削减 rollout 的训练元数据成本

先在固定 actor、固定请求、同一可用设备上 profile，分离 drafter proposal、actor verification、teacher top-k、元数据回传、dump 等时间。再评估仅在需要采集训练样本的轮次/请求中启用 top-k 记录，而不是每轮每个 proposal 都承担这部分成本。必须保留被选训练 trace 的完整候选数据。

这与调训练步数是不同的优化方向。即使模型接受长度不变，减少仅供训练使用的采集工作也可能提升整体速度；具体收益不能从现有三次 run 的总计时直接估算。

### 后续消融

- 若要评估更多独立样本，保留 P4 权重，仅改变采集量/更新预算，避免再次与首拒绝加权混在一起；也不要把一个 response 的更多 anchor 当成更多独立题目。
- 固定当前 actor 的评估若确认后半程更新收益消失，再测学习率衰减或基于连续评估窗口的停止更新。当前 raw grad norm 约 17–27，clip norm=1；这是裁剪前值，不能据此直接判定梯度爆炸，也不支持盲目提高 LR。
- 训练平均约 254-token prefix、rollout full prefix 的差异仍值得后续对齐消融。它是三组共同限制，当前证据不能把 P4_01/P4_02 回退归因于它；先不要同时切回 SWA/hybrid。
- 暂不优先扩大到 top-32/64、叠加 P4_01+union、进一步提高 rejected 权重或直接增加到更长训练。

## 8. 可复核产物与口径

产物目录：[p4_comparison_2026-09-08](../outputs/drafter_analysis/p4_comparison_2026-09-08)。

| 文件 | 内容 |
| --- | --- |
| `artifact_integrity.csv` | 101 steps、更新数、dump/SP 去重和实际采集量检查 |
| `rollout_and_training_steps.csv` | 三组共 303 行逐 actor step 指标 |
| `run_and_phase_summary.csv` | 统一窗口和各 20-step 阶段汇总 |
| `training_position_matrix.csv` | 每轮 ×7 位置的 response top-1、CE 和 count 指标 |
| `dump_position_matrix.csv` | 每轮 ×7 位置 ×四种 token 类别的 P/Q、NLL、K3 诊断 |
| `dump_optimizer_matrix.csv` | 每轮内部 optimizer step ×四种 token 类别的诊断 |
| `dump_cycle_stats.csv` | 每轮、每类 token 的加权/不加权统计 |
| `first_rejection_pair_matrix.csv` | 同位置匹配正确/错误 token 的逐轮逐位置比较 |
| `first_rejection_pairs_by_cycle.csv` | 匹配 token 对按轮汇总 |
| `best_checkpoints.json` | 三组两类 best 的完整原始元数据 |
| `summary.json` | 完整性、分段、选定轮次 dump 和 best 汇总 |
| `p4_*.png` / `.svg` | 可直接查看及导出的曲线、矩阵图 |

CSV 中 `p/q` 是词表原始概率，`*_weighted` 使用 dump 中的位置权重；外层 response/rejected 流权重需另计。`count_metric` 是训练日志的聚合指标，含分布式与多次更新的归约口径，不应当作独立样本数。`suffix_wrong_branch` 指首拒绝之后的原始 proposal 分支，该处单个 token 不一定是 teacher 低概率 token，也不会直接被这次首拒绝后的实际输出采用。

结果支持调整下一轮实验优先级，但三次在线 actor 训练不是同一冻结 teacher 的随机重复对照。最终提升仍应以匹配 actor、输入、设备和计时口径后的接受长度、输出吞吐及任务质量为准。
