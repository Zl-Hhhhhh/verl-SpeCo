# SpeCo DSpark P3 阶段实验总结（截至 p3_06）

数据截止时间：2026-09-04 05:05 UTC。本文汇总 DAPO 上的 P3、相关冷启动、
full/SWA 注意力消融、K3 策略修改、采样与训练频率实验。原始统计同时使用了
`outputs/validation_logs`、`outputs/drafter_analysis`、drafter checkpoint 元数据和
W&B 日志；其中 K3 二进制 dump 来自 `outputs/drafter_analysis`。

## 1. 当前结论

1. **没有一个 P3 新策略明确超过原始 `p3`。** 原始 `p3` 的最后 20 steps
   平均接受长度为 `3.9122`，仍是官方 DSpark 冷启动 P3 中最高。`p3_04`、
   `p3_05_restart0`、`p3_06_swa512` 分别为 `3.8659`、`3.8165`、`3.7603`。
   所以目前不能把 full asymmetric K3、动态流权重、更激进的数据采集或 SWA
   训练判定为正向改进。
2. **历史 rollout 注意力消融不能用于判断真正的 SWA。** 固定 checkpoint、
   DAPO 128 条样本的旧实验中，full、标记为 SWA-512、SWA-1024 的平均接受长度
   分别为 `3.8723`、`3.8696`、`3.8642`。但旧路径只写入 draft model config，
   没有设置 vLLM-Ascend 的 `additional_config.draft_window_size`；在 eager、
   noncausal 路径中并未创建 block-table 裁剪器，rollout 很可能仍读取完整 prefix。
   因此这些结果只能说明旧 overlay 没有提速，不能证明真实 SWA 与 full 等价。
3. **只对 rejected stream 使用 asymmetric K3 是合理方向；对 accepted stream
   也开启 asymmetric K3 风险较高。** full-asymmetric run 的 accepted K3 均值
   从 rejected-only 组约 `2.1` 提升到 `4.5-4.8`，约 19% accepted token 进入
   正尾线性化区，梯度和 fixed-cohort 拟合更强，但 rollout 接受长度更差。
4. **`rejected_position_decay=0.5` 与当前问题方向相反。** position 6 的 rejected
   token 原始权重只有 position 0 的 `1/64`；但 position 6 的样本数仅约
   position 0 的 `17%-18%`，准确率更低，legacy clamp/exceed 比例更高。后部
   draft position 正是提高平均接受长度所需的位置，不应被如此强地降权。
5. **增加独立样本没有解决问题。** `p3_06` 将每 cycle 的日志 unique samples
   从约 `8` 提升到 `23.8`，reuse 从约 `61%-62%` 降到 `0.8%`，但最后 20
   steps 接受长度仍只有 `3.7603`。`p3_05_restart0` 每 cycle 收集约 124 条并
   每 10 actor steps 训练 10 steps，也没有优于 5/5 的原始 `p3`。
6. **训练链路有效，但优化目标与 rollout 指标没有对齐。** 所有主要 run 的
   fixed-cohort loss 都下降、top-1 通常上升；下降幅度更大的 run 并没有更好的
   fresh rollout acceptance。因此当前重点应从“让 loss 降得更多”转为修正
   position weighting、控制 rejected 尾部梯度，并使用严格单变量 A/B。

## 2. 比较口径与证据等级

主要指标：

- `key_metrics/mean_acceptance_length`：每个 verify event 的平均接受长度，是
  drafter 质量主指标。
- `timing_per_token_ms/gen`：生成阶段平均耗时，能部分抵消输出长度不同。
- `perf/throughput`：整个 RL step 的 throughput，包含 actor 训练、数据收集、
  drafter 训练和发布开销，不能当作纯 speculative decoding 吞吐。
- `drafter/fixed_cohort/delta_*`：同一次 update 前后在固定 cohort 上的配对变化，
  用于确认 optimizer 和权重发布是否有效。
- `drafter/train/dspark_*` 与 K3 dumps：用于分析 accepted/rejected 分布、clamp、
  position 权重和梯度，不是固定验证集曲线。

证据强度从高到低为：

1. 修正 runtime adapter 后，同一 frozen actor/drafter checkpoint 的 full/SWA
   重复评测；
2. 同一初始化、只改一个变量的 online run；
3. 同时改变训练策略、采样量、训练频率或 attention 的 online run；
4. 不同 actor snapshot 上的单步 throughput 或 dual-best checkpoint 元数据。

限制：online run 很快进入不同的 actor/drafter 轨迹，后期曲线只能作描述性比较；
`p3_04` 的当前 console log 缺少 step 1-20 的 acceptance，因此其全程均值只统计
step 21-101；废弃的旧 `p3_05` 只有约 40 steps，主结论使用完整重跑
`p3_05_restart0`。`p3_swa1024` 同时改变了 attention 和 K3 配方，不能用来单独
估计 SWA-1024 的因果效果。更重要的是，历史 SWA online run 的训练 mask 生效，
但 Ascend rollout 端缺少 `SlidingWindowAdapter`，存在 train/serve context 不一致。

## 3. Run 配置矩阵

P3 默认 actor 为 `/workspace/models/Qwen3-8B`，drafter 为官方
`/workspace/models/dspark_qwen3_8b_block7`。除表中注明外，LR 为 `1e-5`，
accepted/rejected K3 alpha 均为 `1.0`，response stream weight 为 `1.0`，
block size 为 7。

| run | attention | K3 尾部策略 | dynamic weight | rejected weight / decay | drafter steps / actor interval | 采集特征 |
| --- | --- | --- | --- | --- | --- | --- |
| `p3` | 训练 SWA-512；旧 rollout 疑似 full | accepted/rejected 均为 hard clamp | off | `0.5 / 0.8` | `5 / 5` | 16K token budget，训练端 unique 约 `7.75`/cycle |
| `p3_01` | full | full asymmetric | off | `0.5 / 0.8` | `5 / 5` | 16K，unique `8.20` |
| `p3_02` | full | rejected-only asymmetric | on | `0.5 / 0.8` | `5 / 5` | 16K，unique `8.10` |
| `p3_03` | full | full asymmetric | on | `1.0 / 0.8` | `5 / 5` | 16K，unique `8.00` |
| `p3_swa1024` | 训练 SWA-1024；旧 rollout 疑似 full | full asymmetric | off | `0.5 / 0.8` | `5 / 5` | 16K，unique `8.15` |
| `p3_04` | full | rejected-only asymmetric | on | `0.5 / 0.5` | `5 / 5` | 16K，unique `8.10` |
| `p3_05_restart0` | full | rejected-only asymmetric | on | `0.5 / 0.5` | `10 / 10` | 每 cycle 收集 `124.2`，unique `30.3` |
| `p3_06_swa512` | 训练 SWA-512；旧 rollout 疑似 full | rejected-only asymmetric | on | `0.5 / 0.5` | `6 / 5` | 每 cycle 收集 `96.0`，unique `23.8`，reuse `0.8%` |

这里的 unique 是训练日志中的平均 unique sample 数，不等同于 rollout 中生成的
原始 response 数。早期 run 虽然每 cycle 收集约 32 条，但 2048-token window 与
16K token budget 使单个训练 owner 实际只能装入约 8 条，然后在 20 个 batch draw
中产生约 `61%-63%` reuse。

## 4. Online 结果

| run | 全程 acceptance | step 1-20 | 21-40 | 41-60 | 61-80 | 最后 20 | gen ms/token | RL throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p3` | 3.7698 | 3.3928 | 3.6324 | 3.9361 | **3.9753** | **3.9122** | **0.1591** | 303.25 |
| `p3_01` | 3.6762 | 3.4265 | 3.7569 | 3.8339 | 3.7824 | 3.5813 | 0.1561 | 307.31 |
| `p3_02` | 3.6971 | 3.4171 | 3.6692 | 3.7409 | 3.8683 | 3.7903 | 0.1617 | 294.80 |
| `p3_03` | 3.5493 | 3.4141 | 3.6364 | 3.6043 | 3.5707 | 3.5210 | 0.1823 | 271.60 |
| `p3_swa1024` | 3.6570 | 3.3978 | 3.6108 | 3.6638 | 3.7917 | 3.8209 | 0.1780 | 273.76 |
| `p3_04` | 3.8097* | - | 3.6328 | 3.8466 | 3.8897 | 3.8659 | 0.1617 | 298.03 |
| `p3_05_restart0` | 3.7253 | 3.4150 | 3.6342 | 3.8779 | 3.8781 | 3.8165 | 0.1570 | 304.35 |
| `p3_06_swa512` | 3.6906 | 3.4013 | 3.6394 | 3.7895 | 3.8575 | 3.7603 | 0.1686 | 283.37 |

`* p3_04` 的全程值只覆盖日志可恢复的 step 21-101。不同 run 的 response 长度、
reward、actor clip 和生成轨迹不同，`gen ms/token` 与 RL throughput 不宜用于小幅
横向排名。表中最可靠的现象是：新策略没有形成持续、明显高于 `p3` 的 acceptance
平台；`p3_01` 和 `p3_03` 的 full asymmetric 配方后期下降尤其明显。

`p3_05_restart0` 与 `p3_06` 在首次 drafter update 前 4 steps 的 acceptance 分别
约为 `3.3409` 和 `3.3338`，只差 `-0.0071`，也没有显示 SWA-512 对官方 checkpoint
产生明显的即时质量损伤。后续差异已混入训练频率、样本量和 actor 轨迹变化。

## 5. 历史 fixed-checkpoint config-only 消融

本节保留原始测量值用于审计，但这些标为 SWA 的 arm 没有启用 Ascend
block-table adapter，不应再作为真实 full/SWA 因果比较。

### 5.1 2048-token response cap，三次重复

来源：
`outputs/drafter_analysis/dspark_attn_dapo128_20260830_repeated/aggregate.csv`。
三组使用相同 frozen checkpoint、DAPO validation 128 条、固定 262144 个有效输出
token。

| attention | repeats | mean acceptance | run 间 stdev | generation time (s) | response tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 3 | **3.8723** | 0.0135 | **52.225** | **5023.44** |
| SWA-512 | 3 | 3.8696 | 0.0120 | 53.685 | 4887.98 |
| SWA-1024 | 3 | 3.8642 | 0.0104 | 53.488 | 4901.86 |

full 与标记为 SWA-512 的 acceptance 差仅 `0.0026`，数值上远小于单组 run 间
标准差；但由于没有证据表明 KV 实际被裁剪，不能把它解释为上下文窗口不影响质量。
它没有产生加速也与“实际仍为 full”一致。

### 5.2 8192-token quality run

来源：
`outputs/drafter_analysis/dspark_attn_dapo128_quality8192_20260830/summary.csv`。

| attention | verify | accepted tokens | mean acceptance | generation (s) | output tokens | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 237035 | 730345 | 4.0812 | **177.66** | 967241 | **5444.40** |
| SWA-512 | 236072 | 734098 | **4.1096** | 184.28 | 970004 | 5263.79 |
| SWA-1024 | 240830 | 731922 | 4.0392 | 187.81 | 972601 | 5178.54 |

标记为 SWA-512 的 arm 在这一轮 acceptance 比 full 高 `0.0285`，但速度低
`3.32%`；三个 arm
生成的 token/text hash 不同，这不是严格 paired 输出上的延迟比较，单次结果不足以
证明质量提升。另一个单次 R1 含 SWA-2048：full/SWA-512/SWA-1024/SWA-2048
acceptance 为 `3.8583/3.8569/3.8742/3.8264`，同样没有随窗口大小单调变化。

### 5.3 修正前后 SWA 的具体语义

训练时，每个 draft query 看到：

```text
最近 W 个 actor-context token + 当前 7-token draft block 内的双向全连接
```

五个 drafter 层都使用相同窗口，没有 sink token，也没有 global layer。训练端会
遮掉原始 actor prefix 的更早 token，同时保留 draft block 内非因果 attention。

旧 runtime 只设置了 `dflash_config.use_swa/swa_window_size`。vLLM-Ascend 的 eager
noncausal kernel 会走 full sparse mode，而 proposer 又因缺少
`additional_config.draft_window_size` 没有创建 `SlidingWindowAdapter`，所以历史
rollout 很可能没有裁掉 prefix。

修正后的 public `dspark_context_window=W` 始终表示 W 个 actor-context token；runtime
配置和 block-table adapter 使用 `W + K`，其中 K 是完整 draft block（当前为 7）。
adapter 先把 KV/block table 裁成“最近 W 个 actor token + K 个 draft query”，再运行
noncausal attention，因此既保留 block 内双向可见性，也与训练 mask 对齐。proposer
启动后会检查 model window、`draft_window_size` 和 adapter window 三者相等；原生
multi-group DSpark 还会对全部 drafter KV group 应用同一裁剪范围并检查补丁存在，
否则直接失败。由于 paged KV 必须按 block 起点裁剪，runtime 最多会额外保留
`block_size - 1` 个更老 actor token，这是剩余的有界边界差异。

Hybrid（4 local + 1 global）和 sparse-global 已实现为可选训练模式，但当前 vLLM
在线 worker 不支持混合 KV-cache group，因此没有进入这些 P3 在线 run，也不应把它们
当成已验证方案。attention sink/hybrid 应排在修正后 uniform SWA 的配对复测之后；
loss/position 修正仍可独立推进。

## 6. K3 策略与 dump 分析

### 6.1 asymmetric K3 的实际行为

基础 sampled K3 为：

```text
delta = clamp(log p_target(y) - log q_drafter(y), -20, 20)
raw_k3 = exp(delta) - delta - 1
```

旧策略对 `raw_k3` hard clamp 到 10，超限 token 梯度为 0。asymmetric 策略对
`delta > 2` 使用切线线性延拓，对负尾保留原始 K3；它不再 hard clamp，所以 legacy
超限 token 仍有梯度。日志中的 `linearized_ratio` 只统计正尾切线区，不代表所有
恢复梯度的 token，因为大量 rejected 超限来自负尾。

动态流权重按 response/rejected token 数量计算近似平衡乘子并 clip 到 `[0.25, 4]`，
再乘 base stream weight 和 position decay。它能平衡流的数量，但不会自动保证各
draft position 的有效梯度质量。

### 6.2 dump 汇总

`p3` 早于 dump 功能，没有二进制 K3 dump。其余完整 run 的 dump 数量为：
`p3_01/02/03/swa1024/p3_04/p3_05_restart0` 各 1600 个，`p3_06` 为 1920 个；
废弃旧 `p3_05` 的 640 个 dump 不用于主结论。

| run | stream | post-policy K3 mean | legacy exceed | positive linearized | raw position weight mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `p3_01` | accepted | 4.681 | 15.7% | 19.5% | 1.000 |
|  | rejected | 8.968 | 43.1% | 6.6% | 0.478 |
| `p3_02` | accepted | 2.091 | 17.8% | 0.0% | 1.000 |
|  | rejected | 8.872 | 42.0% | 7.0% | 0.478 |
| `p3_03` | accepted | 4.790 | 15.9% | 19.6% | 1.000 |
|  | rejected | 9.066 | 43.9% | 6.3% | 0.480 |
| `p3_swa1024` | accepted | 4.532 | 15.1% | 18.6% | 1.000 |
|  | rejected | 8.808 | 42.1% | 7.0% | 0.477 |
| `p3_04` | accepted | 2.093 | 17.8% | 0.0% | 1.000 |
|  | rejected | 8.759 | 41.2% | 7.3% | **0.170** |
| `p3_05_restart0` | accepted | 2.396 | 20.2% | 0.0% | 1.000 |
|  | rejected | 9.611 | 44.0% | 9.4% | **0.169** |
| `p3_06` | accepted | 2.184 | 17.8% | 0.0% | 1.000 |
|  | rejected | 9.052 | 42.7% | 7.6% | **0.172** |

关键解释：

- full asymmetric 把 accepted stream 的难正尾也放大，accepted K3 大约翻倍，
  fixed-cohort loss 降得更多，但没有转化成 acceptance；accepted hard clamp 暂时应
  保留。
- rejected stream 有 `41%-44%` token 超过旧 hard cap，说明 rejected-only
  asymmetric 有充分动机；其中只有 `6%-9%` 是正尾线性化，其余大量超限来自负尾。
- decay 从 `0.8` 降到 `0.5` 后，rejected raw position weight mean 从约 `0.478`
  降到约 `0.17`。动态 rejected multiplier 后期虽升到约 `3.2-3.6`，仍主要强化
  早期位置，无法抵消 position 6 的 `0.5^6=0.015625` 权重。
- rejected legacy-exceed 从早期约 `28%-33%` 普遍增长到晚期约 `48%-51%`，与
  attention 模式、采样量无关。这说明 fresh rejection 分布持续变难，当前目标仍在
  追逐不断后移的难例尾部。

raw K3 分布有极大离群值，accepted raw p99 可达 `1e5` 量级，因此不能用 raw mean
比较 run；应优先看 median、quantile、legacy-exceed、position 分布和 post-policy
梯度。

## 7. Position、采样与更新强度

主要 run 的 draft-position 诊断均显示相同结构：

| run | position 0 -> 6 accuracy | position 0 -> 6 平均样本数 | fixed loss delta | fixed top-1 delta |
| --- | --- | --- | ---: | ---: |
| `p3` | 0.785 -> 0.575 | 2028 -> 375 | -0.0440 | +0.0051 |
| `p3_04` | 0.778 -> 0.556 | 2021 -> 372 | -0.0459 | +0.0040 |
| `p3_05_restart0` | 0.753 -> 0.538 | 2047 -> 356 | -0.1548 | +0.0145 |
| `p3_06` | 0.762 -> 0.559 | 2046 -> 343 | -0.0561 | +0.0054 |

`p3_05_restart0` 用更大、更新更稀疏的 cycle，产生最强 fixed-cohort 改善，却没有
更好的 rollout acceptance。这表明 10/10 容易对一个较旧 cohort 做过多局部拟合。
`p3_06` 几乎消除 reuse，仍未改善后期 acceptance，说明早期 run 的 2048 window
确实冗余、unique 样本偏少，但它不是当前质量瓶颈。

梯度稳定性也没有随新策略单调改善：`p3_swa1024` 最后一次 grad norm 约
`467.8`，`p3_06` 为 `58.8`，而 `p3_04` 为 `5.17`。因此下一轮不应同时继续增加
每 cycle optimizer steps 和尾部权重。

## 8. 冷启动实验

相关冷启动分为两类，不能混在一起：

- `cold_p2_05/p2_06`：actor 从 Qwen3-8B 重新开始，但 drafter 使用 P2_02 的
  `draft_step_101` 做 weights-only 初始化，不恢复旧 optimizer state。`p2_05`
  采用 5/5、LR `5e-6`，多次在 step 60/90 附近中断；`p2_06` 改为 3/10、
  LR `2e-6`、candidate gate 后完成 100 steps。
- P3：actor 从 Qwen3-8B、drafter 从官方 `dspark_qwen3_8b_block7` 开始，属于
  官方 checkpoint 冷启动，初始 drafter 与 P2 cold run 不同。

完整的 `cold_p2_06` 全程/最后 20 acceptance 为 `3.7660/3.9242`，最后 20 略高于
`p3` 的 `3.9122`，但这不能归因于其在线策略，因为它继承了 P2_02 已训练 drafter。
其 fixed-cohort 平均变化接近零（loss `-0.001`、top-1 `-0.001`），且 late grad norm
出现约 `494` 的离群值。该配置更像“保守维持已有 drafter”，不是高效的在线适配
方案。

## 9. Dual-best checkpoint

`p3_04` 以后同时保存 best acceptance 与 best throughput：

| run | best acceptance checkpoint | acceptance / response tok/s | best throughput checkpoint | acceptance / response tok/s |
| --- | --- | --- | --- | --- |
| `p3_04` | step 70 | 3.9795 / 6065.3 | step 90 | 3.8458 / 7479.5 |
| `p3_05_restart0` | step 70 | 3.9437 / 6461.6 | step 80 | 3.8245 / 7929.7 |
| `p3_06` | step 70 | 3.9357 / 7213.1 | step 70 | 3.9357 / 7213.1 |

这些指标来自各自 rollout step 和对应 actor snapshot，适合在单个 run 内防止只保留
末尾 checkpoint，不适合跨 run 直接选模型。最终选择仍需在同一个 frozen actor
snapshot、同一 DAPO cohort 上统一复测；当前按此前决定只保留两份，不自动复测。

## 10. 下一阶段优先级

### P0：先修 position weighting，保持其余配置不变

以 `p3_04` 的 rejected-only asymmetric、5/5、full attention 为基线，只做
`rejected_position_decay = 0.5 / 0.8 / 1.0` 单变量对照。至少先验证 `0.8`；从现有
证据看，`0.5` 不应继续作为默认候选。主观察量除 acceptance 外，还应包括每个
position 的有效权重、accuracy、legacy exceed 和样本数。

### P1：约束动态 rejected multiplier

在选定 position decay 后，再单独比较 dynamic off 与 dynamic on。若保留 dynamic，
建议测试将 rejected multiplier 上限从 4 收紧到 2，或按施加 position decay 后的
有效权重质量做平衡。不要与 full asymmetric 同时修改。accepted stream 继续使用
hard-clamped K3。

### P2：使用中等更新强度与足够 unique 样本

回到 `5 drafter steps / 5 actor steps`。每 cycle 让训练端约有 20 个左右 unique
样本、基本无 reuse 即可；现有 `p3_06` 已证明继续推到 24-30 个 unique 或改成
10/10 没有明显收益。先把额外算力用于重复 seed，而不是单 cycle 更多 optimizer
steps。

### P3：attention 仅做严格配对验证

质量优化阶段先固定 full，不再同时改变 attention 和 loss。若目标是部署加速，应使用
已修正并带启动校验的 runtime，在完全相同 checkpoint、输出长度桶和策略下重新比较
full/SWA-512。旧实验没有验证 KV 裁剪，不能用于预测真实 SWA 的速度或质量；
attention sink、hybrid、sparse-global 暂不进入下一轮在线主实验。

下一轮判断成功不应只看某个 peak：至少比较共同 step 的最后 20-step acceptance、
两次以上 seed、post-update acceptance drop、position 4-6 指标和 generation
ms/token。fixed-cohort loss 只作为训练链路健康检查，不再作为模型优劣主排序。

## 11. 可复核数据位置

- 固定 checkpoint 注意力汇总：
  `outputs/drafter_analysis/dspark_attn_dapo128_20260830_repeated/`
- 8192-token attention quality run：
  `outputs/drafter_analysis/dspark_attn_dapo128_quality8192_20260830/`
- 含 SWA-2048 的单次结果：
  `outputs/drafter_analysis/dspark_attn_dapo128_20260829_r1/`
- P3 K3 dumps：
  `outputs/drafter_analysis/dspark_online_100step__p3_01/`、
  `dspark_online_100step__p3_02_rejonly_asym_dynw/`、
  `dspark_online_100step__p3_03_full_asym_dynw/`、
  `dspark_online_100step__p3_swa1024/`、
  `dspark_online_101step__p3_04/`、`dspark_online_101step__p3_05_restart0/`、
  `dspark_online_101step__p3_06_swa512/`
- 完整运行日志：`outputs/validation_logs/2026-08-26` 至 `2026-09-03`；
  `p3_06` 于 2026-09-04 UTC 完成，但沿用启动日的日志目录
- Dual-best metadata：`outputs/drafters/dspark_online_101step__p3_04/`、
  `p3_05_restart0/`、`p3_06_swa512/`
