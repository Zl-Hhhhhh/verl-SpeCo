# SpeCo DSpark P2 实验总结（截至 p2_04）

数据截止时间：2026-08-25 10:30 UTC。本文重点总结 DSpark online drafter 的
P2 系列。P0/P1 只作为必要背景，不展开重复描述。

TACO-LCB `p2_04` 在统计时仍在运行，已完成主训练 step 68、drafter update
cycle 22。因此 TACO-LCB `p2_04` 的结果是中期结果，不是最终 100-step
结论。DAPO `p2_04` 已完整完成 100 steps。

## 1. 结论摘要

1. P2 已经解决了“没有真实 trace、短 response 无法训练、采样偏向 batch
   前部、训练样本跨 rank 丢失”等数据链路问题。已检查的 P2 update cycle
   中，trace coverage 和 teacher-logprob coverage 均为 1.0，planned、
   materialized、dispatched、consumed 数量一致，front/middle/tail 基本为
   27/27/26。当前主要矛盾不再是采不到或训不起来。
2. 截至目前，相对相同数据集的 fixed-drafter offline 曲线，只有两个 run
   显示了可复现到逐 step 均值的接受长度提升：
   - DAPO `p2_02`：共同的 step 1-85 上平均接受长度 `3.7308 vs 3.6851`，
     绝对提升 `+0.0457`（`+1.24%`）；共同区间最后 20 steps 提升
     `+0.1504`。
   - TACO-LCB `p2_04`：截至 step 68，平均接受长度
     `3.4635 vs 3.4198`，绝对提升 `+0.0437`（`+1.28%`）；最近 20 steps
     提升 `+0.0446`。该结果尚需剩余 32 steps 确认。
3. DAPO `p2_04` 没有复现 `p2_02` 的接受长度优势：共同 step 1-85 为
   `3.6654 vs 3.6851`，差值 `-0.0197`；共同区间最后 20 steps 差值
   `-0.0081`。更低学习率、更高更新频率提升了稳定性，但没有提升 DAPO
   的最终适配效果。
4. 局部训练本身是有效的。DAPO/TACO `p2_04` 的 fixed cohort 在同一次
   update 前后，total loss、accepted K3 和 rejected K3 的平均 delta 均为
   负，top-1 accuracy 上升。问题是这种局部 teacher fitting 并不保证 fresh
   speculative rollout 的平均接受长度上升。GPQA `p2_03` 是最直接的反例：
   fixed loss 从 `3.7425` 降到 `2.8554`，但平均接受长度仍比 offline 低
   `0.1399`。
5. 目前尚未实现全程端到端加速。DAPO `p2_02` 已接近收支平衡，但同区间
   step time 仍比 offline 高 `0.88%`、总训练 throughput 低 `0.40%`。
   TACO `p2_04` 当前 step time 高 `3.95%`、throughput 低 `4.34%`，且
   generation ms/token 仍高 `2.09%`。接受长度约 `+0.04` 的收益还不足以
   稳定覆盖 hidden-state collection、drafter train 和权重发布开销。
6. 当前最突出的问题是 sampled K3 的饱和。DAPO `p2_04` rejected K3 的
   clamp ratio 平均为 `41.5%`、最后一次为 `50.2%`；TACO `p2_04` 平均
   `33.3%`、最后一次 `40.8%`。这些 token 的 K3 被硬截到 10，超过上限的
   部分不再提供有效梯度。rejected K3 的在线批次曲线上升不能简单解释为
   “优化器没学到”，但高饱和率说明当前 rejected objective 的信息利用率
   确实不足。

## 2. 统计口径与限制

本文从 console log 中逐 step 重建指标；同一 run 发生 resume 时，用后写入的
step 记录覆盖早期重复 step。主要比较指标为：

- `key_metrics/mean_acceptance_length`：当前 rollout 的平均接受长度。
- `timing_per_token_ms/gen`：比绝对 generation time 更能抵消 response 长度变化。
- `perf/throughput` 和 `timing_s/step`：包含 actor 训练在内的端到端训练指标。
- `drafter/fixed_cohort/delta_*`：同一次 drafter update 前后、同一 cohort 上的
  配对变化，用来判断 update 本身是否沿目标方向移动。
- `drafter/train/dspark_*`：当前在线训练批次的损失分量，样本分布随 actor 和
  drafter 演化，不应把其跨 cycle 绝对趋势当作固定验证曲线。

比较存在以下限制：

- DAPO offline 基线是 8 月 12 日的旧 run，日志只完整记录到 step 85，并且
  代码版本早于最终 P2。DAPO 的所有 offline 差值只使用共同 step 1-85。
- TACO offline 是当前代码附近完整完成的 100-step run，和 TACO `p2_04` 的
  可比性更强；但 `p2_04` 当前只到 step 68。
- online 和 offline 并非严格共享每一步的生成 token、actor 参数和随机采样
  轨迹。逐 step 配对能减少训练阶段差异，但不能替代同 seed、同初始 checkpoint
  的受控 A/B 重跑。
- sequential steps 存在自相关。本文的提升值是描述性结果，不将普通 IID
  置信区间作为最终显著性证明。

## 3. 从 P0/P1 到 P2

P0 的主要贡献是恢复真实 speculative trace、支持短 response、按
front/middle/tail 分层选择窗口、在 collection cap 前做 hash-priority 选择，
并把 restricted CE 换成 exact chunked CE。P1 将 anchor 限制为真实 verify
event 起点，区分 accepted prefix、first rejection 和未验证后缀，并加入 fixed
cohort、unique-first sampling 和 acceptance regression guard。

P2 的核心变化是利用完整 rejected draft branch，并逐步向 Draft-OPD 的 sampled
K3 objective 靠拢。P2 系列实际包含多个不同训练配方，不能只按 run 名后缀把它们
当作同一个实验：

| 版本 | 主要 objective | update 配置 | 主要现象 |
| --- | --- | --- | --- |
| `p2` | accepted FKL `1.0` + L1 `0.1` + rejected K3 `0.25` | 5 drafter steps / 5 actor steps，LR `5e-6` | rejected K3 权重偏弱，DAPO 无稳定优势 |
| `p2_01` | accepted FKL `1.0` + L1 `0.1` + rejected K3 `1.0` | 10/5，LR `1e-5` | 每 cycle 约 50% reuse；DAPO step 21 出现接受长度骤降 `1.445` |
| DAPO `p2_02` | response K3 `1.0` + rejected K3 stream `0.5` | 5/5，LR `1e-5` | 当前 DAPO 最佳；reuse 除首 cycle 外为 0，但 step 51 曾触发一次 `0.754` collapse guard，resume 后完成 |
| TACO `p2_02` | 双 K3，早期 rejected stream 为 `1.0` | 20/5，LR `1e-5` | 每 cycle 约 75% reuse，接受长度低于 offline |
| `p2_03` | 双 K3，rejected stream `0.5` | 8/5，LR `1e-5`，每 cycle 约 124 samples | reuse 已低，但更新偏强；TACO 在 step 51 和 91 两次 collapse，GPQA 完成但效果低于 offline |
| `p2_04` | 双 K3，rejected stream `0.5` | 5/3，LR `5e-6`，每 cycle 80 samples | 无 collapse，DAPO 稳定但无提升；TACO 当前有小幅提升 |

这里最有价值的消融结论不是“样本越多越好”，而是：高 reuse、高 LR、单 cycle
过多 optimizer steps 的组合明显更危险。`p2_04` 每次只做 5 steps、LR 减半、
样本 unique-first，稳定性明显优于 `p2_01`/`p2_03`。

## 4. 当前 p2_04 损失函数

当前实际配置不是旧版 P2 文档中的 accepted FKL + L1。优化权重为：

```text
CE = 0
L1 = 0
accepted FKL = 0
accepted/response K3 = 1.0
rejected K3 = 1.0
response stream weight = 1.0
rejected stream weight = 0.5
rejected suffix position decay = 0.8
```

对于 token `y`：

```text
delta = clamp(log p_target(y) - log q_drafter(y), -20, 20)
K3(y) = min(exp(delta) - delta - 1, 10)
```

response stream 包含真实 target trajectory 上的 accepted prefix、first-rejection
位置的 target replacement，以及全接受 verify event 的 response bonus position。
rejected stream 包含实际递归 draft branch 上从 first rejection 开始的整个后缀，
第 `j` 个 rejected suffix 位置使用 `0.8^j` 权重。实际 objective 是 joint-token
normalization：

```text
L = (sum_response K3 + 0.5 * sum_rejected weighted_K3)
    / (N_response + 0.5 * N_rejected_weighted)
```

所以 `train_loss` 不是 `accepted_k3 + 0.5 * rejected_k3` 两个均值的简单相加。
CE、accepted FKL、top-1/top-5 仍被计算用于诊断，但 CE/FKL 不参与反向优化；
L1 在当前配置中不计算。accepted K3 需要 `dflash_aux_plus_last` 提供 target final
hidden state；当前 batch 已传入 `target_last_hidden_states`，bonus token 也已经进入
response K3。

另有一个配置注释不一致：common script 注释称 rejected suffix decay 为 `0.9`，
实际默认值和已运行日志均为 `0.8`。报告中的分析以实际 `0.8` 为准。

## 5. 接受长度结果

### 5.1 与 offline 的同 step 对比

| 数据集 / run | 可比区间 | online | offline | 差值 | 最近共同 20 steps 差值 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DAPO `p2_02` | 1-85 | 3.7308 | 3.6851 | **+0.0457** | **+0.1504** | 当前 DAPO 最佳，但发生过一次 guard abort |
| DAPO `p2_04` | 1-85 | 3.6654 | 3.6851 | -0.0197 | -0.0081 | 稳定，但未超过 offline |
| TACO `p2_01` | 1-100 | 3.3237 | 3.4671 | -0.1434 | - | 高 reuse、更新过强 |
| TACO `p2_02` | 1-84 | 3.3147 | 3.4470 | -0.1323 | - | 20-step cycle 和 75% reuse 无收益 |
| TACO `p2_03` | 1-90 | 3.3571 | 3.4546 | -0.0976 | - | 两次训练崩溃，不应作为正向结果 |
| TACO `p2_04` | 1-68 | 3.4635 | 3.4198 | **+0.0437** | **+0.0446** | 当前唯一 TACO 正向结果，尚未完成 |
| GPQA `p2_03` | 1-100 | 3.1030 | 3.2429 | -0.1399 | -0.1389 | 明显低于 offline |

DAPO `p2_02` 在共同 85 steps 中有约 68% 的 step 高于 offline；TACO `p2_04`
当前约 88% 的共同 step 高于 offline。DAPO `p2_04` 只有约 47% 的共同 step 高于
offline。这与曲线目视判断一致。

### 5.2 接受长度提升仍然较小

两个正向结果的平均增益都约为 `+0.04 token/verify`，相对增幅约 `1.2%-1.3%`。
这个增幅能说明 online drafter 已经可能改善真实 speculative behavior，但还不足以
单独证明部署价值。特别是 TACO `p2_04`，当前提升从中段开始保持稳定，却没有转化为
更低的 generation ms/token。

## 6. Drafter loss 分布与训练质量

### 6.1 主要 run

| run | update cycles | train loss 均值 | p10-p90 | first -> last | accepted K3 first -> last | rejected K3 first -> last | grad norm mean / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DAPO `p2_02` | 20 | 3.096 | 2.877-3.356 | 3.648 -> 2.997 | 3.257 -> 1.966 | 4.452 -> 5.970 | 7.35 / 9.15 |
| DAPO `p2_04` | 33 | 3.079 | 2.947-3.333 | 3.501 -> 2.961 | 3.309 -> 2.179 | 4.059 -> 6.006 | 8.24 / 10.12 |
| TACO `p2_03` | 18 | 3.232 | 3.016-3.533 | 3.276 -> 4.218 | 2.966 -> 3.095 | 3.884 -> 6.416 | 18.23 / 32.71 |
| TACO `p2_04`（step 68） | 22 | 3.142 | 3.061-3.213 | 3.264 -> 3.181 | 3.136 -> 2.558 | 3.637 -> 5.310 | 7.52 / 9.62 |

`p2_04` 的 total train loss 分布明显比 `p2_03` 稳定。TACO `p2_03` 的最后
train loss 和 grad norm 同时恶化，并对应真实接受长度 collapse；TACO `p2_04`
目前没有这个模式。

### 6.2 为什么 rejected K3 一直上升

在线训练 batch 的 rejected K3 上升，并不与“每次更新能降低 loss”矛盾：

1. 每个 cycle 都从最新 actor/drafter 的 fresh rollout 重新收集 rejected branch。
   随着容易 token 被改善，剩余 rejection 会偏向更难、target/drafter 概率差更大的
   token；跨 cycle 的 batch 分布不是固定的。
2. fixed cohort 的配对结果显示，同一次 optimizer update 确实在降低 rejected K3：
   DAPO `p2_02` 平均 delta `-0.0303`，DAPO `p2_04` 为 `-0.0144`，TACO
   `p2_04` 为 `-0.0269`。
3. 但 rejected batch 变难的速度超过了每次更新的改善幅度。DAPO `p2_04` 的
   `teacher - draft` log-ratio 均值从 `-3.01` 漂到 `-8.84`；TACO `p2_04`
   从 `-2.18` 漂到 `-7.20`。
4. K3 的 hard max=10 进一步造成饱和。DAPO `p2_04` rejected clamp ratio
   平均/最后为 `41.5%/50.2%`，TACO 为 `33.3%/40.8%`。hard clamp 之上的
   token 对应梯度为零，恰好是最难的一批 rejection。

因此，rejected K3 上升主要说明 fresh negative distribution 持续变难；同时，
当前 hard cap 让训练无法充分利用其中相当大的一部分。不能只靠把 rejected stream
weight 从 1.0 降到 0.5 解决这个问题，因为 stream weight 只缩放整体贡献，不恢复
被 clamp token 的梯度。

### 6.3 fixed cohort 证明了“会拟合”，没有证明“会提接受率”

DAPO `p2_04` fixed loss 从 `3.420` 降到 `2.678`，top-1 从 `0.558` 升到
`0.650`；TACO `p2_04` 从 `2.943` 降到 `2.289`，top-1 从 `0.574` 升到
`0.638`。平均单次 update 的 fixed delta 如下：

| run | delta total loss | delta accepted K3 | delta rejected K3 | delta top-1 |
| --- | ---: | ---: | ---: | ---: |
| DAPO `p2_02` | -0.0499 | -0.0580 | -0.0303 | +0.0047 |
| DAPO `p2_04` | -0.0255 | -0.0295 | -0.0144 | +0.0031 |
| TACO `p2_04` | -0.0310 | -0.0326 | -0.0269 | +0.0030 |

这些结果说明梯度、optimizer 和权重发布链路工作正常。`p2_04` 的单次 update
幅度大约只有 `p2_02` 的一半，符合 LR 从 `1e-5` 降到 `5e-6`。但 fixed cohort
保留的是固定输入特征，跨 actor cycle 时 target LM head 仍会变化；它更适合检查
“本次 update 是否正确”，不适合作为真实 speculative acceptance 的替代指标。

### 6.4 后部 draft position 仍明显较弱

`p2_04` 的 diagnostic CE/top-1 虽不参与优化，但清晰展示了 block 内难度分布：

| 数据集 | position 0 loss / acc | position 3 loss / acc | position 6 loss / acc | position 0/6 平均样本数 |
| --- | ---: | ---: | ---: | ---: |
| DAPO | 1.138 / 0.747 | 2.018 / 0.630 | 2.737 / 0.519 | 2047 / 358 |
| TACO | 1.186 / 0.719 | 2.319 / 0.556 | 3.265 / 0.433 | 2047 / 268 |

越靠后的 token 样本越少、准确率越低，而 accepted length 的进一步提升恰好依赖
后部位置。当前 rejected suffix 又使用 `0.8^j` 衰减，会继续降低后部 rejected
token 的有效权重。这可能是局部 loss 能下降、平均接受长度却只能提升约 0.04 的
原因之一。

## 7. 训练吞吐与端到端速度

### 7.1 同 step offline 对比

| run | 接受长度 | gen ms/token | throughput | step time | old-logprob time |
| --- | ---: | ---: | ---: | ---: | ---: |
| DAPO `p2_02` vs offline | +1.24% | **-3.70%** | -0.40% | +0.88% | +9.97%（+1.81 s） |
| DAPO `p2_04` vs offline | -0.53% | **-2.34%** | -1.73% | +1.61% | +14.73%（+2.68 s） |
| TACO `p2_04` vs offline | +1.28% | +2.09% | -4.34% | +3.95% | +16.51%（+3.62 s） |

粗体的 generation ms/token 下降是好方向。DAPO `p2_02` 后半段已有推测解码
收益，但全程 throughput 仍只接近持平。DAPO `p2_04` 的 generation 指标虽较低，
接受长度并未高于 offline，且 response 长度和 actor trajectory 不同，不能把这
部分差异完全归因于 drafter。TACO `p2_04` 当前三个端到端速度指标均未超过
offline。

### 7.2 Online drafter 的直接开销

| run | 每个 update cycle 的 `timing_s/drafter` | 累计 drafter / 总 step wall time |
| --- | ---: | ---: |
| DAPO `p2_02` | 19.71 s | 1.03% |
| DAPO `p2_04` | 13.53 s | 1.15% |
| TACO `p2_04`（step 68） | 约 15.3 s | 约 1.3% |

这还没有包含 old-logprob 阶段为了收集 hidden state 带来的额外时间。`p2_04`
collection step 的 old-logprob 平均约比普通 step 高 5-7 秒；按每 3 steps 收集一次
折算，又增加约 1.8-2.3 秒/actor step。综合来看，当前接受长度需要比 offline
稳定提升更多，才可能覆盖约 2%-3% 的 online drafter 直接及间接开销。

结论是：P2 已经接近“在线训练不显著拖慢”的工程目标，但还没有达到“在线训练使
整个 RL run 更快”。

## 8. 已暴露的训练问题

### 8.1 接受长度 collapse

已记录的 DSpark P2 事故包括：

- DAPO `p2_01`：update step 20 后，step 21 下降 `1.445`。
- DAPO `p2_02`：update step 50 后，step 51 下降 `0.754`，runtime guard
  终止；从较早 checkpoint resume 后最终完成。
- TACO `p2_03`：step 51 下降 `1.553`，resume 后 step 91 再下降 `1.359`。

`p2_04` 没有出现类似事故。DAPO 最大 guard drop 为 `0.183`，TACO 当前最大
为约 `0.070`。LR 减半、5-step update 和低 reuse 是当前最可信的稳定性改进。

### 8.2 Loss 与 acceptance 的评价错位

所有 P2 run，包括效果为负的 GPQA，fixed loss 都能下降。这表明 current fixed
cohort 主要衡量局部 target matching，而 acceptance 还受以下因素影响：

- recursive draft prefix 的 exposure error；
- actor 在 RL 中持续变化造成的 target drift；
- fresh rollout 的 rejection 分布随 drafter 改变；
- `<think>` 长推理段和 final answer/code 段的数据分布差异；
- 后部 block position 的长尾样本不足。

目前没有按 thinking/final、accepted-prefix length、全接受/首拒绝、response
位置分桶的 fresh deterministic acceptance eval，因此还不能准确定位 TACO 有效而
GPQA 无效的原因。

### 8.3 Thinking-mode domain mismatch

当前 actor 生成的是 thinking-mode 长轨迹，而初始 drafter 的训练分布更接近普通
无 thinking 的生成。DAPO/TACO 的 response 平均约 5k-6k tokens，GPQA 只有约
1k tokens；同一套 512-token window 和每 cycle 80/124 samples 对这些数据集代表
的覆盖完全不同。

TACO `p2_04` 的改善可能来自 code/reasoning 局部模式更稳定，不能据此推断所有
thinking workload 都会改善。GPQA 中“fixed loss 下降但 acceptance 持续下降”说明
仅对局部 token distribution 做 K3 仍不能解决 thinking trajectory 的递归分布偏移。

### 8.4 指标与配置文档仍有漂移

旧 `p2_drafter_training_changes.md` 记录的是 accepted FKL + L1 的早期 P2，不能
代表 `p2_02` 之后的实际 objective。common script 的 suffix decay 注释写 0.9，
实际运行是 0.8。后续实验必须把生效配置随 run 固化，否则相同 `p2_*` 名称容易被
误读。

## 9. 下一步建议

1. **先完成 TACO `p2_04`，不在中途改参数。** 重点确认 step 80-100 是否仍
   保持约 `+0.04`，以及 generation ms/token 是否随接受长度继续上升而下降。
2. **做真正同口径的 A/B。** 从同一个 actor checkpoint、同 seed、同代码版本
   分叉 fixed-drafter offline 和 online。DAPO 复跑 `p2_02` 配方，TACO 复跑
   `p2_04` 配方；否则无法把 actor/response 长度差异和 drafter 收益彻底分开。
3. **补 fresh deterministic speculative eval。** 每 5 或 10 actor steps 用固定
   prompt、固定 RNG/temperature、冻结的 actor snapshot 分别评估 online drafter
   和初始 drafter。至少按 `<think>`、final answer/code、draft position、accepted
   prefix length 分桶。
4. **优先处理 K3 saturation，而不是简单继续加样本。** 先记录 unclamped K3 和
   log-ratio 的 p50/p90/p99，再对 hard max=10 做消融：平滑 cap、只对 loss value
   截断但保留受控梯度，或提高 cap 并配合 per-stream gradient norm。当前 33%-50%
   rejected token 被截断，继续增加同类样本的边际价值有限。
5. **提高后部 position 的有效监督。** 比较 rejected decay `0.8`、Draft-OPD
   复现所称的 `0.9` 和不衰减；同时按 position 做采样配额或 inverse-frequency
   weighting。目标应直接观察 position 4-6 accuracy/K3 和 acceptance-length tail，
   而不只看总 loss。
6. **维持 p2_04 的低 reuse 和稳定更新强度。** 已有结果不支持恢复 10/20-step
   cycle 或 50%-75% reuse。若增加 samples，应保持 optimizer token budget 不变，
   用更多 unique samples 替换重复 draw，而不是同时增加训练步数。
7. **把速度验收改成分解式门槛。** 至少同时要求：接受长度高于 offline、
   generation ms/token 下降、端到端 throughput 不低于 offline，并单独报告 hidden
   collection、drafter RPC、target-head sync、publish/update weights 时间。当前只有
   接受长度提升不能判定部署成功。

## 10. 主要数据源

- DAPO offline：`outputs/validation_logs/2026-08-12/dspark_vllm/qwen3_8b_dspark_offline_fixed_150step_b64.log`
- DAPO `p2_02` 首次/恢复：
  `outputs/validation_logs/2026-08-21/dspark_vllm/qwen3_8b_dspark_online_w16_100step_b64__p2_02.log`、
  `outputs/validation_logs/2026-08-22/dspark_vllm/qwen3_8b_dspark_online_w16_100step_b64__p2_02.log`
- DAPO `p2_04`：`outputs/validation_logs/2026-08-24/dspark_vllm/qwen3_8b_dspark_online_w16_100step_b64__p2_04.log`
- TACO offline：`outputs/validation_logs/2026-08-23/dspark_vllm/qwen3_8b_dspark_taco_lcb_offline_fixed_100step_b64.log`
- TACO `p2_03`：`outputs/validation_logs/2026-08-22/dspark_vllm/qwen3_8b_dspark_taco_lcb_online_w16_100step_b64__p2_03.log`、
  `outputs/validation_logs/2026-08-23/dspark_vllm/qwen3_8b_dspark_taco_lcb_online_w16_100step_b64__p2_03.log`
- TACO `p2_04`：`outputs/validation_logs/2026-08-25/dspark_vllm/qwen3_8b_dspark_taco_lcb_online_w16_100step_b64__p2_04.log`
- GPQA online/offline：`outputs/validation_logs/2026-08-23/dspark_vllm/qwen3_8b_dspark_wildsci_gpqa_online_w16_100step_b64__p2_03.log`、
  `outputs/validation_logs/2026-08-24/dspark_vllm/qwen3_8b_dspark_wildsci_gpqa_offline_fixed_100step_b64.log`
- Collapse 记录：`outputs/drafter_acceptance_incidents.jsonl`
