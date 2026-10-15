# Frozen actor drafter LR comparison

采用两个冻结 actor，分别比较原始 DSpark drafter 的 `5e-6 / 1e-5 / 1.5e-5`。
每档 LR 从相同权重和空 Adam 状态开始，连续训练 20 个 optimizer steps，在 5、20 步评估。
实验不会调用 actor 更新；单个 upstream step 仅用于 rollout、提取 teacher 隐藏状态和收集训练数据。

Actor 来源：

- step 50：`outputs/checkpoints/dspark_dapo2000_offline_k3_every5_front512_60step_20260912/global_step_50/actor`
- step 100：`outputs/checkpoints/dspark_dapo2000_offline_k3_entropy_101step_20260916_r2/global_step_100/actor`

两者来自不同 run，主要推断应是同一 actor 内 LR 的作用。跨 actor 结果不能单独归因为训练步数。
原 checkpoint 不变；合并的 HF 权重在 `outputs/frozen_actor_lr_20260921/actor50` 和 `actor100`。

## 设计

- 原 DAPO seed-17 训练集，batch 64 × 5、response 上限 8192；TP/SP=4，16 NPU。
- actor 参数与 optimizer offload 都为 true；drafter 使用原 K3 截断、普通 anchor，无 asymmetric K3。
- 每个 actor 只收集一批 rollout。预构造 20 个 batch，保存每个 rank 的 tensor 与 SHA256；
  三档 LR 顺序复用相同 batch，固定 RNG。5 → 20 步保留参数、Adam moments 和累计步数。
  actor50 缓存审计显示每个 rank 的 20 份 batch hash 相同：这里是对固定数据重复更新，
  不是 20 批独立 rollout。报告会列出 distinct batch 数量，不能把训练 loss 下降视为泛化提升。
- 每档结束恢复 baseline，比较本地权重、optimizer、scheduler 和计数器；不一致则终止。
- 留出集使用原 test 的 30 个独立题，检查与本次训练 prompt 零重叠。
- quality：greedy、512-token cap；丢弃一次预热，然后做 A/A；每个候选做 ABBA。
- load：每档 LR 的 20-step 候选再做 ABBA，320 条并发、greedy、2048-token cap。
  这会重复留出题以模拟并发，不能将 320 条当作 320 个独立样本。
- 逐响应保存 token IDs、长度、hash、verification entropy 和接受轨迹；断言 cap 实际生效。
- 保存全部 6 份候选 drafter state dict，记录训练耗时、发布耗时和 rollout 耗时。
- entropy LR 开关仅用于复用精确 optimizer 回滚与留出集工具；本实验 LR 固定，不执行动态升降或候选淘汰。

## 指标与判断

比较接受长度、配对逐题接受长度变化及区间、低熵确定性占比、低熵首次拒绝率、
accepted/rejected K3、饱和与 log-ratio 截断造成的零梯度比例、梯度范数。
检查 K3 dump 的同 SP 组跨 rank 一致性，以及候选保存/回滚的一致性。

吞吐分别报告 30 题短探针和 320 并发负载。ABBA 对称消除一阶顺序偏差，
A/A 用于判断噪声；用每 token 节约时间计算摊销训练和发布成本所需 token 数。
完整 8192-token、随机采样的实际训练收益仍需后续验证；不从短探针直接声称全程加速。
30 个独立题只支持筛选，单次训练随机种子的结果不能直接证明普适最优 LR。

## 启动及产物

当时使用的一次性启动脚本已归档至
`outputs/experiment_setup/archived_one_off_launchers_20260925.tar.gz`；两个 actor 依次运行，失败则停止。
默认 managed run：`dspark_frozen_actor_lr_20260921_r5`。
首次 r1 在初始化期间主动停止，补齐 vLLM 私有参数清理后以 r2 重启；r1 无 drafter 更新。
r2 完成训练 rollout 与实际 512-token 探针的长度检查，但在探针结果合并时遇到
重复 acceptance 字段；同时发现 trainer.rank 是 SP 组内 rank，诊断文件会重名。
两处修正后以 r3 重启。r2 无 drafter optimizer 更新，不能用其同名 batch 文件回放全 rank。
r3 完成 16 个全局 rank 的 batch 保存和 A/A 基线，但在首档更新前因非全局主 rank
不导出完整权重而触发过严的快照检查。已改为仅要求全局主 rank 的导出非空。
r3 的清理过程耗时较长且出现 subgroup barrier 超时，之后流程确实继续，不能据此定性为死锁。
本实验改用已有的无子组 barrier 的模型/optimizer offload 路径，由协调器等待全部 RPC 完成。
r4/r5 的 actor50 直接使用 r3 的固定 batch：逐 rank 核对原始 SHA256，同时核对 actor 来源和留出题；
不重复采集或写入这些大文件。actor100 仍执行首次 rollout 和完整新数据采集。
actor50 的原始 verification entropy 和 batch 位于 r3/actor50，worker 记录包含具体引用路径。
r4 在首个 forward 前发现缓存路径未初始化 `_current_pad_size`，没有执行 optimizer 更新。
已按每个缓存 batch 的 metadata 设置 padding 状态，并补充缓存恢复回归测试。
重新启动前，使用真实 16 NPU、相同缓存完成 5 + 15 次 optimizer 更新、两次候选导出，
全部 16 个 rank 的权重、Adam、scheduler 和步数均精确恢复。
预检查证据位于 `outputs/frozen_actor_lr_20260921/npu_preflight/worker_XX.jsonl`。
该预检查不包含 vLLM rollout，其训练指标不能替代正式留出集结果。

输出：`outputs/frozen_actor_lr_20260921/r5/actor{50,100}/`：

- `results.jsonl`：实验设计、每次 probe、训练和恢复结果。
- `worker_XX.jsonl`：batch hash、每个 rank 的训练指标、恢复校验。
- `batches_rank_XX.pt`：实际固定训练 batch，可复现回放。
- `drafter_lr_*_step_*.pt`：CPU clone 的完整 drafter publish state dict。
- `*.json.gz`：每次 probe 的完整 token/entropy/acceptance 记录。
- `k3_logprobs/`：每个 optimizer step、每个 rank 的 K3 样本。

CPU 验证：22 个相关测试通过；新 Python 进程实际反序列化 Ray worker，
验证 512 cap、skip-collection、global step 传递及上下文清理通过。
vLLM 边界兼容修复另通过 53 项相关测试（含两种参数传递方式下真实 SamplingParams 构造）。
r3 新增及扩展的 5 项 CPU 测试通过，包括全局 rank 文件命名、探针结果合并与 cap，
以及真实 Adam 更新下完整三档 LR 的 5/20-step、39 次探针调用、3 次精确恢复流程。
r4 扩展至 6 项测试，加入非全局导出 rank 和严格 hash 校验的缓存恢复；实际 rank0 的 20 个缓存 batch
也逐一通过 SHA256 校验。

r3 A/A 初测（无 drafter optimizer 更新）：接受长度 3.4706 / 3.5370，30 题配对变化均值
0.0715，筛选区间 [-0.0250, 0.1679]；仅 6/30 的完整响应 hash 相同。两次实际最长响应均为 512。
这些是测量噪声证据，不是训练收益。
