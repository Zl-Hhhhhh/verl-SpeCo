# Drafter 块状置零与跨 rank 分叉：定位实验设计

状态：两类保存相关问题均已定位并修复；16-NPU 隔离回放和 15-step 在线压力回归均已通过。

## 执行结论（2026-09-19）

已确认一个真实且可重复的跨 DP 分叉根因：当 AdamW 尚无状态时，PyTorch DCP 的
`get_optimizer_state_dict()` 会先以 `lr=0` 隐式调用一次 `optimizer.step()` 来初始化
状态。生产 `save_checkpoint()` 只由 DP0（global rank 0–3）进入，因此第一次完整保存后，
四个 DP replica 的 Adam 状态数从一致的 `[0,0,0,0]` 变成 `[60,0,0,0]`。保存瞬间参数
仍逐字节一致；下一次真实更新因 step/bias-correction 不同而产生参数分叉。

因果对照结果：

- 原始完整保存路径重新启动 3 次，3/3 都观测到 `lr=0` hidden step 和
  `[60,0,0,0]`，严格一致性检查失败。
- 仅将“空 optimizer 的保存导出”替换为无副作用 FQN param-group 导出，重新启动
  3 次，3/3 均保持 `[0,0,0,0]`，无 hidden step；完整模型文件独立重载一致。
- 原始空状态 optimizer 目录约 8.4 GiB；修复后为 28 KiB。该 8.4 GiB 非正常状态还会
  加重原线上运行的主机内存压力。
- 正式修复在 manifest 记录 `state_initialized`。保存空状态时不初始化 Adam；冷恢复
  空状态时跳过 DCP tensor load，只恢复 trainer/scheduler 状态。旧 manifest 缺少该字段
  时仍走原兼容路径。
- 正式修复的 16-NPU 保存、独立文件重载和冷启动恢复全部通过；冷启动后 62 个检查
  shard 与原模型一致，四个 DP optimizer 状态均为空，同输入 forward 精确一致。
- 修复后 F 组合组扩展为 12 周期、每周期 5 次真实更新并轮换 4 份真实采样 batch；
  共覆盖 60 次 optimizer update、12 次 offload/alpha=0 rollback、6 次完整保存和 6 次
  独立重载。16 个 rank 全部 PASS，每 rank 37 次 SP forward 检查通过，离线跨 DP
  复核无参数或 optimizer 差异。

证据入口：

- 全组严格汇总：`outputs/drafter_analysis/integrity_20260919/summary_all.json`
- 12 周期组合组：
  `outputs/drafter_analysis/integrity_20260919/F_fixed_12cycles_4batches/`
- 修复后冷恢复：
  `outputs/drafter_analysis/integrity_20260919/empty_optimizer_cold_resume_fixed/`
- 已知损坏 checkpoint 正控：
  `outputs/drafter_analysis/integrity_20260919/known_bad_positive_control.json`

结论边界：上述证据已满足“原路径复现 → 单因素旁路消失 → 正式修复后扩展验证通过”，
因此保存导致的跨 DP optimizer 分叉已经定位并修复。其后的在线压力回归进一步暴露并
定位了另一条独立的、只影响保存 leader 的存储生命周期问题。

## 30-step 在线压力回归结果与第二项修复（2026-09-20）

2026-09-19 启动的在线回归已经正常结束（exit 0）：

- 有效实验 ID：
  `dspark_dapo2000_online_entropy_lr_integrity_fixed_30step_20260919_dapo2000_r2`
- managed run：`outputs/managed_runs/dspark_20260919T165939Z_1674575`
- 日志：
  `outputs/validation_logs/2026-09-19/dspark_vllm/dspark_dapo2000_online_entropy_lr_integrity_fixed_30step_20260919_dapo2000_r2.log`
- 固定使用 `train_unique_2000_seed17.parquet`、原 test 集、TP/SP=4、actor 参数与 optimizer
  offload、每 5 步 drafter 周期、每 10 步保存、原熵动态 LR 和 K3 记录配置。
- 运行完成 30 个 actor step；候选 drafter 在 step 5/10/15/20 均因验证证据不足而回滚，
  step 25/30 因梯度范数预检失败而拒绝，因此没有候选被发布，最终 drafter 按设计应与
  原始基线一致。
- step 5/10 的四个 SP group 内 K3 dump 精确一致。从 step 15 开始，只有 DP0/SP group
  的 global rank 0 与 rank 1–3 分叉；rank 1–3 彼此一致，其他三个 DP group 仍精确一致。
  输入身份、token 与位置相同，差异只出现在 draft log-prob、raw K3 和 K3。
- 最终 drafter 与原始文件比较发现 7 个矩阵发生变化，全部是原非零元素变成零，共
  21,950,464 个元素；置零区间是 1024 元素对齐的连续块，且只落在第一个 SP shard
  对应的全局行范围。这与 global rank 0 单点分叉完全对应。
- 因此 step 15 及之后的动态 LR/候选收益数据受模型损坏污染，不能用于评价动态 LR；
  step 5/10 仍可作为损坏前的阴性候选结果。

保存时序检查显示，global rank 0 是唯一后台 writer：它在 writer 内清空导出 payload、
执行 GC/allocator reclaim，随后主线程才把 FSDP2 live model 从 NPU offload 到 CPU；其他
rank 则先 offload live model，再 reclaim。原导出还使用
`tensor.detach().cpu().contiguous()`，当源张量已经是连续 CPU tensor 时不保证独立存储。
该 leader-only 顺序与实际唯一损坏的 global rank 0 一致，并能解释为何问题发生在 step 10
首次完整保存之后、step 15 首次可观测。

生产修复包含两项：

- full export 对每个 tensor 强制创建独立 CPU storage，避免后台 writer payload 与 live
  FSDP/offload storage 共享所有权。
- 同步完整保存时，所有 rank 统一先完成 writer，再先 offload live FSDP model，最后才
  清理 payload、GC 和 allocator/file-cache reclaim；不再由 leader 提前回收。

新增生命周期单测与既有 checkpoint/诊断测试共 31 项全部通过。修复后的 16-NPU F 组合
回放连续执行 6 个生命周期、30 次真实 optimizer update、6 次候选回滚、3 次完整保存和
3 次独立重载，16 个 rank 全部 PASS；参数 shard、optimizer 回滚、跨 DP replica 和同输入
forward 均保持一致。证据入口：

- 文件级最终损坏对比：
  `outputs/drafter_analysis/dspark_dapo2000_online_entropy_lr_integrity_fixed_30step_20260919_dapo2000_r2/postrun/final_vs_original.json`
- 修复后 6 周期回放：
  `outputs/drafter_analysis/integrity_20260920/G_reclaim_after_offload_6cycles/`

隔离证据覆盖修复路径后，继续执行了最小 15-step 在线回归，专门跨过 step 10 保存边界
并在 step 15 复查 K3：

- 实验 ID：
  `dspark_dapo2000_online_entropy_lr_reclaim_order_fixed_15step_20260920_r1`
- managed run：
  `outputs/managed_runs/dspark_20260920_reclaim_order_fixed_15step_r1`
- 日志：
  `outputs/validation_logs/2026-09-20/dspark_vllm/dspark_dapo2000_online_entropy_lr_reclaim_order_fixed_15step_20260920_r1.log`
- 运行于 2026-09-20 06:57:15Z 完成，15/15、exit 0，日志中无 traceback、RuntimeError、
  OOM 或 ERROR。
- step 5、10、15 均生成完整 16-rank K3 dump。每个 DP/SP 组内的 token 身份、位置、
  target/draft log-prob、raw K3、截断 K3、权重和掩码全部逐元素精确一致；关键的 step 15
  中 rank 0–3 仍一致，未复现 step 10 保存后的 rank0-only 分叉。
- step 10 与 step 15 完整 drafter checkpoint 各有 62 个 tensor，分别与原始 drafter
  逐元素一致，零个变化、零个缺失；两份 `pytorch_model.bin` 的 SHA-256 也相同。未出现
  原线上实验的块状置零。
- 两次保存均完整执行 optimizer/export/write/offload/reclaim，step 10 总计 14.02 秒，
  step 15 总计 13.02 秒；保存时主机 available memory 约 42–46 GiB，覆盖了预期的在线
  高内存压力窗口。
- 文件比较证据：
  `outputs/drafter_analysis/dspark_dapo2000_online_entropy_lr_reclaim_order_fixed_15step_20260920_r1/postrun/checkpoints_vs_original.json`

因此第二项保存生命周期问题已完成“旧在线路径复现 → 顺序与存储所有权修复 → 隔离回放
通过 → 同配置在线保存后 K3 与文件均保持一致”的闭环。该结论针对 checkpoint 完整性，
不等于动态 LR 策略已获得收益。

本轮动态 LR 的三个候选在 step 5/10/15 均通过梯度预检，但验证接受长度的配对置信区间
均跨零，故全部选择 alpha=0 回滚，没有发布候选。LR 始终为 `1e-5`；certainty EMA 从
0.5058 升至 0.5245，mismatch 从 7.27% 升至 10.00%，K3 saturation 从 22.27% 升至
35.93%。梯度范数分别为 9.86、9.34、8.37，没有旧损坏实验中的异常爆升。该轮证明数据
链已经可信，但尚未证明“确定性越高则提高 LR”能改善 drafter。

另有两个在 NPU 训练前终止的 setup failure：第一次误用了当前入口的 TACO 默认 profile，
第二次使用了完整 DAPO `train.parquet`。二者均在配置核对阶段停止，不计入实验结果，产物不与
上述有效 r2 目录混用。

目标是定位第一次非预期状态变化发生在哪一个操作边界，并区分：训练端参数损坏、
FSDP/运行时缓存不一致、导出 payload 损坏、文件序列化损坏，以及回滚恢复不完整。
本阶段不比较动态 LR 的收益，不运行 101 步 PPO。

## 已有证据与待检验假设

上一轮 r3 的 20 次候选更新全部回滚，但从第 25 步起 SP 组 0 的 rank 0 与
rank 1–3 对匹配 token 的 log-q 分叉；其他 SP 组一致。最终文件中 12 个矩阵
的 71,469,056 个原始非零元素变为零，差异都在前四分之一行。
这是定位线索，不能据此把首四分之一直接认定为某个 rank 的分片；必须核实
DeviceMesh、placements 和实际全局分片偏移。

优先检验四类假设：

1. FSDP full-state 收集、导出或 save_pretrained 改变了存活参数/参数存储。
2. 后台写盘与 offload/复用存在存储别名或时序问题；导出 state dict 清理、GC、
   allocator 回收、文件缓存释放暴露了错误的存储所有权。
3. Offload/reactivate 或回滚恢复了局部分片，但未恢复实际 forward 使用的全量参数、
   混合精度副本、RoPE/cache 或 optimizer 状态。
4. 输入、随机状态、target head 或分布式路径不一致造成表观输出分叉。

## 固定条件与可复现输入

- 从原始 `/workspace/models/dspark_qwen3_8b_block7` 初始化；r3 最终 drafter 只作为
  故障参考，不作为健康基线。冻结原始 Qwen3-8B actor 和 target head。
- 主实验保留原来的 16 NPU、DP=4/SP=4、FSDP2、BF16 forward/FP32 master 参数与
  optimizer、原 loss 和 K3 截断。仅使用 4 卡可能丢失 replica/leader 非对称，不能替代主实验。
- 使用同一批训练 prompt 一次性采集完整 hidden states、tokens、teacher 信息等；
  经正式 batch preparation 固化实际进入后端的输入，以后只重放，不重复生成或更新 actor。
- 每个 DP replica 初始保留至少一个真实 batch（4 个样本、2048-token 窗口），同 SP
  组沿原有分发方式接收对应输入。覆盖前段与较晚绝对位置，保留原始 position IDs，
  不把 cropped window 的位置全部改为从零开始。若小输入不复现，扩展到原来的
  每周期 4 batches 和真实 collection 上限。
- 固定 anchors、masks、teacher log-probs、target head，记录每个完整输入张量的 hash。
  对含随机 mask/noise/dropout 的路径保存并恢复 CPU/NPU RNG；比较时模式、autocast、
  shape、缓存预热条件一致。准备输入与模型 forward 都覆盖，避免只固定 prompt。
- 建立独立健康基线 B0：加载前文件摘要、加载后的 local shard 独立副本、buffer/head
  摘要、重复 forward 结果。先确认初始化权重映射正确，再判断后续操作是否损坏。
- 每个独立实验组使用全新 worker 进程和相同初始化；组内连续循环以触发累计错误。
  不能在已损坏进程中重置几个属性后继续下一组。

## 第一阶段：最小隔离矩阵

先执行以下组；每组最多 6 个生命周期周期，失败立即保存证据并结束该组。
这至少覆盖 3 次与原来“每两个 drafter 周期保存一次”等价的保存机会。
6 个周期没有复现不视为已修复；之后按下面的扩展规则处理。

| 组 | 操作 | 要隔离的问题 |
|---|---|---|
| A 重放基线 | 相同输入重复 forward；无更新、无保存、无 offload | 数值/输入/RNG 本身是否一致 |
| B Offload | forward → 参数与 optimizer offload → activate → 同输入 forward；无保存、无更新 | 设备迁移、重新激活、缓存恢复 |
| C Export-only | forward → full-state gather → 导出 payload → 释放 payload → forward；不写模型文件 | full-state 收集/导出/释放是否有副作用 |
| D 保存链 | forward → 正式 checkpoint 保存并等待完成 → forward；无更新 | 保存整体能否在零 optimizer 更新下致损 |
| E 回滚链 | baseline snapshot → 5 次真实 drafter 更新 → candidate snapshot → alpha=0 finalize → 同输入 forward；无保存 | 候选快照与参数/optimizer/缓存回滚 |
| F 原时序组合 | 原 update/cleanup/offload/rollback 路径；每两周期保存一次，再 activate/replay | 单独正常但组合出错；重现 step 20→25 类路径 |

说明：B/D 的空 optimizer 状态与 r3 的持续回滚情形相符。
E 先按该实际情形测试，再从独立的“完成一次受控更新、Adam moments 非空”的 B1 基线
重复回滚检查，避免仅验证空 optimizer 字典。B1 组只在其自身基线上比较恢复前后。
F 以代码实际调用链为准，保持 worker cleanup、coordinator finalize、保存与再次激活的
原有顺序，不按方法名称重排；启动前将真实事件顺序写入 manifest。

零更新组关闭 optimizer.step；不能用 lr=0 替代，因为那仍可能更新 Adam 状态、计数或缓存。
第一阶段不加载 vLLM rollout 服务；actor 只在一次性制作固定输入时需要，随后可释放资源。

## 第二阶段：只拆分首个失败组

如果 D/F 失败，围绕同一输入与相同调用顺序，逐项改变一个因素：

1. 权重导出与 optimizer DCP 保存分开。各自执行所需的原 process group collectives，
   不能仅让 rank 0 单独进入 collective。
2. 分别观测 payload 刚生成、后台任务提交、save_pretrained 开始/结束、state dict 清空、
   `release_checkpoint_host_memory` 前后、future 完成、再次 offload/activate 后。
3. 对比正式后台线程写盘与明确的调用线程同步写盘。`wait=True` 仍然是提交后台线程后等待，
   不能当作“没有后台线程”的独立对照。
4. 对比原导出 payload 与逐张量独立 CPU clone 的 payload。记录 live 参数、FSDP 内部副本、
   export tensor 的 storage 地址、偏移、字节范围和存活区间；地址重用本身不等于非法别名。
5. 将回收拆为 state-dict clear / Python GC / allocator reclaim / file-cache advice，
   在诊断进程中分别旁路并记录实际 allocator。文件缓存释放不是主动清零 tensor，
   不能仅因名字包含“释放”就判为根因。现有环境没有完整的“禁用回收”开关，需诊断入口实现。
6. 分别在 CPU-offloaded 与 device-resident 状态进入保存，验证 `was_on_device` 分支。

如果 B 失败：拆分参数迁移、optimizer 迁移、target head 迁移与 FSDP 重新 unshard；
检查 shard 字节是否稳定，以及首个 forward 物化后的实际参数/cache 是否变化。

如果 E 失败：在 begin/update/prepare/finalize 每个边界分别检查，覆盖空/非空 Adam 状态，
同时验证 cold reload 与原进程下一次 forward。恢复 local shard 后输出仍不同，优先查
未失效的 full-param/mixed-precision/cache 副本，而不是只看 `state_dict`。

如果所有隔离组通过：F 扩到 12 周期、每周期原来的 4 batches；仍未复现，再恢复原 actor/
rollout 显存压力和同步路径。保留原生时序的低干预版本，防止密集 CPU hash 和全设备同步
掩盖竞态。不要直接退回 101 步盲跑。

## 每个边界的证据与判据

### 参数与内存

- 同一个 rank、同一个参数的 local shard：记录原始字节 hash、dtype、shape、stride、
  placements、mesh 坐标和全局偏移。跨设备迁移字节应一致，设备地址不要求一致。
- 对所有参数做摘要；对上次 12 个受影响矩阵重点记录零值计数、非零→零计数、行/列范围、
  连续 byte 区间，以及这些区间与实际 allocator page/storage/FSDP shard 边界的关系。
- 独立基线必须 clone，避免“快照”与被测参数共用存储而一起改变。
- 不把不同 shard 坐标的 hash 不同当作错误。跨 DP replicas 也先验证 placements 与全局
  偏移相同；完整导出模型按明确的参数映射比较，共享 head/embed 的省略单独记录。

零更新生命周期和完整 alpha=0 回滚后，恢复后的参数应逐元素/逐字节相同。
任何新置零区域或参数差异均判为失败，不使用宽松浮点 tolerance 掩盖参数损坏。

### Forward、buffer 与 optimizer

- 对相同完整输入比较 drafter hidden、logits/log-q、K3 stream、loss；先跑 A 组重复
  forward 建立正常误差范围，再预先固定验收阈值。若 A 组稳定为精确一致，后续沿用该标准。
- `log-q gap > 1e-3` 保留为报警线，不以它替代基线数值误差验证；出现成块置零、明显
  rank 分叉或大于基线波动的系统偏移，应判失败。
- 同 SP 组比较完整逻辑输入一致时的可对应输出；若实际设计有序列切片，先重建或对齐
  相同位置，不能直接比较不同 slice。输入不一致时先定位输入准备路径，不归罪于模型。
- 捕获 target head、RoPE inv_freq/cos/sin、非持久 buffer、普通 tensor 属性、训练/eval
  模式与 FSDP 参数物化状态。允许预期的 cache 扩容，但检查原有有效位置与 forward 等价性。
- E 组要求回滚恢复参数、Adam exp_avg/exp_avg_sq/step、scheduler、各 param group LR、
  训练计数；明确梯度清空状态。RNG 为探针单独恢复，不假定现有 finalize 自动恢复 RNG。

### 文件与独立重载

每次保存核验：存活 local shards → 独立 full export payload → 写完后磁盘文件 →
独立新进程加载。保存结束后立即重载，不能只检查 `complete=true`。

- Live 参数先变：运行时状态损坏。
- Live 参数未变、payload 首先变：full-state/export 路径。
- Payload 正确、磁盘文件变：序列化或 writer 并发/所有权问题。
- 磁盘和重载一致，原进程输出分叉：优先查缓存/参数物化/执行模式。
- 上述都是排查分支，仍需输入、head 与 dtype 相同才能据输出差异做归因。

## 最小诊断工具补充

复用现有 `SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR`，同时通过 Ray runtime_env 传入。
现有 `state_diagnostics.py` 与分析工具已支持 local-shard hash、输入/输出、buffer 与部分
生命周期事件；但不足以直接完成此实验，需要补充：

1. 固定输入重放入口及无 optimizer 更新的 forward 捕获（当前仅训练 forward 有完整记录）。
2. begin/prepare/finalize candidate 的边界，optimizer 快照比较，以及回滚后的实际 forward。
3. 后台 writer、clear、GC、allocator/file-cache reclaim 的边界与 storage ownership 记录。
4. 逐 shard 零值变化定位，以及独立进程的导出文件重载比较。
5. 诊断 harness 对缺失/失败的证据采取严格失败判定；现有训练诊断异常会记录后继续，
   不能把没有记录解释为一致。

诊断会改变时间和内存分配。使用“两档”采集：第一档在原生时序下保留轻量事件与关键
shard 检查，第二档在已缩小的失败路径启用完整 hash/输入/输出与明确同步。
不要把全-rank barrier 插入原本仅某个 SP group 参与的保存路径；新增 collective 的
参与范围和顺序必须先校验。需要 synchronization 的检查点先等本 rank 相关操作完成。

## 停止规则、重复与交付

- 发现首次不一致后停止该组后续更新和保存。先保全每个 rank 的原始 local shard、
  buffer、输入和事件，再做任何可能改变状态的 gather/export；捕获带超时的协调退出。
- 每组目录独立；保留基线、最后一个正常边界、第一个异常边界和受影响文件，不覆盖
  原训练 run，不运行 checkpoint 自动保留/清理 watcher。
- 能复现的最小路径至少重新启动 3 次，记录 3 次中失败次数；只在关键配对组做重复，
  不对全部矩阵机械放大。单次阴性结果不宣称排除了竞态。
- 定位标准：原路径可复现 → 单一因素旁路消失 → 恢复原路径重新出现，并且中间证据
  能解释参数/输出如何变化。达不到该标准则明确记录为缩小范围，而非已找到根因。
- 本阶段产物为实验 manifest、各组 pass/fail 和首个失败事件、异常 tensor/区域、
  磁盘重载对比、因果对照结果，以及据此提出的最小修复方案。
- 资源预算按阶段给出：一次输入采集；A–F 各最多 6 周期；必要时 F 扩到 12；关键
  对照重复 3 次。先测一次初始化和保存成本再给 wall-time 估计，避免虚报几分钟完成。

通过后再做 25–30 actor 步的真实在线回归，覆盖此前第 25 步异常窗口和多次保存。
若单独诊断已复现并修复，线上回归必须检查相同状态不变量，而不仅是训练没有退出。
