# Frozen actor first-rejection CE comparison

冻结 actor100，drafter LR 固定为 `1e-5`，比较
`dspark_first_rejection_ce_alpha = 0 / 0.05 / 0.1`。
每档从相同权重、空 Adam、scheduler 和 RNG 状态开始，累计更新 20 个 optimizer steps，
在 5、20 步进行 greedy 512-token ABBA；20 步额外进行 320 响应、2048-token ABBA。

训练数据复用 `outputs/frozen_actor_lr_20260921/r5/actor100` 的 16-rank 缓存，
逐 rank 核对 SHA256。每 rank 的 20 个 optimizer 输入是同一个固定 batch，
所以这是损失配方筛选，不是独立在线数据或泛化证明。

辅助项只监督 trace 中真实首次拒绝位置的 actor replacement token：

`L = L_original + alpha * mean(first-rejection CE)`

它使用完整词表 CE 并独立按首次拒绝 token 归一化；原 accepted/rejected K3、普通 anchor、
`log(p/q)` 的 `[-20, 20]` 截断和 K3 上限 10 均保持不变，asymmetric K3 关闭。
actor 参数和 optimizer offload 均开启，actor 更新路径禁用。

默认 managed run：`dspark_frozen_actor100_ce_20260921_r1`。
正式输出：`outputs/frozen_actor_ce_20260921/r1/actor100`。

启动前验证：17 项辅助 CE、固定数据和协调器相关测试通过。真实 16 NPU 预检查使用
`alpha=0.1 / LR=1e-5` 完成 5+15 次更新；16 个 rank 的权重、Adam、scheduler 和步数
全部精确恢复，所有记录指标有限。前 5 步累计首次拒绝 CE 为 4.8717，20 步累计为
2.9850，20 步累计梯度范数为 6.4257。预检查记录位于
`outputs/frozen_actor_ce_20260921/npu_preflight_r2/`。
