# Ascend 910 实验环境迁移清单

记录日期：2026-09-08。目标：在另一台同类 Ascend 910 机器上继续 Qwen3-8B + DSpark 的 DAPO 实验。本文以当前 P4_03 为参数基准，优先保持全局训练预算，复用现有启动器；没有创建新的逐实验脚本。

这是一份当前环境的实测清单。已核对文件、依赖和启动配置；尚未在目标机器完成运行验证。

## 1. 先确认目标硬件和系统

- [ ] 在目标机执行下列只读命令，确认 CPU 架构、系统、驱动、设备映射、HBM、主存和共享内存。
- [ ] 确认与本机一样是 **A3 系列**。仅显示“Ascend910”不足以保证算子 ABI 相同，A2/910B 不能直接套用本机 A3 编译产物。
- [ ] 优先使用相同基础镜像、同样的绝对路径；宿主机驱动由目标机器提供，不复制旧机器的驱动目录或设备文件。
- [ ] 本文的扩卡例子是**单机 16 个逻辑设备**，不是多机训练配置。

```bash
uname -m
cat /etc/os-release
npu-smi info
npu-smi info -m
cat /usr/local/Ascend/driver/version.info
free -h
df -h /workspace /dev/shm
```

源机器实测：

| 项目 | 当前值 |
| --- | --- |
| CPU / 系统 | aarch64 / Ubuntu 22.04.5 LTS |
| 驱动 | 26.1.0，ascendhal 7.35.23 |
| NPU 编译目标 | vLLM-Ascend 的 `__device_type__='A3'`；算子目录 `ascend910_93` |
| 物理 / 逻辑映射 | 8 张物理卡，每张 2 个逻辑设备，共 16 个；单逻辑设备约 64 GB HBM |
| 当前允许使用 | 物理 4–7，对应逻辑 8–15，共 8 个逻辑设备 |
| 主存 / /dev/shm 总量 | 约 1 TiB / 503 GiB；是源机容量，不是推导出的最低要求 |
| Python | 3.12.13，实际解释器在 `/usr/local/python3.12.13/bin/python3.12` |
| Python 环境 | `/workspace/main_env_next` |
| 实验运行时 | 隔离安装的 CANN 9.0.1 + ATB 9.0.1 |
| model runner | `VLLM_USE_V2_MODEL_RUNNER=0`，使用 V1 路径 |

actor 参数和 optimizer 有 CPU offload，因此增加卡数时也要留意主存、共享内存和存储带宽。不要只按总 HBM 判断是否能运行。

## 2. 代码迁移：使用 experiment 分支

- [ ] 克隆自己的 fork 的 `experiment` 分支。
- [ ] 核对基准提交，而不是克隆 fork 的 `main` 后直接实验。
- [ ] 不复制原训练目录的 `.git`、暂存区、后台进程状态或临时 worktree。
- [ ] 本清单与依赖清单如果尚未推送，另行复制到新机器。

```bash
# 目标机；/workspace/verl-SpeCo 应是尚不存在的新目录。
git clone --branch experiment --single-branch \
  https://github.com/Zl-Hhhhhh/verl-SpeCo.git /workspace/verl-SpeCo
git -C /workspace/verl-SpeCo rev-parse HEAD
git -C /workspace/verl-SpeCo status --short
```

迁移基准提交：`eb0a3f0af21c6e206756a0ee4da4d13210080cde`
（Snapshot DSpark online training experiments through P4_03）。

此提交包含截至 P4_03 的损失、注意力、双 best 保存、分析工具和实验文档。源训练目录仍显示旧 `main` 加本地改动，是此前刻意保留的工作区状态；这些训练代码已在 `experiment` 快照中。未来分支继续前进时，保留本次实际使用的 commit SHA 即可。

## 3. 必须独立带走的文件

Git 不包含 Python 环境、CANN、自定义算子、模型、DAPO Parquet 和实验输出。

| 源路径 | 实测占用约 | 用途 / 是否必需 |
| --- | ---: | --- |
| `/workspace/main_env_next/` | 3.6 GiB | 必需；包含 site-packages、运行时激活脚本、已安装的自定义算子与备用 wheel |
| `/usr/local/python3.12.13/` | 3.5 GiB | 同路径复制方式必需；venv 的 Python 是指向这里的绝对软链接 |
| `/workspace/cann901_validation/runtime/` | 9.9 GiB | 必需；CANN 9.0.1、NNAL/ATB 及软链接 |
| `/workspace/models/Qwen3-8B/` | 18 GiB | 必需；actor 全部权重、配置、tokenizer |
| `/workspace/models/dspark_qwen3_8b_block7/` | 4.5 GiB | 必需；官方 drafter 冷启动权重和配置 |
| `/workspace/data/dapo_math_17k_verl/` | 218 MiB | 必需；保留现有 train/test，不用同名数据重新加工后代替 |
| `/workspace/cann901_validation/packages/` | 约 4 GiB | 建议；CANN toolkit、A3 ops、NNAL 安装包及 triton-ascend wheel |
| `/workspace/cann901_validation/src/vllm-v0260/` | 433 MiB | 建议；vLLM 源码备份 |
| `/workspace/cann901_validation/src/vllm-ascend-v0260rc/` | 654 MiB | 建议；包含本地构建补丁的源码 |
| `/workspace/cann901_validation/release-src/` | 476 MiB | 建议；另存的构建源码 |
| `outputs/drafter_analysis/`、`outputs/validation_logs/` | 按所选 run | 复盘历史实验时复制；新冷启动不依赖它们 |
| `outputs/drafters/`、`outputs/checkpoints/` | 按所选 run | 用已有 drafter 或恢复 run 时才需要 |

必需目录合计约 40 GiB，不含代码、历史输出、目标机已有系统文件及新训练产物。还要为 actor checkpoint、双 best、临时保存峰值和逐步日志预留额外空间；开启 `DSPARK_DUMP_K3_LOGPROBS` 后输出会持续增长。

DAPO 无需搬 SandboxFusion。只有以后做代码题时，才另带 `/workspace/SandboxFusion`、相关服务及 `/workspace/data/drafter_domains/reward.py`。

### 为什么不能只按 requirements 重装

关键依赖如下，完整已安装包版本见 [environment_packages_2026-09-08.txt](environment_packages_2026-09-08.txt)。该文件是审计清单，**不是可直接安装的 lockfile**。

| 包 | 已安装版本 |
| --- | --- |
| torch / torch-npu | 2.10.0 / 2.10.0.post2 |
| vllm | 0.26.0+empty |
| vllm-ascend | 0.1.dev1+ge592b2e79.d20260811 |
| verl / verl-speco | 0.8.0.dev0 / 0.1.0 editable |
| transformers / ray | 5.14.1 / 2.56.1 |
| triton / triton-ascend | 3.5.0 / 3.2.1 |
| datasets / pyarrow / wandb | 5.0.1 / 25.0.0 / 0.28.1 |

实测发现以下仓库外状态：

- wheel 实际在 `/workspace/main_env_next/artifacts/baseline/`。安装元数据的旧路径 `/workspace/main_env_next_artifacts/baseline/` 已不存在，不能照抄 `pip freeze` 中的 file URL。
- 对照 baseline wheel，vLLM 的 2035 个 Python 文件一致；已安装 verl 的两个 SandboxFusion 文件有差异；vLLM-Ascend 自定义算子下两个 Python 文件有差异。因此 wheel 不能替代整个已安装环境。
- 现在已安装的算子目录有 39 个 A3 kernel 子目录。历史验证记录描述过“最小算子包”，不能用它代表当前安装状态；39 个目录本身也不证明所有算子都验证过。迁移应以当前 `site-packages/vllm_ascend/_cann_ops_custom` 为准。
- 构建源码 `src/vllm-ascend-v0260rc` 的 `setup.py` 与 `csrc/cmake/scripts/util/ascendc_ops_config.py` 有本地修改；源码 SHA 为 `e592b2e79697868550ab40e4486b9ced330bb4a9`。vLLM 源码 SHA 为 `568afb3a13806beb53bb2e6bd518269357b237c0`。
- 若要重建而非复制，应保留以上源码、补丁、wheel 和当前算子树，再核对目标 SoC 与 CANN ABI；没有一条已验证的“纯 pip 从零重建当前环境”命令。

## 4. 同架构、同基础系统的复制步骤

以下命令是供迁移时执行的模板；本次整理没有进行远程复制。推荐目标机使用专属的新容器/目录，保留源绝对路径。若目标已有同名环境，先选择隔离位置或镜像，不直接覆盖在用环境。

- [ ] 在目标机创建目录，由管理员准备 `/usr/local/python3.12.13` 的写入权限。
- [ ] 将 `SOURCE_HOST` 替换为 SSH 源主机别名；命令在目标机执行。
- [ ] 使用 `rsync -aH` 保留软链接。若模型是指向目录外的软链接，还需带走其指向的权重文件。
- [ ] 驱动、内核模块、设备映射由目标宿主机/容器平台配置；不要 rsync `/usr/local/Ascend/driver`。
- [ ] 不复制 SSH 私钥、全量 home、W&B API key 文件或源机代理配置。

```bash
mkdir -p /workspace/models /workspace/data /workspace/cann901_validation
rsync -aH --info=progress2 SOURCE_HOST:/workspace/main_env_next/ /workspace/main_env_next/
rsync -aH --info=progress2 SOURCE_HOST:/usr/local/python3.12.13/ /usr/local/python3.12.13/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/cann901_validation/runtime/ /workspace/cann901_validation/runtime/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/models/Qwen3-8B/ /workspace/models/Qwen3-8B/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/models/dspark_qwen3_8b_block7/ /workspace/models/dspark_qwen3_8b_block7/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/data/dapo_math_17k_verl/ /workspace/data/dapo_math_17k_verl/
```

建议同时备份的构建材料：

```bash
mkdir -p /workspace/cann901_validation/src
rsync -aH --info=progress2 SOURCE_HOST:/workspace/cann901_validation/packages/ /workspace/cann901_validation/packages/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/cann901_validation/src/vllm-v0260/ /workspace/cann901_validation/src/vllm-v0260/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/cann901_validation/src/vllm-ascend-v0260rc/ /workspace/cann901_validation/src/vllm-ascend-v0260rc/
rsync -aH --info=progress2 SOURCE_HOST:/workspace/cann901_validation/release-src/ /workspace/cann901_validation/release-src/
rsync -aH SOURCE_HOST:/workspace/cann901_validation/DSpark_V1_A3_validation_notes.md /workspace/cann901_validation/
```

若目标是不同发行版或基础镜像，目录复制后仍可能缺少系统动态库。优先恢复相同基础镜像；不要把“同为 Ascend 910”当作二进制环境可迁移的充分条件。

### 激活与 Python 路径

运行时入口是：

```bash
source /workspace/main_env_next/bin/activate-speco-cann901-v1.sh
/workspace/main_env_next/bin/python --version
readlink -f /workspace/main_env_next/bin/python
```

该脚本会加载隔离 CANN、ATB 和自定义算子，并设置 `VLLM_PLUGINS=ascend`、`VLLM_USE_V2_MODEL_RUNNER=0`。不要只 source 系统的 `/usr/local/Ascend/ascend-toolkit/set_env.sh`；本机那是另一套旧运行时。

复制后用目标机已安装的 `uv` 重新绑定项目 editable 安装；以下操作只安装当前项目，不升级 Torch/vLLM 等依赖：

```bash
cd /workspace/verl-SpeCo
uv pip install --python /workspace/main_env_next/bin/python --no-deps --no-build-isolation -e .
```

目标机需预先提供 `uv`。源机二进制在 `/root/.local/bin/uv`，不是 venv 自带工具；若复制该工具，仅复制二进制，不复制整个 home。

路径无法保持时，需要一起检查激活脚本、venv Python 链接、CANN 的 `set_env.sh`、ATB/custom-op 环境及动态库依赖。仅设置 `DSPARK_PYTHON` 或改 venv 名称不够，CANN 当前脚本内部也写入了绝对安装路径。

## 5. 迁移后轻量检查，不跑训练烟测

- [ ] 解释器、包路径、运行时路径正确，能导入主要模块。
- [ ] NPU 逻辑编号及独占使用范围确认；随后所有训练命令显式设置设备。
- [ ] W&B 在目标机通过自己的账号交互登录；新实验使用新标签、新 run ID。
- [ ] 命令预览中的 full / top16 / CE 0.2 / 101 step / 双 best 配置正确。
- [ ] 以上检查只是环境预检，不代表完整训练链路已验证。

```bash
source /workspace/main_env_next/bin/activate-speco-cann901-v1.sh
/workspace/main_env_next/bin/python -c 'import torch, torch_npu, vllm, vllm_ascend, verl, verl_speco; print(torch.__version__); print(vllm.__version__); print(verl.__file__); print(verl_speco.__file__)'
/workspace/main_env_next/bin/python -c 'from vllm_ascend._build_info import __device_type__; print(__device_type__)'
sha256sum /workspace/data/dapo_math_17k_verl/train.parquet /workspace/data/dapo_math_17k_verl/test.parquet
/workspace/main_env_next/bin/wandb login
```

本次 DAPO 文件校验值：

```text
train.parquet  cdb223011baebcc1bd600d563b7c0b9559610df96c9ca33acc8556a70fa38b89
test.parquet   a94a87fd700142538db1d0e089e81ca98be477c4208163301adb66aab5979bb6
```

历史依赖标签 `verl==0.8.0.dev0` 与项目要求的 release 标签存在版本提示，不应仅为了去掉提示而换包。遇到导入错误先核对解释器、CANN、自定义算子和动态库，不盲目升级整套依赖。

## 6. 更多卡时保持当前 P4_03 的训练预算

在当前 TP=4、rollout DP=1、PP=1 的设置下，独立 rollout/drafter 数据副本数为：

`副本数 = 逻辑设备数 / GEN_TP_SIZE`

一个 TP 组共享同一批样本，不能把 8 个 worker 都算成独立采样副本。

| 参数 / 预算 | 当前 8 逻辑设备 | 目标 16 逻辑设备 |
| --- | ---: | ---: |
| `N_GPUS` / `DRAFT_WORKER_WORLD_SIZE` | 8 / 8 | 16 / 16 |
| `GEN_TP_SIZE` / `TRAIN_SP_SIZE` | 4 / 4 | 4 / 4 |
| 独立副本数 | 2 | 4 |
| `DRAFT_BATCH_SIZE_PER_GPU` | 8 | 4 |
| `DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA` | 48 | 24 |
| `DRAFT_MAX_COLLECT_TOKENS_PER_REPLICA` | 32768 | 16384 |
| `DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA` | 8 | 4 |
| 全局 drafter batch / 单次 optimizer update | 16 | 16 |
| 每轮全局采集上限 / token cap | 96 / 65536 | 96 / 65536 |
| 全局固定 holdout 数 | 16 | 16 |
| 每轮 drafter optimizer step | 6 | 6 |
| collect / training interval | 5 / 5 | 5 / 5 |
| 每条训练样本的 anchor 上限 | 32 | 32 |
| actor step / actor batch / rollout n | 101 / 64 / 5 | 101 / 64 / 5 |

这保持预算相同，不保证随机轨迹或训练效果逐位相同：副本分配、采样顺序和浮点归约会变化。采集数是上限，实际仍受可用响应、有效 anchor、token cap 和过滤影响。保留 LR=1e-5，不因 worker 翻倍而机械翻倍 LR。

101 个 actor step 按正常节奏在 5、10、…、100 做 20 轮 drafter 更新，每轮 6 个 optimizer step，共 120 次；失败、提前停止或无可训练 batch 时以实际完成计数为准。首次建立固定 holdout 后可用于训练的新样本最多约 80，后续轮上限 96；每轮最多 96 次样本抽取，是否重复取决于实际可训练样本数。

当前是 **full attention + 512 token 的训练采集裁剪**，并非 SWA-512。rollout drafter 使用 full 请求 prefix；训练 query 看裁剪样本内的 actor prefix，draft block 内仍为双向。两个 attention mode 都要显式设为 full 才能保持此基准。

更多卡若仍使用每副本 48、batch 8，则全局采集上限变为 192、batch 32，这属于新的训练预算实验。建议先按表保预算迁移，再单独试更多数据。

### 16 逻辑设备的完整命令

半节点及 P4 wrapper 已于 2026-09-09 删除。新实验统一使用通用入口 `tools/run_dspark_vllm_experiment.sh`，显式设置设备数、数据集与注意力配置。下方命令保留本次迁移的 16 设备配置。

在新 shell / tmux 会话使用以下命令。设备列表是新机全部 16 个逻辑设备均空闲时的例子；若只获得 8 个，按上表恢复 8 设备参数并填写实际逻辑编号。新标签 `p4_03_migrate16` 可改为自己的命名；不要与旧 run 复用 ID 或 checkpoint 目录。可额外设置自己的 `WANDB_ENTITY`。

```bash
(
  cd /workspace/verl-SpeCo
  unset WANDB_RUN_ID WANDB_RESUME CHECKPOINT_DIR EXPERIMENT_ID
  unset DRAFTER_INIT_LABEL SPECO_MANAGED_RUN_ID SPECO_DRAFTER_K3_DUMP_DIR
  export SPECO_ONLINE_STRATEGY_TAG=p4_03_migrate16
  export DSPARK_PYTHON=/workspace/main_env_next/bin/python
  export SPECO_RUNTIME_SETUP=/workspace/main_env_next/bin/activate-speco-cann901-v1.sh
  export MODEL_PATH=/workspace/models/Qwen3-8B
  export DRAFTER_PATH=/workspace/models/dspark_qwen3_8b_block7
  export TRAIN_FILE=/workspace/data/dapo_math_17k_verl/train.parquet
  export TEST_FILE=/workspace/data/dapo_math_17k_verl/test.parquet
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  export N_GPUS=16 DRAFT_WORKER_WORLD_SIZE=16 GEN_TP_SIZE=4 TRAIN_SP_SIZE=4
  export SPECO_SKIP_GLOBAL_RAY_STOP=1 RAY_ADDRESS=local SPECO_WAIT_FOR_HOST_LOCK=1
  export RUN_MODE=online DATASET_PROFILE=dapo
  export TOTAL_STEPS=101 SAVE_FREQ=10 SAVE_OFFSET=9 ENABLE_CHECKPOINTS=1
  export TRAIN_BATCH_SIZE=64 ROLLOUT_N=5 PPO_MINI_BATCH_SIZE=16
  export PPO_MICRO_BATCH_SIZE_PER_GPU=10
  export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=8192
  export MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=16384 GPU_MEMORY_UTILIZATION=0.5
  export DRAFTER_INIT_CHECKPOINT= DRAFTER_INIT_IS_STRONG=false
  export DRAFT_TRAIN_STEPS=6 DRAFT_COLLECT_INTERVAL_STEPS=5 DRAFT_TRAINING_INTERVAL_STEPS=5
  export DRAFT_BATCH_SIZE_PER_GPU=4 DRAFT_TRAIN_BATCHES_PER_CYCLE=4
  export DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA=24
  export DRAFT_MAX_COLLECT_TOKENS_PER_REPLICA=16384
  export DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA=4 DRAFT_TRAIN_UNIQUE_SAMPLES_FIRST=true
  export DRAFT_LR=1e-5 DRAFT_LR_SCHEDULER_TYPE=constant DRAFT_LR_WARMUP_STEPS=0
  export DRAFT_LR_DECAY_STEPS=100 DRAFT_PUBLISH_ASYNC=false DRAFT_PUBLISH_INTERVAL_STEPS=0
  export DSPARK_RESPONSE_LOSS=topk DSPARK_REJECTED_LOSS=topk DSPARK_TOPK=16
  export DSPARK_TOPK_SUPPORT=student DSPARK_FIRST_REJECTION_CE_ALPHA=0.2
  export DSPARK_REJECTED_FIRST_WEIGHT_MULTIPLIER=1.0 DSPARK_REJECTED_SUFFIX_WEIGHT_MULTIPLIER=1.0
  export DSPARK_RESPONSE_STREAM_WEIGHT=1 DSPARK_REJECTED_STREAM_WEIGHT=0.5
  export DSPARK_REJECTED_POSITION_DECAY=0.8
  export DSPARK_LOSS_DECAY_GAMMA=0 DSPARK_CE_LOSS_ALPHA=0 DSPARK_L1_LOSS_ALPHA=0
  export DSPARK_ACCEPTED_K3_ALPHA=1 DSPARK_ACCEPTED_FKL_ALPHA=0 DSPARK_REJECTED_K3_ALPHA=1
  export DSPARK_ASYMMETRIC_K3_ENABLED=false DSPARK_ACCEPTED_ASYMMETRIC_K3_ENABLED=false
  export DSPARK_REJECTED_ASYMMETRIC_K3_ENABLED=false DSPARK_DYNAMIC_ACCEPT_REJECT_WEIGHTING_ENABLED=false
  export DSPARK_TRACE_ALIGNED_ANCHORS=True DSPARK_USE_ABSOLUTE_POSITION_IDS=false
  export DSPARK_VLLM_ATTENTION_MODE=full DSPARK_VLLM_CONTEXT_WINDOW=null
  export DSPARK_TRAINING_ATTENTION_MODE=full DSPARK_TRAINING_CONTEXT_WINDOW=null
  export DSPARK_TRAINING_MAX_WINDOW=512 DSPARK_COLLECTION_WINDOW=512
  export DSPARK_DUMP_K3_LOGPROBS=true
  export DSPARK_POST_UPDATE_MEAN_ACCEPTANCE_LENGTH_DROP_ABORT_THRESHOLD=0.5
  export DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED=true SAVE_BEST_DRAFTER_CHECKPOINT=true
  export SPECO_PROTECTED_ACTOR_STEPS= DELETE_CHECKPOINTS_AFTER_SUCCESS=0
  export WANDB_PROJECT=verl_speco_dapo WANDB_MODE=online
  # 预览不启动训练；正式执行时将 1 改成 0，并确认上面的设备均归本任务使用。
  export DRY_RUN=1
  bash tools/run_dspark_vllm_experiment.sh \
    'actor_rollout_ref.rollout.drafter.training.fixed_eval_export_steps=[]' \
    actor_rollout_ref.rollout.drafter.training.fixed_eval_export_samples=0 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=disable
)
```

上面的命令默认 **DRY_RUN=1**，只打印配置。真正开跑时改成 0；建议在 tmux 中前台运行，启动器已负责日志和串行锁。此配置保留双 best、K3 dump、SAVE_FREQ=10、SAVE_OFFSET=9、101 step；关闭额外固定 actor/轨迹导出，只保留一个滚动 actor checkpoint。

不要调用全局 `ray stop --force`。命令已设置 `SPECO_SKIP_GLOBAL_RAY_STOP=1` 和 `RAY_ADDRESS=local`。串行队列仍复用现有启动器和 `SPECO_WAIT_FOR_HOST_LOCK=1`；多份 checkout 的文件锁不共享，不能据此认为它们会自动互斥。

## 7. 历史产物、双 best 与续跑

- [ ] 仅做新冷启动实验：模型、DAPO 和环境足够，不需要旧 actor checkpoint。
- [ ] 做跨机结果分析：按 run 复制 `outputs/drafter_analysis/`、日志及需要的 `outputs/drafters/`。相同目录结构可保留旧分析文档的相对链接。
- [ ] 双 best 不一定是两个名为 best 的权重目录：保留 `best_acceptance.json`、`best_throughput.json`、对应 iteration 标记及其引用的 `draft_step_N` 目录；两个 best 也可能指向同一步。
- [ ] 要恢复同一个训练 run：带走同一完整保存点的 actor、optimizer、调度器/训练额外状态、drafter 及恢复所需状态，以及根目录的 `latest_checkpointed_iteration.txt`、`.wandb_run_id`、`.wandb_run_path`；仅双 best 权重不是完整续跑包。
- [ ] 对正在写入的 run，不把单次 rsync 当作一致快照。选确认保存完成的 checkpoint；滚动清理可能删除旧目录，完整保存点必须先归档保留。
- [ ] 8→16 worker 的 optimizer/FSDP 状态恢复未在本次验证。首次迁移建议使用本文的新 run 冷启动；使用历史 drafter 权重初始化也应单独记为 warm-start。
- [ ] 不自动重测或选出双 best 胜者，仍保留两者供以后比较。

分析产物可用同样模式复制，例如仅复制某个已完成 run：

```bash
# 将 RUN_DIRECTORY 替换为实际目录名；在目标机运行。
mkdir -p /workspace/verl-SpeCo/outputs/drafter_analysis
rsync -aH --info=progress2 \
  SOURCE_HOST:/workspace/verl-SpeCo/outputs/drafter_analysis/RUN_DIRECTORY/ \
  /workspace/verl-SpeCo/outputs/drafter_analysis/RUN_DIRECTORY/
```

## 8. 正式 run 启动后核对与对比

首轮正式训练直接观察以下内容，不额外安排烟测：

- 日志记录预期逻辑设备、4 个 TP=4 副本、V1 runner，以及正确的 actor / drafter 路径。
- resolved config 为 response/rejected student-top16，first-rejection CE alpha=0.2，full/full，采集裁剪 512，训练间隔 5、每轮 6 step、采集每副本 24。
- 实际采集数、可训练样本数、有效 anchor 数、optimizer 成功步数及首拒绝 CE token 数正常。
- W&B 有 `drafter/spec_decode/mean_acceptance_length`；双 best 元数据和逐步 dump 开始生成。
- 日志、`outputs/drafter_analysis/<experiment>/<wandb_run_id>/k3_logprobs`、checkpoint 目录属于新实验。
- 比较速度时同时记录 verify 次数、accepted tokens、平均接受长度、生成耗时及有效输出 token 数。卡数翻倍带来的系统吞吐增长要与 drafter 接受效果分开判断；首次编译/加载时间也应单独看。
- 关注主存、/dev/shm、磁盘和 HBM 峰值；本配置的全局训练预算不变，不代表 CPU/Ray/副本运行时开销完全不变。

相关现有文档：
[共用启动器说明](../tools/README_vllm_experiments.md)、
[P4 配置记录](p4_top16_full_run_2026-09-05.md)、
[P4 阶段分析](p4_experiment_summary_through_p4_02.md)。

本次检查结果：所有 Bash 代码块通过语法检查；16 设备命令的 DRY_RUN 成功。与现有 P4_03 wrapper 的有效 Hydra 参数逐项对比，只有 8 项预期差异：逻辑设备数、drafter worker 数、batch/采样上限/token cap/holdout 四项每副本预算、新 checkpoint 路径，以及新实验显式禁用 resume。其余显式训练覆盖项一致。目标机的真实通信、显存峰值与完整训练尚未验证。

本次迁移准备的边界：整理当前机器证据、提供复制及启动命令、检查命令预览；不变更当前 run，不复制远程文件，不启动训练，不自动推送文档。
