# SPECO vLLM NPU experiment runners

Use these user-facing entrypoints:

- `tools/run_dspark_vllm_experiment.sh`
- `tools/run_dflash_vllm_experiment.sh`
- `tools/run_eagle3_vllm_experiment.sh`
- `tools/run_target_only_vllm_experiment.sh`

All four entrypoints use the same W&B, logging, checkpoint, resume, retention, and
cleanup implementation in `tools/speco_vllm_experiment_common.sh`.

## Dataset profiles

Set `DATASET_PROFILE` on any of the three entrypoints. Explicit `TRAIN_FILE`,
`TEST_FILE`, `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `TEST_FREQ`, and
`VAL_BEFORE_TRAIN` values override the profile defaults.

| Profile | Train -> validation | Prompt / response | Validation |
| --- | --- | ---: | --- |
| `dapo` | DAPO Math -> DAPO Math | 512 / 8192 | disabled (existing default) |
| `taco_taco` | TACO-Cobalt -> TACO validation | 2048 / 8192 | before train, every 10 steps |
| `taco_lcb` | TACO-Cobalt -> LCB v6-only (175) | 2048 / 8192 | before train, every 10 steps |
| `taco_lcb_full` | TACO-Cobalt -> LCB release_v6 (1,055) | 2048 / 8192 | before train only; prefer `trainer.val_only=True` |
| `wildsci_gpqa` | WildSci All-Aligned -> GPQA-Diamond | 2048 / 4096 | before train, every 10 steps |
| `wildsci_mmlu` | WildSci All-Aligned -> MMLU-Pro STEM (5,878) | 2048 / 2048 | every 50 steps |

Code and science profiles automatically configure the custom reward from
`/workspace/data/drafter_domains/reward.py`. Code generation executes untrusted
model output; set `SANDBOX_FUSION_URL` for isolated execution. Without it, the
reward falls back to VERL's local PRIME code checker. The bundle supplies the
small Python 3.11 compatibility layer needed by PRIME, because the upstream
`pyext==0.7` package no longer installs cleanly on Python 3.11.

## Reward scale

All selectable profiles report training reward on the `[-1, 1]` scale. DAPO
already produces `-1` for an incorrect answer and `1` for a correct answer.
The custom TACO/LiveCodeBench and WildSci rewards convert their original
pass-rate/accuracy `r` with `2 * r - 1`; the unmodified `r` is retained as
`raw_score` in reward metadata. New W&B runs are tagged `reward-pm1`, and the
main `critic/rewards/mean` curve is directly comparable across profiles.

For the historical non-DAPO W&B curves, run the idempotent importer below. It
does not mutate W&B's append-only source history: it creates a same-project,
same-group overlay run containing the standard `critic/rewards/mean`,
`critic/score/mean`, and `key_metrics/critic_reward_mean` fields, all aligned
to the original W&B step.

```bash
/workspace/main_env_next/bin/python tools/backfill_reward_pm1_wandb.py
```

BigCodeBench is deliberately not exposed as a runner profile. Its tests are
Python `unittest` programs, not APPS/TACO input-output cases, and must be scored
with the official harness in
`/workspace/data/drafter_domains/raw/bigcodebench_repo`. Adding its Parquet file
as `TEST_FILE` would reach an unsupported VERL reward path.

Recommended code-domain run:

```bash
DATASET_PROFILE=taco_lcb \
DELETE_CHECKPOINTS_AFTER_SUCCESS=0 \
./tools/run_dflash_vllm_experiment.sh
```

Set `SANDBOX_FUSION_URL` only when a real Sandbox Fusion service is deployed;
literal placeholders such as `http://HOST:PORT/run_code` are rejected at startup.
The endpoint must be reachable from every Ray reward worker and end in
`/run_code`:

```bash
# Run this on a separate CPU/service host with Docker.
docker run -d --name sandbox-fusion --restart unless-stopped \
  -p 8080:8080 volcengine/sandbox-fusion:server-20250609

curl http://localhost:8080/run_code \
  -H 'Content-Type: application/json' \
  --data-raw '{"code":"print(1)","language":"python"}'
```

When training itself runs in a container, do not use `localhost` unless the
sandbox is in that same container. Use the service hostname or a host address
that is reachable from the training container.

```bash
DATASET_PROFILE=taco_lcb \
SANDBOX_FUSION_URL=http://sandbox-host:8080/run_code \
SANDBOX_MEMORY_LIMIT_MB=1024 \
SANDBOX_MAX_CONCURRENT=64 \
DELETE_CHECKPOINTS_AFTER_SUCCESS=0 \
./tools/run_dflash_vllm_experiment.sh
```

Sandbox Fusion is optional for smoke tests but recommended for long code-domain
runs. It is not used by the `dapo` or `wildsci_*` profiles.

Recommended science-domain run:

```bash
DATASET_PROFILE=wildsci_gpqa \
DELETE_CHECKPOINTS_AFTER_SUCCESS=0 \
./tools/run_dflash_vllm_experiment.sh
```

The 512-token DAPO default retains only 29.3% of TACO-Cobalt. Keep the code
profile's 2048-token prompt limit; if memory is tight, first reduce
`MAX_NUM_SEQS` to 128 or reduce the training batch rather than truncating the
problem statement. GPQA uses a 4096-token response budget. MMLU-Pro STEM is
large enough that it should not be validated every 10 steps.

## Typical commands

Run DFlash online training with four draft workers:

```bash
RUN_MODE=online \
EXPERIMENT_ID=dflash_dapo_online_w4_100step \
DRAFT_WORKER_WORLD_SIZE=4 \
TOTAL_STEPS=100 \
./tools/run_dflash_vllm_experiment.sh
```

Run the matching fixed-drafter control:

```bash
RUN_MODE=offline \
EXPERIMENT_ID=dflash_dapo_fixed_100step \
TOTAL_STEPS=100 \
./tools/run_dflash_vllm_experiment.sh
```

Replace the script name with `run_eagle3_vllm_experiment.sh` for Eagle3.

Run the target-only/no-spec baseline matched to the DFlash DAPO profile:

```bash
EXPERIMENT_ID=target_only_150step \
TOTAL_STEPS=150 \
TRAIN_BATCH_SIZE=64 \
./tools/run_target_only_vllm_experiment.sh \
  data.seed=42
```

The target-only entrypoint always forces `drafter.enable=False`,
`drafter.enable_drafter_training=False`, and a null vLLM
`speculative_config`. It defaults to temperature `0.6`, actor KL loss off,
and actor parameter offload off to match the DFlash comparison runs. Use
`ROLLOUT_TEMPERATURE`, `ACTOR_USE_KL_LOSS`, and `ACTOR_PARAM_OFFLOAD` only when
building a separately matched baseline, for example for Eagle3.

Use `DRY_RUN=1` to print the complete command without starting Ray or writing
experiment state:

```bash
DRY_RUN=1 RUN_MODE=online DRAFT_WORKER_WORLD_SIZE=4 \
./tools/run_dflash_vllm_experiment.sh
```

Cold-start a new online run from a trained drafter checkpoint:

```bash
DRAFTER_INIT_CHECKPOINT=/workspace/verl-SpeCo/outputs/drafters/<experiment>/<wandb-run-id>/draft_step_101 \
DRAFTER_INIT_LABEL=p2_02_s101 \
RUN_MODE=online DATASET_PROFILE=dapo \
./tools/run_dspark_vllm_experiment.sh
```

`DRAFTER_INIT_CHECKPOINT` must point directly to a complete managed
`draft_step_N` checkpoint. Only model weights are loaded for a fresh run;
the actor, global step, dataloader, drafter optimizer, and LR scheduler start
fresh. The generated experiment and W&B names include `__cold_<label>` (the
default label is `sN`). Re-running the same command after it has saved an actor
checkpoint switches to a normal full resume, so recovery keeps both actor and
drafter optimizer state. Use `DRAFTER_PATH` instead for an offline fixed-drafter
evaluation.

## DSpark Candidate Gating

When enabled, the DSpark candidate gate does not publish a training result immediately. It keeps
the pre-update drafter as `theta_old`, forms candidates
`theta_alpha = (1 - alpha) * theta_old + alpha * theta_trained`, and runs the
same fixed 16-prompt greedy speculative rollout for every configured alpha.
`alpha=0` is the guaranteed no-op candidate; `alpha=1` is the full optimizer
update, while `0.25` and `0.5` allow a useful but overly aggressive update to
be committed partially. This is parameter interpolation, not an inference-time
ensemble and not four resident drafter models.

The best candidate must satisfy both acceptance and generation-time gates. It
then serves the normal actor rollout. If that full rollout loses at least the
configured acceptance threshold, the runner restores `alpha=0`, regenerates
the actor batch, records the rollback, and continues the run. Fixed-loss,
gradient-norm, and rejected-K3 circuit breakers can reject a candidate before
any serving publication. The best fixed-cohort checkpoint and terminal
checkpoint are both protected by retention and included in the final archive.

Here a *strong checkpoint* means a complete, previously trained drafter
checkpoint selected as the cold-start source because it already has validated
speculative-decoding performance. It does not mean a larger file or merely a
checkpoint containing optimizer state. The managed p2_02 `draft_step_101` is
the current DSpark cold-start default. Because it is already useful, p2_06 adapts it with a
smaller budget by default instead of repeatedly applying the original training
schedule.

## Managed detached execution

The experiment entrypoints start in a detached `nohup` + `setsid` session by
default. The training process, Ray cluster, checkpoint watcher, W&B upload, and
validation log therefore survive an SSH or VSCode Remote terminal disconnect.
The invoking terminal follows the normal validation log while it remains open;
closing it stops only that log observer.

Each launch prints a managed run ID and stores lifecycle state under
`outputs/managed_runs/<run-id>/`. Inspect, follow, or stop that specific run
without relying on the original terminal:

```bash
bash tools/speco_managed_run.sh list
bash tools/speco_managed_run.sh status <run-id>
bash tools/speco_managed_run.sh follow <run-id>
bash tools/speco_managed_run.sh stop <run-id>
```

Set `SPECO_FOLLOW_ON_LAUNCH=0` to return immediately after dispatching a run.
Set `SPECO_LAUNCH_MODE=foreground` only for interactive debugging; foreground
mode intentionally retains the old terminal-bound lifecycle.

## Parameters to edit most often

| Variable | Default | Purpose |
| --- | --- | --- |
| `RUN_MODE` | `online` | `online` trains/publishes the draft; `offline` keeps it fixed. |
| `SPECO_ONLINE_STRATEGY_TAG` | `p1` (`p2_06` for DSpark) | Appended to generated online W&B and local experiment/checkpoint names. The DSpark entrypoint pins `p2_06` so stale shell values cannot silently change the strategy. |
| `SPECO_WAIT_FOR_HOST_LOCK` | `0` | Set to `1` to queue a managed run behind the experiment currently using the host. |
| `DATASET_PROFILE` | `dapo` | Selects pinned train/eval paths plus length and validation defaults. |
| `EXPERIMENT_ID` | derived | Stable local identity. Reuse it to resume the same failed run. |
| `TOTAL_STEPS` | `100` | Main RL training steps. |
| `ENABLE_CHECKPOINTS` | `1` | Set to `0` to disable checkpoint saving and training recovery. |
| `SAVE_FREQ` | `10` | Checkpoint interval. |
| `SAVE_OFFSET` | `9` | Saves at 19, 29, 39... so checkpointing does not overlap drafter update steps. |
| `DRAFT_WORKER_WORLD_SIZE` | `16` | Online draft training world size; divisible by rollout TP x DP x PP. |
| `DRAFT_TRAIN_STEPS` | `3` strong cold start; otherwise `5` | Drafter optimizer steps per online training trigger. |
| `DRAFT_COLLECT_INTERVAL_STEPS` | `10` strong cold start; otherwise `5` | Actor steps between fresh online drafter feature collection. |
| `DRAFT_TRAINING_INTERVAL_STEPS` | `10` strong cold start; otherwise `5` | Actor steps between drafter update triggers; keep aligned with collection. |
| `DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA` | `20` | Per-trigger sample cap used by the unique-first online sampler. |
| `DRAFT_TRAIN_UNIQUE_SAMPLES_FIRST` | `true` | Delay sample reuse until the current unique pool is exhausted. |
| `DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA` | `4` | Persistent held-out samples evaluated before and after each update. Set `0` to disable. |
| `DRAFT_LR` | `2e-6` strong cold start; otherwise `5e-6` | Managed online drafter learning rate. Strong cold starts also use a 2-step warmup and 10-step global cosine schedule. |
| `DRAFT_CANDIDATE_GATE_ENABLED` | `true` for DSpark | Validate a trained candidate with real speculative decoding before serving it. |
| `DRAFT_CANDIDATE_GATE_ALPHAS` | `[0,.25,.5,1]` | Parameter interpolation candidates between the old and fully updated drafter. |
| `DRAFT_CANDIDATE_GATE_SAMPLES` | `16` | Size of the fixed prompt cohort reused by every candidate gate. |
| `DRAFT_CANDIDATE_GATE_MAX_TOKENS` | `256` | Per-response generation cap for candidate validation. |
| `DRAFT_CANDIDATE_GATE_MIN_ACCEPTANCE_DELTA` | `0` | Required fixed-cohort mean-acceptance gain over `alpha=0`. |
| `DRAFT_CANDIDATE_GATE_MAX_MS_PER_TOKEN_REGRESSION` | `0.15` | Maximum allowed fixed-cohort generation latency regression. |
| `DRAFT_CANDIDATE_MAX_FIXED_LOSS_DELTA` | `0` | Reject a candidate before publication when fixed-cohort drafter loss rises beyond this value. |
| `DRAFT_CANDIDATE_MAX_GRAD_NORM` | `50` | Absolute pre-publication gradient-norm circuit breaker. |
| `DRAFT_CANDIDATE_MAX_REJECTED_K3_CLAMP_RATIO` | `0.5` | Pre-publication rejected-K3 clamp-ratio circuit breaker. |
| `SAVE_BEST_DRAFTER_CHECKPOINT` | `true` | Save and retention-protect the best fixed-rollout candidate in addition to the final drafter. |
| `DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED` | `false` | Without extra evaluation rollouts, retain separate best-acceptance and best-effective-throughput drafter checkpoints from the first full rollout after each synchronous publish. Use 101 total steps to score an update published at step 100. |
| `BEST_CHECKPOINT_MIN_ACCEPTANCE_DELTA` | `0` | Minimum absolute mean-acceptance improvement required to replace `best_acceptance`. |
| `BEST_CHECKPOINT_MIN_THROUGHPUT_RELATIVE_DELTA` | `0` | Minimum relative effective-output-TPS improvement required to replace `best_throughput`; `0.01` requires 1%. |
| `DRAFT_DUAL_METRIC_BEST_CHECKPOINT_INCLUDE_OPTIMIZER` | `false` | Include optimizer shards in dual-best checkpoints. Keep disabled when the candidates are only retained for later inference evaluation. |
| `SPECO_PROTECTED_ACTOR_STEPS` | empty | Comma-separated actor checkpoint steps that the managed retention watcher must preserve, for example `100`. Set `trainer.max_actor_ckpt_to_keep` high enough to retain these steps in the upstream checkpoint manager as well. |
| `DSPARK_POST_UPDATE_MEAN_ACCEPTANCE_LENGTH_DROP_ABORT_THRESHOLD` | `0.4` | Candidate mode rolls back and regenerates the actor batch when the full-rollout mean acceptance drops by at least this amount; legacy direct publish still aborts. |
| `DSPARK_K3_DISTRIBUTION_BINS` | `512` | Weighted histogram bins used for accepted/rejected K3 median metrics over the clamped `[0, 10]` range. |
| `DSPARK_DUMP_K3_LOGPROBS` | `false` | Save per-token target/draft logprobs and sampled K3 analysis payloads during DSpark drafter training. |
| `DSPARK_DUMP_K3_LOGPROBS_DIR` | generated | Optional dump root; defaults to `outputs/drafter_analysis/<experiment>/<wandb-run-id>/k3_logprobs`. |
| `DSPARK_DUMP_K3_LOGPROBS_INTERVAL_STEPS` | `1` | Dump every Nth attempted drafter optimizer step. |
| `DSPARK_DUMP_K3_LOGPROBS_MAX_TOKENS_PER_STREAM` | `0` | Per-file accepted/rejected token cap; `0` preserves every sampled K3 token. |
| `DRAFT_UPDATE_BUCKET_MB` | `256` | Weight transfer bucket. Lower values reduce host peak memory. |
| `DRAFT_PUBLISH_ASYNC` | `false` | Keep false on the current A3 stack to avoid overlapping publishes. |
| `TRAIN_BATCH_SIZE` | `64` | Main RL input batch per step. |
| `N_GPUS` | `16` | Main actor/Ray NPU count; the runner derives the visible-device list when it is unset. |
| `ROLLOUT_N` | `5` | Responses sampled per prompt for GRPO. |
| `MAX_PROMPT_LENGTH` | profile-specific | 512 for DAPO and 2048 for code/science profiles. |
| `MAX_RESPONSE_LENGTH` | profile-specific | 8192 for code/DAPO, 4096 for GPQA, and 2048 for MMLU. |
| `TEST_FREQ` | profile-specific | `-1` disables periodic validation; MMLU defaults to 50. |
| `SANDBOX_FUSION_URL` | unset | Isolated code execution endpoint used by TACO/LCB profiles. |
| `SANDBOX_MEMORY_LIMIT_MB` | `1024` | Address-space limit for each submitted code execution; not a sandbox disk quota. |
| `SANDBOX_MAX_CONCURRENT` | `16` | Host-wide limit for concurrent local SandboxFusion requests. |
| `RAY_MEMORY_USAGE_THRESHOLD` | `0.99` | Ray worker-kill threshold; keep the monitor enabled and provide host swap. |
| `MODEL_PATH` | `/workspace/models/Qwen3-8B` | Main model. |
| `DRAFTER_PATH` | algorithm-specific | Initial draft model. |
| `DRAFTER_INIT_CHECKPOINT` | p2_02 step 101 for online DSpark; otherwise unset | Online only: initialize a fresh run from a complete managed `draft_step_N`; optimizer/scheduler state is not restored. Set it empty to disable DSpark cold start. |
| `DRAFTER_INIT_LABEL` | `p2_02_v1` for the default checkpoint; otherwise `sN` | Short source label appended as `__cold_<label>` to generated experiment and W&B names. |
| `DRAFTER_INIT_IS_STRONG` | `1` for the default p2_02 source; otherwise `0` | Explicitly opt another validated cold-start checkpoint into the reduced adaptation budget. |
| `TRAIN_FILE` / `TEST_FILE` | DAPO-Math-17k | Dataset parquet files. |
| `SAVE_DRAFTER_AFTER_SUCCESS` | `1` | For a completed online run, archive the final directly loadable drafter under `outputs/drafters/<experiment>/<wandb_run_id>/` before cleanup. |
| `FINAL_DRAFTER_ROOT` | `outputs/drafters` | Root directory for completed-run drafter archives. |
| `DELETE_CHECKPOINTS_AFTER_SUCCESS` | `1` | Keep checkpoints on failure; after a successful drafter archive, delete the large training checkpoint. |
| `EXPERIMENT_PYTHON` | `/workspace/main_env_next/bin/python` | CANN 9.0.1 / vLLM V1 Python used by the launcher and Ray CLI discovery. |

The managed DSpark `p2` objective and its `DSPARK_*` overrides are documented
in [P2 drafter training](../docs/p2_drafter_training_changes.md).

DSpark training reports `accepted_k3_{mean,min,max,median}` and
`rejected_k3_{mean,min,max,median}` below the `drafter/train/dspark_*` W&B
namespace. Mean, min, and max are exact over the distributed weighted stream.
Median is reconstructed from the mergeable histogram, with resolution
`10 / DSPARK_K3_DISTRIBUTION_BINS`; the reported value is the lower edge of
the median bin. Existing `accepted_k3` and `rejected_k3` mean metrics remain
available for dashboard compatibility.

Enable the optional analysis dump with:

```bash
DSPARK_DUMP_K3_LOGPROBS=true \
DSPARK_DUMP_K3_LOGPROBS_INTERVAL_STEPS=1 \
DSPARK_DUMP_K3_LOGPROBS_MAX_TOKENS_PER_STREAM=0 \
./tools/run_dspark_vllm_experiment.sh
```

Each rank writes an atomic `.pt` file under an `rl_step_N` directory. The
`accepted` and `rejected` stream payloads contain the sampled `token_ids`,
`prev_token_ids`, target/draft `logprobs`, reconstructed `k3`, loss `weights`,
clamp flags, sample/anchor/draft positions, and trace accepted/draft counts.
Only the logprob of the sampled token is stored, not the full vocabulary, so
the dump remains practical for post-run diagnosis. A positive per-stream cap
selects tokens evenly across each stream instead of keeping only its prefix.
The files live outside
managed checkpoints and therefore survive successful checkpoint cleanup.

W&B uses `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_GROUP`,
`WANDB_RUN_NAME`, and optionally `WANDB_RUN_ID`. The generated W&B run ID is
stored in the checkpoint directory and reused automatically on recovery. Do
not place an API key in these scripts; use `wandb login` or `WANDB_API_KEY` in
the calling shell.

Checkpoint directories also store `.wandb_run_path` as
`entity/project/run-id`. Recovery validates explicit `WANDB_ENTITY`,
`WANDB_PROJECT`, and `WANDB_RUN_ID` values against this identity so a moved run
cannot be resumed into the wrong project. When the new default checkpoint name
does not contain state, the runner automatically checks the corresponding
legacy name containing `_vllm`; `CHECKPOINT_DIR` still takes precedence when
set explicitly.

By default, DAPO, TACO-Cobalt, and WildSci runs go to `verl_speco_dapo`,
`verl_speco_taco`, and `verl_speco_wildsci`, respectively. LCB is an evaluation
set under the TACO project; GPQA and MMLU-Pro STEM are evaluation sets under the
WildSci project. W&B run names omit the dataset because the project identifies
the training domain. Local experiment/checkpoint IDs retain the profile suffix
to prevent cross-profile resume collisions.

The two legacy GSM8K smoke runs are kept separately in
`verl_speco_gsm8k_smoke`; this is a historical archive, not a selectable
`DATASET_PROFILE`.

The managed runner keeps `N_GPUS`, `SPECO_ACCELERATOR_COUNT`, and
`ASCEND_RT_VISIBLE_DEVICES` consistent before Ray starts. With no explicit
device mask, `N_GPUS=16` produces `0,1,...,15`. When supplying a mask, it must
contain at least `N_GPUS` comma-separated IDs; use `N_GPUS=8` with an
eight-device mask.

The default actor checkpoint contains `model`, `optimizer`, and `extra`. It
restores actor model weights, actor Adam optimizer moments, RNG/scheduler state,
dataloader position, and global step. The matching `drafter/draft_step_<step>`
directory contains loadable draft weights plus drafter optimizer and scheduler
state. Resume selects the complete drafter checkpoint matching the committed
actor step before Ray worker classes are created. These optimizer shards
substantially increase checkpoint size and temporary serialization memory;
check disk and host-memory capacity before increasing retention.

After a successful online fit, the trainer writes the exact terminal drafter
state even when the periodic checkpoint schedule is offset. The runner then
validates the model, metadata, and optimizer files and archives that checkpoint
under `outputs/drafters/<experiment>/<wandb_run_id>/draft_step_<step>`. On the
default filesystem this uses hard links, so archiving does not temporarily
duplicate the large full drafter checkpoint. The archive remains valid after
the actor checkpoint is deleted. If validation or archiving fails, the run is
reported as failed and the original checkpoint is retained.

To run without checkpoints:

```bash
ENABLE_CHECKPOINTS=0 \
RUN_MODE=online \
EXPERIMENT_ID=dflash_online_no_checkpoint \
./tools/run_dflash_vllm_experiment.sh
```

This sets `trainer.save_freq=-1`, `trainer.resume_mode=disable`, and does not
start the checkpoint-retention watcher. W&B and console logs remain enabled.
After a failure, rerunning starts training again at step 1; use a new
`EXPERIMENT_ID` if you do not want the restarted metrics appended to the same
W&B run.

## Four-worker host memory

Changing `DRAFT_WORKER_WORLD_SIZE=2` to `4` adds two persistent training
processes. The observed process sizes imply roughly 12-18 GB of additional
steady host memory, but this is not a bound on the total peak. A publish also
materializes the approximately 2.1 GB draft state on 16 actor-rollout ranks and
allocates transfer buckets. Checkpoint serialization creates another temporary
peak.

The common runner therefore defaults to synchronous publish, a 256 MB bucket,
explicit post-publish heap release, one retained checkpoint, and a checkpoint
offset of 2. With a frequency of 10, draft publish runs at 5/10/15 while saves
run at 12/22/32, avoiding the combined publish-plus-checkpoint peak. The runner
allows Ray to reach 0.99 only when the host is intentionally configured for
near-capacity execution; inspect host memory after steps 5, 10, 11, and 15 before
treating a new worker configuration as stable. Raising or disabling Ray's memory
monitor does not reduce real memory use.

Extra arguments are Hydra overrides and take precedence. Example:

```bash
./tools/run_dflash_vllm_experiment.sh \
  actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=32 \
  actor_rollout_ref.rollout.drafter.training.dflash_max_window=384
```

## Adding an online drafter algorithm

Keep lifecycle behavior in `speco_vllm_experiment_common.sh`. A new algorithm
entrypoint only needs to set `SPECO_ALGORITHM`, `SPECO_BASE_LAUNCHER`,
`SPECO_DEFAULT_DRAFTER_PATH`, and an optional `SPECO_ALGORITHM_OVERRIDES` array,
then source the common runner. This keeps checkpoint recovery, W&B identity,
runtime isolation, retention, and argument precedence identical across
algorithms. Add the algorithm defaults to `speco_base.yaml` and contract-test
the backend-specific overrides separately from the shared runner.
