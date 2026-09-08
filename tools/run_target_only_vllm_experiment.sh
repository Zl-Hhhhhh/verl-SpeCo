#!/usr/bin/env bash
set -euo pipefail

# User-facing target-only vLLM baseline.
#
# This uses the same PPO, W&B, checkpoint, resume, logging, and Ray lifecycle
# as the speculative runners, but always disables drafter rollout and training.
# Defaults match the DFlash DAPO b64 comparison profile. Override normal
# experiment settings through environment variables or Hydra arguments.
# DATASET_PROFILE accepts dapo, taco_taco, taco_lcb, taco_lcb_full,
# wildsci_gpqa, or wildsci_mmlu, with the same paths and length defaults as the
# speculative runners. Use the identical profile and explicit length/batch
# overrides when measuring target-only versus speculative generation time.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SPECO_ALGORITHM=target_only
SPECO_BASE_LAUNCHER=${PROJECT_ROOT}/examples/run_qwen3-8b_drafter_dflash_vllm_npu.sh

# The common runner currently validates and exports this path before Hydra
# composition. Reuse the target path so this baseline does not require a draft
# model installation; the path is not loaded when drafter.enable=False.
SPECO_DEFAULT_DRAFTER_PATH=${MODEL_PATH:-/workspace/models/Qwen3-8B}

SPECO_ALGORITHM_OVERRIDES=(
    "actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS:-False}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-False}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.6}"
)

# target-only is represented as offline to reuse the common runner without
# allocating online drafter workers. The final Hydra overrides below disable
# fixed speculative decoding as well.
export RUN_MODE=offline
export EXPERIMENT_PYTHON=${TARGET_ONLY_PYTHON:-/workspace/main_env_next/bin/python}
target_dataset_profile=${DATASET_PROFILE:-dapo}
target_dataset_profile=${target_dataset_profile,,}
target_dataset_profile=${target_dataset_profile//-/_}
target_dataset_suffix=
[[ "${target_dataset_profile}" == dapo ]] || target_dataset_suffix=_${target_dataset_profile}
export EXPERIMENT_ID=${EXPERIMENT_ID:-target_only${target_dataset_suffix}_${TOTAL_STEPS:-150}step}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-target_only_${TOTAL_STEPS:-150}step}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_8b_dflash_comparison}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-target_only_baseline}
export WANDB_TAGS=${WANDB_TAGS:-main_env_next,cann901,vllm-v1,npu${N_GPUS:-16},vllm-ascend,target-only,no-spec,dataset-${target_dataset_profile},b${TRAIN_BATCH_SIZE:-64}}

# shellcheck source=speco_vllm_experiment_common.sh
source "${PROJECT_ROOT}/tools/speco_vllm_experiment_common.sh"
# Keep these overrides last. In particular, callers cannot accidentally turn
# this baseline back into fixed or online speculative decoding via "$@".
SPECO_FINAL_HYDRA_OVERRIDES=(
    "actor_rollout_ref.rollout.drafter.enable=False"
    "actor_rollout_ref.rollout.drafter.enable_drafter_training=False"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config=null"
)
speco_launch_vllm_experiment "$0" "$@"
