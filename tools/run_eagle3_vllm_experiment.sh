#!/usr/bin/env bash
set -euo pipefail

# User-facing Eagle3 experiment entrypoint.
#
# Common edits are environment variables, for example:
#   RUN_MODE=online DRAFT_WORKER_WORLD_SIZE=4 TOTAL_STEPS=100 ./tools/run_eagle3_vllm_experiment.sh
#   RUN_MODE=offline TOTAL_STEPS=100 ./tools/run_eagle3_vllm_experiment.sh
#
# Important paths: MODEL_PATH, DRAFTER_PATH, TRAIN_FILE, TEST_FILE.
# Dataset selection: DATASET_PROFILE=dapo|taco_taco|taco_lcb|taco_lcb_full|wildsci_gpqa|wildsci_mmlu.
# Code profiles default to 2048 prompt / 4096 response; set SANDBOX_FUSION_URL for isolated execution.
# wildsci_gpqa defaults to 3072 prompt / 4096 response and validation every 10 steps; wildsci_mmlu uses
# 2048 / 2048 and every 50 steps. Use taco_lcb_full with trainer.val_only=True for final evaluation.
# MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, TEST_FREQ, TRAIN_FILE, and TEST_FILE override the profile.
# BigCodeBench remains an external official-harness evaluation and is not supported as TEST_FILE here.
# W&B: WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_GROUP, WANDB_RUN_NAME, WANDB_RUN_ID.
# Recovery: EXPERIMENT_ID, CHECKPOINT_DIR, SAVE_FREQ, DELETE_CHECKPOINTS_AFTER_SUCCESS.
# Any arguments are appended as Hydra overrides and therefore win over defaults.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SPECO_ALGORITHM=eagle3
SPECO_BASE_LAUNCHER=${PROJECT_ROOT}/examples/run_qwen3-8b_drafter_eagle3_vllm_npu.sh
SPECO_DEFAULT_DRAFTER_PATH=/workspace/models/Tengyunw-qwen3_8b_eagle3

SPECO_ALGORITHM_OVERRIDES=(
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.drafter.vllm.enforce_eager=${DRAFTER_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.drafter.training.use_logits=${EAGLE_USE_LOGITS:-False}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_steps=${SPEC_STEPS:-3}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_topk=${SPEC_TOPK:-1}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPEC_VERIFY_TOKENS:-4}"
)

export EXPERIMENT_PYTHON=${MAIN_PYTHON:-/workspace/main_env_next/bin/python}

# shellcheck source=speco_vllm_experiment_common.sh
source "${PROJECT_ROOT}/tools/speco_vllm_experiment_common.sh"
speco_launch_vllm_experiment "$0" "$@"
