#!/usr/bin/env bash
set -euo pipefail

# User-facing DFlash experiment entrypoint.
#
# Common edits are environment variables, for example:
#   RUN_MODE=online DRAFT_WORKER_WORLD_SIZE=4 TOTAL_STEPS=100 ./tools/run_dflash_vllm_experiment.sh
#   RUN_MODE=offline TOTAL_STEPS=100 ./tools/run_dflash_vllm_experiment.sh
#
# Important paths: MODEL_PATH, DRAFTER_PATH, TRAIN_FILE, TEST_FILE.
# Dataset selection: DATASET_PROFILE=dapo|taco_taco|taco_lcb|taco_lcb_full|wildsci_gpqa|wildsci_mmlu.
#   taco_taco: 2048 prompt / 4096 response, frequent small validation; useful for code-reward smoke tests.
#   taco_lcb: 2048 / 4096, validates on LCB v6-only every 10 steps; use SANDBOX_FUSION_URL when available.
#   taco_lcb_full: 2048 / 4096, intended for trainer.val_only=True final evaluation, not frequent validation.
#   wildsci_gpqa: 3072 prompt / 4096 response, validates every 10 steps and retains all GPQA questions.
#   wildsci_mmlu: 2048 / 2048, validates every 50 steps because the test split has 5,878 questions.
# Profile defaults can be overridden with MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, TEST_FREQ,
# VAL_BEFORE_TRAIN, TRAIN_FILE, or TEST_FILE. For code OOM, try MAX_NUM_SEQS=128 before truncating prompts.
# BigCodeBench is intentionally not a profile: its unittest tasks require the official harness at
# /workspace/data/drafter_domains/raw/bigcodebench_repo rather than VERL's APPS/TACO reward path.
# W&B: WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_GROUP, WANDB_RUN_NAME, WANDB_RUN_ID.
# Recovery: EXPERIMENT_ID, CHECKPOINT_DIR, SAVE_FREQ, DELETE_CHECKPOINTS_AFTER_SUCCESS.
# Memory: DRAFT_UPDATE_BUCKET_MB (128 recommended), DRAFT_PUBLISH_ASYNC=false.
# Any arguments are appended as Hydra overrides and therefore win over defaults.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SPECO_ALGORITHM=dflash
SPECO_BASE_LAUNCHER=${PROJECT_ROOT}/examples/run_qwen3-8b_drafter_dflash_vllm_npu.sh
SPECO_DEFAULT_DRAFTER_PATH=/workspace/models/z-lab-Qwen3-8B-DFlash-b16

# DFlash-only knobs. Override these through environment variables or append the
# full Hydra key on the command line for one-off debugging.
SPECO_ALGORITHM_OVERRIDES=(
    "actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS:-False}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.drafter.vllm.enforce_eager=${DRAFTER_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=${DFLASH_NUM_ANCHORS:-64}"
    "actor_rollout_ref.rollout.drafter.training.dflash_max_window=${DFLASH_MAX_WINDOW:-512}"
    "actor_rollout_ref.rollout.drafter.training.dflash_loss_mode=${DFLASH_LOSS_MODE:-chunked_ce}"
    "actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=False"
    "actor_rollout_ref.rollout.drafter.training.dflash_hard_sample_ratio=${DFLASH_HARD_SAMPLE_RATIO:-0.3}"
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${DRAFT_HIDDEN_WINDOW_TOKENS:-512}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_steps=${SPEC_STEPS:-1}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_topk=${SPEC_TOPK:-1}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPEC_VERIFY_TOKENS:-15}"
)

export EXPERIMENT_PYTHON=${DFLASH_PYTHON:-/workspace/main_env_next/bin/python}

# shellcheck source=speco_vllm_experiment_common.sh
source "${PROJECT_ROOT}/tools/speco_vllm_experiment_common.sh"
speco_launch_vllm_experiment "$0" "$@"
