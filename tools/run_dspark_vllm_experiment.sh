#!/usr/bin/env bash
set -euo pipefail

# User-facing DSpark experiment entrypoint.
#
# Common edits are environment variables, for example:
#   RUN_MODE=online DRAFT_WORKER_WORLD_SIZE=4 TOTAL_STEPS=100 ./tools/run_dspark_vllm_experiment.sh
#   RUN_MODE=offline TOTAL_STEPS=100 ./tools/run_dspark_vllm_experiment.sh
#
# Important paths: MODEL_PATH, DRAFTER_PATH, TRAIN_FILE, TEST_FILE.
# Cold-start an online run from trained drafter weights with
# DRAFTER_INIT_CHECKPOINT=/path/to/draft_step_N and optional DRAFTER_INIT_LABEL.
# Attention ablations use DSPARK_VLLM_ATTENTION_MODE=full|swa and
# DSPARK_VLLM_CONTEXT_WINDOW=512|1024|2048. Matched training is independently
# opt-in through DSPARK_TRAINING_ATTENTION_MODE and DSPARK_TRAINING_CONTEXT_WINDOW.
# Dataset selection: DATASET_PROFILE=dapo|taco_taco|taco_lcb|taco_lcb_full|wildsci_gpqa|wildsci_mmlu.
# Profile defaults can be overridden with MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH,
# TEST_FREQ, TRAIN_FILE, or TEST_FILE.
# W&B: WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_GROUP, WANDB_RUN_NAME, WANDB_RUN_ID.
# Recovery: EXPERIMENT_ID, CHECKPOINT_DIR, SAVE_FREQ, DELETE_CHECKPOINTS_AFTER_SUCCESS.
# Any arguments are appended as Hydra overrides and therefore win over defaults.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

SPECO_ALGORITHM=dspark
SPECO_BASE_LAUNCHER=${PROJECT_ROOT}/examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh
SPECO_DEFAULT_DRAFTER_PATH=/workspace/models/dspark_qwen3_8b_block7

# DSpark-only knobs. The remaining DSpark settings use speco_base.yaml defaults
# and can be changed with a full Hydra override when needed.
SPECO_ALGORITHM_OVERRIDES=(
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}"
    "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=${DSPARK_COLLECT_HIDDEN_STATES_FROM_OLD_LOGPROB:-True}"
    "actor_rollout_ref.rollout.drafter.training.dspark_block_size=${DSPARK_BLOCK_SIZE:-7}"
    "actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=${DSPARK_LOSS_MODE:-chunked_ce}"
    "actor_rollout_ref.rollout.drafter.training.dspark_response_loss=${DSPARK_RESPONSE_LOSS:-k3}"
    "actor_rollout_ref.rollout.drafter.training.dspark_rejected_loss=${DSPARK_REJECTED_LOSS:-k3}"
    "actor_rollout_ref.rollout.drafter.training.dspark_topk=${DSPARK_TOPK:-16}"
    "actor_rollout_ref.rollout.drafter.training.dspark_topk_support=${DSPARK_TOPK_SUPPORT:-student}"
    "actor_rollout_ref.rollout.drafter.training.dspark_first_rejection_ce_alpha=${DSPARK_FIRST_REJECTION_CE_ALPHA:-0.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_first_rejection_ce_low_entropy_multiplier=${DSPARK_FIRST_REJECTION_CE_LOW_ENTROPY_MULTIPLIER:-1.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_first_rejection_ce_low_entropy_threshold=${DSPARK_FIRST_REJECTION_CE_LOW_ENTROPY_THRESHOLD:-0.01}"
    "actor_rollout_ref.rollout.drafter.training.dspark_first_rejection_ce_max_rejected_logprob=${DSPARK_FIRST_REJECTION_CE_MAX_REJECTED_LOGPROB:--4.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_position_entropy_stratified_sampling_enabled=${DSPARK_POSITION_ENTROPY_STRATIFIED_SAMPLING_ENABLED:-false}"
    "actor_rollout_ref.rollout.drafter.training.dspark_low_entropy_max_rejected_logprob=${DSPARK_LOW_ENTROPY_MAX_REJECTED_LOGPROB:--4.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_rejected_aux_alpha=${DSPARK_REJECTED_AUX_ALPHA:-0.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_rejected_prefix_length=${DSPARK_REJECTED_PREFIX_LENGTH:-0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_rejected_first_weight_multiplier=${DSPARK_REJECTED_FIRST_WEIGHT_MULTIPLIER:-1.0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_rejected_suffix_weight_multiplier=${DSPARK_REJECTED_SUFFIX_WEIGHT_MULTIPLIER:-1.0}"
    "actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=${DRAFT_MAX_COLLECT_TOKENS_PER_REPLICA:-16384}"
    "actor_rollout_ref.rollout.drafter.training.dspark_l1_chunk_size=${DSPARK_L1_CHUNK_SIZE:-32}"
    "actor_rollout_ref.rollout.drafter.training.dspark_trace_aligned_anchors=${DSPARK_TRACE_ALIGNED_ANCHORS:-True}"
    "actor_rollout_ref.rollout.drafter.vllm.dspark_attention_mode=${DSPARK_VLLM_ATTENTION_MODE:-full}"
    "actor_rollout_ref.rollout.drafter.vllm.dspark_context_window=${DSPARK_VLLM_CONTEXT_WINDOW:-null}"
    "actor_rollout_ref.rollout.drafter.vllm.dspark_runtime_overlay_root=${DSPARK_RUNTIME_OVERLAY_ROOT:-null}"
    "actor_rollout_ref.rollout.drafter.training.dspark_attention_mode=${DSPARK_TRAINING_ATTENTION_MODE:-full}"
    "actor_rollout_ref.rollout.drafter.training.dspark_context_window=${DSPARK_TRAINING_CONTEXT_WINDOW:-null}"
    "actor_rollout_ref.rollout.drafter.training.dspark_max_window=${DSPARK_TRAINING_MAX_WINDOW:-2048}"
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${DSPARK_COLLECTION_WINDOW:-2048}"
    "actor_rollout_ref.rollout.drafter.training.dspark_use_absolute_position_ids=${DSPARK_USE_ABSOLUTE_POSITION_IDS:-True}"
    "actor_rollout_ref.rollout.drafter.training.dspark_global_layer_count=${DSPARK_GLOBAL_LAYER_COUNT:-1}"
    "actor_rollout_ref.rollout.drafter.training.dspark_global_prefix_tokens=${DSPARK_GLOBAL_PREFIX_TOKENS:-0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_global_prompt_tokens=${DSPARK_GLOBAL_PROMPT_TOKENS:-0}"
    "actor_rollout_ref.rollout.drafter.training.dspark_actor_guided_page_size=${DSPARK_ACTOR_GUIDED_PAGE_SIZE:-16}"
    "actor_rollout_ref.rollout.drafter.training.dspark_actor_guided_page_budget_tokens=${DSPARK_ACTOR_GUIDED_PAGE_BUDGET_TOKENS:-0}"
    "actor_rollout_ref.rollout.drafter.training.post_update_mean_acceptance_length_drop_abort_threshold=${DSPARK_POST_UPDATE_MEAN_ACCEPTANCE_LENGTH_DROP_ABORT_THRESHOLD:-0.4}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPEC_VERIFY_TOKENS:-7}"
)

export EXPERIMENT_PYTHON=${DSPARK_PYTHON:-/workspace/main_env_next/bin/python}
# Keep p3_01 as the default while allowing intentional experiment overrides.
export SPECO_ONLINE_STRATEGY_TAG=${SPECO_ONLINE_STRATEGY_TAG:-p3_01}

# shellcheck source=speco_vllm_experiment_common.sh
source "${PROJECT_ROOT}/tools/speco_vllm_experiment_common.sh"
speco_launch_vllm_experiment "$0" "$@"
