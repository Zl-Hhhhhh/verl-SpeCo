#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export EXPERIMENT_ID=${SPECO_DIRECTIONAL_EXPERIMENT_ID:-dspark_dapo_directional_repair_probe_5step_20261015_halfnode_r1}
export TOTAL_STEPS=5 ENABLE_CHECKPOINTS=0 SAVE_DRAFTER_AFTER_SUCCESS=0
export SPECO_VLLM_VERIFY_ENTROPY=true
export SPECO_VLLM_PAGEABLE_SLEEP=1
output_root=${SPECO_DIRECTIONAL_OUTPUT:-${root}/outputs/directional_repair_diagnostic_20261015/halfnode_r1}

# Physical cards 4--7, with two logical devices per card on this A3 host.
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export N_GPUS=8 DRAFT_WORKER_WORLD_SIZE=8 GEN_TP_SIZE=4 TRAIN_SP_SIZE=4
export DSPARK_PYTHON=/workspace/main_env_next/bin/python
export SPECO_RUNTIME_SETUP=/workspace/main_env_next/bin/activate-speco-local.sh
export SPECO_EXTRA_PYTHONPATH=/workspace/cann901_validation/runtime/cann-9.0.1/python/site-packages
export MODEL_PATH=/workspace/models/Qwen3-8B
export DRAFTER_PATH=/workspace/models/dspark_qwen3_8b_block7
export TRAIN_FILE=${output_root}/actor_train_disjoint.parquet
export TEST_FILE=/workspace/data/dapo_math_17k_verl/test.parquet
export SPECO_SKIP_GLOBAL_RAY_STOP=1 RAY_ADDRESS=local
export RUN_MODE=online DATASET_PROFILE=dapo
export TRAIN_BATCH_SIZE=64 ROLLOUT_N=5 PPO_MINI_BATCH_SIZE=16
export PPO_MICRO_BATCH_SIZE_PER_GPU=10
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=8192
export MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=16384 GPU_MEMORY_UTILIZATION=0.5
export DRAFTER_INIT_CHECKPOINT= DRAFTER_INIT_IS_STRONG=false
export DRAFT_TRAIN_STEPS=5 DRAFT_COLLECT_INTERVAL_STEPS=5 DRAFT_TRAINING_INTERVAL_STEPS=5
# Two replicas replace four. Preserve global collection/holdout budgets.
export DRAFT_BATCH_SIZE_PER_GPU=8 DRAFT_TRAIN_BATCHES_PER_CYCLE=4
export DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA=40 DRAFT_MAX_COLLECT_TOKENS_PER_REPLICA=32768
export DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA=8 DRAFT_TRAIN_UNIQUE_SAMPLES_FIRST=true
export DRAFT_LR=1e-5 DRAFT_LR_SCHEDULER_TYPE=constant DRAFT_LR_WARMUP_STEPS=0
export DRAFT_PUBLISH_ASYNC=false DRAFT_PUBLISH_INTERVAL_STEPS=0 DRAFT_CANDIDATE_GATE_ENABLED=true
export DSPARK_RESPONSE_LOSS=k3 DSPARK_REJECTED_LOSS=k3
export DSPARK_FIRST_REJECTION_CE_ALPHA=0.1 DSPARK_ACCEPTED_K3_ALPHA=1 DSPARK_REJECTED_K3_ALPHA=0
export DSPARK_LOSS_DECAY_GAMMA=0 DSPARK_CE_LOSS_ALPHA=0 DSPARK_L1_LOSS_ALPHA=0
export DSPARK_ASYMMETRIC_K3_ENABLED=false DSPARK_DYNAMIC_ACCEPT_REJECT_WEIGHTING_ENABLED=false
export DSPARK_TRACE_ALIGNED_ANCHORS=True DSPARK_USE_ABSOLUTE_POSITION_IDS=true
export DSPARK_VLLM_ATTENTION_MODE=full DSPARK_VLLM_CONTEXT_WINDOW=null
export DSPARK_TRAINING_ATTENTION_MODE=full DSPARK_TRAINING_CONTEXT_WINDOW=null
export DSPARK_TRAINING_MAX_WINDOW=512 DSPARK_COLLECTION_WINDOW=512
export SAVE_BEST_DRAFTER_CHECKPOINT=false DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED=false
export DELETE_CHECKPOINTS_AFTER_SUCCESS=0 DSPARK_DUMP_K3_LOGPROBS=false
export WANDB_MODE=${WANDB_MODE:-offline}
export SPECO_ONLINE_STRATEGY_TAG=directional_target_diagnostic_halfnode

[[ ! -e ${output_root}/paired_panel_manifest.json ]] || {
    echo "Refusing to overwrite a completed directional diagnostic" >&2
    exit 2
}

bash "${root}/tools/run_dspark_vllm_experiment.sh" \
    data.seed=17 \
    actor_rollout_ref.rollout.drafter.training.candidate_gate_enabled=true \
    ++actor_rollout_ref.rollout.drafter.training.dspark_directional_sampling_enabled=true \
    ++actor_rollout_ref.rollout.drafter.training.dspark_directional_probe_anchors=128 \
    ++actor_rollout_ref.rollout.drafter.training.dspark_directional_diagnostic_enabled=true \
    "++actor_rollout_ref.rollout.drafter.training.dspark_directional_diagnostic_output=${output_root}" \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
    actor_rollout_ref.rollout.drafter.training.fixed_eval_export_samples=0 \
    'actor_rollout_ref.rollout.drafter.training.fixed_eval_export_steps=[]' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=false' \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    trainer.resume_mode=disable \
    '++ray_kwargs.ray_init.runtime_env.env_vars.SPECO_VLLM_VERIFY_ENTROPY="true"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.SPECO_VLLM_PAGEABLE_SLEEP="1"' \
    "++ray_kwargs.ray_init.runtime_env.env_vars.SPECO_DIRECTIONAL_RUN_ID=\"${SPECO_DIRECTIONAL_RUN_ID:-manual}\"" \
    "$@"
