#!/usr/bin/env bash
set -euo pipefail

# P2_02 recipe, both student-top16 streams, physical cards 4-7 only.
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPECO_ONLINE_STRATEGY_TAG=${SPECO_ONLINE_STRATEGY_TAG:-p4} DSPARK_TOPK_ARM=both
export TOTAL_STEPS=101 DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED=true
export SAVE_BEST_DRAFTER_CHECKPOINT=true DSPARK_DUMP_K3_LOGPROBS=true
export DSPARK_VLLM_ATTENTION_MODE=full DSPARK_VLLM_CONTEXT_WINDOW=null
export DSPARK_TRAINING_ATTENTION_MODE=full DSPARK_TRAINING_CONTEXT_WINDOW=null
export DSPARK_TRAINING_MAX_WINDOW=512 DSPARK_COLLECTION_WINDOW=512
# Historical P2_02 uses cropped-local positions, with full block-prefix masks.
export DSPARK_USE_ABSOLUTE_POSITION_IDS=false
export SPECO_PROTECTED_ACTOR_STEPS=${SPECO_PROTECTED_ACTOR_STEPS:-}
export SPECO_FOLLOW_ON_LAUNCH=${SPECO_FOLLOW_ON_LAUNCH:-0}

exec bash "${PROJECT_ROOT}/tools/run_dspark_topk_half_node.sh" \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=100 \
    'actor_rollout_ref.rollout.drafter.training.fixed_eval_export_steps=[60,80,100]' \
    actor_rollout_ref.rollout.drafter.training.fixed_eval_export_samples=128 \
    trainer.max_actor_ckpt_to_keep=null \
    "$@"
