#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export SPECO_DIRECTIONAL_EXPERIMENT_ID=dspark_dapo_directional_repair_probe_5step_20261015_r1
export SPECO_DIRECTIONAL_TOTAL_STEPS=5
export SPECO_DIRECTIONAL_ENABLE_CHECKPOINTS=0
export SPECO_DIRECTIONAL_SAVE_DRAFTER_AFTER_SUCCESS=0
export SPECO_VLLM_VERIFY_ENTROPY=true
output_root=${root}/outputs/directional_repair_diagnostic_20261015/r1

[[ ! -e ${output_root}/paired_panel_manifest.json ]] || {
    echo "Refusing to overwrite a completed directional diagnostic" >&2
    exit 2
}

bash "${root}/outputs/experiment_setup/directional_pair_smoke_20261013/run.sh" \
    data.seed=17 \
    actor_rollout_ref.rollout.drafter.training.candidate_gate_enabled=true \
    ++actor_rollout_ref.rollout.drafter.training.dspark_directional_diagnostic_enabled=true \
    "++actor_rollout_ref.rollout.drafter.training.dspark_directional_diagnostic_output=${output_root}" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.SPECO_VLLM_VERIFY_ENTROPY="true"' \
    "$@"
