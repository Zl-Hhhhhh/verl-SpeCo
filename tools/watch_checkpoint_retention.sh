#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 CHECKPOINT_DIR DONE_SENTINEL [POLL_SECONDS]" >&2
    exit 2
fi

checkpoint_dir=$1
done_sentinel=$2
poll_seconds=${3:-30}
protected_draft_step=
protected_actor_steps=()
if [[ -n "${SPECO_PROTECTED_ACTOR_STEPS:-}" ]]; then
    IFS=',' read -r -a protected_actor_steps <<< "${SPECO_PROTECTED_ACTOR_STEPS}"
    for protected_actor_step in "${protected_actor_steps[@]}"; do
        [[ "${protected_actor_step}" =~ ^[0-9]+$ ]] \
            || { echo "invalid SPECO_PROTECTED_ACTOR_STEPS entry: ${protected_actor_step}" >&2; exit 2; }
    done
fi

is_protected_actor_step() {
    local candidate=$1
    local protected_actor_step
    for protected_actor_step in "${protected_actor_steps[@]}"; do
        [[ "${candidate}" == "${protected_actor_step}" ]] && return 0
    done
    return 1
}

# The active vLLM process may still use the checkpoint selected at startup as
# its bootstrap path. Keep that one checkpoint until this process exits.
initial_latest_file=${checkpoint_dir}/latest_checkpointed_iteration.txt
if [[ -f "${initial_latest_file}" ]]; then
    initial_latest=$(tr -d '[:space:]' < "${initial_latest_file}")
    if [[ "${initial_latest}" =~ ^[0-9]+$ \
        && -d "${checkpoint_dir}/drafter/draft_step_${initial_latest}" ]]; then
        protected_draft_step=${initial_latest}
    fi
fi

while [[ ! -f "${done_sentinel}" ]]; do
    latest_file=${checkpoint_dir}/latest_checkpointed_iteration.txt
    if [[ -f "${latest_file}" ]]; then
        latest=$(tr -d '[:space:]' < "${latest_file}")
        best_draft_step=
        best_draft_file=${checkpoint_dir}/drafter/best_checkpointed_iteration.txt
        if [[ -s "${best_draft_file}" ]]; then
            best_draft_step=$(tr -d '[:space:]' < "${best_draft_file}")
            [[ "${best_draft_step}" =~ ^[0-9]+$ ]] || best_draft_step=
        fi
        best_acceptance_step=
        best_acceptance_file=${checkpoint_dir}/drafter/best_acceptance_checkpointed_iteration.txt
        if [[ -s "${best_acceptance_file}" ]]; then
            best_acceptance_step=$(tr -d '[:space:]' < "${best_acceptance_file}")
            [[ "${best_acceptance_step}" =~ ^[0-9]+$ ]] || best_acceptance_step=
        fi
        best_throughput_step=
        best_throughput_file=${checkpoint_dir}/drafter/best_throughput_checkpointed_iteration.txt
        if [[ -s "${best_throughput_file}" ]]; then
            best_throughput_step=$(tr -d '[:space:]' < "${best_throughput_file}")
            [[ "${best_throughput_step}" =~ ^[0-9]+$ ]] || best_throughput_step=
        fi
        latest_actor=${checkpoint_dir}/global_step_${latest}/actor
        latest_draft=${checkpoint_dir}/drafter/draft_step_${latest}
        first_model=$(find "${latest_actor}" -maxdepth 1 -type f \
            -name 'model_world_size_*_rank_*.pt' -print -quit 2>/dev/null || true)
        actor_ready=false
        if [[ "${latest}" =~ ^[0-9]+$ && -n "${first_model}" ]]; then
            model_name=${first_model##*/}
            world_size=${model_name#model_world_size_}
            world_size=${world_size%%_rank_*}
            if [[ "${world_size}" =~ ^[0-9]+$ ]]; then
                model_count=$(find "${latest_actor}" -maxdepth 1 -type f \
                    -name "model_world_size_${world_size}_rank_*.pt" | wc -l)
                extra_count=$(find "${latest_actor}" -maxdepth 1 -type f \
                    -name "extra_state_world_size_${world_size}_rank_*.pt" | wc -l)
                if [[ "${model_count}" -eq "${world_size}" && "${extra_count}" -eq "${world_size}" ]]; then
                    actor_ready=true
                fi
            fi
        fi
        latest_weight=$(find "${latest_draft}" -maxdepth 1 -type f \
            \( -name '*.safetensors' -o -name '*.bin' -o -name '*.index.json' \) \
            -print -quit 2>/dev/null || true)
        draft_required=false
        draft_ready=false
        if [[ -d "${checkpoint_dir}/drafter" ]]; then
            draft_required=true
        fi
        if [[ -f "${latest_draft}/metadata.json" && -n "${latest_weight}" ]]; then
            draft_ready=true
        fi

        if [[ "${actor_ready}" == true && ( "${draft_required}" == false || "${draft_ready}" == true ) ]]; then
            for global_path in "${checkpoint_dir}"/global_step_*; do
                [[ -d "${global_path}" ]] || continue
                global_step=${global_path##*_}
                if [[ "${global_step}" =~ ^[0-9]+$ \
                    && "${global_step}" -lt "${latest}" ]] \
                    && ! is_protected_actor_step "${global_step}"; then
                    rm -rf -- "${global_path}"
                fi
            done
        fi

        if [[ "${draft_ready}" == true ]]; then
            for draft_path in "${checkpoint_dir}"/drafter/draft_step_*; do
                [[ -d "${draft_path}" ]] || continue
                draft_step=${draft_path##*_}
                if [[ "${draft_step}" =~ ^[0-9]+$ \
                    && "${draft_step}" -lt "${latest}" \
                    && "${draft_step}" != "${protected_draft_step}" \
                    && "${draft_step}" != "${best_draft_step}" \
                    && "${draft_step}" != "${best_acceptance_step}" \
                    && "${draft_step}" != "${best_throughput_step}" ]]; then
                    rm -rf -- "${draft_path}"
                fi
            done
        fi
    fi
    sleep "${poll_seconds}"
done
