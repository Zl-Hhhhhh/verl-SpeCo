#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
setup=${root}/outputs/experiment_setup/directional_repair_diagnostic_20261015
output=${root}/outputs/directional_repair_diagnostic_20261015/halfnode_r1
mkdir -p "${output}"
export SPECO_DIRECTIONAL_RUN_ID
SPECO_DIRECTIONAL_RUN_ID=$(cat /proc/sys/kernel/random/uuid)
printf '%s\n' "${SPECO_DIRECTIONAL_RUN_ID}" > "${output}/run_id.txt"
# A separate session keeps the guard alive while it stops an unhealthy run.
setsid python3 "${root}/tools/monitor_experiment_memory.py" \
    --owner-pid "$$" --run-id "${SPECO_DIRECTIONAL_RUN_ID}" \
    --output "${output}" > "${output}/memory_monitor.log" 2>&1 &
monitor_pid=$!
printf '%s\n' "${monitor_pid}" > "${output}/monitor_pid.txt"
finish() {
    result=$?
    if ((result)); then
        printf 'failed exit=%s\n' "${result}" > "${output}/status.txt"
    else
        printf 'complete\n' > "${output}/status.txt"
    fi
    date -Is > "${output}/finished_at.txt"
    if [[ ! -e ${output}/memory_guard_triggered.json ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
    fi
}
trap finish EXIT
trap 'exit 143' TERM INT
[[ -s ${output}/environment_check.json ]] || { echo 'Environment preflight report missing'; exit 2; }
# Keep the entire training + analysis sequence in this detached session.
export SPECO_LAUNCH_MODE=foreground
printf 'running_actor0_to_actor5_probe\n' > "${output}/status.txt"
date -Is > "${output}/started_at.txt"
git -C "${root}" diff > "${output}/implementation.patch"
bash "${setup}/run.sh"
[[ -s ${output}/paired_panel_manifest.json ]] || { echo 'Paired manifest missing'; exit 1; }
printf 'analyzing_targets\n' > "${output}/status.txt"
source /workspace/main_env_next/bin/activate-speco-local.sh
export PYTHONPATH=${root}
/workspace/main_env_next/bin/python "${setup}/analyze.py" "${output}" > "${output}/analysis.log" 2>&1
