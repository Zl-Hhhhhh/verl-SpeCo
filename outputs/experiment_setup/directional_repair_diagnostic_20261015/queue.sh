#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
setup=${root}/outputs/experiment_setup/directional_repair_diagnostic_20261015
result=${root}/outputs/directional_repair_diagnostic_20261015/r1
mkdir -p "${result}"
printf 'waiting_for_npu\n' > "${result}/status.txt"
finish() {
    code=$?
    if ((code == 0)); then
        printf 'complete\n' > "${result}/status.txt"
    else
        printf 'failed exit=%s\n' "${code}" > "${result}/status.txt"
    fi
}
trap finish EXIT

# An external project shares these devices and does not participate in our host lock.
# Require five consecutive idle minutes before entering the normal experiment launcher.
idle_checks=0
while ((idle_checks < 5)); do
    processes=$(npu-smi info | awk '
        /Process id in container/{found=1; next}
        found && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/{n++}
        END{print n+0}')
    if ((processes == 0)); then
        ((idle_checks += 1))
    else
        idle_checks=0
    fi
    printf '%s npu_processes=%s consecutive_idle_checks=%s\n' "$(date -Is)" "${processes}" "${idle_checks}"
    if ((idle_checks < 5)); then sleep 60; fi
done

export SPECO_LAUNCH_MODE=foreground
printf 'running_probe\n' > "${result}/status.txt"
bash "${setup}/run.sh"

if [[ ! -s ${result}/paired_panel_manifest.json ]]; then
    echo "Paired diagnostic ended without a manifest" >&2
    exit 1
fi
/workspace/main_env_next/bin/python "${setup}/analyze.py" "${result}"
