#!/usr/bin/env bash
# Shared W&B, checkpoint, logging, and recovery logic for vLLM NPU experiments.
# This file is sourced by the managed drafter entrypoints; do not run it directly.

speco_die() {
    echo "error: $*" >&2
    return 1
}

speco_is_true() {
    case "${1,,}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

speco_require_positive_int() {
    local name=$1
    local value=$2
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || speco_die "${name} must be a positive integer, got ${value}"
}

speco_require_positive_number() {
    local name=$1
    local value=$2
    [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
        && awk -v candidate="${value}" 'BEGIN { exit !(candidate > 0) }' \
        || speco_die "${name} must be a positive number, got ${value}"
}

speco_configure_accelerators() {
    local requested_count=$1
    local visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-}
    local device_id
    local -a device_ids=()

    speco_require_positive_int N_GPUS "${requested_count}" || return 1
    if [[ -z "${visible_devices}" ]]; then
        for ((device_id = 0; device_id < requested_count; device_id++)); do
            visible_devices+="${visible_devices:+,}${device_id}"
        done
    fi

    IFS=',' read -r -a device_ids <<< "${visible_devices}"
    (( ${#device_ids[@]} >= requested_count )) \
        || { speco_die "N_GPUS=${requested_count}, but ASCEND_RT_VISIBLE_DEVICES exposes ${#device_ids[@]} device(s)"; return 1; }

    export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
    export SPECO_ACCELERATOR_COUNT="${requested_count}"
}

speco_sanitize_runtime_path() {
    local variable_name=$1
    local forbidden_paths=${SPECO_FORBIDDEN_RUNTIME_PREFIXES:-/workspace/test_env:/tmp/vllm-ascend-v0.18.0}
    local raw_value=${!variable_name-}
    local entry prefix skip
    local cleaned_value=""
    local -a entries=() prefixes=()
    IFS=':' read -r -a entries <<< "${raw_value}"
    IFS=':' read -r -a prefixes <<< "${forbidden_paths}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue
        skip=0
        for prefix in "${prefixes[@]}"; do
            if [[ -n "${prefix}" && ("${entry}" == "${prefix}" || "${entry}" == "${prefix}/"*) ]]; then
                skip=1
                break
            fi
        done
        (( skip == 1 )) && continue
        cleaned_value+="${cleaned_value:+:}${entry}"
    done
    printf -v "${variable_name}" '%s' "${cleaned_value}"
    export "${variable_name}"
}

speco_prepare_runtime_environment() {
    local python_bin=$1
    local project_root=$2
    local runtime_setup=${SPECO_RUNTIME_SETUP:-/workspace/main_env_next/bin/activate-speco-cann901-v1.sh}
    local python_site

    [[ -x "${python_bin}" ]] || speco_die "Python environment is not executable: ${python_bin}"
    [[ -r "${runtime_setup}" ]] || speco_die "V1/A3 runtime setup is not readable: ${runtime_setup}"
    for variable_name in PATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH; do
        speco_sanitize_runtime_path "${variable_name}"
    done
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONHOME _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PYTHONHOME
    unset VLLM_VERSION
    set +u
    # shellcheck disable=SC1090
    source "${runtime_setup}"
    set -u
    export PATH="$(dirname "${python_bin}"):${PATH}"
    export PYTHONPATH="${project_root}${SPECO_EXTRA_PYTHONPATH:+:${SPECO_EXTRA_PYTHONPATH}}"
    export SPECO_PYTHON="${python_bin}"

    python_site=$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    if [[ -d "${python_site}/torch/lib" ]]; then
        export LD_LIBRARY_PATH="${python_site}/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    if [[ -d "${python_site}/torch_npu/lib" ]]; then
        export LD_LIBRARY_PATH="${python_site}/torch_npu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
    export ASCEND_USE_TORCH_NPU_FALLBACKS=${ASCEND_USE_TORCH_NPU_FALLBACKS:-1}
}

speco_safe_remove_checkpoint() {
    local path=$1
    local resolved_path
    resolved_path=$(realpath -m -- "${path}") \
        || speco_die "cannot resolve checkpoint path: ${path}"
    case "${resolved_path}" in
        / | /root | /tmp | /workspace | /workspace/verl-SpeCo | /workspace/verl-SpeCo/outputs)
            speco_die "refusing to remove unsafe checkpoint path: ${resolved_path}"
            ;;
        *)
            [[ -f "${resolved_path}/.wandb_run_id" ]] \
                || speco_die "refusing to remove checkpoint without run marker: ${resolved_path}"
            rm -rf -- "${resolved_path}"
            ;;
    esac
}

speco_copy_drafter_checkpoint() {
    local source_path=$1
    local final_path=$2
    local staging_path=$3

    [[ -d "${final_path}" ]] && return 0
    [[ ! -e "${final_path}" && ! -e "${staging_path}" ]] \
        || { speco_die "drafter archive target already exists: ${final_path}"; return 1; }
    mkdir "${staging_path}" || return 1
    if ! cp -al -- "${source_path}/." "${staging_path}/" 2>/dev/null; then
        rm -rf -- "${staging_path}"
        mkdir "${staging_path}" || return 1
        cp -a -- "${source_path}/." "${staging_path}/" \
            || { rm -rf -- "${staging_path}"; return 1; }
    fi
    mv -- "${staging_path}" "${final_path}"
}

speco_archive_final_drafter() {
    local checkpoint_dir=$1
    local archive_root=$2
    local experiment_id=$3
    local wandb_run_id=$4
    local wandb_run_path=$5
    local python_bin=${6:-python3}
    local drafter_root=${checkpoint_dir}/drafter
    local selection
    local resolved_checkpoint_dir
    local resolved_archive_root

    [[ "${experiment_id}" =~ ^[[:alnum:]_.-]+$ ]] \
        || { speco_die "unsafe experiment ID for drafter archive: ${experiment_id}"; return 1; }
    [[ "${wandb_run_id}" =~ ^[[:alnum:]_.-]+$ ]] \
        || { speco_die "unsafe W&B run ID for drafter archive: ${wandb_run_id}"; return 1; }
    resolved_checkpoint_dir=$(realpath -m -- "${checkpoint_dir}") || return 1
    resolved_archive_root=$(realpath -m -- "${archive_root}") || return 1
    case "${resolved_archive_root}" in
        "${resolved_checkpoint_dir}" | "${resolved_checkpoint_dir}/"*)
            speco_die "final drafter archive must be outside the disposable checkpoint: ${resolved_archive_root}"
            return 1
            ;;
    esac
    [[ -d "${drafter_root}" ]] \
        || { speco_die "completed run has no drafter checkpoint directory: ${drafter_root}"; return 1; }

    selection=$("${python_bin}" - "${drafter_root}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
final_marker = root / "final_checkpointed_iteration.txt"
required_step = None
if final_marker.is_file():
    try:
        required_step = int(final_marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        raise SystemExit(f"invalid final drafter marker: {final_marker}")
    if required_step < 0:
        raise SystemExit(f"invalid final drafter step in {final_marker}: {required_step}")
candidates = []
for path in root.iterdir():
    match = re.fullmatch(r"draft_step_(\d+)", path.name)
    if match is None or not path.is_dir():
        continue
    step = int(match.group(1))
    if required_step is not None and step != required_step:
        continue
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if not isinstance(metadata, dict) or metadata.get("complete") is not True:
        continue
    try:
        if int(metadata.get("step")) != step:
            continue
    except (TypeError, ValueError):
        continue
    if not (path / "config.json").is_file():
        continue
    weights = [
        item
        for pattern in ("*.safetensors", "*.bin", "*.index.json")
        for item in path.glob(pattern)
        if item.is_file() and item.stat().st_size > 0
    ]
    if not weights:
        continue
    optimizer = metadata.get("optimizer")
    if optimizer is not None:
        if not isinstance(optimizer, dict):
            continue
        relative = optimizer.get("path")
        trainer_state = optimizer.get("trainer_state_file")
        if not isinstance(relative, str) or not isinstance(trainer_state, str):
            continue
        normalized = os.path.normpath(relative)
        if os.path.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
            continue
        optimizer_dir = path / normalized
        if not (optimizer_dir / ".metadata").is_file():
            continue
        if Path(trainer_state).name != trainer_state or not (optimizer_dir / trainer_state).is_file():
            continue
    candidates.append((step, path))

if not candidates:
    expected = f" for final step {required_step}" if required_step is not None else ""
    raise SystemExit(f"no complete drafter checkpoint{expected} found under {root}")
step, path = max(candidates, key=lambda item: item[0])
print(f"{step}\t{path}")
PY
    ) || { speco_die "cannot select a complete final drafter under ${drafter_root}"; return 1; }

    local final_step=${selection%%$'\t'*}
    local source_path=${selection#*$'\t'}
    [[ "${final_step}" =~ ^[0-9]+$ && -d "${source_path}" ]] \
        || { speco_die "invalid final drafter selection: ${selection}"; return 1; }

    local run_archive_dir=${archive_root}/${experiment_id}/${wandb_run_id}
    local final_path=${run_archive_dir}/draft_step_${final_step}
    local staging_path=${run_archive_dir}/.draft_step_${final_step}.tmp.$$
    mkdir -p "${run_archive_dir}" || return 1
    speco_copy_drafter_checkpoint "${source_path}" "${final_path}" "${staging_path}" \
        || return 1

    local best_step=
    local best_source=
    local best_marker=${drafter_root}/best_checkpointed_iteration.txt
    if [[ -s "${best_marker}" ]]; then
        best_step=$(tr -d '[:space:]' < "${best_marker}")
        best_source=${drafter_root}/draft_step_${best_step}
        local best_weight=
        if [[ -d "${best_source}" ]]; then
            best_weight=$(find "${best_source}" -maxdepth 1 -type f \
                \( -name '*.safetensors' -o -name '*.bin' -o -name '*.index.json' \) \
                -size +0c -print -quit 2>/dev/null || true)
        fi
        if [[ "${best_step}" =~ ^[0-9]+$ && -d "${best_source}" \
            && -f "${best_source}/metadata.json" \
            && -f "${best_source}/config.json" && -n "${best_weight}" ]]; then
            local best_path=${run_archive_dir}/draft_step_${best_step}
            local best_staging=${run_archive_dir}/.draft_step_${best_step}.best.tmp.$$
            speco_copy_drafter_checkpoint \
                "${best_source}" "${best_path}" "${best_staging}" || return 1
            printf '%s\n' "${best_step}" > "${run_archive_dir}/best_checkpointed_iteration.txt"
            [[ -f "${drafter_root}/best_candidate.json" ]] \
                && cp -a -- "${drafter_root}/best_candidate.json" "${run_archive_dir}/best_candidate.json"
        else
            speco_die "best drafter marker does not reference a complete checkpoint: ${best_marker}"
            return 1
        fi
    fi

    local best_acceptance_step=
    local best_throughput_step=
    local dual_kind=
    for dual_kind in acceptance throughput; do
        local dual_marker=${drafter_root}/best_${dual_kind}_checkpointed_iteration.txt
        [[ -s "${dual_marker}" ]] || continue
        local dual_step
        dual_step=$(tr -d '[:space:]' < "${dual_marker}")
        local dual_source=${drafter_root}/draft_step_${dual_step}
        local dual_manifest=${drafter_root}/best_${dual_kind}.json
        local dual_weight=
        if [[ -d "${dual_source}" ]]; then
            dual_weight=$(find "${dual_source}" -maxdepth 1 -type f \
                \( -name '*.safetensors' -o -name '*.bin' -o -name '*.index.json' \) \
                -size +0c -print -quit 2>/dev/null || true)
        fi
        if [[ "${dual_step}" =~ ^[0-9]+$ && -d "${dual_source}" \
            && -f "${dual_source}/metadata.json" \
            && -f "${dual_source}/config.json" && -n "${dual_weight}" \
            && -f "${dual_manifest}" ]]; then
            local dual_path=${run_archive_dir}/draft_step_${dual_step}
            local dual_staging=${run_archive_dir}/.draft_step_${dual_step}.${dual_kind}.tmp.$$
            speco_copy_drafter_checkpoint \
                "${dual_source}" "${dual_path}" "${dual_staging}" || return 1
            printf '%s\n' "${dual_step}" \
                > "${run_archive_dir}/best_${dual_kind}_checkpointed_iteration.txt"
            cp -a -- "${dual_manifest}" "${run_archive_dir}/best_${dual_kind}.json"
            if [[ "${dual_kind}" == acceptance ]]; then
                best_acceptance_step=${dual_step}
            else
                best_throughput_step=${dual_step}
            fi
        else
            speco_die "best ${dual_kind} drafter marker does not reference a complete checkpoint: ${dual_marker}"
            return 1
        fi
    done

    local latest_tmp=${run_archive_dir}/.latest_checkpointed_iteration.tmp.$$
    printf '%s\n' "${final_step}" > "${latest_tmp}"
    mv -- "${latest_tmp}" "${run_archive_dir}/latest_checkpointed_iteration.txt"
    printf '%s\n' "${final_step}" > "${run_archive_dir}/final_checkpointed_iteration.txt"
    {
        printf 'experiment_id=%s\n' "${experiment_id}"
        printf 'wandb_run_id=%s\n' "${wandb_run_id}"
        printf 'wandb_run_path=%s\n' "${wandb_run_path}"
        printf 'checkpoint_step=%s\n' "${final_step}"
        [[ -n "${best_step}" ]] && printf 'best_checkpoint_step=%s\n' "${best_step}"
        [[ -n "${best_acceptance_step}" ]] \
            && printf 'best_acceptance_checkpoint_step=%s\n' "${best_acceptance_step}"
        [[ -n "${best_throughput_step}" ]] \
            && printf 'best_throughput_checkpoint_step=%s\n' "${best_throughput_step}"
        printf 'archived_at=%s\n' "$(date -u +%FT%TZ)"
    } > "${run_archive_dir}/archive_info"
    printf '%s\n' "${final_path}"
}

SPECO_RETENTION_PID=
SPECO_RAY_COMMAND=()
SPECO_DETACH_PARENT_LAUNCHED=0

speco_maybe_launch_detached() {
    local entrypoint=$1
    shift

    SPECO_DETACH_PARENT_LAUNCHED=0
    local launch_mode=${SPECO_LAUNCH_MODE:-detached}
    launch_mode=${launch_mode,,}
    case "${launch_mode}" in
        detached | foreground) ;;
        *) speco_die "SPECO_LAUNCH_MODE must be detached or foreground, got ${launch_mode}"; return 1 ;;
    esac

    # Keep dry runs and the re-executed child in the current shell. The latter
    # prevents the detached launcher from recursively spawning itself.
    if [[ "${launch_mode}" == foreground ]] \
        || speco_is_true "${DRY_RUN:-0}" \
        || speco_is_true "${SPECO_DETACHED_CHILD:-0}"; then
        return 0
    fi

    entrypoint=$(realpath -m -- "${entrypoint}") \
        || { speco_die "cannot resolve experiment entrypoint: ${entrypoint}"; return 1; }
    [[ -f "${entrypoint}" ]] \
        || { speco_die "experiment entrypoint is not readable: ${entrypoint}"; return 1; }
    command -v setsid >/dev/null 2>&1 \
        || { speco_die "setsid is required for detached managed experiments"; return 1; }

    local project_root
    project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    local managed_root=${SPECO_MANAGED_RUN_ROOT:-${project_root}/outputs/managed_runs}
    local managed_run_id=${SPECO_MANAGED_RUN_ID:-${SPECO_ALGORITHM:-speco}_$(date -u +%Y%m%dT%H%M%SZ)_$$}
    [[ "${managed_run_id}" =~ ^[[:alnum:]_.-]+$ ]] \
        || { speco_die "SPECO_MANAGED_RUN_ID must use only letters, numbers, _, ., or -"; return 1; }

    local managed_run_dir=${managed_root}/${managed_run_id}
    [[ ! -e "${managed_run_dir}" ]] \
        || { speco_die "managed run directory already exists: ${managed_run_dir}"; return 1; }
    mkdir -p "${managed_run_dir}" || return 1

    local launcher_log=${managed_run_dir}/launcher.log
    printf '%s\n' "$(date -u +%FT%TZ)" > "${managed_run_dir}/started_at"
    printf 'starting\n' > "${managed_run_dir}/state"
    printf 'bash' > "${managed_run_dir}/command"
    printf ' %q' "${entrypoint}" "$@" >> "${managed_run_dir}/command"
    printf '\n' >> "${managed_run_dir}/command"

    # The wrapper is a separate session and inherits nohup's ignored SIGHUP.
    # It owns lifecycle files while the entrypoint still owns Ray cleanup.
    SPECO_DETACHED_CHILD=1 SPECO_MANAGED_RUN_DIR="${managed_run_dir}" \
        nohup setsid --fork bash -c '
            managed_run_dir=$1
            shift
            finish_managed_run() {
                status=$1
                printf "%s\\n" "${status}" > "${managed_run_dir}/exit_status"
                if (( status == 0 )); then
                    printf "completed\\n" > "${managed_run_dir}/state"
                elif (( status == 143 )); then
                    printf "stopped\\n" > "${managed_run_dir}/state"
                else
                    printf "failed\\n" > "${managed_run_dir}/state"
                fi
                date -u +%FT%TZ > "${managed_run_dir}/finished_at"
            }
            printf "%s\\n" "$$" > "${managed_run_dir}/pid"
            printf "running\\n" > "${managed_run_dir}/state"
            trap "finish_managed_run 143; exit 143" TERM INT
            "$@"
            status=$?
            finish_managed_run "${status}"
            exit "${status}"
        ' speco-managed-run "${managed_run_dir}" bash "${entrypoint}" "$@" \
        > "${launcher_log}" 2>&1 < /dev/null &

    local attempts
    for ((attempts = 0; attempts < 30; attempts++)); do
        [[ -s "${managed_run_dir}/pid" || -s "${managed_run_dir}/exit_status" ]] && break
        sleep 0.1
    done
    SPECO_DETACH_PARENT_LAUNCHED=1

    local early_status=
    if [[ -s "${managed_run_dir}/exit_status" ]]; then
        IFS= read -r early_status < "${managed_run_dir}/exit_status"
        if [[ "${early_status}" =~ ^[0-9]+$ ]] && (( early_status != 0 )); then
            echo "Managed SPECO launch failed before training setup; inspect ${launcher_log}" >&2
            return "${early_status}"
        fi
    fi

    echo "Managed SPECO run started: ${managed_run_id}"
    [[ -s "${managed_run_dir}/pid" ]] && echo "PID: $(< "${managed_run_dir}/pid")"
    echo "Launcher log: ${launcher_log}"
    echo "Status: bash ${project_root}/tools/speco_managed_run.sh status ${managed_run_id}"
    echo "Follow: bash ${project_root}/tools/speco_managed_run.sh follow ${managed_run_id}"

    # The observer remains attached to the invoking terminal, while the
    # training session does not. Closing the terminal therefore stops only
    # tail, not the managed experiment.
    if speco_is_true "${SPECO_FOLLOW_ON_LAUNCH:-1}"; then
        local follow_wait_seconds=${SPECO_FOLLOW_WAIT_SECONDS:-60}
        [[ "${follow_wait_seconds}" =~ ^[0-9]+$ ]] \
            || { speco_die "SPECO_FOLLOW_WAIT_SECONDS must be a non-negative integer"; return 1; }
        local training_log=
        local follow_attempt
        for ((follow_attempt = 0; follow_attempt < follow_wait_seconds; follow_attempt++)); do
            if [[ -s "${managed_run_dir}/training_log" ]]; then
                IFS= read -r training_log < "${managed_run_dir}/training_log"
                break
            fi
            [[ -s "${managed_run_dir}/exit_status" ]] && break
            sleep 1
        done
        if [[ -n "${training_log}" ]]; then
            echo "Streaming training log; closing this terminal will not stop ${managed_run_id}."
            tail -n "${SPECO_FOLLOW_LINES:-80}" -F --pid="$(< "${managed_run_dir}/pid")" "${training_log}" || true
        elif [[ -s "${managed_run_dir}/exit_status" ]]; then
            echo "Managed run ended before creating its training log; showing launcher output." >&2
            tail -n "${SPECO_FOLLOW_LINES:-80}" "${launcher_log}" >&2 || true
        else
            echo "Training log is not ready yet; follow it with the command above." >&2
        fi
    fi

    # When this terminal stayed attached long enough to observe completion,
    # preserve the real training status for callers and shell automation.
    local final_status=
    if [[ -s "${managed_run_dir}/exit_status" ]]; then
        IFS= read -r final_status < "${managed_run_dir}/exit_status"
        if [[ "${final_status}" =~ ^[0-9]+$ ]] && (( final_status != 0 )); then
            echo "Managed SPECO run ${managed_run_id} exited with status ${final_status}." >&2
            return "${final_status}"
        fi
    fi
}

speco_record_managed_run_metadata() {
    local experiment_id=$1
    local log_file=$2
    local checkpoint_dir=$3
    local wandb_run_id=$4

    if ! speco_is_true "${SPECO_DETACHED_CHILD:-0}"; then
        return 0
    fi

    local managed_run_dir=${SPECO_MANAGED_RUN_DIR:-}
    [[ -d "${managed_run_dir}" ]] \
        || { speco_die "detached managed run directory is unavailable: ${managed_run_dir}"; return 1; }
    printf '%s\n' "${log_file}" > "${managed_run_dir}/training_log"
    {
        printf 'experiment_id=%s\n' "${experiment_id}"
        printf 'checkpoint_dir=%s\n' "${checkpoint_dir}"
        printf 'wandb_run_id=%s\n' "${wandb_run_id}"
    } > "${managed_run_dir}/metadata"
}

speco_launch_vllm_experiment() {
    local entrypoint=$1
    shift

    speco_maybe_launch_detached "${entrypoint}" "$@" || return 1
    if speco_is_true "${SPECO_DETACH_PARENT_LAUNCHED:-0}"; then
        return 0
    fi
    if declare -p SPECO_FINAL_HYDRA_OVERRIDES >/dev/null 2>&1; then
        speco_run_vllm_experiment "$@" "${SPECO_FINAL_HYDRA_OVERRIDES[@]}"
    else
        speco_run_vllm_experiment "$@"
    fi
}

speco_configure_ray_command() {
    local python_bin=$1
    local requested_ray_bin=${RAY_BIN:-}
    local ray_bin=${requested_ray_bin:-$(dirname "${python_bin}")/ray}

    if [[ -x "${ray_bin}" ]]; then
        SPECO_RAY_COMMAND=("${ray_bin}")
        return 0
    fi

    # Some minimal Python environments install Ray's package but omit its
    # console-script wrapper. Invoke Ray through the selected interpreter so
    # cleanup cannot accidentally use a Ray CLI from another environment.
    if [[ -z "${requested_ray_bin}" ]] \
        && "${python_bin}" -c 'from ray.scripts.scripts import main' >/dev/null 2>&1; then
        SPECO_RAY_COMMAND=("${python_bin}" -m ray.scripts.scripts)
        return 0
    fi

    speco_die "Ray CLI is unavailable for Python environment: ${python_bin}" || return 1
}

speco_cleanup_experiment() {
    if [[ -n "${SPECO_RETENTION_PID}" ]]; then
        kill "${SPECO_RETENTION_PID}" >/dev/null 2>&1 || true
        wait "${SPECO_RETENTION_PID}" 2>/dev/null || true
        SPECO_RETENTION_PID=
    fi
    if (( ${#SPECO_RAY_COMMAND[@]} > 0 )) && ! speco_is_true "${SPECO_SKIP_GLOBAL_RAY_STOP:-0}"; then
        "${SPECO_RAY_COMMAND[@]}" stop --force >/dev/null 2>&1 || true
    fi
}

speco_run_vllm_experiment() {
    : "${SPECO_ALGORITHM:?SPECO_ALGORITHM is required}"
    : "${SPECO_BASE_LAUNCHER:?SPECO_BASE_LAUNCHER is required}"
    : "${SPECO_DEFAULT_DRAFTER_PATH:?SPECO_DEFAULT_DRAFTER_PATH is required}"

    local project_root
    project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    local experiment_python=${EXPERIMENT_PYTHON:-/workspace/main_env_next/bin/python}
    local custom_reward_path=/workspace/data/drafter_domains/reward.py
    speco_prepare_runtime_environment "${experiment_python}" "${project_root}"

    # Dataset profiles keep train/eval separation and length limits consistent.
    # Explicit environment variables still override every profile default.
    local dataset_profile=${DATASET_PROFILE:-dapo}
    dataset_profile=${dataset_profile,,}
    dataset_profile=${dataset_profile//-/_}
    local dataset_train_default
    local dataset_test_default
    local profile_max_prompt_length
    local profile_max_response_length
    local profile_test_freq
    local profile_val_before_train
    local profile_uses_custom_reward=false
    local profile_uses_code_reward=false
    local profile_description
    local profile_wandb_project
    case "${dataset_profile}" in
        dapo)
            dataset_train_default=/workspace/data/dapo_math_17k_verl/train.parquet
            dataset_test_default=/workspace/data/dapo_math_17k_verl/test.parquet
            profile_max_prompt_length=512
            profile_max_response_length=8192
            profile_test_freq=-1
            profile_val_before_train=false
            profile_description="DAPO math training and validation"
            profile_wandb_project=verl_speco_dapo
            ;;
        taco_taco)
            dataset_train_default=/workspace/data/drafter_domains/verl/code/taco_cobalt/train.parquet
            dataset_test_default=/workspace/data/drafter_domains/verl/code/taco_cobalt/validation.parquet
            profile_max_prompt_length=2048
            profile_max_response_length=8192
            profile_test_freq=10
            profile_val_before_train=true
            profile_uses_custom_reward=true
            profile_uses_code_reward=true
            profile_description="TACO-Cobalt training with its 250-problem validation split"
            profile_wandb_project=verl_speco_taco
            ;;
        taco_lcb)
            dataset_train_default=/workspace/data/drafter_domains/verl/code/taco_cobalt/train.parquet
            dataset_test_default=/workspace/data/drafter_domains/verl/code/eval/livecodebench/v6_only.parquet
            profile_max_prompt_length=2048
            profile_max_response_length=8192
            profile_test_freq=10
            profile_val_before_train=true
            profile_uses_custom_reward=true
            profile_uses_code_reward=true
            profile_description="TACO-Cobalt training with 175-problem LiveCodeBench v6-only validation"
            profile_wandb_project=verl_speco_taco
            ;;
        taco_lcb_full)
            dataset_train_default=/workspace/data/drafter_domains/verl/code/taco_cobalt/train.parquet
            dataset_test_default=/workspace/data/drafter_domains/verl/code/eval/livecodebench/release_v6.parquet
            profile_max_prompt_length=2048
            profile_max_response_length=8192
            profile_test_freq=-1
            profile_val_before_train=true
            profile_uses_custom_reward=true
            profile_uses_code_reward=true
            profile_description="TACO-Cobalt with full 1,055-problem LiveCodeBench final evaluation"
            profile_wandb_project=verl_speco_taco
            ;;
        wildsci_gpqa)
            dataset_train_default=/workspace/data/drafter_domains/verl/science/wildsci_all_aligned/train.parquet
            dataset_test_default=/workspace/data/drafter_domains/verl/science/eval/gpqa_diamond/test.parquet
            profile_max_prompt_length=2048
            profile_max_response_length=4096
            profile_test_freq=10
            profile_val_before_train=true
            profile_uses_custom_reward=true
            profile_description="WildSci All-Aligned training with GPQA-Diamond validation"
            profile_wandb_project=verl_speco_wildsci
            ;;
        wildsci_mmlu)
            dataset_train_default=/workspace/data/drafter_domains/verl/science/wildsci_all_aligned/train.parquet
            dataset_test_default=/workspace/data/drafter_domains/verl/science/eval/mmlu_pro_stem/test.parquet
            profile_max_prompt_length=2048
            profile_max_response_length=2048
            profile_test_freq=50
            profile_val_before_train=false
            profile_uses_custom_reward=true
            profile_description="WildSci All-Aligned training with full MMLU-Pro STEM validation"
            profile_wandb_project=verl_speco_wildsci
            ;;
        *)
            speco_die "unknown DATASET_PROFILE=${dataset_profile}; expected dapo, taco_taco, taco_lcb, taco_lcb_full, wildsci_gpqa, or wildsci_mmlu" \
                || return 1
            ;;
    esac

    # Frequently changed experiment inputs.
    local run_mode=${RUN_MODE:-online}                  # online or offline (fixed drafter)
    local total_steps=${TOTAL_STEPS:-101}
    local enable_checkpoints=${ENABLE_CHECKPOINTS:-1}
    local save_freq=${SAVE_FREQ:-10}
    local save_offset=${SAVE_OFFSET:-9}
    local train_batch_size=${TRAIN_BATCH_SIZE:-64}
    local draft_workers=${DRAFT_WORKER_WORLD_SIZE:-16}  # online only; must be divisible by GEN_TP_SIZE
    local model_path=${MODEL_PATH:-/workspace/models/Qwen3-8B}
    local drafter_path=${DRAFTER_PATH:-${SPECO_DEFAULT_DRAFTER_PATH}}
    local default_drafter_init_checkpoint=
    local default_drafter_init_label=
    # if [[ "${SPECO_ALGORITHM}" == dspark && "${run_mode}" == online ]]; then
    #     default_drafter_init_checkpoint=/workspace/verl-SpeCo/outputs/drafters/qwen3_8b_dspark_online_w16_100step_b64__p2_02/bc4c6e79b32f/draft_step_101
    #     default_drafter_init_label=p2_02_v1
    # fi
    local drafter_init_checkpoint=${DRAFTER_INIT_CHECKPOINT-${default_drafter_init_checkpoint}}
    local drafter_init_label=
    if [[ ${DRAFTER_INIT_LABEL+x} ]]; then
        drafter_init_label=${DRAFTER_INIT_LABEL}
    elif [[ -n "${default_drafter_init_checkpoint}" \
        && "${drafter_init_checkpoint}" == "${default_drafter_init_checkpoint}" ]]; then
        drafter_init_label=${default_drafter_init_label}
    fi
    local drafter_init_step=
    local drafter_init_active=0
    local train_file=${TRAIN_FILE:-${dataset_train_default}}
    local test_file=${TEST_FILE:-${dataset_test_default}}

    # Main-model and rollout sizing knobs.
    local n_gpus=${N_GPUS:-16}
    local gen_tp=${GEN_TP_SIZE:-4}
    local train_sp=${TRAIN_SP_SIZE:-4}
    local rollout_n=${ROLLOUT_N:-5}
    local max_prompt_length=${MAX_PROMPT_LENGTH:-${profile_max_prompt_length}}
    local max_response_length=${MAX_RESPONSE_LENGTH:-${profile_max_response_length}}
    local max_num_seqs=${MAX_NUM_SEQS:-128}
    local max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}
    local gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6}
    local filter_workers=${FILTER_OVERLONG_PROMPTS_WORKERS:-16}
    local ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
    local ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}
    local test_freq=${TEST_FREQ:-${profile_test_freq}}
    local val_before_train=${VAL_BEFORE_TRAIN:-${profile_val_before_train}}
    local validation_data_dir=${VALIDATION_DATA_DIR:-}
    local sandbox_fusion_url=${SANDBOX_FUSION_URL:-}
    local use_local_sandbox_fusion=${SPECO_USE_LOCAL_SANDBOX_FUSION:-true}
    local local_sandbox_fusion_url=${SPECO_LOCAL_SANDBOX_FUSION_URL:-http://127.0.0.1:8080/run_code}
    local local_sandbox_start=/workspace/SandboxFusion/scripts/start_speco_service.sh
    local sandbox_memory_limit_mb=${SANDBOX_MEMORY_LIMIT_MB:-1024}
    local sandbox_max_concurrent=${SANDBOX_MAX_CONCURRENT:-16}

    # Code rewards must execute outside Ray's Python workers. On this single-node
    # setup, start the local SandboxFusion service unless a remote endpoint was
    # supplied explicitly. Disable only to intentionally retain local PRIME.
    if speco_is_true "${profile_uses_code_reward}" && [[ -z "${sandbox_fusion_url}" ]] \
        && speco_is_true "${use_local_sandbox_fusion}"; then
        [[ -x "${local_sandbox_start}" ]] \
            || { speco_die "local SandboxFusion launcher is unavailable: ${local_sandbox_start}"; return 1; }
        "${local_sandbox_start}" >&2 || return 1
        sandbox_fusion_url=${local_sandbox_fusion_url}
    fi

    # Online-drafter knobs. Synchronous publish plus a smaller bucket minimizes
    # the host-memory peak while keeping all rollout replicas up to date.
    local strong_checkpoint_init=0
    if [[ ${DRAFTER_INIT_IS_STRONG+x} ]]; then
        if speco_is_true "${DRAFTER_INIT_IS_STRONG}"; then
            strong_checkpoint_init=1
        fi
    elif [[ "${run_mode}" == online \
        && -n "${default_drafter_init_checkpoint}" \
        && "${drafter_init_checkpoint}" == "${default_drafter_init_checkpoint}" ]]; then
        strong_checkpoint_init=1
    fi
    local default_draft_train_steps=5
    local default_draft_collect_interval=5
    local default_draft_training_interval=5
    local default_draft_train_batches=4
    local default_draft_lr=1e-5
    if (( strong_checkpoint_init == 1 )); then
        default_draft_train_steps=5
        default_draft_collect_interval=10
        default_draft_training_interval=10
        default_draft_train_batches=4
        default_draft_lr=5e-6
    fi
    local draft_train_steps=${DRAFT_TRAIN_STEPS:-${default_draft_train_steps}}
    local draft_collect_interval=${DRAFT_COLLECT_INTERVAL_STEPS:-${default_draft_collect_interval}}
    local draft_training_interval=${DRAFT_TRAINING_INTERVAL_STEPS:-${default_draft_training_interval}}
    local draft_publish_interval=${DRAFT_PUBLISH_INTERVAL_STEPS:-0}
    local draft_batch_size=${DRAFT_BATCH_SIZE_PER_GPU:-4}
    local draft_train_batches=${DRAFT_TRAIN_BATCHES_PER_CYCLE:-${default_draft_train_batches}}
    local draft_max_samples=${DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA:-32}
    local draft_lr=${DRAFT_LR:-${default_draft_lr}}
    local draft_lr_scheduler=${DRAFT_LR_SCHEDULER_TYPE:-$([[ ${strong_checkpoint_init} == 1 ]] && echo global_cosine || echo constant)}
    local draft_lr_warmup_steps=${DRAFT_LR_WARMUP_STEPS:-$([[ ${strong_checkpoint_init} == 1 ]] && echo 2 || echo 0)}
    local draft_lr_decay_steps=${DRAFT_LR_DECAY_STEPS:-10}
    local draft_min_lr_ratio=${DRAFT_MIN_LR_RATIO:-0.1}
    local draft_unique_samples_first=${DRAFT_TRAIN_UNIQUE_SAMPLES_FIRST:-true}
    local draft_fixed_eval_samples=${DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA:-4}
    local publish_async=${DRAFT_PUBLISH_ASYNC:-false}
    local bucket_mb=${DRAFT_UPDATE_BUCKET_MB:-256}
    local default_candidate_gate_enabled=false
    [[ "${SPECO_ALGORITHM}" == dspark ]] && default_candidate_gate_enabled=false
    local candidate_gate_enabled=${DRAFT_CANDIDATE_GATE_ENABLED:-${default_candidate_gate_enabled}}
    local candidate_gate_alphas=${DRAFT_CANDIDATE_GATE_ALPHAS:-'[0.0,0.25,0.5,1.0]'}
    local candidate_gate_samples=${DRAFT_CANDIDATE_GATE_SAMPLES:-16}
    local candidate_gate_max_tokens=${DRAFT_CANDIDATE_GATE_MAX_TOKENS:-256}
    local candidate_gate_min_acceptance_delta=${DRAFT_CANDIDATE_GATE_MIN_ACCEPTANCE_DELTA:-0.0}
    local candidate_gate_max_latency_regression=${DRAFT_CANDIDATE_GATE_MAX_MS_PER_TOKEN_REGRESSION:-0.15}
    local candidate_max_fixed_loss_delta=${DRAFT_CANDIDATE_MAX_FIXED_LOSS_DELTA:-0.0}
    local candidate_max_grad_norm=${DRAFT_CANDIDATE_MAX_GRAD_NORM:-50.0}
    local candidate_grad_norm_multiplier=${DRAFT_CANDIDATE_GRAD_NORM_ROLLING_MULTIPLIER:-5.0}
    local candidate_max_rejected_clamp_ratio=${DRAFT_CANDIDATE_MAX_REJECTED_K3_CLAMP_RATIO:-0.5}
    local candidate_max_rejected_k3_median=${DRAFT_CANDIDATE_MAX_REJECTED_K3_MEDIAN:-9.5}
    local save_best_drafter_checkpoint=${SAVE_BEST_DRAFTER_CHECKPOINT:-true}
    local best_checkpoint_min_acceptance_delta=${BEST_CHECKPOINT_MIN_ACCEPTANCE_DELTA:-0.0}
    local dual_metric_best_checkpoint_enabled=${DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED:-false}
    local best_checkpoint_min_throughput_relative_delta=${BEST_CHECKPOINT_MIN_THROUGHPUT_RELATIVE_DELTA:-0.0}
    local dual_metric_best_checkpoint_include_optimizer=${DRAFT_DUAL_METRIC_BEST_CHECKPOINT_INCLUDE_OPTIMIZER:-false}

    # DSpark loss coefficients. These defaults reproduce Draft-OPD's direct
    # supervised recipe: sampled K3 on both streams, joint token normalization,
    # no response-position decay, and 0.9 decay on rejected draft suffixes.
    local dspark_loss_decay_gamma=${DSPARK_LOSS_DECAY_GAMMA:-0.0}
    local dspark_ce_loss_alpha=${DSPARK_CE_LOSS_ALPHA:-0.0}
    local dspark_l1_loss_alpha=${DSPARK_L1_LOSS_ALPHA:-0.0}
    local dspark_accepted_k3_alpha=${DSPARK_ACCEPTED_K3_ALPHA:-1.0}
    local dspark_accepted_fkl_alpha=${DSPARK_ACCEPTED_FKL_ALPHA:-0.0}
    local dspark_rejected_k3_alpha=${DSPARK_REJECTED_K3_ALPHA:-1.0}
    local dspark_response_stream_weight=${DSPARK_RESPONSE_STREAM_WEIGHT:-1.0}
    local dspark_rejected_stream_weight=${DSPARK_REJECTED_STREAM_WEIGHT:-0.5}
    local dspark_rejected_position_decay=${DSPARK_REJECTED_POSITION_DECAY:-0.8}
    local dspark_rejected_log_ratio_clamp=${DSPARK_REJECTED_LOG_RATIO_CLAMP:-20.0}
    local dspark_rejected_loss_max_clamp=${DSPARK_REJECTED_LOSS_MAX_CLAMP:-10.0}
    local dspark_asymmetric_k3_enabled=${DSPARK_ASYMMETRIC_K3_ENABLED:-true}
    local dspark_accepted_asymmetric_k3_enabled=${DSPARK_ACCEPTED_ASYMMETRIC_K3_ENABLED:-${dspark_asymmetric_k3_enabled}}
    local dspark_rejected_asymmetric_k3_enabled=${DSPARK_REJECTED_ASYMMETRIC_K3_ENABLED:-${dspark_asymmetric_k3_enabled}}
    local dspark_asymmetric_k3_positive_threshold=${DSPARK_ASYMMETRIC_K3_POSITIVE_THRESHOLD:-2.0}
    local dspark_dynamic_accept_reject_weighting_enabled=${DSPARK_DYNAMIC_ACCEPT_REJECT_WEIGHTING_ENABLED:-false}
    local dspark_dynamic_stream_weight_min_multiplier=${DSPARK_DYNAMIC_STREAM_WEIGHT_MIN_MULTIPLIER:-0.25}
    local dspark_dynamic_stream_weight_max_multiplier=${DSPARK_DYNAMIC_STREAM_WEIGHT_MAX_MULTIPLIER:-4.0}
    local dspark_k3_distribution_bins=${DSPARK_K3_DISTRIBUTION_BINS:-512}
    local dspark_dump_k3_logprobs=${DSPARK_DUMP_K3_LOGPROBS:-true}
    local dspark_dump_k3_logprobs_dir=${DSPARK_DUMP_K3_LOGPROBS_DIR:-}
    local dspark_dump_k3_logprobs_interval=${DSPARK_DUMP_K3_LOGPROBS_INTERVAL_STEPS:-1}
    local dspark_dump_k3_logprobs_max_tokens=${DSPARK_DUMP_K3_LOGPROBS_MAX_TOKENS_PER_STREAM:-0}

    # Save actor optimizer shards with the model and extra training state so
    # resumed runs preserve Adam moments as well as RNG, scheduler, and dataloader state.
    local checkpoint_contents=${ACTOR_CHECKPOINT_CONTENTS:-'["model","optimizer","extra"]'}
    local delete_after_success=${DELETE_CHECKPOINTS_AFTER_SUCCESS:-1}
    local save_drafter_after_success=${SAVE_DRAFTER_AFTER_SUCCESS:-1}
    local ray_num_cpus=${RAY_NUM_CPUS:-64}
    local total_epochs=${TOTAL_EPOCHS:-100}
    local resume_mode=auto

    run_mode=${run_mode,,}
    [[ "${run_mode}" == online || "${run_mode}" == offline ]] \
        || speco_die "RUN_MODE must be online or offline, got ${run_mode}"
    if [[ -n "${drafter_init_checkpoint}" ]]; then
        [[ "${run_mode}" == online ]] \
            || { speco_die "DRAFTER_INIT_CHECKPOINT is only valid with RUN_MODE=online; use DRAFTER_PATH for fixed-drafter evaluation"; return 1; }
        local drafter_init_details
        if ! drafter_init_details=$(PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${experiment_python}" - "${drafter_init_checkpoint}" "${SPECO_ALGORITHM}" <<'PY'
import json
import os
import sys

from verl_speco.trainer.checkpoint import (
    get_drafter_checkpoint_step,
    is_pretrained_drafter_checkpoint,
)

checkpoint = os.path.realpath(sys.argv[1])
algorithm = sys.argv[2].lower()
if "\n" in checkpoint or "\t" in checkpoint:
    raise SystemExit("drafter checkpoint paths cannot contain tabs or newlines")
if not is_pretrained_drafter_checkpoint(checkpoint):
    raise SystemExit(
        f"not a complete, loadable drafter checkpoint: {checkpoint}"
    )
step = get_drafter_checkpoint_step(checkpoint)
if step is None:
    raise SystemExit(
        "DRAFTER_INIT_CHECKPOINT must be a managed trained checkpoint with "
        f"metadata.json: {checkpoint}"
    )

config = {}
for name in ("speco_training_config.json", "config.json"):
    path = os.path.join(checkpoint, name)
    if not os.path.isfile(path):
        continue
    with open(path, "r", encoding="utf-8") as stream:
        candidate = json.load(stream)
    if isinstance(candidate, dict):
        config.update(candidate)
checkpoint_algorithm = str(
    config.get("speco_training_model_type") or config.get("model_type") or ""
).lower()
if algorithm in {"dflash", "dspark"} and checkpoint_algorithm != algorithm:
    raise SystemExit(
        f"drafter checkpoint algorithm mismatch: expected {algorithm}, "
        f"found {checkpoint_algorithm or 'unknown'} in {checkpoint}"
    )
print(f"{checkpoint}\t{step}")
PY
        ); then
            speco_die "invalid DRAFTER_INIT_CHECKPOINT=${drafter_init_checkpoint}" || return 1
        fi
        IFS=$'\t' read -r drafter_init_checkpoint drafter_init_step \
            <<< "${drafter_init_details}"
        drafter_path=${drafter_init_checkpoint}
        if [[ -z "${drafter_init_label}" ]]; then
            drafter_init_label=s${drafter_init_step}
        fi
        [[ "${drafter_init_label}" =~ ^[[:alnum:]_.-]+$ ]] \
            || { speco_die "DRAFTER_INIT_LABEL must use only letters, numbers, _, ., or -"; return 1; }
    elif [[ -n "${drafter_init_label}" ]]; then
        speco_die "DRAFTER_INIT_LABEL requires DRAFTER_INIT_CHECKPOINT" || return 1
    fi
    local archive_final_drafter=0
    if [[ "${run_mode}" == online ]] && speco_is_true "${enable_checkpoints}" \
        && speco_is_true "${save_drafter_after_success}"; then
        archive_final_drafter=1
    fi
    # New online sampling gets an isolated checkpoint namespace. Use a custom
    # tag only when an online variant needs a distinct experiment identity.
    local online_strategy_tag=
    if [[ "${run_mode}" == online ]]; then
        online_strategy_tag=${SPECO_ONLINE_STRATEGY_TAG:-p1}
        [[ "${online_strategy_tag}" =~ ^[[:alnum:]_.-]+$ ]] \
            || { speco_die "SPECO_ONLINE_STRATEGY_TAG must use only letters, numbers, _, ., or -"; return 1; }
    fi
    local online_strategy_suffix=
    [[ -n "${online_strategy_tag}" ]] && online_strategy_suffix=__${online_strategy_tag}
    local drafter_init_suffix=
    [[ -n "${drafter_init_checkpoint}" ]] \
        && drafter_init_suffix=__cold_${drafter_init_label}
    for pair in \
        "TOTAL_STEPS:${total_steps}" \
        "TRAIN_BATCH_SIZE:${train_batch_size}" \
        "N_GPUS:${n_gpus}" \
        "GEN_TP_SIZE:${gen_tp}" \
        "TRAIN_SP_SIZE:${train_sp}" \
        "ROLLOUT_N:${rollout_n}" \
        "MAX_PROMPT_LENGTH:${max_prompt_length}" \
        "MAX_RESPONSE_LENGTH:${max_response_length}" \
        "DRAFT_TRAIN_STEPS:${draft_train_steps}" \
        "DRAFT_BATCH_SIZE_PER_GPU:${draft_batch_size}" \
        "DRAFT_TRAIN_BATCHES_PER_CYCLE:${draft_train_batches}" \
        "DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA:${draft_max_samples}" \
        "DSPARK_K3_DISTRIBUTION_BINS:${dspark_k3_distribution_bins}" \
        "DSPARK_DUMP_K3_LOGPROBS_INTERVAL_STEPS:${dspark_dump_k3_logprobs_interval}" \
        "DRAFT_UPDATE_BUCKET_MB:${bucket_mb}"; do
        speco_require_positive_int "${pair%%:*}" "${pair#*:}"
    done
    [[ "${draft_fixed_eval_samples}" =~ ^[0-9]+$ ]] \
        || { speco_die "DRAFT_FIXED_EVAL_SAMPLES_PER_REPLICA must be a non-negative integer, got ${draft_fixed_eval_samples}"; return 1; }
    [[ "${dspark_dump_k3_logprobs_max_tokens}" =~ ^[0-9]+$ ]] \
        || { speco_die "DSPARK_DUMP_K3_LOGPROBS_MAX_TOKENS_PER_STREAM must be a non-negative integer, got ${dspark_dump_k3_logprobs_max_tokens}"; return 1; }
    case "${dspark_asymmetric_k3_enabled,,}" in
        1 | true | yes | on) dspark_asymmetric_k3_enabled=true ;;
        0 | false | no | off) dspark_asymmetric_k3_enabled=false ;;
        *) speco_die "DSPARK_ASYMMETRIC_K3_ENABLED must be a boolean, got ${dspark_asymmetric_k3_enabled}"; return 1 ;;
    esac
    case "${dspark_accepted_asymmetric_k3_enabled,,}" in
        1 | true | yes | on) dspark_accepted_asymmetric_k3_enabled=true ;;
        0 | false | no | off) dspark_accepted_asymmetric_k3_enabled=false ;;
        *) speco_die "DSPARK_ACCEPTED_ASYMMETRIC_K3_ENABLED must be a boolean, got ${dspark_accepted_asymmetric_k3_enabled}"; return 1 ;;
    esac
    case "${dspark_rejected_asymmetric_k3_enabled,,}" in
        1 | true | yes | on) dspark_rejected_asymmetric_k3_enabled=true ;;
        0 | false | no | off) dspark_rejected_asymmetric_k3_enabled=false ;;
        *) speco_die "DSPARK_REJECTED_ASYMMETRIC_K3_ENABLED must be a boolean, got ${dspark_rejected_asymmetric_k3_enabled}"; return 1 ;;
    esac
    case "${dspark_dynamic_accept_reject_weighting_enabled,,}" in
        1 | true | yes | on) dspark_dynamic_accept_reject_weighting_enabled=true ;;
        0 | false | no | off) dspark_dynamic_accept_reject_weighting_enabled=false ;;
        *) speco_die "DSPARK_DYNAMIC_ACCEPT_REJECT_WEIGHTING_ENABLED must be a boolean, got ${dspark_dynamic_accept_reject_weighting_enabled}"; return 1 ;;
    esac
    speco_require_positive_number \
        DSPARK_ASYMMETRIC_K3_POSITIVE_THRESHOLD \
        "${dspark_asymmetric_k3_positive_threshold}" || return 1
    speco_require_positive_number \
        DSPARK_DYNAMIC_STREAM_WEIGHT_MIN_MULTIPLIER \
        "${dspark_dynamic_stream_weight_min_multiplier}" || return 1
    speco_require_positive_number \
        DSPARK_DYNAMIC_STREAM_WEIGHT_MAX_MULTIPLIER \
        "${dspark_dynamic_stream_weight_max_multiplier}" || return 1
    awk \
        -v minimum="${dspark_dynamic_stream_weight_min_multiplier}" \
        -v maximum="${dspark_dynamic_stream_weight_max_multiplier}" \
        'BEGIN { exit !(minimum <= maximum) }' \
        || { speco_die "DSPARK_DYNAMIC_STREAM_WEIGHT_MIN_MULTIPLIER cannot exceed DSPARK_DYNAMIC_STREAM_WEIGHT_MAX_MULTIPLIER"; return 1; }
    [[ "${test_freq}" =~ ^-?[0-9]+$ ]] \
        || { speco_die "TEST_FREQ must be -1 or a positive integer, got ${test_freq}"; return 1; }
    (( test_freq == -1 || test_freq > 0 )) \
        || { speco_die "TEST_FREQ must be -1 or a positive integer, got ${test_freq}"; return 1; }
    if [[ -n "${sandbox_fusion_url}" ]]; then
        speco_require_positive_int SANDBOX_MEMORY_LIMIT_MB "${sandbox_memory_limit_mb}" || return 1
        speco_require_positive_int SANDBOX_MAX_CONCURRENT "${sandbox_max_concurrent}" || return 1
        [[ "${sandbox_fusion_url}" =~ ^https?://[^[:space:]]+$ ]] \
            || { speco_die "SANDBOX_FUSION_URL must be an http(s) URL, got ${sandbox_fusion_url}"; return 1; }
        [[ "${sandbox_fusion_url}" != *"HOST:PORT"* ]] \
            || { speco_die "SANDBOX_FUSION_URL still contains the HOST:PORT placeholder"; return 1; }
        [[ "${sandbox_fusion_url%%\?*}" == */run_code ]] \
            || { speco_die "SANDBOX_FUSION_URL must end with /run_code, got ${sandbox_fusion_url}"; return 1; }
    fi
    if speco_is_true "${profile_uses_code_reward}" && [[ -z "${sandbox_fusion_url}" ]]; then
        [[ -r "${custom_reward_path}" ]] \
            || { speco_die "custom dataset reward is not readable: ${custom_reward_path}"; return 1; }
        "${experiment_python}" -c \
            "import importlib.util; spec = importlib.util.spec_from_file_location('drafter_domain_reward', '${custom_reward_path}'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.ensure_prime_code_runtime(); from verl.utils.reward_score import prime_code" \
            >/dev/null 2>&1 \
            || { speco_die "DATASET_PROFILE=${dataset_profile} cannot load the local PRIME code checker; configure SANDBOX_FUSION_URL or repair ${custom_reward_path}"; return 1; }
    fi
    if speco_is_true "${enable_checkpoints}"; then
        speco_require_positive_int SAVE_FREQ "${save_freq}"
        [[ "${save_offset}" =~ ^[0-9]+$ ]] \
            || speco_die "SAVE_OFFSET must be a non-negative integer, got ${save_offset}"
        (( save_offset < save_freq )) \
            || speco_die "SAVE_OFFSET=${save_offset} must be smaller than SAVE_FREQ=${save_freq}"
    else
        save_freq=-1
        save_offset=0
        resume_mode=disable
    fi
    if [[ "${run_mode}" == online ]]; then
        speco_require_positive_int DRAFT_WORKER_WORLD_SIZE "${draft_workers}"
        (( draft_workers % gen_tp == 0 )) \
            || speco_die "DRAFT_WORKER_WORLD_SIZE=${draft_workers} must be divisible by GEN_TP_SIZE=${gen_tp}"
    fi
    speco_configure_accelerators "${n_gpus}"
    [[ -r "${train_file}" ]] || speco_die "TRAIN_FILE is not readable: ${train_file}"
    [[ -r "${test_file}" ]] || speco_die "TEST_FILE is not readable: ${test_file}"
    [[ -e "${model_path}" ]] || speco_die "MODEL_PATH does not exist: ${model_path}"
    [[ -e "${drafter_path}" ]] || speco_die "DRAFTER_PATH does not exist: ${drafter_path}"
    [[ -r "${SPECO_BASE_LAUNCHER}" ]] || speco_die "base launcher is not readable: ${SPECO_BASE_LAUNCHER}"

    local worker_label=fixed
    [[ "${run_mode}" == online ]] && worker_label=w${draft_workers}
    local run_name_mode=${run_mode}
    [[ "${run_mode}" == offline ]] && run_name_mode=offline_fixed
    local dataset_id_suffix=
    [[ "${dataset_profile}" == dapo ]] || dataset_id_suffix=_${dataset_profile}
    local default_experiment_id=${SPECO_ALGORITHM}${dataset_id_suffix}_${run_name_mode}_${total_steps}step
    local experiment_id=${EXPERIMENT_ID:-${default_experiment_id}${online_strategy_suffix}${drafter_init_suffix}}
    local checkpoint_root=${CHECKPOINT_ROOT:-${project_root}/outputs/checkpoints}
    local checkpoint_dir=${CHECKPOINT_DIR:-${checkpoint_root}/${experiment_id}}
    local final_drafter_root=${FINAL_DRAFTER_ROOT:-${project_root}/outputs/drafters}
    local run_date=${RUN_DATE:-$(date -u +%F)}
    local log_dir=${LOG_DIR:-${project_root}/outputs/validation_logs/${run_date}/${SPECO_ALGORITHM}_vllm}
    local log_file=${LOG_FILE:-${log_dir}/${experiment_id}.log}
    local done_sentinel=${DONE_SENTINEL:-${log_dir}/.${experiment_id}.complete}
    local default_wandb_name=${SPECO_ALGORITHM}_${run_name_mode}_${total_steps}step
    local wandb_name=${WANDB_RUN_NAME:-${default_wandb_name}${online_strategy_suffix}${drafter_init_suffix}}
    local wandb_project_override=${WANDB_PROJECT:-}
    local wandb_entity_override=${WANDB_ENTITY:-}
    local wandb_project=${wandb_project_override:-${profile_wandb_project}}
    local wandb_entity=${wandb_entity_override:-linzzz-hong-kong-university-of-science-and-technology}
    local wandb_group=${WANDB_RUN_GROUP:-qwen3_8b_${SPECO_ALGORITHM}_comparison}
    local mode_tag=fixed-drafter
    [[ "${run_mode}" == online ]] && mode_tag=online-drafter
    local checkpoint_tag=ckpt-off
    speco_is_true "${enable_checkpoints}" && checkpoint_tag=ckpt${save_freq}-offset${save_offset}
    local wandb_tags=${WANDB_TAGS:-main_env_next,cann901,vllm-v1,npu${n_gpus},vllm-ascend,${SPECO_ALGORITHM},dataset-${dataset_profile},${mode_tag},${worker_label},${checkpoint_tag},reward-pm1}
    if [[ -n "${drafter_init_checkpoint}" ]]; then
        wandb_tags=${wandb_tags},drafter-cold-start,drafter-init-step-${drafter_init_step}
        local actor_checkpoint_marker=${checkpoint_dir}/latest_checkpointed_iteration.txt
        local actor_checkpoint_step=
        if [[ -s "${actor_checkpoint_marker}" ]]; then
            actor_checkpoint_step=$(tr -d '[:space:]' < "${actor_checkpoint_marker}")
        fi
        if [[ "${actor_checkpoint_step}" =~ ^[0-9]+$ ]] \
            && [[ -d "${checkpoint_dir}/global_step_${actor_checkpoint_step}" ]]; then
            # Re-launching the same cold-start experiment is a normal full
            # resume. The init checkpoint remains only a fallback model path;
            # TaskRunner selects the matching local draft_step checkpoint.
            local local_resume_drafter=${checkpoint_dir}/drafter/draft_step_${actor_checkpoint_step}
            [[ -f "${local_resume_drafter}/metadata.json" ]] \
                || { speco_die "actor checkpoint step ${actor_checkpoint_step} has no matching complete drafter checkpoint: ${local_resume_drafter}"; return 1; }
            local local_resume_complete
            local_resume_complete=$("${experiment_python}" - "${local_resume_drafter}/metadata.json" <<'PY'
import json
import sys
try:
    metadata = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print(0)
else:
    print(int(metadata.get("complete") is True))
PY
            )
            [[ "${local_resume_complete}" == 1 ]] \
                || { speco_die "matching drafter checkpoint is incomplete: ${local_resume_drafter}"; return 1; }
            drafter_init_active=0
        else
            drafter_init_active=1
            resume_mode=disable
        fi
    fi
    local drafter_resume_trainer_state=True
    (( drafter_init_active == 1 )) && drafter_resume_trainer_state=False
    local drafter_init_mode=none
    if [[ -n "${drafter_init_checkpoint}" ]]; then
        drafter_init_mode=full_resume
        (( drafter_init_active == 1 )) && drafter_init_mode=cold_weights
    fi
    local wandb_run_path_file=${checkpoint_dir}/.wandb_run_path
    local stored_wandb_run_id=
    if [[ -s "${wandb_run_path_file}" ]]; then
        local stored_wandb_run_path
        local stored_wandb_entity
        local stored_wandb_project
        local stored_wandb_extra
        stored_wandb_run_path=$(tr -d '[:space:]' < "${wandb_run_path_file}")
        IFS='/' read -r stored_wandb_entity stored_wandb_project \
            stored_wandb_run_id stored_wandb_extra <<< "${stored_wandb_run_path}"
        [[ -n "${stored_wandb_entity}" && -n "${stored_wandb_project}" \
            && -n "${stored_wandb_run_id}" && -z "${stored_wandb_extra}" ]] \
            || { speco_die "invalid W&B run path in ${wandb_run_path_file}: ${stored_wandb_run_path}"; return 1; }
        [[ -z "${wandb_entity_override}" || "${wandb_entity_override}" == "${stored_wandb_entity}" ]] \
            || { speco_die "WANDB_ENTITY=${wandb_entity_override} conflicts with checkpoint W&B entity ${stored_wandb_entity}"; return 1; }
        [[ -z "${wandb_project_override}" || "${wandb_project_override}" == "${stored_wandb_project}" ]] \
            || { speco_die "WANDB_PROJECT=${wandb_project_override} conflicts with checkpoint W&B project ${stored_wandb_project}"; return 1; }
        [[ -z "${WANDB_RUN_ID:-}" || "${WANDB_RUN_ID}" == "${stored_wandb_run_id}" ]] \
            || { speco_die "WANDB_RUN_ID=${WANDB_RUN_ID} conflicts with checkpoint W&B run ID ${stored_wandb_run_id}"; return 1; }
        wandb_entity=${stored_wandb_entity}
        wandb_project=${stored_wandb_project}
    fi

    local val_before_train_override=False
    speco_is_true "${val_before_train}" && val_before_train_override=True

    local -a overrides=(
        "data.train_batch_size=${train_batch_size}"
        "data.max_prompt_length=${max_prompt_length}"
        "data.max_response_length=${max_response_length}"
        "data.filter_overlong_prompts_workers=${filter_workers}"
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size}"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size}"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp}"
        "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp}"
        "actor_rollout_ref.rollout.n=${rollout_n}"
        "actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
        "actor_rollout_ref.actor.fsdp_config.param_offload=True"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        "actor_rollout_ref.ref.fsdp_config.param_offload=False"
        "actor_rollout_ref.actor.checkpoint.save_contents=${checkpoint_contents}"
        "actor_rollout_ref.actor.checkpoint.load_contents=${checkpoint_contents}"
        "actor_rollout_ref.rollout.drafter.enable=True"
        "actor_rollout_ref.rollout.drafter.enable_drafter_training=$([[ "${run_mode}" == online ]] && echo True || echo False)"
        "actor_rollout_ref.rollout.drafter.training.save_full_drafter_checkpoint=$(speco_is_true "${enable_checkpoints}" && echo True || echo False)"
        "actor_rollout_ref.rollout.drafter.training.resume_trainer_state_from_checkpoint=${drafter_resume_trainer_state}"
        "actor_rollout_ref.rollout.drafter.training.publish_async=${publish_async}"
        "actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=${bucket_mb}"
        "trainer.n_gpus_per_node=${n_gpus}"
        "trainer.total_training_steps=${total_steps}"
        "trainer.total_epochs=${total_epochs}"
        "trainer.resume_mode=${resume_mode}"
        "trainer.save_freq=${save_freq}"
        "speco.checkpoint.save_offset=${save_offset}"
        "trainer.test_freq=${test_freq}"
        "trainer.val_before_train=${val_before_train_override}"
        "trainer.max_actor_ckpt_to_keep=1"
        "trainer.default_local_dir=${checkpoint_dir}"
        "ray_kwargs.ray_init.num_cpus=${ray_num_cpus}"
    )
    if speco_is_true "${profile_uses_custom_reward}"; then
        [[ -r "${custom_reward_path}" ]] \
            || { speco_die "custom dataset reward is not readable: ${custom_reward_path}"; return 1; }
        overrides+=(
            "reward.custom_reward_function.path=${custom_reward_path}"
            "reward.custom_reward_function.name=compute_score"
            "+reward.custom_reward_function.reward_kwargs.reward_scale=pm1"
        )
        if [[ -n "${sandbox_fusion_url}" ]]; then
            overrides+=(
                "+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=${sandbox_fusion_url}"
                "+reward.custom_reward_function.reward_kwargs.memory_limit_mb=${sandbox_memory_limit_mb}"
                "+reward.custom_reward_function.reward_kwargs.sandbox_max_concurrent=${sandbox_max_concurrent}"
            )
        fi
    elif [[ -n "${sandbox_fusion_url}" ]]; then
        overrides+=(
            "reward.sandbox_fusion.url=${sandbox_fusion_url}"
            "reward.sandbox_fusion.memory_limit_mb=${sandbox_memory_limit_mb}"
            "reward.sandbox_fusion.max_concurrent=${sandbox_max_concurrent}"
        )
    fi
    if [[ "${SPECO_ALGORITHM}" == dspark ]]; then
        overrides+=(
            "actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=${dspark_loss_decay_gamma}"
            "actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=${dspark_ce_loss_alpha}"
            "actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=${dspark_l1_loss_alpha}"
            "actor_rollout_ref.rollout.drafter.training.dspark_accepted_k3_alpha=${dspark_accepted_k3_alpha}"
            "actor_rollout_ref.rollout.drafter.training.dspark_accepted_fkl_alpha=${dspark_accepted_fkl_alpha}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_k3_alpha=${dspark_rejected_k3_alpha}"
            "actor_rollout_ref.rollout.drafter.training.dspark_response_stream_weight=${dspark_response_stream_weight}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_stream_weight=${dspark_rejected_stream_weight}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_position_decay=${dspark_rejected_position_decay}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_log_ratio_clamp=${dspark_rejected_log_ratio_clamp}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_loss_max_clamp=${dspark_rejected_loss_max_clamp}"
            "actor_rollout_ref.rollout.drafter.training.dspark_asymmetric_k3_enabled=${dspark_asymmetric_k3_enabled}"
            "actor_rollout_ref.rollout.drafter.training.dspark_accepted_asymmetric_k3_enabled=${dspark_accepted_asymmetric_k3_enabled}"
            "actor_rollout_ref.rollout.drafter.training.dspark_rejected_asymmetric_k3_enabled=${dspark_rejected_asymmetric_k3_enabled}"
            "actor_rollout_ref.rollout.drafter.training.dspark_asymmetric_k3_positive_threshold=${dspark_asymmetric_k3_positive_threshold}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dynamic_accept_reject_weighting_enabled=${dspark_dynamic_accept_reject_weighting_enabled}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dynamic_stream_weight_min_multiplier=${dspark_dynamic_stream_weight_min_multiplier}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dynamic_stream_weight_max_multiplier=${dspark_dynamic_stream_weight_max_multiplier}"
            "actor_rollout_ref.rollout.drafter.training.dspark_k3_distribution_bins=${dspark_k3_distribution_bins}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dump_k3_logprobs=${dspark_dump_k3_logprobs}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dump_k3_logprobs_interval_steps=${dspark_dump_k3_logprobs_interval}"
            "actor_rollout_ref.rollout.drafter.training.dspark_dump_k3_logprobs_max_tokens_per_stream=${dspark_dump_k3_logprobs_max_tokens}"
        )
    fi
    if [[ -n "${validation_data_dir}" ]]; then
        overrides+=("trainer.validation_data_dir=${validation_data_dir}")
    fi
    if [[ "${run_mode}" == online ]]; then
        overrides+=(
            "actor_rollout_ref.rollout.drafter.training.worker_world_size=${draft_workers}"
            "actor_rollout_ref.rollout.drafter.training.step=${draft_train_steps}"
            "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=${draft_collect_interval}"
            "actor_rollout_ref.rollout.drafter.training.training_interval_steps=${draft_training_interval}"
            "actor_rollout_ref.rollout.drafter.training.publish_interval_steps=${draft_publish_interval}"
            "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${draft_batch_size}"
            "actor_rollout_ref.rollout.drafter.training.train_batches_per_cycle=${draft_train_batches}"
            "actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=${draft_max_samples}"
            "actor_rollout_ref.rollout.drafter.training.lr=${draft_lr}"
            "actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=${draft_lr_scheduler}"
            "actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=${draft_lr_warmup_steps}"
            "actor_rollout_ref.rollout.drafter.training.lr_decay_steps=${draft_lr_decay_steps}"
            "actor_rollout_ref.rollout.drafter.training.min_lr_ratio=${draft_min_lr_ratio}"
            "actor_rollout_ref.rollout.drafter.training.train_unique_samples_first=${draft_unique_samples_first}"
            "actor_rollout_ref.rollout.drafter.training.fixed_eval_samples_per_replica=${draft_fixed_eval_samples}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_enabled=${candidate_gate_enabled}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_alphas=${candidate_gate_alphas}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_samples=${candidate_gate_samples}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_max_tokens=${candidate_gate_max_tokens}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_min_acceptance_delta=${candidate_gate_min_acceptance_delta}"
            "actor_rollout_ref.rollout.drafter.training.candidate_gate_max_ms_per_token_regression=${candidate_gate_max_latency_regression}"
            "actor_rollout_ref.rollout.drafter.training.candidate_max_fixed_loss_delta=${candidate_max_fixed_loss_delta}"
            "actor_rollout_ref.rollout.drafter.training.candidate_max_grad_norm=${candidate_max_grad_norm}"
            "actor_rollout_ref.rollout.drafter.training.candidate_grad_norm_rolling_multiplier=${candidate_grad_norm_multiplier}"
            "actor_rollout_ref.rollout.drafter.training.candidate_max_rejected_k3_clamp_ratio=${candidate_max_rejected_clamp_ratio}"
            "actor_rollout_ref.rollout.drafter.training.candidate_max_rejected_k3_median=${candidate_max_rejected_k3_median}"
            "actor_rollout_ref.rollout.drafter.training.save_best_drafter_checkpoint=${save_best_drafter_checkpoint}"
            "actor_rollout_ref.rollout.drafter.training.best_checkpoint_min_acceptance_delta=${best_checkpoint_min_acceptance_delta}"
            "actor_rollout_ref.rollout.drafter.training.dual_metric_best_checkpoint_enabled=${dual_metric_best_checkpoint_enabled}"
            "actor_rollout_ref.rollout.drafter.training.best_checkpoint_min_throughput_relative_delta=${best_checkpoint_min_throughput_relative_delta}"
            "actor_rollout_ref.rollout.drafter.training.dual_metric_best_checkpoint_include_optimizer=${dual_metric_best_checkpoint_include_optimizer}"
        )
    fi
    if declare -p SPECO_ALGORITHM_OVERRIDES >/dev/null 2>&1; then
        overrides+=("${SPECO_ALGORITHM_OVERRIDES[@]}")
    fi

    local -a command=(bash "${SPECO_BASE_LAUNCHER}" "${overrides[@]}" "$@")
    if speco_is_true "${DRY_RUN:-0}"; then
        echo "algorithm=${SPECO_ALGORITHM} mode=${run_mode} dataset_profile=${dataset_profile} experiment_id=${experiment_id}"
        echo "dataset=${profile_description}"
        echo "resources=n_gpus=${n_gpus} visible_npus=${ASCEND_RT_VISIBLE_DEVICES} draft_workers=${draft_workers}"
        echo "model=${model_path} drafter=${drafter_path}"
        if [[ -n "${drafter_init_checkpoint}" ]]; then
            echo "drafter_init=mode=${drafter_init_mode} checkpoint=${drafter_init_checkpoint} step=${drafter_init_step} resume_trainer_state=${drafter_resume_trainer_state}"
        fi
        echo "train=${train_file} test=${test_file}"
        echo "sequence=max_prompt=${max_prompt_length} max_response=${max_response_length} test_freq=${test_freq} val_before_train=${val_before_train_override}"
        echo "wandb=project=${wandb_project} group=${wandb_group} name=${wandb_name}"
        echo "checkpoint=${checkpoint_dir} log=${log_file}"
        echo "final_drafter=enabled=${archive_final_drafter} root=${final_drafter_root}"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    if [[ -f "${done_sentinel}" ]]; then
        echo "Experiment already completed: ${done_sentinel}"
        return 0
    fi
    mkdir -p "${checkpoint_dir}" "${log_dir}" "${project_root}/outputs/wandb"

    # One vLLM/Ray training job per host. This protects a running experiment
    # from an accidental second invocation that would call ray stop --force.
    exec 9>"${project_root}/outputs/.speco_vllm_experiment.lock"
    if speco_is_true "${SPECO_WAIT_FOR_HOST_LOCK:-0}"; then
        echo "Waiting for the managed SPECO host lock: ${project_root}/outputs/.speco_vllm_experiment.lock"
        flock 9 || { speco_die "failed while waiting for the managed SPECO host lock"; return 1; }
    else
        flock -n 9 || speco_die "another managed SPECO experiment holds the host lock"
    fi
    if pgrep -f '[v]erl_speco.main' >/dev/null && ! speco_is_true "${ALLOW_CONCURRENT_EXPERIMENT:-0}"; then
        speco_die "a verl_speco.main process is already running; wait for it or set ALLOW_CONCURRENT_EXPERIMENT=1"
    fi

    local run_id_file=${checkpoint_dir}/.wandb_run_id
    local wandb_run_id=${WANDB_RUN_ID:-${stored_wandb_run_id}}
    if [[ -f "${run_id_file}" ]]; then
        local stored_run_id
        stored_run_id=$(tr -d '[:space:]' < "${run_id_file}")
        if [[ -n "${wandb_run_id}" && "${wandb_run_id}" != "${stored_run_id}" ]]; then
            speco_die "WANDB_RUN_ID=${wandb_run_id} does not match checkpoint run ID ${stored_run_id}"
        fi
        wandb_run_id=${stored_run_id}
    fi
    if [[ -z "${wandb_run_id}" ]]; then
        wandb_run_id=$("${experiment_python}" -c 'import uuid; print(uuid.uuid4().hex[:12])')
    fi
    printf '%s\n' "${wandb_run_id}" > "${run_id_file}"
    printf '%s/%s/%s\n' "${wandb_entity}" "${wandb_project}" "${wandb_run_id}" \
        > "${wandb_run_path_file}"
    speco_record_managed_run_metadata \
        "${experiment_id}" "${log_file}" "${checkpoint_dir}" "${wandb_run_id}" || return 1

    export MODEL_PATH="${model_path}"
    export DRAFTER_PATH="${drafter_path}"
    export TRAIN_FILE="${train_file}"
    export TEST_FILE="${test_file}"
    export CKPTS_DIR="${checkpoint_dir}"
    export WANDB_MODE=${WANDB_MODE:-online}
    export WANDB_DIR=${WANDB_DIR:-${project_root}/outputs/wandb}
    export WANDB_ENTITY="${wandb_entity}"
    export WANDB_PROJECT="${wandb_project}"
    export WANDB_RUN_ID="${wandb_run_id}"
    export WANDB_RESUME=allow
    export WANDB_RUN_NAME="${wandb_name}"
    export WANDB_RUN_GROUP="${wandb_group}"
    export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-${run_mode}_drafter_training}"
    export WANDB_TAGS="${wandb_tags}"
    export SPECO_EXPERIMENT_ID="${experiment_id}"
    export SPECO_RUN_MODE="${run_mode}"
    export SPECO_ONLINE_STRATEGY_TAG="${online_strategy_tag}"
    export SPECO_DRAFTER_INIT_CHECKPOINT="${drafter_init_checkpoint}"
    export SPECO_DRAFTER_INIT_STEP="${drafter_init_step}"
    export SPECO_DRAFTER_INIT_MODE="${drafter_init_mode}"
    export SPECO_TRAINING_LOG_FILE="${log_file}"
    export SPECO_DRAFTER_ACCEPTANCE_INCIDENT_LOG="${SPECO_DRAFTER_ACCEPTANCE_INCIDENT_LOG:-${project_root}/outputs/drafter_acceptance_incidents.jsonl}"
    if speco_is_true "${dspark_dump_k3_logprobs}"; then
        export SPECO_DRAFTER_K3_DUMP_DIR="${dspark_dump_k3_logprobs_dir:-${project_root}/outputs/drafter_analysis/${experiment_id}/${wandb_run_id}/k3_logprobs}"
    fi
    export TRAINER_LOGGERS='["console","wandb"]'
    export RAY_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD:-0.99}
    export RAY_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-250}
    export RAY_idle_worker_killing_memory_threshold_bytes=${RAY_IDLE_WORKER_KILLING_MEMORY_THRESHOLD_BYTES:-268435456}
    export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}

    speco_configure_ray_command "${experiment_python}" || return 1
    trap speco_cleanup_experiment EXIT INT TERM
    if ! speco_is_true "${SPECO_SKIP_GLOBAL_RAY_STOP:-0}"; then
        "${SPECO_RAY_COMMAND[@]}" stop --force >/dev/null 2>&1 || true
    fi
    if speco_is_true "${enable_checkpoints}"; then
        "${project_root}/tools/watch_checkpoint_retention.sh" "${checkpoint_dir}" "${done_sentinel}" 10 &
        SPECO_RETENTION_PID=$!
    fi

    if [[ -n "${drafter_init_checkpoint}" ]]; then
        echo "Drafter init: mode=${drafter_init_mode}, checkpoint=${drafter_init_checkpoint}, step=${drafter_init_step}, resume_trainer_state=${drafter_resume_trainer_state}" \
            | tee -a "${log_file}"
    fi
    echo "Starting ${experiment_id}: W&B run ${wandb_run_id}, log ${log_file}"
    set +e
    if speco_is_true "${SPECO_DETACHED_CHILD:-0}"; then
        "${command[@]}" 2>&1 | tee -a "${log_file}" >/dev/null
    else
        "${command[@]}" 2>&1 | tee -a "${log_file}"
    fi
    local training_status=${PIPESTATUS[0]}
    set -e
    if [[ "${training_status}" -ne 0 ]]; then
        if speco_is_true "${enable_checkpoints}"; then
            echo "Training failed with status ${training_status}; checkpoint retained at ${checkpoint_dir}" >&2
        else
            echo "Training failed with status ${training_status}; checkpoints were disabled" >&2
        fi
        return "${training_status}"
    fi

    if [[ -n "${SPECO_RETENTION_PID}" ]]; then
        kill "${SPECO_RETENTION_PID}" >/dev/null 2>&1 || true
        wait "${SPECO_RETENTION_PID}" 2>/dev/null || true
        SPECO_RETENTION_PID=
    fi
    local final_drafter_path=
    if speco_is_true "${archive_final_drafter}"; then
        final_drafter_path=$(speco_archive_final_drafter \
            "${checkpoint_dir}" \
            "${final_drafter_root}" \
            "${experiment_id}" \
            "${wandb_run_id}" \
            "${wandb_entity}/${wandb_project}/${wandb_run_id}" \
            "${experiment_python}") || {
                echo "Training completed, but final drafter archiving failed; checkpoint retained at ${checkpoint_dir}" >&2
                return 1
            }
        echo "Final drafter archived at ${final_drafter_path}"
    fi
    touch "${done_sentinel}"
    if speco_is_true "${delete_after_success}"; then
        speco_safe_remove_checkpoint "${checkpoint_dir}"
    fi
    echo "Completed ${experiment_id}; W&B run ${wandb_run_id}; log ${log_file}"
}
