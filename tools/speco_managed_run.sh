#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANAGED_ROOT=${SPECO_MANAGED_RUN_ROOT:-${PROJECT_ROOT}/outputs/managed_runs}

usage() {
    echo "usage: $0 {list|status|follow|stop} [run-id]" >&2
}

resolve_run_dir() {
    local run_id=${1:-}
    [[ -n "${run_id}" ]] || { usage; return 1; }
    local candidate
    if [[ "${run_id}" == /* ]]; then
        candidate=$(realpath -m -- "${run_id}")
    else
        candidate=$(realpath -m -- "${MANAGED_ROOT}/${run_id}")
    fi
    local resolved_root
    resolved_root=$(realpath -m -- "${MANAGED_ROOT}")
    case "${candidate}" in
        "${resolved_root}"/*) ;;
        *) echo "error: run directory must be inside ${resolved_root}" >&2; return 1 ;;
    esac
    [[ -d "${candidate}" ]] || { echo "error: managed run not found: ${candidate}" >&2; return 1; }
    printf '%s\n' "${candidate}"
}

show_status() {
    local run_dir=$1
    local state=unknown
    local pid=
    local exit_status=
    [[ -r "${run_dir}/state" ]] && IFS= read -r state < "${run_dir}/state"
    [[ -r "${run_dir}/pid" ]] && IFS= read -r pid < "${run_dir}/pid"
    [[ -r "${run_dir}/exit_status" ]] && IFS= read -r exit_status < "${run_dir}/exit_status"

    echo "run: $(basename "${run_dir}")"
    echo "state: ${state}"
    [[ -n "${pid}" ]] && echo "pid: ${pid}"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "process: running"
    else
        echo "process: not running"
    fi
    [[ -n "${exit_status}" ]] && echo "exit_status: ${exit_status}"
    [[ -r "${run_dir}/training_log" ]] && echo "training_log: $(< "${run_dir}/training_log")"
    echo "launcher_log: ${run_dir}/launcher.log"
}

follow_run() {
    local run_dir=$1
    local log_file=${run_dir}/launcher.log
    local pid=
    if [[ -r "${run_dir}/training_log" ]]; then
        IFS= read -r log_file < "${run_dir}/training_log"
    fi
    [[ -r "${run_dir}/pid" ]] && IFS= read -r pid < "${run_dir}/pid"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
        tail -n "${SPECO_TAIL_LINES:-80}" -F --pid="${pid}" "${log_file}"
    else
        tail -n "${SPECO_TAIL_LINES:-80}" "${log_file}"
    fi
}

stop_run() {
    local run_dir=$1
    local pid=
    [[ -r "${run_dir}/pid" ]] && IFS= read -r pid < "${run_dir}/pid"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || { echo "error: no valid managed-run PID in ${run_dir}" >&2; return 1; }
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "Managed run is not running: $(basename "${run_dir}")"
        return 0
    fi
    # setsid makes the wrapper PID its own process-group ID. Signalling the
    # group lets the entrypoint trap clean up Ray and its retention watcher.
    kill -TERM -- "-${pid}"
    echo "Stop requested for managed run $(basename "${run_dir}") (process group ${pid})"
}

command_name=${1:-}
shift || true
case "${command_name}" in
    list)
        if [[ -d "${MANAGED_ROOT}" ]]; then
            find "${MANAGED_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
        fi
        ;;
    status)
        show_status "$(resolve_run_dir "${1:-}")"
        ;;
    follow)
        follow_run "$(resolve_run_dir "${1:-}")"
        ;;
    stop)
        stop_run "$(resolve_run_dir "${1:-}")"
        ;;
    *)
        usage
        exit 1
        ;;
esac
