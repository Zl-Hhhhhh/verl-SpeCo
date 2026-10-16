"""Record memory and stop only a tagged experiment on sustained host pressure."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def sample_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    return {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'available_gib': values['MemAvailable'] / 2**30,
        'swap_used_gib': (values['SwapTotal'] - values['SwapFree']) / 2**30,
    }


def owned_processes(run_id):
    marker = f'SPECO_DIRECTIONAL_RUN_ID={run_id}'.encode()
    result = []
    for path in Path('/proc').glob('[0-9]*/environ'):
        try:
            if marker in path.read_bytes().split(b'\0') and int(path.parent.name) != os.getpid():
                result.append(int(path.parent.name))
        except (OSError, ProcessLookupError):
            pass
    return result


def signal_owned(run_id, sig):
    for pid in owned_processes(run_id):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-gib', type=float, default=32)
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    low_samples = 0
    with (args.output / 'memory_usage.jsonl').open('a', buffering=1) as log:
        while Path(f'/proc/{args.owner_pid}').exists():
            sample = sample_memory()
            try:
                info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=8)
                hbm = {}
                for line in info.stdout.splitlines():
                    match = re.search(r'\|\s*[01]\s+(\d+)\s*\|.*?(\d+)\s*/\s*(\d+)\s*\|\s*$', line)
                    if match and 8 <= int(match[1]) <= 15:
                        hbm[match[1]] = {'used_mib': int(match[2]), 'total_mib': int(match[3])}
                sample['hbm'] = hbm
            except (OSError, subprocess.TimeoutExpired) as error:
                sample['npu_error'] = str(error)
            log.write(json.dumps(sample) + '\n')
            low_samples = low_samples + 1 if sample['available_gib'] < args.minimum_gib else 0
            if low_samples >= 2:
                (args.output / 'memory_guard_triggered.json').write_text(json.dumps(sample, indent=2))
                # The wrapper owns this session; Ray subprocesses also receive
                # the unique tag explicitly through runtime_env.env_vars.
                try:
                    os.killpg(args.owner_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signal_owned(args.run_id, signal.SIGTERM)
                time.sleep(10)
                signal_owned(args.run_id, signal.SIGKILL)
                (args.output / 'status.txt').write_text('stopped_by_memory_guard\n')
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
