from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "run_dspark_attention_ablation.sh"


def test_attention_ablation_runner_is_frozen_dapo_validation(tmp_path: Path) -> None:
    calls = tmp_path / "calls.txt"
    entrypoint = tmp_path / "entrypoint.sh"
    entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"${CALLS_FILE}\"\n"
        "printf 'mode=%s dataset=%s test=%s attention=%s window=%s\\n' \"${RUN_MODE}\" \"${DATASET_PROFILE}\" \"${TEST_FILE}\" \"${DSPARK_VLLM_ATTENTION_MODE}\" \"${DSPARK_VLLM_CONTEXT_WINDOW}\" >> \"${CALLS_FILE}\"\n",
        encoding="utf-8",
    )
    entrypoint.chmod(0o755)
    env = {
        **os.environ,
        "ENTRYPOINT": str(entrypoint),
        "CALLS_FILE": str(calls),
        "ABLATION_ARMS": "full swa1024",
        "ANALYSIS_DIR": str(tmp_path / "analysis"),
        "SPECO_ABLATION_LAUNCH_MODE": "foreground",
    }

    subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, check=True)

    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all("trainer.val_only=True" in line for line in lines[0::2])
    assert all("data.train_max_samples=128" in line for line in lines[0::2])
    assert all("data.val_max_samples=128" in line for line in lines[0::2])
    assert all("actor_rollout_ref.rollout.val_kwargs.do_sample=False" in line for line in lines[0::2])
    assert lines[1].endswith("attention=full window=null")
    assert lines[3].endswith("attention=swa window=1024")
    assert all("mode=offline dataset=dapo" in line for line in lines[1::2])
