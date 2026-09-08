# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_target_only_dry_run_forces_no_spec_overrides_last(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    for filename in ("train.parquet", "test.parquet"):
        (tmp_path / filename).touch()

    env = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_MODE": "online",
        "N_GPUS": "2",
        "MODEL_PATH": str(tmp_path / "model"),
        "TRAIN_FILE": str(tmp_path / "train.parquet"),
        "TEST_FILE": str(tmp_path / "test.parquet"),
        "CANN_ENV": str(tmp_path / "missing-cann-env.sh"),
    }
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [
            str(ROOT / "tools" / "run_target_only_vllm_experiment.sh"),
            "actor_rollout_ref.rollout.drafter.enable=True",
            "actor_rollout_ref.rollout.drafter.enable_drafter_training=True",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "algorithm=target_only mode=offline" in result.stdout
    command = result.stdout.split("command:", maxsplit=1)[1]
    assert command.rfind("drafter.enable=False") > command.rfind(
        "drafter.enable=True"
    )
    assert command.rfind("drafter.enable_drafter_training=False") > command.rfind(
        "drafter.enable_drafter_training=True"
    )
    assert command.rstrip().endswith(
        "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config=null"
    )
    assert "actor_rollout_ref.rollout.temperature=0.6" in command
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=False" in command


def test_target_only_uses_the_same_dataset_profile_defaults(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "DATASET_PROFILE": "wildsci-gpqa",
        "N_GPUS": "2",
        "MODEL_PATH": str(tmp_path / "model"),
        "CANN_ENV": str(tmp_path / "missing-cann-env.sh"),
    }
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [str(ROOT / "tools" / "run_target_only_vllm_experiment.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "dataset_profile=wildsci_gpqa" in result.stdout
    assert "experiment_id=target_only_wildsci_gpqa_150step" in result.stdout
    assert (
        "wandb=project=verl_speco_wildsci group=qwen3_8b_dflash_comparison "
        "name=target_only_150step"
        in result.stdout
    )
    assert "max_prompt=2048 max_response=4096" in result.stdout
