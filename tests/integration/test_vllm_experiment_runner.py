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
import sys
import time
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _start_mock_vllm_process(
    name: str, token: str, *, ignore_term: bool = False
) -> subprocess.Popen:
    code = """
import ctypes
import signal
import sys
import time

ctypes.CDLL(None).prctl(15, sys.argv[1].encode(), 0, 0, 0)
if sys.argv[2] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(1)
"""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            name,
            "ignore-term" if ignore_term else "handle-term",
        ],
        env={**os.environ, "SPECO_RUN_CLEANUP_ID": token},
    )


def _wait_for_process_name(process: subprocess.Popen, name: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if Path(f"/proc/{process.pid}/comm").read_text().strip() == name[:15]:
            return
        time.sleep(0.02)
    pytest.fail(f"process {process.pid} did not set its name")


@pytest.mark.parametrize("ignore_term", [False, True])
def test_vllm_cleanup_only_stops_processes_owned_by_this_run(ignore_term: bool) -> None:
    token = str(uuid.uuid4())
    owned = _start_mock_vllm_process(
        "VLLM::EngineCore", token, ignore_term=ignore_term
    )
    worker = _start_mock_vllm_process("VLLM::Worker_TP0", token)
    other_run = _start_mock_vllm_process("VLLM::EngineCore", str(uuid.uuid4()))
    other_process = _start_mock_vllm_process("unrelated", token)
    processes = (owned, worker, other_run, other_process)
    try:
        names = ("VLLM::EngineCore", "VLLM::Worker_TP0", "VLLM::EngineCore", "unrelated")
        for process, name in zip(processes, names):
            _wait_for_process_name(process, name)
        command = f"""
set -euo pipefail
source {ROOT / 'tools' / 'speco_vllm_experiment_common.sh'}
SPECO_RUN_CLEANUP_ID={token}
SPECO_RETENTION_PID=
SPECO_RAY_COMMAND=()
"""
        if ignore_term:
            command += "sleep() { :; }\n"
        result = subprocess.run(
            ["bash", "-c", command + "speco_cleanup_experiment\n"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        owned.wait(timeout=3)
        worker.wait(timeout=3)
        assert other_run.poll() is None
        assert other_process.poll() is None
        if ignore_term:
            assert "forcing" in result.stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


def _run_dflash(
    tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    use_profile_paths = overrides.pop("_USE_PROFILE_PATHS", "0") == "1"
    for directory in ("model", "drafter"):
        (tmp_path / directory).mkdir()
    for filename in ("train.parquet", "test.parquet"):
        (tmp_path / filename).touch()
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "N_GPUS": "16",
        "DRAFT_WORKER_WORLD_SIZE": "16",
        "MODEL_PATH": str(tmp_path / "model"),
        "DRAFTER_PATH": str(tmp_path / "drafter"),
        "CANN_ENV": str(tmp_path / "missing-cann-env.sh"),
        **overrides,
    }
    if not use_profile_paths:
        env["TRAIN_FILE"] = str(tmp_path / "train.parquet")
        env["TEST_FILE"] = str(tmp_path / "test.parquet")
    if "ASCEND_RT_VISIBLE_DEVICES" not in overrides:
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    return subprocess.run(
        [str(ROOT / "tools" / "run_dflash_vllm_experiment.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_dspark(
    tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    for directory in ("model", "drafter"):
        (tmp_path / directory).mkdir()
    for filename in ("train.parquet", "test.parquet"):
        (tmp_path / filename).touch()
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "N_GPUS": "16",
        "DRAFT_WORKER_WORLD_SIZE": "16",
        "MODEL_PATH": str(tmp_path / "model"),
        "DRAFTER_PATH": str(tmp_path / "drafter"),
        "TRAIN_FILE": str(tmp_path / "train.parquet"),
        "TEST_FILE": str(tmp_path / "test.parquet"),
        "CANN_ENV": str(tmp_path / "missing-cann-env.sh"),
        **overrides,
    }
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    return subprocess.run(
        ["bash", str(ROOT / "tools" / "run_dspark_vllm_experiment.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_drafter_checkpoint(path: Path, step: int) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        '{"model_type":"qwen3","speco_training_model_type":"dspark"}',
        encoding="utf-8",
    )
    (path / "pytorch_model.bin").write_bytes(b"trained drafter weights")
    (path / "metadata.json").write_text(
        f'{{"step":{step},"complete":true}}', encoding="utf-8"
    )


def test_managed_dspark_cold_starts_from_drafter_checkpoint_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "draft_step_37"
    _write_drafter_checkpoint(checkpoint, 37)

    result = _run_dspark(
        tmp_path,
        DRAFTER_INIT_CHECKPOINT=str(checkpoint),
        DRAFTER_INIT_LABEL="p2_02_s37",
        DATASET_PROFILE="dapo",
        TOTAL_STEPS="100",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "experiment_id=dspark_online_100step__p2_06__cold_p2_02_s37"
    ) in result.stdout
    assert f"drafter={checkpoint.resolve()}" in result.stdout
    assert (
        f"drafter_init=mode=cold_weights checkpoint={checkpoint.resolve()} "
        "step=37 resume_trainer_state=False"
    ) in result.stdout
    assert "trainer.resume_mode=disable" in result.stdout
    assert (
        "drafter.training.resume_trainer_state_from_checkpoint=False"
        in result.stdout
    )
    assert "drafter.training.step=5" in result.stdout
    assert "drafter.training.lr=5e-6" in result.stdout


def test_managed_dspark_cold_start_command_resumes_its_existing_run(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "draft_step_37"
    _write_drafter_checkpoint(source, 37)
    checkpoint_dir = tmp_path / "target"
    checkpoint_dir.joinpath("global_step_12").mkdir(parents=True)
    checkpoint_dir.joinpath("latest_checkpointed_iteration.txt").write_text(
        "12\n", encoding="utf-8"
    )
    _write_drafter_checkpoint(
        checkpoint_dir / "drafter" / "draft_step_12", 12
    )

    result = _run_dspark(
        tmp_path,
        DRAFTER_INIT_CHECKPOINT=str(source),
        CHECKPOINT_DIR=str(checkpoint_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "drafter_init=mode=full_resume" in result.stdout
    assert "trainer.resume_mode=auto" in result.stdout
    assert (
        "drafter.training.resume_trainer_state_from_checkpoint=True"
        in result.stdout
    )


def test_managed_dspark_resume_rejects_missing_matching_drafter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "draft_step_37"
    _write_drafter_checkpoint(source, 37)
    checkpoint_dir = tmp_path / "target"
    checkpoint_dir.joinpath("global_step_12").mkdir(parents=True)
    checkpoint_dir.joinpath("latest_checkpointed_iteration.txt").write_text(
        "12\n", encoding="utf-8"
    )

    result = _run_dspark(
        tmp_path,
        DRAFTER_INIT_CHECKPOINT=str(source),
        CHECKPOINT_DIR=str(checkpoint_dir),
    )

    assert result.returncode != 0
    assert "no matching complete drafter checkpoint" in result.stderr


def test_managed_dspark_rejects_cold_start_without_managed_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "draft_step_37"
    checkpoint.mkdir(parents=True)
    checkpoint.joinpath("config.json").write_text(
        '{"model_type":"qwen3","speco_training_model_type":"dspark"}',
        encoding="utf-8",
    )
    checkpoint.joinpath("pytorch_model.bin").write_bytes(b"weights")

    result = _run_dspark(
        tmp_path,
        DRAFTER_INIT_CHECKPOINT=str(checkpoint),
    )

    assert result.returncode != 0
    assert "not a complete, loadable drafter checkpoint" in result.stderr


def test_managed_dflash_dry_run_keeps_ray_and_visible_npus_consistent(
    tmp_path: Path,
) -> None:
    result = _run_dflash(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (
        "resources=n_gpus=16 "
        "visible_npus=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
        "draft_workers=16" in result.stdout
    )
    assert "trainer.n_gpus_per_node=16" in result.stdout
    assert "wandb=project=verl_speco_dapo" in result.stdout


def test_managed_dspark_online_run_enables_draft_opd_loss(tmp_path: Path) -> None:
    result = _run_dspark(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "experiment_id=dspark_online_100step__p2_06__cold_p2_02_v1" in result.stdout
    for override in (
        "drafter.training.step=3",
        "drafter.training.collect_interval_steps=10",
        "drafter.training.training_interval_steps=10",
        "drafter.training.max_collect_samples_per_step_per_replica=20",
        "drafter.training.lr=2e-6",
        "drafter.training.lr_scheduler_type=global_cosine",
        "drafter.training.candidate_gate_enabled=true",
        "drafter.training.candidate_gate_alphas=\\[0.0\\,0.25\\,0.5\\,1.0\\]",
        "drafter.training.save_best_drafter_checkpoint=true",
        "drafter.training.train_unique_samples_first=true",
        "drafter.training.fixed_eval_samples_per_replica=4",
        "drafter.training.dspark_trace_aligned_anchors=True",
        "drafter.training.dspark_loss_decay_gamma=0.0",
        "drafter.training.dspark_ce_loss_alpha=0.0",
        "drafter.training.dspark_l1_loss_alpha=0.0",
        "drafter.training.dspark_accepted_k3_alpha=1.0",
        "drafter.training.dspark_accepted_fkl_alpha=0.0",
        "drafter.training.dspark_rejected_k3_alpha=1.0",
        "drafter.training.dspark_response_stream_weight=1.0",
        "drafter.training.dspark_rejected_stream_weight=0.5",
        "drafter.training.dspark_rejected_position_decay=0.8",
    ):
        assert override in result.stdout


def test_managed_dspark_loss_coefficients_are_environment_overridable(
    tmp_path: Path,
) -> None:
    result = _run_dspark(
        tmp_path,
        DSPARK_ACCEPTED_K3_ALPHA="0.25",
        DSPARK_ACCEPTED_FKL_ALPHA="0.5",
        DSPARK_REJECTED_STREAM_WEIGHT="2.0",
    )

    assert result.returncode == 0, result.stderr
    assert "drafter.training.dspark_accepted_k3_alpha=0.25" in result.stdout
    assert "drafter.training.dspark_accepted_fkl_alpha=0.5" in result.stdout
    assert "drafter.training.dspark_rejected_stream_weight=2.0" in result.stdout


def test_managed_dspark_dual_best_checkpoint_is_optional(tmp_path: Path) -> None:
    result = _run_dspark(
        tmp_path,
        DRAFT_DUAL_METRIC_BEST_CHECKPOINT_ENABLED="true",
        BEST_CHECKPOINT_MIN_ACCEPTANCE_DELTA="0.02",
        BEST_CHECKPOINT_MIN_THROUGHPUT_RELATIVE_DELTA="0.01",
    )

    assert result.returncode == 0, result.stderr
    assert "drafter.training.dual_metric_best_checkpoint_enabled=true" in result.stdout
    assert "drafter.training.best_checkpoint_min_acceptance_delta=0.02" in result.stdout
    assert (
        "drafter.training.best_checkpoint_min_throughput_relative_delta=0.01"
        in result.stdout
    )
    assert (
        "drafter.training.dual_metric_best_checkpoint_include_optimizer=false"
        in result.stdout
    )


def test_managed_dspark_strategy_tag_override_names_wandb_run(tmp_path: Path) -> None:
    result = _run_dspark(tmp_path, SPECO_ONLINE_STRATEGY_TAG="p3_04")

    assert result.returncode == 0, result.stderr
    assert "experiment_id=dspark_online_101step__p3_04" in result.stdout
    assert "name=dspark_online_101step__p3_04" in result.stdout


def test_managed_dspark_k3_analysis_controls_are_environment_overridable(
    tmp_path: Path,
) -> None:
    result = _run_dspark(
        tmp_path,
        DSPARK_K3_DISTRIBUTION_BINS="64",
        DSPARK_DUMP_K3_LOGPROBS="true",
        DSPARK_DUMP_K3_LOGPROBS_INTERVAL_STEPS="3",
        DSPARK_DUMP_K3_LOGPROBS_MAX_TOKENS_PER_STREAM="128",
    )

    assert result.returncode == 0, result.stderr
    for override in (
        "drafter.training.dspark_k3_distribution_bins=64",
        "drafter.training.dspark_dump_k3_logprobs=true",
        "drafter.training.dspark_dump_k3_logprobs_interval_steps=3",
        "drafter.training.dspark_dump_k3_logprobs_max_tokens_per_stream=128",
    ):
        assert override in result.stdout


def test_managed_dspark_asymmetric_k3_is_environment_overridable(
    tmp_path: Path,
) -> None:
    result = _run_dspark(
        tmp_path,
        DSPARK_ASYMMETRIC_K3_ENABLED="true",
        DSPARK_ACCEPTED_ASYMMETRIC_K3_ENABLED="false",
        DSPARK_REJECTED_ASYMMETRIC_K3_ENABLED="true",
        DSPARK_ASYMMETRIC_K3_POSITIVE_THRESHOLD="1.75",
        DSPARK_DYNAMIC_ACCEPT_REJECT_WEIGHTING_ENABLED="true",
        DSPARK_DYNAMIC_STREAM_WEIGHT_MIN_MULTIPLIER="0.5",
        DSPARK_DYNAMIC_STREAM_WEIGHT_MAX_MULTIPLIER="3.0",
    )

    assert result.returncode == 0, result.stderr
    assert "drafter.training.dspark_asymmetric_k3_enabled=true" in result.stdout
    assert (
        "drafter.training.dspark_accepted_asymmetric_k3_enabled=false"
        in result.stdout
    )
    assert (
        "drafter.training.dspark_rejected_asymmetric_k3_enabled=true"
        in result.stdout
    )
    assert (
        "drafter.training.dspark_asymmetric_k3_positive_threshold=1.75"
        in result.stdout
    )
    assert (
        "drafter.training.dspark_dynamic_accept_reject_weighting_enabled=true"
        in result.stdout
    )
    assert (
        "drafter.training.dspark_dynamic_stream_weight_min_multiplier=0.5"
        in result.stdout
    )
    assert (
        "drafter.training.dspark_dynamic_stream_weight_max_multiplier=3.0"
        in result.stdout
    )


def test_managed_dflash_rejects_an_undersized_visible_mask(tmp_path: Path) -> None:
    result = _run_dflash(
        tmp_path,
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )

    assert result.returncode != 0
    assert "N_GPUS=16" in result.stderr
    assert "exposes 8 device(s)" in result.stderr


@pytest.mark.parametrize(
    ("profile", "train_path", "test_path", "sequence"),
    [
        (
            "taco_lcb",
            "/workspace/data/drafter_domains/verl/code/taco_cobalt/train.parquet",
            "/workspace/data/drafter_domains/verl/code/eval/livecodebench/v6_only.parquet",
            "sequence=max_prompt=2048 max_response=8192 test_freq=10 val_before_train=True",
        ),
        (
            "wildsci_gpqa",
            "/workspace/data/drafter_domains/verl/science/wildsci_all_aligned/train.parquet",
            "/workspace/data/drafter_domains/verl/science/eval/gpqa_diamond/test.parquet",
            "sequence=max_prompt=3072 max_response=4096 test_freq=10 val_before_train=True",
        ),
        (
            "wildsci_mmlu",
            "/workspace/data/drafter_domains/verl/science/wildsci_all_aligned/train.parquet",
            "/workspace/data/drafter_domains/verl/science/eval/mmlu_pro_stem/test.parquet",
            "sequence=max_prompt=2048 max_response=2048 test_freq=50 val_before_train=False",
        ),
    ],
)
def test_dataset_profiles_select_paths_lengths_and_custom_reward(
    tmp_path: Path,
    profile: str,
    train_path: str,
    test_path: str,
    sequence: str,
) -> None:
    result = _run_dflash(
        tmp_path,
        _USE_PROFILE_PATHS="1",
        DATASET_PROFILE=profile,
    )

    assert result.returncode == 0, result.stderr
    assert f"dataset_profile={profile}" in result.stdout
    assert f"train={train_path} test={test_path}" in result.stdout
    assert sequence in result.stdout
    assert (
        "reward.custom_reward_function.path="
        "/workspace/data/drafter_domains/reward.py" in result.stdout
    )
    if profile != "dapo":
        assert (
            "+reward.custom_reward_function.reward_kwargs.reward_scale=pm1"
            in result.stdout
        )


def test_code_profile_passes_sandbox_as_custom_reward_kwargs(tmp_path: Path) -> None:
    result = _run_dflash(
        tmp_path,
        DATASET_PROFILE="taco_lcb",
        SANDBOX_FUSION_URL="http://sandbox.invalid/run_code",
        SANDBOX_MEMORY_LIMIT_MB="2048",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "+reward.custom_reward_function.reward_kwargs.sandbox_fusion_url="
        "http://sandbox.invalid/run_code" in result.stdout
    )
    assert (
        "+reward.custom_reward_function.reward_kwargs.memory_limit_mb=2048"
        in result.stdout
    )
    assert (
        "+reward.custom_reward_function.reward_kwargs.sandbox_max_concurrent=16"
        in result.stdout
    )


def test_code_profile_executes_local_prime_checker_without_pyext(tmp_path: Path) -> None:
    script = """
import importlib.util
import json

spec = importlib.util.spec_from_file_location(
    "drafter_domain_reward_test", "/workspace/data/drafter_domains/reward.py"
)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)
for data_source in ("livecodebench", "taco"):
    score = reward.compute_score(
        data_source=data_source,
        solution_str="```python\\nprint(input())\\n```",
        ground_truth=json.dumps({"inputs": ["hello\\n"], "outputs": ["hello\\n"]}),
    )
    assert score["score"] == 1.0, (data_source, score)
    assert score["raw_score"] == 1.0, (data_source, score)
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source /workspace/main_env_next/bin/activate-speco-cann901-v1.sh "
            "&& exec /workspace/main_env_next/bin/python -c \"$1\"",
            "bash",
            script,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dataset_profile_selects_wandb_project_without_dataset_in_run_name(
    tmp_path: Path,
) -> None:
    result = _run_dflash(tmp_path, DATASET_PROFILE="taco_lcb")

    assert result.returncode == 0, result.stderr
    assert (
        "experiment_id=dflash_taco_lcb_online_100step__p1"
        in result.stdout
    )
    assert (
        "wandb=project=verl_speco_taco group=qwen3_8b_dflash_comparison "
        "name=dflash_online_100step__p1"
        in result.stdout
    )


def test_offline_runs_do_not_append_an_online_strategy_tag(tmp_path: Path) -> None:
    result = _run_dflash(
        tmp_path,
        RUN_MODE="offline",
        SPECO_ONLINE_STRATEGY_TAG="custom-policy",
    )

    assert result.returncode == 0, result.stderr
    assert "name=dflash_offline_fixed_100step\n" in result.stdout


def test_explicit_wandb_name_is_not_changed_by_the_online_strategy_tag(
    tmp_path: Path,
) -> None:
    result = _run_dflash(
        tmp_path,
        WANDB_RUN_NAME="dflash-legacy-control",
        SPECO_ONLINE_STRATEGY_TAG="custom-policy",
    )

    assert result.returncode == 0, result.stderr
    assert "name=dflash-legacy-control\n" in result.stdout


def test_new_online_strategy_does_not_resume_a_legacy_checkpoint(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    legacy = checkpoint_root / "qwen3_8b_dflash_online_w16_100step_b64"
    legacy.mkdir(parents=True)
    (legacy / "latest_checkpointed_iteration.txt").write_text("82\n", encoding="utf-8")

    result = _run_dflash(tmp_path, CHECKPOINT_ROOT=str(checkpoint_root))

    expected = (
        checkpoint_root
        / "dflash_online_100step__p1"
    )
    assert result.returncode == 0, result.stderr
    assert f"checkpoint={expected}" in result.stdout


def test_checkpoint_wandb_path_rejects_conflicting_project(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "dflash_online_100step__p1"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".wandb_run_path").write_text(
        "linzzz-hong-kong-university-of-science-and-technology/"
        "verl_speco_dapo/f5c60c4fb99d\n",
        encoding="utf-8",
    )

    result = _run_dflash(
        tmp_path,
        CHECKPOINT_ROOT=str(checkpoint_root),
        WANDB_PROJECT="verl_speco_taco",
    )

    assert result.returncode != 0
    assert "conflicts with checkpoint W&B project verl_speco_dapo" in result.stderr


def test_checkpoint_wandb_path_rejects_conflicting_run_id(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "dflash_online_100step__p1"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".wandb_run_path").write_text(
        "linzzz-hong-kong-university-of-science-and-technology/"
        "verl_speco_dapo/f5c60c4fb99d\n",
        encoding="utf-8",
    )

    result = _run_dflash(
        tmp_path,
        CHECKPOINT_ROOT=str(checkpoint_root),
        WANDB_RUN_ID="different-run",
    )

    assert result.returncode != 0
    assert "conflicts with checkpoint W&B run ID f5c60c4fb99d" in result.stderr


def test_code_profile_rejects_placeholder_sandbox_url(tmp_path: Path) -> None:
    result = _run_dflash(
        tmp_path,
        DATASET_PROFILE="taco_lcb",
        SANDBOX_FUSION_URL="http://HOST:PORT/run_code",
    )

    assert result.returncode != 0
    assert "still contains the HOST:PORT placeholder" in result.stderr


def test_code_profile_rejects_sandbox_url_without_run_code(tmp_path: Path) -> None:
    result = _run_dflash(
        tmp_path,
        DATASET_PROFILE="taco_lcb",
        SANDBOX_FUSION_URL="http://sandbox.invalid",
    )

    assert result.returncode != 0
    assert "must end with /run_code" in result.stderr


def test_managed_dflash_rejects_unknown_dataset_profile(tmp_path: Path) -> None:
    result = _run_dflash(tmp_path, DATASET_PROFILE="not_a_dataset")

    assert result.returncode != 0
    assert "unknown DATASET_PROFILE=not_a_dataset" in result.stderr


def _write_complete_checkpoint(root: Path, step: int) -> None:
    actor = root / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    (actor / "model_world_size_1_rank_0.pt").touch()
    (actor / "extra_state_world_size_1_rank_0.pt").touch()
    draft = root / "drafter" / f"draft_step_{step}"
    draft.mkdir(parents=True)
    (draft / "model.safetensors").touch()
    (draft / "metadata.json").write_text(
        '{"complete": true}', encoding="utf-8"
    )


def test_retention_keeps_drafter_checkpoint_used_at_process_start(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_complete_checkpoint(checkpoint_root, 42)
    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_file.write_text("42\n", encoding="utf-8")
    sentinel = tmp_path / "done"
    process = subprocess.Popen(
        [
            str(ROOT / "tools" / "watch_checkpoint_retention.sh"),
            str(checkpoint_root),
            str(sentinel),
            "0.05",
        ],
        env={**os.environ, "SPECO_PROTECTED_ACTOR_STEPS": "48"},
    )
    try:
        time.sleep(0.1)
        _write_complete_checkpoint(checkpoint_root, 47)
        checkpoint_root.joinpath(
            "drafter", "best_checkpointed_iteration.txt"
        ).write_text("47\n", encoding="utf-8")
        _write_complete_checkpoint(checkpoint_root, 48)
        checkpoint_root.joinpath(
            "drafter", "best_acceptance_checkpointed_iteration.txt"
        ).write_text("48\n", encoding="utf-8")
        _write_complete_checkpoint(checkpoint_root, 49)
        checkpoint_root.joinpath(
            "drafter", "best_throughput_checkpointed_iteration.txt"
        ).write_text("49\n", encoding="utf-8")
        _write_complete_checkpoint(checkpoint_root, 52)
        replacement = checkpoint_root / "latest.tmp"
        replacement.write_text("52\n", encoding="utf-8")
        replacement.replace(latest_file)

        deadline = time.monotonic() + 3
        while (
            checkpoint_root.joinpath("global_step_42").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

        assert not checkpoint_root.joinpath("global_step_42").exists()
        assert checkpoint_root.joinpath("global_step_48").is_dir()
        assert checkpoint_root.joinpath("drafter", "draft_step_42").is_dir()
        assert checkpoint_root.joinpath("drafter", "draft_step_47").is_dir()
        assert checkpoint_root.joinpath("drafter", "draft_step_48").is_dir()
        assert checkpoint_root.joinpath("drafter", "draft_step_49").is_dir()
        assert checkpoint_root.joinpath("drafter", "draft_step_52").is_dir()
    finally:
        sentinel.touch()
        process.wait(timeout=3)


def test_retention_preserves_explicit_actor_and_drafter_steps(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    for step in (40, 50, 60):
        _write_complete_checkpoint(checkpoint_root, step)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("60\n")
    sentinel = tmp_path / "done"
    process = subprocess.Popen(
        [str(ROOT / "tools" / "watch_checkpoint_retention.sh"),
         str(checkpoint_root), str(sentinel), "0.05"],
        env={**os.environ, "SPECO_PROTECTED_ACTOR_STEPS": "50,60",
             "SPECO_PROTECTED_DRAFTER_STEPS": "50,60"},
    )
    try:
        deadline = time.monotonic() + 3
        while (checkpoint_root / "drafter" / "draft_step_40").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not (checkpoint_root / "global_step_40").exists()
        assert not (checkpoint_root / "drafter" / "draft_step_40").exists()
        for step in (50, 60):
            assert (checkpoint_root / f"global_step_{step}" / "actor").is_dir()
            assert (checkpoint_root / "drafter" / f"draft_step_{step}").is_dir()
    finally:
        sentinel.touch()
        process.wait(timeout=3)


def test_completed_drafter_archive_survives_checkpoint_cleanup(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints" / "experiment"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / ".wandb_run_id").write_text("run123\n", encoding="utf-8")
    drafter = checkpoint_root / "drafter" / "draft_step_100"
    optimizer = drafter / "optimizer"
    optimizer.mkdir(parents=True)
    (drafter / "config.json").write_text("{}", encoding="utf-8")
    (drafter / "pytorch_model.bin").write_bytes(b"trained weights")
    (optimizer / ".metadata").write_bytes(b"optimizer metadata")
    (optimizer / "trainer_state.pt").write_bytes(b"trainer state")
    (drafter / "metadata.json").write_text(
        '{"step":100,"complete":true,"optimizer":'
        '{"path":"optimizer","trainer_state_file":"trainer_state.pt"}}',
        encoding="utf-8",
    )
    incomplete = checkpoint_root / "drafter" / "draft_step_101"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    (incomplete / "pytorch_model.bin").write_bytes(b"partial weights")
    (incomplete / "metadata.json").write_text(
        '{"step":101,"complete":false}', encoding="utf-8"
    )
    checkpoint_root.joinpath("drafter", "final_checkpointed_iteration.txt").write_text(
        "100\n", encoding="utf-8"
    )
    best = checkpoint_root / "drafter" / "draft_step_90"
    best.mkdir()
    (best / "config.json").write_text("{}", encoding="utf-8")
    (best / "pytorch_model.bin").write_bytes(b"best weights")
    (best / "metadata.json").write_text(
        '{"step":90,"complete":true}', encoding="utf-8"
    )
    checkpoint_root.joinpath("drafter", "best_checkpointed_iteration.txt").write_text(
        "90\n", encoding="utf-8"
    )
    checkpoint_root.joinpath("drafter", "best_candidate.json").write_text(
        '{"step":90,"mean_acceptance_length":4.0}\n', encoding="utf-8"
    )
    for kind, step, score in (
        ("acceptance", 80, '"mean_acceptance_length":4.1'),
        ("throughput", 85, '"response_tokens_per_second":5000.0'),
    ):
        dual = checkpoint_root / "drafter" / f"draft_step_{step}"
        dual.mkdir()
        (dual / "config.json").write_text("{}", encoding="utf-8")
        (dual / "pytorch_model.bin").write_bytes(f"best {kind}".encode())
        (dual / "metadata.json").write_text(
            f'{{"step":{step},"complete":true}}', encoding="utf-8"
        )
        checkpoint_root.joinpath(
            "drafter", f"best_{kind}_checkpointed_iteration.txt"
        ).write_text(f"{step}\n", encoding="utf-8")
        checkpoint_root.joinpath("drafter", f"best_{kind}.json").write_text(
            f'{{"step":{step},{score}}}\n', encoding="utf-8"
        )
    archive_root = tmp_path / "drafters"
    command = f"""
set -euo pipefail
source {ROOT / 'tools' / 'speco_vllm_experiment_common.sh'}
saved=$(speco_archive_final_drafter \
    {checkpoint_root} {archive_root} experiment run123 entity/project/run123 python3)
speco_safe_remove_checkpoint {checkpoint_root}
printf '%s\\n' "$saved"
"""

    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    saved = Path(result.stdout.strip())
    assert saved == archive_root / "experiment" / "run123" / "draft_step_100"
    assert not checkpoint_root.exists()
    assert (saved / "pytorch_model.bin").read_bytes() == b"trained weights"
    assert (saved / "optimizer" / "trainer_state.pt").is_file()
    assert (
        saved.parent.joinpath("draft_step_90", "pytorch_model.bin").read_bytes()
        == b"best weights"
    )
    assert saved.parent.joinpath("best_checkpointed_iteration.txt").read_text(
        encoding="utf-8"
    ) == "90\n"
    assert saved.parent.joinpath(
        "best_acceptance_checkpointed_iteration.txt"
    ).read_text(encoding="utf-8") == "80\n"
    assert saved.parent.joinpath(
        "best_throughput_checkpointed_iteration.txt"
    ).read_text(encoding="utf-8") == "85\n"
    assert (
        saved.parent.joinpath("draft_step_80", "pytorch_model.bin").read_bytes()
        == b"best acceptance"
    )
    assert (
        saved.parent.joinpath("draft_step_85", "pytorch_model.bin").read_bytes()
        == b"best throughput"
    )
    assert saved.parent.joinpath("best_acceptance.json").is_file()
    assert saved.parent.joinpath("best_throughput.json").is_file()
    assert (
        saved.parent.joinpath("latest_checkpointed_iteration.txt").read_text(
            encoding="utf-8"
        )
        == "100\n"
    )
    assert (
        saved.parent.joinpath("final_checkpointed_iteration.txt").read_text(
            encoding="utf-8"
        )
        == "100\n"
    )
