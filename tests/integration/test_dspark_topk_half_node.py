import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("arm,response,rejected", [
    ("k3", "k3", "k3"), ("response", "topk", "k3"),
    ("rejected", "k3", "topk"), ("both", "topk", "topk"),
])
def test_half_node_preserves_global_budgets(tmp_path, arm, response, rejected):
    model, drafter = tmp_path / "model", tmp_path / "drafter"
    model.mkdir()
    drafter.mkdir()
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    train.touch()
    test.touch()
    env = {**os.environ, "DRY_RUN": "1", "DSPARK_TOPK_ARM": arm,
           "MODEL_PATH": str(model), "DRAFTER_PATH": str(drafter),
           "TRAIN_FILE": str(train), "TEST_FILE": str(test),
           "CANN_ENV": str(tmp_path / "missing.sh")}
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_dspark_topk_half_node.sh")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "visible_npus=8,9,10,11,12,13,14,15" in result.stdout
    command = next(line.removeprefix("command: ") for line in result.stdout.splitlines() if line.startswith("command: "))
    overrides = dict(value.split("=", 1) for value in shlex.split(command)[2:])
    prefix = "actor_rollout_ref.rollout.drafter.training."
    assert overrides[prefix + "dspark_response_loss"] == response
    assert overrides[prefix + "dspark_rejected_loss"] == rejected
    assert overrides[prefix + "dspark_topk"] == "16"
    world = int(overrides["trainer.n_gpus_per_node"])
    tp = int(overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"])
    replicas = world // tp
    assert (world, tp, replicas) == (8, 4, 2)
    assert replicas * int(overrides[prefix + "batch_size_per_gpu"]) == 16
    assert replicas * int(overrides[prefix + "max_collect_samples_per_step_per_replica"]) == 80
    assert replicas * int(overrides[prefix + "max_collect_tokens_per_step_per_replica"]) == 65536
    assert replicas * int(overrides[prefix + "fixed_eval_samples_per_replica"]) == 16
    assert overrides[prefix + "step"] == "5"
    assert overrides["data.train_batch_size"] == "64"
    assert overrides["actor_rollout_ref.rollout.n"] == "5"
    assert overrides["trainer.total_training_steps"] == "101"
    assert overrides[prefix + "dspark_context_window"] == "512"
    assert overrides[prefix + "dspark_max_window"] == "1024"
    assert overrides[prefix + "dual_metric_best_checkpoint_enabled"] == "true"
    assert overrides["trainer.save_freq"] == "10"
    assert overrides["speco.checkpoint.save_offset"] == "9"


@pytest.mark.parametrize("tag,steps,samples,token_cap,first,suffix,ce", [
    ("p4_01", "8", "64", "32832", "2.0", "0.4", "0.0"),
    ("p4_03", "6", "48", "32768", "1.0", "1.0", "0.2"),
])
def test_p4_accepts_first_rejection_experiment_overrides(tmp_path, tag, steps, samples, token_cap, first, suffix, ce):
    model, drafter = tmp_path / "model", tmp_path / "drafter"
    model.mkdir()
    drafter.mkdir()
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    train.touch()
    test.touch()
    env = {**os.environ, "DRY_RUN": "1", "SPECO_ONLINE_STRATEGY_TAG": tag,
           "DRAFT_TRAIN_STEPS": steps, "DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA": samples,
           "DRAFT_MAX_COLLECT_TOKENS_PER_REPLICA": token_cap,
           "DSPARK_REJECTED_FIRST_WEIGHT_MULTIPLIER": first,
           "DSPARK_REJECTED_SUFFIX_WEIGHT_MULTIPLIER": suffix,
           "DSPARK_FIRST_REJECTION_CE_ALPHA": ce,
           "MODEL_PATH": str(model), "DRAFTER_PATH": str(drafter),
           "TRAIN_FILE": str(train), "TEST_FILE": str(test),
           "CANN_ENV": str(tmp_path / "missing.sh")}
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_dspark_p4.sh"),
         "actor_rollout_ref.rollout.drafter.training.fixed_eval_export_steps=[]",
         "actor_rollout_ref.rollout.drafter.training.fixed_eval_export_samples=0",
         "trainer.max_actor_ckpt_to_keep=1"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert f"experiment_id=dspark_online_101step__{tag}" in result.stdout
    command = next(line.removeprefix("command: ") for line in result.stdout.splitlines() if line.startswith("command: "))
    overrides = dict(value.split("=", 1) for value in shlex.split(command)[2:])
    prefix = "actor_rollout_ref.rollout.drafter.training."
    assert overrides[prefix + "dspark_rejected_first_weight_multiplier"] == first
    assert overrides[prefix + "dspark_rejected_suffix_weight_multiplier"] == suffix
    assert overrides[prefix + "dspark_first_rejection_ce_alpha"] == ce
    assert float(overrides[prefix + "dspark_ce_loss_alpha"]) == 0.0
    assert overrides[prefix + "dspark_rejected_position_decay"] == "0.8"
    assert overrides[prefix + "dspark_rejected_stream_weight"] == "0.5"
    assert overrides[prefix + "dspark_response_loss"] == "topk"
    assert overrides[prefix + "dspark_rejected_loss"] == "topk"
    assert overrides[prefix + "dspark_attention_mode"] == "full"
    assert overrides["actor_rollout_ref.rollout.drafter.vllm.dspark_attention_mode"] == "full"
    assert overrides[prefix + "dspark_use_absolute_position_ids"] == "false"
    assert overrides[prefix + "step"] == steps
    assert overrides[prefix + "batch_size_per_gpu"] == "8"
    assert overrides[prefix + "max_collect_samples_per_step_per_replica"] == samples
    assert overrides[prefix + "max_collect_tokens_per_step_per_replica"] == token_cap
    assert int(token_cap) >= int(samples) * (512 + 1)
    assert overrides[prefix + "fixed_eval_export_steps"] == "[]"
    assert overrides[prefix + "fixed_eval_export_samples"] == "0"
    assert overrides[prefix + "dspark_dump_k3_logprobs"] == "true"
    assert overrides[prefix + "dual_metric_best_checkpoint_enabled"] == "true"
    assert overrides["trainer.max_actor_ckpt_to_keep"] == "1"
    assert overrides["trainer.save_freq"] == "10"
    assert overrides["speco.checkpoint.save_offset"] == "9"
    assert overrides["trainer.total_training_steps"] == "101"
    assert overrides[prefix + "dspark_context_window"] == "null"
    assert overrides[prefix + "dspark_max_window"] == "512"


def test_half_node_refuses_first_half_devices():
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_dspark_topk_half_node.sh")],
        env={**os.environ, "DRY_RUN": "1", "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "physical cards 4-7" in result.stderr


@pytest.mark.parametrize("support", ["student", "union"])
def test_p4_overrides_swa_profile_with_p2_02_full_attention(tmp_path, support):
    model, drafter = tmp_path / "model", tmp_path / "drafter"
    model.mkdir()
    drafter.mkdir()
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    train.touch()
    test.touch()
    env = {**os.environ, "DRY_RUN": "1", "DSPARK_TOPK_SUPPORT": support,
           "MODEL_PATH": str(model), "DRAFTER_PATH": str(drafter),
           "TRAIN_FILE": str(train), "TEST_FILE": str(test),
           "CANN_ENV": str(tmp_path / "missing.sh")}
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_dspark_p4.sh")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "experiment_id=dspark_online_101step__p4" in result.stdout
    command = next(line.removeprefix("command: ") for line in result.stdout.splitlines() if line.startswith("command: "))
    overrides = dict(value.split("=", 1) for value in shlex.split(command)[2:])
    prefix = "actor_rollout_ref.rollout.drafter.training."
    assert overrides[prefix + "dspark_response_loss"] == "topk"
    assert overrides[prefix + "dspark_rejected_loss"] == "topk"
    assert overrides[prefix + "dspark_topk"] == "16"
    assert overrides[prefix + "dspark_topk_support"] == support
    assert overrides[prefix + "dspark_attention_mode"] == "full"
    assert overrides[prefix + "dspark_context_window"] == "null"
    assert overrides[prefix + "dspark_max_window"] == "512"
    assert overrides[prefix + "hidden_state_window_tokens_per_sample"] == "512"
    assert overrides[prefix + "dspark_use_absolute_position_ids"] == "false"
    assert overrides["actor_rollout_ref.rollout.drafter.vllm.dspark_attention_mode"] == "full"
    assert overrides["actor_rollout_ref.rollout.drafter.vllm.dspark_context_window"] == "null"
    assert overrides[prefix + "dspark_rejected_position_decay"] == "0.8"
    assert overrides[prefix + "dspark_dynamic_accept_reject_weighting_enabled"] == "false"
    assert overrides[prefix + "step"] == "5"
    assert overrides[prefix + "max_collect_samples_per_step_per_replica"] == "40"
    assert overrides[prefix + "dspark_rejected_first_weight_multiplier"] == "1.0"
    assert overrides[prefix + "dspark_rejected_suffix_weight_multiplier"] == "1.0"
    assert overrides[prefix + "fixed_eval_export_steps"] == "[60,80,100]"
    assert overrides[prefix + "fixed_eval_export_samples"] == "128"
    assert overrides[prefix + "dspark_dump_k3_logprobs"] == "true"
    assert overrides[prefix + "dual_metric_best_checkpoint_enabled"] == "true"
    assert overrides["trainer.max_actor_ckpt_to_keep"] == "null"
    assert overrides["trainer.total_training_steps"] == "101"
