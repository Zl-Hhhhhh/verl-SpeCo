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

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_HIDDEN_REFS_KEY,
    _assign_oldlogprob_non_tensor_output,
)
from verl_speco.workers.speco_worker import _resolve_sample_hidden_state_ref

torch = pytest.importorskip("torch")


_speco_ray_trainer = pytest.importorskip(
    "verl_speco.trainer.speco_ray_trainer",
    reason="drafter runtime control contract needs the trainer dependency stack",
)
SpecoRayPPOTrainer = _speco_ray_trainer.SpecoRayPPOTrainer


class _FakeOldLogProbBatch:
    non_tensor_batch = {}

    def __init__(self) -> None:
        self.selected_non_tensor_keys = None

    def select(self, *, non_tensor_batch_keys=None, **kwargs):
        self.selected_non_tensor_keys = non_tensor_batch_keys
        return self

    def to_tensordict(self):
        raise AssertionError(
            "non-collect old-logprob steps should not enter the collection compute path"
        )


class _FakeRolloutWorkerGroup:
    def __init__(self) -> None:
        self.compute_log_prob_calls = 0

    def generate_sequences(self, *args, **kwargs):
        return SimpleNamespace(meta_info={"metrics": {}})

    def compute_log_prob(self, batch):
        self.compute_log_prob_calls += 1
        raise AssertionError(
            "non-collect old-logprob steps should use the original compute path"
        )


def _trainer(training_cfg: dict, *, step: int = 1) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=True,
                    enable_drafter_training=True,
                    training=training_cfg,
                )
            ),
        )
    )
    trainer._pending_drafter_publish_refs = None
    trainer._speco_last_collected_samples = 0
    trainer._ray_get_if_needed = lambda value: value
    return trainer


def _oldlogprob_plan_trainer(
    training_cfg: dict, *, owner_count: int = 1
) -> SpecoRayPPOTrainer:
    trainer = _trainer(training_cfg)
    trainer._speco_oldlogprob_collection_enabled = lambda: True
    trainer._speco_should_collect_drafter_this_step = lambda: True
    trainer._speco_should_train_drafter_this_step = lambda: True
    trainer._speco_owner_bucket_count = lambda: owner_count
    return trainer


def _oldlogprob_batch(response_lengths: list[int]):
    batch_size = len(response_lengths)
    prompt_width = 4
    response_width = max(response_lengths)
    prompt_mask = torch.ones(batch_size, prompt_width, dtype=torch.long)
    response_mask = torch.zeros(batch_size, response_width, dtype=torch.long)
    for index, response_len in enumerate(response_lengths):
        response_mask[index, :response_len] = 1
    return SimpleNamespace(
        batch={
            "prompts": torch.ones(batch_size, prompt_width, dtype=torch.long),
            "responses": torch.ones(batch_size, response_width, dtype=torch.long),
            "attention_mask": torch.cat([prompt_mask, response_mask], dim=1),
            "response_mask": response_mask,
        },
        non_tensor_batch={},
    )


def test_fixed_eval_export_is_optional_and_step_scoped() -> None:
    _trainer({})._speco_export_fixed_eval_snapshot(None, None)
    _trainer({"fixed_eval_export_steps": [60]}, step=59)._speco_export_fixed_eval_snapshot(None, None)


def test_fixed_eval_export_records_pre_update_actor(tmp_path, monkeypatch) -> None:
    trainer = _trainer({"fixed_eval_export_steps": [60], "fixed_eval_export_samples": 2}, step=60)
    trainer.config.trainer = SimpleNamespace(experiment_name="p4-test", default_local_dir=str(tmp_path / "checkpoints"))
    actor_dir = tmp_path / "checkpoints" / "global_step_59" / "actor"
    actor_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    trainer._speco_export_fixed_eval_snapshot(
        _oldlogprob_batch([8, 10, 12]), {"collect_mask": torch.tensor([True, False, False])}
    )
    manifest = json.loads((tmp_path / "outputs/drafter_analysis/p4-test/test-run/fixed_eval_samples/step_60/manifest.json").read_text())
    assert manifest["actor_checkpoint"] == str(actor_dir)
    assert manifest["actor_snapshot_step"] == 59
    assert set(manifest["sample_indices"]) == {1, 2}
    assert trainer._speco_fixed_eval_export_metrics["drafter/fixed_export/samples"] == 2


def test_oldlogprob_plan_keeps_responses_shorter_than_fixed_window() -> None:
    trainer = _oldlogprob_plan_trainer(
        {
            "collection_sample_rate": 1.0,
            "hidden_state_window_tokens_per_sample": 512,
            "hidden_state_window_mode": "stratified",
            "max_collect_samples_per_step_per_replica": 8,
        }
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(
        _oldlogprob_batch([8, 128, 512])
    )

    assert plan is not None
    assert plan["selected_count"] == 3
    assert plan["hidden_position_mask"].sum(dim=1).tolist() == [8, 128, 512]


def test_oldlogprob_cap_uses_hash_priority_before_batch_order() -> None:
    trainer = _oldlogprob_plan_trainer(
        {
            "collection_sample_rate": 1.0,
            "hidden_state_window_tokens_per_sample": 16,
            "hidden_state_window_mode": "front",
            "max_collect_samples_per_step_per_replica": 1,
        },
        owner_count=2,
    )
    response_lengths = [32] * 8

    batch = _oldlogprob_batch(response_lengths)
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan is not None
    priorities = [
        (
            trainer._speco_hash_fraction(
                "1:"
                f"{batch_idx}:"
                f"{trainer._speco_sample_identity(batch, batch_idx, batch.batch['prompts'][batch_idx])}:"
                "4:32:collection-reservoir"
            ),
            batch_idx,
        )
        for batch_idx in range(len(response_lengths))
    ]
    expected = {index for _, index in sorted(priorities)[:2]}
    selected = set(
        torch.nonzero(plan["collect_mask"], as_tuple=False).reshape(-1).tolist()
    )
    assert selected == expected


def test_oldlogprob_stratified_windows_cover_each_sequence_third() -> None:
    seen = {}
    for index in range(1000):
        offset, stratum = SpecoRayPPOTrainer._speco_oldlogprob_stratified_window_offset(
            sample_key=f"sample-{index}",
            max_start_offset=299,
            weights=(0.34, 0.33, 0.33),
        )
        seen.setdefault(stratum, offset)
        if len(seen) == 3:
            break

    assert set(seen) == {"front", "middle", "tail"}
    assert 0 <= seen["front"] < 100
    assert 100 <= seen["middle"] < 200
    assert 200 <= seen["tail"] <= 299


def test_oldlogprob_strata_weights_accept_omegaconf_listconfig() -> None:
    training_cfg = OmegaConf.create(
        {"hidden_state_window_strata_weights": [2.0, 3.0, 5.0]}
    )

    weights = SpecoRayPPOTrainer._speco_oldlogprob_window_strata_weights(training_cfg)

    assert weights == pytest.approx((0.2, 0.3, 0.5))


def test_oldlogprob_stratified_plan_honors_configured_proportions() -> None:
    trainer = _oldlogprob_plan_trainer(
        {
            "collection_sample_rate": 1.0,
            "hidden_state_window_tokens_per_sample": 512,
            "hidden_state_window_mode": "stratified",
            "hidden_state_window_strata_weights": [0.34, 0.33, 0.33],
            "max_collect_samples_per_step_per_replica": 9,
        }
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(
        _oldlogprob_batch([900] * 12)
    )

    assert plan is not None
    selected_strata = [
        stratum for stratum in plan["window_strata"] if stratum is not None
    ]
    assert selected_strata.count("front") == 3
    assert selected_strata.count("middle") == 3
    assert selected_strata.count("tail") == 3


def test_oldlogprob_sample_aligned_refs_survive_dp_concat() -> None:
    shards = []
    for rank in range(4):
        shard = TensorDict({"rank": torch.tensor([rank])}, batch_size=[1])
        _assign_oldlogprob_non_tensor_output(
            shard, OLD_LOGPROB_HIDDEN_REFS_KEY, [f"ref-{rank}"]
        )
        shards.append(shard)

    merged = torch.cat(shards, dim=0)

    assert list(tu.get(merged, OLD_LOGPROB_HIDDEN_REFS_KEY)) == [
        "ref-0",
        "ref-1",
        "ref-2",
        "ref-3",
    ]


def test_worker_slices_sample_from_shared_hidden_ref_chunk() -> None:
    shared_chunk = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    hidden = _resolve_sample_hidden_state_ref(
        shared_chunk,
        {"chunk_start": 4, "chunk_length": 3},
        expected_rows=3,
    )

    assert hidden.shape == (1, 3, 3)
    assert torch.equal(hidden[0], shared_chunk[4:7])


def test_drafter_collection_guard_rejects_missing_trace() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "spec_trace_min_sample_coverage": 0.95,
        }
    )
    trainer._speco_last_oldlogprob_planned_samples = 16
    trainer._speco_last_oldlogprob_collected_samples = 16
    trainer._speco_last_oldlogprob_dispatched_samples = 16
    trainer._speco_last_oldlogprob_consumed_samples = 16
    trainer._speco_last_spec_trace_samples = 0
    trainer._speco_last_spec_trace_diagnostics = {
        "engine_scheduled_draft_tokens": 48
    }

    assert trainer._speco_drafter_collection_guard() == (
        False,
        "spec_trace_coverage",
    )


def test_drafter_collection_guard_rejects_missing_teacher_logprobs() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "spec_trace_min_sample_coverage": 0.95,
            "spec_trace_min_teacher_logprob_coverage": 0.95,
            "dspark_rejected_k3_alpha": 0.25,
        }
    )
    trainer._speco_last_oldlogprob_planned_samples = 16
    trainer._speco_last_oldlogprob_collected_samples = 16
    trainer._speco_last_oldlogprob_dispatched_samples = 16
    trainer._speco_last_oldlogprob_consumed_samples = 16
    trainer._speco_last_spec_trace_samples = 16
    trainer._speco_last_spec_trace_draft_tokens = 100
    trainer._speco_last_spec_trace_teacher_logprob_tokens = 50
    trainer._speco_last_spec_trace_diagnostics = {
        "engine_scheduled_draft_tokens": 100
    }

    assert trainer._speco_drafter_collection_guard() == (
        False,
        "spec_trace_teacher_logprob_coverage",
    )


def test_drafter_collection_guard_rejects_dp_consumption_loss() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "oldlogprob_min_consumption_ratio": 0.95,
        }
    )
    trainer._speco_last_oldlogprob_planned_samples = 64
    trainer._speco_last_oldlogprob_collected_samples = 64
    trainer._speco_last_oldlogprob_dispatched_samples = 64
    trainer._speco_last_oldlogprob_consumed_samples = 16

    assert trainer._speco_drafter_collection_guard() == (
        False,
        "worker_consumption_coverage",
    )


def _no_drafter_trainer(*, calculate_entropy=Ellipsis) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    actor = SimpleNamespace()
    if calculate_entropy is not Ellipsis:
        actor.calculate_entropy = calculate_entropy
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=actor,
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=False,
                    enable_drafter_training=False,
                    training={},
                )
            ),
        )
    )
    return trainer


def test_drafter_collect_train_and_publish_intervals() -> None:
    trainer = _trainer(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
        },
        step=6,
    )

    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is True
    assert trainer._speco_should_publish_drafter_weights(True) is False

    trainer.global_steps = 8
    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is False
    assert trainer._speco_should_publish_drafter_weights(True) is True
    assert trainer._speco_should_publish_drafter_weights(False) is False


@pytest.mark.parametrize(
    ("step", "expected"),
    [(1, False), (10, False), (11, True), (20, False), (21, True), (100, False)],
)
def test_shifted_checkpoint_schedule(step: int, expected: bool) -> None:
    assert _speco_ray_trainer.speco_step_matches_save_schedule(step, 10, 1) is expected


def test_checkpoint_offset_hook_saves_shifted_steps_and_restores_config() -> None:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_freq=10),
        speco=SimpleNamespace(checkpoint=SimpleNamespace(save_offset=1)),
    )
    trainer.global_steps = 10
    actor_calls = []
    saved_steps = []
    trainer._update_actor = lambda batch: actor_calls.append(batch) or {"batch": batch}
    trainer._save_checkpoint = lambda: saved_steps.append(trainer.global_steps)

    with trainer._speco_checkpoint_offset_hook():
        assert trainer.config.trainer.save_freq == -1
        assert trainer._update_actor("step-10") == {"batch": "step-10"}
        trainer.global_steps = 11
        assert trainer._update_actor("step-11") == {"batch": "step-11"}
        trainer._update_actor("step-11-repeat")
        trainer.global_steps = 21
        trainer._update_actor("step-21")

    assert trainer.config.trainer.save_freq == 10
    assert actor_calls == ["step-10", "step-11", "step-11-repeat", "step-21"]
    assert saved_steps == [11, 21]


def test_successful_fit_saves_terminal_drafter_checkpoint(monkeypatch) -> None:
    trainer = _trainer({"save_full_drafter_checkpoint": True}, step=100)
    events = []
    trainer._speco_checkpoint_offset_hook = lambda: nullcontext()
    trainer._speco_tracking_metrics_hook = lambda: nullcontext()
    trainer._speco_online_fit_hooks = lambda: nullcontext()
    trainer._speco_activate_drafter_training_model_before_fit = lambda: None
    trainer._speco_wait_pending_drafter_checkpoint = lambda: events.append("wait")
    trainer._speco_save_drafter_checkpoint = (
        lambda *, wait=True: events.append(("save", trainer.global_steps, wait))
    )
    trainer._speco_mark_final_drafter_checkpoint = lambda: events.append("mark")
    monkeypatch.setattr(
        _speco_ray_trainer.RayPPOTrainer,
        "fit",
        lambda self: events.append("fit") or "finished",
    )

    assert trainer.fit() == "finished"
    assert events == ["fit", "wait", ("save", 100, True), "mark"]


def test_failed_fit_does_not_mark_drafter_as_final(monkeypatch) -> None:
    trainer = _trainer({"save_full_drafter_checkpoint": True}, step=100)
    events = []
    trainer._speco_checkpoint_offset_hook = lambda: nullcontext()
    trainer._speco_tracking_metrics_hook = lambda: nullcontext()
    trainer._speco_online_fit_hooks = lambda: nullcontext()
    trainer._speco_activate_drafter_training_model_before_fit = lambda: None
    trainer._speco_wait_pending_drafter_checkpoint = lambda: events.append("wait")
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: events.append("save")
    trainer._speco_mark_final_drafter_checkpoint = lambda: events.append("mark")

    def fail_fit(self):
        events.append("fit")
        raise RuntimeError("training failed")

    monkeypatch.setattr(_speco_ray_trainer.RayPPOTrainer, "fit", fail_fit)

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit()
    assert events == ["fit", "wait"]


def test_drafter_worker_pool_limit_uses_rollout_replica_multiple(monkeypatch) -> None:
    full_pool = SimpleNamespace(world_size=16)
    limited_pool = SimpleNamespace(world_size=4)
    rollout = SimpleNamespace(
        tensor_model_parallel_size=2,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    splits = []
    monkeypatch.setattr(
        _speco_ray_trainer,
        "split_resource_pool",
        lambda pool, split_size: splits.append((pool, split_size))
        or [limited_pool, object()],
    )

    assert (
        _speco_ray_trainer._speco_limit_drafter_resource_pool(full_pool, 4, rollout)
        is limited_pool
    )
    assert splits == [(full_pool, [4, 12])]
    assert (
        _speco_ray_trainer._speco_limit_drafter_resource_pool(
            full_pool, None, rollout
        )
        is full_pool
    )

    with pytest.raises(ValueError, match="divisible by the rollout replica"):
        _speco_ray_trainer._speco_limit_drafter_resource_pool(full_pool, 3, rollout)
    with pytest.raises(ValueError, match=r"must be in \[1, 16\]"):
        _speco_ray_trainer._speco_limit_drafter_resource_pool(full_pool, 17, rollout)


def test_drafter_training_attempt_requires_interval_and_samples() -> None:
    trainer = _trainer({"training_interval_steps": 5}, step=4)
    trainer._speco_last_collected_samples = 10
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer.global_steps = 5
    trainer._speco_last_collected_samples = 0
    trainer._speco_oldlogprob_collection_requested = lambda: True
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer._speco_last_collected_samples = 1
    assert trainer._speco_should_attempt_drafter_train_this_step() is True


def test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config() -> None:
    assert (
        _no_drafter_trainer(
            calculate_entropy=False
        )._speco_oldlogprob_entropy_hook_enabled()
        is True
    )
    assert (
        _no_drafter_trainer(
            calculate_entropy=True
        )._speco_oldlogprob_entropy_hook_enabled()
        is False
    )
    assert _no_drafter_trainer()._speco_oldlogprob_entropy_hook_enabled() is False


def test_no_drafter_vllm_path_disables_async_scheduling_without_hiding_config(
    monkeypatch,
) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter scheduler contract needs verl and Ray",
    )
    from omegaconf import OmegaConf
    from verl_speco.integration import vllm_runtime

    bridge_calls = []
    monkeypatch.setattr(
        vllm_runtime,
        "install_upstream_vllm_runtime_bridge",
        lambda: bridge_calls.append("installed") or True,
    )

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"enable": False},
                    "engine_kwargs": {"vllm": {}},
                }
            }
        }
    )

    with task_runner._prepare_no_drafter_runtime_config(config):
        from verl_speco.integration.vllm_runtime import (
            SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
        )

        assert config.actor_rollout_ref.rollout.drafter.enable is False
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm["no-async-scheduling"]
            is True
        )
        assert (
            config.actor_rollout_ref.rollout.engine_kwargs.vllm["worker_extension_cls"]
            == SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS
        )
    assert bridge_calls == ["installed"]

    assert "drafter" in config.actor_rollout_ref.rollout
    assert (
        "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )
    assert (
        "worker_extension_cls"
        not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )


def test_task_runner_installs_vllm_import_compat_in_its_own_process(
    monkeypatch,
) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="task-runner import compatibility needs verl and Ray",
    )
    from omegaconf import OmegaConf
    from verl_speco.integration import verl_npu_vllm_compat

    calls = []
    monkeypatch.setattr(
        verl_npu_vllm_compat,
        "install_verl_npu_vllm_import_compat",
        lambda: calls.append("compat") or True,
    )

    assert task_runner._install_vllm_import_compat_for_task_runner(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm"}}})
    )
    assert not task_runner._install_vllm_import_compat_for_task_runner(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "sglang"}}})
    )
    assert calls == ["compat"]


def test_no_drafter_run_keeps_speco_entropy_control(monkeypatch) -> None:
    task_runner = pytest.importorskip(
        "verl_speco.integration.task_runner",
        reason="no-drafter trainer contract needs verl and Ray",
    )
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {"enable": False},
                    "engine_kwargs": {"vllm": {}},
                }
            }
        }
    )
    runner = task_runner.SpecoTaskRunner.__new__(task_runner.SpecoTaskRunner)
    observed = {}

    def fake_run_with_speco_trainer(self, active_config):
        del self
        observed["drafter_present"] = (
            "drafter" in active_config.actor_rollout_ref.rollout
        )
        observed["no_async"] = (
            active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
                "no-async-scheduling"
            ]
        )
        observed["worker_extension_cls"] = (
            active_config.actor_rollout_ref.rollout.engine_kwargs.vllm[
                "worker_extension_cls"
            ]
        )
        return "ran"

    monkeypatch.setattr(
        task_runner.SpecoTaskRunner,
        "_run_with_speco_trainer",
        fake_run_with_speco_trainer,
    )

    assert runner.run(config) == "ran"
    from verl_speco.integration.vllm_runtime import (
        SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    )

    assert observed == {
        "drafter_present": True,
        "no_async": True,
        "worker_extension_cls": SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    }
    assert (
        "no-async-scheduling" not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )
    assert (
        "worker_extension_cls"
        not in config.actor_rollout_ref.rollout.engine_kwargs.vllm
    )


def test_oldlogprob_non_collect_step_uses_original_compute_path() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 2,
            "training_interval_steps": 1,
        },
        step=1,
    )
    trainer.config.actor_rollout_ref.actor.calculate_entropy = True
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp"
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer._update_actor = lambda *args, **kwargs: SimpleNamespace(
        meta_info={"metrics": {}}
    )
    original_calls = []

    def original_compute_old_log_prob(batch):
        original_calls.append(batch)
        return "old-log-prob", 0.5

    trainer._compute_old_log_prob = original_compute_old_log_prob
    batch = _FakeOldLogProbBatch()

    with trainer._speco_online_fit_hooks():
        result = trainer._compute_old_log_prob(batch)

    assert result == ("old-log-prob", 0.5)
    assert original_calls == [batch]
    assert batch.selected_non_tensor_keys is None
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 0
    assert trainer._speco_last_collect_interval_matched == 0


def test_dspark_l1_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({"dspark_l1_loss_alpha": 0.9}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"


def test_dspark_default_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"
    assert trainer._speco_get_drafter_target_lm_head_row_selection() is None


def test_dspark_ce_only_oldlogprob_layout_keeps_aux_only_hidden() -> None:
    trainer = _trainer(
        {
            "dspark_l1_loss_alpha": 0.0,
            "dspark_accepted_k3_alpha": 0.0,
            "dspark_accepted_fkl_alpha": 0.0,
        },
        step=1,
    )
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux"


def test_dspark_accepted_k3_collects_final_hidden_without_l1() -> None:
    trainer = _trainer(
        {
            "dspark_l1_loss_alpha": 0.0,
            "dspark_accepted_k3_alpha": 1.0,
            "dspark_accepted_fkl_alpha": 0.0,
        },
        step=1,
    )
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"
    assert trainer._speco_get_drafter_target_lm_head_row_selection() is None


def test_key_metric_aliases_collect_exactly_four_dashboard_metrics() -> None:
    metrics = {
        "drafter/spec_decode/mean_acceptance_length": 2.5,
        "timing_s/gen": 6.25,
        "timing_s/step": 37.5,
        "critic/rewards/mean": 0.75,
        "perf/throughput": 3.4,
    }

    result = _speco_ray_trainer._speco_add_key_metric_aliases(metrics)

    assert {key: value for key, value in result.items() if key.startswith("key_metrics/")} == {
        "key_metrics/mean_acceptance_length": 2.5,
        "key_metrics/generation_time_s": 6.25,
        "key_metrics/step_time_s": 37.5,
        "key_metrics/critic_reward_mean": 0.75,
    }
    assert result["perf/throughput"] == 3.4


def test_validation_rollout_metrics_aggregate_spec_decode_and_throughput() -> None:
    trainer = _trainer({}, step=0)
    trainer._speco_validation_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 20.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 36.0,
    }
    trainer._speco_validation_generation_elapsed_sec = 4.0
    trainer._speco_validation_response_tokens = 100

    metrics = trainer._speco_validation_spec_decode_metrics()
    prefix = _speco_ray_trainer._SPECO_VALIDATION_SPEC_DECODE_PREFIX

    assert metrics[f"{prefix}/mean_acceptance_length"] == pytest.approx(2.8)
    assert metrics[f"{prefix}/verify_steps"] == 20.0
    assert metrics[f"{prefix}/accepted_tokens"] == 36.0
    assert metrics[f"{prefix}/generation_time_s"] == 4.0
    assert metrics[f"{prefix}/response_tokens"] == 100.0
    assert metrics[f"{prefix}/response_tokens_per_second"] == 25.0


def test_rollout_metrics_hook_collects_validation_without_training_metrics() -> None:
    class FakeRolloutManager:
        def generate_sequences(self, batch):
            del batch
            return SimpleNamespace(
                batch={"response_mask": torch.ones((2, 3), dtype=torch.long)},
                non_tensor_batch={
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": [4, 6],
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": [5, 9],
                },
                meta_info={"validate": True},
            )

    trainer = _trainer({}, step=0)
    trainer.async_rollout_manager = FakeRolloutManager()
    trainer._speco_validation_rollout_metrics = {}
    trainer._speco_validation_generation_elapsed_sec = 0.0
    trainer._speco_validation_response_tokens = 0

    with trainer._speco_rollout_metrics_fit_hook():
        trainer.async_rollout_manager.generate_sequences(
            SimpleNamespace(meta_info={"validate": True})
        )

    prefix = _speco_ray_trainer._SPECO_VALIDATION_SPEC_DECODE_PREFIX
    metrics = trainer._speco_validation_spec_decode_metrics()
    assert metrics[f"{prefix}/mean_acceptance_length"] == pytest.approx(2.4)
    assert metrics[f"{prefix}/verify_steps"] == 10.0
    assert metrics[f"{prefix}/response_tokens"] == 6.0
    assert not hasattr(trainer, "_speco_last_rollout_metrics")


def test_rollout_metrics_hook_collects_training_effective_throughput() -> None:
    class FakeRolloutManager:
        def generate_sequences(self, batch):
            del batch
            return SimpleNamespace(
                batch={"response_mask": torch.ones((2, 3), dtype=torch.long)},
                non_tensor_batch={
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": [4, 6],
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": [5, 9],
                },
                meta_info={"validate": False},
            )

    trainer = _trainer({}, step=7)
    trainer.async_rollout_manager = FakeRolloutManager()

    with trainer._speco_rollout_metrics_fit_hook():
        trainer.async_rollout_manager.generate_sequences(
            SimpleNamespace(meta_info={"validate": False})
        )

    quality = trainer._speco_current_step_rollout_quality_metrics()
    assert quality["mean_acceptance_length"] == pytest.approx(2.4)
    assert quality["verify_steps"] == 10.0
    assert quality["accepted_tokens"] == 14.0
    assert quality["response_tokens"] == 6.0
    assert quality["generation_time_s"] > 0.0
    assert quality["response_tokens_per_second"] > 0.0


def test_online_hook_records_training_rollout_quality() -> None:
    class FakeRolloutManager:
        def generate_sequences(self, batch):
            del batch
            return SimpleNamespace(
                batch={"response_mask": torch.ones((1, 4), dtype=torch.long)},
                non_tensor_batch={
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": [5],
                    f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": [10],
                },
                meta_info={"validate": False},
            )

    trainer = _trainer({}, step=7)
    trainer.async_rollout_manager = FakeRolloutManager()
    trainer._compute_old_log_prob = lambda batch: batch
    trainer._update_actor = lambda batch: batch
    trainer._speco_pending_candidate_update = None
    trainer._speco_pending_candidate_metrics = {}
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_collect_generation_samples = lambda output: 0

    with trainer._speco_online_fit_hooks():
        trainer.async_rollout_manager.generate_sequences(
            SimpleNamespace(meta_info={"validate": False})
        )

    quality = trainer._speco_current_step_rollout_quality_metrics()
    assert quality["verify_steps"] == 5.0
    assert quality["accepted_tokens"] == 10.0
    assert quality["response_tokens"] == 4.0
    assert quality["generation_time_s"] > 0.0
    assert quality["response_tokens_per_second"] > 0.0


def test_dual_best_uses_step_101_rollout_to_save_step_100_once(
    tmp_path,
) -> None:
    trainer = _trainer(
        {
            "dual_metric_best_checkpoint_enabled": True,
            "save_best_drafter_checkpoint": True,
            "best_checkpoint_min_acceptance_delta": 0.0,
            "best_checkpoint_min_throughput_relative_delta": 0.0,
            "dual_metric_best_checkpoint_include_optimizer": False,
            "publish_async": False,
        },
        step=100,
    )
    trainer._speco_pending_dual_best_observation = None
    trainer._speco_ensure_drafter_checkpoint_path = lambda: str(tmp_path)
    saved = []

    def save_checkpoint(**kwargs):
        saved.append(kwargs)
        checkpoint = tmp_path / f"draft_step_{kwargs['step']}"
        checkpoint.mkdir()
        checkpoint.joinpath("config.json").write_text("{}", encoding="utf-8")
        checkpoint.joinpath("pytorch_model.bin").write_bytes(b"weights")
        checkpoint.joinpath("metadata.json").write_text(
            json.dumps({"step": kwargs["step"], "complete": True}),
            encoding="utf-8",
        )

    trainer._speco_save_drafter_checkpoint = save_checkpoint
    armed = trainer._speco_arm_dual_best_observation({"drafter/published": 1})
    assert armed["drafter/dual_best/pending_update_step"] == 100

    trainer.global_steps = 101
    trainer._speco_last_rollout_metrics_step = 101
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 250.0,
    }
    trainer._speco_last_rollout_generation_elapsed_sec = 2.0
    trainer._speco_last_rollout_response_tokens = 1000

    metrics = trainer._speco_maybe_save_dual_best_from_rollout()

    assert saved == [{"wait": True, "step": 100, "include_optimizer": False}]
    assert metrics["drafter/dual_best/acceptance_saved"] == 1
    assert metrics["drafter/dual_best/throughput_saved"] == 1
    assert tmp_path.joinpath(
        "best_acceptance_checkpointed_iteration.txt"
    ).read_text(encoding="utf-8") == "100\n"
    assert tmp_path.joinpath(
        "best_throughput_checkpointed_iteration.txt"
    ).read_text(encoding="utf-8") == "100\n"
    acceptance = json.loads(
        tmp_path.joinpath("best_acceptance.json").read_text(encoding="utf-8")
    )
    throughput = json.loads(
        tmp_path.joinpath("best_throughput.json").read_text(encoding="utf-8")
    )
    assert acceptance["step"] == throughput["step"] == 100
    assert acceptance["actor_snapshot_step"] == 100
    assert acceptance["mean_acceptance_length"] == pytest.approx(3.5)
    assert throughput["response_tokens_per_second"] == pytest.approx(500.0)
    assert acceptance["checkpoint_includes_optimizer"] is False


def test_dual_best_updates_acceptance_and_throughput_independently(tmp_path) -> None:
    trainer = _trainer(
        {
            "dual_metric_best_checkpoint_enabled": True,
            "save_best_drafter_checkpoint": True,
            "best_checkpoint_min_acceptance_delta": 0.0,
            "best_checkpoint_min_throughput_relative_delta": 0.01,
        },
        step=11,
    )
    trainer._speco_ensure_drafter_checkpoint_path = lambda: str(tmp_path)
    tmp_path.joinpath("best_acceptance.json").write_text(
        '{"mean_acceptance_length":3.5}\n', encoding="utf-8"
    )
    tmp_path.joinpath("best_throughput.json").write_text(
        '{"response_tokens_per_second":500.0}\n', encoding="utf-8"
    )
    trainer._speco_pending_dual_best_observation = {"update_step": 10}
    trainer._speco_last_rollout_metrics_step = 11
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 240.0,
    }
    trainer._speco_last_rollout_generation_elapsed_sec = 1.0
    trainer._speco_last_rollout_response_tokens = 600

    def save_checkpoint(**kwargs):
        checkpoint = tmp_path / f"draft_step_{kwargs['step']}"
        checkpoint.mkdir()
        checkpoint.joinpath("config.json").write_text("{}", encoding="utf-8")
        checkpoint.joinpath("pytorch_model.bin").write_bytes(b"weights")
        checkpoint.joinpath("metadata.json").write_text(
            json.dumps({"step": kwargs["step"], "complete": True}),
            encoding="utf-8",
        )

    trainer._speco_save_drafter_checkpoint = save_checkpoint
    metrics = trainer._speco_maybe_save_dual_best_from_rollout()

    assert metrics["drafter/dual_best/acceptance_saved"] == 0
    assert metrics["drafter/dual_best/throughput_saved"] == 1
    assert json.loads(
        tmp_path.joinpath("best_acceptance.json").read_text(encoding="utf-8")
    )["mean_acceptance_length"] == 3.5
    throughput = json.loads(
        tmp_path.joinpath("best_throughput.json").read_text(encoding="utf-8")
    )
    assert throughput["step"] == 10
    assert throughput["response_tokens_per_second"] == pytest.approx(600.0)


def test_validation_diagnostics_dump_tokens_and_position_acceptance(
    tmp_path, monkeypatch
) -> None:
    diagnostics_dir = tmp_path / "diagnostics"
    monkeypatch.setenv(
        "SPECO_VALIDATION_DIAGNOSTICS_DIR", str(diagnostics_dir)
    )
    trainer = _trainer({}, step=0)
    trainer._speco_validation_rollout_metrics = {}
    trainer._speco_validation_generation_elapsed_sec = 0.0
    trainer._speco_validation_response_tokens = 0
    trainer._speco_validation_token_records = []
    trainer._speco_validation_position_attempts = []
    trainer._speco_validation_position_accepted = []
    output = SimpleNamespace(
        batch={
            "responses": torch.tensor([[10, 11, 0], [20, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        },
        non_tensor_batch={
            f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": [
                2,
                1,
            ],
            f"{_speco_ray_trainer.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": [
                3,
                3,
            ],
            "request_id": ["request-a", "request-b"],
            _speco_ray_trainer.SPECO_VLLM_SPECULATIVE_TRACE_KEY: [
                [
                    {"draft_count": 3, "accepted_count": 2},
                    {"draft_count": 3, "accepted_count": 1},
                ],
                [{"draft_count": 3, "accepted_count": 3}],
            ],
        },
    )

    trainer._speco_store_validation_rollout_metrics(output, 2.0)
    trainer._speco_write_validation_diagnostics()

    prefix = _speco_ray_trainer._SPECO_VALIDATION_SPEC_DECODE_PREFIX
    metrics = trainer._speco_validation_spec_decode_metrics()
    assert metrics[f"{prefix}/position_1_acceptance_rate"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/position_2_acceptance_rate"] == pytest.approx(2 / 3)
    assert metrics[f"{prefix}/position_3_acceptance_rate"] == pytest.approx(1 / 3)
    token_records = [
        json.loads(line)
        for line in (diagnostics_dir / "token_ids.step-000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["response_token_ids"] for record in token_records] == [
        [10, 11],
        [20],
    ]
    summary = json.loads(
        (diagnostics_dir / "spec_decode.step-000000.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["sample_count"] == 2
    assert summary["trace_event_count"] == 3
    assert summary["acceptance_by_position"][1] == {
        "position": 2,
        "attempted": 3,
        "accepted": 2,
        "acceptance_rate": pytest.approx(2 / 3),
    }


def test_post_update_acceptance_guard_records_then_aborts_on_a_half_length_drop(
    tmp_path, monkeypatch
) -> None:
    incident_log = tmp_path / "drafter_acceptance_incidents.jsonl"
    monkeypatch.setenv("SPECO_DRAFTER_ACCEPTANCE_INCIDENT_LOG", str(incident_log))
    monkeypatch.setenv("SPECO_ONLINE_STRATEGY_TAG", "p2_02")
    monkeypatch.setenv("SPECO_EXPERIMENT_ID", "dspark-guard-contract")
    monkeypatch.setenv("SPECO_RUN_MODE", "online")
    trainer = _trainer(
        {"post_update_mean_acceptance_length_drop_abort_threshold": 0.5},
        step=10,
    )
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"
    trainer._speco_last_rollout_metrics_step = 10
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 150.0,
    }

    armed = trainer._speco_arm_post_update_acceptance_guard(
        {"drafter/published": 1}
    )

    assert armed["drafter/post_update_acceptance_guard/armed"] == 1
    trainer.global_steps = 11
    trainer._speco_last_rollout_metrics_step = 11
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 100.0,
    }

    with pytest.raises(RuntimeError, match="regressed global mean acceptance length"):
        trainer._speco_check_post_update_acceptance_guard()

    records = [json.loads(line) for line in incident_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0] == {
        "schema_version": 1,
        "incident_id": records[0]["incident_id"],
        "event": "post_update_mean_acceptance_length_regression",
        "source": "runtime_guard",
        "recorded_at_utc": records[0]["recorded_at_utc"],
        "action": "aborted",
        "algorithm": "dspark",
        "strategy_version": "p2_02",
        "experiment_id": "dspark-guard-contract",
        "run_mode": "online",
        "wandb_run_id": "",
        "training_log": "",
        "update_step": 10,
        "observed_step": 11,
        "baseline_mean_acceptance_length": 2.5,
        "observed_mean_acceptance_length": 2.0,
        "drop": 0.5,
        "threshold": 0.5,
    }


def test_post_update_acceptance_guard_allows_a_smaller_drop() -> None:
    trainer = _trainer(
        {"post_update_mean_acceptance_length_drop_abort_threshold": 0.5},
        step=10,
    )
    trainer._speco_last_rollout_metrics_step = 10
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 150.0,
    }
    trainer._speco_arm_post_update_acceptance_guard({"drafter/published": 1})
    trainer.global_steps = 11
    trainer._speco_last_rollout_metrics_step = 11
    trainer._speco_last_rollout_metrics = {
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 100.0,
        _speco_ray_trainer._SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 101.0,
    }

    metrics = trainer._speco_check_post_update_acceptance_guard()

    assert metrics["drafter/post_update_acceptance_guard/checked"] == 1
    assert metrics["drafter/post_update_acceptance_guard/drop"] == pytest.approx(0.49)


def test_async_publish_sets_pending_ref_and_waits_before_next_publish() -> None:
    calls: list[tuple[str, object, int]] = []
    waited: list[object] = []
    trainer = _trainer({"publish_interval_steps": 1, "publish_async": True}, step=10)
    trainer._pending_drafter_publish_refs = ["old-ref"]
    trainer._ray_get_if_needed = lambda value: waited.append(value) or value
    trainer._speco_get_published_drafter_weights = lambda: {"weights": 1}
    trainer._speco_actor_rollout_method = lambda name: (
        lambda payload, global_steps=None: calls.append((name, payload, global_steps))
        or ["new-ref"]
    )

    metrics = trainer._speco_publish_drafter_weights(True)

    assert waited == [["old-ref"]]
    assert calls == [("update_draft_weights_async", {"weights": 1}, 10)]
    assert trainer._pending_drafter_publish_refs == ["new-ref"]
    assert metrics["drafter/publish_attempted"] == 1
    assert metrics["drafter/published"] == 1
    for key in (
        "timing_s/drafter_publish_wait_pending",
        "timing_s/drafter_publish_fetch_snapshot",
        "timing_s/drafter_publish_update_weights",
    ):
        assert metrics[key] >= 0


def test_disabled_or_untrained_drafter_does_not_publish() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=1)
    assert trainer._speco_publish_drafter_weights(False) == {
        "drafter/publish_attempted": 0,
        "drafter/published": 0,
    }


def test_candidate_precheck_rejects_unstable_metrics() -> None:
    trainer = _trainer(
        {
            "candidate_max_fixed_loss_delta": 0.0,
            "candidate_max_grad_norm": 10.0,
            "candidate_grad_norm_rolling_multiplier": 5.0,
            "candidate_max_rejected_k3_clamp_ratio": 0.5,
            "candidate_max_rejected_k3_median": 9.5,
        }
    )
    trainer._speco_candidate_grad_norm_history = [2.0, 2.5, 3.0]

    passed, reason, metrics = trainer._speco_candidate_precheck(
        {
            "drafter/fixed_cohort/delta_loss": -0.01,
            "drafter/train_grad_norm": 11.0,
        }
    )

    assert passed is False
    assert reason == "grad_norm"
    assert metrics["drafter/candidate/precheck_grad_norm"] == 11.0


def test_candidate_gate_selects_best_safe_interpolation() -> None:
    trainer = _trainer(
        {
            "candidate_gate_alphas": [0.0, 0.25, 0.5, 1.0],
            "candidate_gate_min_acceptance_delta": 0.0,
            "candidate_gate_max_ms_per_token_regression": 0.15,
        },
        step=11,
    )
    trainer._speco_pending_candidate_update = {"update_step": 10}
    trainer._speco_candidate_gate_batch = lambda args, kwargs: (args, kwargs)
    published = []
    trainer._speco_publish_candidate_alpha = (
        lambda step, alpha: published.append((step, alpha)) or 0.1
    )
    results = iter(
        [
            {"acceptance": 3.0, "ms_per_token": 1.0},
            {"acceptance": 3.1, "ms_per_token": 1.02},
            {"acceptance": 3.2, "ms_per_token": 1.05},
            {"acceptance": 3.3, "ms_per_token": 1.3},
        ]
    )
    trainer._speco_measure_candidate_rollout = (
        lambda generate, args, kwargs: (None, next(results))
    )

    selected, metrics = trainer._speco_select_candidate_with_rollout(
        lambda *args, **kwargs: None, (), {}
    )

    assert selected == 0.5
    assert metrics["drafter/candidate/selected_alpha"] == 0.5
    assert published == [
        (10, 0.0),
        (10, 0.25),
        (10, 0.5),
        (10, 1.0),
        (10, 0.5),
    ]


def test_drafter_checkpoint_results_require_a_successful_training_replica() -> None:
    SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
        [
            {"saved": True, "reason": "saved"},
            {"saved": False, "reason": "not_checkpoint_replica"},
            {"saved": False, "reason": "not_in_training_group"},
        ],
        require_saved=True,
    )

    with pytest.raises(RuntimeError, match="produced no saved state"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "not_checkpoint_replica"}],
            require_saved=True,
        )


def test_drafter_checkpoint_results_propagate_save_failure() -> None:
    with pytest.raises(RuntimeError, match="missing_checkpoint_dir"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "missing_checkpoint_dir"}],
            require_saved=True,
        )


def test_drafter_checkpoint_saves_before_actor_checkpoint(monkeypatch) -> None:
    trainer = _trainer({}, step=20)
    events = []
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: events.append(
        ("drafter", kwargs)
    )
    parent_cls = SpecoRayPPOTrainer.__mro__[1]
    monkeypatch.setattr(
        parent_cls,
        "_save_checkpoint",
        lambda self: events.append(("actor", {})) or "saved",
    )

    assert trainer._save_checkpoint() == "saved"
    assert events == [
        ("drafter", {"wait": True}),
        ("actor", {}),
    ]


def test_actor_checkpoint_failure_preserves_previous_drafter(monkeypatch) -> None:
    trainer = _trainer({}, step=20)
    events = []
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: events.append(
        ("drafter", kwargs)
    )
    parent_cls = SpecoRayPPOTrainer.__mro__[1]

    def fail_actor_checkpoint(self):
        del self
        events.append(("actor", {}))
        raise RuntimeError("actor save failed")

    monkeypatch.setattr(parent_cls, "_save_checkpoint", fail_actor_checkpoint)

    with pytest.raises(RuntimeError, match="actor save failed"):
        trainer._save_checkpoint()
    assert events == [
        ("drafter", {"wait": True}),
        ("actor", {}),
    ]


@pytest.mark.parametrize(("raises", "expected_exit_code"), [(False, 0), (True, 1)])
def test_tracking_hook_finishes_wandb_before_actor_teardown(
    monkeypatch, raises: bool, expected_exit_code: int
) -> None:
    from verl.utils.tracking import Tracking

    class WandbBackend:
        def __init__(self) -> None:
            self.finish_codes = []
            self.teardown_codes = []
            self.logged = []

        def log(self, *, data, step) -> None:
            self.logged.append((data, step))

        def finish(self, *, exit_code) -> None:
            self.finish_codes.append(exit_code)

        def teardown(self, *, exit_code) -> None:
            self.teardown_codes.append(exit_code)

    trainer = _trainer({}, step=1)
    trainer._speco_current_step_rollout_metrics = lambda: {}
    backend = WandbBackend()

    def initialize_tracking(tracking_self, *args, **kwargs) -> None:
        tracking_self.logger = {"wandb": backend}

    monkeypatch.setattr(Tracking, "__init__", initialize_tracking)
    tracking = None
    with pytest.raises(RuntimeError) if raises else nullcontext():
        with trainer._speco_tracking_metrics_hook():
            tracking = Tracking("project", "experiment")
            tracking.log(data={"training/global_step": 1}, step=1)
            if raises:
                raise RuntimeError("training failed")

    assert tracking is not None
    assert backend.logged == [({"training/global_step": 1}, 1)]
    assert backend.finish_codes == [expected_exit_code]
    assert backend.teardown_codes == [expected_exit_code]
    assert "wandb" not in tracking.logger
