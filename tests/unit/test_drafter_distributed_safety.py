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

import random
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
base_trainer_mod = pytest.importorskip("verl_speco.trainer.base_trainer")
DrafterBaseTrainer = base_trainer_mod.DrafterBaseTrainer
has_mappable_verify_event = base_trainer_mod._has_mappable_verify_event


def _metric_trainer(block_size: int = 3):
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"dspark_block_size": block_size})
        )
    )
    trainer.runtime_device = torch.device("cpu")
    trainer._training_metric_sums = {}
    trainer._training_metric_extrema = {}
    trainer._training_metric_steps = 0
    trainer._training_mesh_batch_skips = 0
    trainer._training_batch_prepare_failures = 0
    trainer.optimizer_steps_total = 0
    trainer.optimizer = None
    trainer.rank = 0
    trainer.rollout_dp_rank = 0
    trainer.dspark_dump_k3_logprobs = False
    return trainer


def test_block_drafter_diagnostics_use_one_fixed_shape_reduction():
    trainer = _metric_trainer(block_size=3)
    reduced_inputs = []

    def capture_reduction(value):
        reduced_inputs.append(value.clone())
        return value

    trainer._reduce_training_metric = capture_reduction
    scalar_keys = (
        "response_topk_enabled",
        "rejected_topk_enabled",
        "topk_union_enabled",
        "response_topk_loss_sum",
        "response_topk_weighted_count",
        "response_topk_student_mass_sum",
        "response_topk_teacher_mass_sum",
        "response_topk_support_size_sum",
        "rejected_topk_loss_sum",
        "rejected_topk_weighted_count",
        "rejected_topk_student_mass_sum",
        "rejected_topk_teacher_mass_sum",
        "rejected_topk_support_size_sum",
        "correct_count",
        "eval_token_count",
        "top1_correct_count",
        "top5_correct_count",
        "quality_token_count",
        "valid_token_count",
        "weighted_token_count",
        "ce_loss_sum",
        "ce_weighted_token_count",
        "first_rejection_ce_enabled",
        "first_rejection_ce_alpha",
        "first_rejection_ce_loss_sum",
        "first_rejection_ce_token_count",
        "first_rejection_ce_weight_sum",
        "first_rejection_ce_boosted_count",
        "first_rejection_ce_q_prob_sum",
        "rejected_aux_alpha",
        "rejected_prefix_length",
        "l1_loss_sum",
        "l1_weighted_token_count",
        "accepted_fkl_sum",
        "accepted_fkl_weighted_count",
        "accepted_k3_sum",
        "accepted_k3_weighted_count",
        "accepted_k3_clamped_count",
        "accepted_k3_linearized_count",
        "accepted_q_prob_sum",
        "accepted_p_prob_sum",
        "accepted_log_ratio_sum",
        "rejected_k3_sum",
        "rejected_k3_weighted_count",
        "rejected_q_prob_sum",
        "rejected_p_prob_sum",
        "rejected_teacher_minus_draft_log_ratio_sum",
        "rejected_k3_clamped_count",
        "rejected_k3_linearized_count",
        "asymmetric_k3_enabled",
        "accepted_asymmetric_k3_enabled",
        "rejected_asymmetric_k3_enabled",
        "asymmetric_k3_positive_threshold",
        "dynamic_accept_reject_weighting_enabled",
        "response_stream_effective_weight",
        "rejected_stream_effective_weight",
        "response_stream_weight_multiplier",
        "rejected_stream_weight_multiplier",
        "metric_replica_count",
        "response_objective_sum",
        "rejected_objective_sum",
        "objective_weighted_token_count",
        "trace_anchor_count",
        "trace_accepted_position_count",
        "trace_first_rejection_count",
        "trace_unverified_suffix_count",
        "trace_all_accepted_anchor_count",
        "trace_bonus_response_position_count",
        "sanitized_rows",
        "masked_rows",
        "sampled_vocab_size",
        "loss_mode_id",
        "absolute_anchor_position_sum",
        "absolute_anchor_position_count",
        "local_context_token_count_sum",
    )
    diagnostics = {
        key: torch.tensor(float(idx + 1)) for idx, key in enumerate(scalar_keys)
    }
    diagnostics.update(
        {
            "loss_sum_per_position": torch.tensor([1.0, 2.0, 3.0]),
            "correct_per_position": torch.tensor([4.0, 5.0, 6.0]),
            "count_per_position": torch.tensor([7.0, 8.0, 9.0]),
        }
    )

    trainer._record_dflash_training_metrics({"diagnostics": diagnostics})

    assert len(reduced_inputs) == 1
    assert reduced_inputs[0].shape == (len(scalar_keys) + 3 * 3 + 2 * 512 + 4,)
    assert trainer._training_metric_steps == 1
    assert trainer._training_metric_sums["dspark/correct_count"] == diagnostics["correct_count"].item()
    assert trainer._training_metric_sums["dspark/loss_sum_per_position/2"] == 3.0
    assert trainer._training_metric_sums["dspark/count_per_position/0"] == 7.0


def test_unique_first_sampling_avoids_reuse_until_pool_is_exhausted() -> None:
    trainer = _metric_trainer()
    trainer.config.rollout.drafter.training["train_unique_samples_first"] = True
    trainer._training_unique_sample_ids = {"sample-0", "sample-1"}
    items = [{"sample_id": f"sample-{index}"} for index in range(8)]

    selected = trainer._sample_training_items(items, 4, random.Random(7))

    assert len(selected) == 4
    assert {item["sample_id"] for item in selected}.isdisjoint(
        trainer._training_unique_sample_ids
    )


def test_first_rejection_ce_metrics_report_actual_auxiliary_weight_and_probability():
    trainer = _metric_trainer()
    trainer._reduce_training_metric = lambda value: value
    trainer._record_dflash_training_metrics({"diagnostics": {
        "metric_replica_count": torch.tensor(8.),
        "first_rejection_ce_enabled": torch.tensor(8.),
        "first_rejection_ce_alpha": torch.tensor(8. * .2),
        "first_rejection_ce_loss_sum": torch.tensor(60.),
        "first_rejection_ce_token_count": torch.tensor(20.),
        "first_rejection_ce_q_prob_sum": torch.tensor(5.),
    }})
    metrics = trainer.get_training_metrics()
    assert metrics["dspark/first_rejection_ce_enabled"] == 1.
    assert metrics["dspark/first_rejection_ce_alpha"] == pytest.approx(.2)
    assert metrics["dspark/first_rejection_ce_loss"] == 3.
    assert metrics["dspark/first_rejection_ce_contribution"] == pytest.approx(.6)
    assert metrics["dspark/first_rejection_ce_q_prob"] == .25


def test_k3_distribution_metrics_include_mean_extrema_and_median() -> None:
    trainer = _metric_trainer(block_size=3)
    trainer.config.rollout.drafter.training.update(
        {
            "dspark_k3_distribution_bins": 4,
            "dspark_rejected_loss_max_clamp": 8.0,
        }
    )
    trainer._reduce_training_metric = lambda value: value
    diagnostics = {
        "accepted_k3_sum": torch.tensor(6.0),
        "accepted_k3_weighted_count": torch.tensor(2.0),
        "accepted_k3_histogram": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "accepted_k3_min": torch.tensor(2.5),
        "accepted_k3_max": torch.tensor(7.5),
        "rejected_k3_sum": torch.tensor(15.0),
        "rejected_k3_weighted_count": torch.tensor(3.0),
        "rejected_k3_histogram": torch.tensor([0.0, 0.0, 3.0, 0.0]),
        "rejected_k3_min": torch.tensor(4.5),
        "rejected_k3_max": torch.tensor(5.5),
        "accepted_asymmetric_k3_enabled": torch.tensor(0.0),
        "rejected_asymmetric_k3_enabled": torch.tensor(1.0),
        "dynamic_accept_reject_weighting_enabled": torch.tensor(1.0),
        "response_stream_effective_weight": torch.tensor(0.75),
        "rejected_stream_effective_weight": torch.tensor(0.75),
        "response_stream_weight_multiplier": torch.tensor(0.75),
        "rejected_stream_weight_multiplier": torch.tensor(1.5),
        "metric_replica_count": torch.tensor(1.0),
    }

    trainer._record_dflash_training_metrics({"diagnostics": diagnostics})
    metrics = trainer.get_training_metrics()

    assert metrics["dspark/accepted_k3"] == pytest.approx(3.0)
    assert metrics["dspark/accepted_k3_mean"] == pytest.approx(3.0)
    assert metrics["dspark/accepted_k3_min"] == pytest.approx(2.5)
    assert metrics["dspark/accepted_k3_max"] == pytest.approx(7.5)
    assert metrics["dspark/accepted_k3_median"] == pytest.approx(2.0)
    assert metrics["dspark/rejected_k3_mean"] == pytest.approx(5.0)
    assert metrics["dspark/rejected_k3_min"] == pytest.approx(4.5)
    assert metrics["dspark/rejected_k3_max"] == pytest.approx(5.5)
    assert metrics["dspark/rejected_k3_median"] == pytest.approx(4.0)
    assert metrics["dspark/accepted_asymmetric_k3_enabled"] == pytest.approx(0.0)
    assert metrics["dspark/rejected_asymmetric_k3_enabled"] == pytest.approx(1.0)
    assert metrics["dspark/dynamic_accept_reject_weighting_enabled"] == pytest.approx(
        1.0
    )
    assert metrics["dspark/response_stream_effective_weight"] == pytest.approx(0.75)
    assert metrics["dspark/rejected_stream_effective_weight"] == pytest.approx(0.75)
    assert metrics["dspark/response_stream_weight_multiplier"] == pytest.approx(0.75)
    assert metrics["dspark/rejected_stream_weight_multiplier"] == pytest.approx(1.5)


def test_optional_k3_logprob_dump_is_atomic_and_reloadable(tmp_path) -> None:
    trainer = _metric_trainer()
    trainer.dspark_dump_k3_logprobs = True
    trainer.dspark_dump_k3_logprobs_dir = str(tmp_path)
    trainer.dspark_dump_k3_logprobs_interval_steps = 1
    trainer.dspark_dump_k3_logprobs_max_tokens_per_stream = 2
    trainer._k3_dump_files = 0
    trainer._k3_dump_tokens = 0
    trainer._k3_dump_failures = 0
    trainer._k3_dump_sequence = 0
    payload = {
        "token_ids": torch.tensor([11, 12, 13]),
        "prev_token_ids": torch.tensor([1, 11, 12]),
        "target_logprobs": torch.tensor([-0.1, -0.2, -0.3]),
        "draft_logprobs": torch.tensor([-0.4, -0.5, -0.6]),
        "k3": torch.tensor([0.1, 0.2, 0.3]),
        "weights": torch.ones(3),
    }
    loss_dict = {
        "diagnostics": {
            "k3_logprob_dump": {
                "accepted": payload,
                "rejected": payload,
            }
        }
    }

    trainer._maybe_dump_dspark_k3_logprobs(loss_dict, rl_step=5)

    output_files = list(tmp_path.rglob("*.pt"))
    assert len(output_files) == 1
    saved = torch.load(output_files[0], weights_only=False)
    assert saved["rl_step"] == 5
    assert saved["optimizer_step"] == 1
    assert saved["streams"]["accepted"]["token_ids"].tolist() == [11, 13]
    assert saved["streams"]["rejected"]["target_logprobs"].tolist() == pytest.approx(
        [-0.1, -0.3]
    )
    assert saved["writer_pid"] > 0
    assert trainer._k3_dump_files == 1
    assert trainer._k3_dump_tokens == 4
    assert "k3_logprob_dump" not in loss_dict["diagnostics"]


def test_candidate_update_interpolates_and_restores_local_training_state() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1.0
    )
    trainer.training_device_mesh = None
    trainer.training_process_group = None
    trainer.training_group_world_size = 1
    trainer.rank = 0
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"publish_dtype": None})
        )
    )
    trainer.training_steps = 4
    trainer.optimizer_steps_total = 4
    trainer._candidate_baseline_state_dict = None
    trainer._candidate_trained_state_dict = None
    trainer._candidate_local_baseline = None
    trainer._candidate_step = None
    trainer._candidate_training_steps = None
    trainer._candidate_optimizer_steps_total = None
    trainer._candidate_scheduler_state = None
    trainer._candidate_optimizer_lrs = None
    trainer._pending_publish_state_dict = None
    trainer._pending_publish_step = None
    trainer._pending_publish_ready = False
    baseline = trainer.model.weight.detach().clone()

    assert trainer.begin_candidate_update(10)
    with torch.no_grad():
        trainer.model.weight.add_(2.0)
    trainer.training_steps = 5
    trainer.optimizer_steps_total = 5
    assert trainer.prepare_candidate_update(10)

    ready, mixed = trainer.get_candidate_state_dict(10, 0.25)
    assert ready
    assert torch.allclose(mixed["weight"], baseline + 0.5)
    assert trainer.finalize_candidate_update(10, 0.25)
    assert torch.allclose(trainer.model.weight, baseline + 0.5)
    assert trainer.training_steps == 5
    assert trainer.optimizer_steps_total == 5
    assert trainer._candidate_step is None


def test_fixed_cohort_requires_a_verify_start_inside_the_feature_window() -> None:
    item = {
        "input_ids": torch.arange(8),
        "_verl_prompt_len": 6,
        "_verl_feature_start": 3,
        "speculative_trace": [
            {"response_start": 1, "draft_count": 3, "generation_index": 0}
        ],
    }

    assert has_mappable_verify_event(item) is True
    item["speculative_trace"][0]["generation_index"] = 1
    assert has_mappable_verify_event(item) is False
    item["speculative_trace"][0]["trajectory_response_start"] = 1
    assert has_mappable_verify_event(item) is True


def test_batch_readiness_reduces_over_sp_then_dp(monkeypatch):
    trainer = _metric_trainer()
    trainer.rank = 0
    sp_group = object()
    dp_group = object()
    trainer._get_sp_group = lambda: sp_group
    trainer._get_dp_group = lambda: dp_group
    trainer._get_sp_world_size = lambda: 4
    trainer._get_dp_world_size = lambda: 2
    calls = []

    def fake_all_reduce(value, *, op, group):
        calls.append(group)
        if group is sp_group:
            value.fill_(0)

    monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", fake_all_reduce)

    assert trainer._sync_batch_readiness(True) is False
    assert calls == [sp_group, dp_group]
    assert trainer._training_mesh_batch_skips == 1
