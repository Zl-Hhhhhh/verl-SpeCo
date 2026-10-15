from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from verl_speco.backends.dspark_trainer_backend import (
    DSparkTrainerBackend,
    _trace_anchor_rows_for_item,
)


def trace_item():
    return {
        "sample_id": "entropy-selection", "_verl_prompt_len": 2,
        "speculative_trace": [
            {"response_start": i * 3, "draft_count": 3, "accepted_count": 1,
             "draft_token_ids": [7, 8, 9],
             "draft_teacher_entropies": [0.5, 0.001 if i < 20 else 0.5, 0.0]}
            for i in range(80)
        ],
    }


def select(item, **kwargs):
    return _trace_anchor_rows_for_item(
        item, crop_start=0, crop_end=250, num_anchors=16, block_size=3, **kwargs
    )


def test_mixed_sampling_preserves_budget_and_increases_target_coverage():
    item = trace_item()
    stats = {}
    selected = select(item, low_entropy_fraction=0.5, selection_stats=stats)
    assert len(selected) == len(select(item)) == 16
    assert len({row["anchor"] for row in selected}) == 16
    assert stats["selected_low_entropy"] >= 8
    assert stats["selected_low_entropy"] > stats["uniform_low_entropy"]
    assert selected == select(deepcopy(item), low_entropy_fraction=0.5)
    assert select(item, low_entropy_fraction=0.0) == select(item)


@pytest.mark.parametrize("entropy", [None, float("nan"), float("inf"), -0.1, 0.01, 0.5])
def test_suffix_entropy_and_invalid_or_high_first_rejection_are_not_prioritized(entropy):
    item = trace_item()
    for event in item["speculative_trace"]:
        event["draft_teacher_entropies"] = [0.0, entropy, 0.0]
    stats = {}
    assert select(item, low_entropy_fraction=0.5, selection_stats=stats) == select(item)
    assert stats["eligible"] == 0
    assert stats["selected"] == 16


def test_cropped_out_first_rejection_is_not_eligible():
    item = trace_item()
    stats = {}
    _trace_anchor_rows_for_item(
        item, crop_start=0, crop_end=3, num_anchors=16, block_size=3,
        low_entropy_fraction=0.5, selection_stats=stats,
    )
    assert stats["rejections"] == stats["eligible"] == 0


def test_position_stratification_reserves_strict_mismatches_across_response():
    item = trace_item()
    for index, event in enumerate(item["speculative_trace"]):
        event["response_start"] = index * 100
        event["draft_teacher_entropies"] = [0.5, 0.001, 0.0]
        event["draft_teacher_logprobs"] = [-0.2, -5.0 if index % 2 == 0 else -2.0, -6.0]
    stats = {}
    selected = _trace_anchor_rows_for_item(
        item, crop_start=0, crop_end=8100, num_anchors=12, block_size=3,
        low_entropy_fraction=0.5, position_stratified_low_entropy=True,
        low_entropy_max_rejected_logprob=-4.0, selection_stats=stats,
    )
    assert len(selected) == 12
    assert stats["eligible"] == 40
    assert stats["reserved"] == 6
    assert [stats[key] for key in (
        "reserved_position_0_2047", "reserved_position_2048_4095",
        "reserved_position_4096_plus",
    )] == [2, 2, 2]
    assert len({row["anchor"] for row in selected}) == 12


def test_position_stratification_requires_first_rejection_logprob():
    item = trace_item()
    for event in item["speculative_trace"]:
        event["draft_teacher_logprobs"] = [-0.2, -5.0, -8.0]
    item["speculative_trace"][0]["draft_teacher_logprobs"] = [-0.2]
    stats = {}
    select(item, low_entropy_fraction=0.5,
           position_stratified_low_entropy=True, selection_stats=stats)
    assert stats["eligible"] == 19
    for event in item["speculative_trace"]:
        event["draft_teacher_logprobs"][1:2] = [-2.0]
    assert select(item, low_entropy_fraction=0.5,
                  position_stratified_low_entropy=True) == select(item)


def test_explicit_global_reservations_are_used_and_validated():
    item = trace_item()
    for event in item["speculative_trace"]:
        event["draft_teacher_entropies"] = [0.5, 0.001, 0.5]
        event["draft_teacher_logprobs"] = [-0.2, -5.0, -5.0]
    candidates = select(item, low_entropy_fraction=0.5,
                        position_stratified_low_entropy=True)
    forced = [candidates[0]["anchor"], candidates[1]["anchor"]]
    selected = select(item, low_entropy_fraction=0.5,
                      position_stratified_low_entropy=True,
                      forced_reserved_anchors=forced)
    assert set(forced) <= {row["anchor"] for row in selected}
    with pytest.raises(ValueError, match="ineligible"):
        select(item, low_entropy_fraction=0.5,
               position_stratified_low_entropy=True,
               forced_reserved_anchors=[-1])


def test_low_entropy_arm_can_apply_same_rejected_logprob_filter_without_position_balancing():
    item = trace_item()
    for index, event in enumerate(item["speculative_trace"]):
        event["draft_teacher_logprobs"] = [-0.2, -5.0 if index % 2 else -2.0, -8.0]
    stats = {}
    select(item, low_entropy_fraction=0.5, require_rejected_logprob=True,
           selection_stats=stats)
    assert stats["eligible"] == 10


def test_preprocessing_applies_policy_but_keeps_fixed_eval_uniform():
    class Config(dict):
        __getattr__ = dict.__getitem__

    cfg = Config(dspark_max_window=250, dspark_num_anchors=16, dspark_block_size=3,
                 dspark_low_entropy_anchor_fraction=0.5, dspark_low_entropy_threshold=0.01)
    backend = object.__new__(DSparkTrainerBackend)
    backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training=cfg)))
    model_config = SimpleNamespace(hidden_size=2, target_hidden_size=2, num_context_layers=1)
    item = trace_item()
    item.update(input_ids=torch.ones(250, dtype=torch.long),
                hidden_states=torch.zeros(250, 2), loss_mask=torch.ones(250))
    trained = backend.preprocess_individual_items([item], torch.device("cpu"), model_config)
    item["_speco_fixed_eval_holdout"] = True
    evaluated = backend.preprocess_individual_items([item], torch.device("cpu"), model_config)
    assert trained["trace_anchor_positions"][0].tolist() == [
        row["anchor"] for row in select(item, low_entropy_fraction=0.5)]
    assert trained["trace_first_rejection_teacher_entropies"][0].tolist() == pytest.approx([
        row["first_rejection_entropy"] for row in select(item, low_entropy_fraction=0.5)
    ])
    assert evaluated["trace_anchor_positions"][0].tolist() == [row["anchor"] for row in select(item)]
    item["_speco_fixed_eval_holdout"] = False
    item["low_entropy_sampling_active"] = False
    gated = backend.preprocess_individual_items([item], torch.device("cpu"), model_config)
    assert gated["trace_anchor_positions"][0].tolist() == evaluated["trace_anchor_positions"][0].tolist()
    cfg["dspark_certainty_share_threshold"] = 0.7
    missing_flag = dict(item)
    missing_flag.pop("low_entropy_sampling_active")
    with pytest.raises(ValueError, match="collection-time sampling flag"):
        backend.preprocess_individual_items([missing_flag], torch.device("cpu"), model_config)
    for event in item["speculative_trace"]:
        event.pop("draft_teacher_entropies")
    with pytest.raises(ValueError, match="requires draft_teacher_entropies"):
        backend.preprocess_individual_items([item], torch.device("cpu"), model_config)


@pytest.mark.parametrize("active", [False, True])
def test_worker_preserves_collection_time_gate_flag(active):
    from verl_speco.workers.speco_worker import SpecoWorker

    received = []
    worker = object.__new__(SpecoWorker)
    worker._store_rollout_sample = lambda **kwargs: received.append(kwargs["batch"]) or 1
    worker._flush_rollout_features_for_step = lambda: None
    sample = {key: torch.ones(1, 3) for key in
              ("input_ids", "prompts", "responses", "hidden_states")}
    sample.update(sample_id="gate-transport", low_entropy_sampling_active=active)
    result = SpecoWorker.collect_rollout_features(worker, [sample])
    assert result["stored_samples"] == 1
    assert received[0]["low_entropy_sampling_active"] is active
