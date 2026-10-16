import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.trainer.directional_compare_trainer import DirectionalCompareTrainer
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer
from verl_speco.workers.speco_worker import SpecoWorker


@pytest.mark.parametrize("world_size", [8, 16])
def test_diagnostic_manifest_uses_configured_replica_count(tmp_path, world_size):
    import json
    from verl_speco.trainer.directional_diagnostic_trainer import DirectionalDiagnosticTrainer

    for rank in range(0, world_size, 4):
        (tmp_path / f"candidates_rank{rank:02d}.jsonl").write_text(
            json.dumps({"sample_id": str(rank), "selected": True}) + "\n"
        )
        (tmp_path / f"paired_rank{rank:02d}_sample.pt").touch()
    source = tmp_path / "train.parquet"
    source.write_bytes(b"training-source")
    trainer = object.__new__(DirectionalDiagnosticTrainer)
    trainer.global_steps = 5
    trainer._directional_pair_seconds = 1.0
    trainer.config = OmegaConf.create({
        "trainer": {"n_gpus_per_node": world_size, "nnodes": 1},
        "data": {"seed": 17, "train_files": [str(source)]},
        "actor_rollout_ref": {
            "actor": {"ulysses_sequence_parallel_size": 4},
            "model": {"path": "actor"},
            "rollout": {"tensor_model_parallel_size": 4, "drafter": {
                "model_path": "draft", "training": {
                    "training_interval_steps": 5, "worker_world_size": world_size,
                    "dspark_directional_diagnostic_output": str(tmp_path),
                    "dspark_num_anchors": 32, "dspark_max_window": 512,
                    "max_collect_tokens_per_step_per_replica": 32768,
                },
            }},
        },
    })
    trainer._ablation_generation_cohorts = [["training"]]
    trainer._speco_entropy_lr = lambda: SimpleNamespace(prepare_validation_cohort=lambda:
        SimpleNamespace(non_tensor_batch={"raw_prompt": ["validation"]}))
    trained, _ = trainer._speco_train_drafter()
    manifest = json.loads((tmp_path / "paired_panel_manifest.json").read_text())
    assert not trained
    assert manifest["independent_data_replicas"] == world_size // 4
    assert manifest["selected_count"] == world_size // 4
    assert manifest["drafter_optimizer_steps"] == 0
    (tmp_path / "candidates_rank01.jsonl").write_text('{}\n')
    with pytest.raises(RuntimeError, match="SP-leader"):
        trainer._speco_train_drafter()


def test_directional_compare_records_prompts_and_calls_parent_generation(monkeypatch):
    trainer = object.__new__(DirectionalCompareTrainer)
    batch = SimpleNamespace(non_tensor_batch={"raw_prompt": [[{"role": "user", "content": "x"}]]})
    monkeypatch.setattr(SpecoRayPPOTrainer, "_get_gen_batch", lambda self, value: value)
    assert trainer._get_gen_batch(batch) is batch
    assert len(trainer._ablation_generation_cohorts) == 1
    assert '"content": "x"' in trainer._ablation_generation_cohorts[0][0]


def test_directional_compare_requires_verification_entropy_before_training(monkeypatch):
    trainer = object.__new__(DirectionalCompareTrainer)
    trainer._speco_drafter_training_config = lambda: {
        "candidate_gate_enabled": True, "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
    }
    monkeypatch.setattr(SpecoRayPPOTrainer, "fit", lambda self: "ready")
    monkeypatch.delenv("SPECO_VLLM_VERIFY_ENTROPY", raising=False)
    with pytest.raises(ValueError, match="verification entropy"):
        trainer.fit()
    monkeypatch.setenv("SPECO_VLLM_VERIFY_ENTROPY", "true")
    assert trainer.fit() == "ready"


def test_paired_probe_uses_identical_prefix_and_current_actor_head():
    trainer = object.__new__(DrafterBaseTrainer)
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    trainer.backend = SimpleNamespace(
        model_type="dspark", target_lm_head=SimpleNamespace(fc=head)
    )
    trainer._training_active = True
    trainer._target_lm_head_weight_step = 5
    trainer.current_rl_step = 5

    class FrozenDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.draft_model = SimpleNamespace(num_context_layers=2)

        def forward(self, **kwargs):
            assert kwargs["return_first_token_logits"]
            return torch.tensor([[[0.6, 0.4]]]).log()

    trainer.model = FrozenDraft()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "hidden_states": torch.zeros(1, 3, 4),
        "loss_mask": torch.ones(1, 3),
        "target_last_hidden_states": torch.tensor([[[0.4, 0.6]] * 3]).log(),
        "trace_anchor_positions": torch.tensor([[0]]),
        "trace_anchor_mask": torch.tensor([[True]]),
        "position_ids": torch.tensor([[10, 11, 12]]),
    }

    def prepare(*, items_override, track_sample_usage):
        assert items_override[0]["_speco_directional_probe"] is True
        assert track_sample_usage is False
        return batch

    trainer._prepare_training_batch = prepare
    item = {
        "input_ids": torch.tensor([1, 2, 3]),
        "hidden_positions": torch.tensor([10, 11, 12]),
        "_verl_feature_start": 0,
        "_verl_feature_end": 3,
    }
    old_sample = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "hidden_positions": torch.tensor([[10, 11, 12]]),
    }
    old_hidden = torch.tensor([[[0.6, 0.4]] * 3]).log()
    scored = trainer.score_directional_candidates(item, old_sample, old_hidden, head.weight)
    assert len(scored) == 1 and scored[0][0] == 0
    assert scored[0][1:] == pytest.approx((0.2, 0.2), abs=1e-6)
    detailed = trainer.score_directional_candidates(
        item, old_sample, old_hidden, head.weight,
        diagnostic=True, selected_anchors=[0],
    )
    assert detailed["rows"][0]["absolute_position"] == 10
    assert detailed["old"].shape == detailed["new"].shape == detailed["draft"].shape == (1, 2)
    assert torch.allclose(detailed["draft"], detailed["repeat_draft"])
    assert "_speco_directional_probe" not in item

    item["hidden_states_layout"] = "dflash_aux_plus_last"
    old_aux_plus_last = torch.cat([torch.zeros(1, 3, 4), old_hidden], dim=-1)
    detailed = trainer.score_directional_candidates(
        item, old_sample, old_aux_plus_last, head.weight,
        diagnostic=True, selected_anchors=[0],
    )
    assert detailed["old_feature_draft"] is not None
    assert torch.allclose(detailed["old_feature_draft"], detailed["draft"])

    old_sample["input_ids"][0, 0] = 9
    with pytest.raises(ValueError, match="different token prefixes"):
        trainer.score_directional_candidates(item, old_sample, old_hidden, head.weight)


def test_pair_worker_keeps_only_positive_m_without_backfill():
    worker = object.__new__(SpecoWorker)
    worker.in_drafter_train_group = True
    worker.last_global_step = 5
    worker.device_name = "cpu"
    worker.is_drafter_group_leader = True
    worker.training_process_group = None
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"dspark_num_anchors": 2})
        )
    )

    class FakeTrainer:
        collected_data = deque(maxlen=8)
        fixed_eval_data = []

        async def activate_training_model(self):
            return True

        def score_directional_candidates(self, item, old_batch, old_hidden, old_head):
            assert old_head.shape == (2, 2)
            return [(1, 0.3, 0.4), (2, 0.1, 0.3), (3, 0.0, 0.2)]

    worker.trainer = FakeTrainer()

    def store(batch, hidden, target_logprobs):
        worker.trainer.collected_data.append({"sample_id": batch["sample_id"], "step": 5})
        return 1

    worker._store_rollout_sample = store
    sample = {
        "sample_id": "a", "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "prompts": torch.tensor([[1]]), "responses": torch.tensor([[2, 3, 4]]),
        "hidden_positions": torch.tensor([[0, 1, 2]]),
        "hidden_states": torch.zeros(1, 3, 4),
    }
    old = [{**sample, "global_step": 1}]
    new = [{**sample, "global_step": 5}]
    result = asyncio.run(SpecoWorker.collect_directional_pair(
        worker, old, new, {"weight": torch.eye(2)}, 1, 5
    ))
    assert result["selected_samples"] == 1
    assert result["candidate_anchors"] == 3
    assert result["selected_anchors"] == 2
    assert worker.trainer.collected_data[0]["_speco_directional_selected_anchors"] == [1, 2]

    with pytest.raises(ValueError, match="wrong source step"):
        asyncio.run(SpecoWorker.collect_directional_pair(
            worker, old, new, {"weight": torch.eye(2)}, 2, 5
        ))


def test_pair_worker_preserves_equal_budget_for_matched_comparison():
    worker = object.__new__(SpecoWorker)
    worker.in_drafter_train_group = True
    worker.last_global_step = 5
    worker.device_name = "cpu"
    worker.is_drafter_group_leader = True
    worker.training_process_group = None
    worker.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(
        training={"dspark_num_anchors": 3,
                  "dspark_directional_compare_enabled": True,
                  "dspark_directional_compare_seed": 17}
    )))

    class FakeTrainer:
        collected_data = deque(maxlen=8)
        fixed_eval_data = []

        async def activate_training_model(self):
            return True

        def score_directional_candidates(self, *_):
            return [(1, 0.0, 0.8), (2, 0.2, 0.3),
                    (3, 0.1, 0.9), (4, 0.0, 0.2)]

    worker.trainer = FakeTrainer()

    def store(batch, hidden, target_logprobs):
        worker.trainer.collected_data.append({"sample_id": batch["sample_id"], "step": 5})
        return 1

    worker._store_rollout_sample = store
    sample = {
        "sample_id": "a", "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "prompts": torch.tensor([[1]]), "responses": torch.tensor([[2, 3, 4]]),
        "hidden_positions": torch.tensor([[0, 1, 2, 3]]),
        "hidden_states": torch.zeros(1, 4, 4),
    }
    result = asyncio.run(SpecoWorker.collect_directional_pair(
        worker, [{**sample, "global_step": 1}],
        [{**sample, "global_step": 5}], {"weight": torch.eye(2)}, 1, 5
    ))
    assert result["selected_anchors"] == 2
    assert worker._ablation_directional_selections["a"]["top_m"] == [2, 3]
    assert worker._ablation_directional_selections["a"]["top_tv"] == [1, 3]
    assert all(len(selected) == 2 for selected in
               worker._ablation_directional_selections["a"].values())


def test_pair_worker_cleans_up_when_no_directional_anchor_qualifies():
    worker = object.__new__(SpecoWorker)
    worker.in_drafter_train_group = True
    worker.last_global_step = 5
    worker.device_name = "cpu"
    worker.is_drafter_group_leader = True
    worker.training_process_group = None
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"dspark_num_anchors": 2})
        )
    )

    class FakeTrainer:
        def __init__(self):
            self.collected_data = deque(maxlen=8)
            self.fixed_eval_data = []
            self.cleaned = False

        async def activate_training_model(self):
            return True

        def score_directional_candidates(self, *args):
            return [(1, 0.0, 0.2)]

        async def cleanup_training(self, *, clear_data):
            self.cleaned = clear_data
            self.collected_data.clear()

    worker.trainer = FakeTrainer()

    def store(batch, hidden, target_logprobs):
        worker.trainer.collected_data.append({"sample_id": batch["sample_id"], "step": 5})
        return 1

    worker._store_rollout_sample = store
    sample = {
        "sample_id": "a", "input_ids": torch.tensor([[1, 2, 3]]),
        "prompts": torch.tensor([[1]]), "responses": torch.tensor([[2, 3]]),
        "hidden_positions": torch.tensor([[0, 1, 2]]),
        "hidden_states": torch.zeros(1, 3, 4),
    }
    result = asyncio.run(SpecoWorker.collect_directional_pair(
        worker, [{**sample, "global_step": 1}],
        [{**sample, "global_step": 5}], {"weight": torch.eye(2)}, 1, 5
    ))
    assert result["selected_samples"] == 0
    assert result["selected_anchors"] == 0
    assert worker.trainer.cleaned
    assert not worker.trainer.collected_data


def test_directional_panel_rolls_from_initial_actor_to_cycle_boundary_actor():
    trainer = object.__new__(SpecoRayPPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"drafter": {"training": {
            "training_interval_steps": 5,
        }}}},
        "trainer": {"total_training_steps": 15},
    })
    trainer.global_steps = 5
    actor0_samples = [[{
        "sample_id": "initial", "global_step": 1,
        "hidden_positions": torch.tensor([[0]]),
        "hidden_states": torch.zeros(1, 1, 2),
    }]]
    actor5_samples = [[{
        "sample_id": "boundary", "global_step": 5,
        "hidden_positions": torch.tensor([[0]]),
        "hidden_states": torch.zeros(1, 1, 2),
    }]]
    actor0_head = {"weight": torch.eye(2)}
    actor5_head = {"weight": torch.eye(2)}
    actor5_training_batch = ("step5-training-data", {"selected_count": 1})
    seen = []
    trainer._speco_directional_panel = {
        "data": ("step1-training-data", {"selected_count": 1}),
        "buckets": actor0_samples, "head": actor0_head,
        "source_step": 1, "source_actor_step": 0,
    }
    trainer._speco_directional_next_panel = actor5_training_batch
    trainer._speco_directional_capture_panel = lambda panel: (
        [[{"sample_id": "initial"}]]
        if panel[0] == "step1-training-data" else actor5_samples
    )
    trainer._speco_set_drafter_global_step = lambda: None
    trainer._speco_sync_target_lm_head_weight = lambda: {
        "drafter/target_lm_head_synced": 1,
    }
    trainer._speco_directional_actor_head = lambda: actor5_head
    trainer._ray_get_if_needed = lambda value: value

    def pair(old, new, head, source_step, actor_step):
        seen.append((old[0][0]["sample_id"], new[0][0]["sample_id"], head,
                     source_step, actor_step))
        return {"candidate_samples": 1, "selected_samples": 1,
                "selected_anchors": 1, "candidate_anchors": 1}

    trainer.speco_collect_directional_pair = pair
    selected, _ = trainer._speco_directional_finish_pair()
    assert selected == 1
    assert seen == [("initial", "initial", actor0_head, 1, 5)]
    trainer._speco_directional_stage_next_panel()
    assert trainer._speco_directional_panel["data"] is actor5_training_batch
    assert trainer._speco_directional_panel["buckets"] is actor5_samples
    assert trainer._speco_directional_panel["source_actor_step"] == 5
    assert trainer._speco_directional_panel["head"] is actor5_head

    trainer.global_steps = 10
    trainer._speco_directional_next_panel = None
    selected, _ = trainer._speco_directional_finish_pair()
    assert selected == 1
    assert seen[-1] == ("boundary", "boundary", actor5_head, 5, 10)


def test_dspark_probe_logits_and_selected_trace_anchors():
    config = DSparkConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        num_target_layers=4, num_context_layers=2, target_hidden_size=8,
        target_num_hidden_layers=4, target_layer_ids=[1, 3],
        mask_token_id=31, block_size=2, num_anchors=2,
    )
    model = DSparkTrainingModel(
        draft_model=DSparkDraftModel(config), block_size=2, num_anchors=2,
        trace_aligned_anchors=True,
    )
    output = model(
        input_ids=torch.arange(1, 9).unsqueeze(0),
        hidden_states_list=[torch.randn(1, 8, 8), torch.randn(1, 8, 8)],
        loss_mask=torch.ones(1, 8), lm_head_weight=torch.randn(32, 8),
        trace_anchor_positions=torch.tensor([[0, 3]]),
        trace_anchor_mask=torch.tensor([[True, True]]),
        return_first_token_logits=True,
    )
    assert output.shape == (1, 2, 32)
    assert torch.isfinite(output).all()

    backend = object.__new__(DSparkTrainerBackend)
    backend.config = OmegaConf.create({"rollout": {"drafter": {"training": {
        "dspark_max_window": 8, "dspark_block_size": 2,
        "dspark_num_anchors": 2, "dspark_directional_probe_anchors": 8,
    }}}})
    event = {
        "response_start": 0, "draft_count": 2, "accepted_count": 0,
        "draft_token_ids": [2, 3], "draft_teacher_logprobs": [-1.0, -2.0],
    }
    item = {
        "input_ids": torch.arange(1, 9),
        "hidden_states": torch.randn(8, 24),
        "hidden_states_layout": "dflash_aux_plus_last",
        "hidden_positions": torch.arange(8),
        "loss_mask": torch.ones(8),
        "_verl_prompt_len": 1, "_verl_feature_start": 0,
        "sample_id": "paired", "speculative_trace": [event, {**event, "response_start": 3}],
        "_speco_directional_selected_anchors": [3],
    }
    processed = backend.preprocess_individual_items([item], torch.device("cpu"), config)
    assert processed["trace_anchor_positions"][0].tolist() == [3]
    item["_speco_directional_selected_anchors"] = []
    processed = backend.preprocess_individual_items([item], torch.device("cpu"), config)
    assert processed["trace_anchor_positions"][0].numel() == 0
