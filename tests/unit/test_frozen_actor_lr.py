"""CPU checks for matched LR arms and preserving updates between probes."""
import asyncio
from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.backends.dspark_trainer_backend import _trace_anchor_rows_for_item
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.frozen_actor_lr_trainer import FrozenActorLRTrainer, _SkipInitialBatches
from verl_speco.workers.frozen_actor_lr_worker import (
    FrozenActorLRWorker, fingerprint, global_position_reservations, nested_collection_pools,
)


def make_worker(tmp_path):
    worker = object.__new__(FrozenActorLRWorker)
    worker._rank = 0
    cfg = OmegaConf.create({"rollout": {"drafter": {"training": {
        "entropy_lr_enabled": True, "frozen_actor_lr_output": str(tmp_path)}}}})
    t = object.__new__(DrafterBaseTrainer)
    t.model = torch.nn.Linear(1, 1, bias=False)
    t.model.weight.data.fill_(1.0)
    t.optimizer = torch.optim.AdamW(t.model.parameters(), lr=1e-5)
    t.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda _: 1.0)
    t.config, t.rank, t.training_device_mesh = cfg, 0, None
    t.training_steps = t.optimizer_steps_total = t._current_pad_size = 0
    t.collected_data, t.current_rl_step, t.use_data_buffer = [], 1, False
    t.runtime_device, t.device_module = "cpu", torch
    t.get_model_state_dict = lambda: {k: v.detach().clone() for k, v in t.model.state_dict().items()}
    t.clear_pending_publish_state_dict = lambda: None
    t.prepare_training_batch = lambda: {"x": torch.tensor([[float(t.training_steps + 1)]])}
    t.reset_training_metrics = lambda: None
    t.get_training_metrics = lambda: {}

    async def activate():
        return True

    async def cleanup(clear_data=False):
        t.training_steps = 0

    async def train(batch, step):
        assert t._current_pad_size == batch.get("_speco_pad_size", 0)
        loss = t.model(batch["x"]).square().sum()
        loss.backward()
        t.optimizer.step()
        t.optimizer.zero_grad(set_to_none=True)
        t.lr_scheduler.step()
        t.optimizer_steps_total += 1
        t.training_steps += 1
        return True

    t.activate_training_model, t.cleanup_training = activate, cleanup
    t.release_training_memory_after_activation = cleanup
    t.training_step_from_batch = train
    worker.trainer, worker.config = t, cfg
    worker.device_name, worker.in_drafter_train_group, worker.last_global_step = "cpu", True, 1
    return worker


def test_resume_skips_only_the_first_two_training_batches():
    batches = _SkipInitialBatches(["warmup", "round1", "round2", "round3"], 2)
    assert len(batches) == 2
    assert list(batches) == ["round2", "round3"]
    with pytest.raises(ValueError, match="Invalid training batch skip"):
        _SkipInitialBatches(["only"], 2)


def test_arm_restoration_and_cumulative_twenty_steps(tmp_path):
    worker = make_worker(tmp_path)

    async def run():
        prepared = await worker.ablation_prepare()
        assert len(set(prepared["batch_hashes"])) == 20
        assert prepared["sample_draws"] is None or prepared["sample_draws"] >= 0
        assert worker._ablation_batches is None
        assert len(worker._ablation_batch_files) == 20
        outcomes = []
        for _ in range(2):
            await worker.ablation_train(1e-5, 5, start=True)
            assert worker.trainer.optimizer_steps_total == 5
            moment_at_five = fingerprint(worker.trainer.optimizer.state_dict())
            await worker.ablation_train(1e-5, 20)
            assert worker.trainer.optimizer_steps_total == 20
            assert fingerprint(worker.trainer.optimizer.state_dict()) != moment_at_five
            outcomes.append(fingerprint(worker.trainer.model.state_dict()))
            assert worker.trainer.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_check_restored())["ok"]
        assert outcomes[0] == outcomes[1]
        worker.trainer.model.weight.data.add_(1)
        with pytest.raises(RuntimeError, match="Baseline restoration mismatch"):
            await worker.ablation_train(1e-5, 5, start=True)

    asyncio.run(run())


def test_arm_restoration_from_resumed_optimizer_step(tmp_path):
    worker = make_worker(tmp_path)
    worker.trainer.optimizer_steps_total = 50

    async def run():
        await worker.ablation_prepare()
        await worker.ablation_train(1e-5, 5, start=True)
        assert worker.trainer.optimizer_steps_total == 55
        await worker.ablation_train(1e-5, 10)
        assert worker.trainer.optimizer_steps_total == 60
        assert worker.trainer.finalize_candidate_update(1, 0.0)
        assert (await worker.ablation_check_restored())["ok"]
        assert worker.trainer.optimizer_steps_total == 50

    asyncio.run(run())


def test_multiround_branches_keep_independent_optimizer_state(tmp_path):
    worker = make_worker(tmp_path)
    worker.trainer.fixed_eval_data = []

    async def run():
        await worker.ablation_prepare()
        worker.trainer.current_rl_step = 2
        source = (await worker.ablation_multiround_init())["source"]
        first_round = {}
        for arm, lr in (("uniform", 1e-5), ("global_position_low_entropy", 2e-5)):
            loaded = await worker.ablation_multiround_load_branch(arm)
            assert loaded["state"] == source
            await worker.ablation_train(lr, 5, start=True, save_artifact=False)
            first_round[arm] = (await worker.ablation_multiround_store_branch(arm))["state"]
            assert worker.trainer.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_multiround_restore_source())["state"] == source
        assert first_round["uniform"] != first_round["global_position_low_entropy"]
        assert not list(tmp_path.glob("drafter_*.pt"))

        worker.trainer.current_rl_step = 3
        second_round = {}
        for arm, lr in (("global_position_low_entropy", 2e-5), ("uniform", 1e-5)):
            loaded = await worker.ablation_multiround_load_branch(arm)
            assert loaded["state"] == first_round[arm]
            assert worker.trainer.optimizer_steps_total == 5
            await worker.ablation_train(lr, 5, start=True, save_artifact=False)
            state = (await worker.ablation_multiround_store_branch(arm))["state"]
            second_round[arm] = state
            assert state["optimizer_steps"] == 10
            assert state["optimizer"] != first_round[arm]["optimizer"]
            assert worker.trainer.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_multiround_restore_source())["state"] == source
        worker.trainer.model.weight.data.add_(0.25)
        reset = await worker.ablation_multiround_restore_source()
        assert reset["drift"] == ["weights"]
        assert reset["state"] == source

        with (tmp_path / "results.jsonl").open("w") as stream:
            stream.write(json.dumps(dict(event="initialized", workers=[dict(rank=0, source=source)])) + "\n")
            for arm, state in second_round.items():
                stream.write(json.dumps(dict(event="trained", round=2, arm=arm,
                                             branch_states=[dict(rank=0, state=state)])) + "\n")
        resumed_root = tmp_path / "resume"
        resumed_root.mkdir()
        resumed = make_worker(resumed_root)
        resumed.config.rollout.drafter.training.frozen_actor_multiround_resume_state_dir = str(tmp_path)
        await resumed.ablation_prepare()
        assert (await resumed.ablation_multiround_init())["source"] == source
        resumed.trainer.current_rl_step = 4
        for arm, lr in (("uniform", 1e-5), ("global_position_low_entropy", 2e-5)):
            assert (await resumed.ablation_multiround_load_branch(arm))["state"] == second_round[arm]
            await resumed.ablation_train(lr, 5, start=True, save_artifact=False)
            assert (await resumed.ablation_multiround_store_branch(arm))["state"]["optimizer_steps"] == 15
            assert resumed.trainer.finalize_candidate_update(1, 0.0)
            assert (await resumed.ablation_multiround_restore_source())["state"] == source
        assert resumed.ablation_multiround_cleanup_states()["removed"] == 2
        assert len(list(tmp_path.glob("state_rank_*.pt"))) == 3
        assert worker.ablation_multiround_cleanup_states()["removed"] == 3
        assert not list(tmp_path.glob("state_rank_*.pt"))

    asyncio.run(run())


def test_nested_collection_pools_enforce_token_budget_and_subset():
    items = [dict(sample_id=str(index), hidden_states=torch.zeros(rows, 2))
             for index, rows in enumerate((60, 80, 100, 120, 140, 160))]
    pools = nested_collection_pools(items, [256, 1024], seed=17)
    low = {item["sample_id"] for item in pools[256]}
    high = {item["sample_id"] for item in pools[1024]}
    assert low < high
    assert sum(item["hidden_states"].shape[0] for item in pools[256]) <= 256
    assert high == {item["sample_id"] for item in items}
    assert nested_collection_pools(list(reversed(items)), [256, 1024], seed=17)[256] == pools[256]
    with pytest.raises(ValueError, match="exceeds"):
        nested_collection_pools(items, [256, 512], seed=17)
    with pytest.raises(ValueError, match="Duplicate"):
        nested_collection_pools(items + items[:1], [256, 1024], seed=17)


def test_global_position_reservations_balance_across_collection_windows():
    items = []
    for bucket, response_start in enumerate((0, 2500, 5000)):
        events = [dict(response_start=response_start + index * 3,
                       draft_count=1, accepted_count=0, draft_token_ids=[7],
                       draft_teacher_entropies=[0.001],
                       draft_teacher_logprobs=[-5.0]) for index in range(12)]
        items.append(dict(sample_id=f"bucket-{bucket}", _verl_prompt_len=2,
                          _verl_feature_start=response_start,
                          input_ids=torch.ones(512, dtype=torch.long),
                          hidden_states=torch.zeros(512, 2),
                          loss_mask=torch.ones(512), speculative_trace=events))
    plan, coverage = global_position_reservations(
        items, max_window=512, num_anchors=32, block_size=7)
    assert coverage["by_position"] == [6, 6, 6]
    assert coverage["selected"] == coverage["budget"] == 18
    assert all(len(plan[item["sample_id"]]) == 6 for item in items)
    with pytest.raises(ValueError, match="Duplicate"):
        global_position_reservations(items + items[:1], max_window=512,
                                     num_anchors=32, block_size=7)


def test_global_position_reservations_exclude_rejection_past_window():
    item = dict(
        sample_id="boundary", _verl_prompt_len=2, _verl_feature_start=0,
        input_ids=torch.ones(512, dtype=torch.long),
        hidden_states=torch.zeros(512, 2), loss_mask=torch.ones(512),
        speculative_trace=[
            dict(response_start=0, draft_count=1, accepted_count=1,
                 draft_token_ids=[7], draft_teacher_entropies=[0.001],
                 draft_teacher_logprobs=[-5.0]),
            dict(response_start=508, draft_count=7, accepted_count=3,
                 draft_token_ids=[7] * 7, draft_teacher_entropies=[0.001] * 7,
                 draft_teacher_logprobs=[-5.0] * 7),
        ],
    )
    plan, coverage = global_position_reservations(
        [item], max_window=512, num_anchors=32, block_size=7)
    assert plan["boundary"] == []
    assert coverage["selected"] == 0
    assert coverage["eligible_by_position"] == [0, 0, 0]
    _trace_anchor_rows_for_item(
        item, crop_start=0, crop_end=512, num_anchors=32, block_size=7,
        low_entropy_fraction=0.5, position_stratified_low_entropy=True,
        forced_reserved_anchors=plan["boundary"],
    )


def test_nested_worker_reuses_one_rollout_and_restores_between_caps(tmp_path):
    worker = make_worker(tmp_path)
    t = worker.trainer
    cfg = worker.config.rollout.drafter.training
    cfg.frozen_actor_nested_caps_enabled = True
    cfg.frozen_actor_nested_caps = [512, 2048]
    cfg.frozen_actor_nested_seed = 17
    t.batch_size = 4
    original = [dict(sample_id=str(index), step=1, hidden_states=torch.zeros(128, 2))
                for index in range(16)]
    t.collected_data = original

    def reset():
        t._training_sample_draws = 0
        t._training_unique_sample_ids = set()

    def prepare():
        chosen = [t.collected_data[(t.training_steps * 4 + offset) % len(t.collected_data)]
                  for offset in range(4)]
        t._training_sample_draws += len(chosen)
        t._training_unique_sample_ids.update(item["sample_id"] for item in chosen)
        value = sum(int(item["sample_id"]) for item in chosen) + t.training_steps
        return {"x": torch.tensor([[float(value + 1)]])}

    t.reset_training_metrics = reset
    t.prepare_training_batch = prepare

    async def run():
        result = await worker.ablation_prepare()
        assert t.collected_data is original
        assert len(result["caps"][512]["sample_ids"]) == 4
        assert len(result["caps"][2048]["sample_ids"]) == 16
        assert result["caps"][512]["batch_hashes"] != result["caps"][2048]["batch_hashes"]
        for cap in (512, 2048):
            await worker.ablation_train(1e-5, 3, start=True, nested_cap=cap,
                                        artifact_tag=f"cap_{cap}")
            await worker.ablation_train(1e-5, 5, nested_cap=cap,
                                        artifact_tag=f"cap_{cap}")
            assert t.optimizer_steps_total == 5
            assert t.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_check_restored())["ok"]
            assert t.optimizer_steps_total == 0
    asyncio.run(run())


def test_sampling_worker_builds_matched_batches_and_restores_config(tmp_path):
    worker = make_worker(tmp_path)
    t = worker.trainer
    cfg = worker.config.rollout.drafter.training
    cfg.frozen_actor_sampling_compare_enabled = True
    cfg.dspark_max_window = 512
    cfg.dspark_num_anchors = 32
    cfg.dspark_block_size = 7
    cfg.dspark_low_entropy_threshold = 0.01
    cfg.dspark_low_entropy_max_rejected_logprob = -4.0
    cfg.dspark_low_entropy_anchor_fraction = 0.0
    cfg.dspark_position_entropy_stratified_sampling_enabled = False
    t.batch_size = 4
    source = []
    for index in range(8):
        source.append(dict(sample_id=f"sample-{index}", step=1,
                           _verl_prompt_len=2, _verl_feature_start=0,
                           input_ids=torch.ones(512, dtype=torch.long),
                           hidden_states=torch.zeros(512, 2), loss_mask=torch.ones(512),
                           speculative_trace=[dict(response_start=3 * offset,
                                                   draft_count=1, accepted_count=0,
                                                   draft_token_ids=[7],
                                                   draft_teacher_entropies=[0.001],
                                                   draft_teacher_logprobs=[-5.0])
                                              for offset in range(12)]))
    t.collected_data = source

    def reset():
        t._training_sample_draws = 0
        t._training_unique_sample_ids = set()

    def prepare():
        chosen = [t.collected_data[(t.training_steps * 4 + offset) % len(t.collected_data)]
                  for offset in range(4)]
        t._training_sample_draws += 4
        t._training_unique_sample_ids.update(item["sample_id"] for item in chosen)
        reserved = sum(len(item.get("_speco_global_reserved_trace_anchors", [])) for item in chosen)
        strict = sum(bool(item.get("_speco_sampling_require_strict_rejection")) for item in chosen)
        return {"x": torch.tensor([[float(1 + reserved + strict)]])}

    t.reset_training_metrics = reset
    t.prepare_training_batch = prepare

    async def run():
        result = await worker.ablation_prepare()
        assert t.collected_data is source
        assert cfg.dspark_low_entropy_anchor_fraction == 0.0
        assert not cfg.dspark_position_entropy_stratified_sampling_enabled
        assert result["global_plan"]["selected"] > 0
        sample_ids = [arm["sample_ids"] for arm in result["arms"].values()]
        assert sample_ids[0] == sample_ids[1] == sample_ids[2]
        for arm in ("uniform", "low_entropy", "global_position_low_entropy"):
            await worker.ablation_train(1e-5, 5, start=True,
                                        sampling_arm=arm, artifact_tag=arm)
            assert t.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_check_restored())["ok"]
    asyncio.run(run())


@pytest.mark.parametrize("overlap", [False, True])
def test_sampling_cohort_provenance_survives_validation_preparation(tmp_path, overlap):
    import hashlib
    import json
    from verl import DataProto

    coordinator = object.__new__(FrozenActorLRTrainer)
    coordinator.config = OmegaConf.create({
        "data": {"seed": 17},
        "actor_rollout_ref": {
            "model": {"path": "actor100"},
            "rollout": {"drafter": {
                "model_path": "initial-drafter",
                "training": {
                    "frozen_actor_lr_output": str(tmp_path / "sampling"),
                    "max_collect_tokens_per_step_per_replica": 16384,
                    "dspark_first_rejection_ce_alpha": 0.1,
                    "dspark_max_window": 2048,
                    "lr": 1e-5,
                },
            }},
        },
    })
    coordinator._speco_drafter_training_config = (
        lambda: coordinator.config.actor_rollout_ref.rollout.drafter.training
    )
    training = json.dumps("held out" if overlap else "training prompt")
    coordinator._ablation_generation_cohorts = [[training]]
    cohort = DataProto.from_dict(non_tensors={
        "raw_prompt": np.array(["held out"], dtype=object),
    })

    def prepare_validation_cohort():
        coordinator._ablation_generation_cohorts.append([json.dumps("held out")])
        return cohort

    coordinator._speco_entropy_lr = lambda: SimpleNamespace(
        prepare_validation_cohort=prepare_validation_cohort
    )

    def stop_after_provenance_check():
        raise RuntimeError("stop after provenance check")

    coordinator._require_speco_worker_group = stop_after_provenance_check
    with pytest.raises(RuntimeError, match=(
        "prompts overlap" if overlap else "stop after provenance check"
    )):
        coordinator._speco_train_drafter_sampling_compare()
    result = tmp_path / "sampling" / "results.jsonl"
    if overlap:
        assert not result.exists()
    else:
        design = json.loads(result.read_text().splitlines()[0])
        assert design["training_prompt_sha256"] == [
            hashlib.sha256(training.encode()).hexdigest()
        ]
        assert design["training_validation_overlap"] == 0


def test_actor_update_does_not_call_actor_worker():
    trainer = object.__new__(FrozenActorLRTrainer)
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda *a: pytest.fail("Frozen actor updated"))
    result = trainer._update_actor(None)
    assert result.meta_info["metrics"]["actor/frozen_for_lr_ablation"] == 1


def test_frozen_drafter_comparison_only_probes(tmp_path):
    import json
    import numpy as np
    from verl import DataProto

    trainer = object.__new__(FrozenActorLRTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {
        "model": {"path": "actor100"},
        "rollout": {"drafter": {
            "model_path": "saved-drafter",
            "training": {"frozen_drafter_compare_output": str(tmp_path)},
        }},
    }})
    trainer._speco_drafter_training_config = lambda: trainer.config.actor_rollout_ref.rollout.drafter.training
    cohort = DataProto.from_dict(non_tensors={"raw_prompt": np.array(["a", "b"], dtype=object)})
    trainer._speco_entropy_lr = lambda: SimpleNamespace(prepare_validation_cohort=lambda: cohort)
    calls = []
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: pytest.fail("Frozen comparison republished actor weights"),
        wake_up_replicas=lambda: calls.append(("wake", None)),
    )
    trainer._ablation_probe = lambda label, performance=False: calls.append((label, performance))

    trained, metrics = trainer.fit()

    assert not trained and metrics == {"drafter/frozen_compare_complete": 1}
    assert calls == [("wake", None), ("warmup_discard", False),
                     ("quality_0", False), ("quality_1", False),
                     ("load320_0", True), ("load320_1", True)]
    records = [json.loads(row) for row in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert records[0]["actor"] == "actor100"
    assert records[0]["drafter"] == "saved-drafter"
    assert len(records[0]["validation_prompt_sha256"]) == 2
    assert records[-1]["event"] == "completed"


def test_full_frozen_eval_validates_all_problems_without_publishing(tmp_path):
    import json

    trainer = object.__new__(FrozenActorLRTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {
        "model": {"path": "actor100"},
        "rollout": {"drafter": {
            "model_path": "saved-drafter",
            "training": {"frozen_full_eval_output": str(tmp_path),
                         "frozen_full_eval_expected_count": 500},
        }},
    }})
    trainer._speco_drafter_training_config = lambda: trainer.config.actor_rollout_ref.rollout.drafter.training
    trainer.val_dataset = range(500)
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: pytest.fail("Frozen evaluation republished actor weights"),
        wake_up_replicas=lambda: None,
    )
    calls = []
    @contextmanager
    def metrics_hook():
        calls.append("hook_enter")
        yield
        calls.append("hook_exit")
    trainer._speco_rollout_metrics_fit_hook = metrics_hook
    trainer._validate = lambda: calls.append("validate") or {
        "val-aux/drafter/spec_decode/mean_acceptance_length": np.float32(2.0),
        "val-aux/drafter/spec_decode/verify_steps": np.int32(30),
        "val-aux/drafter/spec_decode/response_tokens_per_second": np.float32(42.0),
    }
    trainer._shutdown_dump_executor = lambda: calls.append("drain")

    trained, metrics = trainer.fit()

    assert not trained and metrics == {"drafter/frozen_full_eval_complete": 1}
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert result["problem_count"] == 500
    assert result["actor_updates"] == result["drafter_updates"] == 0
    assert result["actor"] == "actor100" and result["drafter"] == "saved-drafter"
    assert result["metrics"]["val-aux/drafter/spec_decode/verify_steps"] == 30
    assert calls == ["hook_enter", "validate", "hook_exit", "drain"]
    with pytest.raises(FileExistsError):
        trainer.fit()
    (tmp_path / "metrics.json").unlink()
    trainer._validate = lambda: {"val-core/math_dapo/acc/mean@1": 0.5}
    with pytest.raises(RuntimeError, match="missing speculative decoding metrics"):
        trainer.fit()


def test_full_frozen_eval_rejects_incomplete_dataset(tmp_path):
    trainer = object.__new__(FrozenActorLRTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {
        "rollout": {"drafter": {"training": {
            "frozen_full_eval_output": str(tmp_path),
            "frozen_full_eval_expected_count": 500,
        }}},
    }})
    trainer._speco_drafter_training_config = lambda: trainer.config.actor_rollout_ref.rollout.drafter.training
    trainer.val_dataset = range(499)
    trainer.checkpoint_manager = SimpleNamespace(
        wake_up_replicas=lambda: pytest.fail("Incomplete dataset reached rollout"),
    )
    with pytest.raises(RuntimeError, match="expected 500 problems"):
        trainer.fit()


def test_target_only_repeats_without_actor_updates(tmp_path):
    import json

    trainer = object.__new__(FrozenActorLRTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"model": {"path": "actor100"},
                              "rollout": {"drafter": {"enable": False, "training": {
                                  "frozen_target_repeat_output": str(tmp_path / "repeat")}}}},
        "trainer": {"validation_data_dir": None},
    })
    trainer._speco_drafter_training_config = lambda: trainer.config.actor_rollout_ref.rollout.drafter.training
    trainer.val_dataset = range(64)
    calls = []
    trainer.checkpoint_manager = SimpleNamespace(
        wake_up_replicas=lambda: calls.append("wake"),
        update_weights=lambda step: pytest.fail("Target-only repeat republished actor weights"),
    )

    def validate():
        path = tmp_path / "repeat" / "generations"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{trainer.global_steps}.jsonl").write_text("{}\n" * 64)
        calls.append(f"validate-{trainer.global_steps}")
        return {"accuracy": 0.5}

    trainer._validate = validate
    trainer._shutdown_dump_executor = lambda: calls.append("drain")
    trainer._init_dump_executor = lambda: calls.append("init")
    completed, metrics = trainer.fit()
    assert not completed and metrics == {"drafter/frozen_target_repeat_complete": 1}
    assert calls == ["wake", "validate-0", "drain", "init", "validate-1", "drain"]
    result = json.loads((tmp_path / "repeat" / "metrics.json").read_text())
    assert result["actor_updates"] == result["drafter_updates"] == 0
    assert len(result["repeats"]) == 2


def test_worker_artifacts_use_global_not_group_local_rank(tmp_path):
    worker = make_worker(tmp_path)
    worker._rank = 4
    worker.trainer.rank = 0
    result = asyncio.run(worker.ablation_prepare())
    assert result["rank"] == 4 and result["local_training_rank"] == 0
    assert (tmp_path / "batches_rank_04" / "batch_00.pt").exists()
    assert (tmp_path / "worker_04.jsonl").exists()
    assert not (tmp_path / "batches_rank_00").exists()
    # Only global rank 0 owns the full export, even when local SP rank is 0.
    worker.trainer.get_model_state_dict = lambda: {}
    asyncio.run(worker.ablation_train(1e-5, 5, start=True))
    assert not list(tmp_path.glob("drafter_lr_*.pt"))
    assert worker.trainer.finalize_candidate_update(1, 0.0)
    assert asyncio.run(worker.ablation_check_restored())["ok"]


def test_probe_combines_stats_unpads_and_enforces_cap(tmp_path):
    import numpy as np
    from verl import DataProto
    from verl_speco.integration.vllm_runtime import SPECO_VLLM_SPECULATIVE_TRACE_KEY

    trainer = object.__new__(FrozenActorLRTrainer)
    cohort = DataProto.from_dict(non_tensors={"raw_prompt": np.array(["a", "b", "c"], dtype=object)})
    runtime = SimpleNamespace(cohort=cohort, validation_batch=lambda count: cohort)
    trainer._speco_entropy_lr = lambda: runtime
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 8}}}})
    trainer._ablation_root = tmp_path
    trainer._speco_rollout_generation_target = lambda: SimpleNamespace(generate_sequences=lambda batch: batch)
    length = 10

    def measure(generate, args, kwargs):
        batch = args[0]
        assert len(batch) == 8
        assert batch.meta_info["speco_candidate_gate_max_tokens"] == 512
        traces = np.empty(8, dtype=object)
        for i in range(8):
            traces[i] = [dict(draft_count=7, accepted_count=2, draft_teacher_entropies=[.001]*3)]
        output = DataProto.from_dict(tensors={"responses": torch.ones(8, length, dtype=torch.long),
                                             "response_mask": torch.ones(8, length, dtype=torch.long)},
                                     non_tensors={SPECO_VLLM_SPECULATIVE_TRACE_KEY: traces})
        return output, dict(acceptance=3., response_tokens=float(length*8), elapsed_sec=1., ms_per_token=1000/(length*8))

    trainer._speco_measure_candidate_rollout = measure
    result = trainer._ablation_probe("test")
    assert result["acceptance"] == 3
    assert len(result["per_prompt"]) == 3
    assert result["response_tokens_unpadded"] == 30
    assert result["tokens_per_second_dispatched"] == 80
    assert (tmp_path / "test.json.gz").exists()
    trainer.config.actor_rollout_ref.rollout.drafter = {"training": {
        "frozen_drafter_compare_expected_prefix": [2]}}
    with pytest.raises(RuntimeError, match="Frozen actor output prefix mismatch"):
        trainer._ablation_probe("wrong_actor")
    trainer.config.actor_rollout_ref.rollout.drafter.training.frozen_drafter_compare_expected_prefix = [1]
    length = 513
    with pytest.raises(RuntimeError, match="cap violation"):
        trainer._ablation_probe("overlong")


@pytest.mark.parametrize(
    "ce_alphas,prefixes,fkl_factorial,objective_compare,lrs,checkpoints,performance_checkpoints,artifact_pattern,expected_arms,probes_per_arm",
    [
        (None, None, False, False, None, None, None, "drafter_lr_*_step_*.pt", 3, 12),
        ([0.0, 0.05, 0.1], None, False, False, None, None, None, "drafter_ce_*_step_*.pt", 3, 12),
        (None, [0, 1, 2, 3, 4], False, False, None, None, None, "drafter_*_step_*.pt", 5, 12),
        (None, None, True, False, None, None, None, "drafter_fkl_*_step_*.pt", 4, 12),
        (None, None, False, True, [1e-5], [5, 10], [5, 10],
         "drafter_*_ce01_step_*.pt", 2, 16),
        (
            [0.1, 0.2], None, False, False, [5e-6, 1e-5], [5, 10, 15, 20], [5, 10, 15, 20],
            "drafter_ce_*_step_*.pt", 4, 32,
        ),
    ],
)
def test_complete_grid_keeps_models_asleep_during_training_and_restores_each_arm(
    tmp_path, ce_alphas, prefixes, fkl_factorial, objective_compare, lrs, checkpoints, performance_checkpoints,
    artifact_pattern, expected_arms, probes_per_arm,
):
    import json
    import numpy as np
    from verl import DataProto

    worker = make_worker(tmp_path)
    coordinator = object.__new__(FrozenActorLRTrainer)
    cfg = worker.config.rollout.drafter.training
    if ce_alphas is not None:
        cfg.frozen_actor_ce_alphas = ce_alphas
        worker.trainer.model.first_rejection_ce_alpha = 0.0
    if prefixes is not None:
        cfg.frozen_actor_rejected_prefix_lengths = prefixes
        cfg.frozen_actor_rejected_aux_alpha = 0.1
        worker.trainer.model.first_rejection_ce_alpha = 0.0
        worker.trainer.model.rejected_aux_alpha = 0.0
        worker.trainer.model.rejected_prefix_length = 0
    if fkl_factorial:
        cfg.frozen_actor_fkl_factorial = True
        cfg.dspark_accepted_k3_alpha = 0.0
        cfg.dspark_accepted_fkl_alpha = 1.0
        cfg.dspark_rejected_k3_alpha = 0.0
        cfg.dspark_rejected_stream_weight = 0.0
        cfg.dspark_block_size = 7
        cfg.frozen_actor_rejected_aux_alpha = 0.1
        cfg.frozen_actor_rejected_prefix_length = 7
        worker.trainer.model.first_rejection_ce_alpha = 0.0
        worker.trainer.model.rejected_aux_alpha = 0.0
        worker.trainer.model.rejected_prefix_length = 0
    if objective_compare:
        cfg.frozen_actor_response_objective_compare = True
        cfg.dspark_accepted_k3_alpha = 1.0
        cfg.dspark_accepted_fkl_alpha = 0.0
        cfg.dspark_rejected_k3_alpha = 0.0
        cfg.dspark_rejected_stream_weight = 0.0
        cfg.dspark_rejected_aux_alpha = 0.0
        cfg.dspark_trace_aligned_anchors = True
        worker.trainer.model.accepted_k3_alpha = 1.0
        worker.trainer.model.accepted_fkl_alpha = 0.0
        worker.trainer.model.first_rejection_ce_alpha = 0.0
        worker.trainer.model.rejected_aux_alpha = 0.0
        worker.trainer.model.rejected_prefix_length = 0
    if lrs is not None:
        cfg.frozen_actor_lr_values = lrs
    if checkpoints is not None:
        cfg.frozen_actor_checkpoints = checkpoints
    if performance_checkpoints is not None:
        cfg.frozen_actor_performance_checkpoints = performance_checkpoints
    coordinator.config = OmegaConf.create({"actor_rollout_ref": {
        "model": {"path": "frozen-actor"}, "rollout": {"drafter": {"training": dict(cfg)}}}})
    coordinator.global_steps = 1
    coordinator._ablation_generation_cohorts = [["training prompt"]]
    cohort = DataProto.from_dict(non_tensors={"raw_prompt": np.array(["held out"], dtype=object)})
    coordinator._speco_entropy_lr = lambda: SimpleNamespace(prepare_validation_cohort=lambda: cohort)
    state = dict(awake=False, probes=0, publishes=0, restores=0)

    def wake(step):
        assert not state["awake"]
        state["awake"] = True

    def update_weights(step):
        if fkl_factorial or objective_compare:
            pytest.fail("The frozen actor objective experiment must not republish actor weights")
        wake(step)

    def sleep():
        assert state["awake"]
        state["awake"] = False

    def train(lr, steps, start, **kwargs):
        assert not state["awake"]
        return [asyncio.run(worker.ablation_train(lr, steps, start, **kwargs))]

    def publish(step, alpha):
        assert state["awake"]
        ready, weights = worker.trainer.get_candidate_state_dict(step, alpha)
        assert ready and weights
        state["publishes"] += 1
        return .01

    def probe(label, performance=False):
        assert state["awake"]
        state["probes"] += 1

    def finalize(step, alpha):
        assert not state["awake"]
        assert alpha == 0 and worker.trainer.finalize_candidate_update(step, alpha)
        state["restores"] += 1

    group = SimpleNamespace(ablation_prepare=lambda: [asyncio.run(worker.ablation_prepare())],
                            ablation_train=train,
                            ablation_check_restored=lambda: [asyncio.run(worker.ablation_check_restored())])
    coordinator._require_speco_worker_group = lambda: group
    coordinator._ray_get_if_needed = lambda value: value
    coordinator.checkpoint_manager = SimpleNamespace(
        update_weights=update_weights, wake_up_replicas=lambda: wake(None), sleep_replicas=sleep)
    coordinator._speco_publish_candidate_alpha = publish
    coordinator._speco_finalize_candidate = finalize
    coordinator._ablation_probe = probe
    trained, metrics = coordinator._speco_train_drafter()
    assert not trained and metrics["drafter/frozen_lr_ablation_complete"] == 1
    assert state == dict(
        awake=False,
        probes=3 + probes_per_arm * expected_arms,
        publishes=probes_per_arm * expected_arms,
        restores=expected_arms,
    )
    records = [json.loads(row) for row in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "completed"
    expected_checkpoints = len(checkpoints or [5, 20])
    assert len(list(tmp_path.glob(artifact_pattern))) == expected_checkpoints * expected_arms
    design = next(record for record in records if record["event"] == "design")
    expected_experiment = (
        "response_k3_vs_accepted_fkl_ce01" if objective_compare else
        "accepted_fkl_factorial" if fkl_factorial else
        "rejected_prefix" if prefixes is not None
        else ("first_rejection_ce" if ce_alphas else "learning_rate")
    )
    assert design["experiment"] == expected_experiment
    assert design["accepted_k3_alpha"] == pytest.approx(0.0 if fkl_factorial else 1.0)
    assert design["accepted_fkl_alpha"] == pytest.approx(1.0 if fkl_factorial else 0.0)
    assert design["rejected_k3_alpha"] == pytest.approx(0.0 if fkl_factorial or objective_compare else 1.0)
    assert design["response_stream_weight"] == pytest.approx(1.0)
    assert design["rejected_stream_weight"] == pytest.approx(0.0 if fkl_factorial or objective_compare else 1.0)
    if objective_compare:
        trained = [row for row in records if row["event"] == "trained"]
        assert {(row["arm"], row["accepted_k3_alpha"], row["accepted_fkl_alpha"],
                 row["first_rejection_ce_alpha"]) for row in trained} == {
            ("response_k3_ce01", 1.0, 0.0, 0.1),
            ("accepted_fkl_ce01", 0.0, 1.0, 0.1),
        }
    if fkl_factorial:
        trained = [row for row in records if row["event"] == "trained" and row["steps"] == 5]
        assert {(row["first_rejection_ce_alpha"], row["rejected_aux_alpha"],
                 row["rejected_prefix_length"]) for row in trained} == {
            (0.0, 0.0, 0), (0.0, 0.1, 7), (0.1, 0.0, 0), (0.1, 0.1, 7)
        }


def test_cached_batches_require_exact_saved_hashes(tmp_path):
    import json

    source = tmp_path / "source"
    first = make_worker(source)
    prepared = asyncio.run(first.ablation_prepare())
    (source / "results.jsonl").write_text(json.dumps(dict(event="prepared", workers=[prepared])) + "\n")
    second = make_worker(tmp_path / "resumed")
    del second.trainer._current_pad_size
    second.config.rollout.drafter.training.frozen_actor_lr_cached_source = str(source)
    second.trainer.prepare_training_batch = lambda: pytest.fail("Cached replay recollected data")
    restored = asyncio.run(second.ablation_prepare())
    assert restored["batch_hashes"] == prepared["batch_hashes"]
    assert restored["batch_file"] == str(source / "batches_rank_00")
    assert not list((tmp_path / "resumed").glob("batches_rank_*"))
    asyncio.run(second.ablation_train(1e-5, 5, start=True))
    assert second.trainer.optimizer_steps_total == 5
    prepared["batch_hashes"][0] = "corrupted"
    (source / "results.jsonl").write_text(json.dumps(dict(event="prepared", workers=[prepared])) + "\n")
    with pytest.raises(RuntimeError, match="Cached batch hash mismatch"):
        asyncio.run(second.ablation_prepare())


def test_worker_sets_first_rejection_ce_and_uses_distinct_artifact_name(tmp_path):
    worker = make_worker(tmp_path)
    worker.trainer.model.first_rejection_ce_alpha = 0.0

    async def run():
        await worker.ablation_prepare()
        result = await worker.ablation_train(
            1e-5, 5, start=True, first_rejection_ce_alpha=.1, artifact_tag="ce_0.10")
        assert result["first_rejection_ce_alpha"] == pytest.approx(.1)
        assert worker.trainer.model.first_rejection_ce_alpha == pytest.approx(.1)
        assert (tmp_path / "drafter_ce_0.10_step_5.pt").exists()
        assert worker.trainer.finalize_candidate_update(1, 0.0)
        assert (await worker.ablation_check_restored())["ok"]

    asyncio.run(run())


def test_worker_sets_rejected_prefix_arm_parameters(tmp_path):
    worker = make_worker(tmp_path)
    worker.trainer.model.first_rejection_ce_alpha = 0.0
    worker.trainer.model.rejected_aux_alpha = 0.0
    worker.trainer.model.rejected_prefix_length = 0

    async def run():
        await worker.ablation_prepare()
        result = await worker.ablation_train(
            1e-5,
            5,
            start=True,
            first_rejection_ce_alpha=0.1,
            rejected_aux_alpha=0.1,
            rejected_prefix_length=4,
            artifact_tag="rejected_prefix_4",
        )
        assert result["first_rejection_ce_alpha"] == pytest.approx(0.1)
        assert result["rejected_aux_alpha"] == pytest.approx(0.1)
        assert result["rejected_prefix_length"] == 4
        assert worker.trainer.model.rejected_aux_alpha == pytest.approx(0.1)
        assert worker.trainer.model.rejected_prefix_length == 4
        assert (tmp_path / "drafter_rejected_prefix_4_step_5.pt").exists()

    asyncio.run(run())


def test_worker_switches_accepted_objective_from_common_baseline(tmp_path):
    worker = make_worker(tmp_path)
    model = worker.trainer.model
    model.accepted_k3_alpha = 1.0
    model.accepted_fkl_alpha = 0.0
    model.first_rejection_ce_alpha = 0.0
    model.rejected_aux_alpha = 0.0
    model.rejected_prefix_length = 0

    async def run():
        await worker.ablation_prepare()
        outcomes = []
        for k3, fkl, name in ((1.0, 0.0, "response_k3_ce01"),
                              (0.0, 1.0, "accepted_fkl_ce01")):
            result = await worker.ablation_train(
                1e-5, 5, start=True, first_rejection_ce_alpha=0.1,
                rejected_aux_alpha=0.0, rejected_prefix_length=0,
                accepted_k3_alpha=k3, accepted_fkl_alpha=fkl, artifact_tag=name)
            assert (model.accepted_k3_alpha, model.accepted_fkl_alpha) == (k3, fkl)
            assert result["accepted_k3_alpha"] == k3
            assert result["accepted_fkl_alpha"] == fkl
            assert (tmp_path / f"drafter_{name}_step_5.pt").exists()
            outcomes.append(result["weights"])
            assert worker.trainer.finalize_candidate_update(1, 0.0)
            assert (await worker.ablation_check_restored())["ok"]
        assert outcomes[0] == outcomes[1]  # Mock loss ignores objective alphas.

    asyncio.run(run())
