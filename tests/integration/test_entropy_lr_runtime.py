from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.trainer.entropy_lr_runtime import EntropyLRRuntime
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto
from verl_speco.integration.vllm_runtime import SPECO_VLLM_SPECULATIVE_TRACE_KEY


def runtime(tmp_path):
    cfg = dict(candidate_gate_enabled=True, lr_scheduler_type='constant', lr_warmup_steps=0,
               training_interval_steps=5, lr=1e-5)
    calls = []
    trainer = SimpleNamespace(
        global_steps=5, config=SimpleNamespace(trainer=OmegaConf.create({'default_local_dir': str(tmp_path)})),
        _speco_drafter_training_config=lambda: cfg, _speco_resume_global_step_hint=lambda: None,
        _speco_current_step_rollout_quality_metrics=lambda: {'response_tokens': 1e6},
        _speco_publish_candidate_alpha=lambda step, alpha: calls.append(('publish', step, alpha)),
        _speco_finalize_candidate=lambda step, alpha: calls.append(('finalize', step, alpha)),
        _speco_pending_candidate_update={'update_step': 5},
    )
    result = EntropyLRRuntime(trainer)
    result.control.observe(dict(low_entropy_share=.85, mismatch=.15))
    result.observed_step = 5
    result.cohort = [None] * 64
    return result, calls


@pytest.mark.parametrize('kind,selected,next_lr', [('gain', 1., 1.1e-5), ('loss', 0., 5e-6), ('noise', 0., 1e-5)])
def test_candidate_commit_or_full_rollback_and_feedback(tmp_path, kind, selected, next_lr):
    control, calls = runtime(tmp_path)
    counts = []

    def measure(generate, count):
        candidate = calls[-1][2] == 1.
        gain = .1 if kind == 'gain' else (-.2 if kind == 'loss' else 0.)
        value = 3. + (gain if candidate else 0.)
        counts.append(count)
        return dict(per_prompt=[value] * count, acceptance=value,
                    mismatch=.12 if candidate and kind == 'gain' else .15,
                    ms_per_token=.9 if candidate and kind == 'gain' else 1.)

    control.measure = measure
    metrics = control.validate(None, {'timing_s/drafter_train_rpc': 1})
    assert calls[-1] == ('finalize', 5, selected)
    assert control.trainer._speco_pending_candidate_update is None
    assert control.control.current == pytest.approx(next_lr)
    assert metrics['drafter/candidate/selected_alpha'] == selected
    assert counts == ([32, 32, 64, 64] if kind == 'noise' else [32, 32])
    assert (tmp_path / 'entropy_lr_decisions.jsonl').is_file()
    control.save()
    assert (tmp_path / 'global_step_5/entropy_lr_state.json').is_file()


def test_probe_exception_restores_baseline_then_propagates(tmp_path):
    control, calls = runtime(tmp_path)

    def fail(*args):
        raise RuntimeError('missing trace')

    control.measure = fail
    with pytest.raises(RuntimeError, match='missing trace'):
        control.validate(None, {})
    assert calls[-2:] == [('publish', 5, 0.), ('finalize', 5, 0.)]
    assert control.control.current == 5e-6
    assert control.trainer._speco_pending_candidate_update is None


def test_validation_uses_unique_holdout_prompts_and_disables_collection(tmp_path):
    control, _ = runtime(tmp_path)
    control.cohort = None
    # Real AgentLoop dataset contract: only a dummy tensor before generation.
    # Exercise duplicates across batches and numpy/list prompt representations.
    rows = [dict(dummy_tensor=torch.tensor([0], dtype=torch.uint8),
                 raw_prompt=[{'role': 'user', 'content': f'Question {i}'}],
                 reward_model={'ground_truth': str(i)}, data_source='test')
            for i in range(64)]
    repeated = [dict(row, raw_prompt=np.array(row['raw_prompt'], dtype=object)) for row in rows]
    control.trainer.val_dataloader = [collate_fn(rows[:20]), collate_fn(repeated), collate_fn(rows)]
    control.trainer._get_gen_batch = lambda batch: RayPPOTrainer._get_gen_batch(control.trainer, batch)
    control.trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    batch = control.validation_batch(32)
    assert len(batch) == 32
    assert len({prompt[0]['content'] for prompt in batch.non_tensor_batch['raw_prompt']}) == 32
    assert 'input_ids' not in (batch.batch.keys() if batch.batch is not None else [])
    assert batch.non_tensor_batch['reward_model'][0]['ground_truth'] == '0'
    assert len(control.cohort) == 64
    assert batch.meta_info['validate'] is True
    assert batch.meta_info['do_sample'] is False
    assert batch.meta_info['speco_candidate_gate_max_tokens'] == 512
    assert control.validation_batch(64).non_tensor_batch['uid'][0] == batch.non_tensor_batch['uid'][0]


@pytest.mark.parametrize('unique_count,expected', [(1, None), (30, 30), (33, 33), (63, 63), (64, 64)])
def test_small_validation_cohort_counts_only_unique_prompts(tmp_path, unique_count, expected):
    control, _ = runtime(tmp_path)
    control.cohort = None
    control.trainer.val_dataloader = [collate_fn([
        dict(dummy_tensor=torch.tensor([0]), raw_prompt=[{'role': 'user', 'content': str(i)}])
        for i in range(unique_count)])]
    control.trainer._get_gen_batch = lambda batch: RayPPOTrainer._get_gen_batch(control.trainer, batch)
    control.trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    if expected is None:
        with pytest.raises(ValueError, match='two unique'):
            control.validation_batch(32)
    else:
        assert len(control.validation_batch(64)) == expected


def test_measure_pads_for_workers_but_excludes_padding_from_statistics(tmp_path):
    control, _ = runtime(tmp_path)
    control.cohort = None
    rows = [dict(dummy_tensor=torch.tensor([0]), raw_prompt=[{'role': 'user', 'content': str(i)}])
            for i in range(30)]
    control.trainer.val_dataloader = [collate_fn(rows * 32)]
    control.trainer._get_gen_batch = lambda batch: RayPPOTrainer._get_gen_batch(control.trainer, batch)
    control.trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    control.trainer.config.actor_rollout_ref = SimpleNamespace(rollout=SimpleNamespace(
        agent=SimpleNamespace(num_workers=8)))

    def generate(batch):
        assert len(batch) == 32
        assert batch.non_tensor_batch['uid'][0] == batch.non_tensor_batch['uid'][30]
        traces = np.empty(32, dtype=object)
        traces[:30] = [[dict(draft_count=1, accepted_count=1, draft_teacher_entropies=[.001])]] * 30
        traces[30:] = [None, None]  # Would fail trace statistics unless removed.
        return DataProto.from_dict(tensors={'response_mask': torch.ones(32, 4)},
                                   non_tensors={SPECO_VLLM_SPECULATIVE_TRACE_KEY: traces})

    def measure(generate_fn, args, kwargs):
        output = generate_fn(*args, **kwargs)
        assert len(output) == 30
        assert output.batch['response_mask'].sum().item() == 120
        return output, dict(ms_per_token=1.)

    control.trainer._speco_measure_candidate_rollout = measure
    stats = control.measure(generate, 32)
    assert len(stats['per_prompt']) == 30
    assert stats['reached'] == 30
    assert stats['acceptance'] == 2.
