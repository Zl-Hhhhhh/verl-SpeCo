import math
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.entropy_lr import (
    EntropyLRController, paired_evidence, set_cycle_lr, trace_statistics,
)


def event(a, h):
    return dict(draft_count=len(h), accepted_count=a, draft_teacher_entropies=h)


def test_entropy_mismatch_excludes_unreached_suffix():
    result = trace_statistics([[event(1, [.001, .001, 0.0]), event(3, [.001, .5, .001])]])
    assert result['reached'] == 5
    assert result['low'] == 4
    assert result['rejected_low'] == 1
    assert result['mismatch'] == .25
    assert result['acceptance'] == 3


@pytest.mark.parametrize('traces', [[], [None], [[]], [[event(0, [math.nan])]], [[event(0, [-1])]]])
def test_missing_or_bad_entropy_fails_closed(traces):
    with pytest.raises(ValueError):
        trace_statistics(traces)


def test_one_validated_gain_raises_lr_with_mismatch_and_available_gradient():
    control = EntropyLRController(ema_beta=1)
    control.observe(dict(low_entropy_share=.85, mismatch=.12))
    assert control.current == 1e-5  # certainty alone does not increase LR
    control.feedback(improved=True)
    assert control.current == pytest.approx(1.1e-5)
    for _ in range(20):
        control.feedback(improved=True)
    assert control.current == pytest.approx(1.425e-5)
    control.feedback(degraded=True)
    assert control.current == pytest.approx(7.125e-6)
    control.feedback(degraded=True)
    assert control.current == 5e-6
    control.feedback(improved=True, saturation=.5)
    control.feedback(improved=True, saturation=.5)
    assert control.current == pytest.approx(6.05e-6)
    control.observe(dict(low_entropy_share=.85, mismatch=.05))
    control.feedback(improved=True)
    control.feedback(improved=True)
    assert control.current == pytest.approx(6.05e-6)


def test_entropy_ceiling_and_state_roundtrip():
    control = EntropyLRController(ema_beta=.3)
    control.observe(dict(low_entropy_share=.7, mismatch=.12))
    control.observe(dict(low_entropy_share=.9, mismatch=.12))
    assert control.certainty_ema == pytest.approx(.76)
    assert control.ceiling == pytest.approx(1.38e-5)
    restored = EntropyLRController(**control.state_dict())
    assert restored.state_dict() == control.state_dict()


def test_scheduler_does_not_overwrite_selected_lr():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    set_cycle_lr(optimizer, scheduler, 1.4e-5)
    for _ in range(5):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 1.4e-5


def test_paired_probe_requires_gain_mismatch_and_net_savings():
    a = dict(per_prompt=[3.] * 32, acceptance=3., mismatch=.15, ms_per_token=1.)
    b = dict(per_prompt=[3.1] * 32, acceptance=3.1, mismatch=.13, ms_per_token=.9)
    assert paired_evidence(a, b, projected_tokens=1e6, overhead_s=10)['improved']
    assert not paired_evidence(a, b, projected_tokens=100, overhead_s=10)['improved']
    b['per_prompt'] = [2., 4.2] * 16
    assert not paired_evidence(a, b, projected_tokens=1e6, overhead_s=10)['improved']
    b.update(per_prompt=[2.8] * 32, acceptance=2.8)
    assert paired_evidence(a, b, projected_tokens=1e6, overhead_s=10)['degraded']


def test_small_but_significant_gain_can_raise_lr():
    baseline = dict(per_prompt=[3.] * 32, acceptance=3., mismatch=.15, ms_per_token=1.)
    candidate = dict(per_prompt=[3.01] * 32, acceptance=3.01, mismatch=.14, ms_per_token=.9)
    assert paired_evidence(
        baseline, candidate, projected_tokens=1e6, overhead_s=10
    )['improved']


def test_certainty_ceiling_has_no_activation_threshold():
    control = EntropyLRController(ema_beta=1)
    control.observe(dict(low_entropy_share=.5, mismatch=.12))
    assert control.ceiling == pytest.approx(1.25e-5)


def test_transaction_rollback_restores_adam_moments_and_scheduler():
    import copy
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-5)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    trainer.training_device_mesh = None
    trainer.rank = 0
    trainer.backend = SimpleNamespace(model_type='dspark')
    trainer.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(
        training={'entropy_lr_enabled': True, 'publish_dtype': None})))
    trainer.training_steps = trainer.optimizer_steps_total = 1
    trainer.model(torch.ones(1, 2)).sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer.lr_scheduler.step()
    before_weight = trainer.model.weight.detach().clone()
    before_opt = copy.deepcopy(trainer.optimizer.state_dict())
    before_scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    assert trainer.begin_candidate_update(5)
    set_cycle_lr(trainer.optimizer, trainer.lr_scheduler, 1.5e-5)
    trainer.model(torch.ones(1, 2)).square().sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer.lr_scheduler.step()
    trainer.training_steps = trainer.optimizer_steps_total = 2
    assert trainer.prepare_candidate_update(5)
    assert trainer.finalize_candidate_update(5, 0.)
    assert torch.equal(trainer.model.weight, before_weight)
    assert trainer.lr_scheduler.state_dict() == before_scheduler
    after = trainer.optimizer.state_dict()
    assert after['param_groups'] == before_opt['param_groups']
    for key, state in before_opt['state'].items():
        for name, value in state.items():
            assert torch.equal(after['state'][key][name], value)
    assert trainer.optimizer_steps_total == 1


@pytest.mark.parametrize('latency', [math.nan, math.inf, 0.0, -1.0])
def test_invalid_probe_measurement_cannot_support_lr_feedback(latency):
    baseline = dict(per_prompt=[3.] * 32, acceptance=3., mismatch=.15, ms_per_token=1.)
    candidate = dict(per_prompt=[3.1] * 32, acceptance=3.1, mismatch=.13, ms_per_token=latency)
    with pytest.raises(ValueError):
        paired_evidence(baseline, candidate, projected_tokens=1e6, overhead_s=10)
