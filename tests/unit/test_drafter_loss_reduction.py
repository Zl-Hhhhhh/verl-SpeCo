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

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

base_trainer_mod = pytest.importorskip("verl_speco.trainer.base_trainer")

DrafterBaseTrainer = base_trainer_mod.DrafterBaseTrainer


def _bare_trainer(sp_size=1, dp_size=1, sp_group=None, dp_group=None):
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.training_device_mesh = None
    trainer._get_sp_group = lambda: sp_group
    trainer._get_dp_group = lambda: dp_group
    trainer._get_sp_world_size = lambda: sp_size
    trainer._get_dp_world_size = lambda: dp_size
    return trainer


def test_reduce_loss_metrics_single_rank_passthrough():
    trainer = _bare_trainer()
    l_v = torch.tensor(2.0, requires_grad=True) * 1.0
    l_p = torch.tensor(3.0, requires_grad=True) * 1.0
    l_n = torch.tensor(4.0)

    global_vloss, global_ploss, global_tokens, world_size = (
        trainer._reduce_loss_metrics(l_v, l_p, l_n)
    )

    assert world_size == 1
    assert not global_vloss.requires_grad
    assert not global_ploss.requires_grad
    assert torch.equal(global_vloss, torch.tensor(2.0))
    assert torch.equal(global_ploss, torch.tensor(3.0))
    assert torch.equal(global_tokens, torch.tensor(4.0))


def test_reduce_loss_metrics_world_size_is_sp_times_dp(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_trainer_mod.dist,
        "all_reduce",
        lambda tensor, group=None: calls.append(group),
    )

    trainer = _bare_trainer(sp_size=4, dp_size=2, sp_group="sp", dp_group="dp")
    zeros = torch.zeros(())
    _, _, _, world_size = trainer._reduce_loss_metrics(zeros, zeros, zeros)

    assert world_size == 8
    assert calls == ["sp", "dp"]


def test_dp_local_loss_scaling_recovers_global_token_mean_gradient(monkeypatch):
    """Emulate 2 DP ranks: per-rank scaled local losses, backward, then an
    FSDP-style gradient mean must equal the global token-mean gradient."""
    torch.manual_seed(0)
    rank_inputs = [torch.randn(3, 4), torch.randn(5, 4)]
    total_tokens = sum(x.shape[0] for x in rank_inputs)

    def token_sum_loss(weight, inputs):
        return ((inputs @ weight) ** 2).sum()

    weight_ref = torch.randn(4, requires_grad=True)
    reference = (
        token_sum_loss(weight_ref, rank_inputs[0])
        + token_sum_loss(weight_ref, rank_inputs[1])
    ) / float(total_tokens)
    reference.backward()

    per_rank_grads = []
    for rank in (0, 1):
        weight = weight_ref.detach().clone().requires_grad_(True)
        l_v = torch.zeros(())
        l_p = token_sum_loss(weight, rank_inputs[rank])
        l_n = torch.tensor(float(rank_inputs[rank].shape[0]))

        other = 1 - rank
        other_metrics = torch.stack(
            [
                torch.zeros(()),
                token_sum_loss(weight_ref.detach(), rank_inputs[other]).detach(),
                torch.tensor(float(rank_inputs[other].shape[0])),
            ]
        )

        def fake_all_reduce(tensor, group=None, _other=other_metrics):
            tensor += _other

        monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", fake_all_reduce)
        trainer = _bare_trainer(dp_size=2, dp_group="dp")

        _, global_ploss, global_tokens, world_size = trainer._reduce_loss_metrics(
            l_v, l_p, l_n
        )
        assert world_size == 2
        assert torch.allclose(global_tokens, torch.tensor(float(total_tokens)))

        denom = global_tokens.clamp(min=1.0)
        local_loss = (0.0 * l_v + 1.0 * l_p) * (float(world_size) / denom)
        local_loss.backward()
        per_rank_grads.append(weight.grad)

    fsdp_mean_grad = (per_rank_grads[0] + per_rank_grads[1]) / 2.0
    assert torch.allclose(fsdp_mean_grad, weight_ref.grad, atol=1e-6)


@pytest.mark.parametrize("counts", [(1, 3), (0, 3), (0, 0)])
def test_auxiliary_ce_uses_global_first_rejection_mean_including_empty_ranks(monkeypatch, counts):
    torch.manual_seed(13)
    rank_inputs = [torch.randn(n, 4) for n in counts]
    reference_weight = torch.randn(4, 7, requires_grad=True)

    def loss_sum(weight, inputs):
        return .2 * torch.nn.functional.cross_entropy(
            inputs @ weight, torch.zeros(len(inputs), dtype=torch.long), reduction="sum"
        )

    reference = sum(loss_sum(reference_weight, x) for x in rank_inputs) / max(sum(counts), 1)
    reference.backward()
    gradients = []
    for rank in (0, 1):
        weight = reference_weight.detach().clone().requires_grad_(True)
        trainer = _bare_trainer(dp_size=2, dp_group="dp")
        other = 1 - rank
        other_metrics = torch.stack([
            torch.zeros(()), loss_sum(reference_weight.detach(), rank_inputs[other]),
            torch.tensor(float(counts[other])),
        ])
        monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", lambda t, group=None: t.add_(other_metrics))
        metric, backward, count = trainer._reduce_auxiliary_loss({
            # The base token count deliberately differs from the auxiliary one.
            "total_local_ploss": torch.tensor(100.), "local_num_tokens": torch.tensor(80.),
            "auxiliary_loss": {"loss_sum": loss_sum(weight, rank_inputs[rank]),
                               "token_count": torch.tensor(float(counts[rank]))},
        })
        assert count.item() == sum(counts)
        assert metric.item() == pytest.approx(reference.item())
        backward.backward()
        gradients.append(weight.grad)
    assert torch.allclose(sum(gradients) / 2, reference_weight.grad, atol=1e-6)


def test_disabled_auxiliary_loss_does_not_reduce(monkeypatch):
    trainer = _bare_trainer(dp_size=2, dp_group="dp")
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled auxiliary objective must not introduce a collective")
    monkeypatch.setattr(base_trainer_mod.dist, "all_reduce", unexpected)
    assert all(v.item() == 0 for v in trainer._reduce_auxiliary_loss({"total_local_ploss": torch.tensor(3.)}))


def test_multiple_auxiliary_losses_keep_independent_denominators():
    trainer = _bare_trainer()
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(6.0, requires_grad=True)
    metric, backward, count = trainer._reduce_auxiliary_losses({
        "total_local_ploss": torch.tensor(0.0),
        "auxiliary_losses": [
            {"loss_sum": first, "token_count": torch.tensor(2.0)},
            {"loss_sum": second, "token_count": torch.tensor(3.0)},
        ],
    })
    assert metric.item() == pytest.approx(3.0)
    assert count.item() == pytest.approx(5.0)
    backward.backward()
    assert first.grad.item() == pytest.approx(0.5)
    assert second.grad.item() == pytest.approx(1 / 3)


def test_optimizer_step_includes_separately_normalized_auxiliary_gradient(monkeypatch):
    trainer = _bare_trainer()
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(.3)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
    trainer.lr_scheduler = None
    trainer.optimizer_steps_total = trainer.training_steps = 0
    trainer._current_pad_size = 0
    trainer._training_metric_sums = {}
    trainer._ulysses_group_context = nullcontext
    trainer._record_dflash_training_metrics = lambda value: None
    trainer._maybe_dump_dspark_k3_logprobs = lambda *args, **kwargs: None
    monkeypatch.setattr(base_trainer_mod, "device_name", "cpu")

    def compute_loss(model, batch, pad):
        w = model.weight.squeeze()
        return {
            "total_local_vloss": w * 0, "total_local_ploss": 80 * w.square(),
            "local_num_tokens": torch.tensor(80.), "v_weight": 0., "p_weight": 1.,
            "auxiliary_loss": {"loss_sum": .2 * 3 * (w - 1).square(), "token_count": torch.tensor(3.)},
        }
    trainer.backend = SimpleNamespace(compute_loss=compute_loss)
    assert asyncio.run(trainer._training_step_on_batch({}, 1))
    # d/dw [w^2 + .2*(w-1)^2] at .3 = .32, below grad clip=1.
    assert trainer.model.weight.item() == pytest.approx(.3 - .1 * .32)
    assert trainer._training_metric_sums["drafter/train_loss_sum"] == pytest.approx(.3**2 + .2 * .7**2)
    assert trainer.optimizer_steps_total == 1
