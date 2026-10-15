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
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

dspark_backend = pytest.importorskip("verl_speco.backends.dspark_trainer_backend")
dspark_models = pytest.importorskip("verl_speco.models.dspark")
dflash_backend = pytest.importorskip("verl_speco.backends.dflash_trainer_backend")

DSparkTrainingModel = dspark_backend.DSparkTrainingModel
DSparkTrainerBackend = dspark_backend.DSparkTrainerBackend
trace_anchor_rows_for_item = dspark_backend._trace_anchor_rows_for_item
DSparkConfig = dspark_models.DSparkConfig
DSparkDraftModel = dspark_models.DSparkDraftModel
create_dense_attention_mask = dflash_backend._create_dflash_dense_attention_mask


def test_dspark_checkpoint_preserves_source_config_and_vllm_weight_names(
    tmp_path,
) -> None:
    initial_config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_config = initial_config.to_dict()
    source_config.update(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3DSparkModel"],
            "num_anchors": 17,
            "source_only_field": {"preserved": True},
        }
    )
    source_config.pop("enable_confidence_head", None)
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    config = DSparkConfig.from_dspark_pretrained(str(source_dir))
    config.num_anchors = 99

    model = DSparkDraftModel(config)
    state_keys = set(model.state_dict())
    assert {"fc.weight", "hidden_norm.weight", "norm.weight"}.issubset(state_keys)
    assert not {
        "context_proj.weight",
        "context_norm.weight",
        "final_norm.weight",
    }.intersection(state_keys)

    assert config.model_type == "dspark"
    exported_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.save_pretrained(
        output_dir, state_dict=exported_state, safe_serialization=False
    )
    saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    saved_state = torch.load(
        output_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    for key, value in source_config.items():
        assert saved_config[key] == json.loads(json.dumps(value))
    assert saved_config["enable_confidence_head"] is False
    assert {"fc.weight", "hidden_norm.weight", "norm.weight"}.issubset(saved_state)

    with pytest.raises(ValueError, match="contains no model weights"):
        dflash_backend.DFlashTrainerBackend._load_draft_checkpoint(
            None, model, str(output_dir), normalized_state={}
        )

    reloaded = DSparkConfig.from_dspark_pretrained(str(output_dir))
    assert reloaded.model_type == "dspark"
    assert reloaded.to_dict()["model_type"] == source_config["model_type"]
    assert reloaded.to_dict()["architectures"] == source_config["architectures"]


def _small_dspark_training_model(
    block_size: int = 4,
    num_hidden_layers: int = 1,
    l1_loss_alpha: float = 0.0,
    l1_chunk_size: int = 0,
    loss_mode: str = "full_vocab",
    first_rejection_ce_alpha: float = 0.0,
    first_rejection_ce_low_entropy_multiplier: float = 1.0,
    trace_aligned_anchors: bool = False,
    accepted_k3_alpha: float = 0.0,
    accepted_fkl_alpha: float = 0.0,
    rejected_k3_alpha: float = 0.0,
    rejected_aux_alpha: float = 0.0,
    rejected_prefix_length: int = 0,
    response_stream_weight: float = 1.0,
    rejected_stream_weight: float = 1.0,
    rejected_first_weight_multiplier: float = 1.0,
    rejected_suffix_weight_multiplier: float = 1.0,
    asymmetric_k3_enabled: bool = False,
    accepted_asymmetric_k3_enabled: bool | None = None,
    rejected_asymmetric_k3_enabled: bool | None = None,
    asymmetric_k3_positive_threshold: float = 2.0,
    dynamic_accept_reject_weighting_enabled: bool = False,
    dynamic_stream_weight_min_multiplier: float = 0.25,
    dynamic_stream_weight_max_multiplier: float = 4.0,
    k3_distribution_bins: int = 512,
    dump_k3_logprobs: bool = False,
    response_loss: str = "k3",
    rejected_loss: str = "k3",
    topk_support: str = "student",
):
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=block_size,
        num_anchors=2,
        markov_rank=4,
        markov_head_type="vanilla",
    )
    draft_model = DSparkDraftModel(config)
    return DSparkTrainingModel(
        draft_model=draft_model,
        block_size=block_size,
        num_anchors=2,
        loss_mode=loss_mode,
        first_rejection_ce_alpha=first_rejection_ce_alpha,
        first_rejection_ce_low_entropy_multiplier=(
            first_rejection_ce_low_entropy_multiplier
        ),
        l1_loss_alpha=l1_loss_alpha,
        l1_chunk_size=l1_chunk_size,
        trace_aligned_anchors=trace_aligned_anchors,
        accepted_k3_alpha=accepted_k3_alpha,
        accepted_fkl_alpha=accepted_fkl_alpha,
        rejected_k3_alpha=rejected_k3_alpha,
        rejected_aux_alpha=rejected_aux_alpha,
        rejected_prefix_length=rejected_prefix_length,
        response_stream_weight=response_stream_weight,
        rejected_stream_weight=rejected_stream_weight,
        rejected_first_weight_multiplier=rejected_first_weight_multiplier,
        rejected_suffix_weight_multiplier=rejected_suffix_weight_multiplier,
        asymmetric_k3_enabled=asymmetric_k3_enabled,
        accepted_asymmetric_k3_enabled=accepted_asymmetric_k3_enabled,
        rejected_asymmetric_k3_enabled=rejected_asymmetric_k3_enabled,
        asymmetric_k3_positive_threshold=asymmetric_k3_positive_threshold,
        dynamic_accept_reject_weighting_enabled=(
            dynamic_accept_reject_weighting_enabled
        ),
        dynamic_stream_weight_min_multiplier=dynamic_stream_weight_min_multiplier,
        dynamic_stream_weight_max_multiplier=dynamic_stream_weight_max_multiplier,
        k3_distribution_bins=k3_distribution_bins,
        dump_k3_logprobs=dump_k3_logprobs,
        response_loss=response_loss,
        rejected_loss=rejected_loss,
        topk_support=topk_support,
    )


def test_state_probe_preserves_loss_gradients_and_rng(monkeypatch, tmp_path):
    torch.manual_seed(73)
    model = _small_dspark_training_model(
        block_size=2, trace_aligned_anchors=True, accepted_k3_alpha=1.,
        rejected_k3_alpha=1., dump_k3_logprobs=True,
    )
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3]]),
        hidden_states_list=[torch.randn(1, 3, 8), torch.randn(1, 3, 8)],
        loss_mask=torch.ones(1, 3), lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(1, 3, 8),
        trace_anchor_positions=torch.tensor([[0]]), trace_anchor_mask=torch.tensor([[True]]),
        trace_accepted_counts=torch.tensor([[1]]), trace_draft_counts=torch.tensor([[2]]),
        trace_rejected_token_ids=torch.tensor([[12]]), trace_draft_token_ids=torch.tensor([[[2, 12]]]),
        trace_draft_teacher_logprobs=torch.tensor([[[-.2, -2.]]]),
        trace_draft_teacher_logprob_mask=torch.ones(1, 1, 2, dtype=torch.bool),
    )
    monkeypatch.delenv("SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR", raising=False)
    baseline = model(**inputs)[0]
    baseline.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    rng = torch.random.get_rng_state().clone()
    monkeypatch.setenv("SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR", str(tmp_path))
    loss, *_, diagnostics = model(**inputs)
    loss.backward()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(loss, baseline)
    for name, p in model.named_parameters():
        if name in gradients:
            assert torch.equal(p.grad, gradients[name])
    assert diagnostics["state_probe"]["context_feature"].shape[:2] == (1, 3)
    assert not diagnostics["state_probe"]["draft_hidden"].requires_grad


@pytest.mark.parametrize("response_loss,rejected_loss", [("k3", "k3"), ("topk", "k3"), ("k3", "topk"), ("topk", "topk")])
@pytest.mark.parametrize("support", ["student", "union"])
def test_topk_stream_selectors_preserve_joint_weighting(response_loss, rejected_loss, support):
    torch.manual_seed(73)
    model = _small_dspark_training_model(
        block_size=2, trace_aligned_anchors=True, accepted_k3_alpha=1., rejected_k3_alpha=1.,
        response_loss=response_loss, rejected_loss=rejected_loss,
        topk_support=support,
        rejected_stream_weight=.5, dynamic_accept_reject_weighting_enabled=True,
        dump_k3_logprobs=True,
    )
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3]]),
        hidden_states_list=[torch.randn(1, 3, 8), torch.randn(1, 3, 8)],
        loss_mask=torch.ones(1, 3), lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(1, 3, 8),
        trace_anchor_positions=torch.tensor([[0]]), trace_anchor_mask=torch.tensor([[True]]),
        trace_accepted_counts=torch.tensor([[1]]), trace_draft_counts=torch.tensor([[2]]),
        trace_rejected_token_ids=torch.tensor([[12]]), trace_draft_token_ids=torch.tensor([[[2, 12]]]),
        trace_draft_teacher_logprobs=torch.tensor([[[-.2, -2.]]]),
        trace_draft_teacher_logprob_mask=torch.ones(1, 1, 2, dtype=torch.bool),
    )
    if rejected_loss == "topk":
        width = 32 if support == "union" else 16
        inputs.update(
            trace_draft_topk_ids=torch.arange(width).expand(1, 1, 2, width),
            trace_draft_topk_teacher_logprobs=torch.randn(1, 1, 2, 32).log_softmax(-1)[..., :width],
        )
    loss, *_, stats = model(**inputs)
    loss.backward()
    assert torch.isfinite(loss)
    response_sum = stats["response_topk_loss_sum" if response_loss == "topk" else "accepted_k3_sum"]
    rejected_sum = stats["rejected_topk_loss_sum" if rejected_loss == "topk" else "rejected_k3_sum"]
    expected = (
        response_sum * stats["response_stream_effective_weight"]
        + rejected_sum * stats["rejected_stream_effective_weight"]
    ) / stats["objective_weighted_token_count"]
    assert loss.item() == pytest.approx(expected.item())
    assert stats["response_topk_enabled"].item() == (response_loss == "topk")
    assert stats["rejected_topk_enabled"].item() == (rejected_loss == "topk")
    assert stats["topk_union_enabled"].item() == (support == "union")
    if rejected_loss == "topk":
        del inputs["trace_draft_topk_ids"]
        with pytest.raises(ValueError, match="requires aligned"):
            model(**inputs)


@pytest.mark.parametrize("mode", ["k3", "topk"])
@pytest.mark.parametrize("first,suffix", [(1.0, 1.0), (2.0, 0.4), (2.0, 0.0), (0.0, 0.0)])
def test_rejected_multipliers_follow_first_rejection_and_joint_normalization(mode, first, suffix):
    torch.manual_seed(73)
    model = _small_dspark_training_model(
        block_size=3, trace_aligned_anchors=True,
        accepted_k3_alpha=1.0, rejected_k3_alpha=1.0,
        response_loss=mode, rejected_loss=mode, rejected_stream_weight=0.5,
        rejected_first_weight_multiplier=first,
        rejected_suffix_weight_multiplier=suffix, dump_k3_logprobs=True,
    )
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).expand(2, -1),
        hidden_states_list=[torch.randn(2, 7, 8), torch.randn(2, 7, 8)],
        loss_mask=torch.ones(2, 7), lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(2, 7, 8),
        trace_anchor_positions=torch.tensor([[0, 3], [0, 3]]),
        trace_anchor_mask=torch.ones(2, 2, dtype=torch.bool),
        trace_accepted_counts=torch.tensor([[0, 1], [2, 3]]),
        trace_draft_counts=torch.full((2, 2), 3),
        trace_rejected_token_ids=torch.tensor([[12, 12], [12, -1]]),
        trace_draft_token_ids=torch.tensor([
            [[12, 13, 14], [5, 12, 14]], [[2, 3, 12], [5, 6, 7]],
        ]),
        trace_draft_teacher_logprobs=torch.full((2, 2, 3), -2.0),
        trace_draft_teacher_logprob_mask=torch.ones(2, 2, 3, dtype=torch.bool),
    )
    if mode == "topk":
        inputs.update(
            trace_draft_topk_ids=torch.arange(16).expand(2, 2, 3, 16),
            trace_draft_topk_teacher_logprobs=torch.randn(2, 2, 3, 32).log_softmax(-1)[..., :16],
        )
    loss, *_, stats = model(**inputs)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    decay = model.rejected_position_decay
    expected_rejected_count = first * (1 + decay + decay**2) + suffix * (decay + 2 * decay**2)
    assert stats["rejected_k3_weighted_count"].item() == pytest.approx(expected_rejected_count)
    assert stats["ce_weighted_token_count"].item() == pytest.approx(9.0)
    assert stats["objective_weighted_token_count"].item() == pytest.approx(9 + 0.5 * expected_rejected_count)
    assert loss.item() == pytest.approx((
        (stats["response_objective_sum"] + 0.5 * stats["rejected_objective_sum"])
        / stats["objective_weighted_token_count"]
    ).item())
    if mode == "topk":
        assert stats["rejected_topk_weighted_count"].item() == pytest.approx(expected_rejected_count)
    rejected = stats["k3_logprob_dump"]["rejected"]
    if first == suffix == 0:
        assert not rejected or rejected["weights"].numel() == 0
    else:
        positions = rejected["draft_positions"]
        expected_weights = decay ** positions.float() * torch.where(
            positions == rejected["trace_accepted_counts"], first, suffix,
        )
        assert torch.allclose(rejected["weights"], expected_weights)
        assert (rejected["trace_accepted_counts"] < 3).all()
    assert (stats["k3_logprob_dump"]["accepted"]["weights"] == 1).all()


@pytest.mark.parametrize("name", ["rejected_first_weight_multiplier", "rejected_suffix_weight_multiplier"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_rejected_multipliers_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _small_dspark_training_model(trace_aligned_anchors=True, **{name: value})


@pytest.mark.parametrize("name", ["rejected_first_weight_multiplier", "rejected_suffix_weight_multiplier"])
def test_rejected_multipliers_require_trace_alignment(name):
    with pytest.raises(ValueError, match="requires trace_aligned_anchors"):
        _small_dspark_training_model(**{name: 2.0})


def _first_rejection_ce_inputs():
    return dict(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).expand(2, -1),
        hidden_states_list=[torch.randn(2, 7, 8), torch.randn(2, 7, 8)],
        loss_mask=torch.ones(2, 7), lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(2, 7, 8),
        trace_anchor_positions=torch.tensor([[0, 3], [0, 3]]),
        trace_anchor_mask=torch.ones(2, 2, dtype=torch.bool),
        trace_accepted_counts=torch.tensor([[0, 1], [2, 3]]),
        trace_draft_counts=torch.full((2, 2), 3),
        trace_rejected_token_ids=torch.tensor([[12, 12], [12, -1]]),
        trace_draft_token_ids=torch.tensor([
            [[12, 13, 14], [5, 12, 14]], [[2, 3, 12], [5, 6, 7]],
        ]),
        trace_draft_teacher_logprobs=torch.full((2, 2, 3), -2.0),
        trace_draft_teacher_logprob_mask=torch.ones(2, 2, 3, dtype=torch.bool),
        trace_draft_topk_ids=torch.arange(16).expand(2, 2, 3, 16),
        trace_draft_topk_teacher_logprobs=torch.randn(2, 2, 3, 32).log_softmax(-1)[..., :16],
    )


def test_first_rejection_ce_targets_only_correct_replacements_without_low_q_clamp(monkeypatch):
    model = _small_dspark_training_model(
        block_size=3, trace_aligned_anchors=True, first_rejection_ce_alpha=0.2,
    )
    logits = torch.zeros(9, 32)
    logits[0, 2] = -30.0
    logits.requires_grad_(True)
    monkeypatch.setattr(model, "_full_vocab_logits", lambda *args: logits)
    loss, *_, stats = model(**_first_rejection_ce_inputs())
    # The nine response rows contain three first rejections at rows 0, 2, 5.
    # Correct replacements are 2, 6, 4, not the recorded wrong token 12.
    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = 0.2 * torch.nn.functional.cross_entropy(
        reference_logits[[0, 2, 5]], torch.tensor([2, 6, 4])
    )
    loss.backward()
    expected.backward()
    assert loss.item() == pytest.approx(expected.item())
    assert stats["first_rejection_ce_token_count"].item() == 3
    assert torch.allclose(logits.grad, reference_logits.grad)
    assert logits.grad[0, 2].item() == pytest.approx(-0.2 / 3)
    assert logits.grad[[1, 3, 4, 6, 7, 8]].abs().sum() == 0


def test_weighted_first_rejection_ce_uses_both_gates_and_global_weight_sum(monkeypatch):
    model = _small_dspark_training_model(
        block_size=3, trace_aligned_anchors=True,
        first_rejection_ce_alpha=0.1,
        first_rejection_ce_low_entropy_multiplier=2.0,
    )
    inputs = _first_rejection_ce_inputs()
    inputs["trace_first_rejection_teacher_entropies"] = torch.tensor(
        [[0.005, 0.2], [0.005, float("nan")]]
    )
    inputs["trace_draft_teacher_logprobs"][0, 0, 0] = -5.0
    inputs["trace_draft_teacher_logprobs"][0, 1, 1] = -5.0
    inputs["trace_draft_teacher_logprobs"][1, 0, 2] = -2.0
    logits = torch.randn(9, 32, requires_grad=True)
    monkeypatch.setattr(model, "_full_vocab_logits", lambda *args: logits)
    loss, *_, stats = model(**inputs)
    reference_logits = logits.detach().clone().requires_grad_(True)
    ce = torch.nn.functional.cross_entropy(
        reference_logits[[0, 2, 5]], torch.tensor([2, 6, 4]), reduction="none"
    )
    expected = 0.1 * (ce * torch.tensor([2.0, 1.0, 1.0])).sum() / 4
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert stats["first_rejection_ce_token_count"].item() == 3
    assert stats["first_rejection_ce_weight_sum"].item() == 4
    assert stats["first_rejection_ce_boosted_count"].item() == 1
    loss.backward()
    expected.backward()
    assert torch.allclose(logits.grad, reference_logits.grad, atol=1e-7)

    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=inputs["lm_head_weight"]))
    batch = {k: v for k, v in inputs.items() if k not in ("lm_head_weight", "hidden_states_list")}
    batch["hidden_states"] = torch.cat(inputs["hidden_states_list"], dim=-1)
    auxiliary = backend.compute_loss(model, batch, 0)["auxiliary_loss"]
    assert auxiliary["token_count"].item() == 4


def test_weighted_first_rejection_ce_rejects_missing_verify_data():
    model = _small_dspark_training_model(
        block_size=3, trace_aligned_anchors=True,
        first_rejection_ce_alpha=0.1,
        first_rejection_ce_low_entropy_multiplier=2.0,
    )
    inputs = _first_rejection_ce_inputs()
    with pytest.raises(ValueError, match="requires verify entropies"):
        model(**inputs)
    inputs["trace_first_rejection_teacher_entropies"] = torch.full((2, 2), float("nan"))
    with pytest.raises(ValueError, match="requires finite entropy"):
        model(**inputs)
    inputs["trace_first_rejection_teacher_entropies"].fill_(0.005)
    inputs["trace_draft_teacher_logprob_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="candidate logprob"):
        model(**inputs)


@pytest.mark.parametrize("prefix_length,expected", [(1, 3), (2, 5), (3, 6)])
def test_rejected_auxiliary_uses_positions_relative_to_first_rejection(
    prefix_length, expected
):
    torch.manual_seed(79)
    model = _small_dspark_training_model(
        block_size=3,
        trace_aligned_anchors=True,
        accepted_k3_alpha=1.0,
        rejected_k3_alpha=0.0,
        rejected_aux_alpha=0.02,
        rejected_prefix_length=prefix_length,
        dump_k3_logprobs=True,
    )
    inputs = _first_rejection_ce_inputs()
    full_loss, *_, diagnostics = model(**inputs)
    rejected = diagnostics["k3_logprob_dump"]["rejected"]
    relative = rejected["draft_positions"] - rejected["trace_accepted_counts"]
    assert len(relative) == expected
    assert (relative >= 0).all() and (relative < prefix_length).all()
    expected_auxiliary = (
        0.02
        * diagnostics["rejected_k3_sum"]
        / diagnostics["rejected_k3_weighted_count"]
    )
    model.rejected_aux_alpha = 0.0
    base_loss, *_ = model(**inputs)
    model.rejected_aux_alpha = 0.02
    assert full_loss.item() == pytest.approx(
        (base_loss + expected_auxiliary).item(), abs=1e-6
    )

    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=inputs["lm_head_weight"]))
    batch = {k: v for k, v in inputs.items() if k not in ("lm_head_weight", "hidden_states_list")}
    batch["hidden_states"] = torch.cat(inputs["hidden_states_list"], dim=-1)
    result = backend.compute_loss(model, batch, 0)
    auxiliary = result["auxiliary_loss"]
    assert auxiliary["name"] == "rejected_k3"
    assert auxiliary["token_count"].item() == pytest.approx(
        diagnostics["rejected_k3_weighted_count"].item()
    )


def test_first_ce_and_rejected_k3_are_separate_auxiliary_means():
    model = _small_dspark_training_model(
        block_size=3,
        trace_aligned_anchors=True,
        accepted_k3_alpha=1.0,
        first_rejection_ce_alpha=0.1,
        rejected_k3_alpha=0.0,
        rejected_aux_alpha=0.02,
        rejected_prefix_length=2,
    )
    inputs = _first_rejection_ce_inputs()
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=inputs["lm_head_weight"]))
    batch = {k: v for k, v in inputs.items() if k not in ("lm_head_weight", "hidden_states_list")}
    batch["hidden_states"] = torch.cat(inputs["hidden_states_list"], dim=-1)
    result = backend.compute_loss(model, batch, 0)
    assert "auxiliary_loss" not in result
    assert [item["name"] for item in result["auxiliary_losses"]] == [
        "first_rejection_ce", "rejected_k3"
    ]


@pytest.mark.parametrize("ce_alpha", [0.0, 0.1])
@pytest.mark.parametrize("rejected_alpha", [0.0, 0.1])
def test_accepted_fkl_factorial_keeps_auxiliary_means_independent(
    ce_alpha, rejected_alpha
):
    torch.manual_seed(89)
    model = _small_dspark_training_model(
        block_size=3, trace_aligned_anchors=True, accepted_fkl_alpha=1.0,
        accepted_k3_alpha=0.0, rejected_k3_alpha=0.0,
        rejected_stream_weight=0.0, first_rejection_ce_alpha=ce_alpha,
        rejected_aux_alpha=rejected_alpha,
        rejected_prefix_length=3 if rejected_alpha else 0,
    )
    loss, *_, stats = model(**_first_rejection_ce_inputs())
    expected = stats["accepted_fkl_sum"] / stats["objective_weighted_token_count"]
    if ce_alpha:
        expected = expected + ce_alpha * stats["first_rejection_ce_loss_sum"] / stats[
            "first_rejection_ce_token_count"
        ]
    if rejected_alpha:
        expected = expected + rejected_alpha * stats["rejected_k3_sum"] / stats[
            "rejected_k3_weighted_count"
        ]
        assert stats["rejected_k3_weighted_count"].item() > 0
    else:
        assert stats["rejected_k3_weighted_count"].item() == 0
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


@pytest.mark.parametrize("loss_mode", ["full_vocab", "chunked_ce"])
@pytest.mark.parametrize("all_accepted", [False, True])
def test_first_rejection_ce_backend_keeps_base_objective_and_separate_mean(loss_mode, all_accepted):
    torch.manual_seed(73)
    model = _small_dspark_training_model(
        block_size=3, loss_mode=loss_mode, trace_aligned_anchors=True,
        accepted_k3_alpha=1., rejected_k3_alpha=1.,
        response_loss="topk", rejected_loss="topk", rejected_stream_weight=.5,
        first_rejection_ce_alpha=.2, dump_k3_logprobs=True,
    )
    inputs = _first_rejection_ce_inputs()
    if all_accepted:
        inputs["trace_accepted_counts"].fill_(3)
        inputs["trace_rejected_token_ids"].fill_(-1)
        inputs["trace_draft_token_ids"] = torch.tensor([[[2, 3, 4], [5, 6, 7]]]).expand(2, -1, -1)
    full_loss, *_, stats = model(**inputs)
    accepted = stats["k3_logprob_dump"]["accepted"]
    first = (accepted["draft_positions"] == accepted["trace_accepted_counts"]) & (
        accepted["trace_accepted_counts"] < accepted["trace_draft_counts"]
    )
    expected_ce_sum = -accepted["draft_logprobs"][first].sum()
    assert stats["first_rejection_ce_loss_sum"].item() == pytest.approx(expected_ce_sum.item(), abs=1e-5)
    model.first_rejection_ce_alpha = 0.0
    base_loss, *_, base_stats = model(**inputs)
    model.first_rejection_ce_alpha = .2
    for name in ("response_objective_sum", "rejected_objective_sum", "objective_weighted_token_count"):
        assert torch.equal(stats[name], base_stats[name])
    contribution = .2 * expected_ce_sum / first.sum().clamp(min=1)
    assert full_loss.item() == pytest.approx((base_loss + contribution).item())

    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=inputs["lm_head_weight"]))
    batch = {k: v for k, v in inputs.items() if k not in ("lm_head_weight", "hidden_states_list")}
    batch["hidden_states"] = torch.cat(inputs["hidden_states_list"], dim=-1)
    result = backend.compute_loss(model, batch, 0)
    assert result["total_local_ploss"].item() == pytest.approx((base_loss * result["local_num_tokens"]).item())
    assert result["auxiliary_loss"]["loss_sum"].item() == pytest.approx(.2 * expected_ce_sum.item(), abs=1e-5)
    assert result["auxiliary_loss"]["token_count"].item() == first.sum().item()
    assert result["auxiliary_loss"]["loss_sum"].requires_grad
    assert not any(k.startswith("_") for k in result["diagnostics"])
    if all_accepted:
        assert torch.isfinite(full_loss) and contribution == 0
    model.first_rejection_ce_alpha = 0.
    disabled = backend.compute_loss(model, batch, 0)
    assert "auxiliary_loss" not in disabled


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_first_rejection_ce_rejects_invalid_alpha(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _small_dspark_training_model(first_rejection_ce_alpha=value)


def test_first_rejection_ce_requires_trace_and_exact_ce():
    with pytest.raises(ValueError, match="requires trace_aligned_anchors"):
        _small_dspark_training_model(first_rejection_ce_alpha=.2)
    with pytest.raises(ValueError, match="requires full-vocabulary CE"):
        _small_dspark_training_model(first_rejection_ce_alpha=.2, trace_aligned_anchors=True, loss_mode="sampled_ce")


@pytest.mark.parametrize("support", ["student", "union"])
def test_topk_candidates_survive_crop_filter_and_padded_training_batch(support):
    from omegaconf import OmegaConf
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    model = _small_dspark_training_model(
        block_size=2, trace_aligned_anchors=True, accepted_k3_alpha=1., rejected_k3_alpha=1.,
        response_loss="topk", rejected_loss="topk", topk_support=support,
    ).to(dtype=torch.bfloat16)
    config = OmegaConf.create({"rollout": {"drafter": {"training": {
        "dspark_trace_aligned_anchors": True, "dspark_rejected_loss": "topk", "dspark_topk": 16,
        "dspark_topk_support": support,
        "dspark_num_anchors": 2, "dspark_block_size": 2, "dspark_max_window": 8,
    }}}})
    candidate_ids = [list(range(16)), list(range(16, 32))]
    width = 32 if support == "union" else 16
    if support == "union":
        candidate_ids = [ids + list(range(8, 24)) for ids in candidate_ids]
    event = {
        "response_start": 0, "draft_count": 2, "accepted_count": 0,
        "draft_token_ids": [2, 16], "draft_teacher_logprobs": [-2., -3.],
        "draft_teacher_entropies": [0.005, 0.2],
        "draft_topk_ids": candidate_ids,
        "draft_topk_teacher_logprobs": [[-4.] * width, [-5.] * width],
    }
    item = {
        "input_ids": torch.arange(1, 9), "hidden_states": torch.randn(8, 24),
        "hidden_states_layout": "dflash_aux_plus_last", "loss_mask": torch.ones(8),
        "_verl_prompt_len": 1, "sample_id": "a", "speculative_trace": [event],
    }
    longer = {**item, "sample_id": "b", "speculative_trace": [event, {**event, "response_start": 3}]}
    skipped = {**item, "sample_id": "skip", "speculative_trace": []}
    backend = object.__new__(DSparkTrainerBackend)
    backend.config = config
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.backend = backend
    trainer.config = config
    trainer.model = model
    trainer.model_config = model.draft_model.config
    trainer.batch_size = 3
    trainer.current_rl_step = 5
    trainer.training_steps = 0
    trainer.use_ulysses_sp = False
    trainer.rank = 0
    batch = trainer._prepare_training_batch(items_override=[item, skipped, longer], track_sample_usage=False)
    assert batch["trace_draft_topk_ids"].shape == (2, 2, 2, width)
    assert batch["trace_first_rejection_teacher_entropies"].shape == (2, 2)
    assert batch["trace_first_rejection_teacher_entropies"][0, 0].item() == pytest.approx(0.005)
    assert torch.isnan(batch["trace_first_rejection_teacher_entropies"][0, 1])
    assert batch["trace_draft_topk_ids"][0, 0].tolist() == candidate_ids
    assert (batch["trace_draft_topk_ids"][0, 1] == -1).all()
    assert batch["trace_draft_topk_ids"][1, 1].tolist() == candidate_ids
    assert batch["trace_draft_topk_teacher_logprobs"][1, 1, 1, 0].item() == -5.
    backend.target_lm_head = SimpleNamespace(fc=torch.nn.Linear(8, 32, bias=False).to(dtype=torch.bfloat16))
    result = backend.compute_loss(model, batch, 0)
    result["total_local_ploss"].backward()
    assert result["diagnostics"]["rejected_topk_weighted_count"].item() > 0
    trainer.reset_training_metrics()
    trainer.runtime_device = torch.device("cpu")
    trainer.optimizer = None
    trainer.optimizer_steps_total = 1
    trainer._reduce_training_metric = lambda value: value
    trainer._record_dflash_training_metrics(result)
    metrics = trainer.get_training_metrics()
    stats = result["diagnostics"]
    assert metrics["dspark/rejected_topk_enabled"] == 1.
    assert metrics["dspark/rejected_topk_loss"] == pytest.approx(
        (stats["rejected_topk_loss_sum"] / stats["rejected_topk_weighted_count"]).item()
    )
    assert metrics["dspark/response_topk_teacher_mass"] > 0
    assert metrics["dspark/topk_union_enabled"] == (support == "union")
    assert metrics["dspark/rejected_topk_support_size"] == pytest.approx(24 if support == "union" else 16)
    assert 16 <= metrics["dspark/response_topk_support_size"] <= width


def test_trace_anchor_rows_map_real_verify_starts_into_crop() -> None:
    item = {
        "sample_id": "sample-a",
        "_verl_prompt_len": 5,
        "_verl_feature_start": 2,
        "speculative_trace": [
            {
                "response_start": 0,
                "draft_count": 3,
                "accepted_count": 1,
                "draft_token_ids": [7, 8, 9],
                "draft_teacher_logprobs": [-0.2, -2.0, -3.0],
                "rejected_draft_token_id": 8,
            },
            {
                "response_start": 2,
                "draft_count": 3,
                "accepted_count": 3,
                "draft_token_ids": [10, 11, 12],
                "draft_teacher_logprobs": [-0.3, -0.4, -0.5],
                "all_drafts_accepted": True,
            },
            {
                "response_start": 4,
                "draft_count": 3,
                "accepted_count": 0,
                "draft_token_ids": [13, 14, 15],
                "draft_teacher_logprobs": [-1.0, -2.0, -3.0],
            },
        ],
    }

    rows = trace_anchor_rows_for_item(
        item,
        crop_start=1,
        crop_end=7,
        num_anchors=8,
        block_size=3,
    )

    assert [row["anchor"] for row in rows] == [1, 3]
    assert rows[0]["rejected_token_id"] == 8
    assert rows[0]["draft_teacher_logprobs"] == [-0.2, -2.0, -3.0]
    assert rows[1]["all_accepted"] is True


def test_all_accepted_trace_adds_response_only_bonus_anchor() -> None:
    item = {
        "sample_id": "sample-bonus",
        "_verl_prompt_len": 2,
        "_verl_feature_start": 0,
        "speculative_trace": [
            {
                "response_start": 0,
                "draft_count": 3,
                "accepted_count": 3,
                "draft_token_ids": [7, 8, 9],
                "draft_teacher_logprobs": [-0.2, -0.3, -0.4],
                "verified_token_ids": [7, 8, 9, 10],
                "verification_token_id": 10,
                "all_drafts_accepted": True,
            }
        ],
    }

    rows = trace_anchor_rows_for_item(
        item,
        crop_start=0,
        crop_end=7,
        num_anchors=8,
        block_size=3,
    )

    assert [row["anchor"] for row in rows] == [1, 4]
    assert rows[0]["bonus_response_anchor"] is False
    assert rows[1] == {
        "anchor": 4,
        "accepted_count": 1,
        "draft_count": 1,
        "rejected_token_id": -1,
        "draft_token_ids": [10],
        "draft_teacher_logprobs": [0.0],
        "draft_teacher_logprob_mask": [False],
        "all_accepted": True,
        "bonus_response_anchor": True,
        "priority": rows[1]["priority"],
    }


def test_bonus_anchor_enters_response_k3_without_rejected_stream() -> None:
    torch.manual_seed(19)
    model = _small_dspark_training_model(
        block_size=3,
        trace_aligned_anchors=True,
        accepted_k3_alpha=1.0,
        rejected_k3_alpha=1.0,
    )
    input_ids = torch.tensor([[1, 2, 7, 8, 9, 10, 11]], dtype=torch.long)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32)
    hidden_states = [torch.randn(1, 7, 8), torch.randn(1, 7, 8)]
    target_last_hidden_states = torch.randn(1, 7, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
        trace_anchor_positions=torch.tensor([[1, 4]], dtype=torch.long),
        trace_anchor_mask=torch.tensor([[True, True]]),
        trace_accepted_counts=torch.tensor([[3, 1]], dtype=torch.long),
        trace_draft_counts=torch.tensor([[3, 1]], dtype=torch.long),
        trace_rejected_token_ids=torch.tensor([[-1, -1]], dtype=torch.long),
        trace_draft_token_ids=torch.tensor(
            [[[7, 8, 9], [10, -1, -1]]], dtype=torch.long
        ),
        trace_draft_teacher_logprobs=torch.tensor(
            [[[-0.2, -0.3, -0.4], [0.0, 0.0, 0.0]]], dtype=torch.float32
        ),
        trace_draft_teacher_logprob_mask=torch.tensor(
            [[[True, True, True], [False, False, False]]], dtype=torch.bool
        ),
        trace_bonus_response_anchor_mask=torch.tensor(
            [[False, True]], dtype=torch.bool
        ),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["trace_accepted_position_count"].item() == 4
    assert diagnostics["trace_bonus_response_position_count"].item() == 1
    assert diagnostics["trace_all_accepted_anchor_count"].item() == 1
    assert diagnostics["rejected_k3_weighted_count"].item() == 0
    assert diagnostics["accepted_k3_weighted_count"].item() == 4


def test_trace_aligned_training_masks_post_rejection_suffix() -> None:
    torch.manual_seed(11)
    model = _small_dspark_training_model(
        block_size=3,
        l1_loss_alpha=0.5,
        trace_aligned_anchors=True,
        accepted_fkl_alpha=1.0,
        rejected_k3_alpha=1.0,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    loss_mask[:, 0] = 0.0
    hidden_states = [torch.randn(1, 7, 8), torch.randn(1, 7, 8)]
    target_last_hidden_states = torch.randn(1, 7, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
        trace_anchor_positions=torch.tensor([[0, 3]], dtype=torch.long),
        trace_anchor_mask=torch.tensor([[True, True]]),
        trace_accepted_counts=torch.tensor([[1, 0]], dtype=torch.long),
        trace_draft_counts=torch.tensor([[3, 3]], dtype=torch.long),
        trace_rejected_token_ids=torch.tensor([[12, 13]], dtype=torch.long),
        trace_draft_token_ids=torch.tensor(
            [[[2, 12, 14], [13, 15, 16]]], dtype=torch.long
        ),
        trace_draft_teacher_logprobs=torch.tensor(
            [[[-0.2, -2.0, -3.0], [-1.5, -2.5, -3.5]]],
            dtype=torch.float32,
        ),
        trace_draft_teacher_logprob_mask=torch.ones(
            1, 2, 3, dtype=torch.bool
        ),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["trace_anchor_count"].item() == 2
    assert diagnostics["trace_accepted_position_count"].item() == 1
    assert diagnostics["trace_first_rejection_count"].item() == 2
    assert diagnostics["trace_unverified_suffix_count"].item() == 3
    assert diagnostics["rejected_k3_weighted_count"].item() == pytest.approx(
        0.9 + 0.9**2 + 1.0 + 0.9 + 0.9**2
    )
    assert diagnostics["accepted_fkl_weighted_count"].item() > 0
    assert diagnostics["l1_weighted_token_count"].item() > 0
    assert diagnostics["ce_weighted_token_count"].item() == pytest.approx(3.0)
    expected_joint_loss = (
        diagnostics["response_objective_sum"]
        + diagnostics["rejected_objective_sum"]
    ) / diagnostics["objective_weighted_token_count"]
    assert loss.item() == pytest.approx(expected_joint_loss.item())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_dynamic_stream_weights_are_applied_to_joint_objective() -> None:
    torch.manual_seed(31)
    model = _small_dspark_training_model(
        block_size=2,
        trace_aligned_anchors=True,
        accepted_k3_alpha=1.0,
        rejected_k3_alpha=1.0,
        response_stream_weight=1.0,
        rejected_stream_weight=0.5,
        dynamic_accept_reject_weighting_enabled=True,
    )

    loss, *_rest, diagnostics = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        hidden_states_list=[torch.randn(1, 3, 8), torch.randn(1, 3, 8)],
        loss_mask=torch.ones(1, 3, dtype=torch.float32),
        lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(1, 3, 8),
        trace_anchor_positions=torch.tensor([[0]], dtype=torch.long),
        trace_anchor_mask=torch.tensor([[True]]),
        trace_accepted_counts=torch.tensor([[1]], dtype=torch.long),
        trace_draft_counts=torch.tensor([[2]], dtype=torch.long),
        trace_rejected_token_ids=torch.tensor([[12]], dtype=torch.long),
        trace_draft_token_ids=torch.tensor([[[2, 12]]], dtype=torch.long),
        trace_draft_teacher_logprobs=torch.tensor(
            [[[-0.2, -2.0]]], dtype=torch.float32
        ),
        trace_draft_teacher_logprob_mask=torch.ones(1, 1, 2, dtype=torch.bool),
    )

    response_count = diagnostics["ce_weighted_token_count"]
    rejected_count = diagnostics["rejected_k3_weighted_count"]
    total_count = response_count + rejected_count
    expected_response_multiplier = total_count / (2.0 * response_count)
    expected_rejected_multiplier = total_count / (2.0 * rejected_count)
    assert diagnostics["response_stream_weight_multiplier"].item() == pytest.approx(
        expected_response_multiplier.item()
    )
    assert diagnostics["rejected_stream_weight_multiplier"].item() == pytest.approx(
        expected_rejected_multiplier.item()
    )
    expected_weighted_sum = (
        diagnostics["response_objective_sum"]
        * diagnostics["response_stream_effective_weight"]
        + diagnostics["rejected_objective_sum"]
        * diagnostics["rejected_stream_effective_weight"]
    )
    assert loss.item() == pytest.approx(
        (expected_weighted_sum / diagnostics["objective_weighted_token_count"]).item()
    )


def test_trace_aligned_k3_dump_can_reconstruct_each_sampled_loss() -> None:
    torch.manual_seed(23)
    model = _small_dspark_training_model(
        block_size=3,
        trace_aligned_anchors=True,
        accepted_k3_alpha=1.0,
        rejected_k3_alpha=1.0,
        k3_distribution_bins=16,
        dump_k3_logprobs=True,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    hidden_states = [torch.randn(1, 7, 8), torch.randn(1, 7, 8)]
    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32),
        lm_head_weight=torch.randn(32, 8),
        target_last_hidden_states=torch.randn(1, 7, 8),
        trace_anchor_positions=torch.tensor([[0, 3]], dtype=torch.long),
        trace_anchor_mask=torch.tensor([[True, True]]),
        trace_accepted_counts=torch.tensor([[1, 0]], dtype=torch.long),
        trace_draft_counts=torch.tensor([[3, 3]], dtype=torch.long),
        trace_rejected_token_ids=torch.tensor([[12, 13]], dtype=torch.long),
        trace_draft_token_ids=torch.tensor(
            [[[2, 12, 14], [13, 15, 16]]], dtype=torch.long
        ),
        trace_draft_teacher_logprobs=torch.tensor(
            [[[-0.2, -2.0, -3.0], [-1.5, -2.5, -3.5]]], dtype=torch.float32
        ),
        trace_draft_teacher_logprob_mask=torch.ones(1, 2, 3, dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    streams = diagnostics["k3_logprob_dump"]
    assert streams["accepted"]["token_ids"].numel() > 0
    assert streams["rejected"]["token_ids"].numel() > 0
    for stream_name in ("accepted", "rejected"):
        stream = streams[stream_name]
        token_count = int(stream["token_ids"].numel())
        for key in (
            "prev_token_ids",
            "target_logprobs",
            "draft_logprobs",
            "k3",
            "raw_k3",
            "weights",
            "clamped",
            "linearized",
            "sample_indices",
            "anchor_indices",
            "anchor_positions",
            "draft_positions",
            "trace_accepted_counts",
            "trace_draft_counts",
        ):
            assert int(stream[key].numel()) == token_count
        delta = (stream["target_logprobs"] - stream["draft_logprobs"]).clamp(
            min=-model.rejected_log_ratio_clamp,
            max=model.rejected_log_ratio_clamp,
        )
        reconstructed = (torch.exp(delta) - delta - 1.0).clamp(
            max=model.rejected_loss_max_clamp
        )
        assert torch.allclose(stream["k3"], reconstructed)
        assert torch.allclose(stream["raw_k3"], torch.exp(delta) - delta - 1.0)
        assert not torch.any(stream["linearized"])
        assert torch.isfinite(stream["target_logprobs"]).all()
        assert torch.isfinite(stream["draft_logprobs"]).all()
        assert diagnostics[f"{stream_name}_k3_histogram"].sum().item() == pytest.approx(
            stream["weights"].sum().item()
        )


def test_rejected_sampled_k3_penalizes_excess_rejected_token_probability() -> None:
    model = _small_dspark_training_model(block_size=2)
    with torch.no_grad():
        for parameter in model.draft_model.markov_head.parameters():
            parameter.zero_()
    draft_hidden = torch.zeros(1, 8)
    draft_hidden[0, 0] = 2.0
    draft_hidden.requires_grad_(True)
    lm_head_weight = torch.zeros(32, 8)
    lm_head_weight[3, 0] = 1.0

    k3_sum, k3_den, q_sum, p_sum, delta_sum, clamped = (
        model._compute_rejected_sampled_k3(
        draft_hidden=draft_hidden,
        prev_token_ids=torch.tensor([1]),
        rejected_token_ids=torch.tensor([3]),
        teacher_logprobs=torch.tensor([-4.0]),
        weights=torch.ones(1),
        lm_head_weight=lm_head_weight,
        )
    )
    k3_sum.backward()

    assert k3_den.item() == pytest.approx(1.0)
    assert k3_sum.item() > 0
    assert q_sum.item() > p_sum.item()
    assert delta_sum.item() < 0
    assert clamped.item() == 0
    expected_k3 = math.exp(delta_sum.item()) - delta_sum.item() - 1.0
    assert k3_sum.item() == pytest.approx(expected_k3)
    assert draft_hidden.grad is not None
    assert draft_hidden.grad.abs().sum().item() > 0


def test_default_k3_policy_preserves_symmetric_hard_clamp() -> None:
    model = _small_dspark_training_model(block_size=2)
    delta = torch.tensor([-12.0, 4.0], requires_grad=True)

    k3, raw_k3, legacy_clamped, linearized = model._apply_k3_tail_policy(delta)
    k3.sum().backward()

    assert torch.all(raw_k3 > model.rejected_loss_max_clamp)
    assert torch.all(k3 == model.rejected_loss_max_clamp)
    assert torch.all(legacy_clamped)
    assert not torch.any(linearized)
    assert torch.equal(delta.grad, torch.zeros_like(delta))


def test_asymmetric_k3_linearizes_only_positive_tail() -> None:
    threshold = 2.0
    model = _small_dspark_training_model(
        block_size=2,
        asymmetric_k3_enabled=True,
        asymmetric_k3_positive_threshold=threshold,
    )
    delta = torch.tensor([-12.0, 1.0, 4.0], requires_grad=True)

    k3, raw_k3, legacy_clamped, linearized = model._apply_k3_tail_policy(delta)
    k3.sum().backward()

    exp_threshold = math.exp(threshold)
    expected_positive_tail = (
        exp_threshold
        - threshold
        - 1.0
        + (exp_threshold - 1.0) * (4.0 - threshold)
    )
    assert k3[0].item() == pytest.approx(raw_k3[0].item())
    assert k3[0].item() > model.rejected_loss_max_clamp
    assert k3[1].item() == pytest.approx(raw_k3[1].item())
    assert k3[2].item() == pytest.approx(expected_positive_tail)
    assert legacy_clamped.tolist() == [True, False, True]
    assert linearized.tolist() == [False, False, True]
    assert delta.grad.tolist() == pytest.approx(
        [math.exp(-12.0) - 1.0, math.e - 1.0, exp_threshold - 1.0]
    )


def test_rejected_sampled_k3_uses_asymmetric_tail_policy() -> None:
    threshold = 2.0
    model = _small_dspark_training_model(
        block_size=2,
        asymmetric_k3_enabled=True,
        asymmetric_k3_positive_threshold=threshold,
    )
    with torch.no_grad():
        for parameter in model.draft_model.markov_head.parameters():
            parameter.zero_()

    outputs = model._compute_rejected_sampled_k3(
        draft_hidden=torch.zeros(1, 8),
        prev_token_ids=torch.tensor([1]),
        rejected_token_ids=torch.tensor([3]),
        teacher_logprobs=torch.tensor([-0.01]),
        weights=torch.ones(1),
        lm_head_weight=torch.zeros(32, 8),
        return_diagnostics=True,
    )

    k3_sum, k3_den, _, _, delta_sum, legacy_clamped, distribution = outputs
    delta = delta_sum.item()
    exp_threshold = math.exp(threshold)
    expected = (
        exp_threshold
        - threshold
        - 1.0
        + (exp_threshold - 1.0) * (delta - threshold)
    )
    assert delta > threshold
    assert k3_den.item() == pytest.approx(1.0)
    assert k3_sum.item() == pytest.approx(expected)
    assert legacy_clamped.item() == pytest.approx(1.0)
    assert distribution["linearized_count"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("delta", [-30.0, -1000.0])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_rejected_negative_tail_restores_corrective_gradient(
    monkeypatch, delta: float, asymmetric: bool,
) -> None:
    model = _small_dspark_training_model(
        accepted_asymmetric_k3_enabled=False,
        rejected_asymmetric_k3_enabled=asymmetric,
    )
    logits = torch.zeros(1, 32, requires_grad=True)
    monkeypatch.setattr(model, "_full_vocab_logits", lambda *args: logits)
    log_q = -math.log(32)
    loss, denominator, *_ = model._compute_rejected_sampled_k3(
        draft_hidden=torch.zeros(1, 8),
        prev_token_ids=torch.tensor([1]),
        rejected_token_ids=torch.tensor([3]),
        teacher_logprobs=torch.tensor([log_q + delta]),
        weights=torch.ones(1),
        lm_head_weight=torch.zeros(32, 8),
    )
    loss.backward()
    assert denominator.item() == 1
    assert torch.isfinite(logits.grad).all()
    if asymmetric:
        assert loss.item() == pytest.approx(-delta - 1, abs=1e-4)
        expected = torch.full_like(logits, -1 / 32)
        expected[0, 3] += 1
        # Gradient descent must suppress the rejected token, not reinforce it.
        torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-6)
    else:
        assert loss.item() == pytest.approx(10)
        assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("delta", [4.0, 30.0])
def test_rejected_positive_tail_keeps_linearization_and_upper_cap(monkeypatch, delta):
    model = _small_dspark_training_model(rejected_asymmetric_k3_enabled=True)
    # The candidate is unlikely under q, allowing positive log(p/q) with valid p.
    logits = torch.zeros(1, 32)
    logits[0, 3] = -40
    logits.requires_grad_()
    monkeypatch.setattr(model, "_full_vocab_logits", lambda *args: logits)
    log_q = torch.log_softmax(logits.detach(), dim=-1)[0, 3]
    outputs = model._compute_rejected_sampled_k3(
        draft_hidden=torch.zeros(1, 8), prev_token_ids=torch.tensor([1]),
        rejected_token_ids=torch.tensor([3]), teacher_logprobs=(log_q + delta).reshape(1),
        weights=torch.ones(1), lm_head_weight=torch.zeros(32, 8),
    )
    loss = outputs[0]
    loss.backward()
    capped_delta = min(delta, 20)
    slope = math.exp(2) - 1
    assert loss.item() == pytest.approx(math.exp(2) - 3 + slope * (capped_delta - 2))
    if delta > 20:
        assert torch.count_nonzero(logits.grad) == 0
    else:
        expected = slope * torch.softmax(logits.detach(), dim=-1)
        expected[0, 3] -= slope
        torch.testing.assert_close(logits.grad, expected, atol=1e-5, rtol=1e-5)


def test_asymmetric_k3_stream_switches_can_select_rejected_only() -> None:
    model = _small_dspark_training_model(
        block_size=2,
        asymmetric_k3_enabled=True,
        accepted_asymmetric_k3_enabled=False,
        rejected_asymmetric_k3_enabled=True,
    )
    delta = torch.tensor([-12.0, 4.0])

    accepted_k3, accepted_raw, _, accepted_linearized = (
        model._apply_k3_tail_policy(
            delta,
            asymmetric_enabled=model.accepted_asymmetric_k3_enabled,
        )
    )
    rejected_k3, rejected_raw, _, rejected_linearized = (
        model._apply_k3_tail_policy(
            delta,
            asymmetric_enabled=model.rejected_asymmetric_k3_enabled,
        )
    )

    assert torch.all(accepted_k3 == model.rejected_loss_max_clamp)
    assert torch.all(accepted_raw > model.rejected_loss_max_clamp)
    assert not torch.any(accepted_linearized)
    assert rejected_k3[0].item() == pytest.approx(rejected_raw[0].item())
    assert rejected_k3[1].item() < rejected_raw[1].item()
    assert rejected_linearized.tolist() == [False, True]


def test_dynamic_accept_reject_weighting_balances_base_stream_mass() -> None:
    model = _small_dspark_training_model(
        block_size=2,
        response_stream_weight=1.0,
        rejected_stream_weight=0.5,
        dynamic_accept_reject_weighting_enabled=True,
    )

    response_weight, rejected_weight, response_multiplier, rejected_multiplier = (
        model._effective_stream_weights(
            response_token_count=torch.tensor(8.0),
            rejected_token_count=torch.tensor(2.0),
            response_stream_enabled=True,
            rejected_stream_enabled=True,
        )
    )

    assert response_multiplier.item() == pytest.approx(10.0 / 16.0)
    assert rejected_multiplier.item() == pytest.approx(10.0 / 4.0)
    assert response_weight.item() == pytest.approx(10.0 / 16.0)
    assert rejected_weight.item() == pytest.approx(1.25)
    response_mass = response_weight.item() * 8.0
    rejected_mass = rejected_weight.item() * 2.0
    assert rejected_mass / response_mass == pytest.approx(0.5)


def test_dynamic_accept_reject_weighting_is_bounded_and_optional() -> None:
    enabled_model = _small_dspark_training_model(
        block_size=2,
        response_stream_weight=1.0,
        rejected_stream_weight=0.5,
        dynamic_accept_reject_weighting_enabled=True,
        dynamic_stream_weight_min_multiplier=0.5,
        dynamic_stream_weight_max_multiplier=2.0,
    )
    weights = enabled_model._effective_stream_weights(
        response_token_count=torch.tensor(100.0),
        rejected_token_count=torch.tensor(1.0),
        response_stream_enabled=True,
        rejected_stream_enabled=True,
    )
    assert weights[2].item() == pytest.approx(0.505)
    assert weights[3].item() == pytest.approx(2.0)

    disabled_model = _small_dspark_training_model(
        block_size=2,
        response_stream_weight=1.0,
        rejected_stream_weight=0.5,
    )
    disabled_weights = disabled_model._effective_stream_weights(
        response_token_count=torch.tensor(100.0),
        rejected_token_count=torch.tensor(1.0),
        response_stream_enabled=True,
        rejected_stream_enabled=True,
    )
    assert [value.item() for value in disabled_weights] == pytest.approx(
        [1.0, 0.5, 1.0, 1.0]
    )


def test_accepted_forward_kl_matches_binary_teacher_distribution() -> None:
    model = _small_dspark_training_model(block_size=2)
    with torch.no_grad():
        for parameter in model.draft_model.markov_head.parameters():
            parameter.zero_()
    draft_hidden = torch.zeros(1, 8)
    draft_hidden[0, 0] = -1.0
    draft_hidden.requires_grad_(True)
    target_hidden = torch.zeros(1, 8)
    target_hidden[0, 0] = 1.0
    lm_head_weight = torch.zeros(32, 8)
    lm_head_weight[3, 0] = 1.0

    fkl_sum, fkl_den, q_sum, p_sum, _ = model._compute_accepted_distribution_losses(
        active_hidden=draft_hidden,
        active_prev_tokens=torch.tensor([1]),
        active_target_hidden=target_hidden,
        active_target_ids=torch.tensor([3]),
        active_weights=torch.ones(1),
        lm_head_weight=lm_head_weight,
        compute_l1=False,
    )[:5]
    fkl_sum.backward()

    q = q_sum.item()
    p = p_sum.item()
    expected = p * math.log(p / q) + (1.0 - p) * math.log(
        (1.0 - p) / (1.0 - q)
    )
    assert fkl_den.item() == pytest.approx(1.0)
    assert fkl_sum.item() == pytest.approx(expected)
    assert p > q
    assert draft_hidden.grad is not None
    assert draft_hidden.grad.abs().sum().item() > 0


def test_accepted_sampled_k3_matches_draft_opd_estimator() -> None:
    model = _small_dspark_training_model(block_size=2, accepted_k3_alpha=1.0)
    with torch.no_grad():
        for parameter in model.draft_model.markov_head.parameters():
            parameter.zero_()
    draft_hidden = torch.zeros(1, 8)
    draft_hidden[0, 0] = 1.0
    draft_hidden.requires_grad_(True)
    target_hidden = torch.zeros(1, 8)
    target_hidden[0, 0] = -1.0
    lm_head_weight = torch.zeros(32, 8)
    lm_head_weight[3, 0] = 1.0

    outputs = model._compute_accepted_distribution_losses(
        active_hidden=draft_hidden,
        active_prev_tokens=torch.tensor([1]),
        active_target_hidden=target_hidden,
        active_target_ids=torch.tensor([3]),
        active_weights=torch.ones(1),
        lm_head_weight=lm_head_weight,
    )
    q_sum, p_sum = outputs[2], outputs[3]
    k3_sum, k3_den, clamped = outputs[7], outputs[8], outputs[9]
    k3_sum.backward()

    delta = math.log(p_sum.item()) - math.log(q_sum.item())
    expected_k3 = math.exp(delta) - delta - 1.0
    assert k3_den.item() == pytest.approx(1.0)
    assert k3_sum.item() == pytest.approx(expected_k3)
    assert clamped.item() == 0
    assert draft_hidden.grad is not None
    assert draft_hidden.grad.abs().sum().item() > 0


def test_dspark_default_loss_weights_match_draft_opd():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
    )
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(config))

    assert config.ce_loss_alpha == pytest.approx(0.1)
    assert config.l1_loss_alpha == pytest.approx(0.9)
    assert model.ce_loss_alpha == pytest.approx(0.0)
    assert model.first_rejection_ce_alpha == pytest.approx(0.0)
    assert model.l1_loss_alpha == pytest.approx(0.0)
    assert model.accepted_k3_alpha == pytest.approx(1.0)
    assert model.accepted_fkl_alpha == pytest.approx(0.0)
    assert model.rejected_k3_alpha == pytest.approx(1.0)
    assert model.rejected_aux_alpha == pytest.approx(0.0)
    assert model.rejected_prefix_length == 0
    assert model.response_stream_weight == pytest.approx(1.0)
    assert model.rejected_stream_weight == pytest.approx(1.0)
    assert model.rejected_position_decay == pytest.approx(0.9)
    assert model.topk_support == "student"
    assert model.rejected_first_weight_multiplier == pytest.approx(1.0)
    assert model.rejected_suffix_weight_multiplier == pytest.approx(1.0)
    assert model.accepted_asymmetric_k3_enabled is False
    assert model.rejected_asymmetric_k3_enabled is False
    assert model.dynamic_accept_reject_weighting_enabled is False
    assert model.l1_chunk_size == 0


def test_chunked_ce_matches_full_vocab_loss_and_hidden_gradient():
    torch.manual_seed(7)
    model = _small_dspark_training_model(loss_mode="chunked_ce")
    model.ce_token_chunk_size = 2
    full_hidden = torch.randn(7, 8, requires_grad=True)
    chunked_hidden = full_hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 32, (7,), dtype=torch.long)
    prev_tokens = torch.randint(0, 32, (7,), dtype=torch.long)
    lm_head_weight = torch.randn(32, 8)

    full_logits = model._full_vocab_logits(full_hidden, lm_head_weight, prev_tokens)
    full_loss = torch.nn.functional.cross_entropy(
        full_logits.float(), targets, reduction="none"
    )
    full_gradient = torch.autograd.grad(full_loss.sum(), full_hidden)[0]

    chunked_loss = model._chunked_exact_cross_entropy(
        active_hidden=chunked_hidden,
        active_targets=targets,
        lm_head_weight=lm_head_weight,
        active_prev_tokens=prev_tokens,
    )
    chunked_gradient = torch.autograd.grad(chunked_loss.sum(), chunked_hidden)[0]

    assert torch.allclose(chunked_loss, full_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_gradient, full_gradient, atol=1e-6, rtol=1e-6)


def test_restricted_ce_compatibility_alias_uses_exact_chunked_ce():
    model = _small_dspark_training_model(loss_mode="restricted_ce")

    assert model.loss_mode == "chunked_ce"


def test_dspark_untrained_confidence_head_is_kept_but_excluded_from_optimizer():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        markov_rank=4,
        enable_confidence_head=True,
    )
    model = DSparkTrainingModel(
        draft_model=DSparkDraftModel(config),
        confidence_head_alpha=0.0,
    )

    confidence_head = model.draft_model.confidence_head
    assert confidence_head is not None
    assert "draft_model.confidence_head.proj.weight" in model.state_dict()
    assert all(
        not parameter.requires_grad for parameter in confidence_head.parameters()
    )
    optimizer_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert optimizer_parameter_ids.isdisjoint(
        id(parameter) for parameter in confidence_head.parameters()
    )


def test_dspark_label_and_prev_token_alignment():
    model = _small_dspark_training_model(block_size=4)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    anchor_positions = torch.tensor([[2]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])

    target_ids, prev_token_ids, eval_mask, label_indices = model._build_label_tensors(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )

    assert label_indices.tolist() == [[[3, 4, 5, 6]]]
    assert target_ids.tolist() == [[[13, 14, 15, 16]]]
    assert prev_token_ids.tolist() == [[[12, 13, 14, 15]]]
    assert eval_mask.tolist() == [[[True, True, True, True]]]


def test_dspark_dense_attention_mask_matches_deepspec_block_contract():
    anchor_positions = torch.tensor([[2, 4]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, True]])

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=2,
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 4, 10)
    assert torch.nonzero(mask[0, 0, 0], as_tuple=False).flatten().tolist() == [
        0,
        1,
        6,
        7,
    ]
    assert torch.nonzero(mask[0, 0, 1], as_tuple=False).flatten().tolist() == [
        0,
        1,
        6,
        7,
    ]
    assert torch.nonzero(mask[0, 0, 2], as_tuple=False).flatten().tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
    ]
    assert torch.nonzero(mask[0, 0, 3], as_tuple=False).flatten().tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
    ]


def test_dspark_swa_crops_only_context_and_keeps_full_draft_block():
    anchor_positions = torch.tensor([[4]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=3,
        context_window=2,
    )

    expected = [2, 3, 6, 7, 8]
    for query_index in range(3):
        assert (
            torch.nonzero(mask[0, 0, query_index], as_tuple=False)
            .flatten()
            .tolist()
            == expected
        )


def test_dspark_sparse_global_adds_selected_context_outside_recent_window():
    anchor_positions = torch.tensor([[5]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])
    global_context_mask = torch.tensor(
        [[True, False, False, False, False, False]], dtype=torch.bool
    )

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=2,
        context_window=2,
        global_context_mask=global_context_mask,
    )

    expected = [0, 3, 4, 6, 7]
    assert torch.nonzero(mask[0, 0, 0], as_tuple=False).flatten().tolist() == expected
    assert torch.nonzero(mask[0, 0, 1], as_tuple=False).flatten().tolist() == expected


def test_dspark_position_ids_preserve_absolute_crop_offsets():
    model = _small_dspark_training_model(block_size=3)
    anchors = torch.tensor([[1]], dtype=torch.long)
    absolute_positions = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)

    context_positions, draft_positions = model._create_position_ids(
        anchors, seq_len=4, context_position_ids=absolute_positions
    )

    assert torch.equal(context_positions, absolute_positions)
    assert draft_positions.tolist() == [[101, 102, 103]]


def test_dspark_layers_accept_distinct_masks_with_absolute_rope_positions():
    model = _small_dspark_training_model(block_size=2, num_hidden_layers=2)
    draft_model = model.draft_model
    context_feature = torch.randn(1, 4, 8)
    noise_embedding = torch.randn(1, 2, 8)
    context_positions = torch.tensor([[100, 101, 102, 103]], dtype=torch.long)
    draft_positions = torch.tensor([[102, 103]], dtype=torch.long)
    local_mask = torch.tensor(
        [[[[False, True, True, False, True, True],
           [False, True, True, False, True, True]]]],
        dtype=torch.bool,
    )
    global_mask = torch.ones((1, 1, 2, 6), dtype=torch.bool)

    output = draft_model(
        draft_input_ids=None,
        context_feature=context_feature,
        draft_position_ids=draft_positions,
        context_position_ids=context_positions,
        dense_attention_mask=[local_mask, global_mask],
        noise_embedding=noise_embedding,
    )

    assert output.shape == (1, 2, 8)
    assert torch.isfinite(output).all()


def test_dspark_sparse_global_preprocessing_preserves_positions_and_pages():
    backend = object.__new__(DSparkTrainerBackend)
    backend.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "dspark_attention_mode": "sparse_global",
                    "dspark_max_window": 8,
                    "dspark_block_size": 2,
                    "dspark_num_anchors": 2,
                    "dspark_global_prefix_tokens": 102,
                    "dspark_global_prompt_tokens": 2,
                    "dspark_actor_guided_page_size": 2,
                    "dspark_actor_guided_page_budget_tokens": 2,
                    "dspark_actor_guided_page_starts_field": "selected_pages",
                }
            )
        )
    )
    model_config = SimpleNamespace(
        hidden_size=8,
        target_hidden_size=8,
        num_context_layers=2,
        pad_token_id=0,
    )
    item = {
        "input_ids": torch.arange(8),
        "hidden_states": torch.randn(8, 16),
        "hidden_states_layout": "dflash_aux",
        "hidden_positions": torch.arange(100, 108),
        "loss_mask": torch.ones(8),
        "_verl_prompt_len": 105,
        "selected_pages": [106],
    }

    result = backend.preprocess_individual_items([item], torch.device("cpu"), model_config)

    assert result["position_ids"][0].tolist() == list(range(100, 108))
    assert result["global_context_masks"][0].tolist() == [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
    ]


def test_dspark_dense_attention_mask_keeps_dummy_rows_finite_safe():
    anchor_positions = torch.tensor([[2, 0]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, False]])

    mask = create_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=6,
        block_size=2,
    )

    assert torch.nonzero(mask[0, 0, 2], as_tuple=False).flatten().tolist() == [8]
    assert torch.nonzero(mask[0, 0, 3], as_tuple=False).flatten().tolist() == [9]


def test_dspark_first_position_is_masked_when_first_target_invalid():
    model = _small_dspark_training_model(block_size=4)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    loss_mask[0, 3] = 0.0
    anchor_positions = torch.tensor([[2]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True]])

    _, _, eval_mask, _ = model._build_label_tensors(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )

    assert eval_mask.tolist() == [[[False, False, False, False]]]


def test_dspark_markov_rank_zero_keeps_base_logits():
    config = DSparkConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=2,
        markov_rank=0,
    )
    draft_model = DSparkDraftModel(config)
    base_logits = torch.randn(1, 2, 4, 32)
    prev_ids = torch.zeros(1, 2, 4, dtype=torch.long)
    hidden = torch.randn(1, 2, 4, 8)

    corrected = draft_model.apply_markov_logits(
        base_logits,
        prev_token_ids=prev_ids,
        draft_hidden=hidden,
    )

    assert torch.equal(corrected, base_logits)


def test_dspark_markov_bias_changes_logits():
    model = _small_dspark_training_model(block_size=4).draft_model
    assert model.markov_head is not None
    base_logits = torch.zeros(1, 1, 4, 32)
    prev_ids = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long)
    hidden = torch.zeros(1, 1, 4, 8)

    with torch.no_grad():
        model.markov_head.markov_w1.weight.fill_(0.5)
        model.markov_head.markov_w2.weight.fill_(0.25)

    corrected = model.apply_markov_logits(
        base_logits,
        prev_token_ids=prev_ids,
        draft_hidden=hidden,
    )

    assert corrected.abs().sum().item() > 0
    assert not torch.equal(corrected, base_logits)


def test_dspark_l1_loss_uses_target_last_hidden_states():
    model = _small_dspark_training_model(block_size=2, l1_loss_alpha=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_rest, diagnostics = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )

    assert torch.isfinite(loss)
    assert diagnostics["ce_weighted_token_count"].item() > 0
    assert diagnostics["ce_loss_sum"].item() >= 0
    assert diagnostics["l1_weighted_token_count"].item() > 0
    assert diagnostics["l1_loss_sum"].item() >= 0


@pytest.mark.parametrize(
    ("loss_mode", "expects_reused_log_probs"),
    [("full_vocab", True), ("restricted_ce", False)],
)
def test_dspark_l1_reuses_only_full_vocab_ce_log_probs(
    monkeypatch, loss_mode, expects_reused_log_probs
):
    model = _small_dspark_training_model(
        block_size=2,
        l1_loss_alpha=0.5,
        loss_mode=loss_mode,
    )
    captured_log_probs = []
    original_compute_l1 = model._compute_l1_loss_for_active

    def capture_compute_l1(**kwargs):
        captured_log_probs.append(kwargs.get("active_draft_log_probs"))
        return original_compute_l1(**kwargs)

    monkeypatch.setattr(model, "_compute_l1_loss_for_active", capture_compute_l1)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    hidden_states = [torch.randn(1, 5, 8), torch.randn(1, 5, 8)]
    target_last_hidden_states = torch.randn(1, 5, 8)
    lm_head_weight = torch.randn(32, 8)

    loss, *_ = model(
        input_ids=input_ids,
        hidden_states_list=hidden_states,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        target_last_hidden_states=target_last_hidden_states,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert len(captured_log_probs) == 1
    assert (captured_log_probs[0] is not None) is expects_reused_log_probs
    if captured_log_probs[0] is not None:
        probability_mass = captured_log_probs[0].exp().sum(dim=-1)
        assert torch.allclose(
            probability_mass, torch.ones_like(probability_mass), atol=1e-5, rtol=1e-5
        )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
