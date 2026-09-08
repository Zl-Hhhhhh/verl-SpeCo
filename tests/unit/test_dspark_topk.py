import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.integration.dspark_topk_runtime import (
    cache_topk_proposal,
    capture_topk_proposal,
    topk_payload_by_request,
    verified_topk_payload,
)
from verl_speco.trainer.dspark_topk import (
    pack_trace_topk,
    student_support_reverse_kl,
    validate_topk_modes,
)


def test_full_support_equals_exact_reverse_kl_and_stops_teacher_gradient():
    torch.manual_seed(17)
    draft = torch.randn(3, 16, requires_grad=True)
    teacher = torch.randn(3, 16, requires_grad=True)
    log_q, log_p = draft.log_softmax(-1), teacher.log_softmax(-1)
    weights = torch.tensor([1., .8, .64])
    actual = student_support_reverse_kl(log_q, log_p, weights, topk=16)
    expected = ((log_q.exp() * (log_q - log_p.detach())).sum(-1) * weights).sum()
    assert torch.allclose(actual["loss_sum"], expected, atol=1e-6)
    actual["loss_sum"].backward()
    assert draft.grad.abs().sum() > 0
    assert teacher.grad is None
    assert actual["student_mass_sum"].item() == pytest.approx(weights.sum().item())


def test_student_not_teacher_support_and_outside_mass_blind_spot():
    log_q = torch.tensor([[.54, .36, .10]]).log()
    log_p = torch.tensor([[.06, .04, .90]]).log()
    stats = student_support_reverse_kl(log_q, log_p, torch.ones(1), topk=2)
    assert stats["loss_sum"].item() == pytest.approx(0., abs=1e-6)
    assert stats["student_mass_sum"].item() == pytest.approx(.9)
    assert stats["teacher_mass_sum"].item() == pytest.approx(.1)


def test_rejected_support_is_fixed_and_gradient_survives_negative_k3_clamp():
    logits = torch.tensor([[3., 2., 10.]], requires_grad=True)
    teacher = torch.tensor([[-50., -1.]], requires_grad=True)
    stats = student_support_reverse_kl(
        logits.log_softmax(-1), teacher, torch.ones(1), topk=2,
        candidate_ids=torch.tensor([[0, 1]]),
    )
    stats["loss_sum"].backward()
    assert logits.grad[0, 0] > 1
    assert logits.grad[0, 1] < -1
    assert abs(logits.grad[0, 2]) < 1e-5
    assert teacher.grad is None


@pytest.mark.parametrize("teacher_values", [
    [4., 3., 2., 1.], [1., 2., 3., 4.], [1., 3., 4., 2.],
])
@pytest.mark.parametrize("explicit", [False, True])
def test_union_loss_and_gradient_match_exact_deduplicated_support(teacher_values, explicit):
    draft = torch.tensor([[4., 3., 2., 1.]], requires_grad=True)
    teacher = torch.tensor([teacher_values], requires_grad=True)
    log_q, log_p = draft.log_softmax(-1), teacher.log_softmax(-1)
    student_ids = log_q.detach().topk(2, dim=-1).indices
    teacher_ids = log_p.detach().topk(2, dim=-1).indices
    ids = torch.cat([student_ids, teacher_ids], dim=-1)
    unique_ids = ids[0].unique()
    actual = student_support_reverse_kl(
        log_q, log_p.gather(-1, ids) if explicit else log_p, torch.tensor([0.4]),
        topk=2, candidate_ids=ids if explicit else None, support="union",
    )
    ref_log_q = draft[:, unique_ids].log_softmax(-1)
    ref_log_p = teacher.detach()[:, unique_ids].log_softmax(-1)
    expected = 0.4 * (ref_log_q.exp() * (ref_log_q - ref_log_p)).sum()
    actual_grad = torch.autograd.grad(actual["loss_sum"], draft, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, draft)[0]
    assert torch.allclose(actual["loss_sum"], expected, atol=1e-6)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6)
    assert torch.isfinite(actual_grad).all()
    assert actual["teacher_mass_sum"].item() == pytest.approx(0.4 * log_p[:, unique_ids].exp().sum().item())
    assert actual["student_mass_sum"].item() == pytest.approx(0.4 * log_q[:, unique_ids].exp().sum().item())
    assert actual["support_size_sum"].item() == pytest.approx(0.4 * unique_ids.numel())
    actual["loss_sum"].backward()
    assert teacher.grad is None


def test_union_supplies_gradient_to_teacher_token_missing_from_student_support():
    draft = torch.tensor([[.54, .36, .10]]).log().requires_grad_(True)
    teacher = torch.tensor([[.06, .04, .90]]).log()
    actual = student_support_reverse_kl(
        draft.log_softmax(-1), teacher, torch.ones(1), topk=2, support="union",
    )
    actual["loss_sum"].backward()
    assert actual["loss_sum"].item() > 0
    assert draft.grad[0, 2] < 0
    assert actual["teacher_mass_sum"].item() == pytest.approx(1.)
    assert actual["support_size_sum"].item() == pytest.approx(3.)


def test_union_rejects_student_only_or_corrupt_metadata():
    with pytest.raises(ValueError, match="candidate shape"):
        student_support_reverse_kl(
            torch.zeros(1, 4), torch.zeros(1, 2), torch.ones(1),
            topk=2, support="union", candidate_ids=torch.tensor([[0, 1]]),
        )
    with pytest.raises(ValueError, match="unique"):
        student_support_reverse_kl(
            torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1),
            topk=2, support="union", candidate_ids=torch.tensor([[0, 0, 2, 3]]),
        )
    with pytest.raises(ValueError, match="topk_support"):
        validate_topk_modes("topk", "topk", 16, True, "invalid")
    with pytest.raises(ValueError, match="expected"):
        pack_trace_topk([
            {"draft_count": 1, "draft_topk_ids": [[0, 1]], "draft_topk_teacher_logprobs": [[-1., -2.]]},
        ], block_size=2, topk=2, device="cpu", support="union")


@pytest.mark.parametrize("ids", [[0, -1], [0, 3], [0, 0]])
def test_invalid_candidates_fail_closed(ids):
    with pytest.raises(ValueError):
        student_support_reverse_kl(
            torch.zeros(1, 3).log_softmax(-1), torch.zeros(1, 2), torch.ones(1),
            topk=2, candidate_ids=torch.tensor([ids]),
        )


def test_modes_and_trace_packing():
    validate_topk_modes("topk", "k3", 16, False)
    with pytest.raises(ValueError, match="trace_aligned"):
        validate_topk_modes("k3", "topk", 16, False)
    with pytest.raises(ValueError, match="at least 2"):
        validate_topk_modes("k3", "k3", 1, True)
    rows = [
        {"draft_count": 2, "draft_topk_ids": [[1, 2], [3, 4]],
         "draft_topk_teacher_logprobs": [[-1., -2.], [-3., -4.]]},
        {"draft_count": 1, "bonus_response_anchor": True},
    ]
    packed = pack_trace_topk(rows, block_size=3, topk=2, device="cpu")
    assert packed["trace_draft_topk_ids"].shape == (2, 3, 2)
    assert packed["trace_draft_topk_ids"][0, 1].tolist() == [3, 4]
    assert (packed["trace_draft_topk_ids"][1] == -1).all()


def test_runtime_topk_config_and_trace_transport_are_optional():
    from verl_speco.integration.vllm_runtime import (
        _build_vllm_speculative_trace_events,
        _configure_vllm_ascend_dspark_swa_window,
    )

    drafter = {"speculative_algorithm": "DSPARK", "training": {"dspark_trace_aligned_anchors": True}}
    engine = {}
    _configure_vllm_ascend_dspark_swa_window(engine, drafter, {})
    assert engine == {}
    drafter["training"]["dspark_rejected_loss"] = "topk"
    _configure_vllm_ascend_dspark_swa_window(engine, drafter, {})
    assert engine["additional_config"]["speco_dspark_rejected_topk"] == 16
    request = SimpleNamespace(_output_token_ids=[], is_finished=lambda: False)
    scheduler = SimpleNamespace(requests={"r": request})
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"r": [1, 2]})
    topk = {"draft_topk_ids": [[1, 0], [2, 3]], "draft_topk_teacher_logprobs": [[-1., -2.], [-3., -4.]]}
    output = SimpleNamespace(
        sampled_token_ids=[[3]], req_id_to_index={"r": 0},
        _speco_verified_draft_topk_by_request={"r": topk},
    )
    trace = _build_vllm_speculative_trace_events(scheduler, scheduled, output)["r"][0]
    assert trace["draft_topk_ids"] == topk["draft_topk_ids"]
    assert trace["draft_topk_teacher_logprobs"] == topk["draft_topk_teacher_logprobs"]


@pytest.mark.parametrize("initial", [{}, {"additional_config": None}, {"additional_config": {"other": 1}}])
@pytest.mark.parametrize("support", ["student", "union"])
def test_runtime_topk_survives_omegaconf_assignment(initial, support):
    from omegaconf import OmegaConf
    from verl_speco.integration.vllm_runtime import _configure_vllm_ascend_dspark_swa_window

    engine = OmegaConf.create(initial)
    OmegaConf.set_struct(engine, True)
    drafter = {
        "speculative_algorithm": "DSPARK",
        "training": {"dspark_rejected_loss": "topk", "dspark_topk": 16, "dspark_trace_aligned_anchors": True, "dspark_topk_support": support},
    }
    _configure_vllm_ascend_dspark_swa_window(engine, drafter, {})
    assert engine.additional_config.speco_dspark_rejected_topk == 16
    assert engine.additional_config.speco_dspark_topk_support == support
    if initial.get("additional_config"):
        assert engine.additional_config.other == 1


def _config(k=2, support="student"):
    return SimpleNamespace(
        additional_config={"speco_dspark_rejected_topk": k, "speco_dspark_topk_support": support},
        speculative_config=SimpleNamespace(disable_padded_drafter_batch=False),
    )


class _FakeModel:
    def compute_logits(self):
        return torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])


def test_proposer_captures_post_markov_support_without_changing_proposals():
    proposer = SimpleNamespace(vllm_config=_config(), method="dspark", model=_FakeModel())

    def native(owner):
        logits = owner.model.compute_logits().view(1, 2, 4)
        logits[:, 0, 0].add_(10)
        logits[:, 1, 3].add_(10)
        return logits.argmax(-1)

    expected = native(proposer)
    result = capture_topk_proposal(native, proposer)
    assert torch.equal(result, expected)
    assert proposer._speco_last_topk_proposal[1].tolist() == [[[0, 3], [3, 0]]]
    assert "compute_logits" not in vars(proposer.model)
    result.fill_(0)
    assert proposer._speco_last_topk_proposal[0].tolist() == [[0, 3]]


def test_disabled_capture_does_not_touch_model_or_runner():
    owner = SimpleNamespace(vllm_config=_config(0))
    assert capture_topk_proposal(lambda _: 42, owner) == 42
    assert cache_topk_proposal(lambda _: 43, owner) == 43
    assert verified_topk_payload(owner, None, None) is None


def test_proposer_restores_method_on_failure():
    proposer = SimpleNamespace(vllm_config=_config(), method="dspark", model=_FakeModel())

    def broken(_):
        raise RuntimeError("native failure")

    with pytest.raises(RuntimeError, match="native failure"):
        capture_topk_proposal(broken, proposer)
    assert "compute_logits" not in vars(proposer.model)


@pytest.mark.parametrize("support", ["student", "union"])
def test_verify_support_follows_request_reordering_and_trimmed_prefix(support):
    runner = SimpleNamespace(
        vllm_config=_config(support=support), input_batch=SimpleNamespace(req_ids=["a", "b"]),
        drafter=SimpleNamespace(), requests={"a": None, "b": None},
    )
    proposals = torch.tensor([[1, 2], [3, 0]])
    candidates = torch.tensor([[[1, 0], [2, 0]], [[3, 1], [0, 2]]])

    def native(owner):
        owner.drafter._speco_last_topk_proposal = proposals, candidates
        return proposals

    cache_topk_proposal(native, runner)
    runner.input_batch.req_ids = ["b", "prefill", "a"]
    metadata = SimpleNamespace(
        num_draft_tokens=[1, 0, 2], draft_token_ids=torch.tensor([3, 1, 2]),
        target_logits_indices=torch.tensor([2, 0, 4]),
    )
    logits = torch.tensor([[8., 1., 5., 2.], [0., 2., 1., 4.], [1., 4., 2., 3.], [4., 3., 2., 1.], [0., 2., 3., 4.]])
    original_logits = logits.clone()
    payload = verified_topk_payload(runner, logits, metadata)
    by_request = topk_payload_by_request(payload)
    expected_b = [3, 1] + ([1, 3] if support == "union" else [])
    expected_a = [[1, 0] + ([0, 2] if support == "union" else []), [2, 0] + ([3, 2] if support == "union" else [])]
    assert by_request["b"]["draft_topk_ids"] == [expected_b]
    assert by_request["a"]["draft_topk_ids"] == expected_a
    assert "prefill" not in by_request
    assert by_request["b"]["draft_topk_teacher_logprobs"][0] == pytest.approx(
        logits[2].log_softmax(-1)[expected_b].tolist()
    )
    for row, positions in enumerate(expected_a):
        assert by_request["a"]["draft_topk_teacher_logprobs"][row] == pytest.approx(
            logits[[0, 4][row]].log_softmax(-1)[positions].tolist(), abs=1e-6,
        )
    if support == "union":
        from verl_speco.integration.vllm_runtime import _build_vllm_speculative_trace_events
        scheduler = SimpleNamespace(requests={"b": SimpleNamespace(_output_token_ids=[], is_finished=lambda: False)})
        output = SimpleNamespace(
            sampled_token_ids=[[0]], req_id_to_index={"b": 0},
            _speco_verified_draft_topk_by_request=by_request,
        )
        traces = _build_vllm_speculative_trace_events(
            scheduler, SimpleNamespace(scheduled_spec_decode_tokens={"b": [3]}), output,
        )
        assert traces["b"][0]["draft_topk_ids"] == [expected_b]
        packed = pack_trace_topk([
            {"draft_count": 1, **by_request["b"]},
        ], block_size=2, topk=2, support="union", device="cpu")
        assert packed["trace_draft_topk_ids"].shape == (1, 2, 4)
        assert packed["trace_draft_topk_ids"][0, 0].tolist() == expected_b
    assert torch.equal(logits, original_logits)
    metadata.draft_token_ids[0] = 0
    with pytest.raises(RuntimeError, match="stale"):
        verified_topk_payload(runner, logits, metadata)
    runner._speco_topk_proposals_by_request.clear()
    with pytest.raises(RuntimeError, match="Missing"):
        verified_topk_payload(runner, logits, metadata)


@pytest.mark.skipif(os.getenv("SPECO_TEST_NPU_TOPK") != "1", reason="explicit NPU smoke opt-in")
def test_npu_top16_loss_backward_and_post_markov_capture():
    pytest.importorskip("torch_npu")
    torch.npu.set_device(0)
    logits = torch.randn(7, 128, device="npu", requires_grad=True)
    teacher = torch.randn_like(logits).log_softmax(-1)
    log_q = logits.log_softmax(-1)
    ids = log_q.detach().topk(16, dim=-1).indices
    stats = student_support_reverse_kl(log_q, teacher, torch.ones(7, device="npu"), topk=16)
    rejected = student_support_reverse_kl(
        log_q, teacher.gather(-1, ids), torch.ones(7, device="npu"), topk=16, candidate_ids=ids
    )
    assert torch.allclose(stats["loss_sum"], rejected["loss_sum"])
    (stats["loss_sum"] + rejected["loss_sum"]).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    with torch.device("npu"):
        test_proposer_captures_post_markov_support_without_changing_proposals()
    from verl_speco.integration.vllm_runtime import (
        _maybe_apply_vllm_ascend_global_patch,
        patch_vllm_ascend_verified_draft_capture,
        patch_vllm_dspark_runtime,
    )

    _maybe_apply_vllm_ascend_global_patch()
    patch_vllm_dspark_runtime()
    assert patch_vllm_ascend_verified_draft_capture()
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    assert patch_vllm_ascend_verified_draft_capture()
    assert AscendDSparkProposer._propose._speco_topk_capture
    assert NPUModelRunner.propose_draft_token_ids._speco_topk_capture
