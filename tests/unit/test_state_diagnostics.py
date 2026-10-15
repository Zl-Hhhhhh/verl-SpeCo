import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.state_diagnostics import StateDiagnostics, tensor_record, trace_state


def trainer():
    model = torch.nn.Linear(3, 2, bias=False)
    model.register_buffer("cos_cached", torch.arange(6).reshape(2, 3).bfloat16(), persistent=False)
    model._tensor_template_cache = {("positions", 3): torch.arange(3)}
    return SimpleNamespace(model=model, backend=SimpleNamespace(model_type="dspark", target_lm_head=torch.nn.Linear(2, 4)),
                           rank=0, rollout_dp_rank=0, current_rl_step=20, optimizer_steps_total=15)


def test_exact_hash_detects_single_bfloat16_change_and_handles_noncontiguous():
    x = torch.arange(12).bfloat16().reshape(3, 4).T
    before, copy = tensor_record(x)
    assert before["sha256"] == tensor_record(x.contiguous())[0]["sha256"]
    assert torch.equal(copy, x)
    x[1, 1] += 1
    assert before["sha256"] != tensor_record(x)[0]["sha256"]
    assert tensor_record(torch.tensor(1.))[0]["local_shape"] == []
    assert tensor_record(torch.empty(0))[0]["local_shape"] == [0]


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided", "scalar", "empty"])
def test_hash_only_matches_exact_bytes_with_bounded_transfers(monkeypatch, layout):
    x = torch.arange(120).bfloat16().reshape(4, 5, 6)
    x = {"contiguous": x, "transpose": x.transpose(0, 2),
         "strided": x[:, ::2, ::2], "scalar": x[0, 0, 0], "empty": x[:0]}[layout]
    expected = hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    original_to = torch.Tensor.to
    transfers = []

    def tracked_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            transfers.append(tensor.numel() * tensor.element_size())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    record, snapshot = tensor_record(x, retain_tensor=False, chunk_bytes=16)
    assert snapshot is None
    assert record["sha256"] == expected
    assert max(transfers) <= 16


def test_cpu_snapshot_does_not_alias_input():
    x = torch.arange(12).reshape(3, 4).T
    _, snapshot = tensor_record(x)
    assert snapshot.is_contiguous()
    assert snapshot.data_ptr() != x.data_ptr()
    x.zero_()
    assert snapshot.sum() == 66


def test_state_captures_nonpersistent_rope_and_plain_cache_without_rng_or_model_mutation(tmp_path):
    t = trainer()
    d = StateDiagnostics(t, tmp_path)
    rng = torch.random.get_rng_state().clone()
    weights = t.model.weight.detach().clone()
    before = d.snapshot("save.before")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(weights, t.model.weight)
    assert before["state"]["buffers"]["model.cos_cached"]["persistent"] is False
    assert before["state"]["runtime"]
    with torch.no_grad():
        t.model.cos_cached[0, 0] += 1
    after = d.snapshot("save.after")
    assert before["state"]["parameters"] == after["state"]["parameters"]
    assert before["state"]["buffers"] != after["state"]["buffers"]
    buffer_path = d.root / before["state"]["buffers"]["model.cos_cached"]["artifact"]
    assert torch.load(buffer_path, weights_only=True)[0, 0] == 0


def test_full_input_and_output_evidence_are_independent_snapshots(tmp_path):
    t = trainer()
    d = StateDiagnostics(t, tmp_path)
    batch = {"input_ids": torch.tensor([[1, 2]]), "hidden_states": torch.randn(1, 2, 9),
             "position_ids": torch.tensor([[511, 512]]), "_speco_sample_ids": ["example"]}
    saved_hidden = batch["hidden_states"].clone()
    d.before_forward(batch, 0)
    batch["hidden_states"].zero_()
    payload = torch.load(d.last_forward["input_path"], weights_only=True)
    assert torch.equal(payload["batch"]["hidden_states"], saved_hidden)
    assert payload["batch"]["_speco_sample_ids"] == ["example"]
    loss = {"diagnostics": {"k3_logprob_dump": {"accepted": {"draft_logprobs": torch.tensor([-22.87])}},
                            "state_probe": {"draft_hidden": saved_hidden}}}
    d.after_forward(loss)
    events = [json.loads(l) for l in (d.root / "events.jsonl").read_text().splitlines()]
    output = torch.load(events[-1]["output_path"], weights_only=True)
    assert torch.equal(output["state_probe"]["draft_hidden"], saved_hidden)
    assert output["streams"]["accepted"]["draft_logprobs"].item() == pytest.approx(-22.87)
    assert events[0]["forward_id"] == events[-1]["forward_id"]


def test_disabled_decorator_preserves_async_return_and_exception(monkeypatch):
    monkeypatch.delenv("SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR", raising=False)
    class Dummy:
        @trace_state
        async def activate(self, value):
            return value
        @trace_state
        def save(self):
            raise ValueError("original")
    assert asyncio.run(Dummy().activate(7)) == 7
    with pytest.raises(ValueError, match="original"):
        Dummy().save()


def test_comparison_aligns_tokens_and_excludes_different_teacher_values():
    from tools.analyze_drafter_state_diagnostics import compare_streams
    def stream(ids, q, p):
        return {"accepted": {"sample_indices": torch.zeros(2), "anchor_positions": torch.ones(2),
                             "draft_positions": torch.zeros(2), "token_ids": torch.tensor(ids),
                             "prev_token_ids": torch.ones(2), "draft_logprobs": torch.tensor(q),
                             "target_logprobs": torch.tensor(p)}}
    left = stream([7, 8], [-1., -2.], [-3., -4.])
    right = stream([8, 7], [-30., -23.87], [-5., -3.])
    result = compare_streams(left, right)["accepted"]
    assert result["matched_teacher_rows"] == 1
    assert result["max_student_logprob_gap"] == pytest.approx(22.87)
