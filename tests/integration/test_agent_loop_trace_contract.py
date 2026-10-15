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

from types import SimpleNamespace
import asyncio

import pytest

from verl_speco.integration.agent_loop_runtime import (
    _CURRENT_MAX_TOKENS,
    _CURRENT_SPEC_TRACE_HISTORY,
    _CURRENT_VALIDATE,
    _accumulate_agent_loop_speculative_trace,
    _sampling_params_with_speco_context,
)


def test_candidate_gate_caps_tokens_and_disables_training_collection() -> None:
    validate_token = _CURRENT_VALIDATE.set(True)
    max_tokens_token = _CURRENT_MAX_TOKENS.set(128)
    try:
        params = _sampling_params_with_speco_context(
            {"max_tokens": 8192, "temperature": 0.0}
        )
    finally:
        _CURRENT_MAX_TOKENS.reset(max_tokens_token)
        _CURRENT_VALIDATE.reset(validate_token)

    assert params["max_tokens"] == 128
    assert params["_verl_skip_drafter_collection"] is True


def test_multi_turn_agent_loop_accumulates_speculative_trace_by_reference() -> None:
    history_token = _CURRENT_SPEC_TRACE_HISTORY.set({})
    try:
        first = SimpleNamespace(
            extra_fields={"speculative_trace": []}
        )
        second = SimpleNamespace(
            extra_fields={"speculative_trace": [{"accepted_count": 0}]}
        )

        _accumulate_agent_loop_speculative_trace(first, "request-1")
        first_trace = first.extra_fields["speculative_trace"]
        _accumulate_agent_loop_speculative_trace(second, "request-1")

        assert first_trace is second.extra_fields["speculative_trace"]
        assert first_trace == [
            {"accepted_count": 0, "generation_index": 1},
        ]
    finally:
        _CURRENT_SPEC_TRACE_HISTORY.reset(history_token)


def test_multi_turn_agent_loop_accumulates_trace_diagnostics() -> None:
    history_token = _CURRENT_SPEC_TRACE_HISTORY.set({})
    try:
        first = SimpleNamespace(
            extra_fields={
                "speculative_trace": [],
                "speculative_trace_diagnostics": {
                    "engine_scheduled_draft_tokens": 3,
                    "frontend_received_events": 0,
                },
            }
        )
        second = SimpleNamespace(
            extra_fields={
                "speculative_trace": [{"accepted_count": 1}],
                "speculative_trace_diagnostics": {
                    "engine_scheduled_draft_tokens": 2,
                    "frontend_received_events": 1,
                },
            }
        )

        _accumulate_agent_loop_speculative_trace(first, "request-1")
        diagnostics = first.extra_fields["speculative_trace_diagnostics"]
        _accumulate_agent_loop_speculative_trace(second, "request-1")

        assert diagnostics is second.extra_fields["speculative_trace_diagnostics"]
        assert diagnostics["engine_scheduled_draft_tokens"] == 5
        assert diagnostics["frontend_received_events"] == 1
        assert diagnostics["agent_generations_received"] == 2
        assert diagnostics["agent_trace_events_received"] == 1
    finally:
        _CURRENT_SPEC_TRACE_HISTORY.reset(history_token)


@pytest.mark.parametrize("positional", [False, True])
def test_vllm_boundary_preserves_cap_and_strips_private_metadata(positional):
    from vllm import SamplingParams
    from verl_speco.integration.vllm_runtime import _SpecoVLLMHttpServerMixin

    class Upstream:
        async def generate(self, prompt_ids, sampling_params, request_id):
            parsed = SamplingParams(**sampling_params)
            assert parsed.max_tokens == 512
            assert parsed.temperature == 0
            assert not any(k.startswith("_verl_") for k in sampling_params)
            sampling_params.pop("max_tokens")
            return SimpleNamespace(extra_fields=None)

    class Server(_SpecoVLLMHttpServerMixin, Upstream):
        pass

    params = dict(max_tokens=512, temperature=0, _verl_global_steps=5,
                  _verl_skip_drafter_collection=True)
    original = params.copy()
    server = Server()
    if positional:
        asyncio.run(server.generate([1], params, "probe"))
    else:
        asyncio.run(server.generate(prompt_ids=[1], sampling_params=params, request_id="probe"))
    assert params == original
