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

import torch

from tools.repair_dflash_checkpoint import translate_legacy_dflash_state_dict


def test_translate_legacy_dflash_state_dict_uses_current_weight_names() -> None:
    translated = translate_legacy_dflash_state_dict(
        {
            "context_proj.weight": torch.ones(2, 3),
            "context_norm.weight": torch.ones(2),
            "final_norm.weight": torch.ones(2),
            "layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "embed_tokens.weight": torch.ones(4, 2),
        }
    )

    assert set(translated) == {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "layers.0.self_attn.q_proj.weight",
    }
    assert all(tensor.dtype == torch.bfloat16 for tensor in translated.values())
    assert all(tensor.device.type == "cpu" for tensor in translated.values())
