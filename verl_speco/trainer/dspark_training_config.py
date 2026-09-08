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
"""Shared DSpark training configuration predicates."""

from __future__ import annotations

from typing import Any


def dspark_target_distribution_loss_enabled(training_cfg: Any) -> bool:
    """Return whether DSpark needs the target model's final hidden state."""

    return any(
        float(training_cfg.get(name, default) or 0.0) > 0
        for name, default in (
            ("dspark_l1_loss_alpha", 0.9),
            ("dspark_accepted_k3_alpha", 1.0),
            ("dspark_accepted_fkl_alpha", 0.0),
        )
    )
