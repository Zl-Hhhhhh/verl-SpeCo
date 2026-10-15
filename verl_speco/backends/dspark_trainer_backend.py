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

import hashlib
import logging
import math
import os
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl_speco.backends.dflash_trainer_backend import (
    DFlashTrainerBackend,
    DFlashTrainingModel,
    _create_dflash_dense_attention_mask,
    _create_dflash_mask_mod,
)
from verl_speco.models.dflash.flex_attention import compile_friendly_create_block_mask
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step
from verl_speco.trainer.dspark_topk import (
    pack_trace_topk,
    student_support_reverse_kl,
    topk_candidate_width,
    validate_topk_modes,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _trace_anchor_rows_for_item(
    item: dict[str, Any],
    *,
    crop_start: int,
    crop_end: int,
    num_anchors: int,
    block_size: int,
    low_entropy_fraction: float = 0.0,
    low_entropy_threshold: float = 0.01,
    position_stratified_low_entropy: bool = False,
    require_rejected_logprob: bool = False,
    low_entropy_max_rejected_logprob: float = -4.0,
    selection_stats: Optional[dict[str, int]] = None,
    eligible_reserve_anchors: Optional[set[int]] = None,
    forced_reserved_anchors: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """Map request-level verify events into a cropped DSpark training window."""

    if not math.isfinite(low_entropy_fraction) or not 0 <= low_entropy_fraction <= 1:
        raise ValueError("low_entropy_fraction must be in [0, 1]")
    if not math.isfinite(low_entropy_threshold) or low_entropy_threshold <= 0:
        raise ValueError("low_entropy_threshold must be finite and positive")
    if position_stratified_low_entropy and not math.isfinite(low_entropy_max_rejected_logprob):
        raise ValueError("low_entropy_max_rejected_logprob must be finite")

    trace = item.get("speculative_trace")
    if not isinstance(trace, (list, tuple)) or not trace:
        return []
    try:
        feature_start = int(item.get("_verl_feature_start", 0) or 0)
        prompt_len = int(item.get("_verl_prompt_len", 0) or 0)
    except (TypeError, ValueError):
        return []
    if prompt_len <= 0 or crop_end - crop_start <= 1:
        return []

    sample_id = str(item.get("sample_id", ""))
    rows_by_anchor: dict[int, dict[str, Any]] = {}
    low_entropy_anchors: set[int] = set()
    rejection_count = entropy_count = 0

    def keep_row(row: dict[str, Any]) -> None:
        anchor = int(row["anchor"])
        previous = rows_by_anchor.get(anchor)
        if previous is None or int(row["priority"]) < int(previous["priority"]):
            rows_by_anchor[anchor] = row

    for event_index, event in enumerate(trace):
        if not isinstance(event, dict):
            continue
        try:
            trajectory_response_start = event.get("trajectory_response_start")
            generation_index = int(event.get("generation_index", 0) or 0)
            if trajectory_response_start is None and generation_index > 0:
                # Multi-turn response_start is local to one generation. Until the
                # agent loop supplies its trajectory offset, using it would create
                # a plausible-looking but false anchor.
                continue
            response_start = int(
                event.get("response_start", -1)
                if trajectory_response_start is None
                else trajectory_response_start
            )
            reported_draft_count = int(event.get("draft_count", 0) or 0)
            reported_accepted_count = int(event.get("accepted_count", 0) or 0)
        except (TypeError, ValueError):
            continue
        if response_start < 0 or reported_draft_count <= 0:
            continue
        raw_draft_ids = event.get("draft_token_ids")
        if not isinstance(raw_draft_ids, (list, tuple)):
            continue
        try:
            draft_token_ids = [int(token_id) for token_id in raw_draft_ids]
        except (TypeError, ValueError):
            continue
        verified_draft_count = min(reported_draft_count, len(draft_token_ids))
        draft_count = min(verified_draft_count, block_size)
        if draft_count <= 0:
            continue
        draft_token_ids = draft_token_ids[:draft_count]
        accepted_count = min(max(reported_accepted_count, 0), draft_count)
        absolute_anchor = prompt_len - 1 + response_start
        local_anchor = absolute_anchor - feature_start - crop_start
        primary_anchor_in_crop = (
            local_anchor >= 0 and local_anchor + 1 < crop_end - crop_start
        )

        full_event_all_accepted = bool(
            event.get("all_drafts_accepted", False)
        ) or (
            reported_accepted_count >= verified_draft_count
        )
        all_accepted = full_event_all_accepted or accepted_count >= draft_count
        if all_accepted:
            accepted_count = draft_count
        rejected_token_id = -1
        if not all_accepted:
            raw_rejected = event.get("rejected_draft_token_id")
            if raw_rejected is None:
                if accepted_count < len(draft_token_ids):
                    raw_rejected = draft_token_ids[accepted_count]
            try:
                rejected_token_id = int(raw_rejected)
            except (TypeError, ValueError):
                rejected_token_id = -1

        priority_payload = (
            f"{sample_id}:{absolute_anchor}:{event_index}:"
            f"{accepted_count}:{draft_count}:{rejected_token_id}"
        )
        priority = int.from_bytes(
            hashlib.blake2b(priority_payload.encode(), digest_size=8).digest(),
            "big",
        )
        raw_teacher_logprobs = event.get("draft_teacher_logprobs")
        raw_entropies = event.get("draft_teacher_entropies")
        first_rejection_entropy = None
        if not all_accepted and isinstance(raw_entropies, (list, tuple)):
            try:
                entropy = float(raw_entropies[accepted_count])
                if math.isfinite(entropy) and entropy >= 0:
                    first_rejection_entropy = entropy
            except (TypeError, ValueError, IndexError):
                pass
        teacher_logprobs = [0.0] * draft_count
        teacher_logprob_mask = [False] * draft_count
        if isinstance(raw_teacher_logprobs, (list, tuple)):
            for token_offset, raw_logprob in enumerate(
                raw_teacher_logprobs[:draft_count]
            ):
                try:
                    teacher_logprob = float(raw_logprob)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(teacher_logprob):
                    teacher_logprobs[token_offset] = teacher_logprob
                    teacher_logprob_mask[token_offset] = True

        row: dict[str, Any] = {
            "anchor": local_anchor,
            "accepted_count": accepted_count,
            "draft_count": draft_count,
            "rejected_token_id": rejected_token_id,
            "draft_token_ids": draft_token_ids,
            "draft_teacher_logprobs": teacher_logprobs,
            "draft_teacher_logprob_mask": teacher_logprob_mask,
            "first_rejection_entropy": first_rejection_entropy,
            "response_position": response_start + accepted_count,
            "all_accepted": all_accepted,
            "bonus_response_anchor": False,
            "priority": priority,
        }
        for key in ("draft_topk_ids", "draft_topk_teacher_logprobs"):
            if key in event:
                row[key] = event[key]
        if primary_anchor_in_crop:
            keep_row(row)
            # Only the actual first rejection is a reached teacher context.
            # Unverified suffix entropy must not influence the sampling policy.
            if not all_accepted and local_anchor + 1 + accepted_count < crop_end - crop_start:
                rejection_count += 1
                entropies = event.get("draft_teacher_entropies")
                try:
                    entropy = float(entropies[accepted_count])
                except (TypeError, ValueError, IndexError, KeyError):
                    entropy = float("nan")
                if math.isfinite(entropy) and entropy >= 0:
                    entropy_count += 1
                    if entropy < low_entropy_threshold:
                        low_entropy_anchors.add(local_anchor)

        # DSpark emits K query states for K proposals, so the K+1 target-model
        # bonus token has no query in the original all-accepted verify block.
        # Replay it at q0 from the last accepted draft token instead. This is a
        # response-only continuation anchor and never enters the rejected branch.
        if full_event_all_accepted:
            raw_bonus_token_id = event.get("verification_token_id")
            if raw_bonus_token_id is None:
                raw_verified_ids = event.get("verified_token_ids")
                if (
                    isinstance(raw_verified_ids, (list, tuple))
                    and verified_draft_count < len(raw_verified_ids)
                ):
                    raw_bonus_token_id = raw_verified_ids[verified_draft_count]
            try:
                bonus_token_id = int(raw_bonus_token_id)
            except (TypeError, ValueError):
                bonus_token_id = -1
            absolute_bonus_anchor = absolute_anchor + verified_draft_count
            local_bonus_anchor = absolute_bonus_anchor - feature_start - crop_start
            if (
                bonus_token_id >= 0
                and local_bonus_anchor >= 0
                and local_bonus_anchor + 1 < crop_end - crop_start
            ):
                bonus_priority_payload = (
                    f"{sample_id}:{absolute_bonus_anchor}:{event_index}:bonus:"
                    f"{bonus_token_id}"
                )
                bonus_priority = int.from_bytes(
                    hashlib.blake2b(
                        bonus_priority_payload.encode(), digest_size=8
                    ).digest(),
                    "big",
                )
                keep_row(
                    {
                        "anchor": local_bonus_anchor,
                        "accepted_count": 1,
                        "draft_count": 1,
                        "rejected_token_id": -1,
                        "draft_token_ids": [bonus_token_id],
                        "draft_teacher_logprobs": [0.0],
                        "draft_teacher_logprob_mask": [False],
                        "all_accepted": True,
                        "bonus_response_anchor": True,
                        "priority": bonus_priority,
                    }
                )

    ranked = sorted(
        rows_by_anchor.values(),
        key=lambda row: (int(row["priority"]), int(row["anchor"])),
    )
    budget = min(max(int(num_anchors), 0), len(ranked))
    eligible = [
        row for row in ranked
        if row["anchor"] in low_entropy_anchors and not row["all_accepted"]
        and (not (position_stratified_low_entropy or require_rejected_logprob) or (
            row["draft_teacher_logprob_mask"][row["accepted_count"]]
            and row["draft_teacher_logprobs"][row["accepted_count"]]
            <= low_entropy_max_rejected_logprob
        ))
    ]
    if eligible_reserve_anchors is not None:
        eligible_reserve_anchors.clear()
        eligible_reserve_anchors.update(int(row["anchor"]) for row in eligible)
    reserve_count = int(budget * low_entropy_fraction)
    if forced_reserved_anchors is not None:
        forced = set(forced_reserved_anchors)
        eligible_by_anchor = {row["anchor"]: row for row in eligible}
        if len(forced) != len(forced_reserved_anchors) or not forced <= eligible_by_anchor.keys():
            raise ValueError("Global sampling plan contains duplicate or ineligible anchors")
        if len(forced) > reserve_count:
            raise ValueError("Global sampling plan exceeds the per-window reserve")
        reserved = [eligible_by_anchor[anchor] for anchor in forced_reserved_anchors]
    elif position_stratified_low_entropy:
        buckets = [[], [], []]
        for row in eligible:
            buckets[min(int(row["response_position"]) // 2048, 2)].append(row)
        reserved = []
        while len(reserved) < reserve_count and any(buckets):
            for bucket in buckets:
                if bucket and len(reserved) < reserve_count:
                    reserved.append(bucket.pop(0))
    else:
        reserved = eligible[:reserve_count]
    reserved_anchors = {row["anchor"] for row in reserved}
    selected = reserved + [
        row for row in ranked if row["anchor"] not in reserved_anchors
    ][: budget - len(reserved)]
    if selection_stats is not None:
        eligible_anchors = {row["anchor"] for row in eligible}
        for key, value in {
            "candidates": len(ranked), "rejections": rejection_count,
            "entropy_observed": entropy_count, "eligible": len(eligible),
            "reserved": len(reserved), "selected": len(selected),
            "selected_low_entropy": sum(row["anchor"] in eligible_anchors for row in selected),
            "uniform_low_entropy": sum(row["anchor"] in eligible_anchors for row in ranked[:budget]),
            "reserved_position_0_2047": sum(int(row["response_position"]) < 2048 for row in reserved),
            "reserved_position_2048_4095": sum(2048 <= int(row["response_position"]) < 4096 for row in reserved),
            "reserved_position_4096_plus": sum(int(row["response_position"]) >= 4096 for row in reserved),
        }.items():
            selection_stats[key] = selection_stats.get(key, 0) + value
    return sorted(selected, key=lambda row: int(row["anchor"]))


class DSparkTrainingModel(DFlashTrainingModel):
    """Training wrapper around DSparkDraftModel.

    It reuses the DFlash block drafter backbone and changes the target alignment
    to DSpark semantics:

    - anchor token `x[a]` seeds position 0,
    - labels are `x[a + 1 : a + 1 + block_size]`,
    - previous tokens are `[x[a], labels[:-1]]`,
    - every supervised position, including position 0, can contribute to the
      response stream; Draft-OPD defaults use sampled K3 on that stream.
    """

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 0.0,
        loss_mode: str = "full_vocab",
        sampled_ce_negatives: int = 0,
        ce_token_chunk_size: int = 64,
        ce_loss_alpha: float = 0.0,
        first_rejection_ce_alpha: float = 0.0,
        first_rejection_ce_low_entropy_multiplier: float = 1.0,
        first_rejection_ce_low_entropy_threshold: float = 0.01,
        first_rejection_ce_max_rejected_logprob: float = -4.0,
        l1_loss_alpha: float = 0.0,
        confidence_head_alpha: float = 0.0,
        accepted_k3_alpha: float = 1.0,
        accepted_fkl_alpha: float = 0.0,
        rejected_k3_alpha: float = 1.0,
        rejected_aux_alpha: float = 0.0,
        rejected_prefix_length: int = 0,
        response_loss: str = "k3",
        rejected_loss: str = "k3",
        topk: int = 16,
        topk_support: str = "student",
        response_stream_weight: float = 1.0,
        rejected_stream_weight: float = 1.0,
        rejected_position_decay: float = 0.9,
        rejected_first_weight_multiplier: float = 1.0,
        rejected_suffix_weight_multiplier: float = 1.0,
        rejected_log_ratio_clamp: float = 20.0,
        rejected_loss_max_clamp: float = 10.0,
        asymmetric_k3_enabled: bool = False,
        accepted_asymmetric_k3_enabled: Optional[bool] = None,
        rejected_asymmetric_k3_enabled: Optional[bool] = None,
        asymmetric_k3_positive_threshold: float = 2.0,
        dynamic_accept_reject_weighting_enabled: bool = False,
        dynamic_stream_weight_min_multiplier: float = 0.25,
        dynamic_stream_weight_max_multiplier: float = 4.0,
        k3_distribution_bins: int = 512,
        dump_k3_logprobs: bool = False,
        trace_aligned_anchors: bool = False,
        attention_mode: str = "full",
        context_window: Optional[int] = None,
        global_layer_count: int = 1,
        use_absolute_position_ids: bool = False,
        l1_chunk_size: int = 0,
        debug_log: bool = False,
        debug_log_first_n: int = 2,
        debug_log_interval: int = 100,
    ):
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            front_position_weight=1.0,
            front_position_count=0,
            loss_mode=loss_mode,
            sampled_ce_negatives=sampled_ce_negatives,
            ce_token_chunk_size=ce_token_chunk_size,
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.first_rejection_ce_alpha = float(first_rejection_ce_alpha)
        self.first_rejection_ce_low_entropy_multiplier = float(
            first_rejection_ce_low_entropy_multiplier
        )
        self.first_rejection_ce_low_entropy_threshold = float(
            first_rejection_ce_low_entropy_threshold
        )
        self.first_rejection_ce_max_rejected_logprob = float(
            first_rejection_ce_max_rejected_logprob
        )
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.accepted_k3_alpha = float(accepted_k3_alpha)
        self.accepted_fkl_alpha = float(accepted_fkl_alpha)
        self.rejected_k3_alpha = float(rejected_k3_alpha)
        self.rejected_aux_alpha = float(rejected_aux_alpha)
        self.rejected_prefix_length = int(rejected_prefix_length)
        self.response_loss = str(response_loss).strip().lower()
        self.rejected_loss = str(rejected_loss).strip().lower()
        self.topk = int(topk)
        self.topk_support = str(topk_support).strip().lower()
        validate_topk_modes(
            self.response_loss, self.rejected_loss, self.topk, trace_aligned_anchors,
            self.topk_support,
        )
        self.response_stream_weight = float(response_stream_weight)
        self.rejected_stream_weight = float(rejected_stream_weight)
        self.rejected_position_decay = float(rejected_position_decay)
        self.rejected_first_weight_multiplier = float(rejected_first_weight_multiplier)
        self.rejected_suffix_weight_multiplier = float(rejected_suffix_weight_multiplier)
        self.rejected_log_ratio_clamp = float(rejected_log_ratio_clamp)
        self.rejected_loss_max_clamp = float(rejected_loss_max_clamp)
        self.asymmetric_k3_enabled = bool(asymmetric_k3_enabled)
        self.accepted_asymmetric_k3_enabled = (
            self.asymmetric_k3_enabled
            if accepted_asymmetric_k3_enabled is None
            else bool(accepted_asymmetric_k3_enabled)
        )
        self.rejected_asymmetric_k3_enabled = (
            self.asymmetric_k3_enabled
            if rejected_asymmetric_k3_enabled is None
            else bool(rejected_asymmetric_k3_enabled)
        )
        self.asymmetric_k3_positive_threshold = float(
            asymmetric_k3_positive_threshold
        )
        self.dynamic_accept_reject_weighting_enabled = bool(
            dynamic_accept_reject_weighting_enabled
        )
        self.dynamic_stream_weight_min_multiplier = float(
            dynamic_stream_weight_min_multiplier
        )
        self.dynamic_stream_weight_max_multiplier = float(
            dynamic_stream_weight_max_multiplier
        )
        self.k3_distribution_bins = int(k3_distribution_bins)
        self.dump_k3_logprobs = bool(dump_k3_logprobs)
        self.attention_mode = str(attention_mode or "full").strip().lower()
        self.context_window = (
            int(context_window) if context_window not in (None, 0) else None
        )
        self.global_layer_count = int(global_layer_count)
        self.use_absolute_position_ids = bool(use_absolute_position_ids)
        valid_attention_modes = {"full", "swa", "hybrid", "sparse_global"}
        if self.attention_mode not in valid_attention_modes:
            raise ValueError(
                "DSpark attention_mode must be one of "
                f"{sorted(valid_attention_modes)}, got {self.attention_mode!r}"
            )
        if self.attention_mode != "full" and (
            self.context_window is None or self.context_window <= 0
        ):
            raise ValueError(
                f"DSpark attention_mode={self.attention_mode!r} requires a positive context_window"
            )
        if self.attention_mode != "full" and not self.use_absolute_position_ids:
            raise ValueError(
                f"DSpark attention_mode={self.attention_mode!r} requires "
                "use_absolute_position_ids=True for cropped-window RoPE parity"
            )
        num_draft_layers = len(self.draft_model.layers)
        if self.attention_mode in {"hybrid", "sparse_global"} and not (
            0 < self.global_layer_count <= num_draft_layers
        ):
            raise ValueError(
                "DSpark global_layer_count must be in [1, num_hidden_layers] for "
                f"{self.attention_mode}, got {self.global_layer_count}"
            )
        loss_weights = {
            "ce_loss_alpha": self.ce_loss_alpha,
            "l1_loss_alpha": self.l1_loss_alpha,
            "accepted_k3_alpha": self.accepted_k3_alpha,
            "accepted_fkl_alpha": self.accepted_fkl_alpha,
            "rejected_k3_alpha": self.rejected_k3_alpha,
            "rejected_aux_alpha": self.rejected_aux_alpha,
            "response_stream_weight": self.response_stream_weight,
            "rejected_stream_weight": self.rejected_stream_weight,
        }
        for name, value in loss_weights.items():
            if value < 0:
                raise ValueError(f"DSpark {name} must be non-negative, got {value}")
        if not math.isfinite(self.rejected_aux_alpha) or self.rejected_aux_alpha < 0:
            raise ValueError("DSpark rejected_aux_alpha must be finite and non-negative")
        if self.rejected_prefix_length < 0:
            raise ValueError("DSpark rejected_prefix_length must be non-negative")
        if self.rejected_aux_alpha > 0:
            if not trace_aligned_anchors:
                raise ValueError("DSpark rejected_aux_alpha requires trace_aligned_anchors")
            if self.rejected_prefix_length <= 0:
                raise ValueError("DSpark rejected_aux_alpha requires a positive rejected_prefix_length")
        if (
            not math.isfinite(self.first_rejection_ce_alpha)
            or self.first_rejection_ce_alpha < 0
        ):
            raise ValueError("DSpark first_rejection_ce_alpha must be finite and non-negative")
        if self.first_rejection_ce_alpha > 0:
            if not trace_aligned_anchors:
                raise ValueError("DSpark first_rejection_ce_alpha requires trace_aligned_anchors")
            if self.loss_mode == "sampled_ce":
                raise ValueError("DSpark first_rejection_ce_alpha requires full-vocabulary CE")
            logger.info(
                "[dspark-trainer] first-rejection correct-token CE enabled: "
                "alpha=%s, separate global first-rejection mean",
                self.first_rejection_ce_alpha,
            )
        if not math.isfinite(self.first_rejection_ce_low_entropy_multiplier) or (
            self.first_rejection_ce_low_entropy_multiplier < 1.0
        ):
            raise ValueError("DSpark first-rejection CE low-entropy multiplier must be finite and >= 1")
        if not math.isfinite(self.first_rejection_ce_low_entropy_threshold) or (
            self.first_rejection_ce_low_entropy_threshold <= 0
        ):
            raise ValueError("DSpark first-rejection CE low-entropy threshold must be finite and positive")
        if not math.isfinite(self.first_rejection_ce_max_rejected_logprob) or (
            self.first_rejection_ce_max_rejected_logprob > 0
        ):
            raise ValueError("DSpark first-rejection CE maximum rejected logprob must be finite and <= 0")
        if self.first_rejection_ce_low_entropy_multiplier > 1 and self.first_rejection_ce_alpha <= 0:
            raise ValueError("DSpark weighted first-rejection CE requires positive first_rejection_ce_alpha")
        if not 0.0 < self.rejected_position_decay <= 1.0:
            raise ValueError("DSpark rejected_position_decay must be in (0, 1]")
        for name in ("rejected_first_weight_multiplier", "rejected_suffix_weight_multiplier"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"DSpark {name} must be finite and non-negative, got {value}")
            if value != 1.0 and not trace_aligned_anchors:
                raise ValueError(f"DSpark {name} requires trace_aligned_anchors")
        if self.rejected_log_ratio_clamp <= 0:
            raise ValueError("DSpark rejected_log_ratio_clamp must be positive")
        if self.rejected_loss_max_clamp <= 0:
            raise ValueError("DSpark rejected_loss_max_clamp must be positive")
        if self.asymmetric_k3_positive_threshold <= 0:
            raise ValueError(
                "DSpark asymmetric_k3_positive_threshold must be positive"
            )
        if (
            (
                self.accepted_asymmetric_k3_enabled
                or self.rejected_asymmetric_k3_enabled
            )
            and self.asymmetric_k3_positive_threshold
            > self.rejected_log_ratio_clamp
        ):
            raise ValueError(
                "DSpark asymmetric_k3_positive_threshold cannot exceed "
                "rejected_log_ratio_clamp when asymmetric K3 is enabled"
            )
        if self.dynamic_stream_weight_min_multiplier <= 0:
            raise ValueError(
                "DSpark dynamic_stream_weight_min_multiplier must be positive"
            )
        if (
            self.dynamic_stream_weight_max_multiplier
            < self.dynamic_stream_weight_min_multiplier
        ):
            raise ValueError(
                "DSpark dynamic_stream_weight_max_multiplier must be greater than "
                "or equal to dynamic_stream_weight_min_multiplier"
            )
        if self.k3_distribution_bins <= 0:
            raise ValueError("DSpark k3_distribution_bins must be positive")
        self.trace_aligned_anchors = bool(trace_aligned_anchors)
        self.l1_chunk_size = int(l1_chunk_size or 0)
        if self.confidence_head_alpha > 0:
            raise NotImplementedError(
                "DSpark confidence loss needs target acceptance targets from target logits; "
                "set dspark_confidence_loss_alpha=0 for the current CE-only trainer path."
            )
        confidence_head = getattr(self.draft_model, "confidence_head", None)
        if confidence_head is not None:
            confidence_head.requires_grad_(False)
            logger.info(
                "[dspark-trainer] confidence head is loaded but frozen because confidence loss is disabled"
            )
        self.debug_log = bool(debug_log)
        self.debug_log_first_n = max(int(debug_log_first_n), 0)
        self.debug_log_interval = max(int(debug_log_interval), 1)
        self._debug_forward_count = 0

    def _apply_k3_tail_policy(
        self,
        delta: torch.Tensor,
        *,
        asymmetric_enabled: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the configured tail policy to sampled K3 values.

        The asymmetric policy preserves ordinary K3 for the negative tail and
        replaces the positive tail after ``tau`` with K3's tangent at ``tau``.
        This keeps the value and first derivative continuous while bounding the
        positive-tail gradient. ``legacy_clamped`` remains comparable with old
        runs and reports values that would exceed the hard-clamp ceiling.
        """

        raw_k3 = torch.exp(delta) - delta - 1.0
        legacy_clamped = raw_k3 > self.rejected_loss_max_clamp
        if asymmetric_enabled is None:
            asymmetric_enabled = self.asymmetric_k3_enabled
        if not asymmetric_enabled:
            return (
                raw_k3.clamp(max=self.rejected_loss_max_clamp),
                raw_k3,
                legacy_clamped,
                torch.zeros_like(delta, dtype=torch.bool),
            )

        threshold = delta.new_tensor(self.asymmetric_k3_positive_threshold)
        exp_threshold = torch.exp(threshold)
        tangent_k3 = (
            exp_threshold
            - threshold
            - 1.0
            + (exp_threshold - 1.0) * (delta - threshold)
        )
        linearized = delta > threshold
        return (
            torch.where(linearized, tangent_k3, raw_k3),
            raw_k3,
            legacy_clamped,
            linearized,
        )

    def _effective_stream_weights(
        self,
        *,
        response_token_count: torch.Tensor,
        rejected_token_count: torch.Tensor,
        response_stream_enabled: bool,
        rejected_stream_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return effective stream weights and their count-balancing multipliers."""

        response_weight = response_token_count.new_tensor(
            self.response_stream_weight, dtype=torch.float32
        )
        rejected_weight = rejected_token_count.new_tensor(
            self.rejected_stream_weight, dtype=torch.float32
        )
        response_multiplier = response_weight.new_ones(())
        rejected_multiplier = rejected_weight.new_ones(())
        if not (
            self.dynamic_accept_reject_weighting_enabled
            and response_stream_enabled
            and rejected_stream_enabled
        ):
            return (
                response_weight,
                rejected_weight,
                response_multiplier,
                rejected_multiplier,
            )

        response_count = response_token_count.detach().float()
        rejected_count = rejected_token_count.detach().float()
        both_nonzero = (response_count > 0) & (rejected_count > 0)
        total_count = response_count + rejected_count
        response_multiplier = torch.where(
            both_nonzero,
            total_count / (2.0 * response_count.clamp(min=1e-6)),
            response_multiplier,
        ).clamp(
            min=self.dynamic_stream_weight_min_multiplier,
            max=self.dynamic_stream_weight_max_multiplier,
        )
        rejected_multiplier = torch.where(
            both_nonzero,
            total_count / (2.0 * rejected_count.clamp(min=1e-6)),
            rejected_multiplier,
        ).clamp(
            min=self.dynamic_stream_weight_min_multiplier,
            max=self.dynamic_stream_weight_max_multiplier,
        )
        return (
            response_weight * response_multiplier,
            rejected_weight * rejected_multiplier,
            response_multiplier,
            rejected_multiplier,
        )

    def _k3_distribution_stats(
        self,
        k3: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build mergeable weighted K3 statistics without retaining autograd state."""

        values = k3.detach().float()
        effective_weights = torch.where(
            valid_mask & (weights > 0),
            weights.detach().float(),
            torch.zeros_like(weights, dtype=torch.float32),
        )
        histogram = torch.zeros(
            self.k3_distribution_bins,
            device=values.device,
            dtype=torch.float32,
        )
        bin_indices = torch.floor(
            values.clamp(min=0.0, max=self.rejected_loss_max_clamp)
            * (float(self.k3_distribution_bins) / self.rejected_loss_max_clamp)
        ).long()
        bin_indices = bin_indices.clamp(max=self.k3_distribution_bins - 1)
        histogram.scatter_add_(0, bin_indices, effective_weights)
        positive = effective_weights > 0
        local_min = torch.where(
            positive,
            values,
            torch.full_like(values, float("inf")),
        ).min()
        local_max = torch.where(
            positive,
            values,
            torch.full_like(values, float("-inf")),
        ).max()
        return {
            "histogram": histogram,
            "min": local_min,
            "max": local_max,
        }

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        trace_anchor_positions: Optional[torch.Tensor] = None,
        trace_anchor_mask: Optional[torch.Tensor] = None,
    ):
        bsz = loss_mask.shape[0]
        num_candidates = max(seq_len - 1, 0)
        if num_candidates <= 0:
            anchors = torch.zeros(
                bsz, self.num_anchors, dtype=torch.long, device=device
            )
            keep_mask = torch.zeros(
                bsz, self.num_anchors, dtype=torch.bool, device=device
            )
            return anchors, keep_mask

        if self.trace_aligned_anchors:
            if trace_anchor_positions is None or trace_anchor_mask is None:
                raise ValueError(
                    "DSpark trace-aligned anchor mode requires trace anchor tensors"
                )
            anchors = trace_anchor_positions.to(device=device, dtype=torch.long)
            keep_mask = trace_anchor_mask.to(device=device, dtype=torch.bool)
            if anchors.dim() != 2 or keep_mask.shape != anchors.shape:
                raise ValueError(
                    "DSpark trace anchor positions and mask must have matching [batch, anchors] shapes"
                )
            if anchors.size(0) != bsz:
                raise ValueError(
                    f"DSpark trace anchor batch mismatch: {anchors.size(0)} != {bsz}"
                )
            valid_anchor = (anchors >= 0) & (anchors < num_candidates)
            safe_next = (anchors + 1).clamp(min=0, max=max(seq_len - 1, 0))
            next_is_supervised = torch.gather(loss_mask > 0.5, 1, safe_next)
            keep_mask = keep_mask & valid_anchor & next_is_supervised
            return torch.where(keep_mask, anchors, 0), keep_mask

        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        indices = (
            self._cached_arange("dspark_anchor_indices", num_candidates, device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        masked_indices = torch.where(valid, indices, seq_len + 1)
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)
        take_n = min(self.num_anchors, num_candidates)
        _, top_idx = torch.topk(
            random_vals, k=take_n, dim=1, largest=False, sorted=False
        )
        selected = torch.gather(masked_indices, 1, top_idx).sort(dim=1).values
        if take_n < self.num_anchors:
            selected = torch.cat(
                [
                    selected,
                    torch.zeros(
                        bsz, self.num_anchors - take_n, dtype=torch.long, device=device
                    ),
                ],
                dim=1,
            )
        keep_mask = self._cached_arange(
            "dspark_anchor_keep", self.num_anchors, device
        ).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=self.num_anchors)
        return torch.where(keep_mask, selected, 0), keep_mask

    def _build_label_tensors(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        n_blocks = anchor_positions.shape[1]
        label_offsets = (
            self._cached_arange(
                "dspark_label_offsets", self.block_size, device, view_shape=(1, 1, -1)
            )
            + 1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=max(seq_len - 1, 0))
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_mask = (
            valid_label_mask & (target_loss_mask > 0.5) & block_keep_mask.unsqueeze(-1)
        )
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()

        anchor_token_ids = torch.gather(
            input_ids, 1, anchor_positions.clamp(0, max(seq_len - 1, 0))
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        return target_ids, prev_token_ids, eval_mask, label_indices

    def _markov_bias_for_active(
        self,
        *,
        active_hidden: torch.Tensor,
        active_prev_tokens: torch.Tensor,
        restricted_vocab: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        markov_head = getattr(self.draft_model, "markov_head", None)
        if markov_head is None:
            return None
        if markov_head.__class__.__name__ == "DSparkRNNMarkovHead":
            raise NotImplementedError(
                "DSpark rnn markov_head_type requires full-block logits and is not supported yet"
            )

        if restricted_vocab is None:
            return markov_head.compute_step_bias(active_prev_tokens, active_hidden)

        prev_embeddings = markov_head.get_prev_embeddings(active_prev_tokens)
        if hasattr(markov_head, "gate_proj"):
            gate = torch.sigmoid(
                markov_head.gate_proj(
                    torch.cat([active_hidden, prev_embeddings], dim=-1)
                )
            )
            prev_embeddings = gate.to(prev_embeddings.dtype) * prev_embeddings
        markov_w2 = markov_head.markov_w2.weight.index_select(
            0, restricted_vocab.to(markov_head.markov_w2.weight.device)
        )
        return F.linear(prev_embeddings, markov_w2.to(prev_embeddings.dtype))

    def _full_vocab_logits(
        self,
        active_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        active_prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = F.linear(active_hidden, lm_head_weight)
        if active_prev_tokens is None:
            return logits
        markov_bias = self._markov_bias_for_active(
            active_hidden=active_hidden,
            active_prev_tokens=active_prev_tokens,
            restricted_vocab=None,
        )
        return logits if markov_bias is None else logits + markov_bias

    def _gather_aligned_target_hidden(
        self,
        *,
        target_last_hidden_states: Optional[torch.Tensor],
        label_indices: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_last_hidden_states is None:
            return None
        if target_last_hidden_states.dim() != 3:
            raise ValueError(
                "DSpark target_last_hidden_states must have shape [batch, seq, hidden], "
                f"got {tuple(target_last_hidden_states.shape)}"
            )
        seq_len = int(target_last_hidden_states.size(1))
        if seq_len <= 0:
            return None
        target_pred_indices = (label_indices - 1).clamp(min=0, max=seq_len - 1)
        target_pred_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            target_pred_indices,
            torch.zeros_like(target_pred_indices),
        )
        return torch.gather(
            target_last_hidden_states.unsqueeze(1).expand(
                -1, target_pred_indices.size(1), -1, -1
            ),
            2,
            target_pred_indices.unsqueeze(-1).expand(
                -1, -1, -1, target_last_hidden_states.size(-1)
            ),
        )

    def _compute_l1_loss_for_active(
        self,
        *,
        active_hidden: torch.Tensor,
        active_prev_tokens: torch.Tensor,
        active_target_hidden: torch.Tensor,
        active_weights: torch.Tensor,
        lm_head_weight: torch.Tensor,
        active_draft_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_hidden.numel() == 0:
            zero = active_weights.new_zeros(())
            return zero, zero

        l1_sum = active_weights.new_zeros((), dtype=torch.float32)
        l1_den = active_weights.float().sum()
        active_count = int(active_hidden.size(0))
        if active_draft_log_probs is not None:
            expected_shape = (active_count, int(lm_head_weight.size(0)))
            if tuple(active_draft_log_probs.shape) != expected_shape:
                raise ValueError(
                    "DSpark precomputed draft log probabilities must have shape "
                    f"{expected_shape}, got {tuple(active_draft_log_probs.shape)}"
                )
        chunk_size = self.l1_chunk_size if self.l1_chunk_size > 0 else active_count
        for start in range(0, active_count, chunk_size):
            end = min(start + chunk_size, active_count)
            hidden_chunk = active_hidden[start:end]
            prev_chunk = active_prev_tokens[start:end]
            target_hidden_chunk = active_target_hidden[start:end]
            weights_chunk = active_weights[start:end].float()

            if active_draft_log_probs is None:
                draft_logits = F.linear(hidden_chunk, lm_head_weight)
                markov_bias = self._markov_bias_for_active(
                    active_hidden=hidden_chunk,
                    active_prev_tokens=prev_chunk,
                    restricted_vocab=None,
                )
                if markov_bias is not None:
                    draft_logits = draft_logits + markov_bias
                draft_probs = torch.softmax(draft_logits.float(), dim=-1)
            else:
                draft_probs = active_draft_log_probs[start:end].exp()
            target_logits = F.linear(target_hidden_chunk, lm_head_weight)
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
            l1_sum = l1_sum + (l1_dist * weights_chunk).sum()
        return l1_sum, l1_den

    def _compute_accepted_distribution_losses(
        self,
        *,
        active_hidden: torch.Tensor,
        active_prev_tokens: torch.Tensor,
        active_target_hidden: torch.Tensor,
        active_target_ids: torch.Tensor,
        active_weights: torch.Tensor,
        lm_head_weight: torch.Tensor,
        active_draft_log_probs: Optional[torch.Tensor] = None,
        compute_l1: bool = False,
        return_diagnostics: bool = False,
        active_flat_indices: Optional[torch.Tensor] = None,
    ) -> tuple[Any, ...]:
        if active_hidden.numel() == 0:
            zero = active_weights.new_zeros((), dtype=torch.float32)
            result = (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)
            if not return_diagnostics:
                return result
            return (*result, {
                "histogram": active_weights.new_zeros(
                    self.k3_distribution_bins, dtype=torch.float32
                ),
                "min": active_weights.new_tensor(float("inf"), dtype=torch.float32),
                "max": active_weights.new_tensor(float("-inf"), dtype=torch.float32),
                "linearized_count": zero,
                "dump": {},
            })

        fkl_sum = active_weights.new_zeros((), dtype=torch.float32)
        q_sum = active_weights.new_zeros((), dtype=torch.float32)
        p_sum = active_weights.new_zeros((), dtype=torch.float32)
        log_ratio_sum = active_weights.new_zeros((), dtype=torch.float32)
        fkl_den = active_weights.new_zeros((), dtype=torch.float32)
        l1_sum = active_weights.new_zeros((), dtype=torch.float32)
        l1_den = active_weights.new_zeros((), dtype=torch.float32)
        k3_sum = active_weights.new_zeros((), dtype=torch.float32)
        k3_den = active_weights.new_zeros((), dtype=torch.float32)
        k3_clamped_count = active_weights.new_zeros((), dtype=torch.float32)
        k3_linearized_count = active_weights.new_zeros((), dtype=torch.float32)
        k3_histogram = active_weights.new_zeros(
            self.k3_distribution_bins, dtype=torch.float32
        )
        k3_min = active_weights.new_tensor(float("inf"), dtype=torch.float32)
        k3_max = active_weights.new_tensor(float("-inf"), dtype=torch.float32)
        dump_parts: dict[str, list[torch.Tensor]] = {
            "token_ids": [],
            "prev_token_ids": [],
            "target_logprobs": [],
            "draft_logprobs": [],
            "k3": [],
            "raw_k3": [],
            "weights": [],
            "clamped": [],
            "linearized": [],
            "flat_indices": [],
        }
        active_count = int(active_hidden.size(0))
        chunk_size = (
            self.l1_chunk_size
            if compute_l1 and self.l1_chunk_size > 0
            else self.ce_token_chunk_size
        )
        chunk_size = max(int(chunk_size), 1)
        max_logprob = math.log1p(-1.0e-6)
        topk_stats = {}
        for start in range(0, active_count, chunk_size):
            end = min(start + chunk_size, active_count)
            if active_draft_log_probs is None:
                draft_logits = self._full_vocab_logits(
                    active_hidden[start:end],
                    lm_head_weight,
                    active_prev_tokens[start:end],
                )
                draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
            else:
                draft_log_probs = active_draft_log_probs[start:end]
            target_logits = F.linear(active_target_hidden[start:end], lm_head_weight)
            target_log_probs = F.log_softmax(target_logits.float(), dim=-1).detach()
            token_indices = active_target_ids[start:end].long().unsqueeze(-1)
            raw_log_q = draft_log_probs.gather(-1, token_indices).squeeze(-1)
            raw_log_p = target_log_probs.gather(-1, token_indices).squeeze(-1)
            log_q = raw_log_q.clamp(min=-10.0, max=max_logprob)
            log_p = raw_log_p.clamp(min=-10.0, max=max_logprob)
            q = log_q.exp()
            p = log_p.exp()
            fkl = p * (log_p - log_q) + (1.0 - p) * (
                torch.log1p(-p) - torch.log1p(-q)
            )
            weights = active_weights[start:end].float()
            if self.response_loss == "topk":
                chunk_topk = student_support_reverse_kl(
                    draft_log_probs, target_log_probs, weights, topk=self.topk,
                    support=self.topk_support,
                )
                for key, value in chunk_topk.items():
                    topk_stats[key] = topk_stats.get(key, 0) + value
            finite = torch.isfinite(fkl) & torch.isfinite(q) & torch.isfinite(p)
            effective_weights = weights * finite.float()
            fkl = torch.where(finite, fkl, torch.zeros_like(fkl))
            fkl_sum = fkl_sum + (fkl * effective_weights).sum()
            fkl_den = fkl_den + effective_weights.sum()
            q_sum = q_sum + (q.detach() * effective_weights).sum()
            p_sum = p_sum + (p.detach() * effective_weights).sum()
            log_ratio_sum = log_ratio_sum + (
                (log_q.detach() - log_p.detach()) * effective_weights
            ).sum()
            delta = (raw_log_p - raw_log_q).clamp(
                min=-self.rejected_log_ratio_clamp,
                max=self.rejected_log_ratio_clamp,
            )
            k3, raw_k3, legacy_clamped, linearized = self._apply_k3_tail_policy(
                delta,
                asymmetric_enabled=self.accepted_asymmetric_k3_enabled,
            )
            finite_k3 = (
                torch.isfinite(k3)
                & torch.isfinite(raw_log_q)
                & torch.isfinite(raw_log_p)
            )
            k3_weights = weights * finite_k3.float()
            k3 = torch.where(finite_k3, k3, torch.zeros_like(k3))
            k3_sum = k3_sum + (k3 * k3_weights).sum()
            k3_den = k3_den + k3_weights.sum()
            k3_clamped_count = k3_clamped_count + (
                (finite_k3 & legacy_clamped).float() * weights
            ).sum()
            k3_linearized_count = k3_linearized_count + (
                (finite_k3 & linearized).float() * weights
            ).sum()
            distribution = self._k3_distribution_stats(k3, weights, finite_k3)
            k3_histogram = k3_histogram + distribution["histogram"]
            k3_min = torch.minimum(k3_min, distribution["min"])
            k3_max = torch.maximum(k3_max, distribution["max"])
            if self.dump_k3_logprobs:
                dump_mask = finite_k3 & (weights > 0)
                dump_parts["token_ids"].append(
                    active_target_ids[start:end][dump_mask].detach()
                )
                dump_parts["prev_token_ids"].append(
                    active_prev_tokens[start:end][dump_mask].detach()
                )
                dump_parts["target_logprobs"].append(raw_log_p[dump_mask].detach())
                dump_parts["draft_logprobs"].append(raw_log_q[dump_mask].detach())
                dump_parts["k3"].append(k3[dump_mask].detach())
                dump_parts["raw_k3"].append(raw_k3[dump_mask].detach())
                dump_parts["weights"].append(weights[dump_mask].detach())
                dump_parts["clamped"].append(legacy_clamped[dump_mask].detach())
                dump_parts["linearized"].append(linearized[dump_mask].detach())
                if active_flat_indices is not None:
                    dump_parts["flat_indices"].append(
                        active_flat_indices[start:end][dump_mask].detach()
                    )
            if compute_l1:
                l1_dist = (
                    draft_log_probs.exp() - target_log_probs.exp()
                ).abs().sum(dim=-1)
                finite_l1 = torch.isfinite(l1_dist)
                l1_weights = weights * finite_l1.float()
                l1_sum = l1_sum + (
                    torch.where(finite_l1, l1_dist, torch.zeros_like(l1_dist))
                    * l1_weights
                ).sum()
                l1_den = l1_den + l1_weights.sum()
        result = (
            fkl_sum,
            fkl_den,
            q_sum,
            p_sum,
            log_ratio_sum,
            l1_sum,
            l1_den,
            k3_sum,
            k3_den,
            k3_clamped_count,
        )
        if not return_diagnostics:
            return result
        dump = {
            key: torch.cat(parts, dim=0) if parts else active_weights.new_empty(0)
            for key, parts in dump_parts.items()
        }
        return (*result, {
            "histogram": k3_histogram,
            "min": k3_min,
            "max": k3_max,
            "linearized_count": k3_linearized_count,
            "topk": topk_stats,
            "dump": dump,
        })

    def _compute_rejected_sampled_k3(
        self,
        *,
        draft_hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        rejected_token_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        weights: torch.Tensor,
        lm_head_weight: torch.Tensor,
        return_diagnostics: bool = False,
        flat_indices: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        topk_teacher_logprobs: Optional[torch.Tensor] = None,
    ) -> tuple[Any, ...]:
        if draft_hidden.numel() == 0:
            zero = weights.new_zeros((), dtype=torch.float32)
            result = (zero, zero, zero, zero, zero, zero)
            if not return_diagnostics:
                return result
            return (*result, {
                "histogram": weights.new_zeros(
                    self.k3_distribution_bins, dtype=torch.float32
                ),
                "min": weights.new_tensor(float("inf"), dtype=torch.float32),
                "max": weights.new_tensor(float("-inf"), dtype=torch.float32),
                "linearized_count": zero,
                "dump": {},
            })

        draft_logits = self._full_vocab_logits(
            draft_hidden, lm_head_weight, prev_token_ids
        )
        draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
        token_indices = rejected_token_ids.long().unsqueeze(-1)
        log_q = draft_log_probs.gather(-1, token_indices).squeeze(-1)
        log_p = teacher_logprobs.float().detach()
        # Asymmetric K3 has a bounded negative-tail derivative (exp(delta)-1
        # approaches -1). A lower log-ratio clamp would silently discard this
        # corrective gradient for confident teacher/drafter disagreements.
        # Keep the positive cap and preserve both caps for the legacy policy.
        delta = (log_p - log_q).clamp(
            min=(
                None if self.rejected_asymmetric_k3_enabled
                else -self.rejected_log_ratio_clamp
            ),
            max=self.rejected_log_ratio_clamp,
        )
        k3, raw_k3, legacy_clamped, linearized = self._apply_k3_tail_policy(
            delta,
            asymmetric_enabled=self.rejected_asymmetric_k3_enabled,
        )
        finite = (
            torch.isfinite(k3)
            & torch.isfinite(log_q)
            & torch.isfinite(log_p)
        )
        effective_weights = weights.float() * finite.float()
        k3 = torch.where(finite, k3, torch.zeros_like(k3))
        clamped = finite & legacy_clamped
        result = (
            (k3 * effective_weights).sum(),
            effective_weights.sum(),
            (log_q.detach().exp() * effective_weights).sum(),
            (log_p.detach().exp() * effective_weights).sum(),
            (delta.detach() * effective_weights).sum(),
            (clamped.float() * weights.float()).sum(),
        )
        if not return_diagnostics:
            return result
        distribution = self._k3_distribution_stats(k3, weights, finite)
        if self.rejected_loss == "topk":
            if topk_ids is None or topk_teacher_logprobs is None:
                raise ValueError(
                    "DSpark rejected topk requires verified rollout candidate metadata"
                )
            distribution["topk"] = student_support_reverse_kl(
                draft_log_probs,
                topk_teacher_logprobs,
                weights,
                topk=self.topk,
                candidate_ids=topk_ids,
                support=self.topk_support,
            )
        distribution["linearized_count"] = (
            (finite & linearized).float() * weights.float()
        ).sum()
        dump_mask = finite & (weights > 0)
        distribution["dump"] = (
            {
                "token_ids": rejected_token_ids[dump_mask].detach(),
                "prev_token_ids": prev_token_ids[dump_mask].detach(),
                "target_logprobs": log_p[dump_mask].detach(),
                "draft_logprobs": log_q[dump_mask].detach(),
                "k3": k3[dump_mask].detach(),
                "raw_k3": raw_k3[dump_mask].detach(),
                "weights": weights[dump_mask].detach(),
                "clamped": clamped[dump_mask].detach(),
                "linearized": linearized[dump_mask].detach(),
                **(
                    {"flat_indices": flat_indices[dump_mask].detach()}
                    if flat_indices is not None
                    else {}
                ),
            }
            if self.dump_k3_logprobs
            else {}
        )
        return (*result, distribution)

    def _should_debug_log(self) -> bool:
        if not self.debug_log:
            return False
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank != 0:
            return False
        count = self._debug_forward_count
        return count <= self.debug_log_first_n or count % self.debug_log_interval == 0

    def _log_debug_preview(
        self,
        *,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        target_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        active_logits: Optional[torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        if not self._should_debug_log():
            return
        valid_anchor = torch.nonzero(block_keep_mask, as_tuple=False)
        preview: dict[str, Any] = {}
        if valid_anchor.numel() > 0:
            batch_idx = int(valid_anchor[0, 0].detach().cpu().item())
            block_idx = int(valid_anchor[0, 1].detach().cpu().item())
            anchor = int(anchor_positions[batch_idx, block_idx].detach().cpu().item())
            preview = {
                "anchor": anchor,
                "anchor_token": int(input_ids[batch_idx, anchor].detach().cpu().item()),
                "labels": target_ids[batch_idx, block_idx]
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
                "prev": prev_token_ids[batch_idx, block_idx]
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
                "eval": eval_mask[batch_idx, block_idx]
                .to(torch.int32)
                .detach()
                .cpu()
                .tolist()[: min(self.block_size, 8)],
            }
        logger.info(
            "[dspark-trainer] batch input_shape=%s loss_tokens=%s anchors=%s sampled=%s preview=%s logits_shape=%s loss=%.6f",
            tuple(input_ids.shape),
            int(eval_mask.detach().sum().cpu().item()),
            int(block_keep_mask.numel()),
            int(block_keep_mask.detach().sum().cpu().item()),
            preview,
            tuple(active_logits.shape) if active_logits is not None else None,
            float(loss.detach().float().cpu().item()),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: list[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        trace_anchor_positions: Optional[torch.Tensor] = None,
        trace_anchor_mask: Optional[torch.Tensor] = None,
        trace_accepted_counts: Optional[torch.Tensor] = None,
        trace_draft_counts: Optional[torch.Tensor] = None,
        trace_rejected_token_ids: Optional[torch.Tensor] = None,
        trace_draft_token_ids: Optional[torch.Tensor] = None,
        trace_draft_teacher_logprobs: Optional[torch.Tensor] = None,
        trace_draft_teacher_logprob_mask: Optional[torch.Tensor] = None,
        trace_first_rejection_teacher_entropies: Optional[torch.Tensor] = None,
        trace_bonus_response_anchor_mask: Optional[torch.Tensor] = None,
        trace_draft_topk_ids: Optional[torch.Tensor] = None,
        trace_draft_topk_teacher_logprobs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        global_context_mask: Optional[torch.Tensor] = None,
        return_first_token_logits: bool = False,
    ):
        bsz, seq_len = input_ids.shape
        response_topk_stats = {}
        rejected_topk_stats = {}
        device = input_ids.device
        self._debug_forward_count += 1
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            trace_anchor_positions=trace_anchor_positions,
            trace_anchor_mask=trace_anchor_mask,
        )
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        requested_position_ids = position_ids if self.use_absolute_position_ids else None
        if self.use_absolute_position_ids and requested_position_ids is None:
            raise ValueError(
                "DSpark absolute position IDs are enabled but the training batch "
                "has no position_ids"
            )
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, requested_position_ids
        )
        draft_len = n_blocks * self.block_size

        block_mask = None
        dense_attention_mask = None
        if global_context_mask is not None:
            if global_context_mask.shape != (bsz, seq_len):
                raise ValueError(
                    "global_context_mask must have shape [batch, context_length], "
                    f"got {tuple(global_context_mask.shape)}"
                )
            global_context_mask = global_context_mask.to(device=device, dtype=torch.bool)

        def build_attention_mask(
            context_window: Optional[int],
            selected_global_context: Optional[torch.Tensor] = None,
        ):
            if device.type == "cuda":
                return compile_friendly_create_block_mask(
                    mask_mod=_create_dflash_mask_mod(
                        anchor_positions,
                        block_keep_mask,
                        seq_len,
                        self.block_size,
                        context_window=context_window,
                        global_context_mask=selected_global_context,
                    ),
                    B=bsz,
                    H=None,
                    Q_LEN=draft_len,
                    KV_LEN=seq_len + draft_len,
                    device=device,
                )
            return _create_dflash_dense_attention_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                context_window=context_window,
                global_context_mask=selected_global_context,
            )

        local_window = self.context_window if self.attention_mode != "full" else None
        local_mask = build_attention_mask(local_window)
        layer_masks = None
        if self.attention_mode in {"hybrid", "sparse_global"}:
            selected_global_context = (
                global_context_mask if self.attention_mode == "sparse_global" else None
            )
            global_window = (
                self.context_window if self.attention_mode == "sparse_global" else None
            )
            global_mask = build_attention_mask(
                global_window, selected_global_context=selected_global_context
            )
            local_layer_count = len(self.draft_model.layers) - self.global_layer_count
            layer_masks = [local_mask] * local_layer_count + [
                global_mask
            ] * self.global_layer_count
        if device.type == "cuda":
            block_mask = layer_masks if layer_masks is not None else local_mask
        else:
            dense_attention_mask = layer_masks if layer_masks is not None else local_mask

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            dense_attention_mask=dense_attention_mask,
            noise_embedding=noise_embedding,
        ).view(bsz, n_blocks, self.block_size, -1)

        if return_first_token_logits:
            first_hidden = draft_hidden[:, :, 0, :].reshape(-1, draft_hidden.size(-1))
            first_prev_tokens = torch.gather(
                input_ids, 1, anchor_positions.clamp(0, seq_len - 1)
            ).reshape(-1)
            return self._full_vocab_logits(
                first_hidden, lm_head_weight, first_prev_tokens
            ).view(bsz, n_blocks, -1)

        target_ids, prev_token_ids, eval_mask, label_indices = (
            self._build_label_tensors(
                input_ids=input_ids,
                loss_mask=loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
            )
        )
        aligned_target_hidden = self._gather_aligned_target_hidden(
            target_last_hidden_states=target_last_hidden_states,
            label_indices=label_indices,
            block_keep_mask=block_keep_mask,
        )

        weight_mask = eval_mask.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = self._cached_arange(
                "dspark_decay_positions", self.block_size, device, view_shape=(1, 1, -1)
            )
            weight_mask = weight_mask * torch.exp(
                -positions.float() / float(self.loss_decay_gamma)
            )

        sanitized_rows = torch.zeros((), dtype=torch.float32, device=device)
        local_ploss_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_ce_den = torch.zeros((), dtype=torch.float32, device=device)
        local_first_ce_sum = (
            draft_hidden.reshape(-1)[:1].sum() * 0.0
            if self.first_rejection_ce_alpha > 0 else local_ce_den.clone()
        )
        local_first_ce_count = local_ce_den.clone()
        local_first_ce_weight_sum = local_ce_den.clone()
        local_first_ce_boosted_count = local_ce_den.clone()
        local_first_ce_q_sum = local_ce_den.clone()
        local_l1_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_l1_den = torch.zeros((), dtype=torch.float32, device=device)
        local_accepted_fkl_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_accepted_fkl_den = torch.zeros((), dtype=torch.float32, device=device)
        local_accepted_k3_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_accepted_k3_den = torch.zeros((), dtype=torch.float32, device=device)
        accepted_k3_histogram = torch.zeros(
            self.k3_distribution_bins, dtype=torch.float32, device=device
        )
        accepted_k3_min = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        accepted_k3_max = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
        accepted_k3_dump: dict[str, torch.Tensor] = {}
        accepted_k3_clamped_count = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        accepted_k3_linearized_count = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        accepted_q_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
        accepted_p_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
        accepted_log_ratio_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_rejected_k3_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_rejected_k3_den = torch.zeros((), dtype=torch.float32, device=device)
        rejected_k3_histogram = torch.zeros(
            self.k3_distribution_bins, dtype=torch.float32, device=device
        )
        rejected_k3_min = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        rejected_k3_max = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
        rejected_k3_dump: dict[str, torch.Tensor] = {}
        rejected_q_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
        rejected_p_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
        rejected_log_ratio_sum = torch.zeros((), dtype=torch.float32, device=device)
        rejected_k3_clamped_count = torch.zeros((), dtype=torch.float32, device=device)
        rejected_k3_linearized_count = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        objective_weighted_token_count = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        response_objective_sum = torch.zeros((), dtype=torch.float32, device=device)
        rejected_objective_sum = torch.zeros((), dtype=torch.float32, device=device)
        effective_response_stream_weight = torch.tensor(
            self.response_stream_weight, dtype=torch.float32, device=device
        )
        effective_rejected_stream_weight = torch.tensor(
            self.rejected_stream_weight, dtype=torch.float32, device=device
        )
        response_stream_weight_multiplier = torch.ones(
            (), dtype=torch.float32, device=device
        )
        rejected_stream_weight_multiplier = torch.ones(
            (), dtype=torch.float32, device=device
        )
        accepted_trace_positions = torch.zeros((), dtype=torch.float32, device=device)
        first_rejection_positions = torch.zeros((), dtype=torch.float32, device=device)
        unverified_trace_positions = torch.zeros((), dtype=torch.float32, device=device)
        all_accepted_trace_anchors = torch.zeros((), dtype=torch.float32, device=device)
        bonus_response_positions = torch.zeros((), dtype=torch.float32, device=device)
        rejection_position_mask = torch.zeros_like(eval_mask)
        first_rejection_target_mask = torch.zeros_like(eval_mask)
        rejected_prev_token_ids = prev_token_ids
        rejected_draft_token_ids = target_ids
        rejected_teacher_logprobs = weight_mask.new_zeros(weight_mask.shape)
        rejected_weights = weight_mask.new_zeros(weight_mask.shape)
        if self.trace_aligned_anchors:
            trace_tensors = (
                trace_accepted_counts,
                trace_draft_counts,
                trace_rejected_token_ids,
                trace_draft_token_ids,
                trace_draft_teacher_logprobs,
                trace_draft_teacher_logprob_mask,
            )
            if any(value is None for value in trace_tensors):
                raise ValueError(
                    "DSpark trace-aligned anchor mode requires accepted, draft, and rejected-token metadata"
                )
            accepted_counts = trace_accepted_counts.to(device=device, dtype=torch.long)
            draft_counts = trace_draft_counts.to(device=device, dtype=torch.long)
            rejected_ids = trace_rejected_token_ids.to(device=device, dtype=torch.long)
            draft_ids = trace_draft_token_ids.to(device=device, dtype=torch.long)
            teacher_logprobs = trace_draft_teacher_logprobs.to(
                device=device, dtype=torch.float32
            )
            teacher_logprob_mask = trace_draft_teacher_logprob_mask.to(
                device=device, dtype=torch.bool
            )
            expected_shape = anchor_positions.shape
            if any(
                value.shape != expected_shape
                for value in (accepted_counts, draft_counts, rejected_ids)
            ):
                raise ValueError(
                    "DSpark trace metadata must match the selected [batch, anchors] shape"
                )
            if trace_bonus_response_anchor_mask is None:
                bonus_response_anchor_mask = torch.zeros_like(
                    block_keep_mask, dtype=torch.bool
                )
            else:
                bonus_response_anchor_mask = trace_bonus_response_anchor_mask.to(
                    device=device, dtype=torch.bool
                )
                if bonus_response_anchor_mask.shape != expected_shape:
                    raise ValueError(
                        "DSpark bonus response anchor mask must match the selected "
                        "[batch, anchors] shape"
                    )
            expected_token_shape = (*expected_shape, self.block_size)
            if any(
                value.shape != expected_token_shape
                for value in (draft_ids, teacher_logprobs, teacher_logprob_mask)
            ):
                raise ValueError(
                    "DSpark draft trace metadata must match [batch, anchors, block_size]"
                )
            trace_offsets = self._cached_arange(
                "dspark_trace_offsets", self.block_size, device, view_shape=(1, 1, -1)
            )
            accepted_mask = (
                (trace_offsets < accepted_counts.unsqueeze(-1))
                & (trace_offsets < draft_counts.unsqueeze(-1))
                & block_keep_mask.unsqueeze(-1)
                & eval_mask
            )
            first_rejection_target_mask = (
                (trace_offsets == accepted_counts.unsqueeze(-1))
                & (accepted_counts < draft_counts).unsqueeze(-1)
                & block_keep_mask.unsqueeze(-1)
                & eval_mask
            )
            rejection_position_mask = (
                (trace_offsets >= accepted_counts.unsqueeze(-1))
                & (trace_offsets < draft_counts.unsqueeze(-1))
                & (accepted_counts < draft_counts).unsqueeze(-1)
                & block_keep_mask.unsqueeze(-1)
                & teacher_logprob_mask
                & (draft_ids >= 0)
            )
            if self.rejected_prefix_length > 0:
                rejection_position_mask = rejection_position_mask & (
                    trace_offsets - accepted_counts.unsqueeze(-1)
                    < self.rejected_prefix_length
                )
            if self.rejected_loss == "topk":
                required_rejected = (
                    (trace_offsets >= accepted_counts.unsqueeze(-1))
                    & (trace_offsets < draft_counts.unsqueeze(-1))
                    & block_keep_mask.unsqueeze(-1)
                )
                if self.rejected_prefix_length > 0:
                    required_rejected = required_rejected & (
                        trace_offsets - accepted_counts.unsqueeze(-1)
                        < self.rejected_prefix_length
                    )
                if (required_rejected & ~rejection_position_mask).any():
                    raise ValueError(
                        "DSpark rejected topk cannot skip missing verified teacher metadata"
                    )
            unverified_mask = (
                (trace_offsets > accepted_counts.unsqueeze(-1))
                & (trace_offsets < draft_counts.unsqueeze(-1))
                & (accepted_counts < draft_counts).unsqueeze(-1)
                & block_keep_mask.unsqueeze(-1)
            )
            accepted_trace_positions = accepted_mask.float().sum()
            bonus_response_positions = (
                accepted_mask & bonus_response_anchor_mask.unsqueeze(-1)
            ).float().sum()
            first_rejection_positions = first_rejection_target_mask.float().sum()
            unverified_trace_positions = unverified_mask.float().sum()
            all_accepted_trace_anchors = (
                (accepted_counts >= draft_counts)
                & block_keep_mask
                & ~bonus_response_anchor_mask
            ).float().sum()
            anchor_token_ids = torch.gather(
                input_ids,
                1,
                anchor_positions.clamp(0, max(seq_len - 1, 0)),
            )
            rejected_prev_token_ids = torch.cat(
                [anchor_token_ids.unsqueeze(-1), draft_ids[:, :, :-1]], dim=-1
            ).clamp(min=0)
            rejected_draft_token_ids = draft_ids.clamp(min=0)
            rejected_teacher_logprobs = teacher_logprobs
            rejected_weights = torch.pow(
                weight_mask.new_tensor(self.rejected_position_decay),
                trace_offsets.float(),
            ).expand_as(weight_mask)
            if (
                self.rejected_first_weight_multiplier != 1.0
                or self.rejected_suffix_weight_multiplier != 1.0
            ):
                # Split relative to the actual first rejection, not block position 0.
                rejected_weights = rejected_weights * torch.where(
                    trace_offsets == accepted_counts.unsqueeze(-1),
                    self.rejected_first_weight_multiplier,
                    self.rejected_suffix_weight_multiplier,
                )
            # Positive supervision follows the target trajectory. Rejected K3
            # follows the recursively proposed draft branch with its own weights.
            weight_mask = weight_mask * (
                accepted_mask | first_rejection_target_mask
            ).float()

        flat_targets = target_ids.reshape(-1)
        flat_prev_tokens = prev_token_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        active_mask = flat_weights > 0
        flat_hidden = draft_hidden.reshape(-1, draft_hidden.size(-1))
        active_hidden = flat_hidden[active_mask]
        active_targets = flat_targets[active_mask]
        active_prev_tokens = flat_prev_tokens[active_mask]
        active_weights = flat_weights[active_mask]
        active_flat_indices = active_mask.nonzero(as_tuple=False).flatten()
        active_target_hidden = (
            aligned_target_hidden.reshape(-1, aligned_target_hidden.size(-1))[
                active_mask
            ]
            if aligned_target_hidden is not None
            else None
        )
        loss_per_token = torch.zeros_like(flat_weights)
        active_logits = None
        active_log_probs = None
        restricted_vocab = None
        if active_targets.numel() == 0:
            loss = flat_weights.sum() * 0.0
        else:
            if self.loss_mode == "sampled_ce":
                restricted_vocab = self._build_restricted_vocab(
                    input_ids, active_targets, int(lm_head_weight.shape[0])
                )
                restricted_weight = lm_head_weight.index_select(0, restricted_vocab)
                active_logits = F.linear(active_hidden, restricted_weight)
                markov_bias = self._markov_bias_for_active(
                    active_hidden=active_hidden,
                    active_prev_tokens=active_prev_tokens,
                    restricted_vocab=restricted_vocab,
                )
                if markov_bias is not None:
                    active_logits = active_logits + markov_bias
                active_ce_targets = torch.searchsorted(restricted_vocab, active_targets)
                active_loss = F.cross_entropy(
                    active_logits, active_ce_targets, reduction="none"
                )
            elif self.loss_mode == "chunked_ce":
                active_loss = self._chunked_exact_cross_entropy(
                    active_hidden=active_hidden,
                    active_targets=active_targets,
                    lm_head_weight=lm_head_weight,
                    active_prev_tokens=active_prev_tokens,
                )
            else:
                active_logits = self._full_vocab_logits(
                    active_hidden, lm_head_weight, active_prev_tokens
                )
                active_log_probs = F.log_softmax(active_logits.float(), dim=-1)
                active_loss = F.nll_loss(
                    active_log_probs, active_targets, reduction="none"
                )

            finite_loss = torch.isfinite(active_loss)
            sanitized_rows = (~finite_loss).sum().to(dtype=torch.float32)
            active_loss = torch.where(
                finite_loss, active_loss, torch.zeros_like(active_loss)
            )
            active_loss_weights = active_weights * finite_loss.to(
                dtype=active_weights.dtype
            )
            loss_per_token[active_mask] = active_loss
            local_ce_den = active_loss_weights.sum()
            local_ploss_sum = (active_loss * active_loss_weights).sum()
            if self.first_rejection_ce_alpha > 0:
                # Correct response replacement at the first rejection only.
                # Do not inherit stream weights or absolute-position decay:
                # this auxiliary objective has its own global token mean.
                first_active = (
                    first_rejection_target_mask.reshape(-1)[active_mask] & finite_loss
                )
                first_weights = torch.ones_like(active_loss[first_active])
                if self.first_rejection_ce_low_entropy_multiplier > 1 and first_active.any():
                    if trace_first_rejection_teacher_entropies is None:
                        raise ValueError("Weighted first-rejection CE requires verify entropies")
                    if trace_first_rejection_teacher_entropies.shape != anchor_positions.shape:
                        raise ValueError("First-rejection verify entropies must match [batch, anchors]")
                    first_entropies = trace_first_rejection_teacher_entropies.to(
                        device=device, dtype=torch.float32
                    ).unsqueeze(-1).expand_as(first_rejection_target_mask)
                    first_entropies = first_entropies.reshape(-1)[active_mask][first_active]
                    first_logprob_mask = teacher_logprob_mask.reshape(-1)[active_mask][first_active]
                    first_logprobs = teacher_logprobs.reshape(-1)[active_mask][first_active]
                    if not torch.isfinite(first_entropies).all() or not first_logprob_mask.all():
                        raise ValueError("Weighted first-rejection CE requires finite entropy and candidate logprob")
                    boosted = (
                        (first_entropies < self.first_rejection_ce_low_entropy_threshold)
                        & (first_logprobs <= self.first_rejection_ce_max_rejected_logprob)
                    )
                    first_weights = first_weights + boosted.to(first_weights.dtype) * (
                        self.first_rejection_ce_low_entropy_multiplier - 1.0
                    )
                    local_first_ce_boosted_count = boosted.float().sum()
                local_first_ce_sum = (active_loss[first_active] * first_weights).sum()
                local_first_ce_count = first_active.float().sum()
                local_first_ce_weight_sum = first_weights.sum()
                local_first_ce_q_sum = (-active_loss[first_active].detach()).exp().sum()
            needs_accepted_distribution = (
                self.accepted_k3_alpha > 0 or self.accepted_fkl_alpha > 0
            )
            if needs_accepted_distribution:
                if active_target_hidden is None:
                    raise ValueError(
                        "DSpark accepted K3/FKL requires target_last_hidden_states"
                    )
                (
                    local_accepted_fkl_sum,
                    local_accepted_fkl_den,
                    accepted_q_prob_sum,
                    accepted_p_prob_sum,
                    accepted_log_ratio_sum,
                    joint_l1_sum,
                    joint_l1_den,
                    local_accepted_k3_sum,
                    local_accepted_k3_den,
                    accepted_k3_clamped_count,
                    accepted_distribution,
                ) = self._compute_accepted_distribution_losses(
                    active_hidden=active_hidden[finite_loss],
                    active_prev_tokens=active_prev_tokens[finite_loss],
                    active_target_hidden=active_target_hidden[finite_loss],
                    active_target_ids=active_targets[finite_loss],
                    active_weights=active_loss_weights[finite_loss],
                    lm_head_weight=lm_head_weight,
                    active_draft_log_probs=(
                        active_log_probs[finite_loss]
                        if active_log_probs is not None
                        else None
                    ),
                    compute_l1=self.l1_loss_alpha > 0,
                    return_diagnostics=True,
                    active_flat_indices=active_flat_indices[finite_loss],
                )
                accepted_k3_histogram = accepted_distribution["histogram"]
                accepted_k3_min = accepted_distribution["min"]
                accepted_k3_max = accepted_distribution["max"]
                accepted_k3_linearized_count = accepted_distribution[
                    "linearized_count"
                ]
                accepted_k3_dump = accepted_distribution["dump"]
                response_topk_stats = accepted_distribution.get("topk", {})
                if self.l1_loss_alpha > 0:
                    local_l1_sum = joint_l1_sum
                    local_l1_den = joint_l1_den
            if self.l1_loss_alpha > 0:
                if active_target_hidden is None:
                    raise ValueError(
                        "DSpark L1 loss requires target_last_hidden_states. "
                        "Enable old-logprob dflash_aux_plus_last collection or set dspark_l1_loss_alpha=0."
                    )
                if not needs_accepted_distribution:
                    finite_target_hidden = torch.isfinite(active_target_hidden).all(
                        dim=-1
                    )
                    l1_mask = finite_loss & finite_target_hidden
                    if l1_mask.any():
                        reusable_draft_log_probs = None
                        # Non-chunked full-vocab CE already has reusable probabilities.
                        if active_log_probs is not None:
                            reusable_draft_log_probs = (
                                active_log_probs
                                if l1_mask.all()
                                else active_log_probs[l1_mask]
                            )
                        local_l1_sum, local_l1_den = (
                            self._compute_l1_loss_for_active(
                                active_hidden=active_hidden[l1_mask],
                                active_prev_tokens=active_prev_tokens[l1_mask],
                                active_target_hidden=active_target_hidden[l1_mask],
                                active_weights=active_loss_weights[l1_mask],
                                lm_head_weight=lm_head_weight,
                                active_draft_log_probs=reusable_draft_log_probs,
                            )
                        )
            if rejection_position_mask.any() and (
                self.rejected_k3_alpha > 0 or self.rejected_aux_alpha > 0
            ):
                if self.rejected_loss == "topk":
                    expected_shape = (
                        *rejection_position_mask.shape,
                        topk_candidate_width(self.topk, self.topk_support),
                    )
                    if (
                        trace_draft_topk_ids is None
                        or trace_draft_topk_teacher_logprobs is None
                        or tuple(trace_draft_topk_ids.shape) != expected_shape
                        or tuple(trace_draft_topk_teacher_logprobs.shape) != expected_shape
                    ):
                        raise ValueError(
                            "DSpark rejected topk requires aligned [batch, anchor, block, k] metadata"
                        )
                (
                    local_rejected_k3_sum,
                    local_rejected_k3_den,
                    rejected_q_prob_sum,
                    rejected_p_prob_sum,
                    rejected_log_ratio_sum,
                    rejected_k3_clamped_count,
                    rejected_distribution,
                ) = self._compute_rejected_sampled_k3(
                    draft_hidden=draft_hidden[rejection_position_mask],
                    prev_token_ids=rejected_prev_token_ids[rejection_position_mask],
                    rejected_token_ids=rejected_draft_token_ids[
                        rejection_position_mask
                    ],
                    teacher_logprobs=rejected_teacher_logprobs[
                        rejection_position_mask
                    ],
                    weights=rejected_weights[rejection_position_mask],
                    lm_head_weight=lm_head_weight,
                    return_diagnostics=True,
                    flat_indices=rejection_position_mask.reshape(-1).nonzero(
                        as_tuple=False
                    ).flatten(),
                    topk_ids=(
                        trace_draft_topk_ids.to(device)[rejection_position_mask]
                        if self.rejected_loss == "topk"
                        else None
                    ),
                    topk_teacher_logprobs=(
                        trace_draft_topk_teacher_logprobs.to(device)[rejection_position_mask]
                        if self.rejected_loss == "topk"
                        else None
                    ),
                )
                rejected_k3_histogram = rejected_distribution["histogram"]
                rejected_k3_min = rejected_distribution["min"]
                rejected_k3_max = rejected_distribution["max"]
                rejected_k3_linearized_count = rejected_distribution[
                    "linearized_count"
                ]
                rejected_k3_dump = rejected_distribution["dump"]
                rejected_topk_stats = rejected_distribution.get("topk", {})
            response_stream_enabled = any(
                alpha > 0
                for alpha in (
                    self.ce_loss_alpha,
                    self.l1_loss_alpha,
                    self.accepted_k3_alpha,
                    self.accepted_fkl_alpha,
                )
            ) and self.response_stream_weight > 0
            rejected_stream_enabled = (
                self.rejected_k3_alpha > 0 and self.rejected_stream_weight > 0
            )
            response_distribution_sum = local_accepted_k3_sum
            rejected_distribution_sum = local_rejected_k3_sum
            if self.response_loss == "topk":
                response_distribution_sum = response_topk_stats.get(
                    "loss_sum", local_ploss_sum.new_zeros(())
                )
            if self.rejected_loss == "topk":
                rejected_distribution_sum = rejected_topk_stats.get(
                    "loss_sum", local_ploss_sum.new_zeros(())
                )
            response_objective_sum = (
                (local_ploss_sum * self.ce_loss_alpha)
                + (local_l1_sum * self.l1_loss_alpha)
                + (response_distribution_sum * self.accepted_k3_alpha)
                + (local_accepted_fkl_sum * self.accepted_fkl_alpha)
            )
            rejected_objective_sum = (
                rejected_distribution_sum * self.rejected_k3_alpha
            )
            (
                effective_response_stream_weight,
                effective_rejected_stream_weight,
                response_stream_weight_multiplier,
                rejected_stream_weight_multiplier,
            ) = self._effective_stream_weights(
                response_token_count=local_ce_den,
                rejected_token_count=local_rejected_k3_den,
                response_stream_enabled=response_stream_enabled,
                rejected_stream_enabled=rejected_stream_enabled,
            )
            weighted_loss_sum = local_ploss_sum.new_zeros(())
            if response_stream_enabled:
                weighted_loss_sum = weighted_loss_sum + (
                    response_objective_sum * effective_response_stream_weight
                )
                objective_weighted_token_count = (
                    objective_weighted_token_count
                    + local_ce_den * effective_response_stream_weight
                )
            if rejected_stream_enabled:
                weighted_loss_sum = weighted_loss_sum + (
                    rejected_objective_sum * effective_rejected_stream_weight
                )
                objective_weighted_token_count = (
                    objective_weighted_token_count
                    + local_rejected_k3_den * effective_rejected_stream_weight
                )
            loss = weighted_loss_sum / objective_weighted_token_count.clamp(min=1e-6)

        base_loss = loss
        if self.first_rejection_ce_alpha > 0:
            # Standalone forward returns the complete local objective. The
            # backend exposes the two sums separately for exact distributed
            # normalization; averaging these local CE means would be biased.
            loss = loss + self.first_rejection_ce_alpha * (
                local_first_ce_sum / local_first_ce_weight_sum.clamp(min=1.0)
            )
        if self.rejected_aux_alpha > 0:
            loss = loss + self.rejected_aux_alpha * (
                local_rejected_k3_sum / local_rejected_k3_den.clamp(min=1.0)
            )

        with torch.no_grad():
            flat_eval_mask = eval_mask.reshape(-1)
            binary_eval_mask = flat_eval_mask & (flat_weights > 0)
            correct = torch.zeros_like(flat_weights, dtype=torch.bool)
            top1_correct = torch.zeros((), dtype=torch.float32, device=device)
            top5_correct = torch.zeros((), dtype=torch.float32, device=device)
            quality_token_count = torch.zeros((), dtype=torch.float32, device=device)
            sampled_vocab_size = torch.tensor(
                float(restricted_vocab.numel())
                if restricted_vocab is not None
                else float(lm_head_weight.shape[0]),
                dtype=torch.float32,
                device=device,
            )
            if active_targets.numel() > 0:
                if active_logits is None:
                    top_tokens = self._chunked_full_vocab_topk(
                        active_hidden=active_hidden,
                        lm_head_weight=lm_head_weight,
                        k=5,
                        active_prev_tokens=active_prev_tokens,
                    )
                    pred_tokens = top_tokens[:, 0]
                elif restricted_vocab is not None:
                    pred_local = active_logits.argmax(dim=-1)
                    pred_tokens = restricted_vocab[pred_local]
                    topk = min(5, int(restricted_vocab.numel()))
                    top_tokens = restricted_vocab[
                        active_logits.topk(topk, dim=-1).indices
                    ]
                else:
                    pred_tokens = active_logits.argmax(dim=-1)
                    topk = min(5, active_logits.shape[-1])
                    top_tokens = active_logits.topk(topk, dim=-1).indices
                active_correct = pred_tokens.eq(active_targets)
                correct[active_mask] = active_correct
                top1_correct = active_correct.float().sum()
                top5_correct = (
                    top_tokens.eq(active_targets.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .sum()
                )
                quality_token_count = active_targets.new_tensor(
                    float(active_targets.numel()), dtype=torch.float32
                )

            binary_weights = binary_eval_mask.view(
                bsz, n_blocks, self.block_size
            ).float()
            loss_3d = loss_per_token.view(bsz, n_blocks, self.block_size)
            correct_3d = correct.view(bsz, n_blocks, self.block_size).float()
            count_per_position = binary_weights.sum(dim=(0, 1)).to(torch.float32)
            loss_sum_per_position = (loss_3d * binary_weights).sum(dim=(0, 1))
            correct_per_position = correct_3d.sum(dim=(0, 1))
            loss_per_position = loss_sum_per_position / count_per_position.clamp(
                min=1.0
            )
            acc_per_position = correct_per_position / count_per_position.clamp(min=1.0)
            valid_token_count = active_weights.sum().to(dtype=torch.float32)
            weighted_token_count = flat_weights.sum().to(dtype=torch.float32)
            accuracy = correct.float().sum() / binary_eval_mask.float().sum().clamp(
                min=1.0
            )

        self._log_debug_preview(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_ids=target_ids,
            prev_token_ids=prev_token_ids,
            eval_mask=eval_mask,
            active_logits=active_logits,
            loss=loss,
        )

        k3_dump_payload: dict[str, dict[str, torch.Tensor]] = {}
        if self.dump_k3_logprobs:
            tokens_per_sample = n_blocks * self.block_size

            def annotate_dump(
                stream_dump: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                if not stream_dump or "flat_indices" not in stream_dump:
                    return stream_dump
                flat_indices = stream_dump["flat_indices"].long()
                sample_indices = torch.div(
                    flat_indices, tokens_per_sample, rounding_mode="floor"
                )
                within_sample = flat_indices.remainder(tokens_per_sample)
                anchor_indices = torch.div(
                    within_sample, self.block_size, rounding_mode="floor"
                )
                draft_positions = within_sample.remainder(self.block_size)
                annotated = dict(stream_dump)
                annotated.update(
                    {
                        "sample_indices": sample_indices.detach(),
                        "anchor_indices": anchor_indices.detach(),
                        "anchor_positions": anchor_positions[
                            sample_indices, anchor_indices
                        ].detach(),
                        "draft_positions": draft_positions.detach(),
                    }
                )
                if self.trace_aligned_anchors:
                    annotated["trace_accepted_counts"] = accepted_counts[
                        sample_indices, anchor_indices
                    ].detach()
                    annotated["trace_draft_counts"] = draft_counts[
                        sample_indices, anchor_indices
                    ].detach()
                return annotated

            k3_dump_payload = {
                "accepted": annotate_dump(accepted_k3_dump),
                "rejected": annotate_dump(rejected_k3_dump),
            }

        absolute_anchor_positions = torch.gather(
            context_position_ids,
            1,
            anchor_positions.clamp(0, max(seq_len - 1, 0)),
        )
        absolute_anchor_bucket_ids = torch.bucketize(
            absolute_anchor_positions,
            absolute_anchor_positions.new_tensor([512, 1024, 2048]),
        )
        absolute_anchor_bucket_counts = torch.zeros(
            4, dtype=torch.float32, device=device
        )
        absolute_anchor_bucket_counts.scatter_add_(
            0,
            absolute_anchor_bucket_ids.reshape(-1),
            block_keep_mask.float().reshape(-1),
        )
        diagnostics = {
            **{
                f"{stream}_topk_{key}": value.detach()
                for stream, stats in (
                    ("response", response_topk_stats), ("rejected", rejected_topk_stats)
                )
                for key, value in stats.items()
            },
            "response_topk_enabled": local_ploss_sum.new_tensor(
                float(self.response_loss == "topk")
            ),
            "rejected_topk_enabled": local_ploss_sum.new_tensor(
                float(self.rejected_loss == "topk")
            ),
            "topk_union_enabled": local_ploss_sum.new_tensor(
                float(self.topk_support == "union")
            ),
            "correct_count": correct.float().sum().detach(),
            "eval_token_count": binary_eval_mask.float().sum().detach(),
            "top1_correct_count": top1_correct.detach(),
            "top5_correct_count": top5_correct.detach(),
            "quality_token_count": quality_token_count.detach(),
            "valid_token_count": valid_token_count.detach(),
            "weighted_token_count": weighted_token_count.detach(),
            "ce_loss_sum": local_ploss_sum.detach(),
            "ce_weighted_token_count": local_ce_den.detach(),
            "first_rejection_ce_enabled": local_ce_den.new_tensor(
                float(self.first_rejection_ce_alpha > 0)
            ),
            "first_rejection_ce_alpha": local_ce_den.new_tensor(self.first_rejection_ce_alpha),
            "first_rejection_ce_loss_sum": local_first_ce_sum.detach(),
            "first_rejection_ce_token_count": local_first_ce_count.detach(),
            "first_rejection_ce_weight_sum": local_first_ce_weight_sum.detach(),
            "first_rejection_ce_boosted_count": local_first_ce_boosted_count.detach(),
            "first_rejection_ce_q_prob_sum": local_first_ce_q_sum.detach(),
            "l1_loss_sum": local_l1_sum.detach(),
            "l1_weighted_token_count": local_l1_den.detach(),
            "accepted_fkl_sum": local_accepted_fkl_sum.detach(),
            "accepted_fkl_weighted_count": local_accepted_fkl_den.detach(),
            "accepted_k3_sum": local_accepted_k3_sum.detach(),
            "accepted_k3_weighted_count": local_accepted_k3_den.detach(),
            "accepted_k3_clamped_count": accepted_k3_clamped_count.detach(),
            "accepted_k3_linearized_count": accepted_k3_linearized_count.detach(),
            "accepted_k3_histogram": accepted_k3_histogram.detach(),
            "accepted_k3_min": accepted_k3_min.detach(),
            "accepted_k3_max": accepted_k3_max.detach(),
            "accepted_q_prob_sum": accepted_q_prob_sum.detach(),
            "accepted_p_prob_sum": accepted_p_prob_sum.detach(),
            "accepted_log_ratio_sum": accepted_log_ratio_sum.detach(),
            "rejected_k3_sum": local_rejected_k3_sum.detach(),
            "rejected_k3_weighted_count": local_rejected_k3_den.detach(),
            "rejected_k3_histogram": rejected_k3_histogram.detach(),
            "rejected_k3_min": rejected_k3_min.detach(),
            "rejected_k3_max": rejected_k3_max.detach(),
            "rejected_q_prob_sum": rejected_q_prob_sum.detach(),
            "rejected_p_prob_sum": rejected_p_prob_sum.detach(),
            "rejected_teacher_minus_draft_log_ratio_sum": rejected_log_ratio_sum.detach(),
            "rejected_k3_clamped_count": rejected_k3_clamped_count.detach(),
            "rejected_k3_linearized_count": rejected_k3_linearized_count.detach(),
            "asymmetric_k3_enabled": torch.tensor(
                float(self.asymmetric_k3_enabled), dtype=torch.float32, device=device
            ),
            "accepted_asymmetric_k3_enabled": torch.tensor(
                float(self.accepted_asymmetric_k3_enabled),
                dtype=torch.float32,
                device=device,
            ),
            "rejected_asymmetric_k3_enabled": torch.tensor(
                float(self.rejected_asymmetric_k3_enabled),
                dtype=torch.float32,
                device=device,
            ),
            "asymmetric_k3_positive_threshold": torch.tensor(
                self.asymmetric_k3_positive_threshold,
                dtype=torch.float32,
                device=device,
            ),
            "dynamic_accept_reject_weighting_enabled": torch.tensor(
                float(self.dynamic_accept_reject_weighting_enabled),
                dtype=torch.float32,
                device=device,
            ),
            "response_stream_effective_weight": (
                effective_response_stream_weight.detach()
            ),
            "rejected_stream_effective_weight": (
                effective_rejected_stream_weight.detach()
            ),
            "response_stream_weight_multiplier": (
                response_stream_weight_multiplier.detach()
            ),
            "rejected_stream_weight_multiplier": (
                rejected_stream_weight_multiplier.detach()
            ),
            "rejected_aux_alpha": local_ce_den.new_tensor(self.rejected_aux_alpha),
            "rejected_prefix_length": local_ce_den.new_tensor(
                float(self.rejected_prefix_length)
            ),
            "metric_replica_count": torch.ones(
                (), dtype=torch.float32, device=device
            ),
            "response_objective_sum": response_objective_sum.detach(),
            "rejected_objective_sum": rejected_objective_sum.detach(),
            "objective_weighted_token_count": objective_weighted_token_count.detach(),
            "trace_anchor_count": block_keep_mask.float().sum().detach(),
            "trace_accepted_position_count": accepted_trace_positions.detach(),
            "trace_first_rejection_count": first_rejection_positions.detach(),
            "trace_unverified_suffix_count": unverified_trace_positions.detach(),
            "trace_all_accepted_anchor_count": all_accepted_trace_anchors.detach(),
            "trace_bonus_response_position_count": bonus_response_positions.detach(),
            "sanitized_rows": sanitized_rows.detach(),
            "masked_rows": (~binary_eval_mask & flat_eval_mask).float().sum().detach(),
            "sampled_vocab_size": sampled_vocab_size.detach(),
            "loss_mode_id": torch.tensor(
                {
                    "full_vocab": 0.0,
                    "sampled_ce": 2.0,
                    "chunked_ce": 3.0,
                }.get(self.loss_mode, 0.0),
                dtype=torch.float32,
                device=device,
            ),
            "loss_sum_per_position": loss_sum_per_position.detach(),
            "correct_per_position": correct_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "local_ploss_sum": local_ploss_sum.detach(),
            "absolute_anchor_position_sum": (
                absolute_anchor_positions.float() * block_keep_mask.float()
            ).sum().detach(),
            "absolute_anchor_position_count": block_keep_mask.float().sum().detach(),
            "absolute_anchor_bucket_counts": absolute_anchor_bucket_counts.detach(),
            "local_context_token_count_sum": (
                (
                    anchor_positions
                    if self.context_window is None
                    else anchor_positions.clamp(max=self.context_window)
                ).float()
                * block_keep_mask.float()
            ).sum().detach(),
        }
        if self.dump_k3_logprobs:
            diagnostics["k3_logprob_dump"] = k3_dump_payload
        if os.getenv("SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR"):
            diagnostics["state_probe"] = {
                "context_feature": context_feature.detach(),
                "draft_hidden": draft_hidden.detach(),
                "context_position_ids": context_position_ids.detach(),
                "draft_position_ids": draft_position_ids.detach(),
                "anchor_positions": anchor_positions.detach(),
                "block_keep_mask": block_keep_mask.detach(),
                "noise_embedding": noise_embedding.detach(),
            }
        if self.first_rejection_ce_alpha > 0 or self.rejected_aux_alpha > 0:
            # These private tensors intentionally retain autograd until the
            # backend has separated the base and auxiliary loss numerators.
            diagnostics["_base_loss"] = base_loss
            if self.first_rejection_ce_alpha > 0:
                diagnostics["_first_rejection_ce_loss_sum"] = local_first_ce_sum
            if self.rejected_aux_alpha > 0:
                diagnostics["_rejected_aux_loss_sum"] = local_rejected_k3_sum
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            diagnostics,
        )


class DSparkTrainerBackend(DFlashTrainerBackend):
    @property
    def model_type(self):
        return "dspark"

    def _training_value(
        self, training_cfg, dspark_key: str, dflash_key: str, default: Any
    ):
        value = training_cfg.get(dspark_key, None)
        if value is not None:
            return value
        return training_cfg.get(dflash_key, default)

    def _normalize_dflash_config(
        self, drafter_config, target_hf_config, normalized_state, spec_model_path
    ):
        training_cfg = self.config.rollout.drafter.training
        if training_cfg.get("dspark_num_target_layers", None) is not None:
            if getattr(drafter_config, "num_context_layers", None) is None:
                drafter_config.num_context_layers = int(
                    training_cfg["dspark_num_target_layers"]
                )
        return super()._normalize_dflash_config(
            drafter_config,
            target_hf_config,
            normalized_state,
            spec_model_path,
        )

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = self._training_value(
            training_cfg, "dspark_hidden_size", "dflash_hidden_size", None
        )
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(
            self._training_value(
                training_cfg, "dspark_num_target_layers", "dflash_num_target_layers", 5
            )
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 36)
        )
        mask_token_id_cfg = self._training_value(
            training_cfg, "dspark_mask_token_id", "dflash_mask_token_id", None
        )
        mask_token_id = int(
            mask_token_id_cfg
            if mask_token_id_cfg is not None
            else target_text_config.vocab_size - 1
        )
        target_layer_ids = self._training_value(
            training_cfg, "dspark_target_layer_ids", "dflash_target_layer_ids", None
        )
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        return DSparkConfig(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=int(
                self._training_value(
                    training_cfg,
                    "dspark_num_hidden_layers",
                    "dflash_num_hidden_layers",
                    1,
                )
            ),
            num_attention_heads=int(getattr(target_text_config, "num_attention_heads")),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    getattr(target_text_config, "num_attention_heads"),
                )
            ),
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(
                getattr(target_text_config, "max_position_embeddings", 32768)
            ),
            rope_theta=float(getattr(target_text_config, "rope_theta", 10000.0)),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("dspark_block_size", 7)),
            num_anchors=int(training_cfg.get("dspark_num_anchors", 512)),
            markov_rank=int(training_cfg.get("dspark_markov_rank", 256)),
            markov_head_type=str(
                training_cfg.get("dspark_markov_head_type", "vanilla")
            ),
            confidence_head_alpha=float(
                training_cfg.get("dspark_confidence_head_alpha", 0.0)
            ),
            confidence_head_with_markov=bool(
                training_cfg.get("dspark_confidence_head_with_markov", True)
            ),
            ce_loss_alpha=float(training_cfg.get("dspark_ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(training_cfg.get("dspark_l1_loss_alpha", 0.9)),
        )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")
        target_hf_config = self._get_target_hf_config()
        normalized_state = None

        if config_path and os.path.exists(config_path):
            drafter_config = DSparkConfig.from_dspark_pretrained(spec_model_path)
            if spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger, spec_model_path, action="Loading DSpark drafter weights"
                )
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_dict(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DSparkConfig):
            raise TypeError(
                f"DSpark config is not a DSparkConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        if (
            spec_model_path
            and os.path.exists(spec_model_path)
            and os.path.exists(config_path)
        ):
            draft_model = DSparkDraftModel(deepcopy(drafter_config))
            self._load_draft_checkpoint(
                draft_model, spec_model_path, normalized_state=normalized_state
            )
        else:
            draft_model = DSparkDraftModel(deepcopy(drafter_config))
        draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        self.target_lm_head = self._build_target_lm_head(
            target_model_path, target_hf_config
        )
        training_cfg = self.config.rollout.drafter.training
        attention_mode = str(
            training_cfg.get("dspark_attention_mode", "full") or "full"
        ).strip().lower()
        raw_context_window = training_cfg.get("dspark_context_window", None)
        context_window = (
            int(raw_context_window) if raw_context_window not in (None, 0) else None
        )
        if attention_mode != "full":
            max_window = int(training_cfg.get("dspark_max_window", 512))
            collection_window = int(
                training_cfg.get("hidden_state_window_tokens_per_sample", max_window)
            )
            if context_window is None or context_window <= 0:
                raise ValueError(
                    f"dspark_attention_mode={attention_mode!r} requires a positive "
                    "dspark_context_window"
                )
            if max_window <= context_window or collection_window <= context_window:
                raise ValueError(
                    "DSpark context-window training needs both dspark_max_window and "
                    "hidden_state_window_tokens_per_sample to exceed dspark_context_window; "
                    f"got max_window={max_window}, collection_window={collection_window}, "
                    f"context_window={context_window}"
                )
        training_mode = str(training_cfg.get("mode", "online") or "online").lower()
        if training_mode == "online":
            vllm_cfg = self.config.rollout.drafter.get("vllm", {}) or {}
            serving_attention_mode = str(
                vllm_cfg.get("dspark_attention_mode", "full") or "full"
            ).strip().lower()
            if serving_attention_mode != attention_mode:
                raise ValueError(
                    "Online DSpark training requires train-serving attention parity: "
                    f"training={attention_mode!r}, serving={serving_attention_mode!r}. "
                    "Use standalone offline training for hybrid/sparse experiments."
                )
            if attention_mode == "swa":
                serving_context_window = vllm_cfg.get(
                    "dspark_context_window", None
                )
                if int(serving_context_window or 0) != int(context_window or 0):
                    raise ValueError(
                        "Online DSpark SWA requires matching training and serving "
                        f"context windows, got training={context_window}, "
                        f"serving={serving_context_window}"
                    )
        return DSparkTrainingModel(
            draft_model=draft_model,
            block_size=int(
                training_cfg.get(
                    "dspark_block_size", getattr(drafter_config, "block_size", 7)
                )
            ),
            num_anchors=int(
                training_cfg.get(
                    "dspark_num_anchors", getattr(drafter_config, "num_anchors", 512)
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "dspark_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 0.0),
                )
            ),
            loss_mode=str(training_cfg.get("dspark_loss_mode", "full_vocab")),
            sampled_ce_negatives=int(
                training_cfg.get("dspark_sampled_ce_negatives", 0)
            ),
            ce_token_chunk_size=int(training_cfg.get("dspark_ce_token_chunk_size", 64)),
            ce_loss_alpha=float(
                training_cfg.get(
                    "dspark_ce_loss_alpha",
                    getattr(drafter_config, "ce_loss_alpha", 0.1),
                )
            ),
            first_rejection_ce_alpha=float(
                training_cfg.get("dspark_first_rejection_ce_alpha", 0.0)
            ),
            first_rejection_ce_low_entropy_multiplier=float(
                training_cfg.get("dspark_first_rejection_ce_low_entropy_multiplier", 1.0)
            ),
            first_rejection_ce_low_entropy_threshold=float(
                training_cfg.get("dspark_first_rejection_ce_low_entropy_threshold", 0.01)
            ),
            first_rejection_ce_max_rejected_logprob=float(
                training_cfg.get("dspark_first_rejection_ce_max_rejected_logprob", -4.0)
            ),
            l1_loss_alpha=float(
                training_cfg.get(
                    "dspark_l1_loss_alpha",
                    getattr(drafter_config, "l1_loss_alpha", 0.9),
                )
            ),
            confidence_head_alpha=float(
                training_cfg.get("dspark_confidence_loss_alpha", 0.0)
            ),
            accepted_k3_alpha=float(
                training_cfg.get("dspark_accepted_k3_alpha", 1.0)
            ),
            accepted_fkl_alpha=float(
                training_cfg.get("dspark_accepted_fkl_alpha", 0.0)
            ),
            rejected_k3_alpha=float(
                training_cfg.get("dspark_rejected_k3_alpha", 1.0)
            ),
            rejected_aux_alpha=float(
                training_cfg.get("dspark_rejected_aux_alpha", 0.0)
            ),
            rejected_prefix_length=int(
                training_cfg.get("dspark_rejected_prefix_length", 0)
            ),
            response_loss=str(training_cfg.get("dspark_response_loss", "k3")),
            rejected_loss=str(training_cfg.get("dspark_rejected_loss", "k3")),
            topk=int(training_cfg.get("dspark_topk", 16)),
            topk_support=str(training_cfg.get("dspark_topk_support", "student")),
            response_stream_weight=float(
                training_cfg.get("dspark_response_stream_weight", 1.0)
            ),
            rejected_stream_weight=float(
                training_cfg.get("dspark_rejected_stream_weight", 1.0)
            ),
            rejected_position_decay=float(
                training_cfg.get("dspark_rejected_position_decay", 0.9)
            ),
            rejected_first_weight_multiplier=float(
                training_cfg.get("dspark_rejected_first_weight_multiplier", 1.0)
            ),
            rejected_suffix_weight_multiplier=float(
                training_cfg.get("dspark_rejected_suffix_weight_multiplier", 1.0)
            ),
            rejected_log_ratio_clamp=float(
                training_cfg.get("dspark_rejected_log_ratio_clamp", 20.0)
            ),
            rejected_loss_max_clamp=float(
                training_cfg.get("dspark_rejected_loss_max_clamp", 10.0)
            ),
            asymmetric_k3_enabled=bool(
                training_cfg.get("dspark_asymmetric_k3_enabled", False)
            ),
            accepted_asymmetric_k3_enabled=training_cfg.get(
                "dspark_accepted_asymmetric_k3_enabled", None
            ),
            rejected_asymmetric_k3_enabled=training_cfg.get(
                "dspark_rejected_asymmetric_k3_enabled", None
            ),
            asymmetric_k3_positive_threshold=float(
                training_cfg.get("dspark_asymmetric_k3_positive_threshold", 2.0)
            ),
            dynamic_accept_reject_weighting_enabled=bool(
                training_cfg.get(
                    "dspark_dynamic_accept_reject_weighting_enabled", False
                )
            ),
            dynamic_stream_weight_min_multiplier=float(
                training_cfg.get(
                    "dspark_dynamic_stream_weight_min_multiplier", 0.25
                )
            ),
            dynamic_stream_weight_max_multiplier=float(
                training_cfg.get(
                    "dspark_dynamic_stream_weight_max_multiplier", 4.0
                )
            ),
            k3_distribution_bins=int(
                training_cfg.get("dspark_k3_distribution_bins", 512)
            ),
            dump_k3_logprobs=bool(
                training_cfg.get("dspark_dump_k3_logprobs", False)
            ),
            trace_aligned_anchors=bool(
                training_cfg.get("dspark_trace_aligned_anchors", False)
            ),
            attention_mode=attention_mode,
            context_window=context_window,
            global_layer_count=int(
                training_cfg.get("dspark_global_layer_count", 1)
            ),
            use_absolute_position_ids=bool(
                training_cfg.get("dspark_use_absolute_position_ids", False)
            ),
            l1_chunk_size=int(training_cfg.get("dspark_l1_chunk_size", 0)),
            debug_log=bool(training_cfg.get("dspark_debug_log", False)),
            debug_log_first_n=int(training_cfg.get("dspark_debug_log_first_n", 2)),
            debug_log_interval=int(training_cfg.get("dspark_debug_log_interval", 100)),
        ), drafter_config

    def preprocess_individual_items(self, items, device, model_config):
        res = {
            "ids": [],
            "h_states": [],
            "masks": [],
            "position_ids": [],
            "target_last_h_states": [],
            "trace_anchor_positions": [],
            "trace_accepted_counts": [],
            "trace_draft_counts": [],
            "trace_rejected_token_ids": [],
            "trace_draft_token_ids": [],
            "trace_draft_teacher_logprobs": [],
            "trace_draft_teacher_logprob_mask": [],
            "trace_first_rejection_teacher_entropies": [],
            "trace_bonus_response_anchor_mask": [],
        }
        max_window = int(
            self.config.rollout.drafter.training.get("dspark_max_window", 512)
        )
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = int(
            getattr(model_config, "target_hidden_size", model_config.hidden_size)
        )
        num_context_layers = int(
            getattr(
                model_config,
                "num_context_layers",
                getattr(model_config, "num_target_layers", 5),
            )
        )
        expected_hidden_dim = h_dim * num_context_layers
        training_cfg = self.config.rollout.drafter.training
        use_sparse_global = (
            str(training_cfg.get("dspark_attention_mode", "full") or "full")
            .strip()
            .lower()
            == "sparse_global"
        )
        if use_sparse_global:
            res["global_context_masks"] = []
        global_prefix_tokens = max(
            int(training_cfg.get("dspark_global_prefix_tokens", 0) or 0), 0
        )
        global_prompt_tokens = max(
            int(training_cfg.get("dspark_global_prompt_tokens", 0) or 0), 0
        )
        actor_page_size = max(
            int(training_cfg.get("dspark_actor_guided_page_size", 16) or 16), 1
        )
        actor_page_budget_tokens = max(
            int(
                training_cfg.get("dspark_actor_guided_page_budget_tokens", 0) or 0
            ),
            0,
        )
        actor_page_field = str(
            training_cfg.get(
                "dspark_actor_guided_page_starts_field",
                "actor_selected_context_page_starts",
            )
        )

        for item in items:
            if (float(training_cfg.get("dspark_certainty_share_threshold", 0.0)) > 0
                    and "low_entropy_sampling_active" not in item):
                raise ValueError("Certainty-gated training is missing its collection-time sampling flag")
            layout = item.get("hidden_states_layout")
            if layout not in (None, "dflash_aux", "dflash_aux_plus_last"):
                raise ValueError(
                    f"DSpark expected hidden_states_layout='dflash_aux' or 'dflash_aux_plus_last', got {layout!r}. "
                    "This usually means EAGLE3 aux+last hidden states were routed into DSpark training."
                )
            ids = item["input_ids"].to(device, non_blocking=True)
            raw_h = item["hidden_states"]
            full_h = (
                torch.cat(raw_h, dim=-1) if isinstance(raw_h, (list, tuple)) else raw_h
            )
            full_h = full_h.to(device, dtype=torch.bfloat16)
            if full_h.size(-1) < expected_hidden_dim:
                raise ValueError(
                    f"DSpark expected at least {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), got {full_h.size(-1)}"
                )
            target_last_h = None
            if layout == "dflash_aux_plus_last":
                expected_with_last = expected_hidden_dim + h_dim
                if full_h.size(-1) != expected_with_last:
                    raise ValueError(
                        "DSpark hidden_states_layout='dflash_aux_plus_last' expected exactly "
                        f"{expected_with_last} hidden dims ({num_context_layers} context layers plus final hidden), "
                        f"got {full_h.size(-1)}"
                    )
                target_last_h = full_h[..., expected_hidden_dim:expected_with_last]
                full_h = full_h[..., :expected_hidden_dim]
            elif layout == "dflash_aux" and full_h.size(-1) != expected_hidden_dim:
                raise ValueError(
                    f"DSpark hidden_states_layout='dflash_aux' expected exactly {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), got {full_h.size(-1)}"
                )

            if item.get("loss_mask") is not None:
                item_loss_mask = item["loss_mask"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
            elif "prompts" in item and "responses" in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                prompt_len = item["prompts"].size(0)
                responses = item["responses"]
                item_loss_mask[prompt_len : prompt_len + responses.size(0)] = (
                    responses != pad_id
                ).float()[: max(0, ids.size(0) - prompt_len)]
            else:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                item_loss_mask[:] = 1.0

            valid_len = min(ids.size(0), full_h.size(0), item_loss_mask.size(0))
            ids = ids[:valid_len]
            full_h = full_h[:valid_len]
            item_loss_mask = item_loss_mask[:valid_len]
            hidden_positions = item.get("hidden_positions")
            if torch.is_tensor(hidden_positions) and hidden_positions.numel() >= valid_len:
                absolute_positions = hidden_positions.reshape(-1)[:valid_len].to(
                    device=device, dtype=torch.long, non_blocking=True
                )
            else:
                feature_start = int(item.get("_verl_feature_start", 0) or 0)
                absolute_positions = torch.arange(
                    feature_start,
                    feature_start + valid_len,
                    device=device,
                    dtype=torch.long,
                )
            nonzero = torch.nonzero(item_loss_mask)
            if nonzero.numel() > 0:
                r_start = nonzero[0, 0]
                start = torch.clamp(
                    r_start - (max_window // 2),
                    min=0,
                    max=max(0, ids.size(0) - max_window),
                ).item()
                end = min(start + max_window, ids.size(0))
            else:
                start, end = max(0, ids.size(0) - max_window), ids.size(0)

            trace_block_size = int(
                self.config.rollout.drafter.training.get("dspark_block_size", 7)
            )
            directional_probe = bool(item.get("_speco_directional_probe", False))
            selected_anchors = item.get("_speco_directional_selected_anchors")
            trace_rows = _trace_anchor_rows_for_item(
                item,
                crop_start=int(start),
                crop_end=int(end),
                num_anchors=(
                    int(training_cfg.get("dspark_directional_probe_anchors", 128))
                    if directional_probe or selected_anchors is not None
                    else self.config.rollout.drafter.training.get("dspark_num_anchors", 512)
                ),
                block_size=trace_block_size,
                low_entropy_fraction=(
                    0.0 if (item.get("_speco_fixed_eval_holdout", False)
                            or not item.get("low_entropy_sampling_active", True))
                    else float(training_cfg.get("dspark_low_entropy_anchor_fraction", 0.0))
                ),
                low_entropy_threshold=float(training_cfg.get("dspark_low_entropy_threshold", 0.01)),
                position_stratified_low_entropy=bool(training_cfg.get(
                    "dspark_position_entropy_stratified_sampling_enabled", False
                )),
                require_rejected_logprob=bool(item.get("_speco_sampling_require_strict_rejection", False)),
                low_entropy_max_rejected_logprob=float(training_cfg.get(
                    "dspark_low_entropy_max_rejected_logprob", -4.0
                )),
                selection_stats=(selection_stats := {}),
                forced_reserved_anchors=item.get("_speco_global_reserved_trace_anchors"),
            )
            if selected_anchors is not None:
                selected_set = {int(anchor) for anchor in selected_anchors}
                available = {int(row["anchor"]) for row in trace_rows}
                if not selected_set <= available:
                    raise ValueError("Directional training anchors differ from the paired probe")
                trace_rows = [
                    row for row in trace_rows if int(row["anchor"]) in selected_set
                ]
            if float(training_cfg.get("dspark_low_entropy_anchor_fraction", 0.0)) > 0:
                if selection_stats.get("rejections", 0) and not selection_stats.get("entropy_observed", 0):
                    raise ValueError("Low-entropy anchor sampling requires draft_teacher_entropies in verify traces")
                logger.info(
                    "[dspark low-entropy sampling] step=%s sample=%s fixed_eval=%s gate_active=%s stats=%s",
                    item.get("global_step"), item.get("sample_id"),
                    bool(item.get("_speco_fixed_eval_holdout", False)),
                    bool(item.get("low_entropy_sampling_active", True)), selection_stats,
                )

            res["ids"].append(ids[start:end])
            res["h_states"].append(full_h[start:end, :expected_hidden_dim])
            res["masks"].append(item_loss_mask[start:end])
            cropped_positions = absolute_positions[start:end]
            res["position_ids"].append(cropped_positions)
            if use_sparse_global:
                global_context_mask = torch.zeros_like(
                    cropped_positions, dtype=torch.bool
                )
                if global_prefix_tokens > 0:
                    global_context_mask |= cropped_positions < global_prefix_tokens
                prompt_len = int(item.get("_verl_prompt_len", 0) or 0)
                if global_prompt_tokens > 0 and prompt_len > 0:
                    global_context_mask |= (
                        (
                            cropped_positions
                            >= max(prompt_len - global_prompt_tokens, 0)
                        )
                        & (cropped_positions < prompt_len)
                    )
                raw_page_starts = item.get(actor_page_field)
                if raw_page_starts is not None and actor_page_budget_tokens > 0:
                    page_starts = torch.as_tensor(
                        raw_page_starts, device=device, dtype=torch.long
                    ).reshape(-1)
                    max_pages = actor_page_budget_tokens // actor_page_size
                    for page_start in page_starts[:max_pages]:
                        global_context_mask |= (
                            (cropped_positions >= page_start)
                            & (cropped_positions < page_start + actor_page_size)
                        )
                res["global_context_masks"].append(global_context_mask)
            if target_last_h is not None:
                res["target_last_h_states"].append(target_last_h[start:end])
            else:
                res["target_last_h_states"].append(None)
            res["trace_anchor_positions"].append(
                torch.tensor(
                    [int(row["anchor"]) for row in trace_rows],
                    dtype=torch.long,
                    device=device,
                )
            )
            if str(training_cfg.get("dspark_rejected_loss", "k3")).strip().lower() == "topk":
                for key, value in pack_trace_topk(
                    trace_rows,
                    block_size=trace_block_size,
                    topk=int(training_cfg.get("dspark_topk", 16)),
                    support=str(training_cfg.get("dspark_topk_support", "student")).strip().lower(),
                    device=device,
                ).items():
                    res.setdefault(key, []).append(value)
            res["trace_accepted_counts"].append(
                torch.tensor(
                    [int(row["accepted_count"]) for row in trace_rows],
                    dtype=torch.long,
                    device=device,
                )
            )
            res["trace_draft_counts"].append(
                torch.tensor(
                    [int(row["draft_count"]) for row in trace_rows],
                    dtype=torch.long,
                    device=device,
                )
            )
            res["trace_rejected_token_ids"].append(
                torch.tensor(
                    [int(row["rejected_token_id"]) for row in trace_rows],
                    dtype=torch.long,
                    device=device,
                )
            )
            res["trace_bonus_response_anchor_mask"].append(
                torch.tensor(
                    [bool(row["bonus_response_anchor"]) for row in trace_rows],
                    dtype=torch.bool,
                    device=device,
                )
            )
            padded_draft_ids = [
                list(row["draft_token_ids"])
                + [-1] * (trace_block_size - len(row["draft_token_ids"]))
                for row in trace_rows
            ]
            padded_teacher_logprobs = [
                list(row["draft_teacher_logprobs"])
                + [0.0]
                * (trace_block_size - len(row["draft_teacher_logprobs"]))
                for row in trace_rows
            ]
            padded_teacher_masks = [
                list(row["draft_teacher_logprob_mask"])
                + [False]
                * (trace_block_size - len(row["draft_teacher_logprob_mask"]))
                for row in trace_rows
            ]
            res["trace_draft_token_ids"].append(
                torch.tensor(
                    padded_draft_ids,
                    dtype=torch.long,
                    device=device,
                ).reshape(-1, trace_block_size)
            )
            res["trace_draft_teacher_logprobs"].append(
                torch.tensor(
                    padded_teacher_logprobs,
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1, trace_block_size)
            )
            res["trace_draft_teacher_logprob_mask"].append(
                torch.tensor(
                    padded_teacher_masks,
                    dtype=torch.bool,
                    device=device,
                ).reshape(-1, trace_block_size)
            )
            res["trace_first_rejection_teacher_entropies"].append(
                torch.tensor(
                    [
                        row.get("first_rejection_entropy")
                        if row.get("first_rejection_entropy") is not None else float("nan")
                        for row in trace_rows
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        return res

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError(
                "DSpark drafter training does not support Ulysses sequence parallel yet"
            )
        if self.target_lm_head is None:
            raise ValueError("DSpark target_lm_head is not initialized")

        draft_model = model.module if hasattr(model, "module") else model
        hidden_states = batch["hidden_states"]
        num_context_layers = draft_model.draft_model.num_context_layers
        per_layer_dim = hidden_states.shape[-1] // num_context_layers
        hidden_states_list = list(hidden_states.split(per_layer_dim, dim=-1))

        loss, accuracy, loss_pp, acc_pp, count_pp, diagnostics = model(
            input_ids=batch["input_ids"],
            hidden_states_list=hidden_states_list,
            loss_mask=batch["loss_mask"],
            lm_head_weight=self.target_lm_head.fc.weight,
            target_last_hidden_states=batch.get("target_last_hidden_states"),
            trace_anchor_positions=batch.get("trace_anchor_positions"),
            trace_anchor_mask=batch.get("trace_anchor_mask"),
            trace_accepted_counts=batch.get("trace_accepted_counts"),
            trace_draft_counts=batch.get("trace_draft_counts"),
            trace_rejected_token_ids=batch.get("trace_rejected_token_ids"),
            trace_draft_token_ids=batch.get("trace_draft_token_ids"),
            trace_draft_teacher_logprobs=batch.get(
                "trace_draft_teacher_logprobs"
            ),
            trace_draft_teacher_logprob_mask=batch.get(
                "trace_draft_teacher_logprob_mask"
            ),
            trace_first_rejection_teacher_entropies=batch.get(
                "trace_first_rejection_teacher_entropies"
            ),
            trace_bonus_response_anchor_mask=batch.get(
                "trace_bonus_response_anchor_mask"
            ),
            trace_draft_topk_ids=batch.get("trace_draft_topk_ids"),
            trace_draft_topk_teacher_logprobs=batch.get(
                "trace_draft_topk_teacher_logprobs"
            ),
            position_ids=batch.get("position_ids"),
            global_context_mask=batch.get("global_context_mask"),
        )
        local_num_tokens = diagnostics.get("objective_weighted_token_count")
        if not torch.is_tensor(local_num_tokens):
            local_num_tokens = diagnostics.get("ce_weighted_token_count")
        if not torch.is_tensor(local_num_tokens):
            local_num_tokens = count_pp.sum()
        local_num_tokens = local_num_tokens.to(loss.device, dtype=loss.dtype)
        base_loss = diagnostics.pop("_base_loss", loss)
        first_ce_auxiliary_sum = diagnostics.pop(
            "_first_rejection_ce_loss_sum", None
        )
        rejected_auxiliary_sum = diagnostics.pop("_rejected_aux_loss_sum", None)
        result = {
            "total_local_vloss": torch.tensor(0.0, device=batch["input_ids"].device),
            "total_local_ploss": base_loss * local_num_tokens,
            "local_num_tokens": local_num_tokens,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_pp.detach(),
            "acc_per_position": acc_pp.detach(),
            "count_per_position": count_pp.detach(),
            "diagnostics": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in diagnostics.items()
            },
        }
        auxiliary_losses = []
        if first_ce_auxiliary_sum is not None:
            auxiliary_losses.append({
                "name": "first_rejection_ce",
                "loss_sum": (
                    first_ce_auxiliary_sum * draft_model.first_rejection_ce_alpha
                ),
                "token_count": diagnostics["first_rejection_ce_weight_sum"],
            })
        if rejected_auxiliary_sum is not None:
            auxiliary_losses.append({
                "name": "rejected_k3",
                "loss_sum": rejected_auxiliary_sum * draft_model.rejected_aux_alpha,
                "token_count": diagnostics["rejected_k3_weighted_count"],
            })
        if len(auxiliary_losses) == 1:
            # Preserve the established one-auxiliary contract for callers and tests.
            result["auxiliary_loss"] = auxiliary_losses[0]
        elif auxiliary_losses:
            result["auxiliary_losses"] = auxiliary_losses
        return result
