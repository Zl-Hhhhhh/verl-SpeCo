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
"""SPECO adapter for verl v0.8.0 RayPPOTrainer."""

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from types import MethodType
from typing import Any, cast

import ray
import torch
from omegaconf import open_dict
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl_speco.integration.agent_loop_runtime import (
    SPECO_AGENT_LOOP_MANAGER_CLASS,
    install_agent_loop_runtime_patch,
)
from verl_speco.integration.rollout_publish import resolve_drafter_publish_payload
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_AUX_LAYER_IDS_KEY,
    OLD_LOGPROB_COLLECT_MASK_KEY,
    OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
    OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY,
    OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
    OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
    OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
    OLD_LOGPROB_HIDDEN_REF_META_KEY,
    OLD_LOGPROB_HIDDEN_REFS_KEY,
    OLD_LOGPROB_HIDDEN_STATES_KEY,
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_TIMING_KEY,
)
from verl_speco.integration.oldlogprob_layer_ids import (
    assert_sglang_aux_last_layer_norm_safe,
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.integration.sglang_adapter import (
    bucket_drafter_samples_by_replica,
    pop_drafter_samples,
    speco_step_matches_interval,
)
from verl_speco.integration.sglang_runtime import (
    clear_sglang_runtime_config,
    configure_sglang_runtime_from_config,
    install_upstream_sglang_runtime_bridge,
    should_install_sglang_base_compat_runtime,
)
from verl_speco.integration.vllm_runtime import (
    SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
    SPECO_VLLM_SPECULATIVE_TRACE_KEY,
    SPECO_VLLM_SPECULATIVE_TRACE_DIAGNOSTICS_KEY,
    configure_vllm_runtime_from_config,
)
from verl_speco.trainer.dspark_training_config import (
    dspark_target_distribution_loss_enabled,
)
from verl_speco.workers import SpecoWorker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC = (
    "drafter/spec_decode/mean_acceptance_length"
)
SPECO_KEY_METRIC_ALIASES = {
    "key_metrics/mean_acceptance_length": SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC,
    "key_metrics/generation_time_s": "timing_s/gen",
    "key_metrics/step_time_s": "timing_s/step",
    "key_metrics/critic_reward_mean": "critic/rewards/mean",
}
_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY = "_speco_vllm_spec_decode_drafts"
_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY = "_speco_vllm_spec_decode_accepted_tokens"
_SPECO_VALIDATION_SPEC_DECODE_PREFIX = "val-aux/drafter/spec_decode"
_SPECO_VALIDATION_DIAGNOSTICS_DIR_ENV = "SPECO_VALIDATION_DIAGNOSTICS_DIR"
_SPECO_DRAFTER_TIMING_DEDUCTED_KEY = "_speco_drafter_timing_deducted_from_update_actor"
_DRAFTER_TARGET_SYNC_MESH = "drafter_target_sync"

_DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS = {
    None,
    "",
    "null",
    "None",
    "/path/to/drafter/checkpoint",
}
_POLICY_MODEL_NON_TENSOR_KEYS = {"multi_modal_inputs", "pad_token_id"}


def speco_step_matches_save_schedule(
    step: int, frequency: int, offset: int = 0
) -> bool:
    """Return whether ``step`` matches a positive periodic save schedule."""

    try:
        step = int(step)
        frequency = int(frequency)
        offset = int(offset)
    except (TypeError, ValueError):
        return False
    if step <= 0 or frequency <= 0 or offset < 0 or offset >= frequency:
        return False
    return step >= frequency + offset and (step - offset) % frequency == 0


def _speco_limit_drafter_resource_pool(
    resource_pool: Any,
    requested_world_size: Any,
    rollout_config: Any,
):
    """Return the actor-pool prefix assigned to online drafter training."""

    if requested_world_size is None:
        return resource_pool
    try:
        requested_world_size = int(requested_world_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "drafter.training.worker_world_size must be an integer, "
            f"got {requested_world_size!r}"
        ) from exc

    full_world_size = int(resource_pool.world_size)
    rollout_world_size = math.prod(
        int(_get_nested(rollout_config, (key,), 1) or 1)
        for key in (
            "tensor_model_parallel_size",
            "data_parallel_size",
            "pipeline_model_parallel_size",
        )
    )
    if requested_world_size <= 0 or requested_world_size > full_world_size:
        raise ValueError(
            "drafter.training.worker_world_size must be in "
            f"[1, {full_world_size}], got {requested_world_size}"
        )
    if requested_world_size % rollout_world_size != 0:
        raise ValueError(
            "drafter.training.worker_world_size must be divisible by the rollout replica world size: "
            f"worker_world_size={requested_world_size}, rollout_world_size={rollout_world_size}"
        )
    if requested_world_size == full_world_size:
        return resource_pool

    limited_pool = split_resource_pool(
        resource_pool,
        split_size=[requested_world_size, full_world_size - requested_world_size],
    )[0]
    logger.warning(
        "SPECO online drafter uses %s of %s actor resource-pool workers",
        requested_world_size,
        full_world_size,
    )
    return limited_pool


def _select_policy_model_batch(batch: DataProto) -> DataProto:
    """Keep rollout/drafter side-channel data out of policy-model forward paths."""
    non_tensor_batch_keys = [
        key for key in _POLICY_MODEL_NON_TENSOR_KEYS if key in batch.non_tensor_batch
    ]
    return batch.select(non_tensor_batch_keys=non_tensor_batch_keys)


def _get_nested(config, path, default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _speco_alpha_counter(value: int) -> str:
    """Encode a positive counter with letters so Ray log dedup keeps each sample."""

    value = max(int(value), 1)
    chars = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))


def _speco_ref_meta_rows(meta: Any) -> int:
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("rows", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _speco_ref_meta_nbytes(meta: Any) -> int:
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("nbytes", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _speco_ref_meta_row_count(meta: Any, default: int = 0) -> int:
    if not isinstance(meta, dict):
        return int(default)
    row_indices = meta.get("chunk_row_indices")
    if torch.is_tensor(row_indices):
        row_indices = cast(torch.Tensor, row_indices)
        return int(row_indices.numel())
    if isinstance(row_indices, (list, tuple)):
        return len(row_indices)
    try:
        return int(meta.get("chunk_length", meta.get("rows", default)) or 0)
    except (TypeError, ValueError):
        return int(default)


def _speco_metric_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _speco_add_key_metric_aliases(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    aliases = {
        alias: value
        for alias, source in SPECO_KEY_METRIC_ALIASES.items()
        if (value := _speco_metric_float(data.get(source))) is not None
    }
    if not aliases:
        return data
    return {**data, **aliases}


def _speco_move_drafter_timing_next_to_update_actor(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    drafter_elapsed = _speco_metric_float(data.get("timing_s/drafter"))
    mean_acceptance_length = _speco_metric_float(
        data.get(SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC)
    )
    update_actor_elapsed = _speco_metric_float(data.get("timing_s/update_actor"))
    already_deducted = bool(data.get(_SPECO_DRAFTER_TIMING_DEDUCTED_KEY))
    if (
        drafter_elapsed is None
        and mean_acceptance_length is None
        and not already_deducted
    ):
        return data

    adjusted_update_actor = None
    adjusted_update_actor_per_token = None
    if (
        drafter_elapsed is not None
        and update_actor_elapsed is not None
        and not already_deducted
    ):
        adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
        update_actor_per_token = _speco_metric_float(
            data.get("timing_per_token_ms/update_actor")
        )
        if update_actor_per_token is not None:
            adjusted_update_actor_per_token = (
                update_actor_per_token * adjusted_update_actor / update_actor_elapsed
                if update_actor_elapsed > 0
                else 0.0
            )

    rewritten = {}
    inserted_drafter_metrics = False
    for key, value in data.items():
        if key in {
            "timing_s/drafter",
            SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC,
            _SPECO_DRAFTER_TIMING_DEDUCTED_KEY,
        }:
            continue
        if key == "timing_s/update_actor":
            rewritten[key] = (
                adjusted_update_actor if adjusted_update_actor is not None else value
            )
            if drafter_elapsed is not None:
                rewritten["timing_s/drafter"] = drafter_elapsed
            if mean_acceptance_length is not None:
                rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = (
                    mean_acceptance_length
                )
            inserted_drafter_metrics = True
        elif (
            key == "timing_per_token_ms/update_actor"
            and adjusted_update_actor_per_token is not None
        ):
            rewritten[key] = adjusted_update_actor_per_token
        else:
            rewritten[key] = value
    if not inserted_drafter_metrics:
        if drafter_elapsed is not None:
            rewritten["timing_s/drafter"] = drafter_elapsed
        if mean_acceptance_length is not None:
            rewritten[SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC] = (
                mean_acceptance_length
            )
    return rewritten


def _speco_float_values(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]

    normalized = []
    for value in values:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _speco_vllm_spec_decode_stats_from_batch(batch: Any) -> dict[str, float]:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if not isinstance(non_tensor_batch, dict):
        return {}

    def values(name: str) -> list[float]:
        return _speco_float_values(
            non_tensor_batch.get(f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_{name}")
        )

    drafts = values("drafts")
    accepted_tokens = values("accepted_tokens")
    total_drafts = float(sum(drafts))
    total_accepted_tokens = float(sum(accepted_tokens))
    if total_drafts <= 0.0 and total_accepted_tokens <= 0.0:
        return {}

    return {
        _SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: total_drafts,
        _SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: total_accepted_tokens,
    }


def _speco_vllm_spec_decode_metrics_from_stats(
    stats: dict[str, float],
) -> dict[str, float]:
    drafts = float(stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0)
    if drafts <= 0.0:
        return {}
    accepted_tokens = float(
        stats.get(_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY, 0.0) or 0.0
    )
    return {
        SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC: 1.0 + accepted_tokens / drafts
    }


def _speco_vllm_spec_decode_metrics_from_batch(batch: Any) -> dict[str, float]:
    return _speco_vllm_spec_decode_metrics_from_stats(
        _speco_vllm_spec_decode_stats_from_batch(batch)
    )


def _speco_truthy_meta_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _speco_generation_meta_info(value: Any) -> dict[str, Any] | None:
    meta_info = getattr(value, "meta_info", None)
    if isinstance(meta_info, dict):
        return meta_info
    if isinstance(value, dict):
        meta_info = value.get("meta_info")
        if isinstance(meta_info, dict):
            return meta_info
    return None


def _speco_is_validation_generation_value(value: Any) -> bool:
    meta_info = _speco_generation_meta_info(value)
    if not isinstance(meta_info, dict):
        return False
    for key in ("validate", "validation", "is_validate", "is_validation", "test"):
        if key in meta_info and _speco_truthy_meta_value(meta_info.get(key)):
            return True
    phase = (
        str(
            meta_info.get("phase")
            or meta_info.get("split")
            or meta_info.get("mode")
            or meta_info.get("stage")
            or ""
        )
        .strip()
        .lower()
    )
    return phase in {"validate", "validation", "val", "test", "eval", "evaluation"}


def _speco_is_validation_generation(
    args: tuple[Any, ...], kwargs: dict[str, Any], output: Any = None
) -> bool:
    candidates = [output, *args]
    for key in ("batch", "prompts", "data", "input_batch"):
        if key in kwargs:
            candidates.append(kwargs[key])
    return any(
        _speco_is_validation_generation_value(candidate) for candidate in candidates
    )


def _speco_merge_vllm_spec_decode_stats(
    existing: dict[str, float] | None,
    current: dict[str, float],
) -> dict[str, float]:
    if not current:
        return existing or {}
    totals = {
        _SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY: 0.0,
        _SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY: 0.0,
    }
    for key in totals:
        totals[key] = float((existing or {}).get(key, 0.0) or 0.0) + float(
            current.get(key, 0.0) or 0.0
        )
    return totals


class SpecoRayPPOTrainer(RayPPOTrainer):
    """External trainer adapter for SPECO.

    Normal PPO still delegates to upstream ``RayPPOTrainer.fit``. SPECO online
    drafter training installs scoped hooks around that loop, delegating normal PPO
    behavior while keeping SPECO collection/training/publishing in
    ``verl_speco`` instead of requiring external ``verl`` source edits.
    """

    def __init__(self, *args, **kwargs):
        self.speco_worker_cls = kwargs.pop("speco_worker_cls", None)
        super().__init__(*args, **kwargs)
        self.drafter_wg = None
        self._pending_drafter_publish_refs = None
        self._pending_drafter_checkpoint_refs = []
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_oldlogprob_candidate_samples = 0
        self._speco_last_oldlogprob_planned_samples = 0
        self._speco_last_oldlogprob_collected_samples = 0
        self._speco_last_oldlogprob_collected_rows = 0
        self._speco_last_oldlogprob_payload_mib = 0.0
        self._speco_last_oldlogprob_window_strata = {}
        self._speco_last_oldlogprob_materialized_window_strata = {}
        self._speco_last_oldlogprob_drop_reasons = {}
        self._speco_last_oldlogprob_dispatched_samples = 0
        self._speco_last_oldlogprob_consumed_samples = 0
        self._speco_last_oldlogprob_unique_samples = 0
        self._speco_last_oldlogprob_consumed_unique_samples = 0
        self._speco_last_spec_trace_samples = 0
        self._speco_last_spec_trace_verify_steps = 0
        self._speco_last_spec_trace_draft_tokens = 0
        self._speco_last_spec_trace_teacher_logprob_tokens = 0
        self._speco_last_spec_trace_accepted_tokens = 0
        self._speco_last_spec_trace_first_rejections = 0
        self._speco_last_spec_trace_unverified_suffix_tokens = 0
        self._speco_last_spec_trace_all_accepted_steps = 0
        self._speco_last_spec_trace_diagnostics = {}
        self._speco_last_fixed_eval_trace_metrics = {}
        self._speco_last_drafter_guard_reason = ""
        self._speco_pending_post_update_acceptance_guard = None
        self._speco_pending_candidate_update = None
        self._speco_pending_candidate_metrics = {}
        self._speco_candidate_grad_norm_history = []
        self._speco_best_candidate_score = None
        self._speco_candidate_gate_cohort = None
        self._speco_pending_best_candidate_save = None
        self._speco_pending_dual_best_observation = None
        self._speco_last_rollout_generation_elapsed_sec = 0.0
        self._speco_last_rollout_response_tokens = 0
        self._speco_last_oldlogprob_select_elapsed_sec = 0.0
        self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
        self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
        self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
        self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
        self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
        self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
        self._speco_last_oldlogprob_total_elapsed_sec = 0.0
        self._speco_last_collect_interval_matched = 0

    def attach_speco_worker_group(self, worker_group):
        self.drafter_wg = worker_group

    def _require_speco_worker_group(self):
        if self.drafter_wg is None:
            raise RuntimeError("SpecoWorker group has not been initialized yet.")
        return self.drafter_wg

    def speco_set_global_step(self, global_step: int):
        return self._require_speco_worker_group().set_global_step(global_step)

    def speco_collect_rollout_features(self, samples: list[list[dict]]):
        return self._require_speco_worker_group().collect_rollout_features(samples)

    def speco_sync_target_lm_head_weight(self, payload: Any, global_step: Any = None):
        return self._require_speco_worker_group().sync_target_lm_head_weight(
            payload, global_step=global_step
        )

    def speco_get_drafter_target_lm_head_row_indices(self):
        return (
            self._require_speco_worker_group().get_drafter_target_lm_head_row_indices()
        )

    def speco_train_drafter(self):
        return self._require_speco_worker_group().train_drafter()

    def speco_activate_drafter_training_model(self):
        return self._require_speco_worker_group().activate_drafter_training_model()

    def speco_maybe_publish(self):
        return self._require_speco_worker_group().maybe_publish()

    def speco_get_candidate_weights(self, global_step: int, alpha: float):
        return self._require_speco_worker_group().get_candidate_weights(
            global_step, alpha
        )

    def speco_finalize_candidate_update(self, global_step: int, alpha: float):
        return self._require_speco_worker_group().finalize_candidate_update(
            global_step, alpha
        )

    def speco_save_checkpoint(
        self,
        global_step: int,
        wait: bool = True,
        include_optimizer: bool = True,
    ):
        return self._require_speco_worker_group().save_checkpoint(
            global_step,
            wait=wait,
            include_optimizer=include_optimizer,
        )

    def speco_wait_checkpoint(self):
        return self._require_speco_worker_group().wait_checkpoint()

    def init_workers(self):
        drafter_rollout_enabled = self.is_drafter_rollout_enabled(self.config)
        online_drafter_enabled = self.is_drafter_training_enabled(self.config)
        if online_drafter_enabled:
            self._speco_prepare_drafter_checkpoint_for_worker_init()
        if drafter_rollout_enabled:
            configure_sglang_runtime_from_config(self.config)
            configure_vllm_runtime_from_config(self.config)
            if online_drafter_enabled:
                install_agent_loop_runtime_patch()
            if (
                _get_nested(self.config, ("actor_rollout_ref", "rollout", "name"), None)
                == "sglang"
            ):
                install_upstream_sglang_runtime_bridge()
        else:
            clear_sglang_runtime_config()
            if should_install_sglang_base_compat_runtime(self.config):
                install_upstream_sglang_runtime_bridge(base_compat_only=True)
        with self._hide_speco_drafter_config_from_upstream_rollout():
            with self._use_speco_agent_loop_manager(online_drafter_enabled):
                super().init_workers()
        if online_drafter_enabled:
            self._init_speco_drafter_workers()
            # Fail closed on the divergent SGLang last-layer-norm combination at
            # init, before any (expensive) rollout generation runs.
            self._speco_validate_sglang_aux_last_layer_norm()

    @contextmanager
    def _use_speco_agent_loop_manager(self, enabled: bool):
        if not enabled:
            yield
            return
        manager_class = SPECO_AGENT_LOOP_MANAGER_CLASS

        rollout_config = _get_nested(
            self.config, ("actor_rollout_ref", "rollout"), None
        )
        if rollout_config is None:
            yield
            return

        missing = object()
        original_agent = (
            rollout_config.get("agent", missing)
            if hasattr(rollout_config, "get")
            else missing
        )
        agent_config = original_agent if original_agent is not missing else {}
        previous_manager_class = (
            agent_config.get("agent_loop_manager_class", missing)
            if hasattr(agent_config, "get")
            else missing
        )
        with open_dict(rollout_config):
            if "agent" not in rollout_config or rollout_config["agent"] is None:
                rollout_config["agent"] = {}
            rollout_config["agent"]["agent_loop_manager_class"] = manager_class
        try:
            yield
        finally:
            with open_dict(rollout_config):
                if original_agent is missing:
                    del rollout_config["agent"]
                elif previous_manager_class is missing:
                    rollout_config["agent"] = original_agent
                    rollout_config["agent"].pop("agent_loop_manager_class", None)
                else:
                    rollout_config["agent"] = original_agent
                    rollout_config["agent"]["agent_loop_manager_class"] = (
                        previous_manager_class
                    )

    @contextmanager
    def _hide_speco_drafter_config_from_upstream_rollout(self):
        rollout_config = _get_nested(
            self.config, ("actor_rollout_ref", "rollout"), None
        )
        missing = object()
        drafter_config = missing
        if rollout_config is not None and "drafter" in rollout_config:
            drafter_config = rollout_config["drafter"]
            with open_dict(rollout_config):
                del rollout_config["drafter"]
        try:
            yield
        finally:
            if drafter_config is not missing:
                with open_dict(rollout_config):
                    rollout_config["drafter"] = drafter_config

    def _init_speco_drafter_workers(self):
        if self.drafter_wg is not None:
            return

        speco_worker_cls = self.speco_worker_cls or ray.remote(SpecoWorker)
        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self.role_worker_mapping
            else Role.ActorRollout
        )
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        training_cfg = self._speco_drafter_training_config()
        resource_pool = _speco_limit_drafter_resource_pool(
            resource_pool,
            training_cfg.get("worker_world_size", None),
            self.config.actor_rollout_ref.rollout,
        )
        drafter_cls = RayClassWithInitArgs(
            cls=speco_worker_cls,
            config=self.config.actor_rollout_ref,
            role="drafter",
            device_name=self.device_name,
        )

        self.drafter_wg = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=drafter_cls,
            name_prefix="speco_drafter",
            device_name=self.device_name,
        )
        self.drafter_wg.init_model()

    def _ray_get_if_needed(self, value):
        if value is None:
            return None
        try:
            import ray
        except Exception:  # noqa: BLE001
            return value

        object_ref_type = getattr(ray, "ObjectRef", ())
        if object_ref_type and isinstance(value, object_ref_type):
            return ray.get(value)
        if isinstance(value, (list, tuple)) and value and object_ref_type:
            if all(isinstance(item, object_ref_type) for item in value):
                return ray.get(list(value))
        return value

    @staticmethod
    def _first_non_null(value):
        if isinstance(value, (list, tuple)):
            non_null = [item for item in value if item is not None]
            if len(non_null) > 1:
                raise RuntimeError(
                    f"Expected at most one non-null SPECO result, got {len(non_null)}"
                )
            return non_null[0] if non_null else None
        return value

    def _speco_online_enabled(self) -> bool:
        return self.is_drafter_training_enabled(self.config)

    def _speco_drafter_training_config(self):
        return _get_nested(
            self.config, ("actor_rollout_ref", "rollout", "drafter", "training"), {}
        )

    def _speco_drafter_config(self):
        return _get_nested(
            self.config, ("actor_rollout_ref", "rollout", "drafter"), None
        )

    @staticmethod
    def _speco_set_config_value(config, key: str, value: Any):
        try:
            with open_dict(config):
                config[key] = value
        except Exception:  # noqa: BLE001
            if hasattr(config, "__setitem__"):
                config[key] = value
            else:
                setattr(config, key, value)

    def _speco_ensure_drafter_checkpoint_path(self) -> str | None:
        drafter_cfg = self._speco_drafter_config()
        if drafter_cfg is None:
            return None

        checkpoint_path = (
            drafter_cfg.get("checkpoint_path", None)
            if hasattr(drafter_cfg, "get")
            else getattr(drafter_cfg, "checkpoint_path", None)
        )
        if checkpoint_path not in _DRAFTER_CHECKPOINT_PATH_PLACEHOLDERS:
            return checkpoint_path

        default_local_dir = _get_nested(
            self.config, ("trainer", "default_local_dir"), None
        )
        if default_local_dir in (None, ""):
            return None

        checkpoint_path = os.path.join(str(default_local_dir), "drafter")
        self._speco_set_config_value(drafter_cfg, "checkpoint_path", checkpoint_path)
        return checkpoint_path

    def _speco_drafter_checkpoint_save_config_enabled(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        if hasattr(training_cfg, "get"):
            return bool(training_cfg.get("save_full_drafter_checkpoint", True))
        return True

    def _speco_resume_global_step_hint(self) -> int | None:
        trainer_cfg = _get_nested(self.config, ("trainer",), None)
        resume_mode = str(
            _get_nested(trainer_cfg, ("resume_mode",), "disable") or "disable"
        )
        if resume_mode == "disable":
            return None

        global_step_folder = None
        if resume_mode == "resume_path":
            global_step_folder = _get_nested(trainer_cfg, ("resume_from_path",), None)
        elif resume_mode == "auto":
            checkpoint_folder = _get_nested(trainer_cfg, ("default_local_dir",), None)
            if checkpoint_folder:
                checkpoint_folder = os.path.abspath(os.fspath(checkpoint_folder))
                try:
                    from verl.utils.checkpoint.checkpoint_manager import (
                        find_latest_ckpt_path,
                    )

                    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Unable to resolve latest actor checkpoint for drafter resume: %s",
                        exc,
                    )

        if not global_step_folder:
            return None
        folder_name = os.path.basename(os.path.normpath(os.fspath(global_step_folder)))
        if not folder_name.startswith("global_step_"):
            return None
        try:
            return int(folder_name.removeprefix("global_step_"))
        except ValueError:
            return None

    def _speco_prepare_drafter_checkpoint_for_worker_init(self):
        drafter_cfg = self._speco_drafter_config()
        if drafter_cfg is None:
            return

        checkpoint_save_enabled = self._speco_drafter_checkpoint_save_config_enabled()
        if checkpoint_save_enabled:
            self._speco_ensure_drafter_checkpoint_path()

        training_cfg = self._speco_drafter_training_config()
        resume_setting = training_cfg.get("resume_trainer_state_from_checkpoint", None)
        if resume_setting is None:
            resume_setting = training_cfg.get(
                "resume_lr_scheduler_from_checkpoint", True
            )
        if not bool(resume_setting):
            return

        explicit_resume_model_path = _get_nested(
            drafter_cfg, ("resume_model_path",), None
        )
        if explicit_resume_model_path:
            from verl_speco.trainer.checkpoint import (
                get_drafter_checkpoint_step,
                is_pretrained_drafter_checkpoint,
            )

            explicit_resume_model_path = os.path.realpath(
                os.fspath(explicit_resume_model_path)
            )
            if not is_pretrained_drafter_checkpoint(explicit_resume_model_path):
                raise RuntimeError(
                    "Explicit drafter resume checkpoint is incomplete or invalid: "
                    f"{explicit_resume_model_path}"
                )
            self._speco_set_config_value(
                drafter_cfg, "model_path", explicit_resume_model_path
            )
            logger.warning(
                "[drafter recovery] explicitly resuming checkpoint=%s drafter_step=%s",
                explicit_resume_model_path,
                get_drafter_checkpoint_step(explicit_resume_model_path),
            )
            return

        resume_step = self._speco_resume_global_step_hint()
        if resume_step is None:
            return

        from verl_speco.trainer.checkpoint import (
            get_drafter_checkpoint_step,
            resolve_drafter_checkpoint_path,
        )

        model_path = _get_nested(drafter_cfg, ("model_path",), None)
        checkpoint_path = _get_nested(drafter_cfg, ("checkpoint_path",), None)
        resolved_path = resolve_drafter_checkpoint_path(
            model_path, checkpoint_path, resume_step
        )
        if resolved_path is None:
            return
        if os.path.normpath(resolved_path) == os.path.normpath(
            os.fspath(model_path or "")
        ):
            if get_drafter_checkpoint_step(resolved_path) != resume_step:
                message = (
                    f"[drafter resume] no complete draft_step_{resume_step} checkpoint under "
                    f"{checkpoint_path}; model_path={model_path}"
                )
                if checkpoint_save_enabled:
                    raise RuntimeError(message)
                logger.warning("%s; starting drafter state from model_path", message)
            return
        self._speco_set_config_value(drafter_cfg, "model_path", resolved_path)
        logger.info(
            "[drafter resume] resolved global_step=%s checkpoint=%s",
            resume_step,
            resolved_path,
        )

    def _speco_should_save_drafter_checkpoint(self) -> bool:
        if not self.is_drafter_training_enabled(self.config):
            return False
        if self._speco_drafter_training_mode() == "collect_only":
            return False
        if self.drafter_wg is None:
            return False
        if not self._speco_drafter_checkpoint_save_config_enabled():
            return False
        return True

    @staticmethod
    def _speco_flatten_checkpoint_results(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                flattened.extend(
                    SpecoRayPPOTrainer._speco_flatten_checkpoint_results(item)
                )
            return flattened
        return []

    @classmethod
    def _speco_validate_drafter_checkpoint_results(
        cls, value: Any, *, require_saved: bool
    ) -> None:
        results = cls._speco_flatten_checkpoint_results(value)
        allowed_skips = {
            "not_checkpoint_leader",
            "not_checkpoint_replica",
            "not_in_training_group",
        }
        failures = [
            result
            for result in results
            if not bool(result.get("saved", False))
            and result.get("reason") not in allowed_skips
        ]
        if failures:
            raise RuntimeError(f"Drafter checkpoint failed: {failures}")
        if require_saved and not any(
            bool(result.get("saved", False)) for result in results
        ):
            raise RuntimeError(f"Drafter checkpoint produced no saved state: {results}")

    def _speco_save_drafter_checkpoint(
        self,
        *,
        wait: bool = True,
        step: int | None = None,
        include_optimizer: bool = True,
    ):
        if not self._speco_should_save_drafter_checkpoint():
            return None
        if self._speco_ensure_drafter_checkpoint_path() is None:
            return None
        checkpoint_step = int(self.global_steps if step is None else step)
        checkpoint_refs = self.speco_save_checkpoint(
            checkpoint_step,
            wait=wait,
            include_optimizer=include_optimizer,
        )
        if wait:
            results = self._ray_get_if_needed(checkpoint_refs)
            self._speco_validate_drafter_checkpoint_results(results, require_saved=True)
            self._speco_last_saved_drafter_step = checkpoint_step
            return results
        if not hasattr(self, "_pending_drafter_checkpoint_refs"):
            self._pending_drafter_checkpoint_refs = []
        self._pending_drafter_checkpoint_refs.append(checkpoint_refs)
        return checkpoint_refs

    def _speco_mark_final_drafter_checkpoint(self) -> str | None:
        if not self._speco_should_save_drafter_checkpoint():
            return None
        checkpoint_root = self._speco_ensure_drafter_checkpoint_path()
        if checkpoint_root is None:
            return None
        os.makedirs(checkpoint_root, exist_ok=True)
        marker_path = os.path.join(
            checkpoint_root, "final_checkpointed_iteration.txt"
        )
        temporary_path = f"{marker_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as stream:
            stream.write(f"{int(self.global_steps)}\n")
        os.replace(temporary_path, marker_path)
        return marker_path

    def _speco_wait_pending_drafter_checkpoint(self) -> int:
        pending_refs = getattr(self, "_pending_drafter_checkpoint_refs", None)
        if not pending_refs:
            return 0
        self._pending_drafter_checkpoint_refs = []
        for refs in pending_refs:
            results = self._ray_get_if_needed(refs)
            self._speco_validate_drafter_checkpoint_results(results, require_saved=True)
        wait_results = self._ray_get_if_needed(self.speco_wait_checkpoint())
        incomplete = [
            result
            for result in self._speco_flatten_checkpoint_results(wait_results)
            if result.get("completed") is False
        ]
        if incomplete:
            raise RuntimeError(f"Drafter checkpoint wait failed: {incomplete}")
        return len(pending_refs)

    def _speco_should_collect_drafter_this_step(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return speco_step_matches_interval(
            self.global_steps, training_cfg.get("collect_interval_steps", 1)
        )

    def _speco_should_train_drafter_this_step(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return speco_step_matches_interval(
            self.global_steps, training_cfg.get("training_interval_steps", 1)
        )

    def _speco_drafter_training_mode(self) -> str:
        training_cfg = self._speco_drafter_training_config()
        return str(training_cfg.get("mode", "online") or "online").strip().lower()

    def _speco_has_collected_drafter_samples_this_step(self) -> bool:
        return int(getattr(self, "_speco_last_collected_samples", 0) or 0) > 0

    def _speco_should_attempt_drafter_train_this_step(self) -> bool:
        if self._speco_drafter_training_mode() == "collect_only":
            return False
        if not self._speco_should_train_drafter_this_step():
            return False
        if self._speco_has_collected_drafter_samples_this_step():
            return True
        training_cfg = self._speco_drafter_training_config()
        if self._speco_oldlogprob_collection_requested():
            return False
        return bool(training_cfg.get("use_data_buffer", False))

    def _speco_drafter_collection_guard(self) -> tuple[bool, str]:
        """Reject a drafter update when collection transport is incomplete."""
        if not self._speco_oldlogprob_collection_requested():
            return True, "not_oldlogprob_collection"

        training_cfg = self._speco_drafter_training_config()
        planned = int(
            getattr(self, "_speco_last_oldlogprob_planned_samples", 0) or 0
        )
        materialized = int(
            getattr(self, "_speco_last_oldlogprob_collected_samples", 0) or 0
        )
        dispatched = int(
            getattr(self, "_speco_last_oldlogprob_dispatched_samples", 0) or 0
        )
        consumed = int(
            getattr(self, "_speco_last_oldlogprob_consumed_samples", 0) or 0
        )
        min_materialization = float(
            training_cfg.get("oldlogprob_min_materialization_ratio", 0.95) or 0.0
        )
        min_consumption = float(
            training_cfg.get("oldlogprob_min_consumption_ratio", 0.95) or 0.0
        )
        min_trace_coverage = float(
            training_cfg.get("spec_trace_min_sample_coverage", 0.95) or 0.0
        )
        min_teacher_coverage = float(
            training_cfg.get("spec_trace_min_teacher_logprob_coverage", 0.95) or 0.0
        )

        if planned > 0 and materialized / planned < min_materialization:
            return False, "materialization_coverage"
        if dispatched > 0 and consumed / dispatched < min_consumption:
            return False, "worker_consumption_coverage"

        diagnostics = getattr(self, "_speco_last_spec_trace_diagnostics", {}) or {}
        selected_draft_tokens = int(
            diagnostics.get("engine_scheduled_draft_tokens", 0) or 0
        )
        rollout_stats = (
            getattr(self, "_speco_last_rollout_metrics", {}) or {}
            if getattr(self, "_speco_last_rollout_metrics_step", None)
            == getattr(self, "global_steps", None)
            else {}
        )
        rollout_draft_tokens = int(
            float(rollout_stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0)
        )
        speculative_decode_ran = selected_draft_tokens > 0 or rollout_draft_tokens > 0
        trace_samples = int(getattr(self, "_speco_last_spec_trace_samples", 0) or 0)
        trace_coverage = trace_samples / materialized if materialized > 0 else 0.0
        if speculative_decode_ran and trace_coverage < min_trace_coverage:
            return False, "spec_trace_coverage"
        rejected_k3_alpha = float(
            training_cfg.get("dspark_rejected_k3_alpha", 0.0) or 0.0
        )
        teacher_tokens = int(
            getattr(self, "_speco_last_spec_trace_teacher_logprob_tokens", 0) or 0
        )
        draft_tokens = int(
            getattr(self, "_speco_last_spec_trace_draft_tokens", 0) or 0
        )
        teacher_coverage = teacher_tokens / draft_tokens if draft_tokens > 0 else 0.0
        if (
            rejected_k3_alpha > 0
            and speculative_decode_ran
            and teacher_coverage < min_teacher_coverage
        ):
            return False, "spec_trace_teacher_logprob_coverage"
        return True, "ok"

    def _speco_set_drafter_global_step(self, *, log_timing: bool = True):
        return self._ray_get_if_needed(self.speco_set_global_step(self.global_steps))

    def _speco_collect_rollout_features_rpc(
        self, source: str, buckets: list[list[dict]]
    ):
        if not buckets or not any(bucket for bucket in buckets):
            return None
        rpc_started = time.perf_counter()
        dispatched = sum(len(bucket) for bucket in buckets)
        result = self._ray_get_if_needed(
            self.speco_collect_rollout_features(buckets)
        )
        if source == "oldlogprob":
            self._speco_last_oldlogprob_collect_rpc_elapsed_sec = (
                time.perf_counter() - rpc_started
            )
            collection_results = self._speco_flatten_checkpoint_results(result)
            consumed = sum(
                int(item.get("stored_samples", 0) or 0)
                for item in collection_results
            )
            consumed_ids = {
                str(sample_id)
                for item in collection_results
                for sample_id in (item.get("sample_ids") or [])
                if sample_id not in (None, "")
            }
            dispatched_ids = {
                str(sample.get("sample_id"))
                for bucket in buckets
                for sample in bucket
                if sample.get("sample_id") not in (None, "")
            }
            self._speco_last_oldlogprob_dispatched_samples = dispatched
            self._speco_last_oldlogprob_consumed_samples = consumed
            self._speco_last_oldlogprob_unique_samples = len(dispatched_ids)
            self._speco_last_oldlogprob_consumed_unique_samples = len(consumed_ids)
            missing_hidden = sum(
                int(item.get("missing_hidden_samples", 0) or 0)
                for item in collection_results
            )
            if missing_hidden:
                drop_reasons = dict(
                    getattr(self, "_speco_last_oldlogprob_drop_reasons", {})
                )
                drop_reasons["worker_missing_hidden"] = int(
                    drop_reasons.get("worker_missing_hidden", 0)
                ) + missing_hidden
                self._speco_last_oldlogprob_drop_reasons = drop_reasons
        return result

    def _speco_oldlogprob_collection_requested(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return bool(training_cfg.get("collect_hidden_states_from_old_logprob", False))

    def _speco_oldlogprob_collection_enabled(self) -> bool:
        if (
            not self._speco_online_enabled()
            or not self._speco_oldlogprob_collection_requested()
        ):
            return False
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("collect_hidden_states_from_sgl", False)):
            raise ValueError(
                "SPECO old-logprob hidden collection requires "
                "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false"
            )
        if bool(training_cfg.get("use_logits", False)):
            raise ValueError(
                "SPECO old-logprob hidden collection currently supports use_logits=false only"
            )
        strategy = str(
            _get_nested(self.config, ("actor_rollout_ref", "actor", "strategy"), "")
            or ""
        ).lower()
        if strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(
                "SPECO old-logprob hidden collection currently supports actor.strategy=fsdp/fsdp2 only, "
                f"got {strategy!r}"
            )
        capture_impl = str(
            training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook")
            or "forward_hook"
        )
        if capture_impl not in {"forward_hook", "output_hidden_states"}:
            raise ValueError(
                f"Unsupported SPECO old-logprob hidden capture impl: {capture_impl!r}"
            )
        return True

    def _speco_oldlogprob_entropy_config_value(self):
        training_cfg = self._speco_drafter_training_config()
        value = training_cfg.get("old_logprob_calculate_entropy", None)
        if value is None:
            value = _get_nested(
                self.config, ("actor_rollout_ref", "actor", "calculate_entropy"), None
            )
        return value

    @staticmethod
    def _speco_bool_config(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _speco_oldlogprob_entropy_hook_enabled(self) -> bool:
        value = self._speco_oldlogprob_entropy_config_value()
        if value is None and not self.is_drafter_rollout_enabled(self.config):
            return False
        return not self._speco_oldlogprob_calculate_entropy()

    def _speco_oldlogprob_calculate_entropy(self) -> bool:
        value = self._speco_oldlogprob_entropy_config_value()
        if value is None:
            value = False
        return self._speco_bool_config(value)

    def _speco_oldlogprob_hidden_capture_impl(self) -> str:
        training_cfg = self._speco_drafter_training_config()
        return str(
            training_cfg.get("old_logprob_hidden_capture_impl", "forward_hook")
            or "forward_hook"
        )

    def _speco_oldlogprob_hidden_layout(self) -> str:
        drafter_cfg = self._speco_drafter_config()
        algorithm = str(
            _get_nested(drafter_cfg, ("speculative_algorithm",), "") or ""
        ).upper()
        training_cfg = self._speco_drafter_training_config()
        if algorithm == "DSPARK" and dspark_target_distribution_loss_enabled(
            training_cfg
        ):
            return "dflash_aux_plus_last"
        return (
            "dflash_aux"
            if algorithm in {"DFLASH", "DSPARK"}
            else "eagle3_aux_plus_last"
        )

    @staticmethod
    def _speco_oldlogprob_window_train_rows(training_cfg) -> int:
        window_rows = training_cfg.get("hidden_state_window_tokens_per_sample")
        if window_rows is None:
            window_rows = training_cfg.get("hidden_state_window_min_rows", 64)
        return int(window_rows or 0)

    @staticmethod
    def _speco_oldlogprob_window_mode(training_cfg) -> str:
        mode = (
            str(
                training_cfg.get("hidden_state_window_mode", "stratified")
                or "stratified"
            )
            .strip()
            .lower()
        )
        if mode not in {"front", "random", "stratified"}:
            return "stratified"
        return mode

    @staticmethod
    def _speco_oldlogprob_window_strata_weights(
        training_cfg,
    ) -> tuple[float, float, float]:
        raw = training_cfg.get("hidden_state_window_strata_weights", [0.34, 0.33, 0.33])
        if isinstance(raw, Mapping):
            values = [raw.get(name, 0.0) for name in ("front", "middle", "tail")]
        else:
            try:
                values = list(raw)
            except TypeError:
                values = []
        if len(values) != 3:
            values = [0.34, 0.33, 0.33]
        weights = tuple(max(float(value or 0.0), 0.0) for value in values)
        total = sum(weights)
        if total <= 0:
            return (1.0 / 3.0,) * 3
        return tuple(value / total for value in weights)

    @classmethod
    def _speco_oldlogprob_stratified_window_offset(
        cls,
        *,
        sample_key: str,
        max_start_offset: int,
        weights: tuple[float, float, float],
        stratum_index: int | None = None,
    ) -> tuple[int, str]:
        if max_start_offset <= 0:
            return 0, "full"
        if stratum_index is None:
            stratum_draw = cls._speco_hash_fraction(f"{sample_key}:window:stratum")
            if stratum_draw < weights[0]:
                stratum = 0
            elif stratum_draw < weights[0] + weights[1]:
                stratum = 1
            else:
                stratum = 2
        else:
            stratum = min(max(int(stratum_index), 0), 2)
        num_offsets = max_start_offset + 1
        start = (stratum * num_offsets + 2) // 3
        end = ((stratum + 1) * num_offsets + 2) // 3
        end = max(end, start + 1)
        offset = start + cls._speco_hash_int(
            f"{sample_key}:window:offset", end - start - 1
        )
        return min(offset, max_start_offset), ("front", "middle", "tail")[stratum]

    @staticmethod
    def _speco_next_oldlogprob_window_stratum(
        assigned_counts: list[int], weights: tuple[float, float, float]
    ) -> int:
        next_total = sum(assigned_counts) + 1
        return max(
            range(3),
            key=lambda index: (
                next_total * weights[index] - assigned_counts[index],
                -index,
            ),
        )

    @staticmethod
    def _speco_oldlogprob_aux_layer_ids_from_config(config) -> Any | None:
        candidates = (
            ("model", "eagle_config", "target_hidden_layer_ids"),
            ("model", "eagle_config", "eagle_aux_hidden_state_layer_ids"),
            ("eagle_config", "target_hidden_layer_ids"),
            ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
            ("target_hidden_layer_ids",),
            ("eagle_aux_hidden_state_layer_ids",),
            ("target_layer_ids",),
        )
        for path in candidates:
            layer_ids = _get_nested(config, path, None)
            if layer_ids is not None:
                return layer_ids
        return None

    @staticmethod
    def _speco_normalize_oldlogprob_aux_layer_ids(layer_ids) -> list[int] | None:
        if layer_ids is None:
            return None
        if isinstance(layer_ids, int):
            return [int(layer_ids)]
        if isinstance(layer_ids, str):
            raw = layer_ids.strip()
            if not raw:
                return None
            if raw.startswith("["):
                layer_ids = json.loads(raw)
            else:
                layer_ids = [part.strip() for part in raw.split(",") if part.strip()]
        return [int(layer_id) for layer_id in list(layer_ids)]

    @staticmethod
    def _speco_load_model_config(model_path: Any) -> dict[str, Any] | None:
        if not model_path:
            return None
        config_path = os.path.join(str(model_path), "config.json")
        try:
            with open(config_path, encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return None
        return config if isinstance(config, dict) else None

    @staticmethod
    def _speco_num_hidden_layers_from_config(config) -> int | None:
        candidates = (
            ("num_hidden_layers",),
            ("text_config", "num_hidden_layers"),
            ("model", "num_hidden_layers"),
            ("n_layer",),
            ("num_layers",),
        )
        for path in candidates:
            value = _get_nested(config, path, None)
            if value is not None:
                return int(value)
        return None

    def _speco_target_num_hidden_layers(self) -> int | None:
        target_model_cfg = _get_nested(
            self.config, ("actor_rollout_ref", "model"), None
        )
        num_layers = self._speco_num_hidden_layers_from_config(target_model_cfg)
        if num_layers is not None:
            return num_layers
        target_model_path = _get_nested(target_model_cfg, ("path",), None)
        target_config = self._speco_load_model_config(target_model_path)
        return self._speco_num_hidden_layers_from_config(target_config)

    @staticmethod
    def _speco_default_eagle3_aux_layer_ids(num_hidden_layers: int) -> list[int]:
        num_hidden_layers = int(num_hidden_layers)
        if num_hidden_layers <= 0:
            raise RuntimeError(
                f"SPECO cannot derive EAGLE3 aux hidden layers from num_hidden_layers={num_hidden_layers}"
            )
        return [2, num_hidden_layers // 2, num_hidden_layers - 3]

    def _speco_validate_sglang_aux_last_layer_norm(self) -> None:
        """Fail closed if SGLang collection would capture the last aux layer pre-norm.

        SGLang's aux/context capture skips the target's final norm, so a last-layer
        (or ``-1``) ``target_layer_id`` diverges from the offline / old-logprob
        (post-norm / embedding) semantics; see ``assert_sglang_aux_last_layer_norm_safe``.
        Best-effort: skips silently when the layer ids or target depth cannot be resolved.
        """
        training_cfg = self._speco_drafter_training_config()
        if not bool(training_cfg.get("collect_hidden_states_from_sgl", False)):
            return
        drafter_cfg = self._speco_drafter_config()
        model_configs = []
        for path_key in ("model_path", "checkpoint_path"):
            model_config = self._speco_load_model_config(
                _get_nested(drafter_cfg, (path_key,), None)
            )
            if model_config is not None:
                model_configs.append(model_config)
        num_hidden_layers = self._speco_target_num_hidden_layers()
        try:
            layer_ids = resolve_oldlogprob_aux_layer_ids(
                drafter_cfg,
                target_num_hidden_layers=num_hidden_layers,
                model_configs=model_configs,
            )
        except Exception:  # noqa: BLE001 -- best-effort guard, never masks the real resolve path
            return
        assert_sglang_aux_last_layer_norm_safe(
            layer_ids,
            num_hidden_layers,
            collect_from_sgl=True,
            allow_prenorm_last=bool(
                training_cfg.get("allow_sglang_prenorm_last_layer", False)
            ),
        )

    def _speco_oldlogprob_aux_layer_ids(self) -> list[int]:
        drafter_cfg = self._speco_drafter_config()
        model_configs = []
        for path_key in ("model_path", "checkpoint_path"):
            model_config = self._speco_load_model_config(
                _get_nested(drafter_cfg, (path_key,), None)
            )
            if model_config is not None:
                model_configs.append(model_config)

        num_hidden_layers = self._speco_target_num_hidden_layers()
        layer_ids = resolve_oldlogprob_aux_layer_ids(
            drafter_cfg,
            target_num_hidden_layers=num_hidden_layers,
            model_configs=model_configs,
        )
        if layer_ids is None:
            raise RuntimeError(
                "SPECO old-logprob hidden collection requires explicit DFlash target_layer_ids, "
                "EAGLE3 eagle_aux_hidden_state_layer_ids/target_hidden_layer_ids in drafter config or checkpoint, "
                "or a readable target model config at actor_rollout_ref.model.path/config.json with "
                "num_hidden_layers. Refusing to guess aux hidden layers."
            )
        return layer_ids

    @staticmethod
    def _speco_hash_fraction(key: str) -> float:
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    @staticmethod
    def _speco_hash_int(key: str, inclusive_max: int) -> int:
        if inclusive_max <= 0:
            return 0
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % (
            inclusive_max + 1
        )

    @classmethod
    def _speco_sample_identity(
        cls,
        batch: DataProto,
        batch_idx: int,
        prompt_ids: torch.Tensor,
    ) -> str:
        non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
        external_id = None
        for key in ("uid", "request_id", "rollout_id", "data_id", "index"):
            external_id = cls._speco_sequence_item(non_tensor_batch.get(key), batch_idx)
            if external_id not in (None, ""):
                break
        prompt_values = ",".join(
            str(int(token_id)) for token_id in prompt_ids.detach().cpu().reshape(-1)
        )
        prompt_digest = hashlib.blake2b(
            prompt_values.encode(), digest_size=12
        ).hexdigest()
        return f"{external_id or 'anonymous'}:{prompt_digest}"

    def _speco_build_oldlogprob_collect_plan(
        self, batch: DataProto
    ) -> dict[str, Any] | None:
        if not self._speco_oldlogprob_collection_enabled():
            return None
        if not self._speco_should_collect_drafter_this_step():
            return None
        if not self._speco_should_train_drafter_this_step():
            return None
        training_cfg = self._speco_drafter_training_config()
        sample_rate = float(training_cfg.get("collection_sample_rate", 1.0) or 0.0)
        if sample_rate <= 0:
            return None
        window_mode = self._speco_oldlogprob_window_mode(training_cfg)

        batch_tensors = batch.batch
        required_keys = ("prompts", "responses", "attention_mask")
        if any(key not in batch_tensors for key in required_keys):
            return None
        prompts = batch_tensors["prompts"]
        attention_mask = batch_tensors["attention_mask"]
        response_mask = batch_tensors.get("response_mask", None)
        batch_size = int(prompts.size(0))
        prompt_width = int(prompts.size(1))

        train_rows = self._speco_oldlogprob_window_train_rows(training_cfg)
        if train_rows <= 0:
            return None
        max_hidden_rows = train_rows + 1
        collect_mask = torch.zeros(batch_size, dtype=torch.bool)
        hidden_positions = torch.zeros(batch_size, max_hidden_rows, dtype=torch.long)
        hidden_position_mask = torch.zeros(
            batch_size, max_hidden_rows, dtype=torch.bool
        )
        owner_rank = torch.zeros(batch_size, dtype=torch.long)

        owner_count = self._speco_owner_bucket_count()
        if owner_count is None:
            owner_count = 1
        owner_count = max(int(owner_count), 1)
        max_per_owner = training_cfg.get("max_collect_samples_per_step_per_replica", 16)
        max_per_owner = int(max_per_owner) if max_per_owner is not None else batch_size
        max_per_owner = max(max_per_owner, 0)
        max_tokens_per_owner = training_cfg.get(
            "max_collect_tokens_per_step_per_replica", None
        )
        max_tokens_per_owner = (
            int(max_tokens_per_owner) if max_tokens_per_owner is not None else None
        )
        if max_tokens_per_owner is not None:
            max_tokens_per_owner = max(max_tokens_per_owner, 0)
        owner_counts = [0 for _ in range(owner_count)]
        owner_token_counts = [0 for _ in range(owner_count)]
        seed_by_step = bool(training_cfg.get("hidden_state_random_seed_by_step", True))
        step_key = self.global_steps if seed_by_step else "request"
        strata_weights = self._speco_oldlogprob_window_strata_weights(training_cfg)

        prompt_lens: list[int] = []
        response_lens: list[int] = []
        candidate_count = 0
        selected_count = 0
        candidates: list[dict[str, Any]] = []
        drop_reasons = {
            "invalid_prompt_or_response": 0,
            "sample_rate": 0,
            "capacity": 0,
        }
        sample_ids = [None for _ in range(batch_size)]
        fixed_eval_cohort = [False for _ in range(batch_size)]
        fixed_eval_seed = int(training_cfg.get("fixed_eval_seed", 17) or 17)
        fixed_eval_sample_rate = min(
            max(float(training_cfg.get("fixed_eval_sample_rate", 0.1) or 0.0), 0.0),
            1.0,
        )
        for batch_idx in range(batch_size):
            prompt_len = int(
                attention_mask[batch_idx, :prompt_width].detach().sum().item()
            )
            if response_mask is not None:
                response_len = int(response_mask[batch_idx].detach().sum().item())
            else:
                response_len = int(
                    attention_mask[batch_idx, prompt_width:].detach().sum().item()
                )
            prompt_lens.append(prompt_len)
            response_lens.append(response_len)
            if prompt_len <= 0 or response_len <= 0:
                drop_reasons["invalid_prompt_or_response"] += 1
                continue
            candidate_count += 1
            prompt_ids = prompts[batch_idx][
                attention_mask[batch_idx, :prompt_width].bool()
            ][:prompt_len]
            stable_identity = self._speco_sample_identity(
                batch, batch_idx, prompt_ids
            )
            sample_key = (
                f"{step_key}:{batch_idx}:{stable_identity}:{prompt_len}:{response_len}"
            )
            sample_id = hashlib.blake2b(
                f"{self.global_steps}:{batch_idx}:{stable_identity}:{response_len}".encode(),
                digest_size=12,
            ).hexdigest()
            sample_ids[batch_idx] = sample_id
            fixed_eval_cohort[batch_idx] = (
                fixed_eval_sample_rate > 0
                and self._speco_hash_fraction(
                    f"{fixed_eval_seed}:{stable_identity}:fixed-eval"
                )
                < fixed_eval_sample_rate
            )
            if (
                sample_rate < 1.0
                and self._speco_hash_fraction(sample_key) >= sample_rate
            ):
                drop_reasons["sample_rate"] += 1
                continue
            hidden_rows = min(max_hidden_rows, response_len)
            max_start_offset = max(response_len - hidden_rows, 0)
            window_stratum = window_mode
            if window_mode == "random":
                random_offset = self._speco_hash_int(
                    f"{sample_key}:window", max_start_offset
                )
            elif window_mode == "stratified":
                random_offset = 0
                window_stratum = "full" if max_start_offset <= 0 else None
            else:
                random_offset = 0
            candidates.append(
                {
                    "priority": self._speco_hash_fraction(
                        f"{sample_key}:collection-reservoir"
                    ),
                    "batch_idx": batch_idx,
                    "hidden_rows": hidden_rows,
                    "train_base": max(prompt_len - 1, 0),
                    "start_offset": random_offset,
                    "max_start_offset": max_start_offset,
                    "sample_key": sample_key,
                    "window_stratum": window_stratum,
                }
            )

        window_strata = [None for _ in range(batch_size)]
        assigned_strata = [0, 0, 0]
        for candidate in sorted(
            candidates, key=lambda item: (item["priority"], item["batch_idx"])
        ):
            batch_idx = int(candidate["batch_idx"])
            hidden_rows = int(candidate["hidden_rows"])
            eligible_owners = [
                owner
                for owner in range(owner_count)
                if owner_counts[owner] < max_per_owner
                and (
                    max_tokens_per_owner is None
                    or owner_token_counts[owner] + hidden_rows <= max_tokens_per_owner
                )
            ]
            if not eligible_owners:
                drop_reasons["capacity"] += 1
                continue
            owner = min(
                eligible_owners,
                key=lambda rank: (
                    owner_counts[rank],
                    owner_token_counts[rank],
                    rank,
                ),
            )
            start_offset = int(candidate["start_offset"])
            window_stratum = candidate["window_stratum"]
            if window_mode == "stratified" and int(
                candidate["max_start_offset"]
            ) > 0:
                stratum_index = self._speco_next_oldlogprob_window_stratum(
                    assigned_strata, strata_weights
                )
                start_offset, window_stratum = (
                    self._speco_oldlogprob_stratified_window_offset(
                        sample_key=str(candidate["sample_key"]),
                        max_start_offset=int(candidate["max_start_offset"]),
                        weights=strata_weights,
                        stratum_index=stratum_index,
                    )
                )
                assigned_strata[stratum_index] += 1
            start = int(candidate["train_base"]) + start_offset
            positions = torch.arange(start, start + hidden_rows, dtype=torch.long)
            collect_mask[batch_idx] = True
            hidden_positions[batch_idx, :hidden_rows] = positions
            hidden_position_mask[batch_idx, :hidden_rows] = True
            owner_rank[batch_idx] = owner
            window_strata[batch_idx] = window_stratum
            owner_counts[owner] += 1
            owner_token_counts[owner] += hidden_rows
            selected_count += 1

        self._speco_last_raw_drafter_samples = candidate_count
        self._speco_last_oldlogprob_candidate_samples = candidate_count
        self._speco_last_oldlogprob_planned_samples = selected_count
        self._speco_last_oldlogprob_drop_reasons = drop_reasons
        self._speco_last_oldlogprob_window_strata = {
            str(stratum): sum(item == stratum for item in window_strata)
            for stratum in {item for item in window_strata if item is not None}
        }
        if selected_count <= 0:
            return None
        return {
            "collect_mask": collect_mask,
            "hidden_positions": hidden_positions,
            "hidden_position_mask": hidden_position_mask,
            "owner_rank": owner_rank,
            "prompt_lens": prompt_lens,
            "response_lens": response_lens,
            "hidden_rows": max_hidden_rows,
            "owner_count": owner_count,
            "selected_count": selected_count,
            "candidate_count": candidate_count,
            "owner_token_counts": owner_token_counts,
            "window_mode": window_mode,
            "window_strata": window_strata,
            "sample_ids": sample_ids,
            "fixed_eval_cohort": fixed_eval_cohort,
        }

    def _speco_export_fixed_eval_snapshot(self, batch, collect_plan):
        training_cfg = self._speco_drafter_training_config()
        export_steps = training_cfg.get("fixed_eval_export_steps") or []
        step = int(self.global_steps)
        if step not in {int(value) for value in export_steps}:
            return
        from verl_speco.trainer.fixed_eval_snapshot import export_fixed_eval_snapshot

        experiment_id = str(_get_nested(self.config, ("trainer", "experiment_name"), "unknown"))
        run_id = os.getenv("WANDB_RUN_ID") or "no_wandb"
        checkpoint_root = _get_nested(self.config, ("trainer", "default_local_dir"), None)
        if not checkpoint_root:
            raise ValueError("Fixed eval export requires trainer.default_local_dir")
        actor_checkpoint = os.path.join(checkpoint_root, f"global_step_{step - 1}", "actor")
        if step == 1:
            actor_checkpoint = _get_nested(self.config, ("actor_rollout_ref", "model", "path"), "")
        root = training_cfg.get("fixed_eval_export_dir") or os.path.join(
            "outputs", "drafter_analysis", experiment_id, run_id, "fixed_eval_samples"
        )
        started = time.perf_counter()
        manifest = export_fixed_eval_snapshot(
            batch, collect_mask=collect_plan["collect_mask"], root=root, step=step,
            sample_count=int(training_cfg.get("fixed_eval_export_samples", 128)),
            seed=int(training_cfg.get("fixed_eval_export_seed", 17)),
            actor_checkpoint=actor_checkpoint, experiment_id=experiment_id, run_id=run_id,
            rollout_config={
                "temperature": _get_nested(self.config, ("actor_rollout_ref", "rollout", "temperature"), None),
                "top_p": _get_nested(self.config, ("actor_rollout_ref", "rollout", "top_p"), None),
                "top_k": _get_nested(self.config, ("actor_rollout_ref", "rollout", "top_k"), None),
                "max_response_length": _get_nested(self.config, ("data", "max_response_length"), None),
                "draft_attention": _get_nested(self.config, ("actor_rollout_ref", "rollout", "drafter", "vllm", "dspark_attention_mode"), "full"),
            },
        )
        self._speco_fixed_eval_export_metrics = {
            "drafter/fixed_export/samples": manifest["sample_count"],
            "drafter/fixed_export/unique_prompts": manifest["unique_prompts"],
            "drafter/fixed_export/actor_snapshot_step": manifest["actor_snapshot_step"],
            "timing_s/drafter_fixed_eval_export": time.perf_counter() - started,
        }
        logger.warning(
            "[SPECO fixed eval] rollout_step=%s actor_step=%s samples=%s prompts=%s directory=%s",
            step, step - 1, manifest["sample_count"], manifest["unique_prompts"], root,
        )

    @staticmethod
    def _speco_tensor_rows(tensor: torch.Tensor | None) -> list[torch.Tensor]:
        if tensor is None:
            return []
        if torch.is_tensor(tensor) and tensor.is_nested:
            return list(tensor.unbind())
        if torch.is_tensor(tensor):
            return [row for row in tensor]
        return []

    @staticmethod
    def _speco_sequence_item(value: Any, index: int):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[index] if 0 <= index < len(value) else None
        if not isinstance(value, (str, bytes, dict, torch.Tensor)):
            try:
                return value[index] if 0 <= index < len(value) else None
            except (IndexError, KeyError, TypeError):
                return None
        return None

    @staticmethod
    def _speco_flatten_non_tensor_rows(value: Any):
        if not isinstance(value, (list, tuple)):
            return value
        if not value or not all(isinstance(item, (list, tuple)) for item in value):
            return value
        flattened = []
        for item in value:
            flattened.extend(item)
        return flattened

    @staticmethod
    def _speco_sum_timing_rows(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            return None
        if torch.is_tensor(tensor) and tensor.is_nested:
            rows = [
                row.reshape(-1).float() for row in tensor.unbind() if row.numel() > 0
            ]
            if not rows:
                return None
            width = min(int(row.numel()) for row in rows)
            return torch.stack([row[:width] for row in rows], dim=0).sum(dim=0).cpu()
        if tensor.numel() == 0:
            return None
        if tensor.dim() == 1:
            return tensor.float().cpu()
        return tensor.reshape(-1, tensor.shape[-1]).float().sum(dim=0).cpu()

    def _speco_collect_oldlogprob_features(
        self,
        batch: DataProto,
        collect_plan: dict[str, Any] | None,
        output: Any,
    ) -> int:
        if not collect_plan:
            return 0
        hidden_states = tu.get(output, OLD_LOGPROB_HIDDEN_STATES_KEY)
        hidden_refs = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_REFS_KEY)
        )
        hidden_ref_meta = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_REF_META_KEY)
        )
        chunk_refs = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY)
        )
        chunk_meta = self._speco_flatten_non_tensor_rows(
            tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_META_KEY)
        )
        if hidden_states is None and hidden_refs is None and chunk_refs is None:
            return 0
        hidden_rows = self._speco_tensor_rows(hidden_states)
        if not hidden_rows and not hidden_refs and not chunk_refs:
            return 0
        timing = self._speco_sum_timing_rows(tu.get(output, OLD_LOGPROB_TIMING_KEY))
        if timing is not None and int(timing.numel()) >= 2:
            self._speco_last_oldlogprob_select_elapsed_sec = (
                float(timing[0].item()) / 1_000_000.0
            )
            self._speco_last_oldlogprob_sp_merge_elapsed_sec = (
                float(timing[1].item()) / 1_000_000.0
            )
            if int(timing.numel()) >= 5:
                self._speco_last_oldlogprob_concat_elapsed_sec = (
                    float(timing[2].item()) / 1_000_000.0
                )
                self._speco_last_oldlogprob_cpu_copy_elapsed_sec = (
                    float(timing[3].item()) / 1_000_000.0
                )
                self._speco_last_oldlogprob_ray_put_elapsed_sec = (
                    float(timing[4].item()) / 1_000_000.0
                )

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask_tensor = batch.batch.get("response_mask", None)
        collect_mask = collect_plan["collect_mask"]
        hidden_positions = collect_plan["hidden_positions"]
        hidden_position_mask = collect_plan["hidden_position_mask"]
        owner_rank = collect_plan["owner_rank"]
        prompt_lens = collect_plan["prompt_lens"]
        response_lens = collect_plan["response_lens"]
        window_strata = collect_plan.get("window_strata") or []
        sample_ids = collect_plan.get("sample_ids") or []
        fixed_eval_cohort = collect_plan.get("fixed_eval_cohort") or []
        non_tensor_batch = getattr(batch, "non_tensor_batch", {})
        speculative_traces = non_tensor_batch.get(SPECO_VLLM_SPECULATIVE_TRACE_KEY)
        speculative_trace_diagnostics = non_tensor_batch.get(
            SPECO_VLLM_SPECULATIVE_TRACE_DIAGNOSTICS_KEY
        )
        buckets: list[list[dict[str, Any]]] = [
            [] for _ in range(int(collect_plan["owner_count"]))
        ]
        collected_rows = 0
        payload_bytes = 0
        trace_samples = 0
        trace_verify_steps = 0
        trace_draft_tokens = 0
        trace_teacher_logprob_tokens = 0
        trace_accepted_tokens = 0
        trace_first_rejections = 0
        trace_unverified_suffix_tokens = 0
        trace_all_accepted_steps = 0
        trace_diagnostics_totals: dict[str, int] = {}
        materialized_strata: dict[str, int] = {}
        materialize_drop_reasons = {
            "invalid_positions": 0,
            "missing_hidden": 0,
        }
        fixed_trace_samples = 0
        fixed_trace_verify_steps = 0
        fixed_trace_draft_tokens = 0
        fixed_trace_accepted_tokens = 0
        fixed_trace_first_rejections = 0
        fixed_trace_unverified_suffix_tokens = 0
        fixed_trace_all_accepted_steps = 0
        fixed_materialized_samples = 0
        sample_ref_chunks: dict[int, list[dict[str, Any]]] = {}
        if isinstance(chunk_refs, (list, tuple)) and isinstance(
            chunk_meta, (list, tuple)
        ):
            for chunk_index, (chunk_ref, chunk_info) in enumerate(
                zip(chunk_refs, chunk_meta, strict=False)
            ):
                if chunk_ref is None or not isinstance(chunk_info, dict):
                    continue
                sample_indices = chunk_info.get("sample_indices") or []
                starts = chunk_info.get("starts") or []
                lengths = chunk_info.get("lengths") or []
                row_indices_payload = chunk_info.get("row_indices") or []
                for item_idx, batch_idx in enumerate(sample_indices):
                    try:
                        batch_idx = int(batch_idx)
                    except (TypeError, ValueError):
                        continue
                    if batch_idx < 0:
                        continue
                    start = int(starts[item_idx]) if item_idx < len(starts) else 0
                    length = int(lengths[item_idx]) if item_idx < len(lengths) else 0
                    row_indices = (
                        row_indices_payload[item_idx]
                        if item_idx < len(row_indices_payload)
                        else None
                    )
                    sample_ref_chunks.setdefault(batch_idx, []).append(
                        {
                            "ref": chunk_ref,
                            "chunk_index": int(chunk_index),
                            "chunk_start": start,
                            "chunk_length": length,
                            "chunk_row_indices": row_indices,
                            "dtype": chunk_info.get("dtype"),
                            "shape": chunk_info.get("shape"),
                        }
                    )

        item_count = max(
            int(collect_mask.numel()),
            len(hidden_rows),
            len(hidden_refs) if isinstance(hidden_refs, (list, tuple)) else 0,
            max(sample_ref_chunks.keys(), default=-1) + 1,
        )
        for batch_idx in range(item_count):
            if batch_idx >= int(collect_mask.numel()) or not bool(
                collect_mask[batch_idx].item()
            ):
                continue
            prompt_len = int(prompt_lens[batch_idx])
            response_len = int(response_lens[batch_idx])
            position_mask = hidden_position_mask[batch_idx].reshape(-1).bool()
            valid_positions = hidden_positions[batch_idx].reshape(-1)[position_mask]
            valid_rows = int(valid_positions.numel())
            if valid_rows <= 0:
                materialize_drop_reasons["invalid_positions"] += 1
                continue
            hidden_ref = self._speco_sequence_item(hidden_refs, batch_idx)
            ref_meta = self._speco_sequence_item(hidden_ref_meta, batch_idx)
            ref_chunks = sample_ref_chunks.get(batch_idx)
            hidden = hidden_rows[batch_idx] if batch_idx < len(hidden_rows) else None
            if ref_chunks:
                collected_rows += sum(
                    _speco_ref_meta_row_count(chunk, 0) for chunk in ref_chunks
                )
                payload_bytes += sum(
                    int(chunk.get("chunk_length", 0) or 0)
                    * int((chunk.get("shape") or [0, 0])[-1] or 0)
                    * 2
                    for chunk in ref_chunks
                )
            elif hidden_ref is None:
                if hidden is None:
                    materialize_drop_reasons["missing_hidden"] += 1
                    continue
                hidden = hidden[:valid_rows].contiguous()
                if hidden.numel() == 0:
                    materialize_drop_reasons["missing_hidden"] += 1
                    continue
                collected_rows += int(hidden.size(0))
                payload_bytes += int(hidden.numel()) * int(hidden.element_size())
            else:
                collected_rows += _speco_ref_meta_rows(ref_meta) or valid_rows
                payload_bytes += _speco_ref_meta_nbytes(ref_meta)
            owner = int(owner_rank[batch_idx].item())
            prompt_mask = attention_mask[batch_idx, : prompts.size(1)].bool()
            if response_mask_tensor is not None:
                response_mask = response_mask_tensor[batch_idx].bool()
            else:
                response_mask = attention_mask[
                    batch_idx, prompts.size(1) : prompts.size(1) + responses.size(1)
                ].bool()
            prompt_ids = prompts[batch_idx][prompt_mask].detach().cpu()
            response_ids = responses[batch_idx][response_mask].detach().cpu()
            prompt_ids = prompt_ids[:prompt_len]
            response_ids = response_ids[:response_len]
            sample_input_ids = torch.cat([prompt_ids, response_ids], dim=0)
            sample = {
                "input_ids": sample_input_ids.unsqueeze(0),
                "prompts": prompt_ids.unsqueeze(0),
                "responses": response_ids.unsqueeze(0),
                "hidden_positions": valid_positions.detach().cpu().unsqueeze(0),
                "hidden_states_layout": self._speco_oldlogprob_hidden_layout(),
                "hidden_position_start": int(valid_positions[0].item()),
                "hidden_position_end": int(valid_positions[-1].item()) + 1,
                "global_step": self.global_steps,
                "replica_rank": owner,
            }
            if batch_idx < len(sample_ids) and sample_ids[batch_idx] is not None:
                sample["sample_id"] = str(sample_ids[batch_idx])
            is_fixed_sample = bool(
                batch_idx < len(fixed_eval_cohort) and fixed_eval_cohort[batch_idx]
            )
            if is_fixed_sample:
                fixed_materialized_samples += 1
            if batch_idx < len(window_strata):
                sample["hidden_window_stratum"] = window_strata[batch_idx]
                stratum = str(window_strata[batch_idx] or "unknown")
                materialized_strata[stratum] = materialized_strata.get(stratum, 0) + 1
            speculative_trace = self._speco_sequence_item(speculative_traces, batch_idx)
            trace_diagnostics = self._speco_sequence_item(
                speculative_trace_diagnostics, batch_idx
            )
            if speculative_trace is not None:
                sample[SPECO_VLLM_SPECULATIVE_TRACE_KEY] = speculative_trace
            if isinstance(trace_diagnostics, dict):
                sample[SPECO_VLLM_SPECULATIVE_TRACE_DIAGNOSTICS_KEY] = (
                    trace_diagnostics
                )
                for key, value in trace_diagnostics.items():
                    try:
                        trace_diagnostics_totals[str(key)] = int(
                            trace_diagnostics_totals.get(str(key), 0)
                        ) + int(value or 0)
                    except (TypeError, ValueError):
                        continue
            if isinstance(speculative_trace, (list, tuple)) and speculative_trace:
                trace_samples += 1
                if is_fixed_sample:
                    fixed_trace_samples += 1
                for event in speculative_trace:
                    if not isinstance(event, dict):
                        continue
                    trace_verify_steps += 1
                    if is_fixed_sample:
                        fixed_trace_verify_steps += 1
                    try:
                        event_drafts = int(event.get("draft_count", 0) or 0)
                        event_accepted = min(
                            max(int(event.get("accepted_count", 0) or 0), 0),
                            max(event_drafts, 0),
                        )
                        trace_draft_tokens += event_drafts
                        teacher_logprobs = event.get("draft_teacher_logprobs")
                        if isinstance(teacher_logprobs, (list, tuple)):
                            trace_teacher_logprob_tokens += sum(
                                1
                                for value in teacher_logprobs[: max(event_drafts, 0)]
                                if isinstance(value, (int, float))
                                and math.isfinite(float(value))
                            )
                        trace_accepted_tokens += event_accepted
                        if event_accepted < event_drafts:
                            trace_first_rejections += 1
                            trace_unverified_suffix_tokens += max(
                                event_drafts - event_accepted - 1, 0
                            )
                        else:
                            trace_all_accepted_steps += 1
                        if is_fixed_sample:
                            fixed_trace_draft_tokens += event_drafts
                            fixed_trace_accepted_tokens += event_accepted
                            if event_accepted < event_drafts:
                                fixed_trace_first_rejections += 1
                                fixed_trace_unverified_suffix_tokens += max(
                                    event_drafts - event_accepted - 1, 0
                                )
                            else:
                                fixed_trace_all_accepted_steps += 1
                    except (TypeError, ValueError):
                        continue
            if ref_chunks:
                sample["hidden_states_ref_chunks"] = ref_chunks
            elif hidden_ref is None:
                hidden = cast(torch.Tensor, hidden)
                sample["hidden_states"] = hidden.detach().cpu().unsqueeze(0)
            else:
                sample["hidden_states_ref"] = hidden_ref
                sample["hidden_states_ref_meta"] = ref_meta
            buckets[owner].append(sample)

        collected = sum(len(bucket) for bucket in buckets)
        if collected <= 0:
            planned_drop_reasons = dict(
                getattr(self, "_speco_last_oldlogprob_drop_reasons", {})
            )
            for key, value in materialize_drop_reasons.items():
                planned_drop_reasons[key] = int(
                    planned_drop_reasons.get(key, 0)
                ) + int(value)
            self._speco_last_oldlogprob_drop_reasons = planned_drop_reasons
            return 0
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        if dispatch_bucket_count is not None and dispatch_bucket_count > len(buckets):
            buckets.extend([[] for _ in range(dispatch_bucket_count - len(buckets))])
        self._speco_set_drafter_global_step(log_timing=False)
        self._speco_collect_rollout_features_rpc("oldlogprob", buckets)
        self._speco_last_collected_samples = collected
        self._speco_last_oldlogprob_collected_samples = collected
        self._speco_last_oldlogprob_collected_rows = collected_rows
        self._speco_last_oldlogprob_payload_mib = payload_bytes / float(1024 * 1024)
        self._speco_last_oldlogprob_materialized_window_strata = materialized_strata
        planned_drop_reasons = dict(
            getattr(self, "_speco_last_oldlogprob_drop_reasons", {})
        )
        for key, value in materialize_drop_reasons.items():
            planned_drop_reasons[key] = int(planned_drop_reasons.get(key, 0)) + int(
                value
            )
        self._speco_last_oldlogprob_drop_reasons = planned_drop_reasons
        self._speco_last_spec_trace_samples = trace_samples
        self._speco_last_spec_trace_verify_steps = trace_verify_steps
        self._speco_last_spec_trace_draft_tokens = trace_draft_tokens
        self._speco_last_spec_trace_teacher_logprob_tokens = (
            trace_teacher_logprob_tokens
        )
        self._speco_last_spec_trace_accepted_tokens = trace_accepted_tokens
        self._speco_last_spec_trace_first_rejections = trace_first_rejections
        self._speco_last_spec_trace_unverified_suffix_tokens = (
            trace_unverified_suffix_tokens
        )
        self._speco_last_spec_trace_all_accepted_steps = trace_all_accepted_steps
        self._speco_last_spec_trace_diagnostics = trace_diagnostics_totals
        self._speco_last_fixed_eval_trace_metrics = {
            "materialized_samples": fixed_materialized_samples,
            "samples": fixed_trace_samples,
            "verify_steps": fixed_trace_verify_steps,
            "draft_tokens": fixed_trace_draft_tokens,
            "accepted_tokens": fixed_trace_accepted_tokens,
            "first_rejections": fixed_trace_first_rejections,
            "unverified_suffix_tokens": fixed_trace_unverified_suffix_tokens,
            "all_accepted_steps": fixed_trace_all_accepted_steps,
            "actor_snapshot_step": int(self.global_steps),
        }
        return collected

    def _speco_num_rollout_replicas(self, samples: list[dict]) -> int:
        sample_max = (
            max((int(sample.get("replica_rank", 0)) for sample in samples), default=0)
            + 1
        )
        rollout_cfg = _get_nested(self.config, ("actor_rollout_ref", "rollout"), None)
        rollout_dp = int(_get_nested(rollout_cfg, ("data_parallel_size",), 1) or 1)
        return max(sample_max, rollout_dp, 1)

    def _speco_collect_generation_samples(self, gen_batch_output: Any) -> int:
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_collect_interval_matched = int(
            self._speco_should_collect_drafter_this_step()
        )
        if not self._speco_online_enabled():
            return 0
        samples = pop_drafter_samples(gen_batch_output)
        self._speco_last_raw_drafter_samples = len(samples)
        if not samples:
            return 0
        if not self._speco_should_collect_drafter_this_step():
            return 0

        num_replicas = self._speco_num_rollout_replicas(samples)
        buckets = bucket_drafter_samples_by_replica(samples, num_replicas=num_replicas)

        # Some verl worker-group dispatch meshes may expose more owner buckets
        # than rollout replicas. Extend with empty buckets when that mapping is
        # discoverable, while keeping sample validation based on replicas.
        dispatch_bucket_count = self._speco_dispatch_bucket_count()
        if dispatch_bucket_count is not None and dispatch_bucket_count > len(buckets):
            buckets.extend([[] for _ in range(dispatch_bucket_count - len(buckets))])

        self._speco_set_drafter_global_step()
        self._speco_collect_rollout_features_rpc("rollout", buckets)
        self._speco_last_collected_samples = len(samples)
        return len(samples)

    def _speco_owner_route_mapping(self):
        worker_group = self.drafter_wg
        if worker_group is None:
            return None
        mapping = None
        dispatch_info = getattr(worker_group, "_dispatch_info", None)
        if isinstance(dispatch_info, dict):
            mapping = dispatch_info.get("drafter_owner_route")
        if mapping is None and hasattr(worker_group, "_query_dispatch_info"):
            mapping = worker_group._query_dispatch_info("drafter_owner_route")
            if isinstance(dispatch_info, dict):
                dispatch_info["drafter_owner_route"] = mapping
        return mapping

    def _speco_owner_route_collect_mask(self):
        worker_group = self.drafter_wg
        if worker_group is None:
            return None
        collect_mask = None
        collect_info = getattr(worker_group, "_collect_info", None)
        if isinstance(collect_info, dict):
            collect_mask = collect_info.get("drafter_owner_route")
        if collect_mask is None and hasattr(worker_group, "_query_collect_info"):
            collect_mask = worker_group._query_collect_info("drafter_owner_route")
            if isinstance(collect_info, dict):
                collect_info["drafter_owner_route"] = collect_mask
        return collect_mask

    def _speco_dispatch_bucket_count(self) -> int | None:
        mapping = self._speco_owner_route_mapping()
        if not mapping:
            return None
        return max(int(dp_rank) for dp_rank in mapping) + 1

    def _speco_owner_bucket_count(self) -> int | None:
        mapping = self._speco_owner_route_mapping()
        if not mapping:
            return None
        collect_mask = self._speco_owner_route_collect_mask()
        if collect_mask and len(collect_mask) == len(mapping):
            owner_ranks = {
                int(dp_rank)
                for dp_rank, is_collect in zip(mapping, collect_mask, strict=False)
                if bool(is_collect)
            }
            if owner_ranks:
                return max(owner_ranks) + 1

        mapping_ranks = {int(dp_rank) for dp_rank in mapping}
        dispatch_bucket_count = max(mapping_ranks) + 1
        return max(dispatch_bucket_count - 1, 1)

    def _speco_get_drafter_target_lm_head_row_selection(self):
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("use_logits", False)):
            return None
        drafter_cfg = self._speco_drafter_config()
        algorithm = str(
            _get_nested(drafter_cfg, ("speculative_algorithm",), "") or ""
        ).upper()
        if algorithm == "DSPARK" and dspark_target_distribution_loss_enabled(
            training_cfg
        ):
            return None
        if not bool(training_cfg.get("target_lm_head_row_restricted_sync", True)):
            return None

        row_infos = (
            self._ray_get_if_needed(self.speco_get_drafter_target_lm_head_row_indices())
            or []
        )
        non_null_infos = [
            info
            for info in row_infos
            if isinstance(info, dict) and info.get("row_indices") is not None
        ]
        if not non_null_infos:
            return None
        source_vocab_sizes = {
            int(info.get("source_vocab_size"))
            for info in non_null_infos
            if info.get("source_vocab_size") is not None
        }
        if len(source_vocab_sizes) > 1:
            raise RuntimeError(
                "Inconsistent SPECO target lm_head source vocab sizes across replicas: "
                f"{sorted(source_vocab_sizes)}"
            )
        source_vocab_size = next(iter(source_vocab_sizes), None)
        row_tensors = []
        for info in non_null_infos:
            row_indices = info.get("row_indices")
            if torch.is_tensor(row_indices):
                rows = row_indices.detach().cpu().long().reshape(-1)
            elif isinstance(row_indices, (list, tuple)):
                rows = torch.tensor([int(idx) for idx in row_indices], dtype=torch.long)
            else:
                continue
            if rows.numel() > 0:
                row_tensors.append(rows)
        if not row_tensors:
            return None
        union_rows = (
            torch.unique(torch.cat(row_tensors), sorted=True)
            .to(dtype=torch.long)
            .contiguous()
        )
        selected_rows = int(union_rows.numel())
        if source_vocab_size is not None and selected_rows >= int(source_vocab_size):
            return None
        return {
            "row_indices": union_rows,
            "source_vocab_size": source_vocab_size,
            "selected_rows": selected_rows,
        }

    def _speco_actor_rollout_method(self, name: str):
        method = getattr(self.actor_rollout_wg, name, None)
        if not callable(method):
            raise RuntimeError(
                f"SPECO online drafter training requires actor_rollout_wg.{name}(). "
                "Attach a rollout worker implementing DraftWeightPublishMixin."
            )
        return method

    def _speco_build_drafter_target_lm_head_sync_args(
        self,
        payload: dict[str, torch.Tensor],
    ) -> tuple[Any, Any, int]:
        worker_group = self.drafter_wg
        if worker_group is None:
            return payload, self.global_steps, 1

        target_sync_mapping = None
        dispatch_info = getattr(worker_group, "_dispatch_info", None)
        if isinstance(dispatch_info, dict):
            target_sync_mapping = dispatch_info.get(_DRAFTER_TARGET_SYNC_MESH)
        if target_sync_mapping is None and hasattr(
            worker_group, "_query_dispatch_info"
        ):
            target_sync_mapping = worker_group._query_dispatch_info(
                _DRAFTER_TARGET_SYNC_MESH
            )
            if isinstance(dispatch_info, dict):
                dispatch_info[_DRAFTER_TARGET_SYNC_MESH] = target_sync_mapping
        if not target_sync_mapping:
            return payload, self.global_steps, 1

        target_sync_bucket_count = (
            max(int(dp_rank) for dp_rank in target_sync_mapping) + 1
        )
        payload_buckets = [payload for _ in range(target_sync_bucket_count)]
        global_step_buckets = [
            self.global_steps for _ in range(target_sync_bucket_count)
        ]
        return payload_buckets, global_step_buckets, target_sync_bucket_count

    def _speco_sync_target_lm_head_weight(self) -> dict[str, Any]:
        sync_started = time.perf_counter()
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("use_logits", False)):
            return {"drafter/target_lm_head_synced": 0}
        if not self._speco_should_train_drafter_this_step():
            return {"drafter/target_lm_head_synced": 0}

        row_selection = self._speco_get_drafter_target_lm_head_row_selection()
        row_indices = (
            row_selection.get("row_indices") if row_selection is not None else None
        )
        selected_rows = (
            int(row_selection.get("selected_rows", 0) or 0)
            if row_selection is not None
            else 0
        )
        source_vocab_size = (
            int(row_selection.get("source_vocab_size", 0) or 0)
            if row_selection is not None
            else 0
        )
        get_actor_lm_head_weight = self._speco_actor_rollout_method(
            "get_actor_lm_head_weight"
        )
        fetch_started = time.perf_counter()
        payloads = self._ray_get_if_needed(get_actor_lm_head_weight(row_indices)) or []
        fetch_elapsed = time.perf_counter() - fetch_started
        payload = self._first_non_null(payloads)
        if payload is None:
            return {
                "drafter/target_lm_head_synced": 0,
                "drafter/target_lm_head_selected_rows": selected_rows,
                "drafter/target_lm_head_source_vocab_size": source_vocab_size,
                "timing_s/drafter_sync_target_lm_head": time.perf_counter()
                - sync_started,
                "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            }

        payload_arg, global_step_arg, _ = (
            self._speco_build_drafter_target_lm_head_sync_args(payload)
        )
        apply_started = time.perf_counter()
        self._ray_get_if_needed(
            self.speco_sync_target_lm_head_weight(
                payload_arg, global_step=global_step_arg
            )
        )
        apply_elapsed = time.perf_counter() - apply_started
        export_strategy = (
            str(payload.get("export_strategy", "unknown"))
            if isinstance(payload, dict)
            else "unknown"
        )
        return {
            "drafter/target_lm_head_synced": 1,
            "drafter/target_lm_head_selected_rows": selected_rows,
            "drafter/target_lm_head_source_vocab_size": source_vocab_size,
            "drafter/target_lm_head_direct_sparse_export": int(
                export_strategy == "direct_sparse"
            ),
            "timing_s/drafter_sync_target_lm_head": time.perf_counter() - sync_started,
            "timing_s/drafter_sync_target_lm_head_fetch": fetch_elapsed,
            "timing_s/drafter_sync_target_lm_head_apply": apply_elapsed,
        }

    def _speco_train_drafter(self) -> tuple[bool, dict[str, Any]]:
        train_rpc_started = time.perf_counter()
        train_results = self._ray_get_if_needed(self.speco_train_drafter()) or []
        train_rpc_elapsed = time.perf_counter() - train_rpc_started
        if not isinstance(train_results, list):
            train_results = [train_results]

        normalized_results = []
        for result in train_results:
            if isinstance(result, dict):
                normalized_results.append(result)
            else:
                normalized_results.append(
                    {
                        "trained": bool(result),
                        "triggered": bool(result),
                        "attempted_steps": int(bool(result)),
                        "successful_steps": int(bool(result)),
                        "elapsed_sec": 0.0,
                        "reason": "legacy_bool_result",
                    }
                )

        trained = any(
            bool(result.get("trained", False)) for result in normalized_results
        )
        successful_steps_max = max(
            (int(result.get("successful_steps", 0)) for result in normalized_results),
            default=0,
        )

        metrics: dict[str, float | int] = {
            "drafter/trained": int(trained),
            "drafter/train_successful_steps_max": successful_steps_max,
            "drafter/train_no_trainable_batch": int(
                any(
                    result.get("reason") == "no_trainable_batch"
                    for result in normalized_results
                )
            ),
            "drafter/train_activation_failed": int(
                any(
                    result.get("reason") == "activation_failed"
                    for result in normalized_results
                )
            ),
        }
        for key in (
            "timing_s/drafter_prepare_batch",
            "timing_s/drafter_forward_loss",
            "timing_s/drafter_reduce_loss",
            "timing_s/drafter_backward",
            "timing_s/drafter_optimizer",
            "timing_s/drafter_publish_snapshot",
            "activation_elapsed_sec",
            "training_loop_elapsed_sec",
            "cleanup_elapsed_sec",
            "elapsed_sec",
        ):
            values = []
            for result in normalized_results:
                value = _speco_metric_float(result.get(key))
                if value is not None:
                    values.append(value)
            if values:
                metric_key = {
                    "activation_elapsed_sec": "timing_s/drafter_worker_activation",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                }.get(key, key)
                metrics[metric_key] = max(values)
        passthrough_prefixes = ("dflash/", "dspark/", "domino/", "drafter/")
        passthrough_keys = {
            key
            for result in normalized_results
            for key, value in result.items()
            if key.startswith(passthrough_prefixes)
            and not key.startswith("timing_s/")
            and _speco_metric_float(value) is not None
        }
        for key in sorted(passthrough_keys):
            values = [
                value
                for result in normalized_results
                if (value := _speco_metric_float(result.get(key))) is not None
            ]
            if not values:
                continue
            if key.startswith("drafter/"):
                metric_key = key
            else:
                metric_key = f"drafter/train/{key.replace('/', '_')}"
            if key in {
                "drafter/k3_dump_files",
                "drafter/k3_dump_tokens",
                "drafter/k3_dump_failures",
            }:
                metrics[metric_key] = sum(values)
            elif key.endswith(("accepted_k3_min", "rejected_k3_min")):
                metrics[metric_key] = min(values)
            else:
                metrics[metric_key] = max(values)
        metrics["timing_s/drafter_train_rpc"] = train_rpc_elapsed
        return trained, metrics

    def _speco_activate_drafter_training_model_before_fit(self) -> None:
        if not self.is_drafter_training_enabled(self.config):
            return
        activation_results = (
            self._ray_get_if_needed(self.speco_activate_drafter_training_model()) or []
        )
        if not isinstance(activation_results, list):
            activation_results = [activation_results]

        active_results = [
            result
            for result in activation_results
            if isinstance(result, dict)
            and result.get("reason") not in {"disabled", "not_in_training_group"}
        ]
        failed_results = [
            result
            for result in active_results
            if not bool(result.get("activated", False))
        ]
        if failed_results:
            raise RuntimeError(
                "SPECO drafter trainer activation failed before RL fit: "
                f"{failed_results[:3]}"
            )

    def _speco_should_publish_drafter_weights(self, drafter_trained: bool) -> bool:
        if not drafter_trained:
            return False
        training_cfg = self._speco_drafter_training_config()
        publish_interval = int(training_cfg.get("publish_interval_steps", 0) or 0)
        return publish_interval <= 0 or self.global_steps % publish_interval == 0

    def _speco_wait_pending_drafter_publish(self) -> int:
        if not self._pending_drafter_publish_refs:
            return 0
        pending_refs = self._pending_drafter_publish_refs
        self._pending_drafter_publish_refs = None
        self._ray_get_if_needed(pending_refs)
        return len(pending_refs) if isinstance(pending_refs, (list, tuple)) else 1

    def _speco_get_published_drafter_weights(self):
        published = self._ray_get_if_needed(self.speco_maybe_publish()) or []
        return self._first_non_null(published)

    def _speco_candidate_gate_enabled(self) -> bool:
        return bool(
            self._speco_drafter_training_config().get(
                "candidate_gate_enabled", False
            )
        )

    def _speco_candidate_alphas(self) -> tuple[float, ...]:
        configured = self._speco_drafter_training_config().get(
            "candidate_gate_alphas", [0.0, 0.25, 0.5, 1.0]
        )
        if isinstance(configured, str):
            configured = configured.split(",")
        try:
            values = [float(value) for value in configured]
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_gate_alphas must be a list of numbers") from exc
        values.append(0.0)
        normalized = sorted(
            {value for value in values if math.isfinite(value) and 0.0 <= value <= 1.0}
        )
        if not normalized:
            raise ValueError("candidate_gate_alphas must contain a value in [0, 1]")
        return tuple(normalized)

    def _speco_publish_candidate_alpha(self, update_step: int, alpha: float) -> float:
        fetch_started = time.perf_counter()
        payloads = self._ray_get_if_needed(
            self.speco_get_candidate_weights(update_step, alpha)
        ) or []
        published = self._first_non_null(payloads)
        if published is None:
            raise RuntimeError(
                f"Missing drafter candidate state for step={update_step}, alpha={alpha}"
            )
        payload = resolve_drafter_publish_payload(published)
        update_draft_weights = self._speco_actor_rollout_method(
            "update_draft_weights"
        )
        self._ray_get_if_needed(
            update_draft_weights(payload, global_steps=update_step)
        )
        return time.perf_counter() - fetch_started

    def _speco_finalize_candidate(self, update_step: int, alpha: float) -> None:
        results = self._ray_get_if_needed(
            self.speco_finalize_candidate_update(update_step, alpha)
        )
        flattened = self._speco_flatten_checkpoint_results(results)
        failed = [
            item
            for item in flattened
            if item.get("reason") != "not_in_training_group"
            and not bool(item.get("finalized", False))
        ]
        if failed:
            raise RuntimeError(
                f"Failed to finalize drafter candidate step={update_step}: {failed}"
            )

    @staticmethod
    def _speco_candidate_response_tokens(output: Any) -> int:
        batch = getattr(output, "batch", None)
        if batch is None:
            return 0
        for key in ("response_mask", "loss_mask"):
            try:
                value = batch.get(key)
            except (AttributeError, KeyError):
                value = None
            if torch.is_tensor(value):
                return int(value.detach().sum().cpu().item())
        try:
            responses = batch.get("responses")
        except (AttributeError, KeyError):
            responses = None
        return int(responses.numel()) if torch.is_tensor(responses) else 0

    def _speco_candidate_gate_batch(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        source = None
        source_arg = None
        for index, value in enumerate(args):
            if isinstance(value, DataProto):
                source = value
                source_arg = index
                break
        source_key = None
        if source is None:
            for key in ("batch", "prompts", "data", "input_batch"):
                if isinstance(kwargs.get(key), DataProto):
                    source = kwargs[key]
                    source_key = key
                    break
        if source is None:
            raise RuntimeError(
                "Cannot locate DataProto input for candidate rollout gate"
            )
        configured_samples = int(
            self._speco_drafter_training_config().get(
                "candidate_gate_samples", 16
            )
            or 16
        )
        cohort = self._speco_candidate_gate_cohort
        if cohort is None:
            count = min(max(configured_samples, 1), len(source))
            cohort = source[:count]
            self._speco_candidate_gate_cohort = cohort
        gate_batch = cohort[:]
        gate_batch.meta_info = dict(getattr(source, "meta_info", {}) or {})
        gate_batch.meta_info.update(
            {
                "validate": True,
                "speco_candidate_gate": True,
                "speco_candidate_gate_max_tokens": int(
                    self._speco_drafter_training_config().get(
                        "candidate_gate_max_tokens", 256
                    )
                    or 256
                ),
            }
        )
        gate_args = list(args)
        gate_kwargs = dict(kwargs)
        if source_arg is not None:
            gate_args[source_arg] = gate_batch
        else:
            gate_kwargs[source_key] = gate_batch
        return tuple(gate_args), gate_kwargs

    def _speco_measure_candidate_rollout(
        self, generate: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, float]]:
        started = time.perf_counter()
        output = generate(*args, **kwargs)
        elapsed = time.perf_counter() - started
        metrics = _speco_vllm_spec_decode_metrics_from_batch(output)
        acceptance = _speco_metric_float(
            metrics.get(SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC)
        )
        if acceptance is None:
            raise RuntimeError("Candidate rollout produced no speculative decode stats")
        response_tokens = self._speco_candidate_response_tokens(output)
        return output, {
            "acceptance": acceptance,
            "elapsed_sec": elapsed,
            "response_tokens": float(response_tokens),
            "ms_per_token": (
                elapsed * 1000.0 / response_tokens
                if response_tokens > 0
                else float("inf")
            ),
        }

    def _speco_candidate_precheck(
        self, train_metrics: Mapping[str, Any]
    ) -> tuple[bool, str, dict[str, float | int]]:
        cfg = self._speco_drafter_training_config()
        checks = (
            (
                "fixed_loss",
                "drafter/fixed_cohort/delta_loss",
                float(cfg.get("candidate_max_fixed_loss_delta", 0.0) or 0.0),
            ),
            (
                "rejected_k3_clamp",
                "drafter/train/dspark_rejected_k3_clamp_ratio",
                float(cfg.get("candidate_max_rejected_k3_clamp_ratio", 0.5) or 0.5),
            ),
            (
                "rejected_k3_median",
                "drafter/train/dspark_rejected_k3_median",
                float(cfg.get("candidate_max_rejected_k3_median", 9.5) or 9.5),
            ),
        )
        metrics: dict[str, float | int] = {}
        for reason, key, maximum in checks:
            value = _speco_metric_float(train_metrics.get(key))
            if value is not None:
                metrics[f"drafter/candidate/precheck_{reason}"] = value
                if value > maximum:
                    return False, reason, metrics

        grad_norm = _speco_metric_float(train_metrics.get("drafter/train_grad_norm"))
        if grad_norm is not None:
            history = self._speco_candidate_grad_norm_history
            fixed_max = float(cfg.get("candidate_max_grad_norm", 50.0) or 50.0)
            rolling_multiplier = float(
                cfg.get("candidate_grad_norm_rolling_multiplier", 5.0) or 5.0
            )
            rolling_median = (
                float(torch.tensor(history[-5:]).median().item()) if history else None
            )
            metrics["drafter/candidate/precheck_grad_norm"] = grad_norm
            if rolling_median is not None:
                metrics["drafter/candidate/grad_norm_rolling_median"] = rolling_median
            history.append(grad_norm)
            del history[:-20]
            if grad_norm > fixed_max or (
                rolling_median is not None
                and rolling_median > 0.0
                and grad_norm > rolling_median * rolling_multiplier
            ):
                return False, "grad_norm", metrics
        return True, "ok", metrics

    def _speco_stage_candidate_update(
        self, drafter_trained: bool, train_metrics: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self._speco_should_publish_drafter_weights(drafter_trained):
            return {"drafter/candidate/staged": 0}
        baseline = _speco_metric_float(
            self._speco_current_step_rollout_metrics().get(
                SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC
            )
        )
        passed, reason, metrics = self._speco_candidate_precheck(train_metrics)
        metrics.update(
            {
                "drafter/candidate/staged": int(passed),
                "drafter/candidate/precheck_passed": int(passed),
            }
        )
        if not passed:
            self._speco_finalize_candidate(int(self.global_steps), 0.0)
            metrics["drafter/candidate/rejected_pre_publish"] = 1
            logger.warning(
                "Rejected drafter candidate before rollout: step=%s reason=%s",
                self.global_steps,
                reason,
            )
            return metrics
        self._speco_pending_candidate_update = {
            "update_step": int(self.global_steps),
            "full_baseline": baseline,
        }
        return metrics

    def _speco_select_candidate_with_rollout(
        self,
        generate: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        pending = self._speco_pending_candidate_update
        update_step = int(pending["update_step"])
        gate_args, gate_kwargs = self._speco_candidate_gate_batch(args, kwargs)
        results = {}
        metrics: dict[str, Any] = {"drafter/candidate/gate_ran": 1}
        for alpha in self._speco_candidate_alphas():
            publish_elapsed = self._speco_publish_candidate_alpha(update_step, alpha)
            _, result = self._speco_measure_candidate_rollout(
                generate, gate_args, gate_kwargs
            )
            results[alpha] = result
            label = str(alpha).replace(".", "_")
            metrics[f"drafter/candidate/alpha_{label}_acceptance"] = result[
                "acceptance"
            ]
            metrics[f"drafter/candidate/alpha_{label}_ms_per_token"] = result[
                "ms_per_token"
            ]
            metrics[f"timing_s/drafter_candidate_alpha_{label}_publish"] = (
                publish_elapsed
            )

        baseline = results[0.0]
        min_delta = float(
            self._speco_drafter_training_config().get(
                "candidate_gate_min_acceptance_delta", 0.0
            )
            or 0.0
        )
        max_latency_regression = float(
            self._speco_drafter_training_config().get(
                "candidate_gate_max_ms_per_token_regression", 0.15
            )
            or 0.15
        )
        selected = 0.0
        selected_result = baseline
        for alpha in self._speco_candidate_alphas():
            if alpha == 0.0:
                continue
            candidate = results[alpha]
            acceptance_ok = (
                candidate["acceptance"]
                >= baseline["acceptance"] + min_delta
            )
            latency_ok = (
                not math.isfinite(baseline["ms_per_token"])
                or candidate["ms_per_token"]
                <= baseline["ms_per_token"] * (1.0 + max_latency_regression)
            )
            if not acceptance_ok or not latency_ok:
                continue
            if candidate["acceptance"] > selected_result["acceptance"] + 1e-12 or (
                abs(candidate["acceptance"] - selected_result["acceptance"]) <= 1e-12
                and candidate["ms_per_token"] < selected_result["ms_per_token"]
            ):
                selected = alpha
                selected_result = candidate
        if selected != self._speco_candidate_alphas()[-1]:
            self._speco_publish_candidate_alpha(update_step, selected)
        metrics["drafter/candidate/selected_alpha"] = selected
        metrics["drafter/candidate/selected_gate_acceptance"] = selected_result[
            "acceptance"
        ]
        metrics["drafter/candidate/gate_acceptance_delta"] = (
            selected_result["acceptance"] - baseline["acceptance"]
        )
        return selected, metrics

    def _speco_best_candidate_manifest_path(self) -> str | None:
        checkpoint_root = self._speco_ensure_drafter_checkpoint_path()
        if checkpoint_root is None:
            return None
        return os.path.join(checkpoint_root, "best_candidate.json")

    def _speco_load_best_candidate_score(self) -> float:
        if self._speco_best_candidate_score is not None:
            return float(self._speco_best_candidate_score)
        score = float("-inf")
        path = self._speco_best_candidate_manifest_path()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as stream:
                    score = float(json.load(stream).get("mean_acceptance_length"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Ignoring invalid best drafter manifest: %s", path)
        self._speco_best_candidate_score = score
        return score

    def _speco_maybe_save_best_candidate(
        self, *, update_step: int, alpha: float, observed: float
    ) -> dict[str, Any]:
        if not bool(
            self._speco_drafter_training_config().get(
                "save_best_drafter_checkpoint", True
            )
        ):
            return {"drafter/candidate/best_saved": 0}
        if not self._speco_should_save_drafter_checkpoint():
            return {"drafter/candidate/best_saved": 0}
        previous = self._speco_load_best_candidate_score()
        min_delta = float(
            self._speco_drafter_training_config().get(
                "best_checkpoint_min_acceptance_delta", 0.0
            )
            or 0.0
        )
        if observed <= previous + min_delta:
            return {"drafter/candidate/best_saved": 0}
        save_started = time.perf_counter()
        self._speco_save_drafter_checkpoint(wait=True, step=update_step)
        save_elapsed = time.perf_counter() - save_started
        path = self._speco_best_candidate_manifest_path()
        if path is None:
            return {"drafter/candidate/best_saved": 0}
        record = {
            "schema_version": 1,
            "step": int(update_step),
            "alpha": float(alpha),
            "mean_acceptance_length": float(observed),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        marker = os.path.join(os.path.dirname(path), "best_checkpointed_iteration.txt")
        temporary_marker = f"{marker}.tmp.{os.getpid()}"
        with open(temporary_marker, "w", encoding="utf-8") as stream:
            stream.write(f"{int(update_step)}\n")
        os.replace(temporary_marker, marker)
        self._speco_best_candidate_score = float(observed)
        return {
            "drafter/candidate/best_saved": 1,
            "drafter/candidate/best_score": float(observed),
            "drafter/candidate/best_step": int(update_step),
            "timing_s/drafter_best_checkpoint": save_elapsed,
        }

    def _speco_dual_metric_best_enabled(self) -> bool:
        training_cfg = self._speco_drafter_training_config()
        return bool(
            training_cfg.get("dual_metric_best_checkpoint_enabled", False)
        ) and bool(training_cfg.get("save_best_drafter_checkpoint", True)) and bool(
            self._speco_drafter_checkpoint_save_config_enabled()
        )

    def _speco_dual_best_manifest_path(self, kind: str) -> str | None:
        if kind not in {"acceptance", "throughput"}:
            raise ValueError(f"Unsupported dual-best metric: {kind}")
        checkpoint_root = self._speco_ensure_drafter_checkpoint_path()
        if checkpoint_root is None:
            return None
        return os.path.join(checkpoint_root, f"best_{kind}.json")

    def _speco_load_dual_best_score(self, kind: str) -> float:
        path = self._speco_dual_best_manifest_path(kind)
        if path is None or not os.path.isfile(path):
            return float("-inf")
        score_key = (
            "mean_acceptance_length"
            if kind == "acceptance"
            else "response_tokens_per_second"
        )
        try:
            with open(path, encoding="utf-8") as stream:
                score = float(json.load(stream).get(score_key))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid dual-best drafter manifest: %s", path)
            return float("-inf")
        return score if math.isfinite(score) else float("-inf")

    def _speco_has_complete_drafter_checkpoint(self, step: int) -> bool:
        checkpoint_root = self._speco_ensure_drafter_checkpoint_path()
        if checkpoint_root is None:
            return False
        checkpoint_path = os.path.join(checkpoint_root, f"draft_step_{int(step)}")
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        try:
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata_step = int(metadata.get("step"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if metadata.get("complete") is not True or metadata_step != int(step):
            return False
        try:
            entries = os.listdir(checkpoint_path)
        except OSError:
            return False
        for name in entries:
            if not name.endswith((".safetensors", ".bin", ".index.json")):
                continue
            try:
                if os.path.getsize(os.path.join(checkpoint_path, name)) > 0:
                    return True
            except OSError:
                continue
        return False

    def _speco_write_dual_best_record(
        self, kind: str, record: Mapping[str, Any]
    ) -> None:
        path = self._speco_dual_best_manifest_path(kind)
        if path is None:
            raise RuntimeError("Dual-best drafter checkpoint path is unavailable")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(dict(record), stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        marker = os.path.join(
            os.path.dirname(path), f"best_{kind}_checkpointed_iteration.txt"
        )
        temporary_marker = f"{marker}.tmp.{os.getpid()}"
        with open(temporary_marker, "w", encoding="utf-8") as stream:
            stream.write(f"{int(record['step'])}\n")
        os.replace(temporary_marker, marker)

    def _speco_maybe_save_dual_best_from_rollout(self) -> dict[str, Any]:
        if not self._speco_dual_metric_best_enabled():
            self._speco_pending_dual_best_observation = None
            return {}
        pending = getattr(self, "_speco_pending_dual_best_observation", None)
        if not isinstance(pending, Mapping):
            return {"drafter/dual_best/checked": 0}
        current_step = int(getattr(self, "global_steps", 0) or 0)
        update_step = int(pending.get("update_step", current_step) or current_step)
        if current_step <= update_step:
            return {"drafter/dual_best/checked": 0}

        quality = self._speco_current_step_rollout_quality_metrics()
        acceptance = _speco_metric_float(quality.get("mean_acceptance_length"))
        throughput = _speco_metric_float(
            quality.get("response_tokens_per_second")
        )
        if acceptance is None or throughput is None or throughput <= 0.0:
            return {
                "drafter/dual_best/checked": 0,
                "drafter/dual_best/waiting_for_complete_rollout_metrics": 1,
            }
        self._speco_pending_dual_best_observation = None

        training_cfg = self._speco_drafter_training_config()
        acceptance_delta = float(
            training_cfg.get("best_checkpoint_min_acceptance_delta", 0.0) or 0.0
        )
        throughput_relative_delta = float(
            training_cfg.get(
                "best_checkpoint_min_throughput_relative_delta", 0.0
            )
            or 0.0
        )
        previous_acceptance = self._speco_load_dual_best_score("acceptance")
        previous_throughput = self._speco_load_dual_best_score("throughput")
        acceptance_improved = acceptance > previous_acceptance + acceptance_delta
        throughput_improved = (
            not math.isfinite(previous_throughput)
            or throughput
            > previous_throughput * (1.0 + throughput_relative_delta)
        )
        metrics: dict[str, Any] = {
            "drafter/dual_best/checked": 1,
            "drafter/dual_best/update_step": update_step,
            "drafter/dual_best/observed_step": current_step,
            "drafter/dual_best/mean_acceptance_length": acceptance,
            "drafter/dual_best/response_tokens_per_second": throughput,
            "drafter/dual_best/acceptance_improved": int(acceptance_improved),
            "drafter/dual_best/throughput_improved": int(throughput_improved),
            "drafter/dual_best/acceptance_saved": 0,
            "drafter/dual_best/throughput_saved": 0,
        }
        if not acceptance_improved and not throughput_improved:
            return metrics

        save_started = time.perf_counter()
        include_optimizer = bool(
            training_cfg.get(
                "dual_metric_best_checkpoint_include_optimizer", False
            )
        )
        if not self._speco_has_complete_drafter_checkpoint(update_step):
            self._speco_save_drafter_checkpoint(
                wait=True,
                step=update_step,
                include_optimizer=include_optimizer,
            )
        save_elapsed = time.perf_counter() - save_started
        common_record = {
            "schema_version": 1,
            "step": update_step,
            "rollout_step": current_step,
            "actor_snapshot_step": max(current_step - 1, 0),
            "mean_acceptance_length": acceptance,
            "response_tokens_per_second": throughput,
            "response_tokens": int(quality["response_tokens"]),
            "generation_time_s": float(quality["generation_time_s"]),
            "verify_steps": int(quality["verify_steps"]),
            "accepted_tokens": int(quality["accepted_tokens"]),
            "checkpoint_includes_optimizer": include_optimizer,
            "requires_final_actor_reevaluation": True,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if acceptance_improved:
            self._speco_write_dual_best_record(
                "acceptance", {**common_record, "selection_metric": "acceptance"}
            )
            metrics["drafter/dual_best/acceptance_saved"] = 1
            metrics["drafter/dual_best/best_acceptance"] = acceptance
        if throughput_improved:
            self._speco_write_dual_best_record(
                "throughput", {**common_record, "selection_metric": "throughput"}
            )
            metrics["drafter/dual_best/throughput_saved"] = 1
            metrics["drafter/dual_best/best_throughput"] = throughput
        metrics["timing_s/drafter_dual_best_checkpoint"] = save_elapsed
        return metrics

    def _speco_commit_or_rollback_candidate(
        self,
        generate: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        selected_alpha: float,
        metrics: dict[str, Any],
    ) -> Any:
        pending = self._speco_pending_candidate_update
        update_step = int(pending["update_step"])
        observed = _speco_metric_float(
            _speco_vllm_spec_decode_metrics_from_batch(output).get(
                SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC
            )
        )
        baseline = _speco_metric_float(pending.get("full_baseline"))
        threshold = self._speco_post_update_acceptance_drop_abort_threshold()
        drop = (
            baseline - observed
            if baseline is not None and observed is not None
            else None
        )
        metrics["drafter/candidate/full_rollout_stats_present"] = int(
            observed is not None
        )
        if baseline is not None:
            metrics["drafter/candidate/full_baseline_acceptance"] = baseline
        if observed is not None:
            metrics["drafter/candidate/full_observed_acceptance"] = observed
        if drop is not None:
            metrics["drafter/candidate/full_acceptance_drop"] = drop

        should_rollback = selected_alpha > 0.0 and (
            observed is None
            or (
                threshold is not None
                and drop is not None
                and drop >= threshold
            )
        )
        if should_rollback:
            self._speco_publish_candidate_alpha(update_step, 0.0)
            self._speco_finalize_candidate(update_step, 0.0)
            metrics["drafter/candidate/rolled_back"] = 1
            metrics["drafter/candidate/committed"] = 0
            if observed is not None and baseline is not None and drop is not None:
                self._speco_record_post_update_acceptance_incident(
                    update_step=update_step,
                    observed_step=int(self.global_steps),
                    baseline=baseline,
                    observed=observed,
                    drop=drop,
                    threshold=float(threshold),
                    action="rolled_back_and_regenerated",
                )
            logger.warning(
                "Rolling back drafter candidate and regenerating actor batch: "
                "update_step=%s selected_alpha=%s baseline=%s observed=%s",
                update_step,
                selected_alpha,
                baseline,
                observed,
            )
            output = generate(*args, **kwargs)
        else:
            self._speco_finalize_candidate(update_step, selected_alpha)
            metrics["drafter/candidate/rolled_back"] = 0
            metrics["drafter/candidate/committed"] = int(selected_alpha > 0.0)
            if selected_alpha > 0.0 and observed is not None:
                fixed_gate_score = _speco_metric_float(
                    metrics.get("drafter/candidate/selected_gate_acceptance")
                )
                self._speco_pending_best_candidate_save = {
                    "update_step": update_step,
                    "alpha": selected_alpha,
                    "observed": (
                        fixed_gate_score
                        if fixed_gate_score is not None
                        else observed
                    ),
                }
        self._speco_pending_candidate_update = None
        return output

    def _speco_publish_drafter_weights(self, drafter_trained: bool) -> dict[str, Any]:
        if not self._speco_should_publish_drafter_weights(drafter_trained):
            return {"drafter/publish_attempted": 0, "drafter/published": 0}

        wait_started = time.perf_counter()
        self._speco_wait_pending_drafter_publish()
        wait_elapsed = time.perf_counter() - wait_started
        fetch_started = time.perf_counter()
        published = self._speco_get_published_drafter_weights()
        fetch_elapsed = time.perf_counter() - fetch_started
        metrics = {
            "drafter/publish_attempted": 1,
            "timing_s/drafter_publish_wait_pending": wait_elapsed,
            "timing_s/drafter_publish_fetch_snapshot": fetch_elapsed,
        }
        if published is None:
            metrics["drafter/published"] = 0
            return metrics

        payload = resolve_drafter_publish_payload(published)
        training_cfg = self._speco_drafter_training_config()
        publish_async = bool(training_cfg.get("publish_async", False))
        method_name = (
            "update_draft_weights_async" if publish_async else "update_draft_weights"
        )
        update_draft_weights = self._speco_actor_rollout_method(method_name)
        update_started = time.perf_counter()
        update_result = update_draft_weights(payload, global_steps=self.global_steps)
        if publish_async:
            self._pending_drafter_publish_refs = update_result
        else:
            self._ray_get_if_needed(update_result)
        update_elapsed = time.perf_counter() - update_started

        metrics.update(
            {
                "drafter/published": 1,
                "timing_s/drafter_publish_update_weights": update_elapsed,
            }
        )
        return metrics

    def _speco_update_output_metrics(self, output: Any, metrics: dict[str, Any]):
        if not metrics:
            return output
        meta_info = getattr(output, "meta_info", None)
        if isinstance(meta_info, dict):
            output_metrics = meta_info.setdefault("metrics", {})
            output_metrics.update(metrics)
            drafter_elapsed = _speco_metric_float(
                output_metrics.get("timing_s/drafter")
            )
            update_actor_elapsed = _speco_metric_float(
                output_metrics.get("timing_s/update_actor")
            )
            if drafter_elapsed is not None and update_actor_elapsed is not None:
                adjusted_update_actor = max(0.0, update_actor_elapsed - drafter_elapsed)
                update_actor_per_token = _speco_metric_float(
                    output_metrics.get("timing_per_token_ms/update_actor")
                )
                if update_actor_per_token is not None:
                    output_metrics["timing_per_token_ms/update_actor"] = (
                        update_actor_per_token
                        * adjusted_update_actor
                        / update_actor_elapsed
                        if update_actor_elapsed > 0
                        else 0.0
                    )
                output_metrics["timing_s/update_actor"] = adjusted_update_actor
                output_metrics[_SPECO_DRAFTER_TIMING_DEDUCTED_KEY] = True
        return output

    def _speco_rollout_generation_target(self):
        for attr_name in ("async_rollout_manager", "actor_rollout_wg"):
            target = getattr(self, attr_name, None)
            if target is not None and callable(
                getattr(target, "generate_sequences", None)
            ):
                return target
        raise RuntimeError(
            "SPECO online drafter training requires a rollout generation object "
            "with generate_sequences(), but neither async_rollout_manager nor "
            "actor_rollout_wg exposes it."
        )

    def _speco_store_rollout_metrics(self, output: Any, elapsed_sec: float) -> None:
        current_step = getattr(self, "global_steps", None)
        if getattr(self, "_speco_last_rollout_metrics_step", None) != current_step:
            self._speco_last_rollout_metrics = {}
            self._speco_last_rollout_metrics_step = current_step
            self._speco_last_rollout_generation_elapsed_sec = 0.0
            self._speco_last_rollout_response_tokens = 0
        self._speco_last_rollout_metrics = _speco_merge_vllm_spec_decode_stats(
            getattr(self, "_speco_last_rollout_metrics", None),
            _speco_vllm_spec_decode_stats_from_batch(output),
        )
        self._speco_last_rollout_generation_elapsed_sec += max(
            float(elapsed_sec), 0.0
        )
        self._speco_last_rollout_response_tokens += self._speco_candidate_response_tokens(
            output
        )

    def _speco_store_validation_rollout_metrics(
        self, output: Any, elapsed_sec: float
    ) -> None:
        self._speco_validation_rollout_metrics = _speco_merge_vllm_spec_decode_stats(
            getattr(self, "_speco_validation_rollout_metrics", None),
            _speco_vllm_spec_decode_stats_from_batch(output),
        )
        self._speco_validation_generation_elapsed_sec = float(
            getattr(self, "_speco_validation_generation_elapsed_sec", 0.0) or 0.0
        ) + max(float(elapsed_sec), 0.0)
        self._speco_validation_response_tokens = int(
            getattr(self, "_speco_validation_response_tokens", 0) or 0
        ) + self._speco_candidate_response_tokens(output)
        self._speco_collect_validation_diagnostics(output)

    def _speco_collect_validation_diagnostics(self, output: Any) -> None:
        """Collect optional raw token hashes and per-draft-position acceptance."""

        if not os.getenv(_SPECO_VALIDATION_DIAGNOSTICS_DIR_ENV, "").strip():
            return
        batch = getattr(output, "batch", None)
        if batch is None:
            return
        try:
            responses = batch.get("responses")
            response_mask = batch.get("response_mask")
        except (AttributeError, KeyError):
            return
        if not torch.is_tensor(responses):
            return
        if not torch.is_tensor(response_mask):
            response_mask = torch.ones_like(responses, dtype=torch.bool)

        non_tensor_batch = getattr(output, "non_tensor_batch", {}) or {}
        traces = non_tensor_batch.get(SPECO_VLLM_SPECULATIVE_TRACE_KEY)
        request_ids = non_tensor_batch.get("request_id")
        records = getattr(self, "_speco_validation_token_records", None)
        if not isinstance(records, list):
            records = []
            self._speco_validation_token_records = records
        position_attempts = getattr(
            self, "_speco_validation_position_attempts", None
        )
        position_accepted = getattr(
            self, "_speco_validation_position_accepted", None
        )
        if not isinstance(position_attempts, list):
            position_attempts = []
            self._speco_validation_position_attempts = position_attempts
        if not isinstance(position_accepted, list):
            position_accepted = []
            self._speco_validation_position_accepted = position_accepted

        for row_index in range(int(responses.size(0))):
            mask = response_mask[row_index].reshape(-1).bool()
            token_ids = (
                responses[row_index].reshape(-1)[mask].detach().cpu().tolist()
            )
            token_ids = [int(token_id) for token_id in token_ids]
            token_payload = json.dumps(token_ids, separators=(",", ":")).encode()
            trace = self._speco_sequence_item(traces, row_index)
            request_id = self._speco_sequence_item(request_ids, row_index)
            records.append(
                {
                    "sample_index": len(records),
                    "request_id": str(request_id) if request_id is not None else None,
                    "response_token_count": len(token_ids),
                    "response_token_sha256": hashlib.sha256(token_payload).hexdigest(),
                    "response_token_ids": token_ids,
                    "speculative_trace_event_count": (
                        len(trace) if isinstance(trace, (list, tuple)) else 0
                    ),
                }
            )
            if not isinstance(trace, (list, tuple)):
                continue
            for event in trace:
                if not isinstance(event, dict):
                    continue
                try:
                    draft_count = max(int(event.get("draft_count", 0) or 0), 0)
                    accepted_count = min(
                        max(int(event.get("accepted_count", 0) or 0), 0),
                        draft_count,
                    )
                except (TypeError, ValueError):
                    continue
                if draft_count > len(position_attempts):
                    missing = draft_count - len(position_attempts)
                    position_attempts.extend([0] * missing)
                    position_accepted.extend([0] * missing)
                for position in range(draft_count):
                    position_attempts[position] += 1
                    if accepted_count > position:
                        position_accepted[position] += 1

    def _speco_write_validation_diagnostics(self) -> None:
        output_dir = os.getenv(_SPECO_VALIDATION_DIAGNOSTICS_DIR_ENV, "").strip()
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        step = int(getattr(self, "global_steps", 0) or 0)
        records = getattr(self, "_speco_validation_token_records", None) or []
        token_path = os.path.join(output_dir, f"token_ids.step-{step:06d}.jsonl")
        token_tmp = f"{token_path}.tmp.{os.getpid()}"
        ordered_digest = hashlib.sha256()
        with open(token_tmp, "w", encoding="utf-8") as stream:
            for record in records:
                digest = str(record.get("response_token_sha256", ""))
                ordered_digest.update(f"{len(digest)}:{digest}".encode())
                stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
                stream.write("\n")
        os.replace(token_tmp, token_path)

        attempts = getattr(self, "_speco_validation_position_attempts", None) or []
        accepted = getattr(self, "_speco_validation_position_accepted", None) or []
        positions = [
            {
                "position": index + 1,
                "attempted": int(attempted),
                "accepted": int(accepted[index]),
                "acceptance_rate": (
                    float(accepted[index]) / float(attempted) if attempted else None
                ),
            }
            for index, attempted in enumerate(attempts)
        ]
        summary = {
            "schema_version": 1,
            "global_step": step,
            "sample_count": len(records),
            "ordered_response_token_sha256": ordered_digest.hexdigest(),
            "token_ids_path": os.path.realpath(token_path),
            "trace_event_count": sum(
                int(record.get("speculative_trace_event_count", 0) or 0)
                for record in records
            ),
            "acceptance_by_position": positions,
        }
        summary_path = os.path.join(
            output_dir, f"spec_decode.step-{step:06d}.json"
        )
        summary_tmp = f"{summary_path}.tmp.{os.getpid()}"
        with open(summary_tmp, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(summary_tmp, summary_path)

    def _speco_validation_spec_decode_metrics(self) -> dict[str, float]:
        stats = getattr(self, "_speco_validation_rollout_metrics", None) or {}
        drafts = float(stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0)
        accepted_tokens = float(
            stats.get(_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY, 0.0) or 0.0
        )
        if drafts <= 0.0:
            return {}
        elapsed_sec = float(
            getattr(self, "_speco_validation_generation_elapsed_sec", 0.0) or 0.0
        )
        response_tokens = float(
            getattr(self, "_speco_validation_response_tokens", 0) or 0
        )
        prefix = _SPECO_VALIDATION_SPEC_DECODE_PREFIX
        metrics = {
            f"{prefix}/mean_acceptance_length": 1.0 + accepted_tokens / drafts,
            f"{prefix}/verify_steps": drafts,
            f"{prefix}/accepted_tokens": accepted_tokens,
            f"{prefix}/generation_time_s": elapsed_sec,
            f"{prefix}/response_tokens": response_tokens,
            f"{prefix}/response_tokens_per_second": (
                response_tokens / elapsed_sec if elapsed_sec > 0.0 else 0.0
            ),
        }
        attempts = getattr(self, "_speco_validation_position_attempts", None) or []
        accepted = getattr(self, "_speco_validation_position_accepted", None) or []
        for index, attempted in enumerate(attempts):
            metrics[f"{prefix}/position_{index + 1}_attempted"] = float(attempted)
            metrics[f"{prefix}/position_{index + 1}_accepted"] = float(
                accepted[index]
            )
            metrics[f"{prefix}/position_{index + 1}_acceptance_rate"] = (
                float(accepted[index]) / float(attempted) if attempted else 0.0
            )
        return metrics

    def _validate(self, merged: bool = False):
        self._speco_validation_rollout_metrics = {}
        self._speco_validation_generation_elapsed_sec = 0.0
        self._speco_validation_response_tokens = 0
        self._speco_validation_token_records = []
        self._speco_validation_position_attempts = []
        self._speco_validation_position_accepted = []
        metrics = super()._validate(merged=merged)
        if merged or not isinstance(metrics, dict):
            return metrics
        self._speco_write_validation_diagnostics()
        return {**metrics, **self._speco_validation_spec_decode_metrics()}

    def _speco_current_step_rollout_metrics(self) -> dict[str, float]:
        if getattr(self, "_speco_last_rollout_metrics_step", None) != getattr(
            self, "global_steps", None
        ):
            return {}
        return _speco_vllm_spec_decode_metrics_from_stats(
            getattr(self, "_speco_last_rollout_metrics", None) or {}
        )

    def _speco_current_step_rollout_quality_metrics(self) -> dict[str, float]:
        if getattr(self, "_speco_last_rollout_metrics_step", None) != getattr(
            self, "global_steps", None
        ):
            return {}
        stats = getattr(self, "_speco_last_rollout_metrics", None) or {}
        verify_steps = float(
            stats.get(_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY, 0.0) or 0.0
        )
        accepted_tokens = float(
            stats.get(_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY, 0.0) or 0.0
        )
        elapsed_sec = float(
            getattr(self, "_speco_last_rollout_generation_elapsed_sec", 0.0)
            or 0.0
        )
        response_tokens = float(
            getattr(self, "_speco_last_rollout_response_tokens", 0) or 0
        )
        if verify_steps <= 0.0 or elapsed_sec <= 0.0 or response_tokens <= 0.0:
            return {}
        return {
            "mean_acceptance_length": 1.0 + accepted_tokens / verify_steps,
            "verify_steps": verify_steps,
            "accepted_tokens": accepted_tokens,
            "generation_time_s": elapsed_sec,
            "response_tokens": response_tokens,
            "response_tokens_per_second": response_tokens / elapsed_sec,
        }

    def _speco_post_update_acceptance_drop_abort_threshold(self) -> float | None:
        """Return the configured absolute drop in mean acceptance length to abort on."""
        raw_threshold = self._speco_drafter_training_config().get(
            "post_update_mean_acceptance_length_drop_abort_threshold", None
        )
        if raw_threshold in (None, ""):
            return None
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "post_update_mean_acceptance_length_drop_abort_threshold must be "
                "a non-negative number or null"
            ) from exc
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "post_update_mean_acceptance_length_drop_abort_threshold must be "
                "a finite non-negative number or null"
            )
        return threshold

    def _speco_acceptance_incident_log_path(self) -> str:
        """Resolve the append-only post-update acceptance incident ledger."""
        training_cfg = self._speco_drafter_training_config()
        configured_path = training_cfg.get(
            "post_update_acceptance_incident_log_path", None
        )
        if configured_path not in (None, ""):
            return os.path.abspath(os.path.expanduser(str(configured_path)))

        environment_path = os.environ.get("SPECO_DRAFTER_ACCEPTANCE_INCIDENT_LOG")
        if environment_path:
            return os.path.abspath(os.path.expanduser(environment_path))

        checkpoint_dir = _get_nested(
            self.config, ("trainer", "default_local_dir"), None
        )
        if checkpoint_dir not in (None, ""):
            checkpoint_root = os.path.dirname(os.path.abspath(str(checkpoint_dir)))
            if os.path.basename(checkpoint_root) == "checkpoints":
                return os.path.join(
                    os.path.dirname(checkpoint_root),
                    "drafter_acceptance_incidents.jsonl",
                )
        return os.path.abspath("outputs/drafter_acceptance_incidents.jsonl")

    def _speco_record_post_update_acceptance_incident(
        self,
        *,
        update_step: int,
        observed_step: int,
        baseline: float,
        observed: float,
        drop: float,
        threshold: float,
        action: str = "aborted",
    ) -> str | None:
        """Append a guard-triggered regression before terminating training."""
        drafter_cfg = self._speco_drafter_config()
        algorithm = str(
            _get_nested(drafter_cfg, ("speculative_algorithm",), "unknown")
            or "unknown"
        ).lower()
        strategy_version = os.environ.get("SPECO_ONLINE_STRATEGY_TAG", "legacy")
        experiment_id = os.environ.get("SPECO_EXPERIMENT_ID", "unknown")
        log_file = os.environ.get("SPECO_TRAINING_LOG_FILE", "")
        event_identity = (
            f"{experiment_id}:{strategy_version}:{update_step}:{observed_step}:"
            f"{baseline:.12g}:{observed:.12g}:{threshold:.12g}"
        )
        record = {
            "schema_version": 1,
            "incident_id": hashlib.blake2b(
                event_identity.encode(), digest_size=12
            ).hexdigest(),
            "event": "post_update_mean_acceptance_length_regression",
            "source": "runtime_guard",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "algorithm": algorithm,
            "strategy_version": strategy_version or "legacy",
            "experiment_id": experiment_id or "unknown",
            "run_mode": os.environ.get("SPECO_RUN_MODE", "unknown"),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID", ""),
            "training_log": log_file,
            "update_step": int(update_step),
            "observed_step": int(observed_step),
            "baseline_mean_acceptance_length": float(baseline),
            "observed_mean_acceptance_length": float(observed),
            "drop": float(drop),
            "threshold": float(threshold),
        }
        path = self._speco_acceptance_incident_log_path()
        encoded_record = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, encoded_record)
            finally:
                os.close(fd)
        except OSError:
            logger.exception(
                "Failed to record post-update drafter acceptance incident at %s", path
            )
            return None
        return path

    def _speco_arm_post_update_acceptance_guard(
        self, publish_metrics: Mapping[str, Any]
    ) -> dict[str, float | int]:
        """Record the pre-update global acceptance metric after a synchronous publish."""
        threshold = self._speco_post_update_acceptance_drop_abort_threshold()
        if threshold is None or not bool(publish_metrics.get("drafter/published", 0)):
            return {}

        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("publish_async", False)):
            return {
                "drafter/post_update_acceptance_guard/armed": 0,
                "drafter/post_update_acceptance_guard/async_publish_skipped": 1,
            }

        baseline = _speco_metric_float(
            self._speco_current_step_rollout_metrics().get(
                SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC
            )
        )
        if baseline is None:
            return {
                "drafter/post_update_acceptance_guard/armed": 0,
                "drafter/post_update_acceptance_guard/missing_baseline": 1,
            }

        self._speco_pending_post_update_acceptance_guard = {
            "baseline": baseline,
            "update_step": int(self.global_steps),
            "threshold": threshold,
        }
        return {
            "drafter/post_update_acceptance_guard/armed": 1,
            "drafter/post_update_acceptance_guard/baseline_mean_acceptance_length": baseline,
            "drafter/post_update_acceptance_guard/threshold": threshold,
        }

    def _speco_arm_dual_best_observation(
        self, publish_metrics: Mapping[str, Any]
    ) -> dict[str, float | int]:
        if not self._speco_dual_metric_best_enabled() or not bool(
            publish_metrics.get("drafter/published", 0)
        ):
            return {}
        training_cfg = self._speco_drafter_training_config()
        if bool(training_cfg.get("publish_async", False)):
            return {
                "drafter/dual_best/armed": 0,
                "drafter/dual_best/async_publish_skipped": 1,
            }
        self._speco_pending_dual_best_observation = {
            "update_step": int(self.global_steps)
        }
        return {
            "drafter/dual_best/armed": 1,
            "drafter/dual_best/pending_update_step": int(self.global_steps),
        }

    def _speco_check_post_update_acceptance_guard(self) -> dict[str, float | int]:
        """Abort when the first rollout using a published update regresses too far."""
        threshold = self._speco_post_update_acceptance_drop_abort_threshold()
        pending = getattr(self, "_speco_pending_post_update_acceptance_guard", None)
        if threshold is None:
            self._speco_pending_post_update_acceptance_guard = None
            return {}
        if not isinstance(pending, Mapping):
            return {"drafter/post_update_acceptance_guard/checked": 0}

        current_step = int(getattr(self, "global_steps", 0) or 0)
        update_step = int(pending.get("update_step", current_step) or current_step)
        if current_step <= update_step:
            return {"drafter/post_update_acceptance_guard/checked": 0}

        observed = _speco_metric_float(
            self._speco_current_step_rollout_metrics().get(
                SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC
            )
        )
        if observed is None:
            return {
                "drafter/post_update_acceptance_guard/checked": 0,
                "drafter/post_update_acceptance_guard/waiting_for_speculative_rollout": 1,
            }

        baseline = float(pending["baseline"])
        drop = baseline - observed
        self._speco_pending_post_update_acceptance_guard = None
        metrics: dict[str, float | int] = {
            "drafter/post_update_acceptance_guard/checked": 1,
            "drafter/post_update_acceptance_guard/baseline_mean_acceptance_length": baseline,
            "drafter/post_update_acceptance_guard/observed_mean_acceptance_length": observed,
            "drafter/post_update_acceptance_guard/drop": drop,
            "drafter/post_update_acceptance_guard/threshold": threshold,
        }
        if drop >= threshold:
            incident_path = self._speco_record_post_update_acceptance_incident(
                update_step=update_step,
                observed_step=current_step,
                baseline=baseline,
                observed=observed,
                drop=drop,
                threshold=threshold,
            )
            message = (
                "Aborting training: a published drafter update regressed global "
                "mean acceptance length by at least the configured threshold; "
                f"update_step={update_step}, observed_step={current_step}, "
                f"baseline={baseline:.6f}, observed={observed:.6f}, "
                f"drop={drop:.6f}, threshold={threshold:.6f}, "
                f"incident_log={incident_path or 'write_failed'}"
            )
            logger.error(message)
            raise RuntimeError(message)
        return metrics

    @contextmanager
    def _speco_rollout_metrics_fit_hook(self):
        rollout_generation_target = self._speco_rollout_generation_target()
        original_generate_sequences = rollout_generation_target.generate_sequences

        def generate_sequences_with_speco_metrics(manager_self, *args, **kwargs):
            started = time.perf_counter()
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            elapsed_sec = time.perf_counter() - started
            if _speco_is_validation_generation(args, kwargs, gen_batch_output):
                self._speco_store_validation_rollout_metrics(
                    gen_batch_output, elapsed_sec
                )
            else:
                self._speco_store_rollout_metrics(gen_batch_output, elapsed_sec)
            return gen_batch_output

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco_metrics,
            rollout_generation_target,
        )
        try:
            yield
        finally:
            rollout_generation_target.generate_sequences = original_generate_sequences

    @contextmanager
    def _speco_tracking_metrics_hook(self):
        try:
            from verl.utils.tracking import Tracking
        except ImportError:
            yield
            return

        original_init = getattr(Tracking, "__init__", None)
        original_log = getattr(Tracking, "log", None)
        if (
            not callable(original_init)
            or not callable(original_log)
            or getattr(original_log, "_speco_drafter_timing_hook", False)
        ):
            yield
            return

        tracking_instances = []

        def init_with_speco_finish(tracking_self, *args, **kwargs):
            original_init(tracking_self, *args, **kwargs)
            tracking_instances.append(tracking_self)

        def log_with_speco_metrics(tracking_self, *args, **kwargs):
            latest_rollout_metrics = self._speco_current_step_rollout_metrics()
            if "data" in kwargs:
                kwargs = dict(kwargs)
                data = kwargs["data"]
                if (
                    isinstance(data, dict)
                    and isinstance(latest_rollout_metrics, dict)
                    and data.get("training/global_step") == self.global_steps
                ):
                    data = dict(data)
                    data.update(latest_rollout_metrics)
                kwargs["data"] = _speco_add_key_metric_aliases(
                    _speco_move_drafter_timing_next_to_update_actor(data)
                )
                return original_log(tracking_self, *args, **kwargs)
            if args:
                data = args[0]
                if (
                    isinstance(data, dict)
                    and isinstance(latest_rollout_metrics, dict)
                    and data.get("training/global_step") == self.global_steps
                ):
                    data = dict(data)
                    data.update(latest_rollout_metrics)
                args = (
                    _speco_add_key_metric_aliases(
                        _speco_move_drafter_timing_next_to_update_actor(data)
                    ),
                    *args[1:],
                )
            return original_log(tracking_self, *args, **kwargs)

        log_with_speco_metrics._speco_drafter_timing_hook = True
        Tracking.__init__ = init_with_speco_finish
        Tracking.log = log_with_speco_metrics
        completed = False
        try:
            yield
            completed = True
        finally:
            Tracking.__init__ = original_init
            Tracking.log = original_log
            for tracking_instance in tracking_instances:
                logger_map = getattr(tracking_instance, "logger", None)
                if not isinstance(logger_map, dict):
                    continue
                wandb_logger = logger_map.pop("wandb", None)
                if wandb_logger is None:
                    continue
                exit_code = 0 if completed else 1
                try:
                    wandb_logger.finish(exit_code=exit_code)
                    teardown = getattr(wandb_logger, "teardown", None)
                    if callable(teardown):
                        teardown(exit_code=exit_code)
                except Exception:
                    logger.exception("Failed to finish the SPECO W&B run cleanly")

    def _speco_compute_old_log_prob_without_forced_entropy(self, batch: DataProto):
        batch = _select_policy_model_batch(batch)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self._speco_oldlogprob_calculate_entropy()
        tu.assign_non_tensor(
            batch_td, calculate_entropy=calculate_entropy, compute_loss=False
        )

        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]

        log_probs = no_padding_2_padding(log_probs, batch_td)
        if entropy is None:
            entropy = torch.zeros_like(log_probs, dtype=torch.float32)
        else:
            entropy = no_padding_2_padding(entropy, batch_td)
        if routed_experts is not None:
            old_log_prob = tu.get_tensordict(
                {
                    "old_log_probs": log_probs.float(),
                    "entropys": entropy.float(),
                    "routed_experts": routed_experts,
                }
            )
        else:
            old_log_prob = tu.get_tensordict(
                {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
            )
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    @contextmanager
    def _speco_oldlogprob_entropy_fit_hook(self):
        original_compute_old_log_prob = self._compute_old_log_prob

        def compute_old_log_prob_without_forced_entropy(trainer_self, batch: DataProto):
            return self._speco_compute_old_log_prob_without_forced_entropy(batch)

        self._compute_old_log_prob = MethodType(
            compute_old_log_prob_without_forced_entropy, self
        )
        try:
            yield
        finally:
            self._compute_old_log_prob = original_compute_old_log_prob

    @contextmanager
    def _speco_online_fit_hooks(self):
        rollout_generation_target = self._speco_rollout_generation_target()
        original_generate_sequences = rollout_generation_target.generate_sequences
        original_compute_old_log_prob = self._compute_old_log_prob
        original_update_actor = self._update_actor
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        original_checkpoint_update_weights = (
            getattr(checkpoint_manager, "update_weights", None)
            if checkpoint_manager is not None
            else None
        )
        defer_publish_until_update_weights = callable(
            original_checkpoint_update_weights
        )
        pending_drafter_publish = {
            "ready": False,
            "drafter_trained": False,
            "actor_output": None,
        }

        def generate_sequences_with_speco(manager_self, *args, **kwargs):
            self._speco_wait_pending_drafter_publish()
            is_validation_input = _speco_is_validation_generation(args, kwargs)
            candidate_metrics: dict[str, Any] = {}
            selected_alpha = None
            if (
                not is_validation_input
                and isinstance(self._speco_pending_candidate_update, Mapping)
            ):
                pending = self._speco_pending_candidate_update
                try:
                    selected_alpha, candidate_metrics = (
                        self._speco_select_candidate_with_rollout(
                            original_generate_sequences, args, kwargs
                        )
                    )
                except Exception:
                    logger.exception(
                        "Candidate rollout gate failed; restoring baseline drafter"
                    )
                    update_step = int(pending["update_step"])
                    self._speco_publish_candidate_alpha(update_step, 0.0)
                    self._speco_finalize_candidate(update_step, 0.0)
                    self._speco_pending_candidate_update = None
                    candidate_metrics = {
                        "drafter/candidate/gate_ran": 1,
                        "drafter/candidate/gate_failed": 1,
                        "drafter/candidate/selected_alpha": 0.0,
                    }
            rollout_started = time.perf_counter()
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            if selected_alpha is not None:
                gen_batch_output = self._speco_commit_or_rollback_candidate(
                    original_generate_sequences,
                    args,
                    kwargs,
                    gen_batch_output,
                    selected_alpha,
                    candidate_metrics,
                )
            rollout_elapsed = time.perf_counter() - rollout_started
            if candidate_metrics:
                self._speco_pending_candidate_metrics.update(candidate_metrics)
            is_validation_generation = _speco_is_validation_generation(
                args, kwargs, gen_batch_output
            )
            if is_validation_generation:
                self._speco_store_validation_rollout_metrics(
                    gen_batch_output, rollout_elapsed
                )
            else:
                self._speco_store_rollout_metrics(gen_batch_output, rollout_elapsed)
                collected = self._speco_collect_generation_samples(gen_batch_output)
                if collected:
                    meta_info = getattr(gen_batch_output, "meta_info", None)
                    if isinstance(meta_info, dict):
                        meta_info.setdefault("metrics", {})[
                            "drafter/collected_samples"
                        ] = collected
            return gen_batch_output

        def compute_old_log_prob_with_speco(trainer_self, batch: DataProto):
            self._speco_fixed_eval_export_metrics = {}
            if not self._speco_oldlogprob_collection_enabled():
                if self._speco_oldlogprob_entropy_hook_enabled():
                    return self._speco_compute_old_log_prob_without_forced_entropy(
                        batch
                    )
                return original_compute_old_log_prob(batch)

            oldlogprob_started = time.perf_counter()
            self._speco_last_oldlogprob_candidate_samples = 0
            self._speco_last_oldlogprob_planned_samples = 0
            self._speco_last_oldlogprob_collected_samples = 0
            self._speco_last_oldlogprob_collected_rows = 0
            self._speco_last_oldlogprob_payload_mib = 0.0
            self._speco_last_oldlogprob_window_strata = {}
            self._speco_last_oldlogprob_materialized_window_strata = {}
            self._speco_last_oldlogprob_drop_reasons = {}
            self._speco_last_oldlogprob_dispatched_samples = 0
            self._speco_last_oldlogprob_consumed_samples = 0
            self._speco_last_oldlogprob_unique_samples = 0
            self._speco_last_oldlogprob_consumed_unique_samples = 0
            self._speco_last_spec_trace_samples = 0
            self._speco_last_spec_trace_verify_steps = 0
            self._speco_last_spec_trace_draft_tokens = 0
            self._speco_last_spec_trace_teacher_logprob_tokens = 0
            self._speco_last_spec_trace_accepted_tokens = 0
            self._speco_last_spec_trace_first_rejections = 0
            self._speco_last_spec_trace_unverified_suffix_tokens = 0
            self._speco_last_spec_trace_all_accepted_steps = 0
            self._speco_last_spec_trace_diagnostics = {}
            self._speco_last_fixed_eval_trace_metrics = {}
            self._speco_last_drafter_guard_reason = ""
            self._speco_last_oldlogprob_select_elapsed_sec = 0.0
            self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
            self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
            self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
            self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
            self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
            self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
            self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
            self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
            self._speco_last_oldlogprob_total_elapsed_sec = 0.0
            collect_interval_matched = self._speco_should_collect_drafter_this_step()
            train_interval_matched = self._speco_should_train_drafter_this_step()
            self._speco_last_collect_interval_matched = int(collect_interval_matched)
            prepare_started = time.perf_counter()
            original_batch = batch

            def compute_old_log_prob_without_collection():
                self._speco_last_oldlogprob_prepare_elapsed_sec = (
                    time.perf_counter() - prepare_started
                )
                compute_started = time.perf_counter()
                if self._speco_oldlogprob_entropy_hook_enabled():
                    old_log_prob, old_log_prob_mfu = (
                        self._speco_compute_old_log_prob_without_forced_entropy(
                            original_batch
                        )
                    )
                else:
                    old_log_prob, old_log_prob_mfu = original_compute_old_log_prob(
                        original_batch
                    )
                self._speco_last_oldlogprob_compute_elapsed_sec = (
                    time.perf_counter() - compute_started
                )
                self._speco_last_oldlogprob_total_elapsed_sec = (
                    time.perf_counter() - oldlogprob_started
                )
                return old_log_prob, old_log_prob_mfu

            if not collect_interval_matched or not train_interval_matched:
                return compute_old_log_prob_without_collection()

            collect_plan = self._speco_build_oldlogprob_collect_plan(original_batch)
            if collect_plan is None:
                return compute_old_log_prob_without_collection()
            self._speco_export_fixed_eval_snapshot(original_batch, collect_plan)
            batch = _select_policy_model_batch(original_batch)
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self._speco_oldlogprob_calculate_entropy()
            tu.assign_non_tensor(
                batch_td, calculate_entropy=calculate_entropy, compute_loss=False
            )
            batch_td[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
            batch_td[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan[
                "hidden_positions"
            ]
            batch_td[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan[
                "hidden_position_mask"
            ]
            batch_td[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_AUX_LAYER_IDS_KEY,
                self._speco_oldlogprob_aux_layer_ids(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
                self._speco_oldlogprob_hidden_capture_impl(),
            )
            tu.assign_non_tensor_data(
                batch_td,
                OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
                self._speco_oldlogprob_hidden_layout(),
            )
            tu.assign_non_tensor_data(batch_td, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True)

            self._speco_last_oldlogprob_prepare_elapsed_sec = (
                time.perf_counter() - prepare_started
            )
            compute_started = time.perf_counter()
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            self._speco_last_oldlogprob_compute_elapsed_sec = (
                time.perf_counter() - compute_started
            )
            collect_started = time.perf_counter()
            self._speco_collect_oldlogprob_features(
                original_batch, collect_plan, output
            )
            self._speco_last_oldlogprob_collect_elapsed_sec = (
                time.perf_counter() - collect_started
            )

            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]

            log_probs = no_padding_2_padding(log_probs, batch_td)
            if entropy is None:
                entropy = torch.zeros_like(log_probs, dtype=torch.float32)
            else:
                entropy = no_padding_2_padding(entropy, batch_td)
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {
                        "old_log_probs": log_probs.float(),
                        "entropys": entropy.float(),
                        "routed_experts": routed_experts,
                    }
                )
            else:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
                )
            old_log_prob = DataProto.from_tensordict(old_log_prob)
            self._speco_last_oldlogprob_total_elapsed_sec = (
                time.perf_counter() - oldlogprob_started
            )
            return old_log_prob, old_log_prob_mfu

        def update_actor_with_speco(trainer_self, *args, **kwargs):
            update_actor_started = time.perf_counter()
            metrics = dict(self._speco_pending_candidate_metrics)
            metrics.update(getattr(self, "_speco_fixed_eval_export_metrics", {}))
            self._speco_pending_candidate_metrics = {}
            metrics.update({
                "drafter/raw_drafter_samples": int(
                    getattr(self, "_speco_last_raw_drafter_samples", 0)
                ),
                "drafter/collected_samples": int(
                    getattr(self, "_speco_last_collected_samples", 0)
                ),
                "drafter/collect_interval_matched": int(
                    getattr(self, "_speco_last_collect_interval_matched", 0)
                ),
                "drafter/train_interval_matched": int(
                    self._speco_should_train_drafter_this_step()
                ),
                "drafter/oldlogprob_candidate_samples": int(
                    getattr(self, "_speco_last_oldlogprob_candidate_samples", 0)
                ),
                "drafter/oldlogprob_planned_samples": int(
                    getattr(self, "_speco_last_oldlogprob_planned_samples", 0)
                ),
                "drafter/oldlogprob_materialized_samples": int(
                    getattr(self, "_speco_last_oldlogprob_collected_samples", 0)
                ),
                "drafter/oldlogprob_dispatched_samples": int(
                    getattr(self, "_speco_last_oldlogprob_dispatched_samples", 0)
                ),
                "drafter/oldlogprob_consumed_samples": int(
                    getattr(self, "_speco_last_oldlogprob_consumed_samples", 0)
                ),
                "drafter/oldlogprob_unique_samples": int(
                    getattr(self, "_speco_last_oldlogprob_unique_samples", 0)
                ),
                "drafter/oldlogprob_consumed_unique_samples": int(
                    getattr(
                        self, "_speco_last_oldlogprob_consumed_unique_samples", 0
                    )
                ),
                "drafter/oldlogprob_collected_rows": int(
                    getattr(self, "_speco_last_oldlogprob_collected_rows", 0)
                ),
                "drafter/spec_trace_samples": int(
                    getattr(self, "_speco_last_spec_trace_samples", 0)
                ),
                "drafter/spec_trace_verify_steps": int(
                    getattr(self, "_speco_last_spec_trace_verify_steps", 0)
                ),
                "drafter/spec_trace_draft_tokens": int(
                    getattr(self, "_speco_last_spec_trace_draft_tokens", 0)
                ),
                "drafter/spec_trace_teacher_logprob_tokens": int(
                    getattr(
                        self, "_speco_last_spec_trace_teacher_logprob_tokens", 0
                    )
                ),
                "drafter/spec_trace_accepted_tokens": int(
                    getattr(self, "_speco_last_spec_trace_accepted_tokens", 0)
                ),
                "drafter/spec_trace_first_rejections": int(
                    getattr(self, "_speco_last_spec_trace_first_rejections", 0)
                ),
                "drafter/spec_trace_unverified_suffix_tokens": int(
                    getattr(
                        self,
                        "_speco_last_spec_trace_unverified_suffix_tokens",
                        0,
                    )
                ),
                "drafter/spec_trace_all_accepted_steps": int(
                    getattr(self, "_speco_last_spec_trace_all_accepted_steps", 0)
                ),
            })
            metrics.update(self._speco_maybe_save_dual_best_from_rollout())
            if not self._speco_candidate_gate_enabled():
                metrics.update(self._speco_check_post_update_acceptance_guard())
            collected_samples = int(
                getattr(self, "_speco_last_oldlogprob_collected_samples", 0)
            )
            planned_samples = int(
                getattr(self, "_speco_last_oldlogprob_planned_samples", 0)
            )
            dispatched_samples = int(
                getattr(self, "_speco_last_oldlogprob_dispatched_samples", 0)
            )
            consumed_samples = int(
                getattr(self, "_speco_last_oldlogprob_consumed_samples", 0)
            )
            metrics["drafter/oldlogprob_materialization_ratio"] = (
                collected_samples / planned_samples if planned_samples > 0 else 0.0
            )
            metrics["drafter/oldlogprob_consumption_ratio"] = (
                consumed_samples / dispatched_samples
                if dispatched_samples > 0
                else 0.0
            )
            draft_tokens = int(
                getattr(self, "_speco_last_spec_trace_draft_tokens", 0)
            )
            accepted_tokens = int(
                getattr(self, "_speco_last_spec_trace_accepted_tokens", 0)
            )
            metrics["drafter/spec_trace_unaccepted_tokens"] = max(
                draft_tokens - accepted_tokens, 0
            )
            metrics["drafter/spec_trace_rejected_tokens"] = metrics[
                "drafter/spec_trace_first_rejections"
            ]
            metrics["drafter/spec_trace_sample_coverage"] = (
                metrics["drafter/spec_trace_samples"] / collected_samples
                if collected_samples > 0
                else 0.0
            )
            metrics["drafter/spec_trace_token_acceptance"] = (
                accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0
            )
            teacher_logprob_tokens = int(
                getattr(
                    self, "_speco_last_spec_trace_teacher_logprob_tokens", 0
                )
            )
            metrics["drafter/spec_trace_teacher_logprob_coverage"] = (
                teacher_logprob_tokens / draft_tokens if draft_tokens > 0 else 0.0
            )
            window_strata = getattr(
                self, "_speco_last_oldlogprob_window_strata", {}
            )
            for stratum in ("front", "middle", "tail", "full"):
                metrics[f"drafter/window_samples_{stratum}"] = int(
                    window_strata.get(stratum, 0)
                )
            materialized_strata = getattr(
                self, "_speco_last_oldlogprob_materialized_window_strata", {}
            )
            for stratum in ("front", "middle", "tail", "full", "unknown"):
                metrics[f"drafter/materialized_window_samples_{stratum}"] = int(
                    materialized_strata.get(stratum, 0)
                )
            for reason, count in (
                getattr(self, "_speco_last_oldlogprob_drop_reasons", {}) or {}
            ).items():
                metrics[f"drafter/collection_drop/{reason}"] = int(count)
            for stage, count in (
                getattr(self, "_speco_last_spec_trace_diagnostics", {}) or {}
            ).items():
                safe_stage = str(stage).replace("/", "_").replace(" ", "_")
                metrics[f"drafter/spec_trace_stage/{safe_stage}"] = int(count)

            fixed_eval = getattr(self, "_speco_last_fixed_eval_trace_metrics", {}) or {}
            for key in (
                "materialized_samples",
                "samples",
                "verify_steps",
                "draft_tokens",
                "accepted_tokens",
                "first_rejections",
                "unverified_suffix_tokens",
                "all_accepted_steps",
                "actor_snapshot_step",
            ):
                metrics[f"drafter/fixed_eval/{key}"] = int(fixed_eval.get(key, 0) or 0)
            fixed_drafts = int(fixed_eval.get("draft_tokens", 0) or 0)
            fixed_accepted = int(fixed_eval.get("accepted_tokens", 0) or 0)
            fixed_verify_steps = int(fixed_eval.get("verify_steps", 0) or 0)
            metrics["drafter/fixed_eval/unaccepted_tokens"] = max(
                fixed_drafts - fixed_accepted, 0
            )
            metrics["drafter/fixed_eval/rejected_tokens"] = int(
                fixed_eval.get("first_rejections", 0) or 0
            )
            metrics["drafter/fixed_eval/token_acceptance"] = (
                fixed_accepted / fixed_drafts if fixed_drafts > 0 else 0.0
            )
            metrics["drafter/fixed_eval/mean_acceptance_length"] = (
                1.0 + fixed_accepted / fixed_verify_steps
                if fixed_verify_steps > 0
                else 0.0
            )

            train_requested = self._speco_should_attempt_drafter_train_this_step()
            guard_applicable = (
                self._speco_oldlogprob_collection_requested()
                and bool(getattr(self, "_speco_last_collect_interval_matched", 0))
                and self._speco_should_train_drafter_this_step()
                and planned_samples > 0
            )
            guard_passed, guard_reason = (
                self._speco_drafter_collection_guard()
                if guard_applicable
                else (True, "not_applicable")
            )
            self._speco_last_drafter_guard_reason = guard_reason
            guard_reason_ids = {
                "not_applicable": 0,
                "not_oldlogprob_collection": 0,
                "ok": 1,
                "materialization_coverage": 2,
                "worker_consumption_coverage": 3,
                "spec_trace_coverage": 4,
                "spec_trace_teacher_logprob_coverage": 5,
            }
            metrics["drafter/training_guard_applicable"] = int(guard_applicable)
            metrics["drafter/training_guard_passed"] = int(guard_passed)
            metrics["drafter/training_guard_blocked"] = int(
                train_requested and not guard_passed
            )
            metrics["drafter/training_guard_reason_id"] = int(
                guard_reason_ids.get(guard_reason, -1)
            )
            should_train_drafter = train_requested and guard_passed
            if train_requested and not guard_passed:
                message = (
                    "Skipping drafter update because collection integrity guard failed: "
                    f"reason={guard_reason}, planned={planned_samples}, "
                    f"materialized={collected_samples}, dispatched={dispatched_samples}, "
                    f"consumed={consumed_samples}, trace_samples="
                    f"{metrics['drafter/spec_trace_samples']}"
                )
                action = str(
                    self._speco_drafter_training_config().get(
                        "collection_integrity_fail_action", "skip"
                    )
                    or "skip"
                ).lower()
                if action == "raise":
                    raise RuntimeError(message)
                logger.warning(message)
            if should_train_drafter:
                self._speco_set_drafter_global_step()
                metrics.update(self._speco_sync_target_lm_head_weight())
            else:
                metrics["drafter/target_lm_head_synced"] = 0
            actor_started = time.perf_counter()
            actor_output = original_update_actor(*args, **kwargs)
            actor_elapsed = time.perf_counter() - actor_started
            pending_best = self._speco_pending_best_candidate_save
            if isinstance(pending_best, Mapping):
                self._speco_pending_best_candidate_save = None
                metrics.update(
                    self._speco_maybe_save_best_candidate(**pending_best)
                )
            if should_train_drafter:
                drafter_trained, train_metrics = self._speco_train_drafter()
            else:
                drafter_trained, train_metrics = (
                    False,
                    {
                        "drafter/trained": 0,
                        "drafter/train_successful_steps_max": 0,
                        "drafter/train_no_trainable_batch": int(
                            self._speco_should_train_drafter_this_step()
                        ),
                        "drafter/train_activation_failed": 0,
                    },
                )
            metrics.update(train_metrics)
            if self._speco_candidate_gate_enabled() and drafter_trained:
                metrics.update(
                    self._speco_stage_candidate_update(
                        drafter_trained, train_metrics
                    )
                )
            elif defer_publish_until_update_weights and drafter_trained:
                pending_drafter_publish["ready"] = True
                pending_drafter_publish["drafter_trained"] = drafter_trained
                pending_drafter_publish["actor_output"] = actor_output
            else:
                publish_metrics = self._speco_publish_drafter_weights(drafter_trained)
                metrics.update(publish_metrics)
                metrics.update(
                    self._speco_arm_post_update_acceptance_guard(publish_metrics)
                )
                metrics.update(self._speco_arm_dual_best_observation(publish_metrics))
            metrics["timing_s/drafter"] = max(
                0.0, time.perf_counter() - update_actor_started - actor_elapsed
            )
            known_drafter_timing = 0.0
            for key in (
                "timing_s/drafter_sync_target_lm_head",
                "timing_s/drafter_train_rpc",
                "timing_s/drafter_publish_wait_pending",
                "timing_s/drafter_publish_fetch_snapshot",
                "timing_s/drafter_publish_update_weights",
            ):
                value = _speco_metric_float(metrics.get(key))
                if value is not None:
                    known_drafter_timing += value
            metrics["timing_s/drafter_outer_unaccounted"] = max(
                0.0,
                metrics["timing_s/drafter"] - known_drafter_timing,
            )
            return self._speco_update_output_metrics(actor_output, metrics)

        def update_weights_with_speco(manager_self, *args, **kwargs):
            result = original_checkpoint_update_weights(*args, **kwargs)
            if (
                pending_drafter_publish["ready"]
                and not self._speco_candidate_gate_enabled()
            ):
                publish_metrics = self._speco_publish_drafter_weights(
                    pending_drafter_publish["drafter_trained"]
                )
                publish_metrics.update(
                    self._speco_arm_post_update_acceptance_guard(publish_metrics)
                )
                publish_metrics.update(
                    self._speco_arm_dual_best_observation(publish_metrics)
                )
                self._speco_update_output_metrics(
                    pending_drafter_publish["actor_output"], publish_metrics
                )
                pending_drafter_publish["ready"] = False
                pending_drafter_publish["drafter_trained"] = False
                pending_drafter_publish["actor_output"] = None
            return result

        rollout_generation_target.generate_sequences = MethodType(
            generate_sequences_with_speco,
            rollout_generation_target,
        )
        if self._speco_oldlogprob_collection_requested():
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco, self
            )
        elif self._speco_oldlogprob_entropy_hook_enabled():
            self._compute_old_log_prob = MethodType(
                compute_old_log_prob_with_speco,
                self,
            )
        self._update_actor = MethodType(update_actor_with_speco, self)
        if defer_publish_until_update_weights:
            checkpoint_manager.update_weights = MethodType(
                update_weights_with_speco, checkpoint_manager
            )
        try:
            yield
        finally:
            rollout_generation_target.generate_sequences = original_generate_sequences
            self._compute_old_log_prob = original_compute_old_log_prob
            self._update_actor = original_update_actor
            if defer_publish_until_update_weights:
                checkpoint_manager.update_weights = original_checkpoint_update_weights
            self._speco_wait_pending_drafter_publish()

    @staticmethod
    def is_drafter_rollout_enabled(config) -> bool:
        return bool(
            _get_nested(
                config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False
            )
        )

    @staticmethod
    def is_drafter_training_enabled(config) -> bool:
        drafter_enabled = bool(
            _get_nested(
                config, ("actor_rollout_ref", "rollout", "drafter", "enable"), False
            )
        )
        training_enabled = bool(
            _get_nested(
                config,
                ("actor_rollout_ref", "rollout", "drafter", "enable_drafter_training"),
                False,
            )
        )
        return drafter_enabled and training_enabled

    def fit(self):
        fit_completed = False
        try:
            with self._speco_checkpoint_offset_hook():
                if self.is_drafter_training_enabled(self.config):
                    self._speco_activate_drafter_training_model_before_fit()
                    with (
                        self._speco_tracking_metrics_hook(),
                        self._speco_online_fit_hooks(),
                    ):
                        result = super().fit()
                        fit_completed = True
                        return result
                if self.is_drafter_rollout_enabled(self.config):
                    with (
                        self._speco_tracking_metrics_hook(),
                        self._speco_rollout_metrics_fit_hook(),
                    ):
                        if self._speco_oldlogprob_entropy_hook_enabled():
                            with self._speco_oldlogprob_entropy_fit_hook():
                                result = super().fit()
                                fit_completed = True
                                return result
                        result = super().fit()
                        fit_completed = True
                        return result
                if self._speco_oldlogprob_entropy_hook_enabled():
                    with self._speco_oldlogprob_entropy_fit_hook():
                        result = super().fit()
                        fit_completed = True
                        return result

                result = super().fit()
                fit_completed = True
                return result
        finally:
            pending_candidate = getattr(
                self, "_speco_pending_candidate_update", None
            )
            if isinstance(pending_candidate, Mapping):
                update_step = int(pending_candidate["update_step"])
                logger.warning(
                    "Discarding unvalidated terminal drafter candidate at step %s",
                    update_step,
                )
                self._speco_finalize_candidate(update_step, 0.0)
                self._speco_pending_candidate_update = None
            self._speco_wait_pending_drafter_checkpoint()
            if fit_completed and getattr(
                self, "_speco_last_saved_drafter_step", None
            ) != int(self.global_steps):
                # Periodic saves may intentionally be shifted away from publish
                # steps. Persist the exact terminal drafter state as well.
                self._speco_save_drafter_checkpoint(wait=True)
            if fit_completed:
                self._speco_mark_final_drafter_checkpoint()

    def _speco_checkpoint_save_offset(self) -> int:
        value = _get_nested(self.config, ("speco", "checkpoint", "save_offset"), 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @contextmanager
    def _speco_checkpoint_offset_hook(self):
        """Shift upstream periodic saves without modifying the pinned VERL checkout."""

        trainer_cfg = _get_nested(self.config, ("trainer",), None)
        frequency = _get_nested(self.config, ("trainer", "save_freq"), -1)
        offset = self._speco_checkpoint_save_offset()
        try:
            frequency = int(frequency)
        except (TypeError, ValueError):
            frequency = -1
        if frequency <= 0 or offset == 0:
            yield
            return
        if offset < 0 or offset >= frequency:
            raise ValueError(
                f"speco.checkpoint.save_offset must satisfy 0 <= offset < save_freq; "
                f"got offset={offset}, save_freq={frequency}"
            )

        original_update_actor = self._update_actor
        original_save_freq = _get_nested(
            self.config, ("trainer", "save_freq"), frequency
        )
        last_saved_step = None

        def update_actor_with_offset(trainer_self, batch):
            nonlocal last_saved_step
            result = original_update_actor(batch)
            step = int(trainer_self.global_steps)
            if step != last_saved_step and speco_step_matches_save_schedule(
                step, frequency, offset
            ):
                logger.warning(
                    "[SPECO checkpoint] saving shifted checkpoint at step=%s frequency=%s offset=%s",
                    step,
                    frequency,
                    offset,
                )
                trainer_self._save_checkpoint()
                last_saved_step = step
            return result

        try:
            with open_dict(trainer_cfg):
                trainer_cfg.save_freq = -1
        except Exception:  # noqa: BLE001
            setattr(trainer_cfg, "save_freq", -1)
        self._update_actor = MethodType(update_actor_with_offset, self)
        try:
            yield
        finally:
            self._update_actor = original_update_actor
            try:
                with open_dict(trainer_cfg):
                    trainer_cfg.save_freq = original_save_freq
            except Exception:  # noqa: BLE001
                setattr(trainer_cfg, "save_freq", original_save_freq)

    def _save_checkpoint(self):
        self._speco_save_drafter_checkpoint(wait=True)
        return super()._save_checkpoint()
