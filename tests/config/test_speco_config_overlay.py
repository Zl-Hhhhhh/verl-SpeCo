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

import os
import shutil
from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra", reason="config overlay tests need hydra-core")
omegaconf = pytest.importorskip(
    "omegaconf", reason="config overlay tests need omegaconf"
)

compose = hydra.compose
initialize_config_dir = hydra.initialize_config_dir
OmegaConf = omegaconf.OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "verl_speco" / "config"


def _upstream_repo_root(upstream_root: str) -> Path:
    base = Path(upstream_root)
    for candidate in (base, base / "verl"):
        if (candidate / "verl" / "trainer" / "config").is_dir():
            return candidate
    raise AssertionError(
        "VERL_SPECO_UPSTREAM_ROOT must point to the upstream verl checkout "
        "or to a directory containing it"
    )


def _copy_overlay_configs(
    upstream_config: Path, composed_config_dir: Path, names: tuple[str, ...]
) -> None:
    shutil.copytree(upstream_config, composed_config_dir)
    for config_name in names:
        config_source = (CONFIG_DIR / config_name).read_text(encoding="utf-8")
        config_source = config_source.replace(
            "pkg://verl.trainer.config",
            composed_config_dir.resolve().as_uri(),
        )
        (composed_config_dir / config_name).write_text(config_source, encoding="utf-8")


def test_overlay_has_expected_default_drafter_shape() -> None:
    raw = OmegaConf.load(CONFIG_DIR / "speco_base.yaml")
    drafter = raw.actor_rollout_ref.rollout.drafter

    assert raw.speco.verl_base.version == "0.8.0"
    assert raw.speco.verl_base.branch == "release/v0.8.0"
    assert raw.speco.checkpoint.save_offset == 0
    assert drafter.enable is False
    assert drafter.enable_drafter_training is False
    assert drafter.resume_model_path is None
    assert drafter.training.worker_world_size is None
    assert drafter.training.collect_hidden_states_from_sgl is False
    assert drafter.training.collect_hidden_states_from_old_logprob is False
    assert drafter.vllm.allow_lossy_speculative_sampling is False
    assert drafter.training.allow_sglang_prenorm_last_layer is False
    assert drafter.training.lr == pytest.approx(1e-5)
    assert drafter.training.lr_scheduler_type == "constant"
    assert drafter.training.lr_decay_steps == 100
    assert drafter.training.min_lr_ratio == pytest.approx(0.1)
    assert drafter.training.warmup_style is None
    assert drafter.training.resume_trainer_state_from_checkpoint is True
    assert drafter.training.fixed_eval_samples_per_replica == 0
    assert drafter.training.train_unique_samples_first is False
    assert drafter.training.dspark_trace_aligned_anchors is False
    assert drafter.training.dspark_loss_decay_gamma == pytest.approx(0.0)
    assert drafter.training.dspark_ce_loss_alpha == pytest.approx(0.0)
    assert drafter.training.dspark_l1_loss_alpha == pytest.approx(0.0)
    assert drafter.training.dspark_accepted_k3_alpha == pytest.approx(1.0)
    assert drafter.training.dspark_accepted_fkl_alpha == pytest.approx(0.0)
    assert drafter.training.dspark_rejected_k3_alpha == pytest.approx(1.0)
    assert drafter.training.dspark_rejected_aux_alpha == pytest.approx(0.0)
    assert drafter.training.dspark_rejected_prefix_length == 0
    assert drafter.training.dspark_response_stream_weight == pytest.approx(1.0)
    assert drafter.training.dspark_rejected_stream_weight == pytest.approx(1.0)
    assert drafter.training.dspark_rejected_position_decay == pytest.approx(0.9)
    assert drafter.training.dspark_topk_support == "student"
    assert drafter.training.dspark_first_rejection_ce_alpha == 0.0
    assert drafter.training.dspark_rejected_first_weight_multiplier == pytest.approx(1.0)
    assert drafter.training.dspark_rejected_suffix_weight_multiplier == pytest.approx(1.0)
    assert drafter.training.dspark_asymmetric_k3_enabled is False
    assert drafter.training.dspark_accepted_asymmetric_k3_enabled is None
    assert drafter.training.dspark_rejected_asymmetric_k3_enabled is None
    assert drafter.training.dspark_asymmetric_k3_positive_threshold == pytest.approx(
        2.0
    )
    assert drafter.training.dspark_dynamic_accept_reject_weighting_enabled is False
    assert drafter.training.dspark_dynamic_stream_weight_min_multiplier == pytest.approx(
        0.25
    )
    assert drafter.training.dspark_dynamic_stream_weight_max_multiplier == pytest.approx(
        4.0
    )
    assert drafter.training.dspark_k3_distribution_bins == 512
    assert drafter.training.dspark_dump_k3_logprobs is False
    assert drafter.training.dspark_dump_k3_logprobs_dir is None
    assert drafter.training.dspark_dump_k3_logprobs_interval_steps == 1
    assert drafter.training.dspark_dump_k3_logprobs_max_tokens_per_stream == 0
    assert drafter.training.post_update_mean_acceptance_length_drop_abort_threshold == pytest.approx(
        0.5
    )
    assert drafter.training.eagle1_num_hidden_layers == 1


def test_overlay_composes_with_release_upstream_verl(tmp_path: Path) -> None:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip(
            "set VERL_SPECO_UPSTREAM_ROOT to check compose against release/v0.8.0 verl"
        )
    upstream_config = _upstream_repo_root(upstream_root) / "verl" / "trainer" / "config"
    assert upstream_config.is_dir()

    composed_config_dir = tmp_path / "config"
    _copy_overlay_configs(
        upstream_config, composed_config_dir, ("speco_base.yaml", "speco_trainer.yaml")
    )

    with initialize_config_dir(config_dir=str(composed_config_dir), version_base=None):
        config = compose(config_name="speco_trainer")

    assert config.speco.verl_base.version == "0.8.0"
    assert config.actor_rollout_ref.rollout.drafter.enable is False
    assert "trainer" in config
    assert "algorithm" in config


def test_draft_trainer_composes_as_primary_config(tmp_path: Path) -> None:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip(
            "set VERL_SPECO_UPSTREAM_ROOT to check compose against release/v0.8.0 verl"
        )
    upstream_config = _upstream_repo_root(upstream_root) / "verl" / "trainer" / "config"
    assert upstream_config.is_dir()

    composed_config_dir = tmp_path / "config"
    _copy_overlay_configs(
        upstream_config, composed_config_dir, ("speco_base.yaml", "draft_trainer.yaml")
    )

    with initialize_config_dir(config_dir=str(composed_config_dir), version_base=None):
        config = compose(config_name="draft_trainer")

    assert config.actor_rollout_ref.rollout.drafter.training.mode == "offline"
    assert config.speco.draft_training.enable is True
    assert "trainer" in config
    assert "algorithm" in config
