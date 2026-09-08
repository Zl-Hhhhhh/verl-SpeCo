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
import json
import shutil
from types import SimpleNamespace

from omegaconf import OmegaConf

from verl_speco.integration.task_runner import _restore_drafter_model_path_from_checkpoint


def test_restore_drafter_model_path_from_committed_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "run"
    (checkpoint_dir / "global_step_20").mkdir(parents=True)
    draft_dir = checkpoint_dir / "drafter" / "draft_step_20"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    (draft_dir / "metadata.json").write_text(json.dumps({"step": 20}), encoding="utf-8")
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "model_path": "/models/original-draft",
                        "checkpoint_path": None,
                    }
                }
            },
            "trainer": {"default_local_dir": str(checkpoint_dir)},
        }
    )

    restored = _restore_drafter_model_path_from_checkpoint(config)

    assert restored == str(draft_dir)
    assert config.actor_rollout_ref.rollout.drafter.model_path == str(draft_dir)
    assert (
        config.actor_rollout_ref.rollout.drafter.runtime_config_source_model_path
        == "/models/original-draft"
    )
    assert config.actor_rollout_ref.rollout.drafter.checkpoint_path == str(checkpoint_dir / "drafter")


def test_restore_drafter_model_path_respects_disabled_resume(tmp_path):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "model_path": "/models/original-draft",
                        "checkpoint_path": None,
                    }
                }
            },
            "trainer": {
                "default_local_dir": str(tmp_path),
                "resume_mode": "disable",
            },
        }
    )

    assert _restore_drafter_model_path_from_checkpoint(config) is None
    assert config.actor_rollout_ref.rollout.drafter.model_path == "/models/original-draft"


def test_restore_drafter_model_path_allows_explicit_safe_rollback(tmp_path):
    checkpoint_dir = tmp_path / "run"
    (checkpoint_dir / "global_step_40").mkdir(parents=True)
    draft_dir = checkpoint_dir / "drafter" / "draft_step_30"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    (draft_dir / "metadata.json").write_text(
        json.dumps({"step": 30, "complete": True}), encoding="utf-8"
    )
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
        "40", encoding="utf-8"
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "model_path": "/models/original-draft",
                        "resume_model_path": str(draft_dir),
                        "checkpoint_path": None,
                    }
                }
            },
            "trainer": {
                "default_local_dir": str(checkpoint_dir),
                "resume_mode": "auto",
            },
        }
    )

    restored = _restore_drafter_model_path_from_checkpoint(config)

    assert restored == str(draft_dir)
    assert config.actor_rollout_ref.rollout.drafter.model_path == str(draft_dir)
    assert (
        config.actor_rollout_ref.rollout.drafter.checkpoint_path
        == str(checkpoint_dir / "drafter")
    )


def test_restore_normalizes_legacy_dflash_checkpoint_for_runtime(tmp_path):
    checkpoint_dir = tmp_path / "run"
    (checkpoint_dir / "global_step_20").mkdir(parents=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DFlashDraftModel"],
                "dflash_config": {"target_layer_ids": [1, 9, 17]},
            }
        ),
        encoding="utf-8",
    )
    draft_dir = checkpoint_dir / "drafter" / "draft_step_20"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dflash",
                "target_layer_ids": [1, 9, 17],
                "mask_token_id": 151669,
            }
        ),
        encoding="utf-8",
    )
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    (draft_dir / "metadata.json").write_text(
        json.dumps({"step": 20}), encoding="utf-8"
    )
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(
        "20", encoding="utf-8"
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "speculative_algorithm": "DFLASH",
                        "model_path": str(source_dir),
                        "checkpoint_path": None,
                    }
                }
            },
            "trainer": {"default_local_dir": str(checkpoint_dir)},
        }
    )

    assert _restore_drafter_model_path_from_checkpoint(config) == str(draft_dir)
    runtime_config = json.loads(
        (draft_dir / "config.json").read_text(encoding="utf-8")
    )
    training_config = json.loads(
        (draft_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )

    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
    assert training_config["model_type"] == "dflash"
    assert (
        config.actor_rollout_ref.rollout.drafter.runtime_config_source_model_path
        == str(source_dir)
    )

    # A retention process may remove the checkpoint selected at startup before
    # the next save. Export must still use the stable original model metadata.
    shutil.rmtree(draft_dir)
    next_draft_dir = checkpoint_dir / "drafter" / "draft_step_30"
    next_draft_dir.mkdir()
    (next_draft_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dflash",
                "target_layer_ids": [1, 9, 17],
                "mask_token_id": 151669,
            }
        ),
        encoding="utf-8",
    )

    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dflash")
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "model_path": str(draft_dir),
                    "runtime_config_source_model_path": str(source_dir),
                }
            }
        }
    )
    trainer._rewrite_exported_block_runtime_config(str(next_draft_dir))

    next_runtime_config = json.loads(
        (next_draft_dir / "config.json").read_text(encoding="utf-8")
    )
    assert next_runtime_config["model_type"] == "qwen3"
