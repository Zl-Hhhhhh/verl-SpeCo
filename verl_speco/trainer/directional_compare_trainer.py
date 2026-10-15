"""Single-cycle paired actor comparison of matched drafter anchor selections."""

import hashlib
import json
import os
from pathlib import Path
import time

from verl_speco.trainer.entropy_lr_runtime import _prompt_json_default
from verl_speco.trainer.frozen_actor_lr_trainer import FrozenActorLRTrainer
from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer


class DirectionalCompareTrainer(SpecoRayPPOTrainer):
    _ablation_record = FrozenActorLRTrainer._ablation_record
    _ablation_probe = FrozenActorLRTrainer._ablation_probe
    _ablation_crossed = FrozenActorLRTrainer._ablation_crossed
    _ablation_wake_rollout = FrozenActorLRTrainer._ablation_wake_rollout

    def fit(self):
        cfg = self._speco_drafter_training_config()
        if (not cfg.get("candidate_gate_enabled")
                or cfg.get("lr_scheduler_type") != "constant"
                or int(cfg.get("lr_warmup_steps", 0))):
            raise ValueError("Directional comparison needs candidate gate and constant LR")
        if os.environ.get("SPECO_VLLM_VERIFY_ENTROPY", "").lower() != "true":
            raise ValueError("Directional comparison needs verification entropy traces")
        return super().fit()

    def _get_gen_batch(self, batch):
        prompts = batch.non_tensor_batch.get("raw_prompt")
        if prompts is not None:
            if not hasattr(self, "_ablation_generation_cohorts"):
                self._ablation_generation_cohorts = []
            self._ablation_generation_cohorts.append([
                json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                           default=_prompt_json_default)
                for prompt in prompts
            ])
        return super()._get_gen_batch(batch)

    def _speco_directional_finish_pair(self):
        started = time.perf_counter()
        result = super()._speco_directional_finish_pair()
        self._directional_pair_seconds = time.perf_counter() - started
        return result

    def _speco_train_drafter(self):
        cfg = self._speco_drafter_training_config()
        if int(self.global_steps) != 5 or int(cfg.training_interval_steps) != 5:
            raise ValueError("Directional comparison must stop at actor step 5")
        if (float(cfg.dspark_first_rejection_ce_alpha) != 0.1
                or float(cfg.dspark_accepted_k3_alpha) != 1.0
                or float(cfg.dspark_rejected_k3_alpha) != 0.0
                or float(cfg.lr) != 1e-5):
            raise ValueError("Directional comparison requires response-K3 + 0.1 first CE, LR=1e-5")
        self._ablation_root = Path(cfg.frozen_actor_lr_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        if (self._ablation_root / "results.jsonl").exists():
            raise FileExistsError("Directional comparison output already exists")
        encode = lambda prompt: json.dumps(
            prompt, sort_keys=True, ensure_ascii=False, default=_prompt_json_default
        )
        training = {
            prompt for group in getattr(self, "_ablation_generation_cohorts", [])
            for prompt in group
        }
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        validation = {encode(prompt) for prompt in cohort.non_tensor_batch["raw_prompt"]}
        if not training or validation.intersection(training):
            raise RuntimeError("Training prompts are missing or overlap held-out validation")
        self._ablation_record(
            "design", experiment="paired_directional_anchor_compare",
            actor_reference_steps=[0, 5], frozen_actor_step=5,
            drafter=str(self.config.actor_rollout_ref.rollout.drafter.model_path),
            arms=["random", "top_tv", "top_m"], training_steps_per_arm=5,
            lr=float(cfg.lr), selection_seed=int(cfg.dspark_directional_compare_seed),
            response_loss="response_k3_plus_0.1_first_rejection_ce",
            pair_seconds=self._directional_pair_seconds,
            training_prompt_sha256=sorted(hashlib.sha256(row.encode()).hexdigest() for row in training),
            validation_prompt_sha256=sorted(hashlib.sha256(row.encode()).hexdigest() for row in validation),
            training_validation_overlap=0,
            quality_probe=dict(responses=len(cohort), max_tokens=512),
            throughput_probe=dict(responses=320, max_tokens=2048),
        )
        group = self._require_speco_worker_group()
        started = time.perf_counter()
        prepared = self._ray_get_if_needed(group.ablation_prepare())
        prepare_seconds = time.perf_counter() - started
        if len(prepared) != 16 or {row["rank"] for row in prepared} != set(range(16)):
            raise RuntimeError("Directional comparison requires all 16 drafter ranks")
        by_rank = {row["rank"]: row for row in prepared}
        for base in range(0, 16, 4):
            reference = by_rank[base]
            for rank in range(base + 1, base + 4):
                current = by_rank[rank]
                if (current["pool_sample_ids"] != reference["pool_sample_ids"]
                        or current["selection_hash"] != reference["selection_hash"]
                        or current["candidate_count"] != reference["candidate_count"]
                        or any(current["arms"][arm]["batch_anchor_counts"]
                               != reference["arms"][arm]["batch_anchor_counts"]
                               for arm in ("random", "top_tv", "top_m"))):
                    raise RuntimeError(f"SP group {base} has divergent directional candidates")
        self._ablation_record("prepared", seconds=prepare_seconds, workers=prepared)
        self._ablation_wake_rollout(cfg)
        self._ablation_probe("warmup_discard", performance=True)
        self._ablation_probe("baseline_AA_0", performance=True)
        self._ablation_probe("baseline_AA_1", performance=True)
        self.checkpoint_manager.sleep_replicas()
        for arm in ("random", "top_tv", "top_m"):
            started = time.perf_counter()
            trained = self._ray_get_if_needed(group.ablation_train(
                float(cfg.lr), 5, start=True, sampling_arm=arm, artifact_tag=arm,
            ))
            self._ablation_record("trained", arm=arm, steps=5,
                                  rpc_seconds=time.perf_counter() - started,
                                  workers=trained)
            self._ablation_wake_rollout(cfg)
            self._ablation_crossed(arm, 5, performance=True)
            self.checkpoint_manager.sleep_replicas()
            self._speco_finalize_candidate(self.global_steps, 0.0)
            restored = self._ray_get_if_needed(group.ablation_check_restored())
            if len(restored) != 16 or not all(row["ok"] for row in restored):
                raise RuntimeError(f"Directional arm {arm} did not restore the initial drafter")
            self._ablation_record("restored", arm=arm, workers=restored)
        self._ablation_record("completed", actor_updates=5,
                              arms=["random", "top_tv", "top_m"],
                              optimizer_steps_per_arm=5)
        return False, {"drafter/directional_compare_complete": 1}
