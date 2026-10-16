"""Paired actor-cycle diagnostic without drafter optimizer updates."""

import hashlib
import json
from pathlib import Path

from verl_speco.trainer.directional_compare_trainer import DirectionalCompareTrainer
from verl_speco.trainer.entropy_lr_runtime import _prompt_json_default


class DirectionalDiagnosticTrainer(DirectionalCompareTrainer):
    def _speco_train_drafter(self):
        cfg = self._speco_drafter_training_config()
        if int(self.global_steps) != 5 or int(cfg.training_interval_steps) != 5:
            raise ValueError("Directional diagnostic requires actor0-to-actor5 pairing")
        root = Path(str(cfg.dspark_directional_diagnostic_output))
        candidate_files = sorted(root.glob("candidates_rank*.jsonl"))
        sample_files = sorted(root.glob("paired_rank*.pt"))
        rollout = self.config.actor_rollout_ref.rollout
        trainer_cfg = self.config.trainer
        worker_world_size = int(cfg.get("worker_world_size") or (
            int(trainer_cfg.n_gpus_per_node) * int(trainer_cfg.nnodes)
        ))
        sp_size = int(self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size)
        if worker_world_size <= 0 or sp_size <= 0 or worker_world_size % sp_size:
            raise ValueError("Directional diagnostic has an invalid worker/SP topology")
        leader_ranks = list(range(0, worker_world_size, sp_size))
        expected_files = {f"candidates_rank{rank:02d}.jsonl" for rank in leader_ranks}
        if {path.name for path in candidate_files} != expected_files or not sample_files:
            raise RuntimeError("Directional diagnostic is missing SP-leader artifacts")
        candidates = [json.loads(line) for path in candidate_files
                      for line in path.read_text(encoding="utf-8").splitlines()]
        selected = [row for row in candidates if row.get("selected")]
        if not len(sample_files) <= len(selected) <= len(sample_files) * int(cfg.dspark_num_anchors):
            raise RuntimeError("Directional diagnostic selected anchor count is incomplete")
        training = {
            prompt for cohort in getattr(self, "_ablation_generation_cohorts", [])
            for prompt in cohort
        }
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        validation = {
            json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                       default=_prompt_json_default)
            for prompt in cohort.non_tensor_batch["raw_prompt"]
        }
        if not training or training.intersection(validation):
            raise RuntimeError("Directional diagnostic train/validation prompts overlap")
        train_files = self.config.data.train_files
        train_paths = [train_files] if isinstance(train_files, str) else list(train_files)
        source_hashes = {}
        for raw_path in train_paths:
            path = Path(str(raw_path))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            source_hashes[str(path)] = digest.hexdigest()
        manifest = {
            "experiment": "directional_repair_target_diagnostic",
            "actor_reference_steps": [0, 5],
            "drafter_optimizer_steps": 0,
            "source_train_sha256": source_hashes,
            "actor_model": str(self.config.actor_rollout_ref.model.path),
            "drafter_model": str(self.config.actor_rollout_ref.rollout.drafter.model_path),
            "seed": self.config.data.seed,
            "worker_world_size": worker_world_size,
            "sequence_parallel_size": sp_size,
            "tensor_parallel_size": int(rollout.tensor_model_parallel_size),
            "independent_data_replicas": len(leader_ranks),
            "sp_leader_ranks": leader_ranks,
            "max_collect_tokens_per_replica": int(cfg.max_collect_tokens_per_step_per_replica),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_sample_files": len(sample_files),
            "independent_training_prompts": len(training),
            "validation_prompts": len(validation),
            "training_validation_overlap": 0,
            "training_prompt_sha256": sorted(hashlib.sha256(row.encode()).hexdigest() for row in training),
            "validation_prompt_sha256": sorted(hashlib.sha256(row.encode()).hexdigest() for row in validation),
            "pair_seconds": self._directional_pair_seconds,
            "window": int(cfg.dspark_max_window),
            "proposal_method": str(self.config.actor_rollout_ref.rollout.drafter.get(
                "draft_sample_method", "greedy")),
            "deploy_sampling": {"temperature": 1.0, "top_k": -1, "top_p": 1.0,
                                 "repetition_penalty": 1.0},
            "probability_dtype": "float32",
            "probability_tolerance": 1e-5,
            "notes": "p_plus is raw softmax; launch must verify deployment generation override matches deploy_sampling",
        }
        output = root / "paired_panel_manifest.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return False, {"drafter/directional_diagnostic_complete": 1,
                       "drafter/directional_diagnostic_selected": len(selected)}
