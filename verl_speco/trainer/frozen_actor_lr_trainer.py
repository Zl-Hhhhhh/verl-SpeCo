"""Frozen actor, matched drafter LR arms, and crossed-order rollout probes."""
import gzip
import hashlib
import itertools
import json
import logging
from pathlib import Path
import time

import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl_speco.integration.rollout_publish import resolve_drafter_publish_payload
from verl_speco.integration.vllm_runtime import SPECO_VLLM_SPECULATIVE_TRACE_KEY
from verl_speco.trainer.entropy_lr import trace_statistics
from verl_speco.trainer.entropy_lr_runtime import _prompt_json_default
from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

logger = logging.getLogger(__name__)


class _SkipInitialBatches:
    def __init__(self, dataloader, count):
        if count < 0 or count >= len(dataloader):
            raise ValueError(f"Invalid training batch skip count {count}")
        self.dataloader = dataloader
        self.count = count

    def __len__(self):
        return len(self.dataloader) - self.count

    def __iter__(self):
        return itertools.islice(iter(self.dataloader), self.count, None)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)


class FrozenActorLRTrainer(SpecoRayPPOTrainer):
    def fit(self):
        cfg = self._speco_drafter_training_config()
        if cfg.get("frozen_target_repeat_output"):
            return self._compare_frozen_target_repeat()
        if cfg.get("frozen_full_eval_output"):
            return self._compare_frozen_full_eval()
        if cfg.get("frozen_drafter_compare_output"):
            return self._compare_frozen_drafter()
        cached = cfg.get("frozen_actor_lr_cached_source")
        if not cached:
            if cfg.get("frozen_actor_multiround_resume_state_dir"):
                skipped = int(cfg.get("frozen_actor_multiround_resume_skip_batches", 0))
                if skipped != 2:
                    raise ValueError("Third-round continuation must skip two training batches")
                self.train_dataloader = _SkipInitialBatches(self.train_dataloader, skipped)
            return super().fit()
        self.global_steps = 1
        self._speco_entropy_lr().prepare_validation_cohort()
        self._speco_set_drafter_global_step()
        self._speco_sync_target_lm_head_weight()
        return self._speco_train_drafter()

    def _compare_frozen_full_eval(self):
        cfg = self._speco_drafter_training_config()
        output_dir = Path(cfg.frozen_full_eval_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "metrics.json"
        if result_path.exists():
            raise FileExistsError(f"Full evaluation already exists: {result_path}")
        expected = int(cfg.frozen_full_eval_expected_count)
        actual = len(self.val_dataset)
        if actual != expected:
            raise RuntimeError(f"Full evaluation expected {expected} problems, got {actual}")
        self.global_steps = 0
        self.checkpoint_manager.wake_up_replicas()
        with self._speco_rollout_metrics_fit_hook():
            metrics = self._validate()
        self._shutdown_dump_executor()
        if not isinstance(metrics, dict) or not metrics:
            raise RuntimeError("Full evaluation returned no metrics")
        required = (
            "val-aux/drafter/spec_decode/mean_acceptance_length",
            "val-aux/drafter/spec_decode/response_tokens_per_second",
        )
        missing = [key for key in required if key not in metrics]
        if missing:
            raise RuntimeError(f"Full evaluation missing speculative decoding metrics: {missing}")
        result = {
            "actor": str(self.config.actor_rollout_ref.model.path),
            "drafter": str(self.config.actor_rollout_ref.rollout.drafter.model_path),
            "problem_count": actual,
            "actor_updates": 0,
            "drafter_updates": 0,
            "metrics": metrics,
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(
            result, indent=2, sort_keys=True, default=_prompt_json_default
        ) + "\n")
        temporary.replace(result_path)
        return False, {"drafter/frozen_full_eval_complete": 1}

    def _compare_frozen_target_repeat(self):
        cfg = self._speco_drafter_training_config()
        output_dir = Path(cfg.frozen_target_repeat_output)
        if len(self.val_dataset) != 64 or self.config.actor_rollout_ref.rollout.drafter.enable:
            raise ValueError("Target-only repeat requires 64 prompts and disabled drafter")
        if output_dir.exists():
            raise FileExistsError(f"Target-only repeat output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        self.config.trainer.validation_data_dir = str(output_dir / "generations")
        self.checkpoint_manager.wake_up_replicas()
        metrics = []
        for step in (0, 1):
            self.global_steps = step
            result = self._validate()
            self._shutdown_dump_executor()
            path = output_dir / "generations" / f"{step}.jsonl"
            if not path.exists() or sum(1 for _ in path.open()) != 64:
                raise RuntimeError(f"Incomplete target-only probe: {path}")
            metrics.append(result)
            if step == 0:
                self._init_dump_executor()
        (output_dir / "metrics.json").write_text(json.dumps(dict(
            actor=str(self.config.actor_rollout_ref.model.path),
            problem_count=64, actor_updates=0, drafter_updates=0,
            repeats=metrics,
        ), indent=2, sort_keys=True, default=_prompt_json_default) + "\n")
        return False, {"drafter/frozen_target_repeat_complete": 1}

    def _get_gen_batch(self, batch):
        prompts = batch.non_tensor_batch.get("raw_prompt")
        if prompts is not None:
            if not hasattr(self, "_ablation_generation_cohorts"):
                self._ablation_generation_cohorts = []
            self._ablation_generation_cohorts.append([
                json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=_prompt_json_default)
                for prompt in prompts])
        return super()._get_gen_batch(batch)

    def _update_actor(self, *args, **kwargs):
        # Intentionally never call actor.update_actor, even with lr=0: Adam,
        # weight decay, and optimizer counters must not change the frozen actor.
        return DataProto(meta_info={"metrics": {"actor/frozen_for_lr_ablation": 1}})

    def _save_checkpoint(self, *args, **kwargs):
        # Frozen actor is already stored as HF weights. Arm weights are saved
        # separately, before exact restoration of the common drafter baseline.
        return None

    def _speco_save_drafter_checkpoint(self, **kwargs):
        return None

    def _ablation_record(self, event, **values):
        record = dict(event=event, time=time.time(), **values)
        with (self._ablation_root / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=_prompt_json_default) + "\n")
        logger.warning("[frozen actor LR] %s", record if event != "probe" else
                       {key: value for key, value in record.items() if key != "per_prompt"})

    def _ablation_probe(self, label, *, performance=False):
        cohort = self._speco_entropy_lr().validation_batch(64)
        if performance:
            # Same held-out prompts repeated to measure load at the original
            # 320-response concurrency. Repeats are NOT independent prompts.
            cohort = cohort[[index % len(cohort) for index in range(320)]]
        cap = 2048 if performance else 512
        cohort.meta_info = dict(cohort.meta_info, speco_candidate_gate_max_tokens=cap)
        workers = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        padded, padding = pad_dataproto_to_divisor(cohort, workers)
        generate = self._speco_rollout_generation_target().generate_sequences
        output, timing = self._speco_measure_candidate_rollout(generate, (padded,), {})
        # Record true dispatched token throughput, then unpad statistical rows.
        output = unpad_dataproto(output, pad_size=padding)
        mask = output.batch["response_mask"]
        lengths = mask.sum(-1).cpu().tolist()
        if not lengths or max(lengths) > cap or min(lengths) <= 0:
            raise RuntimeError(f"Validation response cap violation: {lengths}")
        traces = output.non_tensor_batch.get(SPECO_VLLM_SPECULATIVE_TRACE_KEY)
        if traces is None or len(traces) != len(cohort):
            raise RuntimeError("Missing ablation verification traces")
        stats = trace_statistics(traces)
        ids = [row[row_mask.bool()].cpu().tolist()
               for row, row_mask in zip(output.batch["responses"], mask, strict=True)]
        expected_prefix = self._speco_drafter_training_config().get(
            "frozen_drafter_compare_expected_prefix"
        )
        if expected_prefix is not None:
            expected_prefix = [int(token) for token in expected_prefix]
            mismatches = [index for index, row in enumerate(ids)
                          if row[:len(expected_prefix)] != expected_prefix]
            if mismatches:
                raise RuntimeError(
                    f"Frozen actor output prefix mismatch for {len(mismatches)}/{len(ids)} "
                    f"responses; expected={expected_prefix}, first={ids[mismatches[0]][:len(expected_prefix)]}"
                )
        record = dict(label=label, performance=performance, max_tokens=cap,
                      unique_prompts=len(self._speco_entropy_lr().cohort), responses=len(cohort),
                      padded_responses=padding, response_lengths=lengths,
                      response_hashes=[hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in ids],
                      **{**timing, **stats})
        # Metrics expose both the dispatched and unpadded denominator explicitly.
        record["tokens_per_second_dispatched"] = timing["response_tokens"] / timing["elapsed_sec"]
        record["response_tokens_unpadded"] = sum(lengths)
        with gzip.open(self._ablation_root / f"{label}.json.gz", "wt") as stream:
            json.dump(dict(**record, traces=list(traces), token_ids=ids), stream,
                      default=_prompt_json_default)
        self._ablation_record("probe", **record)
        return record

    def _ablation_crossed(self, arm, steps, *, performance=False):
        prefix = f"{arm}_step_{steps}_{'load320' if performance else 'quality'}"
        # ABBA: symmetric positions for each model, retaining all per-prompt
        # measurements rather than using a one-shot latency gate.
        for index, alpha in enumerate((0.0, 1.0, 1.0, 0.0)):
            publish_seconds = self._speco_publish_candidate_alpha(self.global_steps, alpha)
            label = f"{prefix}_{index}_{'B' if alpha else 'A'}"
            self._ablation_record("publish", label=label, seconds=publish_seconds)
            self._ablation_probe(label, performance=performance)

    def _ablation_wake_rollout(self, cfg):
        if cfg.get("frozen_actor_fkl_factorial", False) or cfg.get("frozen_actor_response_objective_compare", False):
            self.checkpoint_manager.wake_up_replicas()
        else:
            self.checkpoint_manager.update_weights(self.global_steps)

    def _compare_frozen_drafter(self):
        cfg = self._speco_drafter_training_config()
        self._ablation_root = Path(cfg.frozen_drafter_compare_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        if (self._ablation_root / "results.jsonl").exists():
            raise RuntimeError("Use a fresh comparison output directory for each attempt")
        self.global_steps = 1
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        prompts = [json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                              default=_prompt_json_default)
                   for prompt in cohort.non_tensor_batch["raw_prompt"]]
        self._ablation_record(
            "design", actor=self.config.actor_rollout_ref.model.path,
            drafter=self.config.actor_rollout_ref.rollout.drafter.model_path,
            actor_updates=0, drafter_updates=0,
            validation_prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest()
                                      for prompt in prompts],
            quality_probe=dict(responses=len(cohort), max_tokens=512, temperature=0),
            concurrency_probe=dict(responses=320, max_tokens=2048, temperature=0),
        )
        self.checkpoint_manager.wake_up_replicas()
        self._ablation_probe("warmup_discard")
        for index in range(2):
            self._ablation_probe(f"quality_{index}")
        for index in range(2):
            self._ablation_probe(f"load320_{index}", performance=True)
        self._ablation_record("completed", actor_updates=0, drafter_updates=0)
        return False, {"drafter/frozen_compare_complete": 1}

    def _speco_train_drafter(self):
        cfg = self._speco_drafter_training_config()
        if cfg.get("frozen_actor_sampling_multiround_enabled", False):
            return self._speco_train_drafter_sampling_multiround()
        if cfg.get("frozen_actor_sampling_compare_enabled", False):
            return self._speco_train_drafter_sampling_compare()
        if cfg.get("frozen_actor_nested_caps_enabled", False):
            return self._speco_train_drafter_nested()
        self._ablation_root = Path(cfg.frozen_actor_lr_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        if (self._ablation_root / "results.jsonl").exists():
            raise RuntimeError("Use a fresh ablation output directory for each attempt")
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        prompts = cohort.non_tensor_batch["raw_prompt"]
        validation_prompts = {json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                        default=_prompt_json_default) for prompt in prompts}
        cached = cfg.get("frozen_actor_lr_cached_source")
        if cached:
            records = [json.loads(line) for line in (Path(cached) / "results.jsonl").read_text().splitlines()]
            source = next(row for row in records if row["event"] == "design")
            source_prompts = {json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                        default=_prompt_json_default) for prompt in source["validation_prompts"]}
            if source["actor"] != self.config.actor_rollout_ref.model.path or source_prompts != validation_prompts or source["training_validation_overlap"] != 0:
                raise RuntimeError("Cached experiment actor/validation provenance mismatch")
            training_hashes = source["training_prompt_sha256"]
        else:
            training_prompts = self._ablation_generation_cohorts[-1]
            overlap = set(training_prompts).intersection(validation_prompts)
            if overlap:
                raise RuntimeError(f"Training/validation overlap: {len(overlap)} prompts")
            training_hashes = [hashlib.sha256(p.encode()).hexdigest() for p in training_prompts]
        configured_ce = cfg.get("frozen_actor_ce_alphas")
        ce_alphas = [float(value) for value in configured_ce] if configured_ce is not None else None
        configured_prefixes = cfg.get("frozen_actor_rejected_prefix_lengths")
        rejected_prefixes = (
            [int(value) for value in configured_prefixes]
            if configured_prefixes is not None else None
        )
        fkl_factorial = bool(cfg.get("frozen_actor_fkl_factorial", False))
        response_objective_compare = bool(cfg.get("frozen_actor_response_objective_compare", False))
        rejected_aux_alpha = float(cfg.get("frozen_actor_rejected_aux_alpha", 0.1))
        rejected_prefix_length = int(cfg.get("frozen_actor_rejected_prefix_length", 7))
        if sum((ce_alphas is not None, rejected_prefixes is not None,
                fkl_factorial, response_objective_compare)) > 1:
            raise ValueError("Choose exactly one frozen-actor objective grid")
        if response_objective_compare:
            if (float(cfg.get("dspark_accepted_k3_alpha", 1.0)) != 1.0
                    or float(cfg.get("dspark_accepted_fkl_alpha", 0.0)) != 0.0
                    or float(cfg.get("dspark_rejected_k3_alpha", 1.0)) != 0.0
                    or float(cfg.get("dspark_rejected_stream_weight", 1.0)) != 0.0
                    or float(cfg.get("dspark_rejected_aux_alpha", 0.0)) != 0.0
                    or float(cfg.get("dspark_response_stream_weight", 1.0)) != 1.0
                    or float(cfg.get("dspark_ce_loss_alpha", 0.0)) != 0.0
                    or float(cfg.get("dspark_l1_loss_alpha", 0.0)) != 0.0
                    or not bool(cfg.get("dspark_trace_aligned_anchors", False))):
                raise ValueError("Objective comparison requires pure accepted loss and trace-aligned first-rejection CE")
        if fkl_factorial:
            if (float(cfg.get("dspark_accepted_k3_alpha", 1.0)) != 0.0
                    or float(cfg.get("dspark_accepted_fkl_alpha", 0.0)) != 1.0
                    or float(cfg.get("dspark_rejected_k3_alpha", 1.0)) != 0.0
                    or float(cfg.get("dspark_rejected_stream_weight", 1.0)) != 0.0
                    or float(cfg.get("dspark_ce_loss_alpha", 0.0)) != 0.0
                    or float(cfg.get("dspark_l1_loss_alpha", 0.0)) != 0.0
                    or rejected_aux_alpha <= 0
                    or rejected_prefix_length != int(cfg.get("dspark_block_size", 7))):
                raise ValueError("FKL factorial requires pure accepted FKL and full-branch independent rejected K3")
        configured_lrs = cfg.get("frozen_actor_lr_values")
        if configured_lrs is not None:
            lrs = [float(value) for value in configured_lrs]
        elif ce_alphas is not None or rejected_prefixes is not None or fkl_factorial or response_objective_compare:
            lrs = [1e-5]
        else:
            lrs = [5e-6, 1e-5, 1.5e-5]
        if not lrs or len(set(lrs)) != len(lrs) or any(
            not np.isfinite(value) or value <= 0 for value in lrs
        ):
            raise ValueError(f"Frozen-actor learning rates must be distinct and positive: {lrs}")
        checkpoints = [int(value) for value in cfg.get("frozen_actor_checkpoints", [5, 20])]
        if (
            not checkpoints
            or checkpoints != sorted(set(checkpoints))
            or checkpoints[0] <= 0
            or checkpoints[-1] > 20
        ):
            raise ValueError(
                "Frozen-actor checkpoints must be distinct, increasing, and within the 20 cached batches"
            )
        performance_checkpoints = [
            int(value)
            for value in cfg.get("frozen_actor_performance_checkpoints", [checkpoints[-1]])
        ]
        if (
            len(set(performance_checkpoints)) != len(performance_checkpoints)
            or not set(performance_checkpoints).issubset(checkpoints)
        ):
            raise ValueError("Performance checkpoints must be a distinct subset of checkpoints")
        experiment = (
            "response_k3_vs_accepted_fkl_ce01" if response_objective_compare else
            "accepted_fkl_factorial" if fkl_factorial else
            ("rejected_prefix" if rejected_prefixes is not None else
             ("first_rejection_ce" if ce_alphas is not None else "learning_rate"))
        )
        self._ablation_record("design", actor=self.config.actor_rollout_ref.model.path,
                              experiment=experiment,
                              lrs=lrs, first_rejection_ce_alphas=ce_alphas,
                              checkpoints=checkpoints,
                              performance_checkpoints=performance_checkpoints,
                              rejected_prefix_lengths=rejected_prefixes,
                              rejected_aux_alpha=rejected_aux_alpha if rejected_prefixes is not None or fkl_factorial else 0.0,
                              rejected_prefix_length=rejected_prefix_length if fkl_factorial else None,
                              objective_arms=(
                                  [{"name": "response_k3_ce01", "accepted_k3_alpha": 1.0, "accepted_fkl_alpha": 0.0},
                                   {"name": "accepted_fkl_ce01", "accepted_k3_alpha": 0.0, "accepted_fkl_alpha": 1.0}]
                                  if response_objective_compare else None),
                              accepted_k3_alpha=float(cfg.get("dspark_accepted_k3_alpha", 1.0)),
                              accepted_fkl_alpha=float(cfg.get("dspark_accepted_fkl_alpha", 0.0)),
                              rejected_k3_alpha=float(cfg.get("dspark_rejected_k3_alpha", 1.0)),
                              response_stream_weight=float(cfg.get("dspark_response_stream_weight", 1.0)),
                              rejected_stream_weight=float(cfg.get("dspark_rejected_stream_weight", 1.0)),
                              actor_updates=0, validation_prompts=list(prompts),
                              training_prompt_sha256=training_hashes, cached_source=cached,
                              training_validation_overlap=0,
                              concurrency_probe=dict(responses=320, max_tokens=2048, temperature=0))
        group = self._require_speco_worker_group()
        prepared = self._ray_get_if_needed(group.ablation_prepare())
        self._ablation_record("prepared", workers=prepared)
        self._ablation_wake_rollout(cfg)
        self._ablation_probe("warmup_discard")
        self._ablation_probe("baseline_AA_0")
        self._ablation_probe("baseline_AA_1")
        self.checkpoint_manager.sleep_replicas()
        multiple_lrs = len(lrs) > 1
        if response_objective_compare:
            arms = [(lr, 0.1, 0.0, 0, name + (f"_lr_{lr:.1e}" if multiple_lrs else ""))
                    for name in ("response_k3_ce01", "accepted_fkl_ce01") for lr in lrs]
        elif fkl_factorial:
            arms = [
                (lr, ce, aux, rejected_prefix_length if aux else 0,
                 f"fkl_ce_{ce:.1f}_reject_{aux:.1f}" +
                 (f"_lr_{lr:.1e}" if multiple_lrs else ""))
                for ce in (0.0, 0.1)
                for aux in (0.0, rejected_aux_alpha)
                for lr in lrs
            ]
        elif rejected_prefixes is not None:
            arms = [
                (lr, 0.1, 0.0 if prefix == 0 else rejected_aux_alpha, prefix,
                 ("baseline_ce01" if prefix == 0 else f"rejected_prefix_{prefix}")
                 + (f"_lr_{lr:.1e}" if multiple_lrs else ""))
                for prefix in rejected_prefixes
                for lr in lrs
            ]
        elif ce_alphas is not None:
            arms = [
                (lr, ce, None, None,
                 f"ce_{ce:.2f}" + (f"_lr_{lr:.1e}" if multiple_lrs else ""))
                for ce in ce_alphas
                for lr in lrs
            ]
        else:
            arms = [(lr, None, None, None, f"lr_{lr:.1e}") for lr in lrs]
        for lr, ce_alpha, aux_alpha, prefix_length, arm in arms:
            accepted_alphas = (
                (1.0, 0.0) if arm.startswith("response_k3_ce01") else (0.0, 1.0)
            ) if response_objective_compare else None
            for steps in checkpoints:
                started = time.perf_counter()
                if ce_alpha is None:
                    pending = group.ablation_train(lr, steps, start=(steps == checkpoints[0]))
                else:
                    pending = group.ablation_train(
                        lr, steps, start=(steps == checkpoints[0]),
                        first_rejection_ce_alpha=ce_alpha,
                        rejected_aux_alpha=aux_alpha,
                        rejected_prefix_length=prefix_length,
                        artifact_tag=arm,
                        **({"accepted_k3_alpha": accepted_alphas[0],
                            "accepted_fkl_alpha": accepted_alphas[1]}
                           if accepted_alphas is not None else {}))
                results = self._ray_get_if_needed(pending)
                self._ablation_record("trained", lr=lr, first_rejection_ce_alpha=ce_alpha,
                                      arm=arm, steps=steps,
                                      rejected_aux_alpha=aux_alpha,
                                      rejected_prefix_length=prefix_length,
                                      accepted_k3_alpha=accepted_alphas[0] if accepted_alphas else None,
                                      accepted_fkl_alpha=accepted_alphas[1] if accepted_alphas else None,
                                      rpc_seconds=time.perf_counter()-started, workers=results)
                self._ablation_wake_rollout(cfg)
                self._ablation_crossed(arm, steps)
                if steps in performance_checkpoints:
                    self._ablation_crossed(arm, steps, performance=True)
                self.checkpoint_manager.sleep_replicas()
            self._speco_finalize_candidate(self.global_steps, 0.0)
            restored = self._ray_get_if_needed(group.ablation_check_restored())
            self._ablation_record("restored", lr=lr, first_rejection_ce_alpha=ce_alpha,
                                  rejected_aux_alpha=aux_alpha,
                                  rejected_prefix_length=prefix_length,
                                  arm=arm, workers=restored)
        self._ablation_record("completed", actor_updates=0, arms=len(arms),
                              checkpoints_per_arm=len(checkpoints),
                              performance_checkpoints=performance_checkpoints)
        return False, {"drafter/frozen_lr_ablation_complete": 1}

    def _speco_train_drafter_nested(self):
        cfg = self._speco_drafter_training_config()
        caps = [int(cap) for cap in cfg.frozen_actor_nested_caps]
        if caps != [16384, 32768]:
            raise ValueError(f"Expected the paired 16384/32768 caps, got {caps}")
        if int(cfg.get("max_collect_tokens_per_step_per_replica", 0)) != caps[-1]:
            raise ValueError("The rollout collector must use the high nested cap")
        self._ablation_root = Path(cfg.frozen_actor_lr_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        if (self._ablation_root / "results.jsonl").exists():
            raise RuntimeError("Use a fresh output directory for each nested rollout")
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        prompts = cohort.non_tensor_batch["raw_prompt"]
        encode = lambda prompt: json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                            default=_prompt_json_default)
        validation = {encode(prompt) for prompt in prompts}
        training = self._ablation_generation_cohorts[-1]
        if set(training).intersection(validation):
            raise RuntimeError("Nested experiment training and validation prompts overlap")
        self._ablation_record(
            "design", experiment="nested_collection_caps", actor=self.config.actor_rollout_ref.model.path,
            actor_updates=0, rollout_seed=int(self.config.data.seed),
            selection_seed=int(cfg.frozen_actor_nested_seed), caps=caps,
            lr=float(cfg.lr), checkpoints=[0, 3, 5],
            validation_prompt_sha256=[hashlib.sha256(encode(prompt).encode()).hexdigest()
                                      for prompt in prompts],
            training_prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest()
                                    for prompt in training],
            training_validation_overlap=0,
            response_loss="response_k3_plus_0.1_first_rejection_ce",
            collection_scope="post_fixed_eval_training_pool",
        )
        group = self._require_speco_worker_group()
        prepared = self._ray_get_if_needed(group.ablation_prepare())
        if len(prepared) != 16:
            raise RuntimeError(f"Expected 16 prepared workers, got {len(prepared)}")
        by_rank = {row["rank"]: row for row in prepared}
        if set(by_rank) != set(range(16)):
            raise RuntimeError("Nested worker ranks are missing or duplicated")
        for base in range(0, 16, 4):
            for cap in caps:
                reference = by_rank[base]["caps"][cap]
                ids = set(reference["sample_ids"])
                if not ids or len(ids) != len(reference["sample_ids"]):
                    raise RuntimeError(f"Empty or duplicated cap {cap} pool on rank {base}")
                for peer in range(base + 1, base + 4):
                    current = by_rank[peer]["caps"][cap]
                    if current["sample_ids"] != reference["sample_ids"] or current["sample_rows"] != reference["sample_rows"]:
                        raise RuntimeError(f"SP group {base} disagrees on nested cap {cap}")
            low = set(by_rank[base]["caps"][caps[0]]["sample_ids"])
            high = set(by_rank[base]["caps"][caps[1]]["sample_ids"])
            if not low < high:
                raise RuntimeError(f"Nested cap pools are not strict on SP group {base}")
        self._ablation_record("prepared", workers=prepared)
        self._ablation_wake_rollout(cfg)
        self._ablation_probe("warmup_discard", performance=True)
        self._ablation_probe("baseline_AA_0", performance=True)
        self._ablation_probe("baseline_AA_1", performance=True)
        self.checkpoint_manager.sleep_replicas()
        for cap in caps:
            for step in (3, 5):
                started = time.perf_counter()
                trained = self._ray_get_if_needed(group.ablation_train(
                    float(cfg.lr), step, start=(step == 3), nested_cap=cap,
                    artifact_tag=f"cap_{cap}",
                ))
                self._ablation_record("trained", cap=cap, steps=step,
                                      rpc_seconds=time.perf_counter() - started, workers=trained)
                self._ablation_wake_rollout(cfg)
                self._ablation_crossed(f"cap_{cap}", step, performance=True)
                self.checkpoint_manager.sleep_replicas()
            self._speco_finalize_candidate(self.global_steps, 0.0)
            restored = self._ray_get_if_needed(group.ablation_check_restored())
            if len(restored) != 16 or not all(row["ok"] for row in restored):
                raise RuntimeError(f"Nested cap {cap} did not restore every worker")
            self._ablation_record("restored", cap=cap, workers=restored)
        self._ablation_record("completed", actor_updates=0, caps=caps, checkpoints=[0, 3, 5])
        return False, {"drafter/frozen_nested_caps_complete": 1}

    def _speco_train_drafter_sampling_compare(self):
        cfg = self._speco_drafter_training_config()
        if int(cfg.max_collect_tokens_per_step_per_replica) != 16384:
            raise ValueError("Sampling comparison requires a 16K rank-local collection cap")
        if float(cfg.dspark_first_rejection_ce_alpha) != 0.1 or float(cfg.lr) != 1e-5:
            raise ValueError("Sampling comparison requires CE=0.1 and drafter LR=1e-5")
        self._ablation_root = Path(cfg.frozen_actor_lr_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        if (self._ablation_root / "results.jsonl").exists():
            raise RuntimeError("Use a fresh output directory for each sampling rollout")
        if not getattr(self, "_ablation_generation_cohorts", None):
            raise RuntimeError("Sampling comparison has no recorded training rollout")
        training = tuple(self._ablation_generation_cohorts[-1])
        cohort = self._speco_entropy_lr().prepare_validation_cohort()
        encode = lambda prompt: json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                            default=_prompt_json_default)
        validation = {encode(prompt) for prompt in cohort.non_tensor_batch["raw_prompt"]}
        if set(training).intersection(validation):
            raise RuntimeError("Sampling comparison training and validation prompts overlap")
        self._ablation_record(
            "design", experiment="same_rollout_sampling", actor=self.config.actor_rollout_ref.model.path,
            actor_updates=0, rollout_seed=int(self.config.data.seed),
            drafter=self.config.actor_rollout_ref.rollout.drafter.model_path,
            lr=float(cfg.lr), optimizer_steps=5, collection_cap=16384,
            window=int(cfg.dspark_max_window),
            arms=["uniform", "low_entropy", "global_position_low_entropy"],
            training_prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest() for prompt in training],
            validation_prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest()
                                      for prompt in validation],
            training_validation_overlap=0,
        )
        group = self._require_speco_worker_group()
        prepared = self._ray_get_if_needed(group.ablation_prepare())
        if len(prepared) != 16 or {row["rank"] for row in prepared} != set(range(16)):
            raise RuntimeError("Sampling comparison did not prepare exactly 16 drafter ranks")
        by_rank = {row["rank"]: row for row in prepared}
        for base in range(0, 16, 4):
            reference = by_rank[base]
            for peer in range(base + 1, base + 4):
                current = by_rank[peer]
                if (current["pool_sample_ids"] != reference["pool_sample_ids"]
                        or current["pool_rows"] != reference["pool_rows"]
                        or current["global_plan"] != reference["global_plan"]):
                    raise RuntimeError(f"SP group {base} disagrees on the sampling pool or plan")
        self._ablation_record("prepared", workers=prepared)
        self._ablation_wake_rollout(cfg)
        self._ablation_probe("warmup_discard")
        self._ablation_probe("baseline_AA_0")
        self._ablation_probe("baseline_AA_1")
        self.checkpoint_manager.sleep_replicas()
        for arm in ("uniform", "low_entropy", "global_position_low_entropy"):
            started = time.perf_counter()
            trained = self._ray_get_if_needed(group.ablation_train(
                1e-5, 5, start=True, sampling_arm=arm, artifact_tag=arm,
            ))
            self._ablation_record("trained", arm=arm, steps=5,
                                  rpc_seconds=time.perf_counter() - started, workers=trained)
            self._ablation_wake_rollout(cfg)
            self._ablation_crossed(arm, 5)
            self._ablation_crossed(arm, 5, performance=True)
            self.checkpoint_manager.sleep_replicas()
            self._speco_finalize_candidate(self.global_steps, 0.0)
            restored = self._ray_get_if_needed(group.ablation_check_restored())
            if len(restored) != 16 or not all(row["ok"] for row in restored):
                raise RuntimeError(f"Sampling arm {arm} did not restore every worker")
            self._ablation_record("restored", arm=arm, workers=restored)
        self._ablation_record("completed", actor_updates=0, arms=3, checkpoints=[0, 5])
        return False, {"drafter/frozen_sampling_comparison_complete": 1}

    def _speco_train_drafter_sampling_multiround(self):
        cfg = self._speco_drafter_training_config()
        offset = int(cfg.get("frozen_actor_multiround_round_offset", 0))
        resume_root = cfg.get("frozen_actor_multiround_resume_state_dir")
        if (offset, bool(resume_root)) not in ((0, False), (2, True)):
            raise ValueError("Multiround resume requires round offset 2 and saved state directory")
        if (int(cfg.max_collect_tokens_per_step_per_replica) != 16384
                or float(cfg.dspark_first_rejection_ce_alpha) != 0.1
                or float(cfg.lr) != 1e-5
                or int(cfg.step) != 5):
            raise ValueError("Multiround sampling requires 16K, CE=0.1, LR=1e-5, five steps")
        if self.global_steps not in ((1, 2) if offset else (1, 2, 3, 4)):
            raise RuntimeError(f"Unexpected multiround actor step {self.global_steps}")
        self._ablation_root = Path(cfg.frozen_actor_lr_output)
        self._ablation_root.mkdir(parents=True, exist_ok=True)
        group = self._require_speco_worker_group()
        if self.global_steps == 1:
            if (self._ablation_root / "results.jsonl").exists():
                raise RuntimeError("Use a fresh multiround output directory")
            cohort = self._speco_entropy_lr().prepare_validation_cohort()
            prompts = [json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                  default=_prompt_json_default)
                       for prompt in cohort.non_tensor_batch["raw_prompt"]]
            validation_hashes = [hashlib.sha256(prompt.encode()).hexdigest()
                                 for prompt in prompts]
            if resume_root:
                previous = [json.loads(line) for line in
                            (Path(resume_root) / "results.jsonl").open()]
                prior_design = next(row for row in previous if row["event"] == "design")
                if prior_design["actor"] != str(self.config.actor_rollout_ref.model.path):
                    raise RuntimeError("Resumed actor differs from first two rounds")
                if prior_design["validation_prompt_sha256"] != validation_hashes:
                    raise RuntimeError("Resumed 64-prompt validation cohort differs")
            self._ablation_record(
                "design", experiment="continuous_same_source_rollout_sampling",
                actor=str(self.config.actor_rollout_ref.model.path),
                drafter=str(self.config.actor_rollout_ref.rollout.drafter.model_path),
                actor_updates=0, warmup_step=1,
                training_steps=([2] if offset else [2, 3, 4]),
                round_offset=offset, resumed_from=str(resume_root) if resume_root else None,
                skipped_training_batches=int(cfg.get("frozen_actor_multiround_resume_skip_batches", 0)),
                optimizer_steps_per_arm_per_round=5, collection_cap=16384,
                rollout_source="mature_drafter_fixed_across_rounds",
                arms=["uniform", "global_position_low_entropy"],
                validation_prompt_sha256=validation_hashes,
            )
            status = self._ray_get_if_needed(group.ablation_multiround_fixed_eval_status())
            if len(status) != 16 or any(row["samples"] != 4 for row in status):
                raise RuntimeError(f"Fixed validation pool was not filled on all ranks: {status}")
            self._ablation_record("fixed_validation_pool", workers=status,
                                  rollout=self._speco_current_step_rollout_quality_metrics())
            return False, {"drafter/frozen_multiround_warmup": 1}

        if not (self._ablation_root / "results.jsonl").exists():
            raise RuntimeError("Missing warmup and fixed validation pool")
        round_number = offset + self.global_steps - 1
        training = tuple(self._ablation_generation_cohorts[-1])
        if resume_root:
            previous = [json.loads(line) for line in (Path(resume_root) / "results.jsonl").open()]
            prior_round = next(row for row in previous if row["event"] == "round_start"
                               and row["round"] == round_number)
            hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in training]
            if hashes != prior_round["training_prompt_sha256"]:
                raise RuntimeError("Resumed third-round training prompt cohort differs")
        validation = {json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                 default=_prompt_json_default)
                      for prompt in self._speco_entropy_lr().cohort.non_tensor_batch["raw_prompt"]}
        if set(training).intersection(validation):
            raise RuntimeError("Training and fixed validation prompts overlap")
        self._ablation_record(
            "round_start", round=round_number,
            training_prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest()
                                    for prompt in training],
            rollout=self._speco_current_step_rollout_quality_metrics(),
            collection_seconds=float(getattr(self, "_speco_last_oldlogprob_total_elapsed_sec", 0) or 0),
            collection_rpc_seconds=float(getattr(self, "_speco_last_oldlogprob_collect_rpc_elapsed_sec", 0) or 0),
        )
        started = time.perf_counter()
        if self.global_steps > 2:
            reset = self._ray_get_if_needed(group.ablation_multiround_restore_source())
            self._ablation_record("pre_round_source_reset", round=round_number,
                                  workers=reset)
        prepared = self._ray_get_if_needed(group.ablation_prepare())
        if len(prepared) != 16 or {row["rank"] for row in prepared} != set(range(16)):
            raise RuntimeError("Multiround sampling did not prepare exactly 16 ranks")
        by_rank = {row["rank"]: row for row in prepared}
        for base in range(0, 16, 4):
            reference = by_rank[base]
            for peer in range(base + 1, base + 4):
                current = by_rank[peer]
                if (current["pool_sample_ids"] != reference["pool_sample_ids"]
                        or current["pool_rows"] != reference["pool_rows"]
                        or current["global_plan"] != reference["global_plan"]):
                    raise RuntimeError(f"SP group {base} disagrees on sampling pool or plan")
        self._ablation_record("prepared", round=round_number,
                              seconds=time.perf_counter() - started, workers=prepared)
        if self.global_steps == 2:
            initialized = self._ray_get_if_needed(group.ablation_multiround_init())
            self._ablation_record("initialized", workers=initialized)
            self._ablation_wake_rollout(cfg)
            self._ablation_probe("warmup_discard")
            self._ablation_probe("source_AA_0")
            self._ablation_probe("source_AA_1")
            self.checkpoint_manager.sleep_replicas()

        arms = (("uniform", "global_position_low_entropy") if round_number % 2
                else ("global_position_low_entropy", "uniform"))
        for arm in arms:
            loaded = self._ray_get_if_needed(group.ablation_multiround_load_branch(arm))
            self._ablation_record("branch_loaded", round=round_number, arm=arm, workers=loaded)
            started = time.perf_counter()
            trained = self._ray_get_if_needed(group.ablation_train(
                1e-5, 5, start=True, sampling_arm=arm,
                artifact_tag=f"round_{round_number}_{arm}",
                save_artifact=(round_number == 3),
            ))
            stored = self._ray_get_if_needed(group.ablation_multiround_store_branch(arm))
            self._ablation_record("trained", round=round_number, arm=arm, steps=5,
                                  rpc_seconds=time.perf_counter() - started,
                                  workers=trained, branch_states=stored)
            self._ablation_wake_rollout(cfg)
            self._ablation_crossed(f"round_{round_number}_{arm}", round_number * 5)
            self._ablation_crossed(f"round_{round_number}_{arm}", round_number * 5,
                                   performance=True)
            self.checkpoint_manager.sleep_replicas()
            self._speco_finalize_candidate(self.global_steps, 0.0)
            restored = self._ray_get_if_needed(group.ablation_multiround_restore_source())
            if len(restored) != 16:
                raise RuntimeError("Source restore did not reach every drafter rank")
            self._ablation_record("source_restored", round=round_number, arm=arm,
                                  workers=restored)

        self._ablation_wake_rollout(cfg)
        publish_started = time.perf_counter()
        payloads = self._ray_get_if_needed(group.ablation_multiround_source_weights())
        payload = self._first_non_null(payloads)
        if payload is None:
            raise RuntimeError("No source drafter publish payload")
        self._ray_get_if_needed(self._speco_actor_rollout_method("update_draft_weights")(
            resolve_drafter_publish_payload(payload), global_steps=self.global_steps))
        self._ablation_record("source_published", round=round_number,
                              seconds=time.perf_counter() - publish_started)
        cleaned = self._ray_get_if_needed(group.ablation_multiround_cleanup_batches())
        if len(cleaned) != 16 or any(row["removed"] != 10 for row in cleaned):
            raise RuntimeError(f"Round batch cache cleanup incomplete: {cleaned}")
        self._ablation_record("round_complete", round=round_number,
                              batch_cache_removed=cleaned)
        if round_number == 3:
            states = self._ray_get_if_needed(group.ablation_multiround_cleanup_states())
            expected = 2 if resume_root else 3
            if len(states) != 16 or any(row["removed"] != expected for row in states):
                raise RuntimeError(f"Multiround state cleanup incomplete: {states}")
            self._ablation_record("completed", actor_updates=0,
                                  drafter_optimizer_steps_per_arm=15, rounds=3,
                                  state_files_removed=states)
        return False, {"drafter/frozen_multiround_round": round_number}
