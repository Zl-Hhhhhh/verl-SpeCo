"""Coordinator-side entropy LR experiment, using transactional drafter workers."""
import json
import logging
from pathlib import Path
import time

import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl_speco.integration.vllm_runtime import SPECO_VLLM_SPECULATIVE_TRACE_KEY
from verl_speco.trainer.entropy_lr import EntropyLRController, paired_evidence, trace_statistics

logger = logging.getLogger(__name__)


def _prompt_json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f'Unsupported validation prompt value: {type(value).__name__}')


class EntropyLRRuntime:
    def __init__(self, trainer):
        self.trainer = trainer
        cfg = trainer._speco_drafter_training_config()
        if not cfg.get('candidate_gate_enabled') or cfg.get('lr_scheduler_type') != 'constant' or int(cfg.get('lr_warmup_steps', 0)):
            raise ValueError('Entropy LR requires candidate gate and constant LR without warmup')
        self.control = EntropyLRController(
            base=float(cfg.get('lr', 1e-5)), current=float(cfg.get('lr', 1e-5)),
            floor=float(cfg.get('entropy_lr_min', 5e-6)), maximum=float(cfg.get('entropy_lr_max', 1.5e-5)),
            ema_beta=float(cfg.get('entropy_lr_ema_beta', .3)))
        self.observed_step = None
        self.cohort = None
        self.last_stats = {}
        self.root = Path(trainer.config.trainer.default_local_dir)
        resume_step = trainer._speco_resume_global_step_hint()
        if resume_step is not None:
            resume_root = Path(trainer.config.trainer.get('resume_from_path') or self.root / f'global_step_{resume_step}')
            state = json.loads((resume_root / 'entropy_lr_state.json').read_text())
            self.control = EntropyLRController(**state['controller'])

    def observe(self, traces, step):
        if self.observed_step == step:
            return
        self.last_stats = trace_statistics(traces)
        self.control.observe(self.last_stats)
        self.observed_step = step

    def learning_rate(self, step):
        if self.observed_step != step:
            raise RuntimeError('Missing current-rollout entropy for dynamic LR')
        return self.control.current

    def save(self):
        target = self.root / f'global_step_{self.trainer.global_steps}' / 'entropy_lr_state.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps({'controller': self.control.state_dict()}, indent=2))
        temporary.replace(target)

    def record(self, metrics):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'entropy_lr_decisions.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': int(self.trainer.global_steps), **metrics,
                                     'controller': self.control.state_dict()}) + '\n')
        logger.warning('[entropy lr] step=%s metrics=%s', self.trainer.global_steps, metrics)

    def reject_precheck(self, reason, train_metrics):
        saturation = train_metrics.get('drafter/train/dspark_rejected_k3_clamp_ratio', 0.0)
        self.control.feedback(degraded=True, saturation=saturation)
        metrics = {**self.control.metrics(), 'drafter/entropy_lr/precheck_rejected': 1}
        self.record({**metrics, 'reason': reason})
        return metrics

    def prepare_validation_cohort(self):
        """Prepare held-out prompts before upstream fit initializes global_steps."""
        trainer = self.trainer
        if self.cohort is None:
            # Dedicated validation data, never the actor training batch or its repeats.
            parts, seen = [], set()
            for data in trainer.val_dataloader:
                batch = DataProto.from_single_dict(data)
                prompts = batch.non_tensor_batch.get('raw_prompt')
                if prompts is None:
                    raise ValueError('Entropy LR validation requires raw_prompt for AgentLoop generation')
                unique = []
                for i, prompt in enumerate(prompts):
                    key = json.dumps(prompt, sort_keys=True, ensure_ascii=False,
                                     default=_prompt_json_default)
                    if key not in seen:
                        seen.add(key)
                        unique.append(i)
                    if len(seen) == 64:
                        break
                if unique:
                    parts.append(batch[unique])
                if len(seen) == 64:
                    break
            if len(seen) < 2:
                raise ValueError('Entropy LR paired validation needs at least two unique validation prompts')
            batch = DataProto.concat(parts)
            batch.non_tensor_batch['uid'] = np.array([f'entropy-lr-{i}' for i in range(len(batch))], dtype=object)
            self.cohort = trainer._get_gen_batch(batch)
            logger.warning('[entropy lr] validation cohort has %s unique prompts', len(self.cohort))
        return self.cohort

    def validation_batch(self, count):
        trainer = self.trainer
        self.prepare_validation_cohort()
        batch = self.cohort[:min(count, len(self.cohort))]
        batch.meta_info = dict(eos_token_id=trainer.tokenizer.eos_token_id,
                               pad_token_id=trainer.tokenizer.pad_token_id,
                               recompute_log_prob=False, do_sample=False, validate=True,
                               global_steps=trainer.global_steps, speco_candidate_gate=True,
                               speco_candidate_gate_max_tokens=512)
        return batch

    def measure(self, generate, count):
        batch = self.validation_batch(count)
        def generate_unpadded(probe):
            workers = int(self.trainer.config.actor_rollout_ref.rollout.agent.num_workers)
            padded, pad_size = pad_dataproto_to_divisor(probe, workers)
            # Match upstream validation dispatch, excluding padding from both
            # per-prompt evidence and the response-token throughput denominator.
            return unpad_dataproto(generate(padded), pad_size=pad_size)

        output, result = self.trainer._speco_measure_candidate_rollout(generate_unpadded, (batch,), {})
        traces = output.non_tensor_batch.get(SPECO_VLLM_SPECULATIVE_TRACE_KEY)
        if traces is None or len(traces) != len(batch):
            raise RuntimeError('Missing validation traces for entropy LR')
        return {**result, **trace_statistics(traces)}

    def validate(self, generate, train_metrics):
        trainer = self.trainer
        step = int(trainer.global_steps)
        started = time.perf_counter()
        # Estimate amortization over the following five actor rollouts. Probe
        # latency is an estimate only; whole-step throughput remains authoritative.
        quality = trainer._speco_current_step_rollout_quality_metrics()
        interval = int(trainer._speco_drafter_training_config().get('training_interval_steps', 5))
        tokens = quality.get('response_tokens', 0.0) * interval
        selected = 0.0
        try:
            for count in (32, 64):
                trainer._speco_publish_candidate_alpha(step, 0.0)
                baseline = self.measure(generate, count)
                trainer._speco_publish_candidate_alpha(step, 1.0)
                candidate = self.measure(generate, count)
                evidence = paired_evidence(
                    baseline, candidate, projected_tokens=tokens,
                    overhead_s=time.perf_counter()-started + train_metrics.get('timing_s/drafter_train_rpc', 0.0))
                if evidence['improved'] or evidence['degraded'] or len(baseline['per_prompt']) >= len(self.cohort):
                    break
            selected = float(evidence['improved'])
            if not selected:
                trainer._speco_publish_candidate_alpha(step, 0.0)
            trainer._speco_finalize_candidate(step, selected)
        except Exception:
            trainer._speco_publish_candidate_alpha(step, 0.0)
            trainer._speco_finalize_candidate(step, 0.0)
            trainer._speco_pending_candidate_update = None
            self.control.feedback(degraded=True)
            self.record({'validation_failed': 1, **self.control.metrics()})
            raise
        trainer._speco_pending_candidate_update = None
        saturation = train_metrics.get('drafter/train/dspark_rejected_k3_clamp_ratio', 0.0)
        self.control.feedback(improved=bool(selected), degraded=bool(evidence['degraded']), saturation=saturation)
        metrics = {f'drafter/entropy_lr/validation_{key}': value for key, value in evidence.items()}
        metrics.update(self.control.metrics())
        metrics.update({'drafter/candidate/selected_alpha': selected,
                        'drafter/candidate/committed': int(selected),
                        'drafter/candidate/rolled_back': int(not selected),
                        'drafter/published': int(selected),
                        'timing_s/entropy_lr_validation': time.perf_counter()-started})
        self.record(metrics)
        return metrics
