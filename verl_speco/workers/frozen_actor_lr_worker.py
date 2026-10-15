"""Opt-in fixed-data LR experiment; ordinary online workers do not use this path."""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import ray
import torch
from torch.utils._pytree import tree_map
from verl.single_controller.base.decorator import Dispatch, register
from verl_speco.backends.dspark_trainer_backend import _trace_anchor_rows_for_item
from verl_speco.trainer.entropy_lr import set_cycle_lr
from verl_speco.workers.speco_worker import SpecoWorker, _preserve_process_rng_state


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            item = item.detach()
            if hasattr(item, "to_local"):
                item = item.to_local()
            item = item.cpu().contiguous()
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def nested_collection_pools(items, caps, seed):
    """Select nested per-replica training pools from one collected rollout."""
    caps = [int(cap) for cap in caps]
    if not caps or caps != sorted(set(caps)) or caps[0] <= 0:
        raise ValueError(f"Invalid nested collection caps: {caps}")
    keyed = []
    for item in items:
        sample_id = item.get("sample_id")
        if not sample_id or "hidden_states" not in item:
            raise ValueError("Nested collection requires identified hidden-state samples")
        rows = int(item["hidden_states"].shape[0])
        if rows <= 0:
            raise ValueError(f"Invalid hidden-state rows for sample {sample_id}")
        priority = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
        keyed.append((priority, str(sample_id), rows, item))
    keyed.sort(key=lambda row: (row[0], row[1]))
    if len({row[1] for row in keyed}) != len(keyed):
        raise ValueError("Duplicate collected sample IDs in nested experiment")
    if sum(row[2] for row in keyed) > caps[-1]:
        raise ValueError("Collected training pool exceeds the configured high cap")
    pools = {}
    for cap in caps:
        used = 0
        selected = []
        for _, _, rows, item in keyed:
            if used + rows <= cap:
                selected.append(item)
                used += rows
        pools[cap] = selected
    if not all(
        {item["sample_id"] for item in pools[low]}.issubset(
            {item["sample_id"] for item in pools[high]})
        for low, high in zip(caps, caps[1:])
    ):
        raise RuntimeError("Nested collection selection is not monotone")
    return pools


def global_position_reservations(items, *, max_window, num_anchors, block_size,
                                 entropy_threshold=0.01, max_rejected_logprob=-4.0):
    """Balance hard first rejections across a full rank-local collection pool."""
    bins = [[], [], []]
    limits = {}
    seen = set()
    for item in items:
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate collection sample: {sample_id}")
        seen.add(sample_id)
        ids = item["input_ids"]
        hidden = item["hidden_states"]
        hidden_rows = hidden[0].shape[0] if isinstance(hidden, (list, tuple)) else hidden.shape[0]
        mask = item.get("loss_mask")
        if mask is None:
            if "prompts" in item and "responses" in item:
                mask = torch.zeros_like(ids, dtype=torch.float32)
                prompt_len = item["prompts"].numel()
                mask[prompt_len:prompt_len + item["responses"].numel()] = 1
            else:
                mask = torch.ones_like(ids, dtype=torch.float32)
        length = min(ids.numel(), hidden_rows, mask.numel())
        start = max(0, min(int(torch.nonzero(mask[:length])[0, 0]) - max_window // 2,
                           length - max_window)) if torch.any(mask[:length]) else max(0, length - max_window)
        end = min(start + max_window, length)
        eligible_anchors = set()
        candidates = _trace_anchor_rows_for_item(
            item, crop_start=start, crop_end=end, num_anchors=length,
            block_size=block_size, low_entropy_fraction=0,
            low_entropy_threshold=entropy_threshold,
            position_stratified_low_entropy=True,
            low_entropy_max_rejected_logprob=max_rejected_logprob,
            eligible_reserve_anchors=eligible_anchors,
        )
        limits[sample_id] = int(min(num_anchors, len(candidates)) * 0.5)
        for row in candidates:
            if row["anchor"] not in eligible_anchors:
                continue
            bucket = min(int(row["response_position"]) // 2048, 2)
            bins[bucket].append((int(row["priority"]), sample_id, int(row["anchor"])))
    for bucket in bins:
        bucket.sort()
    planned = {sample_id: [] for sample_id in seen}
    cursors = [0, 0, 0]
    total_budget = sum(limits.values())
    count = 0
    while count < total_budget:
        progress = False
        for index, bucket in enumerate(bins):
            while cursors[index] < len(bucket):
                _, sample_id, anchor = bucket[cursors[index]]
                cursors[index] += 1
                if len(planned[sample_id]) < limits[sample_id]:
                    planned[sample_id].append(anchor)
                    count += 1
                    progress = True
                    break
        if not progress:
            break
    return planned, dict(
        selected=count, budget=total_budget,
        by_position=[sum(anchor in planned[sample_id] for _, sample_id, anchor in bucket)
                     for bucket in bins],
        eligible_by_position=[len(bucket) for bucket in bins],
    )


class FrozenActorLRWorker(SpecoWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ablation_multiround_fixed_eval_status(self):
        items = self.trainer.fixed_eval_data
        return dict(rank=self.rank, samples=len(items),
                    sample_ids=[str(item.get("sample_id")) for item in items],
                    rows=sum(int(item["hidden_states"].shape[0]) for item in items))

    def _ablation_rng(self, state=None):
        device = self.trainer.device_module
        if state is None:
            return (random.getstate(), np.random.get_state(), torch.get_rng_state(), device.get_rng_state())
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        device.set_rng_state(state[3])

    def _ablation_state(self):
        trainer = self.trainer
        # This is called only after reshard/offload or get_model_state_dict().
        return dict(weights=fingerprint(dict(trainer.model.named_parameters())),
                    optimizer=fingerprint(trainer.optimizer.state_dict()),
                    scheduler=fingerprint(trainer.lr_scheduler.state_dict()),
                    optimizer_steps=trainer.optimizer_steps_total)

    def _ablation_record(self, event, **data):
        root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"worker_{self.rank:02d}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=event, time=time.time(), **data)) + "\n")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_prepare(self):
        if not self.in_drafter_train_group or self.trainer is None:
            raise RuntimeError("Frozen LR experiment requires all drafter ranks")
        t = self.trainer
        with _preserve_process_rng_state(self.device_name):
            if not await t.activate_training_model():
                raise RuntimeError("Drafter activation failed")
            self._ablation_batch_files = []
            self._ablation_batches = None
            root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output)
            root.mkdir(parents=True, exist_ok=True)
            if self.config.rollout.drafter.training.get("dspark_directional_compare_enabled", False):
                return await self._ablation_prepare_directional(root)
            if self.config.rollout.drafter.training.get("frozen_actor_nested_caps_enabled", False):
                return await self._ablation_prepare_nested(root)
            if (self.config.rollout.drafter.training.get("frozen_actor_sampling_compare_enabled", False)
                    or self.config.rollout.drafter.training.get("frozen_actor_sampling_multiround_enabled", False)):
                return await self._ablation_prepare_sampling(root)
            pool = [item for item in t.collected_data
                    if int(item.get("step", t.current_rl_step)) == int(t.current_rl_step)
                    and "hidden_states" in item] if not t.use_data_buffer else []
            pool_samples = len(pool) if not t.use_data_buffer else None
            pool_unique_samples = len({str(item["sample_id"]) for item in pool
                                       if item.get("sample_id") not in (None, "")}) if not t.use_data_buffer else None
            # Twenty pre-built batches: every arm sees exactly the same tensors
            # in the same order, including sampled anchors and target logprobs.
            cached = self.config.rollout.drafter.training.get("frozen_actor_lr_cached_source")
            if cached:
                batch_file = Path(cached) / f"batches_rank_{self.rank:02d}"
                if batch_file.is_dir():
                    self._ablation_batch_files = [batch_file / f"batch_{index:02d}.pt" for index in range(20)]
                else:
                    batch_file = batch_file.with_suffix(".pt")
                    self._ablation_batches = torch.load(batch_file, map_location="cpu", weights_only=True, mmap=True)
                records = [json.loads(line) for line in (Path(cached) / "results.jsonl").read_text().splitlines()]
                expected = next(row for row in records if row["event"] == "prepared")["workers"]
                expected = next(row for row in expected if row["rank"] == self.rank)["batch_hashes"]
            else:
                batch_file = root / f"batches_rank_{self.rank:02d}"
                batch_file.mkdir(exist_ok=False)
                hashes = []
                for index in range(20):
                    t.training_steps = index
                    batch = t.prepare_training_batch()
                    if batch is None:
                        raise RuntimeError("Cannot prepare the matched training batches")
                    batch["_speco_pad_size"] = int(t._current_pad_size)
                    cpu_batch = tree_map(
                        lambda x: x.detach().cpu().clone() if torch.is_tensor(x) else x, batch)
                    path = batch_file / f"batch_{index:02d}.pt"
                    hashes.append(fingerprint(cpu_batch))
                    torch.save(cpu_batch, path)
                    self._ablation_batch_files.append(path)
                    del batch, cpu_batch
            if len(self._ablation_batch_files or self._ablation_batches) != 20:
                raise RuntimeError("Expected exactly twenty cached training batches")
            t.training_steps = 0
            self._ablation_initial_rng = self._ablation_rng()
            if cached:
                hashes = [fingerprint(self._ablation_load_batch(index)) for index in range(20)]
                if hashes != expected:
                    raise RuntimeError(f"Cached batch hash mismatch: global rank {self.rank}")
            sample_draws = int(getattr(t, "_training_sample_draws", 0)) if not cached else None
            unique_samples = len(getattr(t, "_training_unique_sample_ids", set())) if not cached else None
            # This experiment has no asynchronous checkpoint/data-buffer work.
            # Use the existing barrier-free offload path; the coordinator joins
            # all rank RPCs before switching back to rollout.
            await t.release_training_memory_after_activation()
            self._ablation_baseline = self._ablation_state()
            self._ablation_record("prepared", batch_file=str(batch_file), batch_hashes=hashes,
                                  sample_draws=sample_draws, unique_samples=unique_samples,
                                  pool_samples=pool_samples, pool_unique_samples=pool_unique_samples,
                                  baseline=self._ablation_baseline)
            return dict(rank=self.rank, local_training_rank=t.rank, batch_file=str(batch_file),
                        batch_hashes=hashes, sample_draws=sample_draws,
                        unique_samples=unique_samples, pool_samples=pool_samples,
                        pool_unique_samples=pool_unique_samples, baseline=self._ablation_baseline)

    async def _ablation_prepare_directional(self, root):
        t = self.trainer
        cfg = self.config.rollout.drafter.training
        if t.use_data_buffer or int(t.current_rl_step) != int(self.last_global_step):
            raise RuntimeError("Directional comparison needs current-step samples")
        source = t.collected_data
        pool = [item for item in source if int(item.get("step", -1)) == int(t.current_rl_step)
                and item.get("_speco_directional_selected_anchors")]
        if len(pool) < t.batch_size:
            raise RuntimeError("Too few paired directional samples for a training batch")
        candidate_rows = getattr(self, "_ablation_directional_candidates", None)
        selections = getattr(self, "_ablation_directional_selections", None)
        if not candidate_rows or not selections:
            raise RuntimeError("Paired directional scores and arm selections are missing")
        by_sample = {}
        for sample_id, anchor, m, tv in candidate_rows:
            by_sample.setdefault(sample_id, {})[anchor] = (float(m), float(tv))
        ids = [str(item["sample_id"]) for item in pool]
        if len(ids) != len(set(ids)) or set(ids) != set(selections):
            raise RuntimeError("Directional arm selection does not match the training pool")
        arms = ("random", "top_tv", "top_m")
        if any(set(arms) != set(selections[sample_id]) or
               len({len(selections[sample_id][arm]) for arm in arms}) != 1
               for sample_id in ids):
            raise RuntimeError("Directional arms have different per-sample budgets")
        self._ablation_sampling_files = {}
        initial_rng = self._ablation_rng()
        records = {}
        try:
            for arm in arms:
                t.collected_data = type(source)(
                    (dict(item, _speco_directional_selected_anchors=selections[str(item["sample_id"])][arm])
                     for item in pool), maxlen=source.maxlen,
                )
                self._ablation_rng(initial_rng)
                t.reset_training_metrics()
                batch_dir = root / f"directional_{arm}_rank_{self.rank:02d}"
                batch_dir.mkdir(exist_ok=False)
                files, hashes, anchor_counts = [], [], []
                for index in range(5):
                    t.training_steps = index
                    batch = t.prepare_training_batch()
                    if batch is None:
                        raise RuntimeError(f"Missing {arm} training batch {index} on rank {self.rank}")
                    anchor_counts.append(int(batch["trace_anchor_mask"].sum().item()))
                    batch["_speco_pad_size"] = int(t._current_pad_size)
                    cpu_batch = tree_map(
                        lambda value: value.detach().cpu().clone() if torch.is_tensor(value) else value,
                        batch,
                    )
                    path = batch_dir / f"batch_{index:02d}.pt"
                    hashes.append(fingerprint(cpu_batch))
                    torch.save(cpu_batch, path)
                    files.append(path)
                    del batch, cpu_batch
                self._ablation_sampling_files[arm] = files
                selected_scores = [by_sample[sample_id][anchor]
                                   for sample_id in ids for anchor in selections[sample_id][arm]]
                records[arm] = dict(
                    batch_hashes=hashes, batch_anchor_counts=anchor_counts,
                    sample_draws=int(t._training_sample_draws),
                    unique_samples=len(t._training_unique_sample_ids),
                    sample_ids=sorted(t._training_unique_sample_ids),
                    selected_anchors=len(selected_scores),
                    selected_m_mean=sum(row[0] for row in selected_scores) / len(selected_scores),
                    selected_m_sum=sum(row[0] for row in selected_scores),
                    selected_m_positive=sum(row[0] > 0 for row in selected_scores),
                    selected_tv_mean=sum(row[1] for row in selected_scores) / len(selected_scores),
                )
        finally:
            t.collected_data = source
            t.training_steps = 0
            self._ablation_rng(initial_rng)
        reference = records[arms[0]]
        for arm in arms[1:]:
            current = records[arm]
            if (current["sample_ids"] != reference["sample_ids"]
                    or current["sample_draws"] != reference["sample_draws"]
                    or current["batch_anchor_counts"] != reference["batch_anchor_counts"]
                    or current["selected_anchors"] != reference["selected_anchors"]):
                raise RuntimeError("Directional arm batch or anchor budgets diverged")
        self._ablation_initial_rng = self._ablation_rng()
        await t.release_training_memory_after_activation()
        self._ablation_baseline = self._ablation_state()
        result = dict(rank=self.rank, local_training_rank=t.rank,
                      pool_sample_ids=ids, pool_rows=sum(int(item["hidden_states"].shape[0])
                                                    for item in pool),
                      candidate_count=len(candidate_rows),
                      candidate_hash=fingerprint(candidate_rows),
                      selection_hash=fingerprint(selections),
                      arms=records, baseline=self._ablation_baseline)
        self._ablation_record("prepared_directional", **result)
        return result

    async def _ablation_prepare_nested(self, root):
        t = self.trainer
        cfg = self.config.rollout.drafter.training
        if t.use_data_buffer:
            raise RuntimeError("Nested experiment requires current-step collected data")
        caps = [int(cap) for cap in cfg.frozen_actor_nested_caps]
        source = t.collected_data
        pool = [item for item in source
                if int(item.get("step", t.current_rl_step)) == int(t.current_rl_step)
                and "hidden_states" in item]
        pools = nested_collection_pools(pool, caps, int(cfg.frozen_actor_nested_seed))
        self._ablation_nested_files = {}
        coverage = {}
        try:
            for cap in caps:
                selected = pools[cap]
                if len(selected) < t.batch_size:
                    raise RuntimeError(f"Nested cap {cap} has only {len(selected)} samples on rank {self.rank}")
                t.collected_data = selected
                t.reset_training_metrics()
                batch_dir = root / f"cap_{cap}_rank_{self.rank:02d}"
                batch_dir.mkdir(exist_ok=False)
                files, hashes = [], []
                for index in range(5):
                    t.training_steps = index
                    batch = t.prepare_training_batch()
                    if batch is None:
                        raise RuntimeError(f"No nested batch: cap={cap}, rank={self.rank}, step={index}")
                    batch["_speco_pad_size"] = int(t._current_pad_size)
                    cpu_batch = tree_map(
                        lambda value: value.detach().cpu().clone() if torch.is_tensor(value) else value,
                        batch,
                    )
                    path = batch_dir / f"batch_{index:02d}.pt"
                    hashes.append(fingerprint(cpu_batch))
                    torch.save(cpu_batch, path)
                    files.append(path)
                    del batch, cpu_batch
                self._ablation_nested_files[cap] = files
                coverage[cap] = dict(
                    sample_ids=[str(item["sample_id"]) for item in selected],
                    sample_rows=sum(int(item["hidden_states"].shape[0]) for item in selected),
                    sample_draws=int(t._training_sample_draws),
                    unique_samples=len(t._training_unique_sample_ids),
                    batch_hashes=hashes,
                )
        finally:
            t.collected_data = source
            t.training_steps = 0
        self._ablation_initial_rng = self._ablation_rng()
        await t.release_training_memory_after_activation()
        self._ablation_baseline = self._ablation_state()
        result = dict(rank=self.rank, local_training_rank=t.rank, caps=coverage,
                      fixed_eval_samples=len(getattr(t, "fixed_eval_data", [])),
                      baseline=self._ablation_baseline)
        self._ablation_record("prepared_nested", **result)
        return result

    async def _ablation_prepare_sampling(self, root):
        t = self.trainer
        cfg = self.config.rollout.drafter.training
        multiround = bool(cfg.get("frozen_actor_sampling_multiround_enabled", False))
        if t.use_data_buffer:
            raise RuntimeError("Sampling comparison requires current-step collected data")
        source = t.collected_data
        pool = [item for item in source
                if int(item.get("step", t.current_rl_step)) == int(t.current_rl_step)
                and "hidden_states" in item]
        if len(pool) < t.batch_size:
            raise RuntimeError(f"Only {len(pool)} collected samples on rank {self.rank}")
        if sum(int(item["hidden_states"].shape[0]) for item in pool) > 16384:
            raise RuntimeError("Sampling comparison exceeded the 16K rank-local budget")
        plan, coverage = global_position_reservations(
            pool, max_window=int(cfg.dspark_max_window),
            num_anchors=int(cfg.dspark_num_anchors), block_size=int(cfg.dspark_block_size),
            entropy_threshold=float(cfg.dspark_low_entropy_threshold),
            max_rejected_logprob=float(cfg.dspark_low_entropy_max_rejected_logprob),
        )
        self._ablation_sampling_files = {}
        initial_rng = self._ablation_rng()
        previous = (float(cfg.dspark_low_entropy_anchor_fraction),
                    bool(cfg.dspark_position_entropy_stratified_sampling_enabled))
        arms = (("uniform", "global_position_low_entropy") if multiround
                else ("uniform", "low_entropy", "global_position_low_entropy"))
        arm_records = {}
        try:
            for arm in arms:
                cfg.dspark_low_entropy_anchor_fraction = 0.0 if arm == "uniform" else 0.5
                cfg.dspark_position_entropy_stratified_sampling_enabled = arm == "global_position_low_entropy"
                if arm == "global_position_low_entropy":
                    t.collected_data = [dict(item, _speco_global_reserved_trace_anchors=plan[str(item["sample_id"])])
                                        for item in pool]
                elif arm == "low_entropy":
                    t.collected_data = [dict(item, _speco_sampling_require_strict_rejection=True)
                                        for item in pool]
                else:
                    t.collected_data = pool
                self._ablation_rng(initial_rng)
                t.reset_training_metrics()
                prefix = f"round_{int(t.current_rl_step):02d}_" if multiround else ""
                batch_dir = root / f"{prefix}sampling_{arm}_rank_{self.rank:02d}"
                batch_dir.mkdir(exist_ok=False)
                files, hashes = [], []
                for index in range(5):
                    t.training_steps = index
                    batch = t.prepare_training_batch()
                    if batch is None:
                        raise RuntimeError(f"No {arm} training batch: rank={self.rank}, index={index}")
                    batch["_speco_pad_size"] = int(t._current_pad_size)
                    cpu_batch = tree_map(
                        lambda value: value.detach().cpu().clone() if torch.is_tensor(value) else value,
                        batch,
                    )
                    path = batch_dir / f"batch_{index:02d}.pt"
                    hashes.append(fingerprint(cpu_batch))
                    torch.save(cpu_batch, path)
                    files.append(path)
                    del batch, cpu_batch
                self._ablation_sampling_files[arm] = files
                arm_records[arm] = dict(
                    batch_hashes=hashes, sample_draws=int(t._training_sample_draws),
                    unique_samples=len(t._training_unique_sample_ids),
                    sample_ids=sorted(t._training_unique_sample_ids),
                )
        finally:
            t.collected_data = source
            cfg.dspark_low_entropy_anchor_fraction = previous[0]
            cfg.dspark_position_entropy_stratified_sampling_enabled = previous[1]
            t.training_steps = 0
            self._ablation_rng(initial_rng)
        sample_ids = arm_records[arms[0]]["sample_ids"]
        if any(arm_records[arm]["sample_ids"] != sample_ids for arm in arms[1:]):
            raise RuntimeError("Sampling arms drew different training samples")
        self._ablation_initial_rng = self._ablation_rng()
        await t.release_training_memory_after_activation()
        self._ablation_baseline = self._ablation_state()
        if multiround and hasattr(self, "_ablation_multiround_source"):
            if self._ablation_baseline != self._ablation_multiround_source["fingerprint"]:
                raise RuntimeError(f"Rollout source drifted before round {t.current_rl_step} on rank {self.rank}")
        result = dict(rank=self.rank, local_training_rank=t.rank,
                      pool_sample_ids=[str(item["sample_id"]) for item in pool],
                      pool_rows=sum(int(item["hidden_states"].shape[0]) for item in pool),
                      arms=arm_records, global_plan=coverage, baseline=self._ablation_baseline)
        self._ablation_record("prepared_sampling", **result)
        return result

    def _ablation_snapshot(self):
        t = self.trainer
        return dict(
            parameters={name: t._candidate_local_tensor(parameter).detach().cpu()
                        for name, parameter in t.model.named_parameters() if parameter.requires_grad},
            optimizer=t.optimizer.state_dict(),
            scheduler=t.lr_scheduler.state_dict(),
            optimizer_steps=int(t.optimizer_steps_total),
        )

    def _ablation_save_snapshot(self, name):
        root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output)
        path = root / f"state_rank_{self.rank:02d}_{name}.pt"
        if path.exists():
            raise FileExistsError(f"Multiround state already exists: {path}")
        fingerprint_before = self._ablation_state()
        torch.save(self._ablation_snapshot(), path)
        if self._ablation_state() != fingerprint_before:
            raise RuntimeError(f"State changed during snapshot on rank {self.rank}")
        return dict(path=str(path), fingerprint=fingerprint_before)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_multiround_init(self):
        if hasattr(self, "_ablation_multiround_source"):
            raise RuntimeError("Multiround source already initialized")
        resume_root = self.config.rollout.drafter.training.get(
            "frozen_actor_multiround_resume_state_dir")
        if resume_root:
            resume_root = Path(resume_root)
            records = [json.loads(line) for line in (resume_root / "results.jsonl").open()]
            source_row = next(row for row in records if row["event"] == "initialized")
            source = next(row["source"] for row in source_row["workers"]
                          if row["rank"] == self.rank)
            source_path = resume_root / f"state_rank_{self.rank:02d}_source.pt"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            self._ablation_multiround_source = dict(path=str(source_path), fingerprint=source)
            self._ablation_multiround_branches = {}
            for arm in ("uniform", "global_position_low_entropy"):
                trained = next(row for row in records if row["event"] == "trained"
                               and row.get("round") == 2 and row.get("arm") == arm)
                fingerprint = next(row["state"] for row in trained["branch_states"]
                                   if row["rank"] == self.rank)
                path = resume_root / f"state_rank_{self.rank:02d}_{arm}_round_2.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                self._ablation_multiround_branches[arm] = dict(
                    path=str(path), fingerprint=fingerprint)
            if self._ablation_state() != source:
                raise RuntimeError(f"Resumed source checkpoint differs on rank {self.rank}")
        else:
            self._ablation_multiround_source = self._ablation_save_snapshot("source")
            self._ablation_multiround_branches = {
                arm: self._ablation_multiround_source
                for arm in ("uniform", "global_position_low_entropy")
            }
        return dict(rank=self.rank, source=self._ablation_multiround_source["fingerprint"])

    async def _ablation_load_snapshot(self, saved):
        t = self.trainer
        state = torch.load(saved["path"], map_location="cpu", weights_only=True, mmap=True)
        if not await t.activate_training_model():
            raise RuntimeError("Drafter activation failed while switching multiround branches")
        with torch.no_grad():
            for name, parameter in t.model.named_parameters():
                if name in state["parameters"]:
                    local = t._candidate_local_tensor(parameter)
                    local.copy_(state["parameters"][name].to(device=local.device, dtype=local.dtype))
        t.optimizer.load_state_dict(state["optimizer"])
        t.lr_scheduler.load_state_dict(state["scheduler"])
        t.optimizer_steps_total = state["optimizer_steps"]
        t.training_steps = 0
        await t.release_training_memory_after_activation()
        del state
        actual = self._ablation_state()
        if actual != saved["fingerprint"]:
            raise RuntimeError(f"Multiround state switch mismatch on rank {self.rank}: {actual}")
        return actual

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_multiround_load_branch(self, arm):
        if arm not in self._ablation_multiround_branches:
            raise ValueError(f"Unknown multiround branch {arm}")
        state = await self._ablation_load_snapshot(self._ablation_multiround_branches[arm])
        self._ablation_baseline = state
        return dict(rank=self.rank, arm=arm, state=state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_multiround_store_branch(self, arm):
        if arm not in self._ablation_multiround_branches:
            raise ValueError(f"Unknown multiround branch {arm}")
        previous = self._ablation_multiround_branches[arm]
        current = self._ablation_save_snapshot(f"{arm}_round_{self.trainer.current_rl_step - 1}")
        self._ablation_multiround_branches[arm] = current
        root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output).resolve()
        if (previous["path"] != self._ablation_multiround_source["path"]
                and Path(previous["path"]).parent.resolve() == root):
            Path(previous["path"]).unlink()
        return dict(rank=self.rank, arm=arm,
                    state=current["fingerprint"])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_multiround_restore_source(self):
        before = self._ablation_state()
        state = await self._ablation_load_snapshot(self._ablation_multiround_source)
        self._ablation_baseline = state
        return dict(rank=self.rank, before=before, state=state,
                    drift=[key for key in state if before[key] != state[key]])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ablation_multiround_source_weights(self):
        if self._ablation_state() != self._ablation_multiround_source["fingerprint"]:
            raise RuntimeError(f"Source publish state mismatch on rank {self.rank}")
        weights = self.trainer.get_model_state_dict()
        if not self.is_global_publish_leader:
            return None
        if not weights:
            raise RuntimeError("Missing source drafter weights")
        return {"weights_ref": ray.put(weights)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ablation_multiround_cleanup_batches(self):
        count = 0
        for files in self._ablation_sampling_files.values():
            for path in files:
                path.unlink()
                count += 1
            files[0].parent.rmdir()
        self._ablation_sampling_files = {}
        return dict(rank=self.rank, removed=count)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ablation_multiround_cleanup_states(self):
        root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output).resolve()
        states = list(self._ablation_multiround_branches.values()) + [self._ablation_multiround_source]
        paths = {state["path"] for state in states if Path(state["path"]).parent.resolve() == root}
        for path in paths:
            Path(path).unlink()
        return dict(rank=self.rank, removed=len(paths))

    def _ablation_load_batch(self, index):
        if self._ablation_batches is not None:
            return self._ablation_batches[index]
        return torch.load(self._ablation_batch_files[index], map_location="cpu", weights_only=True, mmap=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_train(
        self,
        lr,
        target_steps,
        start=False,
        first_rejection_ce_alpha=None,
        rejected_aux_alpha=None,
        rejected_prefix_length=None,
        artifact_tag=None,
        accepted_k3_alpha=None,
        accepted_fkl_alpha=None,
        nested_cap=None,
        sampling_arm=None,
        save_artifact=True,
    ):
        t = self.trainer
        with _preserve_process_rng_state(self.device_name):
            if nested_cap is not None:
                cap = int(nested_cap)
                if cap not in self._ablation_nested_files:
                    raise ValueError(f"Unknown nested cap {cap}")
                self._ablation_batch_files = self._ablation_nested_files[cap]
                self._ablation_batches = None
            if sampling_arm is not None:
                if sampling_arm not in self._ablation_sampling_files:
                    raise ValueError(f"Unknown sampling arm {sampling_arm}")
                self._ablation_batch_files = self._ablation_sampling_files[sampling_arm]
                self._ablation_batches = None
            if start:
                current = self._ablation_state()
                if current != self._ablation_baseline:
                    raise RuntimeError(f"Baseline restoration mismatch on rank {t.rank}: {current}")
                self._ablation_done = 0
                self._ablation_saved_rng = self._ablation_initial_rng
            started = time.perf_counter()
            if not await t.activate_training_model():
                raise RuntimeError("Drafter activation failed")
            draft_model = t.model.module if hasattr(t.model, "module") else t.model
            if accepted_k3_alpha is not None or accepted_fkl_alpha is not None:
                if accepted_k3_alpha is None or accepted_fkl_alpha is None:
                    raise ValueError("Accepted K3 and FKL alphas must be set together")
                accepted_alphas = (float(accepted_k3_alpha), float(accepted_fkl_alpha))
                if any(not np.isfinite(value) or value < 0 for value in accepted_alphas):
                    raise ValueError("Accepted objective alphas must be finite and non-negative")
                if not all(hasattr(draft_model, key) for key in ("accepted_k3_alpha", "accepted_fkl_alpha")):
                    raise RuntimeError("Drafter model does not expose accepted objective alphas")
                draft_model.accepted_k3_alpha, draft_model.accepted_fkl_alpha = accepted_alphas
            if first_rejection_ce_alpha is not None:
                ce_alpha = float(first_rejection_ce_alpha)
                if not np.isfinite(ce_alpha) or ce_alpha < 0:
                    raise ValueError("First-rejection CE alpha must be finite and non-negative")
                if not hasattr(draft_model, "first_rejection_ce_alpha"):
                    raise RuntimeError("Drafter model does not expose first_rejection_ce_alpha")
                draft_model.first_rejection_ce_alpha = ce_alpha
            if rejected_aux_alpha is not None or rejected_prefix_length is not None:
                if rejected_aux_alpha is None or rejected_prefix_length is None:
                    raise ValueError(
                        "Rejected auxiliary alpha and prefix length must be set together"
                    )
                aux_alpha = float(rejected_aux_alpha)
                prefix_length = int(rejected_prefix_length)
                if not np.isfinite(aux_alpha) or aux_alpha < 0:
                    raise ValueError("Rejected auxiliary alpha must be finite and non-negative")
                if prefix_length < 0 or (aux_alpha > 0 and prefix_length <= 0):
                    raise ValueError("Rejected auxiliary prefix length is invalid")
                draft_model.rejected_aux_alpha = aux_alpha
                draft_model.rejected_prefix_length = prefix_length
            if start:
                ready = t.begin_candidate_update(self.last_global_step)
                if not ready and self.rank == 0:
                    raise RuntimeError("Cannot snapshot ablation baseline")
                t.reset_training_metrics()
            set_cycle_lr(t.optimizer, t.lr_scheduler, lr)
            self._ablation_rng(self._ablation_saved_rng)
            # cleanup_training resets the display counter; Adam and its moments
            # must continue from step 5 to 20 without a rollback in between.
            t.training_steps = self._ablation_done
            for index in range(self._ablation_done, int(target_steps)):
                batch = tree_map(lambda x: x.to(t.runtime_device) if torch.is_tensor(x) else x,
                                 self._ablation_load_batch(index))
                t._current_pad_size = int(batch.get("_speco_pad_size", 0))
                if not await t.training_step_from_batch(batch, self.last_global_step):
                    raise RuntimeError(f"LR={lr} failed at optimizer step {index + 1}")
                del batch
                self._ablation_done = index + 1
            self._ablation_saved_rng = self._ablation_rng()
            metrics = t.get_training_metrics()
            train_seconds = time.perf_counter() - started
            expected_optimizer_steps = int(self._ablation_baseline["optimizer_steps"]) + int(target_steps)
            if t.optimizer_steps_total != expected_optimizer_steps:
                raise RuntimeError(
                    f"Unexpected optimizer-step count in fixed-LR arm: "
                    f"expected={expected_optimizer_steps}, actual={t.optimizer_steps_total}"
                )
            ready = t.prepare_candidate_update(self.last_global_step)
            if not ready and self.rank == 0:
                raise RuntimeError("Cannot export ablation candidate")
            if self.rank == 0 and save_artifact:
                root = Path(self.config.rollout.drafter.training.frozen_actor_lr_output)
                name = artifact_tag or f"lr_{lr:.1e}"
                torch.save(t._candidate_trained_state_dict,
                           root / f"drafter_{name}_step_{target_steps}.pt")
            await t.release_training_memory_after_activation()
            result = dict(rank=self.rank, local_training_rank=t.rank, lr=float(lr), steps=int(target_steps),
                          train_seconds=train_seconds, metrics=metrics,
                          weights=fingerprint(dict(t.model.named_parameters())))
            if first_rejection_ce_alpha is not None:
                result["first_rejection_ce_alpha"] = float(first_rejection_ce_alpha)
            if nested_cap is not None:
                result["nested_cap"] = int(nested_cap)
            if sampling_arm is not None:
                result["sampling_arm"] = sampling_arm
            if accepted_k3_alpha is not None:
                result["accepted_k3_alpha"] = float(accepted_k3_alpha)
                result["accepted_fkl_alpha"] = float(accepted_fkl_alpha)
            if rejected_aux_alpha is not None:
                result["rejected_aux_alpha"] = float(rejected_aux_alpha)
                result["rejected_prefix_length"] = int(rejected_prefix_length)
            self._ablation_record("trained", **result)
            return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def ablation_check_restored(self):
        await self.trainer.release_training_memory_after_activation()
        current = self._ablation_state()
        ok = current == self._ablation_baseline
        self._ablation_record("restored", ok=ok, state=current)
        if not ok:
            raise RuntimeError(f"Ablation rollback is not exact: rank={self.trainer.rank}")
        return dict(rank=self.rank, ok=ok)
