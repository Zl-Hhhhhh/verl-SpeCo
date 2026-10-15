"""Opt-in, rank-local evidence for drafter state transitions.

No collectives, forward replays, RNG draws, or model mutations are performed.
Parameter hashes describe LOCAL shards; different shard coordinates must not
be compared as if they were replicas. Buffers include nonpersistent RoPE state.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import time

import torch

logger = logging.getLogger(__name__)
ENV = "SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR"


def cpu_tensor(value):
    value = value.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    # Device-to-host already creates independent storage. Clone only CPU inputs;
    # cloning a transferred FP32 target head doubles a multi-GiB staging buffer.
    if value.device.type == "cpu":
        return value.clone(memory_format=torch.contiguous_format)
    return value.to(device="cpu").contiguous()


def _hash_chunks(value, max_elements):
    """Yield bounded views in logical contiguous order, including strided inputs."""
    if value.numel() <= max_elements:
        yield value
    elif value.is_contiguous():
        yield from value.view(-1).split(max_elements)
    else:
        # Avoid flatten/reshape on a large strided tensor: they may copy it all.
        for index in range(value.shape[0]):
            yield from _hash_chunks(value[index], max_elements)


def tensor_record(value, *, retain_tensor=True, chunk_bytes=16 * 1024 * 1024):
    if chunk_bytes < value.element_size():
        raise ValueError("chunk_bytes must fit at least one tensor element")
    local_value = value.detach()
    if hasattr(local_value, "to_local"):
        local_value = local_value.to_local()
    local = cpu_tensor(local_value) if retain_tensor else None
    source = local if local is not None else local_value
    # uint8 preserves BF16 and NaN bit patterns without a dtype conversion.
    digest = hashlib.sha256()
    for chunk in _hash_chunks(source, chunk_bytes // source.element_size()):
        host = chunk.to(device="cpu").contiguous()
        raw = host.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        del raw, host
    record = {
        "sha256": digest.hexdigest(), "shape": list(value.shape),
        "local_shape": list(local_value.shape), "dtype": str(value.dtype),
        "device": str(value.device), "stride": list(value.stride()),
    }
    if hasattr(value, "placements"):
        record["placements"] = [str(p) for p in value.placements]
        record["mesh_coordinate"] = value.device_mesh.get_coordinate()
        record["mesh_names"] = list(value.device_mesh.mesh_dim_names or ())
    return record, local


def clone_tree(value):
    if torch.is_tensor(value):
        return cpu_tensor(value)
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clone_tree(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _tensor_leaves(value, prefix="", depth=0):
    if torch.is_tensor(value):
        yield prefix, value
    elif depth < 3 and isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{prefix}/{key!r}", depth + 1)
    elif depth < 3 and isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            yield from _tensor_leaves(item, f"{prefix}/{key}", depth + 1)


class StateDiagnostics:
    def __init__(self, trainer, root):
        self.trainer = trainer
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(trainer.rank)
        self.rank = rank
        self.root = Path(root) / f"rank_{rank:04d}_pid_{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.forward_sequence = 0
        self.last_forward = None
        self.last_snapshot_step = None

    def metadata(self):
        t = self.trainer
        return {
            "schema_version": 1, "global_rank": self.rank,
            "rollout_dp_rank": int(t.rollout_dp_rank),
            "sp_rank": int(t.rank), "rl_step": int(t.current_rl_step),
            "optimizer_steps_total": int(t.optimizer_steps_total),
            "target_lm_head_step": getattr(t, "_target_lm_head_weight_step", None),
            "target_lm_head_pending": getattr(t, "_pending_target_lm_head_weight", None) is not None,
            "pending_publish_step": getattr(t, "_pending_publish_step", None),
            "last_snapshot_step": self.last_snapshot_step,
            "training_active": bool(getattr(t, "_training_active", False)),
            "wandb_run_id": os.getenv("WANDB_RUN_ID"),
            "time_ns": time.time_ns(),
        }

    def append(self, record):
        self.sequence += 1
        record.update(self.metadata(), event_sequence=self.sequence)
        with (self.root / "events.jsonl").open("a") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def save(self, payload, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)
        return str(path)

    def state(self):
        t = self.trainer
        state = {"parameters": {}, "buffers": {}, "runtime": {}, "modules": {}}
        modules = {"model": getattr(t, "model", None),
                   "target_lm_head": getattr(t.backend, "target_lm_head", None)}
        for prefix, model in modules.items():
            if model is None:
                continue
            for name, value in model.named_parameters():
                record, _ = tensor_record(value, retain_tensor=False)
                state["parameters"][f"{prefix}.{name}"] = record
            for name, module in model.named_modules():
                module_name = ".".join(part for part in (prefix, name) if part)
                scalars = {"training": bool(module.training)}
                for attr in ("max_seq_len_cached", "original_max_seq_len", "base", "dim", "attention_scaling"):
                    value = getattr(module, attr, None)
                    if isinstance(value, (int, float, str, bool)):
                        scalars[attr] = value
                state["modules"][module_name] = scalars
                for name, value in module._buffers.items():
                    if value is None:
                        continue
                    record, local = tensor_record(value)
                    record["persistent"] = name not in module._non_persistent_buffers_set
                    record["artifact"] = self._cache_tensor(record, local)
                    state["buffers"][f"{module_name}.{name}"] = record
                # Plain tensor attributes and template caches are absent from state_dict.
                for attr, value in vars(module).items():
                    if attr in {"_parameters", "_buffers", "_modules"}:
                        continue
                    if torch.is_tensor(value) or "cache" in attr or "inv_freq" in attr:
                        for key, tensor in _tensor_leaves(value, attr):
                            record, local = tensor_record(tensor)
                            record["artifact"] = self._cache_tensor(record, local)
                            state["runtime"][f"{module_name}.{key}"] = record
        return state

    def _cache_tensor(self, record, local):
        # Content-addressed cache keeps repeated RoPE snapshots small.
        tag = hashlib.sha256(json.dumps([record["dtype"], record["local_shape"], record["sha256"]]).encode()).hexdigest()
        relative = f"buffers/{tag}.pt"
        if not (self.root / relative).exists():
            self.save(local, relative)
        return relative

    def snapshot(self, phase, **extra):
        return self.append({"phase": phase, "state": self.state(), **extra})

    def before_forward(self, batch, pad_size):
        self.last_forward = None
        self.forward_sequence += 1
        ident = f"rl_{self.trainer.current_rl_step:06d}_opt_{self.trainer.optimizer_steps_total + 1:06d}_forward_{self.forward_sequence:06d}"
        payload = {"metadata": self.metadata(), "batch": clone_tree(batch), "pad_size": pad_size}
        records = {key: tensor_record(tensor, retain_tensor=False)[0] for key, tensor in _tensor_leaves(payload["batch"])}
        path = self.save(payload, f"inputs/{ident}.pt")
        event = self.snapshot("forward.before", input_path=path, inputs=records, forward_id=ident)
        self.last_forward = {"forward_id": ident, "input_path": path,
                             "state_event_sequence": event["event_sequence"]}

    def after_forward(self, loss_dict):
        if self.last_forward is None:
            logger.error("[drafter state diagnostics] missing before-forward evidence; skip output association")
            return
        diagnostics = loss_dict.get("diagnostics", {})
        payload = diagnostics.get("k3_logprob_dump", {})
        ident = self.last_forward["forward_id"]
        path = self.save({"metadata": self.metadata(), "streams": clone_tree(payload),
                          "state_probe": clone_tree(diagnostics.pop("state_probe", {}))}, f"outputs/{ident}.pt")
        self.snapshot("forward.after", output_path=path, **self.last_forward)

    def publish_payload(self, weights, global_step):
        self.last_snapshot_step = global_step
        fingerprints = {name: tensor_record(value, retain_tensor=False)[0] for name, value in (weights or {}).items() if torch.is_tensor(value)}
        self.append({"phase": "publish.payload", "publish_step": global_step, "weights": fingerprints})


def diagnostics_for(trainer):
    root = os.getenv(ENV)
    if not root or getattr(getattr(trainer, "backend", None), "model_type", None) != "dspark":
        return None
    diag = getattr(trainer, "_state_diagnostics", None)
    if diag is None:
        diag = StateDiagnostics(trainer, root)
        trainer._state_diagnostics = diag
    return diag


def record(trainer, method, *args, **kwargs):
    # Diagnostics failures are visible but must not change optimizer/control flow.
    try:
        diag = diagnostics_for(trainer)
        if diag is not None:
            return getattr(diag, method)(*args, **kwargs)
    except Exception:
        logger.exception("[drafter state diagnostics] %s failed", method)
    return None


def trace_state(method):
    """Record before/after state for synchronous or asynchronous lifecycle calls."""
    phase = method.__name__
    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def wrapped(self, *args, **kwargs):
            record(self, "snapshot", phase + ".before")
            try:
                return await method(self, *args, **kwargs)
            finally:
                record(self, "snapshot", phase + ".after")
    else:
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            record(self, "snapshot", phase + ".before")
            try:
                return method(self, *args, **kwargs)
            finally:
                record(self, "snapshot", phase + ".after")
    return wrapped


def record_rollout_publish(weights, global_step, phase):
    """Actor-side payload digest and RPC acknowledgement, not a claim about vLLM buffers."""
    root = os.getenv(ENV)
    if not root:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        path = Path(root) / f"publish_rank_{rank:04d}_pid_{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"phase": phase, "global_rank": rank, "publish_step": global_step,
                 "time_ns": time.time_ns(), "wandb_run_id": os.getenv("WANDB_RUN_ID")}
        if phase == "send":
            value["weights"] = {name: tensor_record(t, retain_tensor=False)[0] for name, t in weights.items() if torch.is_tensor(t)}
        with path.open("a") as out:
            out.write(json.dumps(value, sort_keys=True) + "\n")
    except Exception:
        logger.exception("[drafter state diagnostics] rollout publish failed")
