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
"""Runtime bridge from SPECO drafter config to upstream verl vLLM rollout.

The upstream verl v0.8.0 rollout config does not know about
``rollout.drafter``.  SPECO therefore injects only the vLLM-native launch
arguments under ``rollout.engine_kwargs.vllm`` before upstream validation, and
keeps draft-only weight publishing as a runtime method on the rollout adapter.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import types
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable, cast

from verl_speco.integration.verl_npu_vllm_compat import (
    install_verl_npu_vllm_import_compat,
)
from verl_speco.trainer.checkpoint import trim_process_host_memory

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"
SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV = "VERL_SPECO_VLLM_DRAFT_UPDATE_USE_SHM"
SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS = (
    "verl_speco.integration.vllm_runtime.SpecoVLLMWeightSyncCompatExtension"
)
SPECO_VLLM_WORKER_EXTENSION_CLS = (
    "verl_speco.integration.vllm_runtime.SpecoVLLMColocateWorkerExtension"
)
SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_ENV = (
    "VERL_SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_SECONDS"
)
SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX = "_speco_vllm_spec_decode"
SPECO_VLLM_SPECULATIVE_TRACE_KEY = "speculative_trace"
SPECO_VLLM_SPECULATIVE_TRACE_DIAGNOSTICS_KEY = "speculative_trace_diagnostics"
_SPECO_VLLM_TRACE_TRANSFER_KEY = "__speco_speculative_trace__"
_SPECO_VLLM_TRACE_DIAGNOSTICS_TRANSFER_KEY = (
    "__speco_speculative_trace_diagnostics__"
)
_SPECO_VLLM_TRACE_HEADER_KEY = "x-speco-speculative-trace"
_SPECO_VLLM_TRACE_DIAGNOSTICS_HEADER_KEY = "x-speco-speculative-trace-diagnostics"
SPECO_VLLM_DRAFT_DIAG_ENV = "VERL_SPECO_VLLM_DRAFT_DIAG"
SPECO_VLLM_TRACE_SPOOL_DIR_ENV = "VERL_SPECO_VLLM_TRACE_SPOOL_DIR"
SPECO_VLLM_TRACE_SPOOL_DIR_DEFAULT = "/dev/shm/verl_speco_vllm_traces"
SPECO_VLLM_NPU_STAGING_ENV = "VERL_SPECO_VLLM_NPU_STAGING"
SPECO_VLLM_VERIFY_ENTROPY_ENV = "SPECO_VLLM_VERIFY_ENTROPY"
SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES = 64 << 20
SPECO_VLLM_ENGINE_STARTUP_LOCK_ENV = "VERL_SPECO_VLLM_ENGINE_STARTUP_LOCK"
SPECO_VLLM_ENGINE_STARTUP_LOCK_DEFAULT = (
    "/tmp/verl_speco_vllm_engine_startup.lock"
)

_VLLM_REPLICA_PATCHED = False
_VLLM_DFLASH_CONFIG_ALIASES_PATCHED = False
_VLLM_DSPARK_RUNTIME_PATCHED = False
_VLLM_DSPARK_REGISTRY_ALIAS_PATCHED = False
_NPU_TARGET_STAGING_STATE = threading.local()
_VLLM_SPEC_TRACE_LOCK = threading.Lock()
_VLLM_SPEC_TRACES_BY_REQUEST: dict[str, list[dict[str, Any]]] = {}
_VLLM_SPEC_TRACE_DIAGNOSTICS_BY_REQUEST: dict[str, dict[str, int]] = {}

_DSPARK_VLLM_ARCHITECTURES = {
    "DSparkDraftModel",
    "Qwen3DSparkModel",
    "DeepSeekDSparkModel",
}
_TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK = (
    "attention",
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",
)


@contextmanager
def _serialize_vllm_engine_startup():
    """Prevent concurrent EngineCore processes from selecting the same port."""
    lock_path = os.getenv(
        SPECO_VLLM_ENGINE_STARTUP_LOCK_ENV,
        SPECO_VLLM_ENGINE_STARTUP_LOCK_DEFAULT,
    ).strip()
    if not lock_path:
        yield
        return

    import fcntl

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        logger.info("Waiting for vLLM EngineCore startup lock: %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _speco_is_npu_vllm_worker(worker: Any) -> bool:
    try:
        from vllm.platforms import current_platform

        return str(getattr(current_platform, "device_type", "")).lower() == "npu"
    except Exception:  # noqa: BLE001
        device = getattr(worker, "device", None)
        return str(getattr(device, "type", "")).lower() == "npu"


def _get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _plain_container(value: Any):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        pass

    if isinstance(value, dict):
        return {key: _plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_container(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: _plain_container(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _open_dict_if_needed(config: Any):
    try:
        from omegaconf import OmegaConf, open_dict

        if OmegaConf.is_config(config):
            return open_dict(config)
    except Exception:  # noqa: BLE001
        pass
    return nullcontext()


def _set_child(container: Any, key: str, value: Any) -> None:
    with _open_dict_if_needed(container):
        if hasattr(container, "__setitem__"):
            container[key] = value
        else:
            setattr(container, key, value)


def _has_config_field(config: Any, key: str) -> bool:
    if config is None:
        return False
    if hasattr(config, "get"):
        try:
            return key in config
        except TypeError:
            return config.get(key, None) is not None
    return hasattr(config, key)


def _ensure_child_mapping(container: Any, key: str) -> Any:
    child = _get_nested(container, (key,), None)
    if child is None:
        child = {}
        _set_child(container, key, child)
        # OmegaConf copies mappings on assignment; mutate the stored child.
        child = _get_nested(container, (key,), None)
    return child


def _ensure_nested_mapping(config: Any, path: tuple[str, ...]) -> Any:
    current = config
    for key in path:
        current = _ensure_child_mapping(current, key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            continue
        return value
    return None


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "on", "yes", "y"}:
            return True
        if normalized in {"0", "false", "off", "no", "n", ""}:
            return False
    return bool(value)


def _describe_vllm_draft_logits(draft_logits: Any, *, missing: bool = False) -> str:
    if missing:
        return "missing"
    if draft_logits is None:
        return "None(greedy)"
    shape = getattr(draft_logits, "shape", None)
    if shape is not None:
        try:
            return f"tensor{tuple(shape)}"
        except TypeError:
            return f"shape={shape}"
    return type(draft_logits).__name__


def _resolve_torch_rebuild_func(func: Any):
    if callable(func):
        return func
    if isinstance(func, str):
        from torch.multiprocessing import reductions

        resolved = getattr(reductions, func.rsplit(".", 1)[-1], None)
        if callable(resolved):
            return resolved
    raise TypeError(f"Unsupported IPC rebuild function: {func!r}")


def _speco_rebuild_ipc_compat(handle: tuple[Any, tuple], device_id: int | None = None):
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        if len(list_args) <= 6:
            raise ValueError(
                f"IPC rebuild args do not include a device id slot: len={len(list_args)}"
            )
        list_args[6] = device_id
    return _resolve_torch_rebuild_func(func)(*list_args)


setattr(_speco_rebuild_ipc_compat, "_speco_compat", True)


def patch_verl_bucketed_weight_transfer_rebuild_ipc(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Patch verl's bucketed IPC rebuild helper for serialized rebuild names.

    Some environments deserialize the first element of a torch IPC handle as a
    rebuild function name string instead of the callable object. SPECO installs
    this runtime compatibility patch without modifying the vendored verl tree.
    """

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import (
                bucketed_weight_transfer as bucketed_weight_transfer_module,
            )
        except Exception:  # noqa: BLE001
            return False
        bucketed_weight_transfer = bucketed_weight_transfer_module

    current = getattr(bucketed_weight_transfer, "rebuild_ipc", None)
    if getattr(current, "_speco_compat", False):
        return False
    bucketed_weight_transfer.rebuild_ipc = _speco_rebuild_ipc_compat
    return True


def _speco_persistent_weight_shm_name(zmq_handle: str, bucket_size: int) -> str:
    """Return a job/rank/bucket-scoped shared-memory name.

    ``zmq_handle`` contains the Ray job id, replica id, and local rank in verl.
    Including the bucket size keeps target and draft transfers isolated when
    they use different bucket sizes.
    """

    identity = f"{zmq_handle}\0{int(bucket_size)}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return f"verl_weights_speco_{digest}"


def patch_verl_bucketed_weight_transfer_shm_reuse(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Reuse one stable SHM mapping per verl weight-transfer channel.

    NPU cannot use torch device IPC, so verl falls back to POSIX shared memory.
    Upstream creates a new UUID-named bucket for every actor update. If an NPU
    runtime keeps the old mmap/pinned registration alive after ``close()``, the
    node can retain one full bucket per rank and update. Keeping the mapping
    open and reusing it bounds host memory to one bucket per channel.

    CUDA/device IPC is delegated to the untouched upstream implementation.
    """

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import (
                bucketed_weight_transfer as bucketed_weight_transfer_module,
            )
        except Exception:  # noqa: BLE001
            return False
        bucketed_weight_transfer = bucketed_weight_transfer_module

    sender_cls = getattr(bucketed_weight_transfer, "BucketedWeightSender", None)
    receiver_cls = getattr(bucketed_weight_transfer, "BucketedWeightReceiver", None)
    if sender_cls is None or receiver_cls is None:
        return False

    sender_init = getattr(sender_cls, "_init_buffer", None)
    receiver_init = getattr(receiver_cls, "_init_buffer", None)
    sender_cleanup = getattr(sender_cls, "_cleanup", None)
    receiver_cleanup = getattr(receiver_cls, "_cleanup", None)
    methods = (sender_init, receiver_init, sender_cleanup, receiver_cleanup)
    if not all(callable(method) for method in methods):
        return False
    if all(getattr(method, "_speco_shm_reuse", False) for method in methods):
        return False

    cache = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_cache", None
    )
    if cache is None:
        cache = {}
        bucketed_weight_transfer._speco_persistent_weight_shm_cache = cache
    owner_names = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_owner_names", None
    )
    if owner_names is None:
        owner_names = set()
        bucketed_weight_transfer._speco_persistent_weight_shm_owner_names = owner_names
    reuse_logged = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_reuse_logged", None
    )
    if reuse_logged is None:
        reuse_logged = set()
        bucketed_weight_transfer._speco_persistent_weight_shm_reuse_logged = (
            reuse_logged
        )
    cache_lock = getattr(
        bucketed_weight_transfer, "_speco_persistent_weight_shm_lock", None
    )
    if cache_lock is None:
        cache_lock = threading.Lock()
        bucketed_weight_transfer._speco_persistent_weight_shm_lock = cache_lock

    if not getattr(
        bucketed_weight_transfer,
        "_speco_persistent_weight_shm_cleanup_registered",
        False,
    ):

        def _cleanup_persistent_weight_shm() -> None:
            with cache_lock:
                entries = list(cache.items())
                cache.clear()
                owned = set(owner_names)
                owner_names.clear()

            # Drop torch.frombuffer views before closing their mmap objects.
            shm_entries = [(name, entry[1]) for name, entry in entries]
            entries.clear()
            gc.collect()
            for name, shm in shm_entries:
                try:
                    shm.close()
                except (BufferError, OSError) as exc:
                    logger.warning(
                        "[speco weight shm] close failed name=%s: %s", name, exc
                    )
                if name in owned:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning(
                            "[speco weight shm] unlink failed name=%s: %s", name, exc
                        )

        bucketed_weight_transfer._speco_cleanup_persistent_weight_shm = (
            _cleanup_persistent_weight_shm
        )
        bucketed_weight_transfer._speco_persistent_weight_shm_cleanup_registered = True
        atexit.register(_cleanup_persistent_weight_shm)

    def _get_cached_buffer(shm_name: str, shm_size: int, *, owner: bool):
        with cache_lock:
            cached = cache.get(shm_name)
            if cached is not None:
                buffer, shm = cached
                if int(getattr(shm, "size", shm_size)) < shm_size:
                    raise RuntimeError(
                        f"Persistent weight SHM {shm_name!r} is smaller than requested: "
                        f"{getattr(shm, 'size', None)} < {shm_size}"
                    )
                if owner:
                    owner_names.add(shm_name)
                reuse_key = ("sender" if owner else "receiver", shm_name)
                if reuse_key not in reuse_logged:
                    reuse_logged.add(reuse_key)
                    logger.warning(
                        "[speco weight shm] persistent mapping reused role=%s name=%s "
                        "size_mb=%.1f pid=%s cache_entries=%s",
                        reuse_key[0],
                        shm_name,
                        shm_size / (1 << 20),
                        os.getpid(),
                        len(cache),
                    )
                return buffer, shm

            if owner:
                shm = bucketed_weight_transfer.create_shared_memory(shm_size, shm_name)
                buffer = bucketed_weight_transfer.torch.frombuffer(
                    shm.buf,
                    dtype=bucketed_weight_transfer.torch.uint8,
                )
                owner_names.add(shm_name)
                role = "sender"
            else:
                buffer, shm = bucketed_weight_transfer.rebuild_shared_memory(
                    shm_name,
                    shm_size,
                    dtype=bucketed_weight_transfer.torch.uint8,
                )
                role = "receiver"
            cache[shm_name] = (buffer, shm)
            logger.warning(
                "[speco weight shm] persistent mapping ready role=%s name=%s size_mb=%.1f",
                role,
                shm_name,
                shm_size / (1 << 20),
            )
            return buffer, shm

    def _sender_init_buffer_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return sender_init(self)
        shm_name = _speco_persistent_weight_shm_name(self.zmq_handle, self.bucket_size)
        buffer, shm = _get_cached_buffer(shm_name, self.bucket_size, owner=True)
        self.socket.send_pyobj({"name": shm_name, "size": self.bucket_size})
        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _receiver_init_buffer_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return receiver_init(self)
        comm_metadata = self.socket.recv_pyobj()
        shm_name = comm_metadata["name"]
        shm_size = int(comm_metadata["size"])
        buffer, shm = _get_cached_buffer(shm_name, shm_size, owner=False)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _sender_cleanup_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return sender_cleanup(self)
        # The module-level cache owns the mapping. Let upstream clean sockets
        # and device caches without closing or unlinking the persistent SHM.
        self.buffer = None
        self.shm = None
        return sender_cleanup(self)

    def _receiver_cleanup_with_shm_reuse(self):
        if not bool(getattr(self, "use_shm", False)):
            return receiver_cleanup(self)
        self.buffer = None
        self.shm = None
        return receiver_cleanup(self)

    for method in (
        _sender_init_buffer_with_shm_reuse,
        _receiver_init_buffer_with_shm_reuse,
        _sender_cleanup_with_shm_reuse,
        _receiver_cleanup_with_shm_reuse,
    ):
        setattr(method, "_speco_shm_reuse", True)

    sender_cls._init_buffer = _sender_init_buffer_with_shm_reuse
    receiver_cls._init_buffer = _receiver_init_buffer_with_shm_reuse
    sender_cls._cleanup = _sender_cleanup_with_shm_reuse
    receiver_cls._cleanup = _receiver_cleanup_with_shm_reuse
    return True


def patch_verl_bucketed_weight_transfer_npu_staging(
    bucketed_weight_transfer: Any = None,
) -> bool:
    """Use one reusable NPU allocation for all SHM buckets in a target reload."""

    if bucketed_weight_transfer is None:
        try:
            from verl.workers.rollout.vllm_rollout import (
                bucketed_weight_transfer as bucketed_weight_transfer_module,
            )
        except Exception:  # noqa: BLE001
            return False
        bucketed_weight_transfer = bucketed_weight_transfer_module

    receiver_cls = getattr(bucketed_weight_transfer, "BucketedWeightReceiver", None)
    if receiver_cls is None:
        return False
    original_receive = getattr(receiver_cls, "receive_weights", None)
    if not callable(original_receive) or getattr(
        original_receive, "_speco_npu_staging", False
    ):
        return False

    logged_pids = getattr(
        bucketed_weight_transfer, "_speco_npu_staging_logged_pids", None
    )
    if logged_pids is None:
        logged_pids = set()
        bucketed_weight_transfer._speco_npu_staging_logged_pids = logged_pids

    def _receive_weights_with_npu_staging(self, on_bucket_received):
        enabled = bool(getattr(_NPU_TARGET_STAGING_STATE, "enabled", False))
        device_type = str(getattr(getattr(self, "device", None), "type", "")).lower()
        if (
            not enabled
            or not bool(getattr(self, "use_shm", False))
            or device_type != "npu"
        ):
            return original_receive(self, on_bucket_received)

        staging_buffer = None
        weights = None
        tensor = None
        metadata = None
        bucket_meta = None
        try:
            self._init_socket()
            self._init_buffer()
            capacity = int(self.buffer.numel())
            staging_buffer = bucketed_weight_transfer.torch.empty(
                capacity,
                dtype=bucketed_weight_transfer.torch.uint8,
                device=self.device,
            )
            if os.getpid() not in logged_pids:
                logged_pids.add(os.getpid())
                logger.warning(
                    "[speco vllm weight sync] using one reusable NPU staging allocation for SHM target reload "
                    "pid=%s capacity_mb=%.1f copy_chunk_mb=%.1f",
                    os.getpid(),
                    capacity / (1 << 20),
                    SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES / (1 << 20),
                )

            while True:
                metadata = self.socket.recv_pyobj()
                bucket_meta = metadata["bucket_meta"]
                used_bytes = max(
                    (
                        int(meta["offset"])
                        + int(meta["dtype"].itemsize * meta["shape"].numel())
                        for meta in bucket_meta.values()
                        if meta["handle"] is None
                    ),
                    default=0,
                )
                if used_bytes > capacity:
                    raise RuntimeError(
                        f"NPU staging bucket overflow: {used_bytes} > {capacity}"
                    )
                if used_bytes:
                    for start in range(
                        0, used_bytes, SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES
                    ):
                        end = min(
                            start + SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES, used_bytes
                        )
                        staging_buffer[start:end].copy_(
                            self.buffer[start:end], non_blocking=False
                        )

                weights = []
                for name, meta in bucket_meta.items():
                    shape = meta["shape"]
                    dtype = meta["dtype"]
                    offset = int(meta["offset"])
                    handle = meta["handle"]
                    if handle is not None:
                        tensor = bucketed_weight_transfer.rebuild_ipc(
                            handle, self.device.index
                        )
                    else:
                        size = int(dtype.itemsize * shape.numel())
                        tensor = (
                            staging_buffer[offset : offset + size]
                            .view(dtype=dtype)
                            .view(shape)
                        )
                    weights.append((name, tensor))

                on_bucket_received(weights)
                bucketed_weight_transfer.get_torch_device().synchronize()
                self.socket.send(b"")
                weights = None
                tensor = None
                if metadata["is_last"]:
                    break
        finally:
            weights = None
            tensor = None
            bucket_meta = None
            metadata = None
            staging_buffer = None
            self._cleanup()

    setattr(_receive_weights_with_npu_staging, "_speco_npu_staging", True)
    receiver_cls.receive_weights = _receive_weights_with_npu_staging
    return True


def _speco_npu_target_staging_decision(
    worker: Any,
    *,
    peft_config: dict | None,
    use_shm: bool,
) -> tuple[bool, str]:
    if _bool_or_none(os.getenv(SPECO_VLLM_NPU_STAGING_ENV)) is False:
        return False, "disabled_by_env"
    if not use_shm:
        return False, "not_shm"
    if peft_config is not None:
        return False, "peft"
    if not _speco_is_npu_vllm_worker(worker):
        return False, "not_npu"
    if bool(getattr(worker, "_is_qat_model", False)) or bool(
        getattr(worker, "_is_modelopt_qat", False)
    ):
        return False, "qat"
    use_mtp_sync = getattr(worker, "_use_mtp_drafter_weight_sync", None)
    if callable(use_mtp_sync) and use_mtp_sync():
        return False, "mtp"
    runner = getattr(worker, "model_runner", None)
    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is None:
        return False, "missing_vllm_config"
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is not None:
        return False, f"quantized:{type(quant_config).__name__}"
    return True, "eligible"


def _speco_can_use_npu_target_staging(
    worker: Any, *, peft_config: dict | None, use_shm: bool
) -> bool:
    enabled, _ = _speco_npu_target_staging_decision(
        worker,
        peft_config=peft_config,
        use_shm=use_shm,
    )
    return enabled


@contextmanager
def _speco_npu_target_staging(worker: Any, *, peft_config: dict | None, use_shm: bool):
    previous = bool(getattr(_NPU_TARGET_STAGING_STATE, "enabled", False))
    enabled, reason = _speco_npu_target_staging_decision(
        worker,
        peft_config=peft_config,
        use_shm=use_shm,
    )
    _NPU_TARGET_STAGING_STATE.enabled = enabled
    if not bool(getattr(worker, "_speco_npu_staging_decision_logged", False)):
        worker._speco_npu_staging_decision_logged = True
        print(
            "[speco vllm weight sync] NPU staging decision "
            f"enabled={int(enabled)} reason={reason} pid={os.getpid()} "
            f"local_rank={getattr(worker, 'local_rank', None)} use_shm={int(bool(use_shm))}",
            flush=True,
        )
    try:
        yield
    finally:
        _NPU_TARGET_STAGING_STATE.enabled = previous


def _int_list_or_none(value: Any, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or isinstance(value, dict):
        raise TypeError(f"{field_name} must be a list of integers")
    try:
        return [int(item) for item in value]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a list of integers") from exc
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain only integers") from exc


def _is_dspark_config(config: Any) -> bool:
    architectures = _get_nested(config, ("architectures",), None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture_names = {str(name) for name in architectures}
    return bool(
        _has_config_field(config, "markov_head_type")
        or _has_config_field(config, "dspark_config")
        or architecture_names.intersection(_DSPARK_VLLM_ARCHITECTURES)
    )


def _normalize_dflash_target_layer_aliases(config: Any) -> bool:
    """Mirror DFlash/DSpark target layer ids into vLLM 0.23 alias fields."""

    target_layer_ids = _int_list_or_none(
        _get_nested(config, ("target_layer_ids",), None),
        "target_layer_ids",
    )
    dflash_config = _get_nested(config, ("dflash_config",), None)
    if dflash_config is not None and not hasattr(dflash_config, "get"):
        raise TypeError("DFlash dflash_config must be a mapping when provided")
    nested_target_layer_ids = _int_list_or_none(
        _get_nested(dflash_config, ("target_layer_ids",), None),
        "dflash_config.target_layer_ids",
    )
    dspark_config = _get_nested(config, ("dspark_config",), None)
    if dspark_config is not None and not hasattr(dspark_config, "get"):
        raise TypeError("DSpark dspark_config must be a mapping when provided")
    dspark_target_layer_ids = _int_list_or_none(
        _get_nested(dspark_config, ("target_layer_ids",), None),
        "dspark_config.target_layer_ids",
    )

    if (
        target_layer_ids is not None
        and nested_target_layer_ids is not None
        and target_layer_ids != nested_target_layer_ids
    ):
        raise ValueError(
            "DFlash target_layer_ids conflict with dflash_config.target_layer_ids: "
            f"{target_layer_ids} != {nested_target_layer_ids}"
        )
    if (
        target_layer_ids is not None
        and dspark_target_layer_ids is not None
        and target_layer_ids != dspark_target_layer_ids
    ):
        raise ValueError(
            "DSpark target_layer_ids conflict with dspark_config.target_layer_ids: "
            f"{target_layer_ids} != {dspark_target_layer_ids}"
        )
    if (
        nested_target_layer_ids is not None
        and dspark_target_layer_ids is not None
        and nested_target_layer_ids != dspark_target_layer_ids
    ):
        raise ValueError(
            "DSpark dflash_config.target_layer_ids conflict with dspark_config.target_layer_ids: "
            f"{nested_target_layer_ids} != {dspark_target_layer_ids}"
        )

    selected_layer_ids = _first_present(
        target_layer_ids, nested_target_layer_ids, dspark_target_layer_ids
    )
    if selected_layer_ids is None:
        return False

    changed = False
    if dflash_config is None:
        dflash_config = {}
        _set_child(config, "dflash_config", dflash_config)
        changed = True
    if nested_target_layer_ids is None:
        _set_child(dflash_config, "target_layer_ids", selected_layer_ids)
        changed = True
    if (
        _is_dspark_config(config)
        and _get_nested(dflash_config, ("mask_token_id",), None) is None
        and _get_nested(config, ("mask_token_id",), None) is not None
    ):
        _set_child(
            dflash_config,
            "mask_token_id",
            _get_nested(config, ("mask_token_id",), None),
        )
        changed = True

    expected_aux_layer_ids = [layer_id + 1 for layer_id in selected_layer_ids]
    existing_aux_layer_ids = _int_list_or_none(
        _get_nested(config, ("eagle_aux_hidden_state_layer_ids",), None),
        "eagle_aux_hidden_state_layer_ids",
    )
    if (
        existing_aux_layer_ids is not None
        and existing_aux_layer_ids != expected_aux_layer_ids
    ):
        raise ValueError(
            "DFlash eagle_aux_hidden_state_layer_ids conflict with target_layer_ids: "
            f"{existing_aux_layer_ids} != {expected_aux_layer_ids}"
        )
    if existing_aux_layer_ids is None:
        _set_child(config, "eagle_aux_hidden_state_layer_ids", expected_aux_layer_ids)
        changed = True

    return changed


def _drafter_algorithm(drafter_cfg: dict[str, Any]) -> str:
    return (
        str(drafter_cfg.get("speculative_algorithm", "EAGLE3") or "EAGLE3")
        .strip()
        .upper()
    )


def _validate_vllm_dflash_drafter_config(
    spec_model_path: Any, algorithm: str = "DFLASH"
) -> None:
    if not spec_model_path:
        return

    config_path = os.path.join(os.fspath(spec_model_path), "config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid DFlash drafter config.json at {config_path}: {exc}"
        ) from exc

    architectures = config.get("architectures") or []
    algorithm = str(algorithm or "DFLASH").strip().upper()
    if algorithm == "DSPARK":
        if not _is_dspark_config(config):
            raise ValueError(
                "vLLM DSpark uses the DFlash speculative path, but requires "
                "actor_rollout_ref.rollout.drafter.model_path to point to a DSpark drafter "
                "checkpoint with markov_head_type or a DSpark architecture in config.json; "
                f"got architectures={architectures!r} from {config_path}."
            )
        return

    if architectures and "DFlashDraftModel" not in architectures:
        raise ValueError(
            "vLLM DFlash requires actor_rollout_ref.rollout.drafter.model_path "
            "to point to a DFlash drafter checkpoint with architectures=['DFlashDraftModel']; "
            f"got architectures={architectures!r} from {config_path}. "
            "Do not use an EAGLE/EAGLE3 drafter path with speculative_algorithm=DFLASH."
        )


def _load_env_drafter_config() -> dict[str, Any]:
    raw = os.getenv(SPECO_DRAFTER_CONFIG_ENV)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {SPECO_DRAFTER_CONFIG_ENV}: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def _vllm_drafter_env_payload(drafter_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(drafter_cfg)


def _rollout_name(config: Any) -> str | None:
    return _get_nested(
        config, ("actor_rollout_ref", "rollout", "name"), None
    ) or _get_nested(config, ("rollout", "name"), None)


def _drafter_config_from_config(config: Any) -> dict[str, Any]:
    drafter = _get_nested(config, ("actor_rollout_ref", "rollout", "drafter"), None)
    if drafter is None:
        drafter = _get_nested(config, ("rollout", "drafter"), {})
    drafter = _plain_container(drafter) or {}
    return drafter if isinstance(drafter, dict) else {}


def _rollout_config_from_config(config: Any) -> Any:
    return _get_nested(config, ("actor_rollout_ref", "rollout"), None) or _get_nested(
        config, ("rollout",), None
    )


def _speculative_method_from_drafter(drafter_cfg: dict[str, Any]) -> str:
    algorithm = _drafter_algorithm(drafter_cfg)
    if algorithm == "PEAGLE":
        # P-EAGLE trains against vLLM's parallel-drafting runtime (speculators
        # PR #480). This overlay wires the training path; enabling vLLM rollout
        # requires that runtime, which is not asserted here yet.
        raise ValueError(
            "P-EAGLE vLLM rollout requires the parallel-drafting runtime and is not wired in this overlay yet; "
            "keep actor_rollout_ref.rollout.drafter.enable=false (train the drafter, serve separately)."
        )
    if algorithm == "DOMINO":
        # Domino is a DFlash variant, not an engine-level method: engines expose it as
        # "dflash" and enable the causal correction head (prefix_gru + embed_proj) from
        # the checkpoint's dflash_config.projector_type="domino" (vllm-project/vllm#48241,
        # sgl-project/sglang#31328). DOMINO is never a valid engine algorithm string, so
        # fail loud and point at DFLASH.
        raise ValueError(
            "DOMINO is not an engine-level speculative algorithm; Domino is served as a DFlash "
            "projector sub-mode. Set actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH "
            "for the rollout/serve path; the trained checkpoint's dflash_config.projector_type=domino "
            "enables the Domino correction head on engines that support it, keeping DOMINO for "
            "drafter training."
        )
    if algorithm == "DSPARK":
        # DSpark has its own native vLLM/vLLM-Ascend proposer.  Treating it as
        # DFlash selects a different draft-model contract on NPU and prevents
        # the DSpark block-7 runtime from being initialized.
        return "dspark"

    method_map = {
        # EAGLE-1 and EAGLE-2 share vLLM's native EAGLE draft (method="eagle");
        # EAGLE-2 is a dynamic-tree decoding policy over the same draft head.
        "EAGLE1": "eagle",
        "EAGLE2": "eagle",
        "EAGLE3": "eagle3",
        "DFLASH": "dflash",
        "DRAFT": "draft_model",
        "DRAFT_MODEL": "draft_model",
        "MTP": "mtp",
    }
    if algorithm not in method_map:
        raise ValueError(
            f"Unsupported SPECO speculative_algorithm for vLLM 0.23: {algorithm}"
        )
    return method_map[algorithm]


def _should_force_eager(drafter_cfg: dict[str, Any]) -> bool:
    explicit = _first_present(
        _get_nested(drafter_cfg, ("vllm", "enforce_eager"), None),
        _get_nested(drafter_cfg, ("vllm", "force_eager"), None),
        _get_nested(drafter_cfg, ("training", "draft_update_enforce_eager"), None),
    )
    return explicit is not None and bool(_bool_or_none(explicit))


# Acceptance settings that break the exact target-distribution guarantee RL relies
# on. greedy DRAFT sampling is lossless (a one-hot proposal fed into rejection
# sampling); only ACCEPTANCE-relaxing knobs are listed here. This is a best-effort
# denylist of the lossy modes we can name confidently, NOT a proof of losslessness --
# it fails closed on the known silent-degradation paths (config overrides via
# drafter.vllm.speculative_config_overrides or engine_kwargs.vllm.speculative_config).
_LOSSY_VLLM_ACCEPTANCE_CHECKS = (
    (
        "acceptance_method",
        lambda v: str(v).strip().lower() == "typical_acceptance_sampler",
        "typical_acceptance_sampler trades exactness for speed",
    ),
    (
        "spec_decoding_acceptance_method",
        lambda v: str(v).strip().lower() == "typical_acceptance_sampler",
        "typical_acceptance_sampler trades exactness for speed",
    ),
    (
        "rejection_sample_method",
        lambda v: str(v).strip().lower() == "synthetic",
        "synthetic acceptance does not sample from the corrected residual distribution",
    ),
    (
        "posterior_threshold",
        lambda v: v is not None and float(v) > 0.0,
        "a nonzero posterior_threshold enables typical/Medusa relaxed acceptance",
    ),
    (
        "posterior_alpha",
        lambda v: v is not None and float(v) > 0.0,
        "a nonzero posterior_alpha enables typical/Medusa relaxed acceptance",
    ),
)


def assert_lossless_vllm_speculative_config(config: Any, *, allow_lossy: bool) -> None:
    """Fail closed on known-lossy vLLM speculative acceptance settings.

    RL rollout under speculative decoding is only unbiased when the verifier samples
    exactly from the target policy. SPECO recomputes PPO's ``old_log_probs`` as the
    target logprob with no importance-sampling correction, so a relaxed acceptance
    method silently miscalibrates the PPO ratio (uncorrected off-policy bias). This
    turns that implicit assumption into an enforced contract for the acceptance modes
    we can name; set ``allow_lossy`` to opt in knowingly.
    """
    if allow_lossy or not isinstance(config, dict):
        return
    offenders = []
    for key, is_lossy, why in _LOSSY_VLLM_ACCEPTANCE_CHECKS:
        if key not in config:
            continue
        try:
            lossy = bool(is_lossy(config[key]))
        except (TypeError, ValueError):
            lossy = False
        if lossy:
            offenders.append(f"{key}={config[key]!r} ({why})")
    if offenders:
        raise ValueError(
            "SPECO refuses a lossy speculative-decoding config that would break the "
            "target-distribution guarantee RL relies on: " + "; ".join(offenders) + ". "
            "The generated tokens would no longer be exactly sampled from the target policy, so "
            "PPO's old_log_probs (recomputed as the target logprob, with no importance-sampling "
            "correction) would be miscalibrated. Set "
            "actor_rollout_ref.rollout.drafter.vllm.allow_lossy_speculative_sampling=true to opt in knowingly."
        )


def build_vllm_speculative_config_from_drafter(
    drafter_cfg: dict[str, Any],
    rollout_cfg: Any = None,
) -> dict[str, Any]:
    """Build a vLLM 0.23 ``speculative_config`` from SPECO drafter config."""

    if not bool(drafter_cfg.get("enable")):
        return {}

    algorithm = _drafter_algorithm(drafter_cfg)
    method = _speculative_method_from_drafter(drafter_cfg)
    spec_model_path = _first_present(
        drafter_cfg.get("model_path"),
        drafter_cfg.get("checkpoint_path"),
        _get_nested(drafter_cfg, ("spec_model", "path"), None),
        _get_nested(drafter_cfg, ("model", "path"), None),
        drafter_cfg.get("spec_model_path"),
    )
    if (
        method in {"eagle", "eagle3", "draft_model", "dflash", "dspark"}
        and spec_model_path is None
    ):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.model_path is required for vLLM speculative decoding"
        )

    rollout_drafter_cfg = drafter_cfg.get("rollout") or {}
    vllm_cfg = drafter_cfg.get("vllm") or {}
    if method in ("dflash", "dspark"):
        if method == "dflash" or algorithm == "DSPARK":
            _validate_vllm_dflash_drafter_config(spec_model_path, algorithm=algorithm)
        num_speculative_tokens = _positive_int_or_none(
            rollout_drafter_cfg.get("spec_verify_tokens")
        )
        if num_speculative_tokens is None:
            raise ValueError(
                "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens "
                f"must be positive for vLLM {method.upper()} speculative decoding"
            )
    else:
        num_speculative_tokens = _positive_int_or_none(
            _first_present(
                rollout_drafter_cfg.get("spec_steps"),
                rollout_drafter_cfg.get("spec_verify_tokens"),
                drafter_cfg.get("num_speculative_tokens"),
            )
        )
        if num_speculative_tokens is None:
            raise ValueError(
                "actor_rollout_ref.rollout.drafter.rollout.spec_steps or spec_verify_tokens "
                "must be positive for vLLM speculative decoding"
            )

    speculative_config: dict[str, Any] = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_sample_method": "greedy",
    }
    if spec_model_path is not None:
        speculative_config["model"] = spec_model_path

    draft_tp = _positive_int_or_none(
        _first_present(
            vllm_cfg.get("draft_tensor_parallel_size"),
            drafter_cfg.get("draft_tensor_parallel_size"),
        )
    )
    if draft_tp is not None:
        speculative_config["draft_tensor_parallel_size"] = draft_tp

    max_model_len = _positive_int_or_none(
        _first_present(vllm_cfg.get("max_model_len"), drafter_cfg.get("max_model_len"))
    )
    if max_model_len is not None:
        speculative_config["max_model_len"] = max_model_len

    if _should_force_eager(drafter_cfg):
        speculative_config["enforce_eager"] = True

    # Keep draft sampling greedy by default. This preserves the NPU/vLLM-Ascend
    # DFlash-family behavior where draft probabilities should not affect
    # rejection sampling; native GPU DSpark can opt in through overrides.

    overrides = vllm_cfg.get("speculative_config_overrides") or {}
    if not isinstance(overrides, dict):
        raise TypeError(
            "drafter.vllm.speculative_config_overrides must be a mapping when provided"
        )
    speculative_config.update(_plain_container(overrides))
    if algorithm == "DSPARK":
        runtime_query_tokens = _positive_int_or_none(
            speculative_config.get("num_speculative_tokens")
        )
        if runtime_query_tokens is None:
            raise ValueError(
                "vLLM DSpark num_speculative_tokens must remain positive after "
                "speculative_config_overrides"
            )
        speculative_config["model"] = _prepare_dspark_attention_runtime_model(
            speculative_config.get("model", spec_model_path),
            vllm_cfg,
            draft_query_tokens=runtime_query_tokens,
        )
    assert_lossless_vllm_speculative_config(
        speculative_config,
        allow_lossy=bool(
            _bool_or_none(vllm_cfg.get("allow_lossy_speculative_sampling", False))
        ),
    )
    return speculative_config


def _prepare_dspark_attention_runtime_model(
    model_path: Any,
    vllm_cfg: dict[str, Any],
    *,
    draft_query_tokens: int,
) -> str:
    """Create a config-only model overlay for uniform DSpark context SWA."""

    attention_mode = str(
        vllm_cfg.get("dspark_attention_mode", "full") or "full"
    ).strip().lower()
    valid_modes = {"full", "swa", "hybrid", "sparse_global"}
    if attention_mode not in valid_modes:
        raise ValueError(
            "drafter.vllm.dspark_attention_mode must be one of "
            f"{sorted(valid_modes)}, got {attention_mode!r}"
        )
    if attention_mode in {"hybrid", "sparse_global"}:
        raise NotImplementedError(
            f"DSpark {attention_mode} serving is not supported by the current vLLM "
            "DFlash KV-cache implementation: mixed full/sliding layers require "
            "multiple cache groups. Use full or uniform swa for train-serving parity."
        )
    if attention_mode == "full":
        return str(model_path)

    context_window = _positive_int_or_none(vllm_cfg.get("dspark_context_window"))
    if context_window is None:
        raise ValueError(
            "drafter.vllm.dspark_context_window must be positive when "
            "dspark_attention_mode=swa"
        )
    query_tokens = _positive_int_or_none(draft_query_tokens)
    if query_tokens is None:
        raise ValueError("DSpark SWA requires a positive draft query-token count")
    # vLLM's draft window includes the non-causal query block. The public SPECO
    # context window counts actor-context tokens only, matching the training mask.
    runtime_window = context_window + query_tokens
    source_path = os.path.realpath(os.fspath(model_path))
    source_config_path = os.path.join(source_path, "config.json")
    if not os.path.isfile(source_config_path):
        raise ValueError(
            "DSpark SWA runtime overlay requires a local model directory containing "
            f"config.json, got {model_path!r}"
        )
    with open(source_config_path, encoding="utf-8") as config_file:
        source_config = json.load(config_file)
    if not isinstance(source_config, dict):
        raise ValueError(f"DSpark config must be a JSON object: {source_config_path}")

    overlay_config = dict(source_config)
    dflash_config = dict(overlay_config.get("dflash_config") or {})
    dflash_config.update(
        {
            "use_swa": True,
            "swa_window_size": runtime_window,
            "causal": False,
        }
    )
    overlay_config["dflash_config"] = dflash_config
    overlay_config["sliding_window"] = runtime_window

    digest_payload = json.dumps(
        {
            "source_path": source_path,
            "source_config": source_config,
            "attention_mode": attention_mode,
            "actor_context_window": context_window,
            "draft_query_tokens": query_tokens,
            "runtime_window": runtime_window,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(digest_payload).hexdigest()[:20]
    overlay_root = os.path.realpath(
        os.fspath(
            vllm_cfg.get("dspark_runtime_overlay_root")
            or "/tmp/verl_speco_dspark_attention"
        )
    )
    if os.path.commonpath([source_path, overlay_root]) == source_path:
        raise ValueError(
            "drafter.vllm.dspark_runtime_overlay_root must not be inside the "
            f"source model directory: source={source_path}, overlay_root={overlay_root}"
        )
    os.makedirs(overlay_root, exist_ok=True)
    overlay_path = os.path.join(overlay_root, digest)
    if os.path.isfile(os.path.join(overlay_path, "config.json")):
        return overlay_path

    staging_path = f"{overlay_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    os.makedirs(staging_path, exist_ok=False)
    try:
        for entry in os.scandir(source_path):
            if entry.name == "config.json":
                continue
            os.symlink(entry.path, os.path.join(staging_path, entry.name))
        with open(os.path.join(staging_path, "config.json"), "w", encoding="utf-8") as output:
            json.dump(overlay_config, output, ensure_ascii=True, indent=2, sort_keys=True)
            output.write("\n")
        try:
            os.rename(staging_path, overlay_path)
        except FileExistsError:
            shutil.rmtree(staging_path)
    except BaseException:
        if os.path.isdir(staging_path):
            shutil.rmtree(staging_path)
        raise
    logger.info(
        "Prepared DSpark uniform non-causal SWA runtime overlay: source=%s "
        "actor_context_window=%s draft_query_tokens=%s runtime_window=%s overlay=%s",
        source_path,
        context_window,
        query_tokens,
        runtime_window,
        overlay_path,
    )
    return overlay_path


def _configure_vllm_ascend_dspark_swa_window(
    engine_kwargs: Any,
    drafter_cfg: dict[str, Any],
    speculative_config: dict[str, Any],
) -> int | None:
    """Configure the draft block-table window consumed by vLLM-Ascend."""

    if _drafter_algorithm(drafter_cfg) != "DSPARK":
        return None
    training_cfg = drafter_cfg.get("training") or {}
    if str(training_cfg.get("dspark_rejected_loss", "k3")).strip().lower() == "topk":
        if not training_cfg.get("dspark_trace_aligned_anchors", False):
            raise ValueError("DSpark rejected topk requires trace-aligned training")
        topk = int(training_cfg.get("dspark_topk", 16))
        if topk < 2:
            raise ValueError("DSpark topk must be at least 2")
        additional_config = _ensure_child_mapping(engine_kwargs, "additional_config")
        _set_child(additional_config, "speco_dspark_rejected_topk", topk)
        support = str(training_cfg.get("dspark_topk_support", "student")).strip().lower()
        if support not in {"student", "union"}:
            raise ValueError("DSpark topk_support must be student or union")
        _set_child(additional_config, "speco_dspark_topk_support", support)
    vllm_cfg = drafter_cfg.get("vllm") or {}
    attention_mode = str(
        vllm_cfg.get("dspark_attention_mode", "full") or "full"
    ).strip().lower()
    if attention_mode != "swa":
        return None

    context_window = _positive_int_or_none(vllm_cfg.get("dspark_context_window"))
    query_tokens = _positive_int_or_none(
        speculative_config.get("num_speculative_tokens")
    )
    if context_window is None or query_tokens is None:
        raise ValueError(
            "DSpark SWA requires positive dspark_context_window and "
            "num_speculative_tokens values"
        )
    runtime_window = context_window + query_tokens

    additional_config = _ensure_child_mapping(engine_kwargs, "additional_config")
    if not hasattr(additional_config, "get"):
        raise TypeError("rollout.engine_kwargs.vllm.additional_config must be a mapping")
    existing = additional_config.get("draft_window_size")
    if existing is not None:
        existing_window = _positive_int_or_none(existing)
        if existing_window != runtime_window:
            raise ValueError(
                "DSpark SWA requires vLLM-Ascend additional_config.draft_window_size "
                f"to equal actor context plus draft queries ({context_window} + "
                f"{query_tokens} = {runtime_window}), got {existing!r}"
            )
    else:
        _set_child(additional_config, "draft_window_size", runtime_window)

    logger.info(
        "Configured vLLM-Ascend DSpark SWA block-table adapter: "
        "actor_context_window=%s draft_query_tokens=%s runtime_window=%s",
        context_window,
        query_tokens,
        runtime_window,
    )
    return runtime_window


def _configure_dspark_swa_window_from_runtime_model(
    engine_kwargs: dict[str, Any],
) -> int | None:
    """Reinforce DSpark SWA at verl's final vLLM CLI construction boundary."""

    speculative_config = engine_kwargs.get("speculative_config")
    if isinstance(speculative_config, str):
        speculative_config = json.loads(speculative_config)
    speculative_config = _plain_container(speculative_config)
    if not isinstance(speculative_config, dict):
        return None
    if str(speculative_config.get("method", "")).strip().lower() != "dspark":
        return None

    model_path = speculative_config.get("model")
    if model_path is None:
        return None
    config_path = os.path.join(os.path.realpath(os.fspath(model_path)), "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as config_file:
        model_config = json.load(config_file)
    dflash_config = model_config.get("dflash_config") or {}
    if not bool(_bool_or_none(dflash_config.get("use_swa", False))):
        return None
    runtime_window = _positive_int_or_none(dflash_config.get("swa_window_size"))
    if runtime_window is None:
        raise ValueError(
            f"DSpark SWA runtime model has no positive swa_window_size: {config_path}"
        )

    additional_config = engine_kwargs.get("additional_config") or {}
    if isinstance(additional_config, str):
        additional_config = json.loads(additional_config)
    additional_config = _plain_container(additional_config)
    if not isinstance(additional_config, dict):
        raise TypeError("vLLM additional_config must be a mapping")
    existing = additional_config.get("draft_window_size")
    if existing is not None and _positive_int_or_none(existing) != runtime_window:
        raise ValueError(
            "DSpark SWA runtime model and vLLM additional_config disagree: "
            f"model_window={runtime_window}, draft_window_size={existing!r}"
        )
    additional_config["draft_window_size"] = runtime_window
    engine_kwargs["additional_config"] = additional_config
    logger.warning(
        "[speco dspark swa] prepared final vLLM CLI window: runtime_window=%s",
        runtime_window,
    )
    return runtime_window


def _merge_speculative_config(
    existing: Any, injected: dict[str, Any]
) -> dict[str, Any]:
    if existing in (None, ""):
        return dict(injected)
    if isinstance(existing, str):
        existing = json.loads(existing)
    existing = _plain_container(existing)
    if not isinstance(existing, dict):
        raise TypeError(
            "rollout.engine_kwargs.vllm.speculative_config must be a mapping for SPECO merge"
        )
    merged = dict(injected)
    merged.update(existing)
    return merged


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_env_or_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _get_vllm_spec_decode_logger():
    try:
        from vllm.logger import init_logger

        spec_logger = init_logger("vllm.spec_decode.acceptance")
    except Exception:  # noqa: BLE001
        spec_logger = logging.getLogger("vllm.spec_decode.acceptance")

    level_name = os.getenv("VLLM_LOGGING_LEVEL")
    if level_name:
        try:
            spec_logger.setLevel(level_name.upper())
        except ValueError:
            pass
    return spec_logger


def _is_vllm_ascend_runtime_hint() -> bool:
    env_hints = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
    )
    if any(os.getenv(name) for name in env_hints):
        return True
    if str(os.getenv("VLLM_TARGET_DEVICE", "")).strip().lower() == "npu":
        return True
    if "ascend" in str(os.getenv("VLLM_PLATFORM", "")).strip().lower():
        return True

    try:
        from verl.utils.device import get_device_name

        if str(get_device_name()).lower() == "npu":
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        from vllm.platforms import current_platform

        return (
            str(
                _first_present(
                    getattr(current_platform, "device_type", None),
                    getattr(current_platform, "device_name", None),
                )
            ).lower()
            == "npu"
        )
    except Exception:  # noqa: BLE001
        return "vllm_ascend" in sys.modules and "torch_npu" in sys.modules


def _maybe_apply_vllm_ascend_global_patch() -> bool:
    patch_transformers_attention_layer_type_constants()
    if not _is_vllm_ascend_runtime_hint():
        return False
    try:
        from vllm_ascend.utils import adapt_patch
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to import vLLM-Ascend global patch hook: %s", exc)
        return False

    try:
        adapt_patch(is_global_patch=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to apply vLLM-Ascend global patch hook: %s", exc)
        return False
    return True


def patch_transformers_attention_layer_type_constants() -> bool:
    """Provide the attention layer type aliases expected by mixed vLLM builds."""

    try:
        from transformers import configuration_utils
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install transformers attention layer type compat: %s", exc
        )
        return False

    has_v5_name = hasattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES")
    has_v4_name = hasattr(configuration_utils, "ALLOWED_LAYER_TYPES")
    if has_v5_name and has_v4_name:
        return False

    existing = None
    if has_v5_name:
        existing = getattr(configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES", None)
    elif has_v4_name:
        existing = getattr(configuration_utils, "ALLOWED_LAYER_TYPES", None)

    try:
        allowed_layer_types = (
            tuple(existing)
            if existing is not None
            else _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK
        )
    except TypeError:
        allowed_layer_types = _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK
    if not allowed_layer_types:
        allowed_layer_types = _TRANSFORMERS_ATTENTION_LAYER_TYPES_FALLBACK

    patched_names = []
    if not has_v5_name:
        configuration_utils.ALLOWED_ATTENTION_LAYER_TYPES = allowed_layer_types
        patched_names.append("ALLOWED_ATTENTION_LAYER_TYPES")
    if not has_v4_name:
        configuration_utils.ALLOWED_LAYER_TYPES = allowed_layer_types
        patched_names.append("ALLOWED_LAYER_TYPES")

    logger.warning(
        "[speco vllm compat] patched transformers.configuration_utils missing %s for vLLM import",
        ", ".join(patched_names),
    )
    return True


# Ray imports this module to deserialize SpecoVLLMHttpServer before the normal
# worker runtime hooks run. Install both import guards before any top-level
# verl/vLLM import below, including Worker_TP extension class resolution.
patch_transformers_attention_layer_type_constants()
install_verl_npu_vllm_import_compat()


def _is_dspark_hf_config(hf_config: Any) -> bool:
    architectures = _get_nested(hf_config, ("architectures",), None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture_names = {str(name).replace("DFlash", "", 1) for name in architectures}
    return bool(
        _has_config_field(hf_config, "markov_head_type")
        or _has_config_field(hf_config, "dspark_config")
        or architecture_names.intersection(_DSPARK_VLLM_ARCHITECTURES)
    )


def _ensure_dspark_dflash_aliases(hf_config: Any) -> bool:
    """Make DSpark HF config consumable by vLLM's DFlash draft model."""

    if not _is_dspark_hf_config(hf_config):
        return False

    target_layer_ids = _int_list_or_none(
        _first_present(
            _get_nested(hf_config, ("target_layer_ids",), None),
            _get_nested(hf_config, ("dflash_config", "target_layer_ids"), None),
            _get_nested(hf_config, ("dspark_config", "target_layer_ids"), None),
        ),
        "DSpark target_layer_ids",
    )
    if target_layer_ids is None:
        return False

    changed = False
    dflash_config = _get_nested(hf_config, ("dflash_config",), None)
    if dflash_config is None:
        dflash_config = {}
        _set_child(hf_config, "dflash_config", dflash_config)
        changed = True
    if _get_nested(dflash_config, ("target_layer_ids",), None) is None:
        _set_child(dflash_config, "target_layer_ids", target_layer_ids)
        changed = True
    if _get_nested(dflash_config, ("mask_token_id",), None) is None:
        mask_token_id = _get_nested(hf_config, ("mask_token_id",), None)
        if mask_token_id is not None:
            _set_child(dflash_config, "mask_token_id", mask_token_id)
            changed = True

    if _get_nested(hf_config, ("eagle_aux_hidden_state_layer_ids",), None) is None:
        _set_child(
            hf_config,
            "eagle_aux_hidden_state_layer_ids",
            [layer_id + 1 for layer_id in target_layer_ids],
        )
        changed = True
    return changed


def _dspark_hf_config_from_vllm_config(vllm_config: Any) -> Any:
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    draft_model_cfg = (
        getattr(spec_cfg, "draft_model_config", None) if spec_cfg is not None else None
    )
    return getattr(draft_model_cfg, "hf_config", None)


def _dspark_hf_config_from_proposer(proposer: Any) -> Any:
    draft_model_cfg = getattr(proposer, "draft_model_config", None)
    if draft_model_cfg is not None:
        hf_config = getattr(draft_model_cfg, "hf_config", None)
        if hf_config is not None:
            return hf_config
    spec_cfg = getattr(proposer, "speculative_config", None)
    draft_model_cfg = (
        getattr(spec_cfg, "draft_model_config", None) if spec_cfg is not None else None
    )
    return getattr(draft_model_cfg, "hf_config", None)


def _validate_vllm_ascend_dspark_swa_adapter(proposer: Any) -> bool:
    """Fail closed when a DSpark SWA model starts without block-table cropping."""

    hf_config = _dspark_hf_config_from_proposer(proposer)
    if not _is_dspark_hf_config(hf_config):
        return False
    dflash_config = _get_nested(hf_config, ("dflash_config",), None) or {}
    if not bool(_bool_or_none(_get_nested(dflash_config, ("use_swa",), False))):
        return False

    expected_window = _positive_int_or_none(
        _get_nested(dflash_config, ("swa_window_size",), None)
    )
    configured_window = _positive_int_or_none(
        getattr(proposer, "draft_window_size", None)
    )
    adapter = getattr(proposer, "sliding_window", None)
    adapter_window = _positive_int_or_none(getattr(adapter, "window_size", None))
    grouped_tables = hasattr(proposer, "_per_group_block_tables")
    grouped_builder = getattr(type(proposer), "build_draft_attn_metadata", None)
    grouped_adapter_patched = bool(
        getattr(grouped_builder, "_speco_dspark_grouped_swa", False)
    )
    if (
        expected_window is None
        or configured_window != expected_window
        or adapter_window != expected_window
        or (grouped_tables and not grouped_adapter_patched)
    ):
        raise RuntimeError(
            "DSpark SWA runtime model loaded without a matching vLLM-Ascend "
            "SlidingWindowAdapter: "
            f"model_window={expected_window}, "
            f"draft_window_size={configured_window}, "
            f"adapter_window={adapter_window}, "
            f"grouped_adapter_patched={grouped_adapter_patched}. Set "
            "rollout.engine_kwargs.vllm.additional_config.draft_window_size to "
            "the runtime window (actor context plus draft query tokens)."
        )
    logger.warning(
        "[speco dspark swa] validated vLLM-Ascend SlidingWindowAdapter: "
        "runtime_window=%s grouped_tables=%s",
        expected_window,
        int(grouped_tables),
    )
    return True


def _crop_vllm_ascend_dspark_group_block_tables(proposer: Any) -> int:
    """Apply the active SWA span to every native DSpark KV-cache group."""

    adapter = getattr(proposer, "sliding_window", None)
    if adapter is None or getattr(proposer, "draft_window_size", None) is None:
        return 0
    active_tables = getattr(proposer, "_per_group_block_table_buffers", None)
    if not isinstance(active_tables, dict) or not active_tables:
        raise RuntimeError(
            "DSpark SWA could not find per-group block tables after input expansion"
        )

    block_size = _positive_int_or_none(getattr(adapter, "block_size", None))
    start_tokens = getattr(adapter, "start_tokens_in_window_rounding", None)
    windowed_seq_lens = getattr(adapter, "_windowed_seq_lens", None)
    max_window_blocks = _positive_int_or_none(
        getattr(adapter, "max_window_blocks", None)
    )
    if (
        block_size is None
        or start_tokens is None
        or windowed_seq_lens is None
        or max_window_blocks is None
    ):
        raise RuntimeError(
            "DSpark SWA grouped adapter ran before the primary window was applied"
        )

    group_block_sizes = getattr(proposer, "_per_group_kernel_block_sizes", {}) or {}
    mismatched_groups = {
        int(group_id): int(group_size)
        for group_id, group_size in group_block_sizes.items()
        if _positive_int_or_none(group_size) != block_size
    }
    if mismatched_groups:
        raise RuntimeError(
            "DSpark SWA currently requires identical logical block sizes across "
            f"draft KV-cache groups; primary={block_size}, mismatched={mismatched_groups}"
        )

    import torch

    num_reqs = int(windowed_seq_lens.shape[0])
    start_block_indices = start_tokens[:num_reqs] // block_size
    needed_blocks = (windowed_seq_lens[:num_reqs] + block_size - 1) // block_size
    needed_blocks = torch.clamp(needed_blocks, max=max_window_blocks).unsqueeze(1)
    cropped_buffers = getattr(
        proposer, "_speco_dspark_swa_group_block_table_buffers", None
    )
    if not isinstance(cropped_buffers, dict):
        cropped_buffers = {}

    for group_id, full_table in active_tables.items():
        if full_table.ndim != 2 or full_table.shape[1] <= 0:
            raise RuntimeError(
                f"DSpark SWA received an invalid block table for group {group_id}: "
                f"shape={tuple(full_table.shape)}"
            )
        output_shape = (full_table.shape[0], max_window_blocks)
        output = cropped_buffers.get(group_id)
        if (
            output is None
            or tuple(output.shape) != output_shape
            or output.device != full_table.device
            or output.dtype != full_table.dtype
        ):
            output = full_table.new_zeros(output_shape)
            cropped_buffers[group_id] = output

        columns = torch.arange(
            max_window_blocks, device=full_table.device
        ).unsqueeze(0)
        source_columns = start_block_indices.unsqueeze(1) + columns
        gathered = torch.gather(
            full_table[:num_reqs],
            1,
            source_columns.clamp(max=full_table.shape[1] - 1),
        )
        valid = (columns < needed_blocks) & (source_columns < full_table.shape[1])
        output[:num_reqs].copy_(gathered * valid.to(gathered.dtype))

    proposer._speco_dspark_swa_group_block_table_buffers = cropped_buffers
    proposer._per_group_block_table_buffers = {
        group_id: cropped_buffers[group_id] for group_id in active_tables
    }
    if not bool(getattr(proposer, "_speco_dspark_swa_crop_logged", False)):
        logger.warning(
            "[speco dspark swa] applied grouped block-table crop: "
            "runtime_window=%s block_size=%s groups=%s",
            getattr(adapter, "window_size", None),
            block_size,
            len(active_tables),
        )
        proposer._speco_dspark_swa_crop_logged = True
    return len(active_tables)


def _patch_vllm_ascend_dspark_grouped_swa() -> bool:
    try:
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install native vLLM-Ascend DSpark grouped SWA adapter: %s",
            exc,
        )
        return False

    current = getattr(AscendDSparkProposer, "build_draft_attn_metadata", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_grouped_swa", False):
        return True

    def patched_build_draft_attn_metadata(self, *args, **kwargs):
        if getattr(self, "draft_window_size", None) is not None:
            _crop_vllm_ascend_dspark_group_block_tables(self)
        return current(self, *args, **kwargs)

    setattr(patched_build_draft_attn_metadata, "_speco_dspark_grouped_swa", True)
    setattr(patched_build_draft_attn_metadata, "_speco_original_builder", current)
    AscendDSparkProposer.build_draft_attn_metadata = patched_build_draft_attn_metadata
    return True


def _patch_vllm_ascend_dspark_swa_adapter_validation() -> bool:
    try:
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install vLLM-Ascend DSpark SWA adapter validation: %s", exc
        )
        return False

    current = getattr(AscendSpecDecodeBaseProposer, "load_model", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_swa_adapter_validation", False):
        return True

    def patched_load_model(self, *args, **kwargs):
        result = current(self, *args, **kwargs)
        _validate_vllm_ascend_dspark_swa_adapter(self)
        return result

    setattr(
        patched_load_model,
        "_speco_dspark_swa_adapter_validation",
        True,
    )
    setattr(
        patched_load_model,
        "_speco_original_load_model",
        current,
    )
    AscendSpecDecodeBaseProposer.load_model = patched_load_model
    return True


def _patch_vllm_dspark_parallel_token() -> bool:
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark parallel-token patch: %s", exc)
        return False

    current = getattr(SpecDecodeBaseProposer, "_init_parallel_drafting_params", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_parallel_token", False):
        return True

    def patched_init_parallel_drafting_params(self):
        model_hf_config = self.draft_model_config.hf_config
        if _is_dspark_hf_config(model_hf_config):
            _ensure_dspark_dflash_aliases(model_hf_config)
            mask_token_id = _get_nested(model_hf_config, ("mask_token_id",), None)
            if mask_token_id is not None:
                self.parallel_drafting_token_id = int(mask_token_id)
                return
        current(self)

    setattr(patched_init_parallel_drafting_params, "_speco_dspark_parallel_token", True)
    setattr(
        patched_init_parallel_drafting_params,
        "_speco_original_init_parallel_drafting_params",
        current,
    )
    SpecDecodeBaseProposer._init_parallel_drafting_params = (
        patched_init_parallel_drafting_params
    )
    return True


def _patch_vllm_dspark_qwen3_heads() -> bool:
    try:
        import torch
        from torch import nn
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark Qwen3 head patch: %s", exc)
        return False

    class DSparkConfidenceHead(nn.Module):
        def __init__(self, vllm_config: Any, prefix: str) -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            rank = int(
                getattr(
                    config, "markov_rank", getattr(config, "dspark_markov_rank", 256)
                )
            )
            self.proj = ReplicatedLinear(
                config.hidden_size + rank,
                1,
                bias=True,
                params_dtype=torch.float32,
                quant_config=None,
                prefix=f"{prefix}.proj",
            )

        def forward(self, hidden_states: Any, markov_embeds: Any) -> Any:
            x = torch.cat([hidden_states, markov_embeds], dim=-1)
            confidence, _ = self.proj(x.float())
            return confidence.squeeze(-1)

    class DSparkMarkovHead(nn.Module):
        def __init__(self, vllm_config: Any, prefix: str) -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            rank = int(
                getattr(
                    config, "markov_rank", getattr(config, "dspark_markov_rank", 256)
                )
            )
            self.markov_w1 = VocabParallelEmbedding(
                config.vocab_size,
                rank,
                prefix=f"{prefix}.markov_w1",
            )
            self.markov_w2 = ParallelLMHead(
                config.vocab_size,
                rank,
                params_dtype=torch.float32,
                org_num_embeddings=config.vocab_size,
                prefix=f"{prefix}.markov_w2",
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)

        def forward(self, token_ids: Any) -> tuple[Any, Any]:
            embeds = self.markov_w1(token_ids)
            logits = self.logits_processor(
                self.markov_w2,
                embeds.view(-1, embeds.shape[-1]).float(),
            )
            return logits.view(*embeds.shape[:-1], -1), embeds

    current = getattr(DFlashQwen3Model, "__init__", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dspark_qwen3_heads", False):
        return True

    def patched_dflash_qwen3_init(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        prefix = kwargs.get("prefix", "")
        hf_config = _dspark_hf_config_from_vllm_config(vllm_config)
        is_dspark = _is_dspark_hf_config(hf_config)
        if is_dspark:
            _ensure_dspark_dflash_aliases(hf_config)

        current(self, *args, **kwargs)

        if is_dspark:
            if not hasattr(self, "markov_head"):
                self.markov_head = DSparkMarkovHead(
                    vllm_config, prefix=f"{prefix}.markov_head"
                )
            if not hasattr(self, "confidence_head"):
                self.confidence_head = DSparkConfidenceHead(
                    vllm_config, prefix=f"{prefix}.confidence_head"
                )

    setattr(patched_dflash_qwen3_init, "_speco_dspark_qwen3_heads", True)
    setattr(patched_dflash_qwen3_init, "_speco_original_dflash_qwen3_init", current)
    DFlashQwen3Model.__init__ = patched_dflash_qwen3_init
    return True


def _import_vllm_ascend_dspark_patch() -> bool:
    try:
        import importlib

        importlib.import_module("vllm_ascend.patch.platform.patch_dspark_proposer")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to import vLLM-Ascend DSpark platform patch: %s", exc)
        return False
    return True


def _source_contains_all(obj: Any, markers: tuple[str, ...]) -> bool:
    if obj is None:
        return False
    try:
        import inspect

        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return False
    return all(marker in source for marker in markers)


def _vllm_ascend_has_dspark_pr11153_k_query_runtime() -> bool:
    """Detect the latest PR #11153 DSpark layout in vLLM-Ascend.

    The current PR logic differs from the older fallback in two important ways:
    DSpark uses K query tokens per request instead of DFlash's K+1, and the
    vLLM-Ascend proposer samples the anchor position plus K-1 mask positions.
    """

    try:
        import vllm_ascend.spec_decode.dflash_proposer as dflash_module
        import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
    except Exception:  # noqa: BLE001
        return False

    dflash_cls = getattr(dflash_module, "AscendDflashProposer", None)
    if dflash_cls is None:
        return False

    has_k_query_proposer = _source_contains_all(
        getattr(dflash_cls, "_num_query_per_req", None),
        (
            "self._is_dspark",
            "self.num_speculative_tokens",
            "1 + self.num_speculative_tokens",
        ),
    ) and _source_contains_all(
        getattr(dflash_cls, "set_inputs_first_pass", None),
        ("_num_query_per_req", "IS_DSPARK"),
    )
    if not has_k_query_proposer:
        return False

    for candidate in vars(proposer_module).values():
        if not isinstance(candidate, type) or "Proposer" not in candidate.__name__:
            continue
        method = getattr(candidate, "_run_merged_draft", None)
        if _source_contains_all(
            method,
            (
                "markov_head_type",
                "blk = self.num_speculative_tokens",
                "markov_head",
                "draft_token_ids[:, 1:]",
            ),
        ):
            return True
    return False


def patch_vllm_dspark_runtime() -> bool:
    """Install DSpark hooks for vLLM-Ascend PR #11153's K-query runtime."""

    global _VLLM_DSPARK_RUNTIME_PATCHED

    patched = bool(_VLLM_DSPARK_RUNTIME_PATCHED)
    # These adapters are also required by newer vLLM-Ascend releases with a
    # native AscendDSparkProposer, which no longer match the PR #11153 detector.
    patched = _patch_vllm_ascend_dspark_grouped_swa() or patched
    patched = _patch_vllm_ascend_dspark_swa_adapter_validation() or patched

    ascend_has_pr11153_k_query = _vllm_ascend_has_dspark_pr11153_k_query_runtime()
    if not ascend_has_pr11153_k_query:
        logger.debug(
            "vLLM-Ascend DSpark runtime does not match PR #11153's latest K-query "
            "layout; SpeCo will not install legacy DSpark fallback patches."
        )
        if patched:
            _VLLM_DSPARK_RUNTIME_PATCHED = True
        return patched

    patched = _import_vllm_ascend_dspark_patch() or patched
    patched = _patch_vllm_dspark_parallel_token() or patched
    patched = _patch_vllm_dspark_qwen3_heads() or patched
    if patched:
        _VLLM_DSPARK_RUNTIME_PATCHED = True
    return patched


def _record_vllm_spec_decode_acceptance(
    scheduler: Any,
    *,
    request_id: Any,
    num_draft_tokens: Any,
    num_accepted_tokens: Any,
    num_invalid_spec_tokens: Any,
) -> None:
    if not getattr(scheduler, "log_stats", True):
        return

    del request_id, num_invalid_spec_tokens
    draft_tokens = _int_or_zero(num_draft_tokens)
    accepted = max(0, _int_or_zero(num_accepted_tokens))
    if draft_tokens <= 0:
        return

    total_drafts = int(getattr(scheduler, "_speco_spec_decode_log_drafts", 0)) + 1
    total_accepted = (
        int(getattr(scheduler, "_speco_spec_decode_log_accepted", 0)) + accepted
    )
    scheduler._speco_spec_decode_log_drafts = total_drafts
    scheduler._speco_spec_decode_log_accepted = total_accepted

    now = time.monotonic()
    last_log_time = float(getattr(scheduler, "_speco_spec_decode_last_log_time", 0.0))
    interval = _float_env_or_default(SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_ENV, 10.0)
    if last_log_time > 0.0 and interval > 0.0 and now - last_log_time < interval:
        return

    spec_logger = getattr(scheduler, "_speco_spec_decode_logger", None)
    if spec_logger is None:
        spec_logger = _get_vllm_spec_decode_logger()
        scheduler._speco_spec_decode_logger = spec_logger

    spec_logger.info(
        "[speco vllm spec decode] mean_acceptance_length=%.3f",
        1.0 + total_accepted / max(1, total_drafts),
    )
    scheduler._speco_spec_decode_last_log_time = now
    scheduler._speco_spec_decode_log_drafts = 0
    scheduler._speco_spec_decode_log_accepted = 0


def _compute_vllm_draft_teacher_statistics(
    logits: Any, metadata: Any, *, calculate_entropy: bool = False
) -> tuple[Any, Any]:
    """Gather raw actor draft log-probs and optionally full-vocabulary entropy."""

    import torch

    target_logits = logits[metadata.target_logits_indices].float()
    draft_ids = metadata.draft_token_ids.long().reshape(-1)
    if target_logits.size(0) != draft_ids.numel():
        raise ValueError(
            "vLLM verify logits and flattened draft token counts do not match"
        )
    selected_logits = target_logits.gather(-1, draft_ids.unsqueeze(-1)).squeeze(-1)
    logprobs = selected_logits - torch.logsumexp(target_logits, dim=-1)
    if not calculate_entropy:
        return logprobs, None
    # Chunk the full-vocabulary reduction to bound temporary device memory.
    entropies = []
    for chunk in target_logits.split(32, dim=0):
        log_partition = torch.logsumexp(chunk, dim=-1, keepdim=True)
        log_probabilities = chunk - log_partition
        probabilities = torch.exp(log_probabilities)
        # Masked vocabulary logits are -inf; 0 * -inf would yield NaN.
        finite_log_probabilities = torch.where(
            torch.isfinite(log_probabilities),
            log_probabilities,
            torch.zeros_like(log_probabilities),
        )
        entropies.append(
            (-(probabilities * finite_log_probabilities).sum(-1)).clamp_min(0)
        )
    return logprobs, torch.cat(entropies)


def _compute_vllm_draft_teacher_logprobs(logits: Any, metadata: Any) -> Any:
    """Gather raw target-model log-probabilities for flattened draft tokens."""

    return _compute_vllm_draft_teacher_statistics(logits, metadata)[0]


def _build_vllm_speculative_trace_events(
    scheduler: Any, scheduler_output: Any, model_runner_output: Any
) -> dict[str, list[dict[str, Any]]]:
    """Build request-aligned trace events from the actual vLLM verify step."""

    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None) or {}
    if not scheduled or sampled is None:
        return {}

    traces: dict[str, list[dict[str, Any]]] = {}
    for request_id, raw_draft_ids in scheduled.items():
        verified_drafts = getattr(
            model_runner_output,
            "_speco_verified_draft_token_ids_by_request",
            None,
        )
        verified_teacher_logprobs = getattr(
            model_runner_output,
            "_speco_verified_draft_teacher_logprobs_by_request",
            None,
        )
        verified_teacher_entropies = getattr(
            model_runner_output,
            "_speco_verified_draft_teacher_entropies_by_request",
            None,
        )
        if isinstance(verified_drafts, dict):
            raw_draft_ids = verified_drafts.get(
                request_id, verified_drafts.get(str(request_id), raw_draft_ids)
            )
        if raw_draft_ids is None or request_id not in req_id_to_index:
            continue
        try:
            if len(raw_draft_ids) <= 0:
                continue
        except TypeError:
            continue
        request = getattr(scheduler, "requests", {}).get(request_id)
        is_finished = getattr(request, "is_finished", None)
        if request is None or bool(
            is_finished() if callable(is_finished) else is_finished
        ):
            continue
        try:
            generated_ids = [
                int(token_id) for token_id in sampled[int(req_id_to_index[request_id])]
            ]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if not generated_ids:
            continue

        draft_ids = [int(token_id) for token_id in raw_draft_ids if int(token_id) >= 0]
        draft_teacher_logprobs: list[float] = []
        if isinstance(verified_teacher_logprobs, dict):
            raw_teacher_logprobs = verified_teacher_logprobs.get(
                request_id, verified_teacher_logprobs.get(str(request_id))
            )
            if isinstance(raw_teacher_logprobs, (list, tuple)):
                try:
                    draft_teacher_logprobs = [
                        float(value) for value in raw_teacher_logprobs
                    ][: len(draft_ids)]
                except (TypeError, ValueError):
                    draft_teacher_logprobs = []
        draft_teacher_entropies: list[float] = []
        if isinstance(verified_teacher_entropies, dict):
            raw_entropies = verified_teacher_entropies.get(
                request_id, verified_teacher_entropies.get(str(request_id))
            )
            if isinstance(raw_entropies, (list, tuple)):
                draft_teacher_entropies = [
                    float(value) for value in raw_entropies[: len(draft_ids)]
                ]
        num_sampled = max(
            _int_or_zero(getattr(scheduler, "num_sampled_tokens_per_step", 1)), 0
        )
        accepted_count = min(
            max(len(generated_ids) - num_sampled, 0), len(draft_ids)
        )
        all_drafts_accepted = accepted_count == len(draft_ids)
        output_ids = getattr(request, "_output_token_ids", None)
        if output_ids is None:
            output_ids = getattr(request, "output_token_ids", ())
        try:
            response_start = len(output_ids)
        except TypeError:
            response_start = 0
        traces[str(request_id)] = [
            {
                "response_start": int(response_start),
                "draft_token_ids": draft_ids,
                "draft_teacher_logprobs": draft_teacher_logprobs,
                "verified_token_ids": generated_ids,
                "accepted_token_ids": generated_ids[:accepted_count],
                "accepted_count": int(accepted_count),
                "draft_count": int(len(draft_ids)),
                "all_drafts_accepted": bool(all_drafts_accepted),
                "verification_token_kind": (
                    "bonus" if all_drafts_accepted else "replacement"
                ),
                "rejected_draft_token_id": (
                    draft_ids[accepted_count] if not all_drafts_accepted else None
                ),
                "verification_token_id": (
                    generated_ids[accepted_count]
                    if accepted_count < len(generated_ids)
                    else None
                ),
            }
        ]
        if draft_teacher_entropies:
            traces[str(request_id)][0]["draft_teacher_entropies"] = draft_teacher_entropies
        topk_by_request = getattr(model_runner_output, "_speco_verified_draft_topk_by_request", {})
        topk_payload = topk_by_request.get(str(request_id))
        if topk_payload is not None:
            traces[str(request_id)][0].update(topk_payload)
    return traces


def _build_vllm_speculative_trace_diagnostics(
    scheduler: Any,
    scheduler_output: Any,
    model_runner_output: Any,
    traces: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Describe trace construction per request without relying on trace success."""

    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None) or {}
    requests = getattr(scheduler, "requests", {}) or {}
    diagnostics: dict[str, dict[str, int]] = {}
    for request_id, raw_draft_ids in scheduled.items():
        request_key = str(request_id)
        verified_drafts = getattr(
            model_runner_output,
            "_speco_verified_draft_token_ids_by_request",
            None,
        )
        verified_draft_ids = (
            verified_drafts.get(request_id, verified_drafts.get(request_key))
            if isinstance(verified_drafts, dict)
            else None
        )
        verified_teacher_logprobs = getattr(
            model_runner_output,
            "_speco_verified_draft_teacher_logprobs_by_request",
            None,
        )
        teacher_logprobs = (
            verified_teacher_logprobs.get(
                request_id, verified_teacher_logprobs.get(request_key)
            )
            if isinstance(verified_teacher_logprobs, dict)
            else None
        )
        if verified_draft_ids is not None:
            raw_draft_ids = verified_draft_ids
        try:
            draft_count = len(raw_draft_ids) if raw_draft_ids is not None else 0
        except TypeError:
            draft_count = 0
        req_index = req_id_to_index.get(request_id)
        sampled_count = 0
        if sampled is not None and req_index is not None:
            try:
                sampled_count = len(sampled[int(req_index)])
            except (IndexError, TypeError, ValueError):
                sampled_count = 0
        request = requests.get(request_id)
        is_finished = getattr(request, "is_finished", None)
        request_finished = bool(
            is_finished() if callable(is_finished) else is_finished
        )
        diagnostics[request_key] = {
            "engine_patch_active": 1,
            "engine_scheduled_requests": 1,
            "engine_scheduled_draft_tokens": int(max(draft_count, 0)),
            "engine_request_present": int(request is not None),
            "engine_request_finished": int(request_finished),
            "engine_request_index_present": int(req_index is not None),
            "engine_sampled_tokens": int(max(sampled_count, 0)),
            "engine_verified_draft_ids_present": int(
                verified_draft_ids is not None
            ),
            "engine_teacher_logprobs_present": int(teacher_logprobs is not None),
            "engine_teacher_logprob_tokens": int(
                len(teacher_logprobs)
                if isinstance(teacher_logprobs, (list, tuple))
                else 0
            ),
            "engine_built_events": int(len(traces.get(request_key, ()))),
        }
    return diagnostics


def _attach_vllm_speculative_trace_events(
    engine_outputs: Any,
    traces: dict[str, list[dict[str, Any]]],
    diagnostics: dict[str, dict[str, int]] | None = None,
) -> dict[str, int]:
    attach_stats = {
        "engine_output_requests": 0,
        "engine_attached_requests": 0,
        "engine_attached_events": 0,
        "engine_attach_misses": 0,
    }
    if not isinstance(engine_outputs, dict):
        attach_stats["engine_attach_misses"] = len(diagnostics or traces)
        return attach_stats
    matched_requests: set[str] = set()
    for output_group in engine_outputs.values():
        for output in getattr(output_group, "outputs", ()) or ():
            request_id = str(getattr(output, "request_id", ""))
            if not request_id:
                continue
            attach_stats["engine_output_requests"] += 1
            events = traces.get(request_id) or []
            diagnostic = dict((diagnostics or {}).get(request_id) or {})
            if not events and not diagnostic:
                continue
            matched_requests.add(request_id)
            emitted_token_ids = [
                int(token_id)
                for token_id in (getattr(output, "new_token_ids", None) or [])
            ]
            events = [
                {**event, "emitted_token_ids": emitted_token_ids} for event in events
            ]
            transfer = getattr(output, "kv_transfer_params", None)
            transfer = dict(transfer) if isinstance(transfer, dict) else {}
            if events:
                transfer[_SPECO_VLLM_TRACE_TRANSFER_KEY] = events
            diagnostic["engine_attached_requests"] = 1
            diagnostic["engine_attached_events"] = len(events)
            diagnostic["engine_emitted_tokens"] = len(emitted_token_ids)
            transfer[_SPECO_VLLM_TRACE_DIAGNOSTICS_TRANSFER_KEY] = diagnostic
            output.kv_transfer_params = transfer
            trace_headers = getattr(output, "trace_headers", None)
            trace_headers = dict(trace_headers) if trace_headers is not None else {}
            if events:
                trace_headers[_SPECO_VLLM_TRACE_HEADER_KEY] = json.dumps(
                    events, separators=(",", ":")
                )
            trace_headers[_SPECO_VLLM_TRACE_DIAGNOSTICS_HEADER_KEY] = json.dumps(
                diagnostic, separators=(",", ":")
            )
            output.trace_headers = trace_headers
            attach_stats["engine_attached_requests"] += 1
            attach_stats["engine_attached_events"] += len(events)
    expected_requests = set((diagnostics or traces).keys())
    attach_stats["engine_attach_misses"] = len(expected_requests - matched_requests)
    return attach_stats


def _external_vllm_request_id(request_id: Any) -> str:
    request_id = str(request_id)
    external_id, separator, random_suffix = request_id.rpartition("-")
    if separator and len(random_suffix) == 8:
        try:
            int(random_suffix, 16)
        except ValueError:
            pass
        else:
            return external_id
    return request_id


def _vllm_trace_spool_path(request_id: Any) -> str:
    spool_dir = os.getenv(
        SPECO_VLLM_TRACE_SPOOL_DIR_ENV, SPECO_VLLM_TRACE_SPOOL_DIR_DEFAULT
    ).strip()
    request_digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
    return os.path.join(spool_dir, f"{request_digest}.json")


def _write_vllm_trace_spool(
    request_id: Any,
    events: list[dict[str, Any]],
    diagnostics: dict[str, int],
) -> bool:
    external_request_id = _external_vllm_request_id(request_id)
    path = _vllm_trace_spool_path(external_request_id)
    directory = os.path.dirname(path)
    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "request_id": str(request_id),
                    "external_request_id": external_request_id,
                    "events": events,
                    "diagnostics": diagnostics,
                },
                output_file,
                separators=(",", ":"),
            )
        os.replace(temporary, path)
        return True
    except OSError as exc:
        logger.warning("Unable to persist vLLM speculative trace spool: %s", exc)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return False


def _pop_vllm_trace_spool(
    request_id: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if request_id is None:
        return [], {}
    path = _vllm_trace_spool_path(request_id)
    try:
        with open(path, encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except FileNotFoundError:
        return [], {}
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Unable to read vLLM speculative trace spool: %s", exc)
        payload = {}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not isinstance(payload, dict) or payload.get("external_request_id") != str(
        request_id
    ):
        return [], {}
    events = payload.get("events")
    diagnostics = payload.get("diagnostics")
    return (
        list(events) if isinstance(events, list) else [],
        dict(diagnostics) if isinstance(diagnostics, dict) else {},
    )


def _persist_finished_vllm_speculative_traces(
    engine_outputs: Any,
    pending_traces: dict[str, list[dict[str, Any]]],
    pending_diagnostics: dict[str, dict[str, int]],
) -> None:
    if not isinstance(engine_outputs, dict):
        return
    for output_group in engine_outputs.values():
        for output in getattr(output_group, "outputs", ()) or ():
            if getattr(output, "finish_reason", None) is None:
                continue
            request_id = str(getattr(output, "request_id", ""))
            if not request_id:
                continue
            events = pending_traces.pop(request_id, [])
            diagnostics = pending_diagnostics.pop(request_id, {})
            diagnostics["engine_spooled_events"] = len(events)
            _write_vllm_trace_spool(request_id, events, diagnostics)


def patch_vllm_speculative_trace_capture() -> bool:
    """Capture real per-request draft/verify events in the EngineCore process."""

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM speculative trace patch: %s", exc)
        return False
    original = getattr(Scheduler, "update_from_output", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_speculative_trace_capture", False):
        return True

    def update_from_output_with_speco_trace(
        self, scheduler_output, model_runner_output
    ):
        scheduled = (
            getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        )
        traces = _build_vllm_speculative_trace_events(
            self, scheduler_output, model_runner_output
        )
        diagnostics = _build_vllm_speculative_trace_diagnostics(
            self, scheduler_output, model_runner_output, traces
        )
        pending_traces = getattr(self, "_speco_pending_traces_by_request", None)
        if not isinstance(pending_traces, dict):
            pending_traces = {}
            self._speco_pending_traces_by_request = pending_traces
        pending_diagnostics = getattr(
            self, "_speco_pending_trace_diagnostics_by_request", None
        )
        if not isinstance(pending_diagnostics, dict):
            pending_diagnostics = {}
            self._speco_pending_trace_diagnostics_by_request = pending_diagnostics
        for request_id, events in traces.items():
            pending_traces.setdefault(request_id, []).extend(events)
        for request_id, diagnostic in diagnostics.items():
            request_diagnostics = pending_diagnostics.setdefault(request_id, {})
            _merge_vllm_trace_diagnostics(request_diagnostics, diagnostic)
        result = original(self, scheduler_output, model_runner_output)
        attach_stats = _attach_vllm_speculative_trace_events(
            result, traces, diagnostics
        )
        _persist_finished_vllm_speculative_traces(
            result, pending_traces, pending_diagnostics
        )
        if scheduled and bool(
            _bool_or_none(os.getenv(SPECO_VLLM_DRAFT_DIAG_ENV))
        ):
            sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
            logger.warning(
                "[speco vllm trace boundary] scheduler=%s scheduled_requests=%s "
                "scheduled_draft_tokens=%s sampled_rows=%s built_events=%s "
                "output_groups=%s attached_events=%s attach_misses=%s",
                type(self).__name__,
                len(scheduled),
                sum(len(tokens or ()) for tokens in scheduled.values()),
                len(sampled),
                sum(len(events) for events in traces.values()),
                len(result) if isinstance(result, dict) else -1,
                attach_stats["engine_attached_events"],
                attach_stats["engine_attach_misses"],
            )
        if diagnostics:
            totals = getattr(self, "_speco_trace_diagnostics_totals", None)
            if not isinstance(totals, dict):
                totals = {}
                self._speco_trace_diagnostics_totals = totals
            totals["built"] = int(totals.get("built", 0)) + sum(
                len(events) for events in traces.values()
            )
            totals["attached"] = int(totals.get("attached", 0)) + int(
                attach_stats["engine_attached_events"]
            )
            totals["attach_miss"] = int(totals.get("attach_miss", 0)) + int(
                attach_stats["engine_attach_misses"]
            )
            now = time.monotonic()
            last_log = float(getattr(self, "_speco_trace_last_log_time", 0.0))
            interval = _float_env_or_default(
                SPECO_VLLM_SPEC_DECODE_LOG_INTERVAL_ENV, 10.0
            )
            if last_log <= 0.0 or interval <= 0.0 or now - last_log >= interval:
                logger.info(
                    "[speco vllm trace] engine_patch_active=1 built=%s attached=%s attach_miss=%s",
                    totals["built"],
                    totals["attached"],
                    totals["attach_miss"],
                )
                self._speco_trace_last_log_time = now
        return result

    setattr(
        update_from_output_with_speco_trace, "_speco_speculative_trace_capture", True
    )
    setattr(
        update_from_output_with_speco_trace,
        "_speco_original_update_from_output",
        original,
    )
    Scheduler.update_from_output = update_from_output_with_speco_trace
    return True


def _merge_vllm_trace_diagnostics(
    target: dict[str, int], current: dict[str, Any]
) -> None:
    for key, value in current.items():
        try:
            target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        except (TypeError, ValueError):
            continue


def _record_frontend_vllm_speculative_traces(engine_core_outputs: Any) -> None:
    for output in engine_core_outputs or ():
        transfer = getattr(output, "kv_transfer_params", None)
        transfer = dict(transfer) if isinstance(transfer, dict) else {}
        events = transfer.pop(_SPECO_VLLM_TRACE_TRANSFER_KEY, None)
        diagnostic = transfer.pop(_SPECO_VLLM_TRACE_DIAGNOSTICS_TRANSFER_KEY, None)
        trace_headers = getattr(output, "trace_headers", None)
        trace_headers = dict(trace_headers) if trace_headers is not None else {}
        encoded_events = trace_headers.pop(_SPECO_VLLM_TRACE_HEADER_KEY, None)
        encoded_diagnostic = trace_headers.pop(
            _SPECO_VLLM_TRACE_DIAGNOSTICS_HEADER_KEY, None
        )
        if not events and encoded_events:
            try:
                events = json.loads(encoded_events)
            except (TypeError, ValueError):
                events = None
        if not isinstance(diagnostic, dict) and encoded_diagnostic:
            try:
                diagnostic = json.loads(encoded_diagnostic)
            except (TypeError, ValueError):
                diagnostic = None
        output.kv_transfer_params = transfer or None
        output.trace_headers = trace_headers or None
        if not events and not isinstance(diagnostic, dict):
            continue
        request_id = _external_vllm_request_id(
            getattr(output, "request_id", "")
        )
        if not request_id:
            continue
        with _VLLM_SPEC_TRACE_LOCK:
            if events:
                _VLLM_SPEC_TRACES_BY_REQUEST.setdefault(request_id, []).extend(events)
            request_diagnostics = _VLLM_SPEC_TRACE_DIAGNOSTICS_BY_REQUEST.setdefault(
                request_id, {}
            )
            if isinstance(diagnostic, dict):
                _merge_vllm_trace_diagnostics(request_diagnostics, diagnostic)
            request_diagnostics["frontend_received_requests"] = int(
                request_diagnostics.get("frontend_received_requests", 0)
            ) + 1
            request_diagnostics["frontend_received_events"] = int(
                request_diagnostics.get("frontend_received_events", 0)
            ) + len(events or ())
        if bool(_bool_or_none(os.getenv(SPECO_VLLM_DRAFT_DIAG_ENV))):
            logger.warning(
                "[speco vllm trace boundary] frontend request_id=%s events=%s "
                "diagnostic_keys=%s",
                request_id,
                len(events or ()),
                sorted(diagnostic) if isinstance(diagnostic, dict) else [],
            )


def _pop_frontend_vllm_speculative_trace(request_id: Any) -> list[dict[str, Any]]:
    with _VLLM_SPEC_TRACE_LOCK:
        return _VLLM_SPEC_TRACES_BY_REQUEST.pop(str(request_id), [])


def _pop_frontend_vllm_speculative_trace_diagnostics(
    request_id: Any,
) -> dict[str, int]:
    with _VLLM_SPEC_TRACE_LOCK:
        return _VLLM_SPEC_TRACE_DIAGNOSTICS_BY_REQUEST.pop(str(request_id), {})


def patch_vllm_output_processor_speculative_trace() -> bool:
    """Collect EngineCore trace payloads in the frontend serving the request."""

    try:
        from vllm.v1.engine.output_processor import OutputProcessor
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM trace output-processor patch: %s", exc)
        return False
    original = getattr(OutputProcessor, "process_outputs", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_speculative_trace_capture", False):
        return True

    def process_outputs_with_speco_trace(self, engine_core_outputs, *args, **kwargs):
        _record_frontend_vllm_speculative_traces(engine_core_outputs)
        return original(self, engine_core_outputs, *args, **kwargs)

    setattr(process_outputs_with_speco_trace, "_speco_speculative_trace_capture", True)
    setattr(
        process_outputs_with_speco_trace, "_speco_original_process_outputs", original
    )
    OutputProcessor.process_outputs = process_outputs_with_speco_trace
    return True


def patch_vllm_core_client_speculative_trace() -> bool:
    """Collect trace payloads immediately after EngineCore deserialization."""

    try:
        from vllm.v1.engine.core_client import AsyncMPClient
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM trace core-client patch: %s", exc)
        return False
    original = getattr(AsyncMPClient, "get_output_async", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_speculative_trace_capture", False):
        return True

    async def get_output_async_with_speco_trace(self, *args, **kwargs):
        outputs = await original(self, *args, **kwargs)
        engine_core_outputs = getattr(outputs, "outputs", None) or ()
        _record_frontend_vllm_speculative_traces(engine_core_outputs)
        return outputs

    setattr(
        get_output_async_with_speco_trace, "_speco_speculative_trace_capture", True
    )
    setattr(
        get_output_async_with_speco_trace,
        "_speco_original_get_output_async",
        original,
    )
    AsyncMPClient.get_output_async = get_output_async_with_speco_trace
    return True


def patch_vllm_async_llm_speculative_trace() -> bool:
    """Collect trace metadata from the RequestOutput stream seen by verl."""

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM AsyncLLM trace patch: %s", exc)
        return False
    original = getattr(AsyncLLM, "generate", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_speculative_trace_capture", False):
        return True

    async def generate_with_speco_trace(self, *args, **kwargs):
        async for request_output in original(self, *args, **kwargs):
            _record_frontend_vllm_speculative_traces((request_output,))
            yield request_output

    setattr(generate_with_speco_trace, "_speco_speculative_trace_capture", True)
    setattr(
        generate_with_speco_trace,
        "_speco_original_generate",
        original,
    )
    AsyncLLM.generate = generate_with_speco_trace
    return True


def _install_vllm_trace_on_async_llm(engine: Any) -> bool:
    """Install a trace collector on the concrete frontend engine instance."""

    output_processor = getattr(engine, "output_processor", None)
    original = getattr(output_processor, "process_outputs", None)
    if output_processor is None or not callable(original):
        return False
    if getattr(original, "_speco_instance_trace_capture", False):
        return True

    def process_outputs_with_speco_instance_trace(
        engine_core_outputs, *args, **kwargs
    ):
        _record_frontend_vllm_speculative_traces(engine_core_outputs)
        return original(engine_core_outputs, *args, **kwargs)

    setattr(
        process_outputs_with_speco_instance_trace,
        "_speco_instance_trace_capture",
        True,
    )
    try:
        output_processor.process_outputs = process_outputs_with_speco_instance_trace
    except (AttributeError, TypeError):
        processor_cls = type(output_processor)
        class_original = getattr(processor_cls, "process_outputs", None)
        if not callable(class_original):
            return False
        if getattr(class_original, "_speco_concrete_trace_capture", False):
            return True

        def process_outputs_with_speco_concrete_trace(
            self, engine_core_outputs, *args, **kwargs
        ):
            _record_frontend_vllm_speculative_traces(engine_core_outputs)
            return class_original(self, engine_core_outputs, *args, **kwargs)

        setattr(
            process_outputs_with_speco_concrete_trace,
            "_speco_concrete_trace_capture",
            True,
        )
        try:
            processor_cls.process_outputs = process_outputs_with_speco_concrete_trace
        except (AttributeError, TypeError):
            return False
    if bool(_bool_or_none(os.getenv(SPECO_VLLM_DRAFT_DIAG_ENV))):
        logger.warning("[speco vllm trace boundary] frontend_instance_hook=1")
    return True


def patch_vllm_spec_decode_acceptance_logging() -> bool:
    """Lightly restore vLLM speculative acceptance logging as INFO logs."""

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Unable to install vLLM spec decode acceptance logging patch: %s", exc
        )
        return False

    original = getattr(Scheduler, "make_spec_decoding_stats", None)
    if not callable(original):
        return False
    if getattr(original, "_speco_spec_decode_acceptance_logging", False):
        return True

    def patched_make_spec_decoding_stats(
        self,
        spec_decoding_stats,
        num_draft_tokens,
        num_accepted_tokens,
        num_invalid_spec_tokens,
        request_id,
    ):
        result = original(
            self,
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )
        try:
            _record_vllm_spec_decode_acceptance(
                self,
                request_id=request_id,
                num_draft_tokens=num_draft_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_invalid_spec_tokens=num_invalid_spec_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log vLLM spec decode acceptance stats: %s", exc)
        return result

    setattr(
        patched_make_spec_decoding_stats,
        "_speco_spec_decode_acceptance_logging",
        True,
    )
    setattr(
        patched_make_spec_decoding_stats,
        "_speco_original_make_spec_decoding_stats",
        original,
    )
    Scheduler.make_spec_decoding_stats = patched_make_spec_decoding_stats
    return True


def patch_vllm_ascend_verified_draft_capture() -> bool:
    """Return real verify draft IDs and their target-model log probabilities."""

    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM-Ascend draft trace patch: %s", exc)
        return False

    original_copy = getattr(NPUModelRunner, "_copy_draft_token_ids_to_cpu", None)
    original_sample = getattr(NPUModelRunner, "sample_tokens", None)
    original_internal_sample = getattr(NPUModelRunner, "_sample", None)
    if not all(
        callable(value)
        for value in (original_copy, original_sample, original_internal_sample)
    ):
        return False
    if getattr(original_sample, "_speco_verified_draft_capture", False):
        return True

    from verl_speco.integration.dspark_topk_runtime import (
        patch_dspark_topk_capture,
        topk_payload_by_request,
        verified_topk_payload,
    )

    patch_dspark_topk_capture(NPUModelRunner)

    def copy_draft_token_ids_for_speco_trace(
        self, scheduler_output, zeros_only=False
    ):
        async_scheduling = getattr(self, "use_async_scheduling", False)
        self.use_async_scheduling = False
        try:
            return original_copy(self, scheduler_output, zeros_only=zeros_only)
        finally:
            self.use_async_scheduling = async_scheduling

    def sample_with_speco_teacher_logprobs(self, logits, spec_decode_metadata):
        self._speco_verified_draft_topk_payload = verified_topk_payload(
            self, logits, spec_decode_metadata
        )
        self._speco_verified_draft_teacher_payload = None
        if logits is not None and spec_decode_metadata is not None:
            try:
                draft_ids = spec_decode_metadata.draft_token_ids.long().reshape(-1)
                teacher_logprobs, teacher_entropies = _compute_vllm_draft_teacher_statistics(
                    logits, spec_decode_metadata,
                    calculate_entropy=bool(
                        _bool_or_none(os.getenv(SPECO_VLLM_VERIFY_ENTROPY_ENV))
                    ),
                )
                self._speco_verified_draft_teacher_payload = (
                    self.input_batch.req_ids.copy(),
                    list(spec_decode_metadata.num_draft_tokens),
                    draft_ids.detach(),
                    teacher_logprobs.detach(),
                    teacher_entropies.detach() if teacher_entropies is not None else None,
                )
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                self._speco_verified_draft_teacher_payload = None
        return original_internal_sample(self, logits, spec_decode_metadata)

    def sample_tokens_with_speco_verified_drafts(self, *args, **kwargs):
        verified_drafts: dict[str, list[int]] = {}
        verified_teacher_logprobs: dict[str, list[float]] = {}
        verified_teacher_entropies: dict[str, list[float]] = {}
        if getattr(self, "_draft_token_ids", None) is not None:
            try:
                draft_token_ids, request_ids = self._get_draft_token_ids_cpu()
                verified_drafts = {
                    str(request_id): [int(token_id) for token_id in token_ids]
                    for request_id, token_ids in zip(request_ids, draft_token_ids)
                }
            except (AttributeError, RuntimeError, TypeError, ValueError):
                verified_drafts = {}
        result = original_sample(self, *args, **kwargs)
        payload = getattr(self, "_speco_verified_draft_teacher_payload", None)
        if isinstance(payload, tuple) and len(payload) == 5:
            try:
                request_ids, draft_counts, draft_ids_tensor, teacher_logprobs_tensor, teacher_entropies_tensor = (
                    payload
                )
                flat_draft_ids = draft_ids_tensor.detach().cpu().tolist()
                flat_teacher_logprobs = (
                    teacher_logprobs_tensor.detach().float().cpu().tolist()
                )
                flat_teacher_entropies = (
                    teacher_entropies_tensor.detach().float().cpu().tolist()
                    if teacher_entropies_tensor is not None else None
                )
                offset = 0
                for request_id, raw_count in zip(request_ids, draft_counts):
                    count = max(int(raw_count), 0)
                    end = offset + count
                    if end > len(flat_draft_ids) or end > len(flat_teacher_logprobs):
                        break
                    request_key = str(request_id)
                    verified_drafts[request_key] = [
                        int(token_id) for token_id in flat_draft_ids[offset:end]
                    ]
                    verified_teacher_logprobs[request_key] = [
                        float(value) for value in flat_teacher_logprobs[offset:end]
                    ]
                    if flat_teacher_entropies is not None:
                        verified_teacher_entropies[request_key] = [
                            float(value) for value in flat_teacher_entropies[offset:end]
                        ]
                    offset = end
            except (AttributeError, RuntimeError, TypeError, ValueError):
                verified_teacher_logprobs = {}
        model_runner_output = getattr(result, "_model_runner_output", result)
        if model_runner_output is not None:
            model_runner_output._speco_verified_draft_topk_by_request = topk_payload_by_request(
                getattr(self, "_speco_verified_draft_topk_payload", None)
            )
            model_runner_output._speco_verified_draft_token_ids_by_request = (
                verified_drafts
            )
            model_runner_output._speco_verified_draft_teacher_logprobs_by_request = (
                verified_teacher_logprobs
            )
            model_runner_output._speco_verified_draft_teacher_entropies_by_request = (
                verified_teacher_entropies
            )
        return result

    setattr(
        copy_draft_token_ids_for_speco_trace,
        "_speco_verified_draft_capture",
        True,
    )
    setattr(
        sample_with_speco_teacher_logprobs,
        "_speco_verified_draft_capture",
        True,
    )
    setattr(
        sample_tokens_with_speco_verified_drafts,
        "_speco_verified_draft_capture",
        True,
    )
    NPUModelRunner._copy_draft_token_ids_to_cpu = (
        copy_draft_token_ids_for_speco_trace
    )
    NPUModelRunner._sample = sample_with_speco_teacher_logprobs
    NPUModelRunner.sample_tokens = sample_tokens_with_speco_verified_drafts
    return True


def patch_vllm_dflash_config_aliases() -> bool:
    """Let vLLM 0.23 consume SPECO DFlash top-level target layer ids."""

    global _VLLM_DFLASH_CONFIG_ALIASES_PATCHED
    if _VLLM_DFLASH_CONFIG_ALIASES_PATCHED:
        return True
    try:
        from vllm.transformers_utils.configs.eagle import EAGLEConfig
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DFlash config alias patch: %s", exc)
        return False

    current = getattr(EAGLEConfig, "__init__", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_dflash_config_aliases", False):
        _VLLM_DFLASH_CONFIG_ALIASES_PATCHED = True
        return True

    def patched_eagle_config_init(self, *args, **kwargs):
        method = kwargs.get("method", None)
        if method is None and len(args) >= 3:
            method = args[2]
        current(self, *args, **kwargs)
        if str(method or "eagle").strip().lower() == "dflash":
            _normalize_dflash_target_layer_aliases(self)
            if _is_dspark_hf_config(self):
                _set_child(self, "architectures", ["DFlashDraftModel"])

    setattr(patched_eagle_config_init, "_speco_dflash_config_aliases", True)
    setattr(patched_eagle_config_init, "_speco_original_eagle_config_init", current)
    EAGLEConfig.__init__ = patched_eagle_config_init
    _VLLM_DFLASH_CONFIG_ALIASES_PATCHED = True
    return True


def patch_vllm_dspark_registry_aliases() -> bool:
    """Let vLLM resolve DSpark draft architectures through the DFlash model."""

    global _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED
    if _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED:
        return True
    try:
        from vllm.model_executor.models import registry
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM DSpark registry aliases: %s", exc)
        return False

    model_registry = getattr(registry, "ModelRegistry", None)
    register_model = getattr(model_registry, "register_model", None)
    if not callable(register_model):
        return False

    existing_models = getattr(model_registry, "models", {})
    for architecture in sorted(
        _DSPARK_VLLM_ARCHITECTURES
        | {"DFlashDSparkDraftModel", "DFlashQwen3DSparkModel"}
    ):
        if architecture not in existing_models:
            register_model(
                architecture,
                "vllm.model_executor.models.qwen3_dflash:DFlashQwen3ForCausalLM",
            )
    _VLLM_DSPARK_REGISTRY_ALIAS_PATCHED = True
    return True


def _speco_vllm_run_engine_core_with_acceptance_logging(*args, **kwargs):
    _maybe_apply_vllm_ascend_global_patch()
    patch_vllm_dflash_config_aliases()
    patch_vllm_dspark_registry_aliases()
    patch_vllm_dspark_runtime()
    acceptance_patched = patch_vllm_spec_decode_acceptance_logging()
    trace_patched = patch_vllm_speculative_trace_capture()
    logger.warning(
        "[speco vllm trace] EngineCore startup acceptance_patch=%s trace_patch=%s pid=%s",
        int(acceptance_patched),
        int(trace_patched),
        os.getpid(),
    )
    patch_vllm_worker_proc_entrypoint()

    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(EngineCoreProc, "_speco_original_run_engine_core", None)
    if original is None:
        original = EngineCoreProc.run_engine_core
    return original(*args, **kwargs)


def patch_vllm_engine_core_entrypoint() -> bool:
    """Install the scheduler logging patch inside vLLM EngineCore subprocesses."""

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM EngineCore entrypoint patch: %s", exc)
        return False

    current = getattr(EngineCoreProc, "run_engine_core", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_engine_core_acceptance_logging", False):
        return True

    EngineCoreProc._speco_original_run_engine_core = current
    setattr(
        _speco_vllm_run_engine_core_with_acceptance_logging,
        "_speco_engine_core_acceptance_logging",
        True,
    )
    EngineCoreProc.run_engine_core = staticmethod(
        _speco_vllm_run_engine_core_with_acceptance_logging
    )
    return True


def _speco_vllm_worker_main_with_runtime_observability(*args, **kwargs):
    _maybe_apply_vllm_ascend_global_patch()
    patch_vllm_dflash_config_aliases()
    patch_vllm_dspark_registry_aliases()
    patch_vllm_dspark_runtime()
    patch_vllm_spec_decode_acceptance_logging()
    verified_draft_patched = patch_vllm_ascend_verified_draft_capture()
    logger.warning(
        "[speco vllm trace] worker verified_draft_patch=%s pid=%s",
        int(verified_draft_patched),
        os.getpid(),
    )

    original = getattr(
        _speco_vllm_worker_main_with_runtime_observability,
        "_speco_original_worker_main",
        None,
    )
    if not callable(original):
        from vllm.v1.executor import multiproc_executor

        original = multiproc_executor.WorkerProc.worker_main
    return original(*args, **kwargs)


def patch_vllm_worker_proc_entrypoint() -> bool:
    """Install runtime patches inside vLLM worker subprocesses."""

    if not _maybe_apply_vllm_ascend_global_patch():
        return False
    try:
        from vllm.v1.executor import multiproc_executor
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install vLLM WorkerProc entrypoint patch: %s", exc)
        return False

    WorkerProc = getattr(multiproc_executor, "WorkerProc", None)
    if WorkerProc is None:
        return False
    current = getattr(WorkerProc, "worker_main", None)
    if not callable(current):
        return False
    if getattr(current, "_speco_worker_proc_runtime_observability", False):
        return True

    WorkerProc._speco_original_worker_main = current
    setattr(
        _speco_vllm_worker_main_with_runtime_observability,
        "_speco_worker_proc_runtime_observability",
        True,
    )
    setattr(
        _speco_vllm_worker_main_with_runtime_observability,
        "_speco_original_worker_main",
        current,
    )
    WorkerProc.worker_main = staticmethod(
        _speco_vllm_worker_main_with_runtime_observability
    )
    return True


def install_vllm_spec_decode_acceptance_logging() -> bool:
    """Install speculative acceptance logs in parent and EngineCore processes."""

    scheduler_patched = patch_vllm_spec_decode_acceptance_logging()
    engine_core_patched = patch_vllm_engine_core_entrypoint()
    return scheduler_patched or engine_core_patched


def install_vllm_runtime_observability() -> bool:
    """Install lightweight vLLM logging hooks for all rollout modes."""

    _maybe_apply_vllm_ascend_global_patch()
    dflash_config_patched = patch_vllm_dflash_config_aliases()
    dspark_registry_patched = patch_vllm_dspark_registry_aliases()
    dspark_runtime_patched = patch_vllm_dspark_runtime()
    acceptance_patched = install_vllm_spec_decode_acceptance_logging()
    trace_capture_patched = patch_vllm_speculative_trace_capture()
    trace_client_patched = patch_vllm_core_client_speculative_trace()
    trace_async_llm_patched = patch_vllm_async_llm_speculative_trace()
    trace_output_patched = patch_vllm_output_processor_speculative_trace()
    worker_proc_patched = patch_vllm_worker_proc_entrypoint()
    return (
        dflash_config_patched
        or dspark_registry_patched
        or dspark_runtime_patched
        or acceptance_patched
        or trace_capture_patched
        or trace_client_patched
        or trace_async_llm_patched
        or trace_output_patched
        or worker_proc_patched
    )


def _new_vllm_spec_decode_stats() -> dict[str, float]:
    return {
        "drafts": 0,
        "accepted_tokens": 0,
    }


def _record_vllm_spec_decode_scheduler_stats(
    target: dict[str, float], scheduler_stats: Any
) -> None:
    spec_stats = getattr(scheduler_stats, "spec_decoding_stats", None)
    if spec_stats is None:
        return
    drafts = _int_or_zero(getattr(spec_stats, "num_drafts", 0))
    accepted_tokens = _int_or_zero(getattr(spec_stats, "num_accepted_tokens", 0))
    if drafts <= 0 and accepted_tokens <= 0:
        return
    target["drafts"] += drafts
    target["accepted_tokens"] += accepted_tokens


def _vllm_spec_decode_stats_to_metrics(stats: dict[str, float]) -> dict[str, float]:
    drafts = float(stats.get("drafts", 0.0) or 0.0)
    accepted_tokens = float(stats.get("accepted_tokens", 0.0) or 0.0)
    return {
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": drafts,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": accepted_tokens,
    }


def _build_speco_vllm_stat_logger(server: Any):
    from vllm.v1.metrics.loggers import StatLoggerBase

    class SpecoVLLMSpecDecodeStatLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0):
            del vllm_config
            self.engine_index = engine_index

        def record(
            self,
            scheduler_stats,
            iteration_stats,
            mm_cache_stats=None,
            engine_idx: int = 0,
        ):
            del iteration_stats, mm_cache_stats
            stats = getattr(server, "_speco_vllm_spec_decode_pending_stats", None)
            if not isinstance(stats, dict):
                stats = _new_vllm_spec_decode_stats()
                server._speco_vllm_spec_decode_pending_stats = stats
            _record_vllm_spec_decode_scheduler_stats(stats, scheduler_stats)

        def log_engine_initialized(self):
            return None

    SpecoVLLMSpecDecodeStatLogger.__module__ = __name__
    return SpecoVLLMSpecDecodeStatLogger


def _ensure_vllm_drafter_speculative_config_from_env(rollout_cfg: Any) -> None:
    drafter_cfg = _load_env_drafter_config()
    if not bool(drafter_cfg.get("enable")):
        return

    speculative_config = build_vllm_speculative_config_from_drafter(
        drafter_cfg, rollout_cfg=rollout_cfg
    )
    engine_kwargs_root = _ensure_child_mapping(rollout_cfg, "engine_kwargs")
    engine_kwargs = _ensure_child_mapping(engine_kwargs_root, "vllm")
    existing_spec = _get_nested(engine_kwargs, ("speculative_config",), None)
    merged_speculative_config = _merge_speculative_config(
        existing_spec, speculative_config
    )
    # Authoritative check: engine_kwargs.vllm.speculative_config (existing_spec) takes
    # priority in the merge, so a lossy acceptance mode injected there must be caught here.
    assert_lossless_vllm_speculative_config(
        merged_speculative_config,
        allow_lossy=bool(
            _bool_or_none(
                _get_nested(
                    drafter_cfg, ("vllm", "allow_lossy_speculative_sampling"), False
                )
            )
        ),
    )
    _set_child(engine_kwargs, "speculative_config", merged_speculative_config)
    _configure_vllm_ascend_dspark_swa_window(
        engine_kwargs, drafter_cfg, merged_speculative_config
    )
    if bool(merged_speculative_config.get("enforce_eager")):
        _set_child(engine_kwargs, "enforce_eager", True)


def _preload_vllm_ascend_ops() -> bool:
    """Initialize Ascend ops before platform quantization imports submodules."""
    import importlib
    import importlib.util

    try:
        spec = importlib.util.find_spec("vllm_ascend.ops")
    except (ImportError, AttributeError, ValueError):
        return False
    if spec is None:
        return False
    importlib.import_module("vllm_ascend.ops")
    return True


class _SpecoVLLMHttpServerMixin:
    def _preprocess_engine_kwargs(self, engine_kwargs: dict[str, Any]) -> None:
        parent_preprocess = getattr(super(), "_preprocess_engine_kwargs", None)
        if callable(parent_preprocess):
            parent_preprocess(engine_kwargs)
        _configure_dspark_swa_window_from_runtime_model(engine_kwargs)

    def _speco_pop_vllm_spec_decode_stats(self) -> dict[str, float]:
        stats = getattr(self, "_speco_vllm_spec_decode_pending_stats", None)
        if not isinstance(stats, dict):
            return _new_vllm_spec_decode_stats()
        snapshot = dict(stats)
        self._speco_vllm_spec_decode_pending_stats = _new_vllm_spec_decode_stats()
        return snapshot

    def _speco_add_vllm_spec_decode_extra_fields(
        self, extra_fields: dict[str, Any]
    ) -> None:
        stats = self._speco_pop_vllm_spec_decode_stats()
        extra_fields.update(_vllm_spec_decode_stats_to_metrics(stats))

    async def launch_server(self, *args, **kwargs):
        self._speco_vllm_spec_decode_pending_stats = _new_vllm_spec_decode_stats()
        _preload_vllm_ascend_ops()
        install_vllm_runtime_observability()
        _ensure_vllm_drafter_speculative_config_from_env(self.config)
        return await super().launch_server(*args, **kwargs)

    async def run_server(self, args):
        try:
            import inspect

            from vllm.v1.engine.async_llm import AsyncLLM
        except Exception:  # noqa: BLE001
            return await super().run_server(args)

        original_from_vllm_config_attr = inspect.getattr_static(
            AsyncLLM, "from_vllm_config"
        )
        original_from_vllm_config = AsyncLLM.from_vllm_config
        try:
            original_signature = inspect.signature(
                original_from_vllm_config_attr.__func__
            )
        except (AttributeError, TypeError, ValueError):
            original_signature = None

        def from_vllm_config_with_speco_stats(cls, *call_args, **call_kwargs):
            del cls
            install_vllm_runtime_observability()
            stat_loggers = list(call_kwargs.get("stat_loggers") or [])
            stat_loggers.append(_build_speco_vllm_stat_logger(self))
            call_kwargs["stat_loggers"] = stat_loggers
            # vLLM probes and releases a TCP port before TP workers bind it.
            # Serializing this one-time startup closes the race between rollout
            # replicas while leaving generation fully parallel.
            with _serialize_vllm_engine_startup():
                engine = original_from_vllm_config(*call_args, **call_kwargs)
            _install_vllm_trace_on_async_llm(engine)
            return engine

        if original_signature is not None:
            from_vllm_config_with_speco_stats.__signature__ = original_signature
        AsyncLLM.from_vllm_config = classmethod(from_vllm_config_with_speco_stats)
        try:
            return await super().run_server(args)
        finally:
            AsyncLLM.from_vllm_config = original_from_vllm_config_attr

    async def generate(self, *args, **kwargs):
        # AgentLoop's SPECO bridge adds side-channel routing metadata. These
        # are not vLLM SamplingParams fields. Keep ordinary controls (including
        # max_tokens) intact, and never mutate the caller's reusable dictionary.
        if "sampling_params" in kwargs:
            kwargs["sampling_params"] = {
                key: value for key, value in kwargs["sampling_params"].items()
                if not key.startswith("_verl_")
            }
        elif len(args) >= 2:
            args = (args[0], {
                key: value for key, value in args[1].items()
                if not key.startswith("_verl_")
            }, *args[2:])
        if os.getenv("SPECO_AUDIT_SAMPLING_PARAMS", "").lower() in ("1", "true", "yes", "on"):
            if not getattr(self, "_speco_sampling_params_audited", False):
                params = kwargs.get("sampling_params", args[1] if len(args) >= 2 else {})
                print("[speco sampling audit]", {
                    key: params.get(key)
                    for key in ("temperature", "top_p", "top_k", "seed", "max_tokens")
                }, flush=True)
                self._speco_sampling_params_audited = True
        request_id = kwargs.get("request_id")
        if request_id is None and len(args) >= 3:
            request_id = args[2]
        _pop_frontend_vllm_speculative_trace(request_id)
        _pop_frontend_vllm_speculative_trace_diagnostics(request_id)
        _pop_vllm_trace_spool(request_id)
        try:
            output = await super().generate(*args, **kwargs)
        except BaseException:
            _pop_frontend_vllm_speculative_trace(request_id)
            _pop_frontend_vllm_speculative_trace_diagnostics(request_id)
            _pop_vllm_trace_spool(request_id)
            raise
        trace = _pop_frontend_vllm_speculative_trace(request_id)
        trace_diagnostics = _pop_frontend_vllm_speculative_trace_diagnostics(
            request_id
        )
        spooled_trace, spooled_diagnostics = _pop_vllm_trace_spool(request_id)
        if not trace:
            trace = spooled_trace
        if not trace_diagnostics:
            trace_diagnostics = spooled_diagnostics
        elif spooled_diagnostics:
            for key, value in spooled_diagnostics.items():
                trace_diagnostics.setdefault(key, value)
        trace_diagnostics["server_spooled_events"] = len(spooled_trace)
        extra_fields = getattr(output, "extra_fields", None)
        if isinstance(extra_fields, dict):
            self._speco_add_vllm_spec_decode_extra_fields(extra_fields)
            drafts = int(
                extra_fields.get(
                    f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts", 0
                )
                or 0
            )
            trace_diagnostics["server_request_id_present"] = int(
                request_id is not None
            )
            trace_diagnostics["server_returned_events"] = len(trace)
            trace_diagnostics["server_trace_match"] = int(bool(trace))
            trace_diagnostics["server_trace_miss_with_drafts"] = int(
                drafts > 0 and not trace
            )
            extra_fields[SPECO_VLLM_SPECULATIVE_TRACE_KEY] = trace
            extra_fields[SPECO_VLLM_SPECULATIVE_TRACE_DIAGNOSTICS_KEY] = (
                trace_diagnostics
            )
        return output


def _build_speco_vllm_http_server_class(upstream_module: Any):
    upstream_cls = upstream_module.vLLMHttpServer
    if issubclass(upstream_cls, _SpecoVLLMHttpServerMixin):
        return upstream_cls
    return type(
        "SpecoVLLMHttpServer",
        (_SpecoVLLMHttpServerMixin, upstream_cls),
        {"__module__": __name__},
    )


def install_upstream_vllm_runtime_bridge() -> bool:
    """Patch upstream verl vLLM rollout classes in the current process."""

    global _VLLM_REPLICA_PATCHED
    install_vllm_runtime_observability()
    if _VLLM_REPLICA_PATCHED:
        return True

    try:
        import ray

        from verl.workers.rollout import replica as replica_module
        from verl.workers.rollout.vllm_rollout import vllm_async_server
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to install SPECO vLLM runtime bridge: %s", exc)
        return False

    upstream_replica = getattr(vllm_async_server, "vLLMReplica", None)
    if upstream_replica is None:
        return False

    speco_http_server_cls = _build_speco_vllm_http_server_class(vllm_async_server)

    upstream_replica_base = cast(type[Any], upstream_replica)

    def _speco_vllm_replica_init(self, *args, **kwargs):
        upstream_replica_base.__init__(self, *args, **kwargs)
        self.server_class = ray.remote(speco_http_server_cls)

    SpecoVLLMReplica = types.new_class(
        "SpecoVLLMReplica",
        (upstream_replica_base,),
        exec_body=lambda namespace: namespace.update(
            {"__init__": _speco_vllm_replica_init}
        ),
    )

    SpecoVLLMReplica.__module__ = __name__
    vllm_async_server.vLLMReplica = SpecoVLLMReplica
    registry = getattr(replica_module, "RolloutReplicaRegistry", None)
    if registry is not None and hasattr(registry, "_registry"):
        registry._registry["vllm"] = lambda: SpecoVLLMReplica
    patch_vllm_server_adapter_update()
    _VLLM_REPLICA_PATCHED = True
    return True


def configure_vllm_runtime_from_config(config: Any) -> dict[str, Any]:
    """Inject vLLM launch kwargs needed by SPECO online drafter rollout."""

    if _rollout_name(config) != "vllm":
        return {}
    drafter_cfg = _drafter_config_from_config(config)
    enabled = bool(drafter_cfg.get("enable"))
    if not enabled:
        os.environ.pop(SPECO_DRAFTER_CONFIG_ENV, None)
        return {}

    os.environ[SPECO_DRAFTER_CONFIG_ENV] = json.dumps(
        _vllm_drafter_env_payload(drafter_cfg), sort_keys=True
    )
    rollout_cfg = _rollout_config_from_config(config)
    speculative_config = build_vllm_speculative_config_from_drafter(
        drafter_cfg, rollout_cfg=rollout_cfg
    )
    install_upstream_vllm_runtime_bridge()

    engine_kwargs = _ensure_nested_mapping(
        config, ("actor_rollout_ref", "rollout", "engine_kwargs", "vllm")
    )
    existing_spec = _get_nested(engine_kwargs, ("speculative_config",), None)
    merged_speculative_config = _merge_speculative_config(
        existing_spec, speculative_config
    )
    assert_lossless_vllm_speculative_config(
        merged_speculative_config,
        allow_lossy=bool(
            _bool_or_none(
                _get_nested(
                    drafter_cfg, ("vllm", "allow_lossy_speculative_sampling"), False
                )
            )
        ),
    )
    _set_child(engine_kwargs, "speculative_config", merged_speculative_config)
    _configure_vllm_ascend_dspark_swa_window(
        engine_kwargs, drafter_cfg, merged_speculative_config
    )
    if bool(drafter_cfg.get("enable")):
        _set_child(
            engine_kwargs, "worker_extension_cls", SPECO_VLLM_WORKER_EXTENSION_CLS
        )
    if bool(merged_speculative_config.get("enforce_eager")):
        _set_child(engine_kwargs, "enforce_eager", True)
    return speculative_config


def _draft_zmq_handle_from_base(zmq_handle: str) -> str:
    if zmq_handle.endswith(".sock"):
        return f"{zmq_handle[:-5]}-draft.sock"
    return f"{zmq_handle}-draft"


def _named_weight_iter(weights: Any) -> Iterable[tuple[str, Any]]:
    if hasattr(weights, "items"):
        return weights.items()
    return weights


def _resolve_vllm_draft_update_use_shm(
    adapter: Any, training_cfg: dict[str, Any]
) -> bool:
    env_forced = _bool_or_none(os.getenv(SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV))
    if env_forced is not None:
        return env_forced
    forced = _bool_or_none(training_cfg.get("draft_update_use_shm", None))
    if forced is not None:
        return forced
    config_forced = _bool_or_none(
        _first_present(
            _get_nested(
                getattr(adapter, "config", None),
                ("drafter", "training", "draft_update_use_shm"),
                None,
            ),
            _get_nested(
                getattr(adapter, "config", None),
                ("rollout", "drafter", "training", "draft_update_use_shm"),
                None,
            ),
            _get_nested(
                getattr(adapter, "config", None),
                (
                    "actor_rollout_ref",
                    "rollout",
                    "drafter",
                    "training",
                    "draft_update_use_shm",
                ),
                None,
            ),
        )
    )
    if config_forced is not None:
        return config_forced
    return bool(getattr(adapter, "use_shm", False)) or _is_vllm_ascend_runtime_hint()


def _draft_param_name_candidates(name: str) -> list[str]:
    prefixes = ("module.", "_orig_mod.", "draft_model.", "model.draft_model.")
    bases = []
    pending = [name]
    while pending:
        candidate = pending.pop(0)
        if candidate in bases:
            continue
        bases.append(candidate)
        for prefix in prefixes:
            if candidate.startswith(prefix):
                pending.append(candidate[len(prefix) :])

    candidates = []
    for candidate in bases:
        candidates.append(candidate)
        if "midlayer." in candidate:
            candidates.append(candidate.replace("midlayer.", "model.layers.0."))
    for candidate in list(candidates):
        if not candidate.startswith("model."):
            candidates.append(f"model.{candidate}")
    return list(dict.fromkeys(candidates))


def _draft_fused_param_candidates(name: str) -> list[tuple[str, Any]]:
    mappings = (
        (".qkv_proj.", ".q_proj.", "q"),
        (".qkv_proj.", ".k_proj.", "k"),
        (".qkv_proj.", ".v_proj.", "v"),
        (".gate_up_proj.", ".gate_proj.", 0),
        (".gate_up_proj.", ".up_proj.", 1),
    )
    candidates = []
    for candidate in _draft_param_name_candidates(name):
        for fused_name, shard_name, shard_id in mappings:
            if shard_name in candidate:
                candidates.append((candidate.replace(shard_name, fused_name), shard_id))
    return list(dict.fromkeys(candidates))


def _load_draft_param(param: Any, tensor: Any, shard_id: Any = None) -> None:
    param_data = getattr(param, "data", None)
    device = getattr(param, "device", getattr(param_data, "device", None))
    dtype = getattr(param, "dtype", getattr(param_data, "dtype", None))
    tensor = tensor.to(device=device, dtype=dtype)
    weight_loader = getattr(param, "weight_loader", None)
    if callable(weight_loader):
        if shard_id is None:
            weight_loader(param, tensor)
        else:
            weight_loader(param, tensor, shard_id)
        return
    param.data.copy_(tensor, non_blocking=True)


def _ensure_vllm_server_handle(adapter: Any) -> None:
    if (
        getattr(adapter, "rollout_rank", None) != 0
        or getattr(adapter, "server_handle", None) is not None
    ):
        return
    import ray

    prefix = adapter._get_server_name_prefix()
    adapter.server_handle = ray.get_actor(
        f"{prefix}server_{adapter.replica_rank}_{adapter.node_rank}"
    )


async def _maybe_call_vllm_server_method(
    adapter: Any, method_name: str, *args, **kwargs
) -> Any:
    if getattr(adapter, "rollout_rank", None) != 0:
        return None
    _ensure_vllm_server_handle(adapter)
    method = getattr(adapter.server_handle, method_name, None)
    if method is None or not hasattr(method, "remote"):
        return None
    return await method.remote(*args, **kwargs)


async def speco_vllm_update_draft_weights(
    self, weights: Any, *args, global_steps: int | None = None, **kwargs
):
    """Update only vLLM draft/speculative model weights from a ServerAdapter."""

    del args
    if not weights:
        return

    drafter_cfg = _load_env_drafter_config()
    training_cfg = drafter_cfg.get("training") or {}
    bucket_mb = _positive_int_or_none(
        training_cfg.get("draft_update_weights_bucket_megabytes")
    )
    if bucket_mb is None:
        bucket_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
    pause_generation = bool(training_cfg.get("draft_update_pause_generation", True))
    flush_before = bool(training_cfg.get("draft_update_flush_before", True))
    flush_after = bool(training_cfg.get("draft_update_flush_after", True))
    generation_paused = False
    use_shm = _resolve_vllm_draft_update_use_shm(self, training_cfg)
    if (
        getattr(self, "replica_rank", -1) == 0
        and getattr(self, "rollout_rank", -1) == 0
    ):
        logger.warning(
            "[speco vllm draft update] starting global_steps=%s transfer=%s env_%s=%r cfg_draft_update_use_shm=%r adapter_use_shm=%r",
            global_steps,
            "shm" if use_shm else "ipc",
            SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV,
            os.getenv(SPECO_VLLM_DRAFT_UPDATE_USE_SHM_ENV),
            training_cfg.get("draft_update_use_shm", None),
            getattr(self, "use_shm", None),
        )

    patch_verl_bucketed_weight_transfer_shm_reuse()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        BucketedWeightSender,
    )

    start_time = time.time()
    try:
        if self.rollout_rank == 0 and pause_generation:
            await _maybe_call_vllm_server_method(
                self, "abort_all_requests", reset_prefix_cache=flush_before
            )
            generation_paused = True
        elif self.rollout_rank == 0 and flush_before:
            await _maybe_call_vllm_server_method(self, "clear_kv_cache")

        future = await self._execute_method(
            "update_draft_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": use_shm},
        )

        sender = BucketedWeightSender(
            zmq_handle=_draft_zmq_handle_from_base(self.zmq_handle),
            bucket_size_mb=int(bucket_mb),
            use_shm=use_shm,
        )
        await sender.async_send_weights(_named_weight_iter(weights))

        if future is not None:
            await future

        if self.rollout_rank == 0:
            if flush_after:
                await _maybe_call_vllm_server_method(self, "clear_kv_cache")
            if global_steps is not None:
                await _maybe_call_vllm_server_method(
                    self, "set_global_steps", global_steps
                )

        if (
            getattr(self, "replica_rank", -1) == 0
            and getattr(self, "rollout_rank", -1) == 0
        ):
            logger.warning(
                "[speco vllm draft update] done global_steps=%s transfer=%s bucket_mb=%s elapsed_sec=%.3f",
                global_steps,
                "shm" if use_shm else "ipc",
                bucket_mb,
                time.time() - start_time,
            )
    finally:
        if generation_paused:
            await _maybe_call_vllm_server_method(self, "resume_generation")


def attach_update_draft_weights_to_rollout(rollout: Any) -> Any:
    """Attach ``update_draft_weights`` to an upstream vLLM ServerAdapter."""

    if rollout is not None and not callable(
        getattr(rollout, "update_draft_weights", None)
    ):
        rollout.update_draft_weights = speco_vllm_update_draft_weights.__get__(
            rollout, type(rollout)
        )
    return rollout


def patch_vllm_server_adapter_update() -> None:
    patch_verl_bucketed_weight_transfer_rebuild_ipc()
    patch_verl_bucketed_weight_transfer_shm_reuse()
    try:
        from verl.workers.rollout.vllm_rollout import vllm_rollout
    except Exception:  # noqa: BLE001
        return

    server_adapter = getattr(vllm_rollout, "ServerAdapter", None)
    if server_adapter is not None and not callable(
        getattr(server_adapter, "update_draft_weights", None)
    ):
        server_adapter.update_draft_weights = speco_vllm_update_draft_weights


def install_vllm_runtime_for_worker(worker: Any) -> None:
    """Install SPECO vLLM runtime hooks inside an actor-rollout worker process."""

    drafter_env = getattr(type(worker), "_speco_sglang_drafter_config_env", None)
    if drafter_env:
        os.environ[SPECO_DRAFTER_CONFIG_ENV] = drafter_env
    install_vllm_runtime_observability()
    patch_verl_bucketed_weight_transfer_rebuild_ipc()
    patch_verl_bucketed_weight_transfer_shm_reuse()
    patch_verl_bucketed_weight_transfer_npu_staging()
    patch_vllm_server_adapter_update()


try:
    from verl.workers.rollout.vllm_rollout.utils import (
        vLLMColocateWorkerExtension as _VLLMWorkerExtensionBase,
    )
except Exception:  # noqa: BLE001
    _VLLMWorkerExtensionBase = object


class SpecoVLLMWeightSyncCompatExtension(_VLLMWorkerExtensionBase):
    """Install the serialized NPU IPC-handle compatibility before target weight sync."""

    def update_weights_from_ipc(
        self,
        peft_config: dict | None = None,
        base_sync_done=False,
        use_shm: bool = False,
    ):
        patched = patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        patch_verl_bucketed_weight_transfer_npu_staging()
        if patched and int(getattr(self, "local_rank", 0) or 0) == 0:
            logger.warning(
                "[speco vllm weight sync] installed IPC rebuild compatibility"
            )
        if not _speco_is_npu_vllm_worker(self):
            return super().update_weights_from_ipc(
                peft_config=peft_config,
                base_sync_done=base_sync_done,
                use_shm=use_shm,
            )

        try:
            with _speco_npu_target_staging(
                self, peft_config=peft_config, use_shm=use_shm
            ):
                return super().update_weights_from_ipc(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                    use_shm=use_shm,
                )
        finally:
            trim_process_host_memory()


class SpecoVLLMColocateWorkerExtension(_VLLMWorkerExtensionBase):
    """vLLM worker extension that can update only the speculative draft model."""

    def __new__(cls, **kwargs):
        from verl_speco.integration.npu_sleep_compat import install_pageable_npu_sleep

        install_pageable_npu_sleep()
        try:
            instance = super().__new__(cls, **kwargs)
        except TypeError:
            instance = super().__new__(cls)
        # vLLM's extension mechanism forbids overriding methods that already
        # exist on Worker (e.g. wake_up). Use __new__ (dunder, skipped by the
        # conflict check) to install an instance-level wrapper instead.
        # Python resolves instance attributes before class methods.
        _orig_wake_up = getattr(type(instance), "wake_up", None)
        if not callable(_orig_wake_up):
            return instance

        def _speco_wake_up_hook(*args, **kwargs):
            result = _orig_wake_up(instance, *args, **kwargs)
            reloaded = instance._speco_reload_draft_from_checkpoint()
            if reloaded > 0:
                logger.warning(
                    "[speco draft wake_up] drafter weights restored after wake_up (%d tensors)",
                    reloaded,
                )
            return result

        instance.wake_up = _speco_wake_up_hook
        return instance

    def _get_speco_draft_zmq_handle(self) -> str:
        get_base = getattr(self, "_get_zmq_handle", None)
        if callable(get_base):
            return _draft_zmq_handle_from_base(get_base())
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{self.local_rank}-draft.sock"

    def _speco_resolve_draft_proposer(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return None
        for attr in ("drafter", "speculator"):
            proposer = getattr(runner, attr, None)
            if proposer is not None:
                return proposer
        return None

    def _speco_resolve_draft_model(self):
        proposer = self._speco_resolve_draft_proposer()
        if proposer is None:
            return None, None
        get_model = getattr(proposer, "get_model", None)
        if callable(get_model):
            try:
                return get_model(), proposer
            except AttributeError:
                return None, proposer
        model = getattr(proposer, "model", None)
        if model is not None:
            return model, proposer
        return None, proposer

    def _speco_draft_method(self) -> str:
        proposer = self._speco_resolve_draft_proposer()
        spec_cfg = (
            getattr(proposer, "speculative_config", None)
            if proposer is not None
            else None
        )
        method = str(getattr(spec_cfg, "method", "") or "").strip().lower()
        if method:
            return method
        draft_model, _ = self._speco_resolve_draft_model()
        model_type = (
            type(draft_model).__name__.lower() if draft_model is not None else ""
        )
        if "dflash" in model_type:
            return "dflash"
        if "eagle3" in model_type:
            return "eagle3"
        return method

    def _speco_is_dflash_draft(self) -> bool:
        return self._speco_draft_method() in ("dflash", "dspark")

    def _speco_update_draft_weights(self, weights: list[tuple[str, Any]]) -> int:
        draft_model, proposer = self._speco_resolve_draft_model()
        if draft_model is None:
            return 0
        del proposer
        named_parameters = getattr(draft_model, "named_parameters", None)
        if not callable(named_parameters):
            raise RuntimeError(
                "Resolved vLLM draft model does not expose named_parameters() for graph-safe update"
            )

        named_params = dict(named_parameters())
        updated = 0
        missing = []
        incompatible = []
        for name, tensor in weights:
            param = None
            matched_name = None
            for candidate in _draft_param_name_candidates(str(name)):
                param = named_params.get(candidate)
                if param is not None:
                    matched_name = candidate
                    break
            shard_id = None
            if param is None:
                for candidate, candidate_shard_id in _draft_fused_param_candidates(
                    str(name)
                ):
                    param = named_params.get(candidate)
                    if param is not None:
                        matched_name = candidate
                        shard_id = candidate_shard_id
                        break
                if param is None:
                    missing.append(name)
                    continue
            if (
                shard_id is None
                and not callable(getattr(param, "weight_loader", None))
                and tuple(param.shape) != tuple(tensor.shape)
            ):
                incompatible.append(
                    f"{name}->{matched_name}: expected {tuple(param.shape)}, got {tuple(tensor.shape)}"
                )
                continue
            try:
                _load_draft_param(param, tensor, shard_id=shard_id)
            except Exception as exc:  # noqa: BLE001
                incompatible.append(
                    f"{name}->{matched_name}: loader failed for shape {tuple(tensor.shape)}: {exc}"
                )
                continue
            updated += 1

        if missing or incompatible:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}")
            if incompatible:
                details.append(f"incompatible={incompatible[:8]}")
            raise RuntimeError(
                "SPECO vLLM graph-safe draft update could not load all weights: "
                + "; ".join(details)
            )
        return updated

    def update_draft_weights_from_ipc(self, use_shm: bool = False):
        """Receive and load draft-model weights through the verl bucketed IPC path.

        Uses draft_model.load_weights() (the same path as checkpoint reload)
        instead of per-param copy_() to ensure weights are correctly placed in
        cumem-managed memory after sleep/wake_up cycles.
        """

        import torch
        from vllm.platforms import current_platform

        patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        is_npu = str(getattr(current_platform, "device_type", "")).lower() == "npu"
        use_shm = bool(use_shm)
        if is_npu and not use_shm:
            raise RuntimeError(
                "SPECO vLLM draft weight update on NPU requires shared-memory transfer"
            )

        if is_npu and getattr(self, "device", None) is None:
            self.device = torch.device(f"npu:{self.local_rank}")
        assert self.device is not None

        all_weights: list[tuple[str, torch.Tensor]] = []

        def finish_update(result, translated_weights=None):
            if not is_npu:
                return result
            if translated_weights is not None:
                translated_weights.clear()
            all_weights.clear()
            trim_process_host_memory()
            return result

        def on_bucket_received(bucket_weights):
            # Clone immediately: bucket views may be freed (IPC) or overwritten
            # by the next update (persistent SHM). Give each tensor independent
            # device storage before receive_weights() returns.
            all_weights.extend(
                [(name, t.detach().clone()) for name, t in bucket_weights]
            )

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_speco_draft_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(on_bucket_received=on_bucket_received)

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None or not all_weights:
            return finish_update(
                {"loaded_params": 0, "has_draft_model": draft_model is not None}
            )

        draft_method = self._speco_draft_method()
        is_dflash = draft_method == "dflash"
        is_dspark = draft_method == "dspark"
        is_eagle3 = draft_method == "eagle3"

        # Translate training-side names.  DFlash/DSpark publish into the inner
        # DFlashQwen3Model/Qwen3DSparkModel, while EAGLE3 publishes into the
        # outer Eagle3LlamaForCausalLM because lm_head.weight lives outside
        # ``draft_model.model`` in vLLM.
        _strip_prefixes: tuple[str, ...] = (
            "module.",
            "_orig_mod.",
            "draft_model.",
            "model.draft_model.",
        )
        if is_dflash or is_dspark:
            _strip_prefixes = (*_strip_prefixes, "model.")
        translated_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in all_weights:
            n = name
            changed = True
            while changed:
                changed = False
                for pfx in _strip_prefixes:
                    if n.startswith(pfx):
                        n = n[len(pfx) :]
                        changed = True
            if "midlayer." in n:
                n = n.replace("midlayer.", "layers.0.")
            # The current DSpark trainer only supports CE/L1 training; when
            # confidence loss is disabled its frozen confidence head is not
            # consumed by the vLLM speculative proposer.  vLLM's
            # Qwen3DSparkModel intentionally omits that training-only module,
            # so do not send its two parameters through strict load_weights.
            if is_dspark and n.startswith("confidence_head."):
                continue
            if (
                is_eagle3
                and n != "lm_head.weight"
                and "." in n
                and not n.startswith("model.")
            ):
                n = f"model.{n}"
            translated_weights.append((n, tensor))

        loaded_params = len(translated_weights)
        if is_eagle3:
            loaded_params = self._speco_update_draft_weights(translated_weights)
        else:
            inner_model = getattr(draft_model, "model", None)
            if inner_model is None:
                return finish_update(
                    {"loaded_params": 0, "has_draft_model": True},
                    translated_weights,
                )
            logger.warning(
                "[speco draft ipc] loading %d translated weights into %s (method=%s), first 5 keys: %s",
                len(translated_weights),
                type(inner_model).__name__,
                draft_method,
                [n for n, _ in translated_weights[:5]],
            )
            inner_model.load_weights(iter(translated_weights))

            # Rebuild fused KV buffers (torch.cat snapshot, not a view)
            try:
                inner_model._build_fused_kv_buffers()
            except Exception as exc:
                logger.warning(
                    "[speco draft update] _build_fused_kv_buffers failed: %s", exc
                )

        self._speco_diag_draft_state("after_draft_ipc_update")
        if loaded_params > 0:
            # NPU hybrid rollout uses sleep(level=1), which preserves the live
            # draft weights. Do not let a later wake-up reload replace them
            # with the checkpoint used to bootstrap this process.
            self._speco_live_draft_weights_loaded = True
        # One-time diagnostic: check whether probabilistic sampling is active
        proposer = self._speco_resolve_draft_proposer()
        if proposer is not None and not getattr(
            self, "_speco_logged_sampling_mode", False
        ):
            self._speco_logged_sampling_mode = True
            missing_draft_logits = not hasattr(proposer, "draft_logits")
            draft_logits = getattr(proposer, "draft_logits", None)
            spec_cfg = getattr(
                getattr(getattr(self, "model_runner", None), "vllm_config", None),
                "speculative_config",
                None,
            )
            dsm = (
                getattr(spec_cfg, "draft_sample_method", "UNKNOWN")
                if spec_cfg
                else "NO_SPEC_CFG"
            )
            logger.warning(
                "[speco-diag:sampling_mode] draft_sample_method=%s draft_logits=%s proposer=%s",
                dsm,
                _describe_vllm_draft_logits(draft_logits, missing=missing_draft_logits),
                type(proposer).__name__,
            )
        return finish_update(
            {"loaded_params": loaded_params, "has_draft_model": True},
            translated_weights,
        )

    # Checkpoint fallback before the first online draft-weight publication.

    def _speco_get_draft_checkpoint_path(self) -> str | None:
        """Resolve the DFlash drafter checkpoint path from speculative_config."""
        if not self._speco_is_dflash_draft():
            return None
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return None
        vllm_cfg = getattr(runner, "vllm_config", None)
        spec_cfg = getattr(vllm_cfg, "speculative_config", None) if vllm_cfg else None
        draft_model_cfg = (
            getattr(spec_cfg, "draft_model_config", None) if spec_cfg else None
        )
        if draft_model_cfg is None:
            return None
        return getattr(draft_model_cfg, "model", None)

    def _speco_reload_draft_from_checkpoint(self) -> int:
        """Reload DFlash/DSpark drafter weights from a pretrained checkpoint.

        GPU hybrid rollout may need this after sleep(level=2). NPU uses
        sleep(level=1), where the bootstrapped draft model remains resident
        across wake-up and target-model synchronization. Loading a full draft
        checkpoint in every NPU TP rank is therefore both unnecessary and a
        substantial host-memory spike.
        """
        import glob as _glob

        if not self._speco_is_dflash_draft():
            return 0

        # Ascend hybrid rollout uses sleep(level=1), which preserves the
        # initial checkpoint load and subsequently published IPC weights. The
        # normal vLLM model loader already handles the checkpoint at process
        # startup; do not re-read it after every target sync.
        if _speco_is_npu_vllm_worker(self):
            return 0

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None:
            logger.warning("[speco draft reload] no draft model found, skip")
            return 0

        ckpt_path = self._speco_get_draft_checkpoint_path()
        if not ckpt_path:
            logger.warning(
                "[speco draft reload] no draft checkpoint path configured, skip"
            )
            return 0

        def _load_state_file(path: str) -> dict[str, Any]:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file

                return load_file(path, device="cpu")

            import torch

            try:
                state_dict = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
            except (RuntimeError, TypeError):
                # mmap is unavailable for a few legacy PyTorch archives.
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict):
                raise TypeError(
                    f"Expected a state dict in {path}, got {type(state_dict).__name__}"
                )
            return state_dict

        index_paths = sorted(_glob.glob(os.path.join(ckpt_path, "*.index.json")))
        weight_files: list[str] = []
        if index_paths:
            if len(index_paths) != 1:
                logger.warning(
                    "[speco draft reload] multiple checkpoint indexes in %s, skip",
                    ckpt_path,
                )
                return 0
            try:
                with open(index_paths[0], "r", encoding="utf-8") as stream:
                    weight_map = json.load(stream).get("weight_map", {})
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                logger.warning(
                    "[speco draft reload] invalid checkpoint index %s: %s",
                    index_paths[0],
                    exc,
                )
                return 0
            if not isinstance(weight_map, dict) or not weight_map:
                logger.warning(
                    "[speco draft reload] empty checkpoint index %s, skip", index_paths[0]
                )
                return 0
            checkpoint_root = os.path.realpath(ckpt_path)
            for filename in sorted(set(weight_map.values())):
                if not isinstance(filename, str):
                    logger.warning(
                        "[speco draft reload] invalid shard name in %s, skip",
                        index_paths[0],
                    )
                    return 0
                shard_path = os.path.realpath(os.path.join(ckpt_path, filename))
                if os.path.commonpath((checkpoint_root, shard_path)) != checkpoint_root:
                    logger.warning(
                        "[speco draft reload] shard outside checkpoint path in %s, skip",
                        index_paths[0],
                    )
                    return 0
                weight_files.append(shard_path)
        else:
            safe_files = sorted(_glob.glob(os.path.join(ckpt_path, "*.safetensors")))
            bin_files = sorted(_glob.glob(os.path.join(ckpt_path, "*.bin")))
            # Prefer safetensors when both forms are intentionally retained.
            weight_files = safe_files or bin_files

        if not weight_files:
            logger.warning(
                "[speco draft reload] no supported weight files in %s, skip", ckpt_path
            )
            return 0

        weights_iter = []
        try:
            for weight_file in weight_files:
                state_dict = _load_state_file(weight_file)
                weights_iter.extend(state_dict.items())
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("[speco draft reload] failed to read %s: %s", ckpt_path, exc)
            return 0

        if not weights_iter:
            logger.warning("[speco draft reload] checkpoint empty, skip")
            return 0

        try:
            draft_model.load_weights(iter(weights_iter))
            loaded_count = len(weights_iter)
        except Exception as exc:
            logger.warning("[speco draft reload] load_weights failed: %s", exc)
            return 0
        finally:
            weights_iter.clear()

        return loaded_count

    def update_weights_from_ipc(
        self,
        peft_config: dict | None = None,
        base_sync_done=False,
        use_shm: bool = False,
    ):
        """Override target weight sync to also reload drafter from checkpoint."""
        patch_verl_bucketed_weight_transfer_rebuild_ipc()
        patch_verl_bucketed_weight_transfer_shm_reuse()
        patch_verl_bucketed_weight_transfer_npu_staging()
        is_npu = _speco_is_npu_vllm_worker(self)
        # Diagnostic: check draft state BEFORE target sync
        self._speco_diag_draft_state("before_target_sync")
        try:
            with _speco_npu_target_staging(
                self, peft_config=peft_config, use_shm=use_shm
            ):
                result = super().update_weights_from_ipc(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                    use_shm=use_shm,
                )
            # Diagnostic: check draft state AFTER target sync (may be zeroed by wake_up)
            self._speco_diag_draft_state("after_target_sync")
            reloaded = self._speco_reload_draft_from_checkpoint()
            if reloaded > 0:
                logger.warning(
                    "[speco draft reload] drafter weights restored after target sync (%d tensors)",
                    reloaded,
                )
            # Diagnostic: check draft state AFTER reload
            self._speco_diag_draft_state("after_draft_reload")
            return result
        finally:
            if is_npu:
                trim_process_host_memory()

    def _speco_diag_draft_state(self, phase: str):
        """Log norms of key draft model parameters for debugging."""
        if not bool(_bool_or_none(os.getenv(SPECO_VLLM_DRAFT_DIAG_ENV))):
            return

        draft_model, _ = self._speco_resolve_draft_model()
        if draft_model is None:
            logger.warning("[speco-diag:%s] no draft model found", phase)
            return
        try:
            params = dict(draft_model.named_parameters())
            diag_keys = [
                k
                for k in params
                if any(s in k for s in ("markov", "fc.", "norm.", "layers.0."))
            ]
            if not diag_keys:
                diag_keys = list(params.keys())[:5]
            norms = {
                k: f"{params[k].data.float().norm().item():.4f}" for k in diag_keys[:6]
            }
            logger.warning("[speco-diag:%s] draft param norms: %s", phase, norms)
        except Exception as exc:
            logger.warning("[speco-diag:%s] failed: %s", phase, exc)
