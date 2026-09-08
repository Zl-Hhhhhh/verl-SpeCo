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
import ctypes
import gc
import json
import logging
import os
import pickletools
import re
import sys
import time
import zipfile
from copy import deepcopy
from functools import lru_cache
from glob import glob
from typing import Any, Optional, Union


_DRAFT_STEP_PATTERN = re.compile(r"^draft_step_(\d+)$")
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.index.json",
)
_OPTIMIZER_DCP_METADATA = ".metadata"
_MAX_WEIGHT_HEADER_BYTES = 16 * 1024 * 1024

logger = logging.getLogger(__name__)

_JEMALLOC_ARENAS_ALL = 4096
_JEMALLOC_RECLAIM_MODE_ENV = "SPECO_JEMALLOC_RECLAIM_MODE"


class DrafterCheckpointMetadataError(ValueError):
    """Raised when a managed drafter checkpoint has invalid metadata."""


def _pytorch_archive_has_tensors(path: str) -> bool:
    """Return false only for a readable PyTorch zip archive with no tensors.

    This inspects pickle opcodes rather than deserializing an untrusted
    checkpoint.  A tensor saved by ``torch.save`` has a persistent storage
    reference; an empty state dict does not.  Unknown legacy formats stay
    accepted here and will retain their existing loader behavior.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            data_entries = [
                name
                for name in archive.namelist()
                if name == "data.pkl" or name.endswith("/data.pkl")
            ]
            if len(data_entries) != 1:
                return True
            data_info = archive.getinfo(data_entries[0])
            if data_info.file_size > _MAX_WEIGHT_HEADER_BYTES:
                return True
            payload = archive.read(data_entries[0])
        return any(
            opcode.name in {"PERSID", "BINPERSID"}
            for opcode, _, _ in pickletools.genops(payload)
        )
    except (OSError, ValueError, zipfile.BadZipFile):
        return True


def _safetensors_file_has_tensors(path: str) -> bool:
    """Return false only for a readable safetensors file with no tensor keys."""
    try:
        with open(path, "rb") as stream:
            header_size = int.from_bytes(stream.read(8), byteorder="little")
            if header_size <= 0 or header_size > _MAX_WEIGHT_HEADER_BYTES:
                return True
            header = stream.read(header_size)
        if len(header) != header_size:
            return True
        entries = json.loads(header)
        return isinstance(entries, dict) and any(
            name != "__metadata__" for name in entries
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True


def _has_nonempty_pretrained_weights(path: str) -> bool:
    """Reject only known-empty model weight files before a resume is accepted."""
    index_paths = glob(os.path.join(path, "*.index.json"))
    for index_path in index_paths:
        try:
            with open(index_path, "r", encoding="utf-8") as stream:
                weight_map = json.load(stream).get("weight_map")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(weight_map, dict) and weight_map:
            return True

    saw_weight_file = False
    for weight_path in glob(os.path.join(path, "*.safetensors")):
        saw_weight_file = True
        if _safetensors_file_has_tensors(weight_path):
            return True
    for weight_path in glob(os.path.join(path, "*.bin")):
        saw_weight_file = True
        if _pytorch_archive_has_tensors(weight_path):
            return True
    return False if saw_weight_file or index_paths else False


def _load_json_object(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DrafterCheckpointMetadataError(f"Invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DrafterCheckpointMetadataError(
            f"Invalid JSON object {path}: expected an object"
        )
    return value


def _atomic_json_dump(payload: dict[str, Any], output_path: str) -> None:
    temporary_path = f"{output_path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _fill_if_missing(
    destination: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]
) -> None:
    for key in keys:
        if key in source and key not in destination:
            destination[key] = deepcopy(source[key])


def _ensure_dict_child(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    config[key] = value
    return value


def rewrite_block_drafter_runtime_config(
    backend_type: Optional[str],
    source_model_path: Optional[Union[str, os.PathLike]],
    checkpoint_path: Union[str, os.PathLike],
) -> bool:
    """Make a DFlash-family training checkpoint directly loadable by runtimes.

    SpeCo trains with an internal ``dflash``/``dspark`` config class, while vLLM
    first requires the target-family config (for example ``qwen3``) and applies
    its speculative-model overrides afterward. Keep the internal config beside
    the runtime-facing config so both training resume and inference retain the
    fields they need.
    """
    if backend_type not in {"dflash", "dspark"}:
        return False

    checkpoint = os.fspath(checkpoint_path)
    config_path = os.path.join(checkpoint, "config.json")
    training_config_path = os.path.join(checkpoint, "speco_training_config.json")
    current_config = _load_json_object(config_path)
    if current_config is None:
        raise DrafterCheckpointMetadataError(
            f"Cannot export block drafter runtime config: missing {config_path}"
        )

    training_config = _load_json_object(training_config_path) or current_config
    source_config = None
    if source_model_path:
        source_config = _load_json_object(
            os.path.join(os.fspath(source_model_path), "config.json")
        )
    if source_config is None:
        source_config = current_config

    runtime_model_type = str(source_config.get("model_type", "") or "").lower()
    if runtime_model_type in {"dflash", "dspark", "domino"} or not runtime_model_type:
        raise DrafterCheckpointMetadataError(
            "Cannot derive a runtime-facing block drafter config from "
            f"model_type={runtime_model_type or None!r}; source_model_path={source_model_path!r}"
        )

    runtime_config = deepcopy(source_config)
    runtime_config["speco_training_model_type"] = backend_type
    common_alias_keys = ("target_layer_ids", "mask_token_id", "num_context_layers")
    _fill_if_missing(runtime_config, training_config, common_alias_keys)

    dflash_config = _ensure_dict_child(runtime_config, "dflash_config")
    _fill_if_missing(dflash_config, training_config, common_alias_keys)

    if backend_type == "dspark":
        dspark_config = _ensure_dict_child(runtime_config, "dspark_config")
        _fill_if_missing(
            dspark_config,
            training_config,
            (
                "block_size",
                "num_anchors",
                "markov_rank",
                "markov_head_type",
                "confidence_head_alpha",
                "confidence_head_with_markov",
                "ce_loss_alpha",
                "l1_loss_alpha",
                "loss_decay_gamma",
                "target_layer_ids",
                "num_context_layers",
                "num_target_layers",
                "target_num_hidden_layers",
                "mask_token_id",
            ),
        )
    else:
        dspark_config = {}

    target_layer_ids = (
        runtime_config.get("target_layer_ids")
        or dflash_config.get("target_layer_ids")
        or dspark_config.get("target_layer_ids")
    )
    if (
        target_layer_ids is not None
        and "eagle_aux_hidden_state_layer_ids" not in runtime_config
    ):
        try:
            runtime_config["eagle_aux_hidden_state_layer_ids"] = [
                int(layer_id) + 1 for layer_id in target_layer_ids
            ]
        except (TypeError, ValueError) as exc:
            raise DrafterCheckpointMetadataError(
                f"Invalid target_layer_ids in exported block drafter config: {target_layer_ids!r}"
            ) from exc

    if not os.path.exists(training_config_path):
        _atomic_json_dump(training_config, training_config_path)
    _atomic_json_dump(runtime_config, config_path)
    return True


def _managed_checkpoint_step(path: str) -> Optional[int]:
    match = _DRAFT_STEP_PATTERN.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match is not None else None


def get_drafter_checkpoint_metadata(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    if not model_path:
        return {}

    metadata_path = os.path.join(os.fspath(model_path), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint metadata {metadata_path}: expected a JSON object"
        )
    return metadata


def get_drafter_trainer_state(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    trainer_state = metadata.get("trainer_state")
    if trainer_state is None:
        return {}
    if not isinstance(trainer_state, dict):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter trainer state metadata: expected a JSON object"
        )
    return dict(trainer_state)


def get_drafter_checkpoint_step(
    model_path: Optional[Union[str, os.PathLike]],
) -> Optional[int]:
    """Return the drafter training step recorded in a saved checkpoint directory."""
    if not model_path:
        return None

    path = os.fspath(model_path)
    metadata = get_drafter_checkpoint_metadata(path)
    if not metadata:
        return None
    try:
        step = metadata.get("step")
        if step is not None:
            return int(step)
    except (TypeError, ValueError) as exc:
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint step in {os.path.join(path, 'metadata.json')}: {step!r}"
        ) from exc
    return None


def get_drafter_optimizer_manifest(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    manifest = metadata.get("optimizer")
    if manifest is None:
        return {}
    if not isinstance(manifest, dict):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter optimizer manifest: expected a JSON object"
        )

    checkpoint_format = manifest.get("format")
    relative_path = manifest.get("path")
    if checkpoint_format != "torch_distributed_checkpoint" or not isinstance(
        relative_path, str
    ):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter optimizer manifest: expected torch_distributed_checkpoint format and path"
        )
    normalized_path = os.path.normpath(relative_path)
    if (
        os.path.isabs(normalized_path)
        or normalized_path == ".."
        or normalized_path.startswith(f"..{os.sep}")
    ):
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter optimizer checkpoint path: {relative_path!r}"
        )
    return dict(manifest)


def get_drafter_optimizer_checkpoint_path(
    model_path: Optional[Union[str, os.PathLike]],
) -> Optional[str]:
    if not model_path:
        return None
    manifest = get_drafter_optimizer_manifest(model_path)
    if not manifest:
        return None
    optimizer_path = os.path.join(os.fspath(model_path), manifest["path"])
    if not os.path.isfile(os.path.join(optimizer_path, _OPTIMIZER_DCP_METADATA)):
        raise DrafterCheckpointMetadataError(
            f"Drafter optimizer checkpoint is incomplete: missing {_OPTIMIZER_DCP_METADATA} in {optimizer_path}"
        )
    trainer_state_file = manifest.get("trainer_state_file")
    if (
        not isinstance(trainer_state_file, str)
        or os.path.basename(trainer_state_file) != trainer_state_file
        or not os.path.isfile(os.path.join(optimizer_path, trainer_state_file))
    ):
        raise DrafterCheckpointMetadataError(
            f"Drafter optimizer checkpoint is incomplete: missing trainer state in {optimizer_path}"
        )
    return optimizer_path


def is_pretrained_drafter_checkpoint(
    model_path: Optional[Union[str, os.PathLike]],
) -> bool:
    if not model_path:
        return False
    path = os.fspath(model_path)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "config.json")):
        return False
    if not any(glob(os.path.join(path, pattern)) for pattern in _WEIGHT_PATTERNS):
        return False
    if not _has_nonempty_pretrained_weights(path):
        return False

    metadata_path = os.path.join(path, "metadata.json")
    managed_step = _managed_checkpoint_step(path)
    if managed_step is not None or os.path.exists(metadata_path):
        metadata = get_drafter_checkpoint_metadata(path)
        if not metadata or metadata.get("complete", True) is not True:
            return False
        recorded_step = get_drafter_checkpoint_step(path)
        if managed_step is not None and recorded_step != managed_step:
            return False
        if metadata.get("optimizer") is not None:
            get_drafter_optimizer_checkpoint_path(path)
    return True


def _read_kib(path: str, keys: set[str]) -> dict[str, int]:
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                name, separator, raw_value = line.partition(":")
                if not separator or name not in keys:
                    continue
                fields = raw_value.strip().split()
                if fields:
                    values[name] = int(fields[0])
    except (OSError, ValueError):
        return {}
    return values


def collect_checkpoint_memory_snapshot() -> dict[str, Optional[int]]:
    """Collect Linux process/system memory counters in KiB."""

    process = _read_kib("/proc/self/status", {"VmRSS", "RssAnon", "RssShmem"})
    system = _read_kib(
        "/proc/meminfo",
        {
            "MemTotal",
            "MemAvailable",
            "Cached",
            "Shmem",
            "Dirty",
            "Writeback",
        },
    )
    dev_shm_used_kib = None
    statvfs = getattr(os, "statvfs", None)
    try:
        if statvfs is not None:
            stat = statvfs("/dev/shm")
            dev_shm_used_kib = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) // 1024
    except OSError:
        pass
    return {
        "rss_gib": process.get("VmRSS"),
        "anon_gib": process.get("RssAnon"),
        "rss_shmem_gib": process.get("RssShmem"),
        "node_unavailable_gib": (
            system["MemTotal"] - system["MemAvailable"]
            if "MemTotal" in system and "MemAvailable" in system
            else None
        ),
        "available_gib": system.get("MemAvailable"),
        "cached_gib": system.get("Cached"),
        "node_shmem_gib": system.get("Shmem"),
        "dev_shm_used_gib": dev_shm_used_kib,
        "dirty_gib": system.get("Dirty"),
        "writeback_gib": system.get("Writeback"),
    }


def format_checkpoint_memory_snapshot(
    counters: Optional[dict[str, Optional[int]]] = None,
) -> str:
    """Return concise Linux process/system memory counters without extra dependencies."""

    counters = (
        counters if counters is not None else collect_checkpoint_memory_snapshot()
    )
    return " ".join(
        f"{name}={value / (1024**2):.2f}" if value is not None else f"{name}=n/a"
        for name, value in counters.items()
    )


@lru_cache(maxsize=1)
def _jemalloc_is_active() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if "jemalloc" in os.getenv("LD_PRELOAD", "").lower():
        return True
    try:
        with open("/proc/self/maps", encoding="utf-8") as stream:
            return any("jemalloc" in line.lower() for line in stream)
    except OSError:
        return False


@lru_cache(maxsize=1)
def _jemalloc_mallctl_function() -> Any:
    try:
        runtime = ctypes.CDLL(None)
        mallctl = getattr(runtime, "mallctl", None) or getattr(
            runtime, "je_mallctl", None
        )
        if mallctl is None:
            return None
        mallctl.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        mallctl.restype = ctypes.c_int
        return mallctl
    except Exception:  # noqa: BLE001
        return None


def _jemalloc_mallctl(name: str) -> bool:
    try:
        mallctl = _jemalloc_mallctl_function()
        if mallctl is None:
            return False
        return mallctl(name.encode("ascii"), None, None, None, 0) == 0
    except Exception:  # noqa: BLE001
        return False


def _jemalloc_reclaim_mode() -> str:
    mode = os.getenv(_JEMALLOC_RECLAIM_MODE_ENV, "decay").strip().lower()
    return mode if mode in {"decay", "purge"} else "decay"


def _reclaim_jemalloc_heap() -> bool:
    _jemalloc_mallctl("thread.tcache.flush")
    return _jemalloc_mallctl(f"arena.{_JEMALLOC_ARENAS_ALL}.{_jemalloc_reclaim_mode()}")


def _host_allocator_name() -> str:
    if not sys.platform.startswith("linux"):
        return "unsupported"
    return "jemalloc" if _jemalloc_is_active() else "glibc"


def _trim_process_heap() -> bool:
    """Return unused allocator pages to the operating system when available."""

    if not sys.platform.startswith("linux"):
        return False
    if _jemalloc_is_active():
        return _reclaim_jemalloc_heap()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except Exception:  # noqa: BLE001
        return False


def trim_process_host_memory() -> dict[str, Any]:
    """Return unused jemalloc/glibc pages without forcing a Python GC cycle."""

    started = time.perf_counter()
    allocator = _host_allocator_name()
    heap_trimmed = _trim_process_heap()
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": heap_trimmed,
        "allocator": allocator,
        "reclaim_action": (
            _jemalloc_reclaim_mode() if allocator == "jemalloc" else "malloc_trim"
        ),
    }


def _flush_and_drop_checkpoint_file_cache(checkpoint_path: str) -> tuple[int, int]:
    """Flush completed checkpoint files and advise Linux to evict their cache."""

    if not sys.platform.startswith("linux") or not hasattr(os, "posix_fadvise"):
        return 0, 0
    try:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return 0, 0
    except OSError:
        return 0, 1

    advised = 0
    failed = 0
    paths = []
    if os.path.isfile(checkpoint_path):
        paths.append(checkpoint_path)
    else:
        for root, _, filenames in os.walk(checkpoint_path):
            paths.extend(os.path.join(root, filename) for filename in filenames)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    advice = getattr(os, "POSIX_FADV_DONTNEED", 4)
    for path in paths:
        try:
            fd = os.open(path, flags)
        except OSError:
            failed += 1
            continue
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, advice)
            advised += 1
        except OSError:
            failed += 1
        finally:
            try:
                os.close(fd)
            except OSError:
                failed += 1
    return advised, failed


def release_checkpoint_host_memory(
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    *,
    drop_file_cache: bool = False,
) -> dict[str, Any]:
    """Best-effort release of checkpoint staging and file-cache memory."""

    started = time.perf_counter()
    try:
        gc.collect()
    except Exception:  # noqa: BLE001
        pass
    allocator = _host_allocator_name()
    trimmed_before = _trim_process_heap()
    advised = 0
    failed = 0
    if drop_file_cache and checkpoint_path:
        try:
            advised, failed = _flush_and_drop_checkpoint_file_cache(
                os.fspath(checkpoint_path)
            )
        except Exception:  # noqa: BLE001
            failed = 1
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": trimmed_before,
        "allocator": allocator,
        "reclaim_action": (
            _jemalloc_reclaim_mode() if allocator == "jemalloc" else "malloc_trim"
        ),
        "files_advised": advised,
        "files_failed": failed,
    }


def resolve_drafter_checkpoint_path(
    model_path: Optional[Union[str, os.PathLike]],
    checkpoint_path: Optional[Union[str, os.PathLike]],
    global_step: Optional[int],
) -> Optional[str]:
    """Resolve a drafter model path to the checkpoint matching ``global_step`` when available."""
    original_model_path = os.fspath(model_path) if model_path is not None else None
    if global_step is None:
        return original_model_path

    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return original_model_path
    if step <= 0:
        return original_model_path

    if (
        original_model_path is not None
        and get_drafter_checkpoint_step(original_model_path) == step
        and is_pretrained_drafter_checkpoint(original_model_path)
    ):
        return original_model_path

    candidates = []
    if checkpoint_path:
        root = os.fspath(checkpoint_path)
        if os.path.basename(os.path.normpath(root)) == f"draft_step_{step}":
            candidates.append(root)
        candidates.append(os.path.join(root, f"draft_step_{step}"))

    for candidate in candidates:
        if get_drafter_checkpoint_step(
            candidate
        ) == step and is_pretrained_drafter_checkpoint(candidate):
            return candidate
    return original_model_path


def log_drafter_checkpoint_step(
    logger: logging.Logger,
    model_path: Optional[Union[str, os.PathLike]],
    *,
    action: str = "Loading drafter weights",
) -> Optional[int]:
    step = get_drafter_checkpoint_step(model_path)
    step_text = str(step) if step is not None else "unknown"
    logger.info("%s from %s (drafter_step=%s)", action, model_path, step_text)
    return step
