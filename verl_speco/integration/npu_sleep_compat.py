"""Optional pageable backups for Ascend's synchronous sleep allocator."""

from __future__ import annotations

import importlib
import os


def install_pageable_npu_sleep() -> bool:
    """Avoid the driver's pinned host pool when explicitly enabled for a run.

    CaMem uses blocking aclrtMemcpy in both directions, so ordinary CPU memory
    can hold the backup until wake_up completes. Persistent allocations and
    tag selection retain the allocator's original semantics.
    """
    if os.getenv("SPECO_VLLM_PAGEABLE_SLEEP", "0") != "1":
        return False
    module = importlib.import_module("vllm_ascend.device_allocator.camem")
    allocator = module.CaMemAllocator
    if getattr(allocator.sleep, "_speco_pageable_backup", False):
        return False

    def sleep(self, offload_tags=None):
        if offload_tags is None:
            offload_tags = (allocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        assert isinstance(offload_tags, tuple)
        module.logger.info("SPECO CaMem sleep: synchronous pageable CPU backups")
        for ptr, data in self.pointer_to_data.items():
            if data.tag == allocator.sleep_persistent_tag:
                continue
            handle = data.handle
            if data.tag in offload_tags:
                size = handle[1]
                backup = module.torch.empty(
                    size, dtype=module.torch.uint8, device="cpu", pin_memory=False
                )
                cpu_ptr = backup.data_ptr()
                module.memcpy(cpu_ptr, cpu_ptr + size * 2, ptr, size, 2)
                data.cpu_backup_tensor = backup
            module.unmap_and_release(handle)
        module.gc.collect()
        module.torch.npu.empty_cache()

    sleep._speco_pageable_backup = True
    allocator.sleep = sleep
    return True
