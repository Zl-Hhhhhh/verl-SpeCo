"""Check the migrated local runtime and prepare a disjoint actor source."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch_npu

root = Path(__file__).resolve().parents[3]
output = root / "outputs/directional_repair_diagnostic_20261015/halfnode_r1"
output.mkdir(parents=True, exist_ok=True)
assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
assert torch.npu.device_count() == 8
expected = {
    "torch": "2.10.0", "torch-npu": "2.10.0.post2", "vllm": "0.26.0+empty",
    "vllm-ascend": "0.1.dev1+ge592b2e79.d20260811", "verl": "0.8.0.dev0",
    "verl-speco": "0.1.0", "ray": "2.56.1", "transformers": "5.14.1",
    "triton-ascend": "3.2.1",
}
versions = {name: importlib.metadata.version(name) for name in expected}
assert versions == expected, versions
from vllm_ascend._build_info import __device_type__
assert __device_type__ == "A3"
ops = Path('/workspace/main_env_next/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer')
loaded = []
for library in ops.glob('**/libcust_*rt2.0.so'):
    ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    loaded.append(str(library))
assert len(loaded) == 2
checks = []
for device in range(8):
    with torch.npu.device(device):
        value = torch.ones((16, 16), device=f"npu:{device}", dtype=torch.bfloat16)
        assert (value @ value).sum().item() == 4096
        checks.append({"local_device": device, "physical_logical_device": device + 8, "matmul": "pass"})
        del value
        torch.npu.empty_cache()
print('All eight selected logical devices passed BF16 matmul', flush=True)

source = Path('/workspace/data/dapo_math_17k_verl')
expected_sha = {
    "train": "cdb223011baebcc1bd600d563b7c0b9559610df96c9ca33acc8556a70fa38b89",
    "test": "a94a87fd700142538db1d0e089e81ca98be477c4208163301adb66aab5979bb6",
}
source_sha = {name: hashlib.sha256((source / f'{name}.parquet').read_bytes()).hexdigest()
              for name in expected_sha}
assert source_sha == expected_sha
train, validation = (pq.read_table(source / f'{name}.parquet') for name in ('train', 'test'))
def prompt_key(prompt):
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
validation_keys = {prompt_key(prompt) for prompt in validation['prompt'].to_pylist()}
train_keys = [prompt_key(prompt) for prompt in train['prompt'].to_pylist()]
excluded = [i for i, key in enumerate(train_keys) if key in validation_keys]
keep = [key not in validation_keys for key in train_keys]
actor_source = output / 'actor_train_disjoint.parquet'
if not excluded:
    if actor_source.exists() and not actor_source.is_symlink():
        actor_source.unlink()
    if not actor_source.exists():
        actor_source.symlink_to(source / 'train.parquet')
else:
    pq.write_table(train.filter(keep), actor_source, use_dictionary=False, row_group_size=50000)
print('Actor source prepared; verifying readback', flush=True)
assert {prompt_key(p) for p in pq.read_table(actor_source, columns=['prompt'])['prompt'].to_pylist()}.isdisjoint(validation_keys)
report = {
    "status": "passed", "platform": platform.platform(), "python": platform.python_version(),
    "versions": versions, "device_type": __device_type__, "visible_devices": os.environ['ASCEND_RT_VISIBLE_DEVICES'],
    "device_checks": checks, "custom_operator_libraries_loaded": loaded,
    "runtime_setup": "/workspace/main_env_next/bin/activate-speco-local.sh",
    "source_sha256": source_sha, "actor_train_file": str(actor_source),
    "actor_train_sha256": hashlib.sha256(actor_source.read_bytes()).hexdigest(),
    "source_training_rows": len(train), "actor_training_rows": sum(keep),
    "validation_rows": len(validation), "independent_validation_prompts": len(validation_keys),
    "excluded_source_row_indices": excluded, "validation_prompt_sha256": sorted(validation_keys),
    "seed": 17, "training_validation_overlap": 0,
    "budget_adaptation": {"logical_devices": 8, "independent_replicas": 2, "tp": 4, "sp": 4,
                          "max_samples_per_replica": 40, "holdout_samples_per_replica": 8,
                          "max_tokens_per_replica": 32768, "global_max_tokens": 65536,
                          "max_selected_training_positions": 2048},
    "base_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
}
(output / 'environment_check.json').write_text(json.dumps(report, indent=2) + '\n')
(output / 'npu_before.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))
print(json.dumps(report, indent=2), flush=True)
