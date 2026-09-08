#!/usr/bin/env python3
"""Convert an internal DFlash training checkpoint to the loadable vLLM format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from verl_speco.trainer.checkpoint import rewrite_block_drafter_runtime_config


_LEGACY_WEIGHT_ALIASES = (
    ("context_proj.", "fc."),
    ("context_norm.", "hidden_norm."),
    ("final_norm.", "norm."),
)


def translate_legacy_dflash_state_dict(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Convert legacy training names to the current vLLM-compatible names."""
    translated = {}
    for name, tensor in state_dict.items():
        if name == "embed_tokens.weight" or name.endswith(".embed_tokens.weight"):
            continue
        translated_name = name
        for source, target in _LEGACY_WEIGHT_ALIASES:
            translated_name = translated_name.replace(source, target)
        translated[translated_name] = (
            tensor.detach().to(dtype=dtype, device="cpu").contiguous()
        )
    return translated


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as stream:
        return {name: stream.get_tensor(name) for name in stream.keys()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--template", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    template = args.template.resolve()
    source_path = checkpoint / "model.safetensors"
    template_path = template / "model.safetensors"
    translated = translate_legacy_dflash_state_dict(load_safetensors(source_path))

    with safe_open(template_path, framework="pt", device="cpu") as stream:
        template_shapes = {
            name: tuple(stream.get_slice(name).get_shape()) for name in stream.keys()
        }
    translated_shapes = {
        name: tuple(tensor.shape) for name, tensor in translated.items()
    }
    if translated_shapes != template_shapes:
        missing = sorted(template_shapes.keys() - translated_shapes.keys())
        unexpected = sorted(translated_shapes.keys() - template_shapes.keys())
        mismatched = sorted(
            name
            for name in template_shapes.keys() & translated_shapes.keys()
            if template_shapes[name] != translated_shapes[name]
        )
        raise ValueError(
            f"translated checkpoint does not match template: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )

    temporary_path = checkpoint / "model.safetensors.repairing"
    save_file(translated, temporary_path)
    rewrite_block_drafter_runtime_config("dflash", template, checkpoint)
    os.replace(temporary_path, source_path)
    for filename in (
        "generation_config.json",
        "dflash.py",
        "modeling_dflash.py",
        "utils.py",
    ):
        source = template / filename
        if source.is_file():
            shutil.copy2(source, checkpoint / filename)

    metadata_path = checkpoint / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "format": "pretrained_drafter_checkpoint",
            "serialization": "safetensors",
            "dtype": "bfloat16",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"repaired {checkpoint}: {len(translated)} tensors, {source_path.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
