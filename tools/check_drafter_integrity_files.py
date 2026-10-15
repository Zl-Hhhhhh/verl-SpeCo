"""Compare replay checkpoint exports with the original BF16 drafter file."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def check(path, reference):
    result = {"path": str(path), "same": [], "changed": {}, "missing_reference": []}
    with safe_open(str(reference), framework="pt") as sf:
        for shard in sorted(path.glob("pytorch_model*.bin")):
            weights = torch.load(shard, map_location="cpu", weights_only=True)
            for key, value in weights.items():
                if key not in sf.keys():
                    result["missing_reference"].append(key)
                    continue
                original = sf.get_tensor(key).to(value.dtype)
                if torch.equal(value, original):
                    result["same"].append(key)
                else:
                    mask = (original != 0) & (value == 0)
                    coords = mask.nonzero()
                    result["changed"][key] = {
                        "different": int((original != value).sum()),
                        "nonzero_to_zero": int(mask.sum()),
                        "max_abs": float((original - value).abs().max()),
                        "bounds": [coords.amin(0).tolist(), coords.amax(0).tolist()] if coords.numel() else [],
                    }
            del weights
    result["ok"] = bool(result["same"]) and not result["changed"] and not result["missing_reference"]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--reference", type=Path, default=Path("/workspace/models/dspark_qwen3_8b_block7/model.safetensors"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    paths = sorted({f.parent for f in args.root.rglob("pytorch_model*.bin")})
    results = [check(path, args.reference) for path in paths]
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps([{**r, "same": len(r["same"])} for r in results], indent=2))
    raise SystemExit(0 if results and all(r["ok"] for r in results) else 1)
