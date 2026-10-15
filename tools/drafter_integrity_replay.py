"""Real DSpark/FSDP2 lifecycle replay. Run with torchrun; never touches old runs."""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from omegaconf import OmegaConf
from transformers import AutoConfig
from safetensors import safe_open

from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.state_diagnostics import clone_tree, tensor_record


def digest(x):
    return tensor_record(x, retain_tensor=False)[0]["sha256"]


def tree_digest(x):
    if torch.is_tensor(x):
        return {"hash": digest(x), "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): tree_digest(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tree_digest(v) for v in x]
    return x


def local(x):
    x = x.detach()
    return x.to_local() if hasattr(x, "to_local") else x


def reload_check(path, expected):
    actual = {}
    for p in sorted(Path(path).glob("pytorch_model*.bin")):
        actual.update(torch.load(p, map_location="cpu", weights_only=True))
    hashes = {k: digest(v) for k, v in actual.items()}
    wanted = json.loads(Path(expected).read_text())
    result = {"ok": hashes == wanted, "keys": len(hashes),
              "different": [k for k in set(hashes) | set(wanted) if hashes.get(k) != wanted.get(k)]}
    Path(path, "integrity_reload.json").write_text(json.dumps(result, indent=2))
    if not result["ok"]:
        raise RuntimeError(result)


class Probe:
    def __init__(self, trainer, root):
        self.t = trainer
        self.rank = dist.get_rank()
        self.root = Path(root) / f"rank_{self.rank:02d}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.baseline = None
        self.baseline_hashes = {}
        self.failed = False
        self.sequence = 0
        names = {id(p): name for name, p in trainer.model.named_parameters()}
        self.fsdp_params = {}
        for module in trainer.model.modules():
            if not hasattr(module, "_get_fsdp_state"):
                continue
            group = module._get_fsdp_state()._fsdp_param_group
            if group is not None:
                for p in group.fsdp_params:
                    self.fsdp_params[names[id(p.sharded_param)]] = p
        assert len(self.fsdp_params) == len(names)

    def parameters(self):
        # named_parameters() may expose BF16 full parameters after no-grad
        # forward. Follow the FSDP-owned FP32 master shards across reshard/to().
        return ((name, p.sharded_param) for name, p in self.fsdp_params.items())

    def emit(self, event, **kw):
        self.sequence += 1
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"event": event, "sequence": self.sequence,
                                "time": time.time(), "rank": self.rank, **kw}) + "\n")
        if self.rank == 0:
            print(f"INTEGRITY {event} {kw.get('ok', '')}", flush=True)

    def snapshot(self):
        return {k: local(v).cpu().clone() for k, v in self.parameters()}

    def state(self, event, compare=True):
        torch.npu.synchronize()
        records, changes = {}, {}
        for name, param in self.parameters():
            v = local(param)
            rec, cpu = tensor_record(param)
            rec.update(storage_ptr=v.untyped_storage().data_ptr(), offset=v.storage_offset(),
                       storage_bytes=v.untyped_storage().nbytes())
            records[name] = rec
            if compare and self.baseline is not None and rec["sha256"] != self.baseline_hashes[name]:
                old = self.baseline[name]
                mask = (old != 0) & (cpu == 0)
                coords = mask.nonzero()
                changes[name] = {"different": int((cpu != old).sum()),
                                 "nonzero_to_zero": int(mask.sum()),
                                 "max_abs": float((cpu - old).abs().max()),
                                 "zero_bounds": [coords.amin(0).tolist(), coords.amax(0).tolist()] if coords.numel() else []}
                torch.save({"before": old, "after": cpu}, self.root / f"failure_{self.sequence}_{name}.pt")
        self.failed |= bool(changes)
        self.emit(event, ok=not changes, compared_to_baseline=compare and self.baseline is not None,
                  changes=changes, parameters=records,
                  buffers=tree_digest(dict(self.t.model.named_buffers())),
                  optimizer=tree_digest(self.t.optimizer.state_dict()),
                  scheduler=self.t.lr_scheduler.state_dict(),
                  counters=[self.t.optimizer_steps_total, self.t.training_steps])
        return not changes

    def stop_if_failed(self):
        flag = torch.tensor(int(self.failed), device=self.t.runtime_device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            # No gather/export before preserving each rank's original local shards.
            torch.save(self.snapshot(), self.root / "failure_local_shards.pt")
            torch.save(clone_tree(self.t.optimizer.state_dict()), self.root / "failure_optimizer.pt")
            self.emit("FAIL", ok=False)
            raise RuntimeError("Integrity mismatch; local shards preserved on all ranks")

    def replicas(self, event):
        values = {"parameters": {k: digest(v) for k, v in self.parameters()},
                  "optimizer": tree_digest(self.t.optimizer.state_dict())}
        peers = [None] * 4
        dist.all_gather_object(peers, values, group=self.t.training_device_mesh["dp"].get_group())
        differences = []
        for dp_rank, other in enumerate(peers):
            if other != peers[0]:
                differences.append({"dp_rank": dp_rank,
                                    "parameters": [k for k in other["parameters"] if other["parameters"][k] != peers[0]["parameters"][k]],
                                    "optimizer_equal": other["optimizer"] == peers[0]["optimizer"]})
        self.failed |= bool(differences)
        self.emit(event, ok=not differences, replica_differences=differences,
                  optimizer_states=[len(p["optimizer"]["state"]) for p in peers])
        self.stop_if_failed()


async def run(args):
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(args.backend, timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 16, "This experiment requires the original DP4/SP4 topology"
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    mesh = dist.device_mesh.DeviceMesh("npu", torch.arange(16).reshape(4, 4), mesh_dim_names=("dp", "sp"))
    full_config = OmegaConf.load(args.config)
    # Resolve interpolations before extracting actor_rollout_ref.
    cfg = OmegaConf.create(OmegaConf.to_container(full_config, resolve=True)["actor_rollout_ref"])
    if args.model_path:
        cfg.rollout.drafter.model_path = args.model_path
    cfg.rollout.drafter.checkpoint_path = str(Path(args.output) / "checkpoints")
    cfg.rollout.drafter.training.dspark_dump_k3_logprobs = True
    cfg.rollout.drafter.training.dspark_dump_k3_logprobs_max_tokens_per_stream = 0
    cfg.rollout.drafter.training.dspark_dump_k3_logprobs_dir = str(Path(args.output) / "k3")
    backend = DSparkTrainerBackend(cfg, AutoConfig.from_pretrained(cfg.model.path))
    trainer = DrafterBaseTrainer(cfg, 16, rank // 4, mesh, backend)
    assert await trainer.activate_training_model()
    trainer.model.train()
    probe = Probe(trainer, args.output)
    # Check initialization directly against the original file, not a gathered export.
    original_checks = {}
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    reference_model_path = args.reference_model_path or cfg.rollout.drafter.model_path
    with safe_open(str(Path(reference_model_path) / "model.safetensors"), framework="pt") as sf:
        for name, param in trainer.model.named_parameters():
            key = name.removeprefix("draft_model.")
            if key.endswith("embed_tokens.weight") or key not in sf.keys():
                continue
            value = local(param).cpu()
            if hasattr(param, "placements"):
                shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
                expected = sf.get_slice(key)[tuple(slice(o, o + n) for o, n in zip(offset, shape))]
            else:
                offset = [0] * param.ndim
                expected = sf.get_tensor(key)
            ok = torch.equal(value, expected.to(value.dtype))
            original_checks[name] = {"ok": ok, "global_offset": list(offset), "hash": digest(value)}
            probe.failed |= not ok
    probe.emit("original_file_matches_local_shards", ok=not probe.failed, tensors=original_checks)
    probe.stop_if_failed()
    import verl_speco.trainer.checkpoint as checkpoint
    import verl_speco.trainer.base_trainer as base
    if args.pure_empty_optimizer_export:
        original_optimizer_state = base._DrafterOptimizerState.state_dict
        def optimizer_state(adapter):
            if adapter.optimizer.state:
                return original_optimizer_state(adapter)
            # Diagnostic single-factor bypass: export an empty Adam state by
            # parameter FQN, without DCP's hidden zero-LR optimizer.step().
            names = {id(p): name for name, p in adapter.model.named_parameters()}
            groups = []
            for group in adapter.optimizer.param_groups:
                groups.append({k: [names[id(p)] for p in v] if k == "params" else copy.deepcopy(v)
                               for k, v in group.items()})
            probe.emit("pure_empty_optimizer_export", ok=True)
            return {"state": {}, "param_groups": groups}
        base._DrafterOptimizerState.state_dict = optimizer_state
    original_optimizer_step = trainer.optimizer.step
    def observed_optimizer_step(*a, **k):
        probe.emit("optimizer_step_call", lrs=[float(g["lr"]) for g in trainer.optimizer.param_groups])
        return original_optimizer_step(*a, **k)
    observed_optimizer_step._wrapped_by_lr_sched = True
    trainer.optimizer.step = observed_optimizer_step
    original_reclaim = checkpoint.release_checkpoint_host_memory
    def reclaim(*a, **k):
        probe.state("reclaim_enter_after_payload_clear")
        if args.no_reclaim:
            result = {"elapsed_sec": 0.0, "files_advised": 0, "files_failed": 0}
        else:
            result = original_reclaim(*a, **k)
        probe.state("reclaim_return")
        return result
    checkpoint.release_checkpoint_host_memory = reclaim
    base.release_checkpoint_host_memory = reclaim
    input_paths = sorted(Path(args.inputs).glob(f"rank_{rank // 4 * 4:04d}_*/inputs/*.pt"))
    assert input_paths, "Missing real prepared input"
    artifact = torch.load(input_paths[0], map_location="cpu", weights_only=False)
    trainer._current_pad_size = int(artifact["pad_size"])
    artifacts = [artifact] + [torch.load(p, map_location="cpu", weights_only=False) for p in input_paths[1:args.batches]]
    assert len(artifacts) == args.batches
    batch = {k: v.to(trainer.runtime_device) if torch.is_tensor(v) else v for k, v in artifact["batch"].items()}
    input_hash = tree_digest(artifact["batch"])
    probe.emit("manifest", case=args.case, config=OmegaConf.to_container(cfg, resolve=True),
               input=str(input_paths[0]), input_hash=input_hash, mesh=mesh.mesh.tolist(),
               environment={k: os.environ.get(k) for k in ["LD_PRELOAD", "MALLOC_CONF", "PYTORCH_NPU_ALLOC_CONF"]},
               options=vars(args))
    torch.save(artifact, probe.root / "fixed_input.pt")
    original_export = trainer._get_pretrained_export_state_dict
    original_save = trainer._get_pretrained_export_model()[0].save_pretrained
    export_model = trainer._get_pretrained_export_model()[0]

    def export():
        probe.state("export_enter", compare=args.case not in ("E", "F", "Ewarm"))
        payload = original_export()
        probe.state("export_return", compare=args.case not in ("E", "F", "Ewarm"))
        if args.clone_payload:
            payload = {k: v.clone() for k, v in payload.items()}
        if payload:
            (probe.root / "export_hashes.json").write_text(json.dumps({k: digest(v) for k, v in payload.items()}))
            probe.emit("payload", tensors={k: {"hash": digest(v), "ptr": v.untyped_storage().data_ptr(), "bytes": v.untyped_storage().nbytes()} for k, v in payload.items()})
        return payload

    def save(path, **kwargs):
        probe.state("writer_enter")
        expected = Path(path) / "integrity_expected.json"
        expected.write_text(json.dumps({k: digest(v) for k, v in kwargs["state_dict"].items()}))
        result = original_save(path, **kwargs)
        probe.state("writer_return")
        return result

    trainer._get_pretrained_export_state_dict = export
    export_model.save_pretrained = save
    if args.inline:
        from concurrent.futures import Future
        class InlineExecutor:
            def submit(self, fn):
                f = Future()
                try:
                    f.set_result(fn())
                except BaseException as exc:
                    f.set_exception(exc)
                return f
            def shutdown(self, **kwargs):
                pass
        trainer._full_checkpoint_executor = InlineExecutor()

    rng_cpu, rng_npu = torch.get_rng_state(), torch.npu.get_rng_state()
    head_hash = digest(backend.target_lm_head.fc.weight)
    def forward(event):
        torch.set_rng_state(rng_cpu)
        torch.npu.set_rng_state(rng_npu)
        trainer.model.train()
        # Enable backend intermediate outputs without enabling trainer decorators.
        os.environ["SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR"] = str(probe.root)
        try:
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                out = backend.compute_loss(trainer.model, {k: v for k, v in batch.items() if not k.startswith("_speco_")}, artifact["pad_size"])
        finally:
            os.environ.pop("SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR", None)
        output = clone_tree(out)
        torch.save(output, probe.root / f"{event}.pt")
        signature = tree_digest(output)
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        peers = [None] * 4
        dist.all_gather_object(peers, fingerprint, group=mesh["sp"].get_group())
        head_ok = digest(backend.target_lm_head.fc.weight) == head_hash
        same = len(set(peers)) == 1 and head_ok
        probe.failed |= not same
        probe.emit(event, output=signature, sp_equal=same, sp_hashes=peers, head_unchanged=head_ok)
        return signature

    baseline_out = forward("baseline_forward")
    if args.case == "Ewarm":
        assert await trainer._training_step_on_batch(batch, 0)
        baseline_out = forward("warm_baseline_forward")
    probe.baseline = probe.snapshot()
    probe.baseline_hashes = {k: digest(v) for k, v in probe.baseline.items()}
    probe.state("baseline")
    probe.replicas("baseline_replicas")
    baseline_opt = tree_digest(trainer.optimizer.state_dict())
    baseline_scheduler = trainer.lr_scheduler.state_dict().copy()
    baseline_counters = [trainer.optimizer_steps_total, trainer.training_steps]
    for cycle in range(1, args.cycles + 1):
        trainer.current_rl_step = cycle * 5
        probe.emit("cycle_start", cycle=cycle)
        if args.case == "B":
            await trainer.cleanup_training()
            probe.state("offloaded")
            assert await trainer.activate_training_model()
            probe.state("activated")
        if args.case == "C":
            payload = trainer._get_pretrained_export_state_dict()
            payload.clear()
            gc.collect()
            probe.state("payload_released")
        if args.case in ("E", "F", "Ewarm"):
            baseline_opt = tree_digest(trainer.optimizer.state_dict())
            baseline_scheduler = trainer.lr_scheduler.state_dict().copy()
            baseline_counters = [trainer.optimizer_steps_total, trainer.training_steps]
            forward(f"cycle_{cycle}_pre_update")
            trainer.begin_candidate_update(cycle * 5)
            assert trainer._candidate_local_baseline is not None
            probe.state("candidate_begin")
            for microstep in range(5):
                if args.fresh_batches:
                    source = artifacts[microstep % len(artifacts)]["batch"]
                    train_batch = {k: v.to(trainer.runtime_device) if torch.is_tensor(v) else v for k, v in source.items()}
                else:
                    train_batch = batch
                assert await trainer._training_step_on_batch(train_batch, cycle * 5)
                del train_batch
            probe.state("updated", compare=False)
            probe.replicas("updated_replicas")
            forward(f"cycle_{cycle}_post_update")
            trainer.prepare_candidate_update(cycle * 5)
            assert trainer._candidate_step == cycle * 5
            probe.state("candidate_prepared", compare=False)
            if args.case == "F":
                await trainer.cleanup_training()
                probe.state("candidate_offloaded", compare=False)
            assert trainer.finalize_candidate_update(cycle * 5, 0.0)
            probe.state("rolled_back")
            restored = (tree_digest(trainer.optimizer.state_dict()) == baseline_opt
                        and trainer.lr_scheduler.state_dict() == baseline_scheduler
                        and [trainer.optimizer_steps_total, trainer.training_steps] == baseline_counters)
            probe.failed |= not restored
            probe.emit("optimizer_restored", ok=restored)
            probe.replicas("rollback_replicas")
        probe.stop_if_failed()
        if args.case == "D" or (args.case == "F" and cycle % 2 == 0):
            if args.offloaded_entry and args.case == "D":
                await trainer.cleanup_training()
                probe.state("save_entry_offloaded")
            if args.optimizer_only:
                result = {"optimizer": trainer._save_optimizer_checkpoint(
                    str(Path(args.output) / "checkpoints" / f"draft_step_{cycle * 5}")
                )}
            else:
                result = trainer.save_checkpoint(cycle * 5, wait=True, include_optimizer=not args.weights_only)
            probe.emit("checkpoint_result", result=result)
            probe.state("save_complete")
            probe.replicas("save_replicas")
            if rank == 0 and not args.optimizer_only:
                path = Path(result["path"])
                subprocess.run([sys.executable, __file__, "--reload", str(path), "--expected", str(path / "integrity_expected.json")], check=True)
                probe.emit("fresh_process_reload", ok=True)
        probe.stop_if_failed()
        assert await trainer.activate_training_model()
        output = forward(f"cycle_{cycle}_forward")
        # Exact same input, mode and RNG; A establishes whether exact equality holds.
        equal = output == baseline_out
        probe.failed |= not equal
        probe.emit("forward_equal", cycle=cycle, ok=equal)
        probe.state("cycle_complete")
        probe.stop_if_failed()
    probe.emit("PASS", ok=True, cycles=args.cycles)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["A", "B", "C", "D", "E", "Ewarm", "F"])
    parser.add_argument("--config", default="outputs/2026-09-18/15-44-03/.hydra/config.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--reference-model-path")
    parser.add_argument("--inputs", default="outputs/drafter_analysis/dspark_dapo2000_online_k3_every5_front512_60step_20260912/state_diagnostics")
    parser.add_argument("--output")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--backend", default="cpu:gloo,npu:hccl")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--fresh-batches", action="store_true")
    parser.add_argument("--clone-payload", action="store_true")
    parser.add_argument("--no-reclaim", action="store_true")
    parser.add_argument("--inline", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--offloaded-entry", action="store_true")
    parser.add_argument("--pure-empty-optimizer-export", action="store_true")
    parser.add_argument("--optimizer-only", action="store_true")
    parser.add_argument("--reload")
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.reload:
        reload_check(args.reload, args.expected)
    else:
        assert args.output and args.case
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
