"""Exercise the real 16-NPU cached training/rollback path without rollout engines."""
import argparse
import asyncio
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from omegaconf import OmegaConf
from transformers import AutoConfig

from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.workers.frozen_actor_lr_worker import FrozenActorLRWorker


async def run(args):
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("cpu:gloo,npu:hccl", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 16
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    mesh = dist.device_mesh.DeviceMesh("npu", torch.arange(16).reshape(4, 4), mesh_dim_names=("dp", "sp"))
    config = OmegaConf.load(args.config)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True)["actor_rollout_ref"])
    cfg.rollout.drafter.training.frozen_actor_lr_output = args.output
    if args.cached_source:
        cfg.rollout.drafter.training.frozen_actor_lr_cached_source = args.cached_source
    cfg.rollout.drafter.training.dspark_dump_k3_logprobs_dir = str(Path(args.output) / "k3_logprobs")
    backend = DSparkTrainerBackend(cfg, AutoConfig.from_pretrained(cfg.model.path))
    trainer = DrafterBaseTrainer(cfg, 16, rank // 4, mesh, backend)
    trainer.increment_rl_step(1)
    worker = object.__new__(FrozenActorLRWorker)
    worker._rank, worker.config, worker.trainer = rank, cfg, trainer
    worker.device_name, worker.in_drafter_train_group, worker.last_global_step = "npu", True, 1
    await worker.ablation_prepare()
    dist.barrier()
    kwargs = dict(first_rejection_ce_alpha=args.ce_alpha, artifact_tag=(
        f"ce_{args.ce_alpha:.2f}" if args.ce_alpha is not None else None))
    first = await worker.ablation_train(args.lr, 5, start=True, **kwargs)
    if args.ce_alpha is not None:
        assert abs(first["metrics"]["dspark/first_rejection_ce_alpha"] - args.ce_alpha) < 1e-7
        assert first["metrics"]["dspark/first_rejection_ce_token_count"] > 0
    dist.barrier()
    await worker.ablation_train(args.lr, 20, **kwargs)
    dist.barrier()
    assert trainer.finalize_candidate_update(1, 0.)
    assert (await worker.ablation_check_restored())["ok"]
    dist.barrier()
    if rank == 0:
        print("PREFLIGHT PASS: 16 ranks, 5+15 real optimizer steps, exact rollback", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ce-alpha", type=float)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--cached-source")
    asyncio.run(run(parser.parse_args()))
