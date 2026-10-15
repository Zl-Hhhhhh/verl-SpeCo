"""Fresh-interpreter Ray serialization regression; no Ray cluster or NPU needed."""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import ray
import ray.cloudpickle as cloudpickle
from verl import DataProto
from verl.experimental.agent_loop import agent_loop as upstream


def child():
    cls = cloudpickle.loads(sys.stdin.buffer.read())
    captured = []

    async def server_generate(self, **kwargs):
        captured.append(kwargs["sampling_params"])
        return SimpleNamespace(extra_fields={})

    # Stub only model/server setup. Keep the actual serialized worker dispatch,
    # its constructor patch installation, and the patched client call boundary.
    upstream.AgentLoopWorker.__init__ = lambda self: None
    upstream.LLMServerClient.generate = server_generate
    worker = object.__new__(cls)
    cls.__init__(worker)
    worker.rollout_config = OmegaConf.create(dict(
        temperature=1, top_p=1, top_k=-1, calculate_log_probs=False,
        val_kwargs=dict(temperature=0, top_p=1, top_k=-1),
        agent=dict(default_agent_loop="single_turn_agent")))

    async def trajectory(step, indices, validate):
        return [dict(step=step, index=i, validate=validate) for i in indices]

    upstream.get_trajectory_info = trajectory
    upstream.RolloutTraceConfig = SimpleNamespace(get_instance=lambda: SimpleNamespace(max_samples_per_step_per_worker=None))
    client = object.__new__(upstream.LLMServerClient)

    async def run(params, trajectory, **kwargs):
        return await client.generate(sampling_params=params, request_id="probe")

    worker._run_agent_loop = run
    worker._postprocess = lambda outputs, **kwargs: SimpleNamespace(non_tensor_batch={})
    for meta in (dict(validate=True, do_sample=False, global_steps=5,
                      speco_candidate_gate_max_tokens=512),
                 dict(validate=False, do_sample=True, global_steps=6)):
        batch = DataProto.from_dict(non_tensors={"raw_prompt": np.array(["diagnostic"], dtype=object)}, meta_info=meta)
        asyncio.run(worker.generate_sequences(batch))
    assert captured[0]["max_tokens"] == 512, captured
    assert captured[0]["_verl_skip_drafter_collection"] is True, captured
    assert captured[0]["_verl_global_steps"] == 5, captured
    assert captured[0]["temperature"] == 0, captured
    assert "max_tokens" not in captured[1], captured
    assert not captured[1].get("_verl_skip_drafter_collection", False), captured
    assert captured[1]["_verl_global_steps"] == 6, captured
    print("RESULT_JSON=" + json.dumps(captured))


def main():
    from verl_speco.integration.agent_loop_runtime import (
        _configure_speco_agent_loop_manager_instance, install_agent_loop_runtime_patch)

    install_agent_loop_runtime_patch()
    manager = SimpleNamespace(config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm"}}}))
    _configure_speco_agent_loop_manager_instance(manager, upstream.AgentLoopWorker)
    cls = manager.agent_loop_workers_class.__ray_metadata__.modified_class
    assert cls._speco_explicit_worker_runtime
    result = subprocess.run([sys.executable, __file__, "child"],
                            input=cloudpickle.dumps(cls), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout.decode() + result.stderr.decode())
    print(next(line for line in result.stdout.decode().splitlines() if line.startswith("RESULT_JSON=")))


if __name__ == "__main__":
    child() if len(sys.argv) > 1 and sys.argv[1] == "child" else main()
