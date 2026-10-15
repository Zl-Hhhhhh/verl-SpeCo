from concurrent.futures import Future

import pytest


torch = pytest.importorskip("torch")
base = pytest.importorskip("verl_speco.trainer.base_trainer")


def test_full_export_state_dict_owns_cpu_storage() -> None:
    trainer = base.DrafterBaseTrainer.__new__(base.DrafterBaseTrainer)
    trainer.model = torch.nn.Linear(4, 3, bias=False)
    trainer.training_device_mesh = None
    original = trainer.model.weight.detach().clone()

    exported = trainer._get_full_export_state_dict()

    assert exported["weight"].untyped_storage().data_ptr() != (
        trainer.model.weight.untyped_storage().data_ptr()
    )
    trainer.model.weight.data.zero_()
    assert torch.equal(exported["weight"], original)


def test_synchronous_checkpoint_reclaims_after_model_offload(
    monkeypatch, tmp_path
) -> None:
    trainer = base.DrafterBaseTrainer.__new__(base.DrafterBaseTrainer)
    trainer.checkpoint_dir = str(tmp_path)
    trainer.rollout_dp_rank = 0
    trainer.rank = 0
    trainer.model = torch.nn.Linear(2, 2, bias=False)
    trainer.training_device_mesh = object()
    trainer._pending_full_checkpoint_future = None
    events = []

    trainer._is_checkpoint_leader = lambda: True
    trainer._sync_checkpoint_phase_error = lambda error, phase: None

    def save_async(step, optimizer_manifest, is_final=False, *, reclaim_in_writer=True):
        events.append(("write", reclaim_in_writer))
        future = Future()
        future.set_result(
            {"export_elapsed": 1.0, "write_elapsed": 2.0, "reclaim_elapsed": 0.0}
        )
        return future

    trainer._save_checkpoint_async = save_async
    monkeypatch.setattr(base, "device_name", "npu")
    monkeypatch.setattr(
        base,
        "load_fsdp_model_to_gpu",
        lambda model: events.append("load"),
    )
    monkeypatch.setattr(
        base,
        "offload_fsdp_model_to_cpu",
        lambda model: events.append("offload"),
    )

    def reclaim(path=None, *, drop_file_cache=False):
        events.append(("reclaim", path, drop_file_cache))
        return {"elapsed_sec": 3.0, "files_advised": 1, "files_failed": 0}

    monkeypatch.setattr(base, "release_checkpoint_host_memory", reclaim)

    result = trainer.save_checkpoint(10, wait=True, include_optimizer=False)

    checkpoint = str(tmp_path / "draft_step_10")
    assert result == {"saved": True, "path": checkpoint, "reason": "saved"}
    assert events == [
        "load",
        ("write", False),
        "offload",
        ("reclaim", checkpoint, True),
    ]
