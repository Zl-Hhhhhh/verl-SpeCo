from __future__ import annotations

import pytest


def test_empty_optimizer_export_does_not_initialize_state(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.trainer.base_trainer import _DrafterOptimizerState

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    monkeypatch.setattr(
        optimizer,
        "step",
        lambda *args, **kwargs: pytest.fail("empty checkpoint export called step()"),
    )

    result = _DrafterOptimizerState(
        model, optimizer, initialize_missing_state=False
    ).state_dict()

    assert result["state"] == {}
    assert result["param_groups"][0]["params"] == ["weight", "bias"]
    assert optimizer.state == {}


def test_nonempty_optimizer_export_uses_distributed_state_dict(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import state_dict as state_dict_module
    from verl_speco.trainer.base_trainer import _DrafterOptimizerState

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.state[next(iter(model.parameters()))]["step"] = torch.tensor(1.0)
    expected = {"state": {"sentinel": 1}, "param_groups": []}
    calls = []

    def fake_get_optimizer_state_dict(received_model, received_optimizer, *, options):
        calls.append((received_model, received_optimizer, options))
        return expected

    monkeypatch.setattr(
        state_dict_module, "get_optimizer_state_dict", fake_get_optimizer_state_dict
    )

    result = _DrafterOptimizerState(
        model, optimizer, initialize_missing_state=False
    ).state_dict()

    assert result is expected
    assert calls and calls[0][0] is model and calls[0][1] is optimizer


def test_empty_optimizer_export_rejects_unowned_parameter() -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.trainer.base_trainer import _DrafterOptimizerState

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)

    with pytest.raises(RuntimeError, match="not owned by the model"):
        _DrafterOptimizerState(
            model, optimizer, initialize_missing_state=False
        ).state_dict()


def test_empty_optimizer_checkpoint_restore_skips_dcp_load(
    monkeypatch, tmp_path
) -> None:
    torch = pytest.importorskip("torch")
    import torch.distributed.checkpoint as dcp
    from verl_speco.trainer import checkpoint as checkpoint_module
    from verl_speco.trainer import base_trainer as base_trainer_module

    optimizer_path = tmp_path / "optimizer"
    optimizer_path.mkdir()
    trainer_state = {
        "version": 1,
        "optimizer_steps_total": 0,
        "training_steps": 0,
        "lr_scheduler": None,
    }
    torch.save(trainer_state, optimizer_path / "trainer_state.pt")
    monkeypatch.setattr(
        checkpoint_module,
        "get_drafter_optimizer_checkpoint_path",
        lambda path: str(optimizer_path),
    )
    monkeypatch.setattr(
        dcp,
        "load",
        lambda *args, **kwargs: pytest.fail("empty restore called dcp.load()"),
    )
    monkeypatch.setattr(
        base_trainer_module, "release_checkpoint_host_memory", lambda: None
    )
    trainer = base_trainer_module.DrafterBaseTrainer.__new__(
        base_trainer_module.DrafterBaseTrainer
    )

    restored = trainer._load_optimizer_checkpoint(
        str(tmp_path),
        {
            "optimizer": {
                "trainer_state_file": "trainer_state.pt",
                "state_initialized": False,
            }
        },
    )

    assert restored == trainer_state
