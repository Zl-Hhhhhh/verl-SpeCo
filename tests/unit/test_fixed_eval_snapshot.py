import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from verl_speco.trainer.fixed_eval_snapshot import export_fixed_eval_snapshot


def _batch():
    return SimpleNamespace(
        batch={
            "prompts": torch.tensor([[0, 10], [0, 10], [0, 20], [0, 20], [0, 30], [0, 30]]),
            "responses": torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0], [9, 10, 0], [11, 12, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 0]] * 6),
            "response_mask": torch.tensor([[1, 1, 0]] * 6),
            "token_level_scores": torch.ones(6, 3),
        },
        non_tensor_batch={
            "uid": np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
            "speculative_trace": [[{"draft_topk_ids": [[1, 2]], "draft_topk_teacher_logprobs": [[-1., -2.]]}]] * 6,
        },
    )


def test_fixed_eval_excludes_training_rows_and_preserves_full_trajectories(tmp_path):
    actor = tmp_path / "checkpoint/global_step_59/actor"
    actor.mkdir(parents=True)
    batch = _batch()
    mask = torch.tensor([True, False, False, False, False, False])
    manifest = export_fixed_eval_snapshot(
        batch, collect_mask=mask, root=tmp_path / "cohorts", step=60,
        sample_count=3, seed=17, actor_checkpoint=actor, experiment_id="p4", run_id="test",
    )
    assert manifest["actor_snapshot_step"] == 59
    assert manifest["sample_count"] == manifest["unique_prompts"] == 3
    assert 0 not in manifest["sample_indices"]
    with gzip.open(tmp_path / "cohorts/step_60/samples.jsonl.gz", "rt") as stream:
        raw = stream.read()
    assert hashlib.sha256(raw.encode()).hexdigest() == manifest["uncompressed_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert rows[0]["speculative_trace"][0]["draft_topk_ids"] == [[1, 2]]
    assert all(len(row["prompt_token_ids"]) == 1 and len(row["response_token_ids"]) == 2 for row in rows)
    assert rows[0]["response_loss_mask"] == [1, 1]
    assert all(row["excluded_from_source_drafter_training"] for row in rows)
    assert mask.tolist() == [True, False, False, False, False, False]
    assert batch.batch["responses"][0].tolist() == [1, 2, 0]


def test_fixed_eval_repeat_is_immutable_and_selection_deterministic(tmp_path):
    actor = tmp_path / "actor"
    actor.mkdir()
    arguments = dict(
        batch=_batch(), collect_mask=torch.zeros(6, dtype=torch.bool), step=80,
        sample_count=4, seed=17, actor_checkpoint=actor, experiment_id="p4", run_id="test",
    )
    first = export_fixed_eval_snapshot(root=tmp_path / "one", **arguments)
    second = export_fixed_eval_snapshot(root=tmp_path / "two", **arguments)
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    arguments["batch"].batch["responses"].add_(100)
    assert export_fixed_eval_snapshot(root=tmp_path / "one", **arguments) == first
    arguments["run_id"] = "other"
    with pytest.raises(ValueError, match="different run"):
        export_fixed_eval_snapshot(root=tmp_path / "one", **arguments)


def test_fixed_eval_requires_actor_and_available_holdout(tmp_path):
    kwargs = dict(
        batch=_batch(), collect_mask=torch.ones(6, dtype=torch.bool), root=tmp_path / "out",
        step=100, sample_count=128, seed=17, actor_checkpoint=tmp_path / "missing",
        experiment_id="p4", run_id="test",
    )
    with pytest.raises(ValueError, match="actor checkpoint"):
        export_fixed_eval_snapshot(**kwargs)
    kwargs["actor_checkpoint"].mkdir()
    with pytest.raises(ValueError, match="No uncollected"):
        export_fixed_eval_snapshot(**kwargs)
