import pytest
import torch

from verl_speco.trainer.directional_sampling import (
    build_repair_targets, matched_directional_selections, paired_repair_scores,
    select_directional_anchors,
)


def test_paired_scores_distinguish_new_from_historical_mismatch():
    draft = torch.tensor([[0.6, 0.4]] * 3).log()
    new = torch.tensor([[0.4, 0.6]] * 3).log()
    old = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]]).log()
    scores = paired_repair_scores(old, new, draft)
    assert scores["m"].tolist() == pytest.approx([0.2, 0.1, 0.0], abs=1e-6)
    assert scores["current_tv"].tolist() == pytest.approx([0.2] * 3, abs=1e-6)
    assert select_directional_anchors([7, 3, 9], scores["m"], 3) == [3, 7]


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_repair_targets_move_equal_mass_and_preserve_probability(eta):
    old = torch.tensor([[.3, .1, .6], [.4, .6, 0.]])
    new = torch.tensor([[.1, .3, .6], [.4, .6, 0.]])
    draft = old.clone()
    targets = build_repair_targets(old, new, draft, eta)
    assert targets["m"].tolist() == pytest.approx([.2, 0], abs=1e-6)
    for key in ("direction", "all_mismatch"):
        result = targets[key]
        assert result[0].tolist() == pytest.approx([.3 - eta * .2, .1 + eta * .2, .6], abs=1e-6)
        assert result[1].tolist() == pytest.approx(draft[1].tolist(), abs=1e-6)
        assert result.sum(-1).tolist() == pytest.approx([1, 1], abs=1e-6)


def test_repair_target_soft_tv_can_improve_without_greedy_change():
    old = torch.tensor([[.3, .1, .6]])
    new = torch.tensor([[.1, .3, .6]])
    targets = build_repair_targets(old, new, old, .5)
    assert old.argmax(-1).item() == targets["direction"].argmax(-1).item()
    assert targets["current_tv"].item() == pytest.approx(.2, abs=1e-6)


def test_directional_selection_does_not_backfill_or_keep_one_sided_mismatch():
    draft = torch.tensor([[0.6, 0.4]]).log()
    new = torch.tensor([[0.4, 0.6]]).log()
    old = torch.tensor([[0.3, 0.7]]).log()
    scores = paired_repair_scores(old, new, draft)
    assert scores["surplus"].item() == 0
    assert scores["m"].item() == 0
    assert select_directional_anchors([4], scores["m"], 32) == []


def test_paired_scores_reject_incomplete_or_nonfinite_distributions():
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="same"):
        paired_repair_scores(logits, logits[:1], logits)
    bad = logits.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_repair_scores(logits, logits, bad)


def test_matched_directional_selections_share_budget_and_are_reproducible():
    candidates = [(1, 0.0, 0.8), (2, 0.2, 0.3),
                  (3, 0.1, 0.9), (4, 0.0, 0.2)]
    first = matched_directional_selections(candidates, 3, seed=17, sample_id="sample")
    second = matched_directional_selections(candidates, 3, seed=17, sample_id="sample")
    assert first == second
    assert all(len(anchors) == 2 for anchors in first.values())
    assert first["top_tv"] == [1, 3]
    assert first["top_m"] == [2, 3]
    with pytest.raises(ValueError, match="finite"):
        matched_directional_selections([(1, float("nan"), 0.1)], 1,
                                       seed=17, sample_id="sample")
