import pytest

from verl_speco.trainer.entropy_sampling import update_certainty_gate


def gate(entropies, accepted=0, streak=0):
    return update_certainty_gate(
        [[{"draft_count": len(entropies), "accepted_count": accepted,
           "draft_teacher_entropies": entropies}]],
        entropy_threshold=0.01, share_threshold=0.7, patience=2, streak=streak,
    )


def test_gate_requires_two_cycles_and_resets_below_threshold():
    first = gate([0.001, 2, 2])
    assert first["active"] == 0
    second = gate([0.001, 2, 2], streak=first["streak"])
    assert second["active"] == 1
    third = gate([0.5, 0.0, 0.0], streak=second["streak"])
    assert third["active"] == third["streak"] == 0


def test_gate_counts_accepted_prefix_and_first_rejection_only():
    result = gate([0.001, 0.5, 0.0], accepted=1)
    assert result["reached_positions"] == 2
    assert result["low_entropy_share"] == 0.5
    result = gate([0.001, 0.001, 0.5], accepted=3)
    assert result["reached_positions"] == 3
    assert result["low_entropy_share"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("entropies", [[float("nan")], [-1], []])
def test_gate_rejects_invalid_or_empty_observations(entropies):
    with pytest.raises(ValueError):
        gate(entropies)
