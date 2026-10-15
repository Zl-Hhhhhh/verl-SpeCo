import pytest

from verl_speco.trainer.verification_entropy import (
    accumulate_verification_entropy,
    verification_entropy_metrics,
)


def event(entropies, accepted):
    return {"draft_count": len(entropies), "accepted_count": accepted,
            "draft_teacher_entropies": entropies}


def test_reached_positions_exclude_counterfactual_suffix():
    stats = {}
    accumulate_verification_entropy(stats, [[event([0.0, 0.5, 99.0], 1)]])
    metrics = verification_entropy_metrics(stats)
    assert metrics["actor/verify_entropy"] == 0.25
    assert metrics["actor/verify_entropy_positions"] == 2
    assert metrics["actor/verify_entropy_accepted"] == 0
    assert metrics["actor/verify_entropy_rejected"] == 0.5
    assert metrics["actor/verify_low_entropy_share_lt_0p01"] == 0.5
    assert metrics["actor/verify_low_entropy_rejection_rate_lt_0p01"] == 0


def test_weighted_merge_all_accepted_and_first_rejection():
    stats = {}
    accumulate_verification_entropy(stats, [[event([1.0, 2.0, 3.0], 3)]])
    accumulate_verification_entropy(stats, [[event([0.005, 99.0], 0)]])
    metrics = verification_entropy_metrics(stats)
    assert metrics["actor/verify_entropy"] == pytest.approx(6.005 / 4)
    assert metrics["actor/verify_entropy_first"] == pytest.approx(1.005 / 2)
    assert metrics["actor/verify_entropy_accepted"] == 2
    assert metrics["actor/verify_entropy_rejected"] == 0.005
    assert metrics["actor/verify_low_entropy_rejection_rate_lt_0p01"] == 1


def test_missing_diagnostics_are_not_fake_zero_entropy():
    stats = {}
    accumulate_verification_entropy(stats, None)
    accumulate_verification_entropy(stats, [None, [{"draft_count": 2}], [event([], 0)]])
    assert verification_entropy_metrics(stats) == {}


@pytest.mark.parametrize("bad", [
    event([float("nan")], 0), event([float("inf")], 0), event([-0.1], 0),
    event([0.1], 2), {"draft_count": 2, "draft_teacher_entropies": [0.1]},
])
def test_malformed_entropy_fails_closed(bad):
    with pytest.raises(RuntimeError, match="Invalid actor verification entropy"):
        accumulate_verification_entropy({}, [[bad]])
