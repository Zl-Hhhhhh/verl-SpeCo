"""Token-weighted actor entropy on actually reached speculative positions."""

import math


def accumulate_verification_entropy(stats, traces):
    """Add raw sufficient statistics; ignore the suffix after first rejection.

    Missing entropy diagnostics are omitted, not represented by zero entropy.
    Low entropy means H < 0.01 nats (natural logarithm).
    """
    if traces is None:
        return

    def add(key, value):
        stats[key] = stats.get(key, 0.0) + value

    for events in traces:
        if events is None:
            continue
        for event in events:
            if not isinstance(event, dict) or "draft_teacher_entropies" not in event:
                continue
            draft_count = int(event.get("draft_count", 0))
            accepted_count = int(event.get("accepted_count", 0))
            entropies = event["draft_teacher_entropies"]
            if (
                not 0 <= accepted_count <= draft_count
                or len(entropies) != draft_count
                or not all(math.isfinite(h) and h >= 0 for h in entropies)
            ):
                raise RuntimeError("Invalid actor verification entropy trace")
            for position, entropy in enumerate(entropies[:min(draft_count, accepted_count + 1)]):
                add("count", 1)
                add("sum", entropy)
                if position == 0:
                    add("first_count", 1)
                    add("first_sum", entropy)
                stream = "accepted" if position < accepted_count else "rejected"
                add(f"{stream}_count", 1)
                add(f"{stream}_sum", entropy)
                if entropy < 0.01:
                    add("low_count", 1)
                    if stream == "rejected":
                        add("low_rejected_count", 1)


def verification_entropy_metrics(stats):
    """Convert accumulated statistics to scalars suitable for training curves."""
    count = stats.get("count", 0)
    if not count:
        return {}
    metrics = {
        "actor/verify_entropy": stats["sum"] / count,
        "actor/verify_entropy_positions": float(count),
        "actor/verify_low_entropy_share_lt_0p01": stats.get("low_count", 0) / count,
    }
    for stream in ("first", "accepted", "rejected"):
        stream_count = stats.get(f"{stream}_count", 0)
        if stream_count:
            metrics[f"actor/verify_entropy_{stream}"] = stats[f"{stream}_sum"] / stream_count
    low_count = stats.get("low_count", 0)
    if low_count:
        metrics["actor/verify_low_entropy_rejection_rate_lt_0p01"] = (
            stats.get("low_rejected_count", 0) / low_count
        )
    return metrics
