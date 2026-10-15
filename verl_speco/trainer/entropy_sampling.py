"""Actor-certainty gate, using only actually reached verification positions."""
import math


def update_certainty_gate(traces, *, entropy_threshold, share_threshold, patience, streak=0):
    if not 0 < share_threshold <= 1 or patience < 1:
        raise ValueError("Certainty gate requires share in (0, 1] and positive patience")
    if not math.isfinite(entropy_threshold) or entropy_threshold <= 0:
        raise ValueError("Entropy threshold must be finite and positive")
    reached = low = 0
    for events in traces:
        if events is None:
            continue
        for event in events:
            draft = int(event.get("draft_count", 0))
            accepted = int(event.get("accepted_count", 0))
            if not 0 <= accepted <= draft:
                raise ValueError("Invalid verification counts for certainty gate")
            count = min(draft, accepted + 1)
            entropies = event.get("draft_teacher_entropies", [])
            if len(entropies) < count:
                raise ValueError("Certainty gate requires entropy for every reached position")
            for raw in entropies[:count]:
                entropy = float(raw)
                if not math.isfinite(entropy) or entropy < 0:
                    raise ValueError("Invalid actor entropy for certainty gate")
                reached += 1
                low += entropy < entropy_threshold
    if not reached:
        raise ValueError("Certainty gate received no reached verification positions")
    share = low / reached
    streak = min(streak + 1, patience) if share >= share_threshold else 0
    return {
        "active": int(streak >= patience), "streak": streak,
        "low_entropy_share": share, "reached_positions": reached,
        "low_entropy_positions": low,
    }
