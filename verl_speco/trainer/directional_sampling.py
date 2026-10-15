"""Scores paired actor/drafter distributions for directional anchor selection."""

from __future__ import annotations

import math
import hashlib

import torch


def paired_repair_scores(
    old_actor_logits: torch.Tensor,
    new_actor_logits: torch.Tensor,
    frozen_drafter_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact full-vocabulary surplus, deficit, m, and current TV."""
    if (
        old_actor_logits.shape != new_actor_logits.shape
        or old_actor_logits.shape != frozen_drafter_logits.shape
        or old_actor_logits.ndim != 2
    ):
        raise ValueError("Paired logits must have the same [positions, vocabulary] shape")
    if old_actor_logits.shape[1] < 2:
        raise ValueError("Paired logits require a full vocabulary")

    old = torch.softmax(old_actor_logits.float(), dim=-1)
    new = torch.softmax(new_actor_logits.float(), dim=-1)
    draft = torch.softmax(frozen_drafter_logits.float(), dim=-1)
    if not (torch.isfinite(old).all() and torch.isfinite(new).all() and torch.isfinite(draft).all()):
        raise ValueError("Paired distributions must be finite")

    surplus = torch.minimum((old - new).clamp_min(0), (draft - new).clamp_min(0)).sum(-1)
    deficit = torch.minimum((new - old).clamp_min(0), (new - draft).clamp_min(0)).sum(-1)
    return {
        "surplus": surplus,
        "deficit": deficit,
        "m": torch.minimum(surplus, deficit),
        "current_tv": 0.5 * (new - draft).abs().sum(-1),
        "actor_tv": 0.5 * (new - old).abs().sum(-1),
        "actor_entropy": -(new * new.clamp_min(1e-30).log()).sum(-1),
    }


def build_repair_targets(
    old: torch.Tensor, new: torch.Tensor, draft: torch.Tensor, eta: float,
) -> dict[str, torch.Tensor]:
    """Build directional and mass-matched targets from full-vocabulary probabilities."""
    if old.shape != new.shape or new.shape != draft.shape or old.ndim != 2:
        raise ValueError("Paired probabilities must have the same [positions, vocabulary] shape")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    old, new, draft = (value.float() for value in (old, new, draft))
    for value in (old, new, draft):
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("Paired probabilities must be finite and nonnegative")
        if not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]), atol=1e-5, rtol=0):
            raise ValueError("Paired probabilities must sum to one")
    surplus = torch.minimum((old - new).clamp_min(0), (draft - new).clamp_min(0))
    deficit = torch.minimum((new - old).clamp_min(0), (new - draft).clamp_min(0))
    s, u = surplus.sum(-1), deficit.sum(-1)
    m = torch.minimum(s, u)
    tv = 0.5 * (new - draft).abs().sum(-1)
    actor_tv = 0.5 * (new - old).abs().sum(-1)
    if (m > torch.minimum(tv, actor_tv) + 1e-5).any():
        raise ValueError("Directional repair mass exceeds actor change or current mismatch")
    moved = eta * m
    direction = draft - (moved / s.clamp_min(1e-30))[:, None] * surplus
    direction = direction + (moved / u.clamp_min(1e-30))[:, None] * deficit
    all_mismatch = (1 - (moved / tv.clamp_min(1e-30)))[:, None] * draft
    all_mismatch = all_mismatch + (moved / tv.clamp_min(1e-30))[:, None] * new
    for target in (direction, all_mismatch):
        if (target < -1e-6).any() or not torch.allclose(
            target.sum(-1), torch.ones_like(m), atol=1e-5, rtol=0
        ):
            raise ValueError("Repair target is not a probability distribution")
        actual = 0.5 * (new - target).abs().sum(-1)
        if not torch.allclose(actual, tv - moved, atol=1e-5, rtol=0):
            raise ValueError("Repair target did not achieve its exact TV improvement")
    return {"direction": direction, "all_mismatch": all_mismatch,
            "surplus": s, "deficit": u, "m": m, "current_tv": tv,
            "actor_tv": actor_tv}


def select_directional_anchors(
    anchors: list[int], scores: torch.Tensor, budget: int, *, min_m: float = 0.0
) -> list[int]:
    """Choose the largest reliable m values; leave unused capacity empty."""
    if scores.ndim != 1 or len(anchors) != scores.numel():
        raise ValueError("Each candidate anchor needs one m score")
    if len(set(anchors)) != len(anchors) or budget < 0 or min_m < 0:
        raise ValueError("Invalid anchor selection budget or candidate IDs")
    values = scores.detach().float().cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite m score")
    ranked = sorted(
        ((-score, anchor) for anchor, score in zip(anchors, values, strict=True) if score > min_m),
        key=lambda pair: (pair[0], pair[1]),
    )
    return sorted(anchor for _, anchor in ranked[:budget])


def matched_directional_selections(
    candidates: list[tuple[int, float, float]], budget: int, *, seed: int, sample_id: str
) -> dict[str, list[int]]:
    """Choose equal-size random, current-TV, and incremental-m anchor sets."""
    anchors = [anchor for anchor, _, _ in candidates]
    if budget < 0 or len(anchors) != len(set(anchors)) or not sample_id:
        raise ValueError("Invalid matched directional candidates")
    if any(not math.isfinite(m) or not math.isfinite(tv) or m < 0 or tv < 0
           for _, m, tv in candidates):
        raise ValueError("Directional candidate scores must be finite and non-negative")
    count = min(budget, sum(m > 0 for _, m, _ in candidates))
    random_rank = lambda anchor: hashlib.sha256(
        f"{seed}:{sample_id}:{anchor}".encode()
    ).hexdigest()
    choices = {
        "random": sorted(candidates, key=lambda row: (random_rank(row[0]), row[0]))[:count],
        "top_tv": sorted(candidates, key=lambda row: (-row[2], row[0]))[:count],
        "top_m": sorted(candidates, key=lambda row: (-row[1], row[0]))[:count],
    }
    return {arm: sorted(row[0] for row in selected) for arm, selected in choices.items()}
