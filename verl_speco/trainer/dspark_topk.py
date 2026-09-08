"""Top-k support reverse KL, with full-vocabulary coverage diagnostics."""

import torch
import torch.nn.functional as F


def topk_candidate_width(topk, support="student"):
    if support not in {"student", "union"}:
        raise ValueError(f"DSpark topk_support must be student or union, got {support!r}")
    if int(topk) < 2:
        raise ValueError("DSpark topk must be at least 2")
    return int(topk) * (2 if support == "union" else 1)


def validate_topk_modes(response_loss, rejected_loss, topk, trace_aligned, support="student"):
    for name, mode in (("response", response_loss), ("rejected", rejected_loss)):
        if mode not in {"k3", "topk"}:
            raise ValueError(f"DSpark {name}_loss must be k3 or topk, got {mode!r}")
    topk_candidate_width(topk, support)
    if rejected_loss == "topk" and not trace_aligned:
        raise ValueError("DSpark rejected topk requires trace_aligned_anchors")


def student_support_reverse_kl(
    draft_logprobs, teacher_logprobs, weights, *, topk, candidate_ids=None, support="student",
):
    """Normalize BOTH distributions on the selected support; teacher is detached.

    With explicit candidate_ids, teacher_logprobs contains only those candidates
    (the verified rollout branch). Otherwise it is a full vocabulary distribution
    and support is selected from the current training student and optionally
    teacher. Union buffers concatenate two top-k lists; overlaps count only once.
    """
    width = topk_candidate_width(topk, support)
    if topk > draft_logprobs.size(-1):
        raise ValueError("DSpark topk exceeds vocabulary size")
    if candidate_ids is None:
        candidate_ids = draft_logprobs.detach().topk(topk, dim=-1).indices
        if support == "union":
            candidate_ids = torch.cat([
                candidate_ids, teacher_logprobs.detach().topk(topk, dim=-1).indices,
            ], dim=-1)
        teacher_selected = teacher_logprobs.gather(-1, candidate_ids).detach().float()
    else:
        if candidate_ids.shape != (*draft_logprobs.shape[:-1], width):
            raise ValueError("DSpark rejected topk candidate shape mismatch")
        if teacher_logprobs.shape != candidate_ids.shape:
            raise ValueError("DSpark rejected topk teacher shape mismatch")
        if (candidate_ids < 0).any() or (candidate_ids >= draft_logprobs.size(-1)).any():
            raise ValueError("DSpark rejected topk has missing/out-of-vocabulary candidates")
        candidate_lists = candidate_ids.reshape(*candidate_ids.shape[:-1], width // topk, topk)
        if (candidate_lists.sort(dim=-1).values.diff(dim=-1) == 0).any():
            raise ValueError("DSpark rejected topk candidates must be unique")
        teacher_selected = teacher_logprobs.detach().float()
    draft_selected = draft_logprobs.gather(-1, candidate_ids).float()
    if not torch.isfinite(teacher_selected).all() or not torch.isfinite(draft_selected).all():
        raise ValueError("DSpark topk requires finite candidate log probabilities")
    if support == "union":
        sorted_ids, order = candidate_ids.sort(dim=-1)
        draft_selected = draft_selected.gather(-1, order)
        teacher_selected = teacher_selected.gather(-1, order)
        unique = torch.cat([
            torch.ones_like(sorted_ids[..., :1], dtype=torch.bool),
            sorted_ids[..., 1:] != sorted_ids[..., :-1],
        ], dim=-1)
        draft_selected = draft_selected.masked_fill(~unique, -torch.inf)
        teacher_selected = teacher_selected.masked_fill(~unique, -torch.inf)
        log_q = F.log_softmax(draft_selected, dim=-1)
        log_p = F.log_softmax(teacher_selected, dim=-1)
        # Avoid undefined -inf - -inf on duplicate slots, including in backward.
        log_ratio = log_q.masked_fill(~unique, 0) - log_p.masked_fill(~unique, 0)
        per_token = (log_q.exp() * log_ratio).sum(dim=-1)
        support_size = unique.sum(dim=-1).float()
    else:
        log_q = F.log_softmax(draft_selected, dim=-1)
        log_p = F.log_softmax(teacher_selected, dim=-1)
        per_token = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        support_size = torch.full_like(per_token, topk)
    weights = weights.float()
    return {
        "loss_sum": (per_token * weights).sum(),
        "weighted_count": weights.sum(),
        "student_mass_sum": (draft_selected.detach().exp().sum(-1) * weights).sum(),
        "teacher_mass_sum": (teacher_selected.exp().sum(-1) * weights).sum(),
        "support_size_sum": (support_size * weights).sum(),
    }


def pack_trace_topk(rows, *, block_size, topk, device, support="student"):
    """Pad optional trace candidates; absent support remains invalid, never invented."""
    width = topk_candidate_width(topk, support)
    ids = torch.full((len(rows), block_size, width), -1, dtype=torch.long, device=device)
    logprobs = torch.zeros_like(ids, dtype=torch.float32)
    for index, row in enumerate(rows):
        raw_ids = row.get("draft_topk_ids")
        raw_logprobs = row.get("draft_topk_teacher_logprobs")
        if raw_ids is None and raw_logprobs is None:
            continue
        candidate_ids = torch.as_tensor(raw_ids, dtype=torch.long, device=device)[:block_size]
        teacher = torch.as_tensor(raw_logprobs, dtype=torch.float32, device=device)[:block_size]
        expected = (int(row["draft_count"]), width)
        if tuple(candidate_ids.shape) != expected or tuple(teacher.shape) != expected:
            raise ValueError(f"DSpark trace topk expected {expected} candidates per anchor")
        ids[index, :expected[0]] = candidate_ids
        logprobs[index, :expected[0]] = teacher
    return {"trace_draft_topk_ids": ids, "trace_draft_topk_teacher_logprobs": logprobs}
