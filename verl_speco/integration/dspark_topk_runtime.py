"""Optional DSpark top-k support capture for rejected-branch KL.

Only candidate IDs are retained from the proposer. The unchanged actor verify
pass scores them on the *actual proposed branch*, before sampling mutates logits.
Union mode appends teacher top-k IDs from that same verification pass.
"""

from functools import wraps

import torch


def rejected_topk_size(owner):
    config = getattr(getattr(owner, "vllm_config", None), "additional_config", None) or {}
    return int(config.get("speco_dspark_rejected_topk", 0))


def capture_topk_proposal(original, proposer, *args, **kwargs):
    topk = rejected_topk_size(proposer)
    if not topk:
        return original(proposer, *args, **kwargs)
    if getattr(proposer, "method", None) != "dspark":
        raise ValueError("Rejected topk capture is supported only for DSpark")
    model = proposer.model
    compute_logits = model.compute_logits
    had_override = "compute_logits" in vars(model)
    captured = []

    def capture_logits(*logit_args, **logit_kwargs):
        logits = compute_logits(*logit_args, **logit_kwargs)
        captured.append(logits)
        return logits

    model.compute_logits = capture_logits
    try:
        proposals = original(proposer, *args, **kwargs)
    finally:
        if had_override:
            model.compute_logits = compute_logits
        else:
            delattr(model, "compute_logits")
    if len(captured) != 1 or not torch.is_tensor(proposals) or proposals.ndim != 2:
        raise RuntimeError("Unsupported DSpark proposal path for rejected topk capture")
    # The native proposer adds the recursive Markov biases to these logits in place.
    logits = captured[0].view(*proposals.shape, -1)
    if topk < 2 or topk > logits.size(-1):
        raise ValueError("Rejected topk must be in [2, vocabulary size]")
    if not torch.equal(logits.argmax(dim=-1), proposals):
        raise RuntimeError("Captured DSpark logits do not match post-Markov proposals")
    proposer._speco_last_topk_proposal = (
        proposals.detach().clone(), logits.detach().topk(topk, dim=-1).indices
    )
    return proposals


def cache_topk_proposal(original, runner, *args, **kwargs):
    topk = rejected_topk_size(runner)
    if not topk:
        return original(runner, *args, **kwargs)
    spec_config = getattr(runner.vllm_config, "speculative_config", None)
    if getattr(spec_config, "disable_padded_drafter_batch", False):
        raise ValueError("DSpark rejected topk currently requires the padded drafter batch path")
    request_ids = list(runner.input_batch.req_ids)
    runner.drafter._speco_last_topk_proposal = None
    result = original(runner, *args, **kwargs)
    payload = runner.drafter._speco_last_topk_proposal
    if payload is None:
        raise RuntimeError("DSpark proposer did not capture rejected topk candidates")
    proposals, candidates = payload
    if proposals.size(0) != len(request_ids) or not torch.equal(proposals, result):
        raise RuntimeError("DSpark candidate cache does not match the scheduled proposals")
    cache = getattr(runner, "_speco_topk_proposals_by_request", {})
    live_requests = getattr(runner, "requests", None)
    if live_requests is not None:
        cache = {key: value for key, value in cache.items() if key in live_requests}
    for index, request_id in enumerate(request_ids):
        cache[request_id] = (proposals[index], candidates[index])
    runner._speco_topk_proposals_by_request = cache
    return result


def verified_topk_payload(runner, logits, metadata):
    if not rejected_topk_size(runner) or logits is None or metadata is None:
        return None
    request_ids = list(runner.input_batch.req_ids)
    counts = [int(count) for count in metadata.num_draft_tokens]
    if len(counts) != len(request_ids) or any(count < 0 for count in counts):
        raise ValueError("DSpark verify request counts do not align")
    if not sum(counts):
        return None
    cache = getattr(runner, "_speco_topk_proposals_by_request", {})
    expected_parts, candidate_parts = [], []
    for request_id, count in zip(request_ids, counts):
        if not count:
            continue
        payload = cache.get(request_id)
        if payload is None or payload[0].numel() < count:
            raise RuntimeError(f"Missing DSpark topk proposal for verified request {request_id}")
        expected_parts.append(payload[0][:count])
        candidate_parts.append(payload[1][:count])
    expected = torch.cat(expected_parts)
    actual = metadata.draft_token_ids.reshape(-1)
    if not torch.equal(expected, actual):
        raise RuntimeError("DSpark topk cache is stale or verify branch IDs do not match")
    candidates = torch.cat(candidate_parts)
    teacher_logits = logits[metadata.target_logits_indices].float()
    if teacher_logits.size(0) != candidates.size(0):
        raise ValueError("DSpark topk teacher logits do not match verified candidates")
    config = runner.vllm_config.additional_config or {}
    support = config.get("speco_dspark_topk_support", "student")
    if support not in {"student", "union"}:
        raise ValueError("DSpark topk_support must be student or union")
    if support == "union":
        # Retain a fixed 2*k trace width; training masks overlaps before softmax.
        candidates = torch.cat([
            candidates, teacher_logits.topk(rejected_topk_size(runner), dim=-1).indices,
        ], dim=-1)
    teacher_logprobs = teacher_logits.gather(-1, candidates) - torch.logsumexp(
        teacher_logits, dim=-1, keepdim=True
    )
    if not torch.isfinite(teacher_logprobs).all():
        raise ValueError("DSpark topk teacher log probabilities are non-finite")
    return request_ids, counts, candidates.detach(), teacher_logprobs.detach()


def topk_payload_by_request(payload):
    if payload is None:
        return {}
    request_ids, counts, ids, logprobs = payload
    ids, logprobs = ids.cpu().tolist(), logprobs.float().cpu().tolist()
    result, offset = {}, 0
    for request_id, count in zip(request_ids, counts):
        if count:
            result[str(request_id)] = {
                "draft_topk_ids": ids[offset:offset + count],
                "draft_topk_teacher_logprobs": logprobs[offset:offset + count],
            }
        offset += count
    return result


def patch_dspark_topk_capture(runner_class):
    try:
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    except ImportError:
        # Older engines can keep the K3 path. Enabling topk without capture
        # will fail the first verified metadata check, never train on false IDs.
        return False

    if not callable(getattr(runner_class, "propose_draft_token_ids", None)):
        return False

    original_propose = AscendDSparkProposer._propose
    if not getattr(original_propose, "_speco_topk_capture", False):
        @wraps(original_propose)
        def propose(self, *args, **kwargs):
            return capture_topk_proposal(original_propose, self, *args, **kwargs)

        propose._speco_topk_capture = True
        AscendDSparkProposer._propose = propose
    original_runner = runner_class.propose_draft_token_ids
    if not getattr(original_runner, "_speco_topk_capture", False):
        @wraps(original_runner)
        def propose_for_runner(self, *args, **kwargs):
            return cache_topk_proposal(original_runner, self, *args, **kwargs)

        propose_for_runner._speco_topk_capture = True
        runner_class.propose_draft_token_ids = propose_for_runner
    return True
