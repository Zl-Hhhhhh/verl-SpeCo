#!/usr/bin/env python3
"""Compare lifecycle evidence without treating different FSDP shards as replicas."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def changed_state(before, after):
    changes = {}
    for category in ("parameters", "buffers", "runtime"):
        left, right = before.get(category, {}), after.get(category, {})
        changed = []
        for name in sorted(left.keys() | right.keys()):
            a, b = left.get(name, {}), right.get(name, {})
            if any(a.get(k) != b.get(k) for k in ("sha256", "dtype", "local_shape")):
                changed.append(name)
        if changed:
            changes[category] = changed
    return changes


def compare_streams(left, right):
    result = {}
    for stream in ("accepted", "rejected"):
        def rows(payload):
            values = payload.get(stream, {})
            keys = ("sample_indices", "anchor_positions", "draft_positions", "token_ids", "prev_token_ids")
            if not all(k in values for k in (*keys, "draft_logprobs", "target_logprobs")):
                return {}
            columns = [values[k].reshape(-1).tolist() for k in keys]
            q = values["draft_logprobs"].reshape(-1).tolist()
            p = values["target_logprobs"].reshape(-1).tolist()
            # Keep duplicate identities separate rather than silently overwriting.
            seen = defaultdict(int)
            mapped = {}
            for identity, logq, logp in zip(zip(*columns), q, p):
                ordinal = seen[identity]
                seen[identity] += 1
                mapped[(*identity, ordinal)] = (logq, logp)
            return mapped
        a, b = rows(left), rows(right)
        common = a.keys() & b.keys()
        comparable = [k for k in common if abs(a[k][1] - b[k][1]) <= 1e-6]
        gaps = [(abs(a[k][0] - b[k][0]), k) for k in comparable]
        gaps.sort(reverse=True)
        result[stream] = {"matched_teacher_rows": len(comparable),
                          "unmatched_identity_rows": len(a.keys() ^ b.keys()),
                          "max_student_logprob_gap": gaps[0][0] if gaps else None,
                          "rows_gap_gt_1e_3": sum(gap > 1e-3 for gap, _ in gaps),
                          "largest_gaps": [{"gap": gap, "identity": key} for gap, key in gaps[:5]]}
    return result


def analyze(root, compare_logprobs=False):
    transitions, forwards = [], defaultdict(list)
    event_count = 0
    for path in sorted(root.glob("rank_*/events.jsonl")):
        pending = defaultdict(list)
        inputs = {}
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:  # A live writer may be mid-line.
                continue
            event_count += 1
            phase = event["phase"]
            if phase == "forward.before":
                inputs[event["forward_id"]] = event["inputs"]
            if phase == "forward.after":
                event["inputs"] = inputs.get(event["forward_id"], {})
                key = (event["rl_step"], event["optimizer_steps_total"], event["rollout_dp_rank"], event["forward_id"])
                forwards[key].append(event)
            if phase.endswith(".before") and phase != "forward.before":
                pending[phase[:-7]].append(event)
            elif phase.endswith(".after") and phase != "forward.after":
                stack = pending[phase[:-6]]
                if stack:
                    before = stack.pop()
                    changes = changed_state(before["state"], event["state"])
                    if changes:
                        transitions.append({"rank": event["global_rank"], "rl_step": event["rl_step"],
                                            "phase": phase[:-6], "changes": changes,
                                            "before_event": before["event_sequence"], "after_event": event["event_sequence"],
                                            "events_file": str(path)})
    comparisons = []
    for key, events in sorted(forwards.items()):
        events.sort(key=lambda e: e["sp_rank"])
        first = events[0]
        for other in events[1:]:
            def signature(e):
                return {k: (v["sha256"], v["dtype"], v["local_shape"]) for k, v in e["inputs"].items()}
            same_inputs = bool(first["inputs"]) and signature(first) == signature(other)
            # Buffers/head are replicated; model parameters can be different shards.
            replicated = {}
            for category in ("buffers", "runtime"):
                a = {k: v for k, v in first["state"][category].items() if "placements" not in v}
                b = {k: v for k, v in other["state"][category].items() if "placements" not in v}
                replicated.update(changed_state({category: a}, {category: b}))
            item = {"rl_step": key[0], "optimizer_steps_total": key[1], "dp": key[2],
                    "forward_id": key[3], "ranks": [first["global_rank"], other["global_rank"]],
                    "same_input_tensors": same_inputs, "replicated_buffer_differences": replicated}
            if compare_logprobs:
                import torch
                a = torch.load(first["output_path"], map_location="cpu", weights_only=True)
                b = torch.load(other["output_path"], map_location="cpu", weights_only=True)
                item["logprobs"] = compare_streams(a["streams"], b["streams"])
            comparisons.append(item)
    return {"event_count": event_count, "lifecycle_state_changes": transitions,
            "rank_comparisons": comparisons,
            "note": "Changes are evidence, not causal conclusions. Cache growth and head synchronization may be expected. Local parameter shards are not compared across ranks."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--compare-logprobs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.compare_logprobs), indent=2))
