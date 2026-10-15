"""Summarize complete or partial frozen-actor LR experiments, with prompt clusters."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def compare(probes, train_seconds, publish_seconds):
    a = [p for p in probes if p["label"].endswith("_A")]
    b = [p for p in probes if p["label"].endswith("_B")]
    if len(a) != 2 or len(b) != 2:
        return {"complete": False, "measurements": len(probes)}
    diff = np.mean([p["per_prompt"] for p in b], axis=0) - np.mean([p["per_prompt"] for p in a], axis=0)
    # Aggregate repeated responses by original prompt BEFORE confidence intervals.
    unique = a[0]["unique_prompts"]
    clustered = np.array([diff[i::unique].mean() for i in range(unique)])
    rng = np.random.default_rng(17)
    boot = rng.choice(clustered, size=(10000, unique), replace=True).mean(axis=1)
    a_latency = sum(p["elapsed_sec"] for p in a) / sum(p["response_tokens"] for p in a)
    b_latency = sum(p["elapsed_sec"] for p in b) / sum(p["response_tokens"] for p in b)
    saving = a_latency - b_latency
    quality_cost = sum(p["elapsed_sec"] for p in probes) + 4 * publish_seconds
    return dict(complete=True, unique_prompts=unique, paired_mean=float(clustered.mean()),
                paired_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                acceptance_delta=float(np.mean([p["acceptance"] for p in b]) - np.mean([p["acceptance"] for p in a])),
                low_entropy_mismatch_delta=float(np.mean([p["mismatch"] for p in b]) - np.mean([p["mismatch"] for p in a])),
                baseline_tokens_per_second=1/a_latency, candidate_tokens_per_second=1/b_latency,
                throughput_relative_change=a_latency/b_latency-1,
                seconds_saved_per_token=saving,
                break_even_tokens_train_and_one_publish=(train_seconds+publish_seconds)/saving if saving > 0 else None,
                break_even_tokens_including_this_validation=(train_seconds+publish_seconds+quality_cost)/saving if saving > 0 else None,
                baseline_repeat_latency_spread=max(p["ms_per_token"] for p in a)/min(p["ms_per_token"] for p in a)-1,
                candidate_repeat_latency_spread=max(p["ms_per_token"] for p in b)/min(p["ms_per_token"] for p in b)-1,
                greedy_response_hash_agreement=sum(x == y for x, y in zip(a[0]["response_hashes"], b[0]["response_hashes"], strict=True))/len(diff))


def k3_audit(root, arm_metadata=None, steps_per_arm=20):
    import torch
    torch.set_num_threads(2)
    grouped = defaultdict(dict)
    for file in sorted((root / "k3_logprobs").rglob("*.pt")):
        payload = torch.load(file, map_location="cpu", weights_only=True)
        grouped[payload["dump_sequence"]][payload["global_rank"]] = payload
    comparisons = 0
    differences, samples = [], []
    first_forward_comparisons, first_forward_differences = 0, []

    def same_streams(left, right):
        return left.keys() == right.keys() and all(
            left[s].keys() == right[s].keys() and all(
                torch.equal(value, right[s][key]) if torch.is_tensor(value) else value == right[s][key]
                for key, value in left[s].items()) for s in left)

    first_sequences = (
        [1 + index * steps_per_arm for index in range(1, len(arm_metadata))]
        if arm_metadata is not None else [21, 41]
    )
    for sequence in first_sequences:
        for rank in set(grouped.get(1, {})).intersection(grouped.get(sequence, {})):
            first_forward_comparisons += 1
            if not same_streams(grouped[1][rank]["streams"], grouped[sequence][rank]["streams"]):
                first_forward_differences.append(dict(sequence=sequence, rank=rank))
    for sequence, ranks in sorted(grouped.items()):
        for anchor in range(0, 16, 4):
            if anchor not in ranks:
                continue
            left = ranks[anchor]["streams"]
            for other in range(anchor+1, anchor+4):
                if other not in ranks:
                    continue
                right = ranks[other]["streams"]
                same = same_streams(left, right)
                comparisons += 1
                if not same:
                    differences.append(dict(sequence=sequence, ranks=[anchor, other]))
            for stream, values in left.items():
                if not values:
                    continue
                first = values["draft_positions"] == values["trace_accepted_counts"]
                masks = {"all": torch.ones_like(first)} if stream == "accepted" else {"first": first, "suffix": ~first}
                for kind, mask in masks.items():
                    if not mask.any():
                        continue
                    ratio = (values["target_logprobs"][mask].float() - values["draft_logprobs"][mask].float())
                    clipped = ratio.clamp(-20, 20)
                    raw = clipped.exp()-clipped-1
                    blocked = (ratio <= -20) | (ratio >= 20) | (raw >= 10)
                    if arm_metadata is None:
                        metadata = {
                            "lr": (5e-6, 1e-5, 1.5e-5)[min((sequence-1)//20, 2)]
                        }
                    else:
                        arm_index = min((sequence - 1) // steps_per_arm, len(arm_metadata) - 1)
                        metadata = arm_metadata[arm_index]
                    samples.append(dict(
                        sequence=sequence,
                        optimizer_step=ranks[anchor]["optimizer_step"],
                        rank=anchor,
                        stream=stream,
                        kind=kind,
                        n=int(mask.sum()),
                        blocked_by_clipping=int(blocked.sum()),
                        hard_loss_clamp=int((raw >= 10).sum()),
                        negative_tail=int((ratio < -20).sum()),
                        positive_tail=int((ratio > 20).sum()),
                        nonfinite=int((~torch.isfinite(ratio)).sum()),
                        **metadata,
                    ))
    return dict(files=sum(len(v) for v in grouped.values()), pair_checks=comparisons,
                differences=differences, samples=samples,
                first_forward_comparisons=first_forward_comparisons,
                first_forward_differences=first_forward_differences,
                incomplete_sequences=[s for s, ranks in grouped.items() if len(ranks) != 16])


def analyze(root, audit_k3=False):
    records = read_records(root / "results.jsonl")
    groups = defaultdict(list)
    train_cost = defaultdict(float)
    training_metrics = {}
    publish_cost = defaultdict(list)
    for row in records:
        if row["event"] == "trained":
            train_cost[(row["lr"], row["steps"])] = row["rpc_seconds"]
            metrics = [worker["metrics"] for worker in row["workers"]]
            training_metrics[(row["lr"], row["steps"])] = {
                key: float(np.mean([m[key] for m in metrics if key in m]))
                for key in {k for m in metrics for k in m}}
        if row["event"] in {"probe", "publish"}:
            match = re.match(r"lr_([\de.+-]+)_step_(\d+)_(quality|load320)_", row["label"])
            if match:
                key = (float(match[1]), int(match[2]), match[3])
                (groups if row["event"] == "probe" else publish_cost)[key].append(row if row["event"] == "probe" else row["seconds"])
    summary = dict(completed=any(r["event"] == "completed" for r in records), arms=[])
    prepared = next((r for r in records if r["event"] == "prepared"), None)
    if prepared:
        summary["fixed_batch_coverage"] = [
            dict(rank=w["rank"], optimizer_inputs=len(w["batch_hashes"]),
                 distinct_batch_hashes=len(set(w["batch_hashes"])))
            for w in prepared["workers"]]
    aa = [r for r in records if r["event"] == "probe" and r["label"].startswith("baseline_AA")]
    if len(aa) == 2:
        summary["baseline_AA"] = dict(
            acceptance=[r["acceptance"] for r in aa], certainty=[r["low_entropy_share"] for r in aa],
            mismatch=[r["mismatch"] for r in aa], ms_per_token=[r["ms_per_token"] for r in aa],
            hash_agreement=sum(x == y for x, y in zip(aa[0]["response_hashes"], aa[1]["response_hashes"], strict=True))/len(aa[0]["response_hashes"]))
    for (lr, steps, mode), probes in sorted(groups.items()):
        cost = train_cost[(lr, 5)] + (train_cost[(lr, 20)] if steps == 20 else 0)
        publish = float(np.mean(publish_cost[(lr, steps, mode)]))
        summary["arms"].append(dict(lr=lr, steps=steps, mode=mode,
                                    training_metrics=training_metrics[(lr, steps)],
                                    **compare(probes, cost, publish)))
    workers = {path.name: read_records(path) for path in root.glob("worker_*.jsonl")}
    summary["restoration_checks"] = [dict(worker=name, ok=r["ok"]) for name, rows in workers.items() for r in rows if r["event"] == "restored"]
    if audit_k3:
        audit = k3_audit(root)
        (root / "k3_audit.json").write_text(json.dumps(audit, indent=2))
        summary["k3"] = {k: v for k, v in audit.items() if k != "samples"}
        sample_totals = defaultdict(lambda: defaultdict(int))
        for row in audit["samples"]:
            key = (row["lr"], row["optimizer_step"], row["stream"], row["kind"])
            for field in ("n", "blocked_by_clipping", "hard_loss_clamp", "negative_tail", "positive_tail", "nonfinite"):
                sample_totals[key][field] += row[field]
        summary["sampled_k3_coverage"] = [
            dict(lr=lr, optimizer_step=step, stream=stream, kind=kind, **values,
                 unclipped_fraction=1-values["blocked_by_clipping"]/values["n"],
                 hard_loss_clamp_fraction=values["hard_loss_clamp"]/values["n"])
            for (lr, step, stream, kind), values in sorted(sample_totals.items())]
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = ["# Frozen actor LR results", "", f"Completed: {summary['completed']}", "",
            "Intervals cluster by independent prompt; repeated load-test responses do not increase sample size.", "",
            "| LR | Steps | Mode | Δ acceptance | Paired 95% interval | Δ low-H mismatch | Throughput Δ |",
            "|---|---:|---|---:|---|---:|---:|"]
    for arm in summary["arms"]:
        if arm["complete"]:
            lo, hi = arm["paired_bootstrap95"]
            rows.append(f"| {arm['lr']:.1e} | {arm['steps']} | {arm['mode']} | {arm['acceptance_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {arm['low_entropy_mismatch_delta']:+.4f} | {arm['throughput_relative_change']:+.2%} |")
    rows += ["", "Break-even token counts, A/A noise, response consistency, fixed-batch reuse, restoration and K3 audits are in summary.json.",
             "Optimizer steps can reuse identical cached batches; they are not independent data samples or online actor steps.",
             "These are single-seed screening results, with greedy 512/2048-token probes; they do not establish speedup for full stochastic training."]
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["completed"]:
        if len(workers) != 16 or len(summary["restoration_checks"]) != 48 or not all(r["ok"] for r in summary["restoration_checks"]):
            raise RuntimeError("Missing or failed rank restoration checks")
        if len(summary["arms"]) != 9 or not all(arm["complete"] for arm in summary["arms"]):
            raise RuntimeError("Incomplete LR evaluation matrix")
        if audit_k3 and (audit["files"] != 960 or audit["differences"] or
                         audit["first_forward_differences"] or audit["incomplete_sequences"] or
                         any(row["nonfinite"] for row in audit["samples"])):
            raise RuntimeError("K3 integrity audit failed; inspect k3_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--k3", action="store_true")
    args = parser.parse_args()
    analyze(args.root, args.k3)
