"""Summarize fixed-LR first-rejection CE ablations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from analyze_frozen_actor_lr import compare, k3_audit, read_records


def analyze(root, audit_k3=False):
    records = read_records(root / "results.jsonl")
    groups, publish_cost = defaultdict(list), defaultdict(list)
    train_cost, training_metrics = defaultdict(float), {}
    pattern = re.compile(r"ce_([\d.]+)_step_(\d+)_(quality|load320)_")
    for row in records:
        if row["event"] == "trained":
            key = (float(row["first_rejection_ce_alpha"]), int(row["steps"]))
            train_cost[key] = row["rpc_seconds"]
            metrics = [worker["metrics"] for worker in row["workers"]]
            training_metrics[key] = {
                name: float(np.mean([metric[name] for metric in metrics if name in metric]))
                for name in {name for metric in metrics for name in metric}
            }
        elif row["event"] in {"probe", "publish"}:
            match = pattern.match(row["label"])
            if match:
                key = (float(match[1]), int(match[2]), match[3])
                (groups if row["event"] == "probe" else publish_cost)[key].append(
                    row if row["event"] == "probe" else row["seconds"])

    summary = dict(completed=any(row["event"] == "completed" for row in records), arms=[])
    aa = [row for row in records if row["event"] == "probe" and row["label"].startswith("baseline_AA")]
    if len(aa) == 2:
        summary["baseline_AA"] = dict(
            acceptance=[row["acceptance"] for row in aa],
            certainty=[row["low_entropy_share"] for row in aa],
            mismatch=[row["mismatch"] for row in aa],
            ms_per_token=[row["ms_per_token"] for row in aa])
    prepared = next((row for row in records if row["event"] == "prepared"), None)
    if prepared:
        summary["fixed_batch_coverage"] = [
            dict(rank=worker["rank"], optimizer_inputs=len(worker["batch_hashes"]),
                 distinct_batch_hashes=len(set(worker["batch_hashes"])))
            for worker in prepared["workers"]]
    for (ce_alpha, steps, mode), probes in sorted(groups.items()):
        cost = train_cost[(ce_alpha, 5)] + (train_cost[(ce_alpha, 20)] if steps == 20 else 0)
        publish = float(np.mean(publish_cost[(ce_alpha, steps, mode)]))
        summary["arms"].append(dict(
            first_rejection_ce_alpha=ce_alpha, lr=1e-5, steps=steps, mode=mode,
            training_metrics=training_metrics[(ce_alpha, steps)],
            **compare(probes, cost, publish)))

    workers = {path.name: read_records(path) for path in root.glob("worker_*.jsonl")}
    summary["restoration_checks"] = [
        dict(worker=name, ok=row["ok"])
        for name, rows in workers.items() for row in rows if row["event"] == "restored"]
    if audit_k3:
        audit = k3_audit(root)
        for row in audit["samples"]:
            row["first_rejection_ce_alpha"] = (0.0, .05, .1)[min((row["sequence"]-1)//20, 2)]
            row.pop("lr", None)
        (root / "k3_audit.json").write_text(json.dumps(audit, indent=2))
        summary["k3"] = {key: value for key, value in audit.items() if key != "samples"}
        totals = defaultdict(lambda: defaultdict(int))
        for row in audit["samples"]:
            key = (row["first_rejection_ce_alpha"], row["optimizer_step"], row["stream"], row["kind"])
            for field in ("n", "blocked_by_clipping", "hard_loss_clamp", "negative_tail", "positive_tail", "nonfinite"):
                totals[key][field] += row[field]
        summary["sampled_k3_coverage"] = [
            dict(first_rejection_ce_alpha=alpha, optimizer_step=step, stream=stream,
                 kind=kind, **values,
                 unclipped_fraction=1-values["blocked_by_clipping"]/values["n"])
            for (alpha, step, stream, kind), values in sorted(totals.items())]

    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    report = ["# First-rejection CE results", "", f"Completed: {summary['completed']}", "",
              "| CE alpha | Steps | Mode | Delta acceptance | Paired 95% interval | Delta low-H mismatch | Throughput delta |",
              "|---:|---:|---|---:|---|---:|---:|"]
    for arm in summary["arms"]:
        if arm["complete"]:
            low, high = arm["paired_bootstrap95"]
            report.append(
                f"| {arm['first_rejection_ce_alpha']:.2f} | {arm['steps']} | {arm['mode']} | "
                f"{arm['acceptance_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                f"{arm['low_entropy_mismatch_delta']:+.4f} | {arm['throughput_relative_change']:+.2%} |")
    report += ["", "Single-seed fixed-data screening; repeated load prompts do not increase independent sample size."]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))

    if summary["completed"]:
        if len(workers) != 16 or len(summary["restoration_checks"]) != 48 or not all(row["ok"] for row in summary["restoration_checks"]):
            raise RuntimeError("Missing or failed restoration checks")
        if len(summary["arms"]) != 9 or not all(arm["complete"] for arm in summary["arms"]):
            raise RuntimeError("Incomplete CE evaluation matrix")
        if audit_k3 and (audit["files"] != 960 or audit["differences"] or
                         audit["first_forward_differences"] or audit["incomplete_sequences"] or
                         any(row["nonfinite"] for row in audit["samples"])):
            raise RuntimeError("K3 integrity audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--k3", action="store_true")
    args = parser.parse_args()
    analyze(args.root, args.k3)
