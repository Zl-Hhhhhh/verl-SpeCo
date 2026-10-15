"""Summarize CE-by-LR frozen-actor update-depth experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from analyze_frozen_actor_lr import compare, k3_audit, read_records


LABEL = re.compile(r"(?P<arm>ce_[\d.]+_lr_[\de.+-]+)_step_(?P<steps>\d+)_(?P<mode>quality|load320)_")


def analyze(root, audit_k3=False):
    records = read_records(root / "results.jsonl")
    design = next((row for row in records if row["event"] == "design"), {})
    checkpoints = [int(value) for value in design.get("checkpoints", [])]
    performance_checkpoints = {
        int(value) for value in design.get("performance_checkpoints", [])
    }
    groups, publish_cost = defaultdict(list), defaultdict(list)
    training_metrics, incremental_train_cost = {}, defaultdict(float)
    arm_metadata = {}
    arm_order = []
    for row in records:
        if row["event"] == "trained":
            arm = row["arm"]
            if arm not in arm_metadata:
                arm_order.append(arm)
            arm_metadata[arm] = {
                "arm": arm,
                "lr": float(row["lr"]),
                "first_rejection_ce_alpha": float(row["first_rejection_ce_alpha"]),
            }
            key = (arm, int(row["steps"]))
            incremental_train_cost[key] = float(row["rpc_seconds"])
            metrics = [worker["metrics"] for worker in row["workers"]]
            training_metrics[key] = {
                name: float(np.mean([metric[name] for metric in metrics if name in metric]))
                for name in {name for metric in metrics for name in metric}
            }
        elif row["event"] in {"probe", "publish"}:
            match = LABEL.match(row["label"])
            if match:
                key = (match["arm"], int(match["steps"]), match["mode"])
                target = groups if row["event"] == "probe" else publish_cost
                target[key].append(row if row["event"] == "probe" else row["seconds"])

    summary = {
        "completed": any(row["event"] == "completed" for row in records),
        "design": design,
        "arms": [],
    }
    aa = [
        row for row in records
        if row["event"] == "probe" and row["label"].startswith("baseline_AA")
    ]
    if len(aa) == 2:
        summary["baseline_AA"] = {
            "acceptance": [row["acceptance"] for row in aa],
            "certainty": [row["low_entropy_share"] for row in aa],
            "mismatch": [row["mismatch"] for row in aa],
            "ms_per_token": [row["ms_per_token"] for row in aa],
            "hash_agreement": sum(
                left == right
                for left, right in zip(
                    aa[0]["response_hashes"], aa[1]["response_hashes"], strict=True
                )
            ) / len(aa[0]["response_hashes"]),
        }
    prepared = next((row for row in records if row["event"] == "prepared"), None)
    if prepared:
        summary["fixed_batch_coverage"] = [
            {
                "rank": worker["rank"],
                "optimizer_inputs": len(worker["batch_hashes"]),
                "distinct_batch_hashes": len(set(worker["batch_hashes"])),
            }
            for worker in prepared["workers"]
        ]
    for (arm, steps, mode), probes in sorted(groups.items()):
        metadata = arm_metadata[arm]
        train_seconds = sum(
            seconds
            for (cost_arm, checkpoint), seconds in incremental_train_cost.items()
            if cost_arm == arm and checkpoint <= steps
        )
        publishing = publish_cost[(arm, steps, mode)]
        summary["arms"].append({
            **metadata,
            "steps": steps,
            "mode": mode,
            "training_metrics": training_metrics[(arm, steps)],
            **compare(probes, train_seconds, float(np.mean(publishing))),
        })

    workers = {
        path.name: read_records(path) for path in root.glob("worker_*.jsonl")
    }
    summary["restoration_checks"] = [
        {"worker": name, "ok": row["ok"]}
        for name, rows in workers.items()
        for row in rows
        if row["event"] == "restored"
    ]
    if audit_k3 and arm_order:
        metadata = [arm_metadata[arm] for arm in arm_order]
        audit = k3_audit(root, arm_metadata=metadata, steps_per_arm=max(checkpoints))
        (root / "k3_audit.json").write_text(json.dumps(audit, indent=2))
        summary["k3"] = {key: value for key, value in audit.items() if key != "samples"}
        totals = defaultdict(lambda: defaultdict(int))
        for row in audit["samples"]:
            key = (
                row["arm"], row["optimizer_step"], row["stream"], row["kind"]
            )
            for field in (
                "n", "blocked_by_clipping", "hard_loss_clamp", "negative_tail",
                "positive_tail", "nonfinite",
            ):
                totals[key][field] += row[field]
        summary["sampled_k3_coverage"] = [
            {
                "arm": arm,
                "optimizer_step": step,
                "stream": stream,
                "kind": kind,
                **values,
                "unclipped_fraction": 1 - values["blocked_by_clipping"] / values["n"],
            }
            for (arm, step, stream, kind), values in sorted(totals.items())
        ]

    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    report = [
        "# Frozen actor CE x LR update-depth results",
        "",
        f"Completed: {summary['completed']}",
        "",
        "| CE | LR | Steps | Mode | Delta acceptance | Paired 95% interval | Delta low-H mismatch | Throughput delta |",
        "|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for arm in summary["arms"]:
        if arm["complete"]:
            low, high = arm["paired_bootstrap95"]
            report.append(
                f"| {arm['first_rejection_ce_alpha']:.2f} | {arm['lr']:.1e} | "
                f"{arm['steps']} | {arm['mode']} | {arm['acceptance_delta']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | "
                f"{arm['low_entropy_mismatch_delta']:+.4f} | "
                f"{arm['throughput_relative_change']:+.2%} |"
            )
    report += [
        "",
        "Optimizer steps replay fixed cached batches; they are not actor-training steps.",
        "Load-test repeats are clustered by independent prompt and do not inflate sample size.",
    ]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))

    if summary["completed"]:
        expected_arms = len(design["lrs"]) * len(design["first_rejection_ce_alphas"])
        expected_rows = expected_arms * (
            len(checkpoints) + len(performance_checkpoints)
        )
        if (
            len(workers) != 16
            or len(summary["restoration_checks"]) != 16 * expected_arms
            or not all(row["ok"] for row in summary["restoration_checks"])
        ):
            raise RuntimeError("Missing or failed rank restoration checks")
        if len(summary["arms"]) != expected_rows or not all(
            arm["complete"] for arm in summary["arms"]
        ):
            raise RuntimeError("Incomplete CE-by-LR update-depth matrix")
        if audit_k3:
            expected_files = expected_arms * max(checkpoints) * 16
            if (
                audit["files"] != expected_files
                or audit["differences"]
                or audit["first_forward_differences"]
                or audit["incomplete_sequences"]
                or any(row["nonfinite"] for row in audit["samples"])
            ):
                raise RuntimeError("K3 integrity audit failed; inspect k3_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--k3", action="store_true")
    arguments = parser.parse_args()
    analyze(arguments.root, arguments.k3)
