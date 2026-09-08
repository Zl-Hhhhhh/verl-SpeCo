#!/usr/bin/env python3
"""Aggregate repeated DSpark attention runs and verify output equivalence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from tools.summarize_dspark_attention_ablation import (
    CORE_METRICS,
    METRICS,
    _latest_log,
    _parse_metrics,
)


def _ordered_text_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        output = str(record.get("output", ""))
        payload = output.encode("utf-8")
        digest.update(f"{len(payload)}:".encode())
        digest.update(payload)
        count += 1
    return digest.hexdigest(), count


def _diagnostic_hashes(run_dir: Path) -> dict[str, object]:
    summaries = sorted(run_dir.glob("diagnostics/spec_decode.step-*.json"))
    text_dumps = sorted(run_dir.glob("text/*.jsonl"))
    result: dict[str, object] = {}
    if summaries:
        payload = json.loads(summaries[-1].read_text(encoding="utf-8"))
        result.update(
            {
                "token_sha256": payload.get("ordered_response_token_sha256"),
                "token_sample_count": payload.get("sample_count"),
                "trace_event_count": payload.get("trace_event_count"),
                "acceptance_by_position": payload.get("acceptance_by_position", []),
                "diagnostics_path": str(summaries[-1].resolve()),
            }
        )
    if text_dumps:
        text_hash, text_count = _ordered_text_digest(text_dumps[-1])
        result.update(
            {
                "text_sha256": text_hash,
                "text_sample_count": text_count,
                "text_path": str(text_dumps[-1].resolve()),
            }
        )
    return result


def analyze(
    *,
    prefixes: tuple[str, ...],
    arms: tuple[str, ...],
    log_root: Path,
    analysis_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for prefix in prefixes:
        for arm in arms:
            experiment_id = f"{prefix}_{arm}"
            log_path = _latest_log(log_root, experiment_id)
            metrics = _parse_metrics(log_path) if log_path else {}
            record: dict[str, object] = {
                "prefix": prefix,
                "arm": arm,
                "experiment_id": experiment_id,
                "status": (
                    "completed"
                    if all(metric in metrics for metric in CORE_METRICS)
                    else "pending"
                ),
                "log_path": str(log_path.resolve()) if log_path else None,
                **{metric: metrics.get(metric) for metric in METRICS},
            }
            if (
                record.get("response_tokens_per_second") is None
                and metrics.get("response_tokens") is not None
                and float(metrics.get("generation_time_s", 0.0) or 0.0) > 0.0
            ):
                record["response_tokens_per_second"] = float(
                    metrics["response_tokens"]
                ) / float(metrics["generation_time_s"])
            record.update(
                _diagnostic_hashes(analysis_root / prefix / "validation" / arm)
            )
            runs.append(record)

    aggregate: list[dict[str, object]] = []
    aggregate_metrics = (
        "verify_steps",
        "accepted_tokens",
        "mean_acceptance_length",
        "generation_time_s",
        "response_tokens_per_second",
        "reward_mean",
        "accuracy_mean",
    )
    for arm in arms:
        arm_runs = [
            record
            for record in runs
            if record["arm"] == arm and record["status"] == "completed"
        ]
        row: dict[str, object] = {"arm": arm, "completed_runs": len(arm_runs)}
        for metric in aggregate_metrics:
            values = [
                float(record[metric])
                for record in arm_runs
                if record.get(metric) is not None
            ]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_median"] = statistics.median(values) if values else None
            row[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        aggregate.append(row)

    hashes = [
        {
            "prefix": record["prefix"],
            "arm": record["arm"],
            "token_sha256": record.get("token_sha256"),
            "text_sha256": record.get("text_sha256"),
            "token_sample_count": record.get("token_sample_count"),
            "text_sample_count": record.get("text_sample_count"),
        }
        for record in runs
        if record.get("token_sha256") or record.get("text_sha256")
    ]
    token_hashes = {item["token_sha256"] for item in hashes if item["token_sha256"]}
    text_hashes = {item["text_sha256"] for item in hashes if item["text_sha256"]}
    equivalence = {
        "runs_with_hashes": len(hashes),
        "token_outputs_all_equal": len(token_hashes) == 1 if token_hashes else None,
        "text_outputs_all_equal": len(text_hashes) == 1 if text_hashes else None,
        "records": hashes,
    }
    payload = {
        "schema_version": 1,
        "prefixes": list(prefixes),
        "arms": list(arms),
        "runs": runs,
        "aggregate": aggregate,
        "output_equivalence": equivalence,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_fields = [
        "prefix",
        "arm",
        "status",
        *METRICS,
        "token_sha256",
        "text_sha256",
        "token_sample_count",
        "text_sample_count",
        "experiment_id",
        "log_path",
    ]
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    aggregate_fields = list(aggregate[0]) if aggregate else ["arm", "completed_runs"]
    with (output_dir / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=["full", "swa512", "swa1024"])
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        prefixes=tuple(args.prefixes),
        arms=tuple(args.arms),
        log_root=args.log_root,
        analysis_root=args.analysis_root,
        output_dir=args.output_dir,
    )
    completed = sum(record["status"] == "completed" for record in payload["runs"])
    print(f"wrote {completed}/{len(payload['runs'])} completed runs to {args.output_dir}")


if __name__ == "__main__":
    main()
