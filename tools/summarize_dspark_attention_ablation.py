#!/usr/bin/env python3
"""Summarize frozen DSpark attention-ablation metrics from managed logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("full", "swa512", "swa1024", "swa2048")
METRIC_PREFIX = "val-aux/drafter/spec_decode/"
CORE_METRICS = (
    "verify_steps",
    "accepted_tokens",
    "mean_acceptance_length",
    "generation_time_s",
    "response_tokens",
)
OPTIONAL_METRICS = (
    "response_tokens_per_second",
    "reward_mean",
    "accuracy_mean",
    *(
        f"position_{position}_{suffix}"
        for position in range(1, 8)
        for suffix in ("attempted", "accepted", "acceptance_rate")
    ),
)
METRICS = (*CORE_METRICS, *OPTIONAL_METRICS)
EXACT_METRIC_ALIASES = {
    "val-aux/math_dapo/reward/mean@1": "reward_mean",
    "val-core/math_dapo/acc/mean@1": "accuracy_mean",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _latest_log(log_root: Path, experiment_id: str) -> Path | None:
    candidates = list(log_root.rglob(f"{experiment_id}.log"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _parse_metrics(log_path: Path) -> dict[str, float]:
    latest: dict[str, float] = {}
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        if METRIC_PREFIX not in line and not any(
            key in line for key in EXACT_METRIC_ALIASES
        ):
            continue
        for item in line.split(" - "):
            if ":" not in item:
                continue
            key, value = item.rsplit(":", 1)
            key = key.strip().strip("'\"")
            name = EXACT_METRIC_ALIASES.get(key)
            if name is None and key.startswith(METRIC_PREFIX):
                name = key.removeprefix(METRIC_PREFIX)
            if name not in METRICS:
                continue
            try:
                latest[name] = float(value.strip().rstrip(",)}'\""))
            except ValueError:
                continue
    return latest


def summarize(
    prefix: str,
    log_root: Path,
    output_dir: Path,
    arms: tuple[str, ...] = ARMS,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for arm in arms:
        experiment_id = f"{prefix}_{arm}"
        log_path = _latest_log(log_root, experiment_id)
        metrics = _parse_metrics(log_path) if log_path else {}
        status = (
            "completed"
            if all(name in metrics for name in CORE_METRICS)
            else "pending"
        )
        records.append(
            {
                "arm": arm,
                "experiment_id": experiment_id,
                "status": status,
                "log_path": str(log_path.resolve()) if log_path else None,
                **{name: metrics.get(name) for name in METRICS},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ablation_prefix": prefix,
        "records": records,
    }
    json_tmp = output_dir / ".summary.json.tmp"
    json_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_tmp.replace(output_dir / "summary.json")

    fieldnames = ["arm", "status", *METRICS, "experiment_id", "log_path"]
    csv_tmp = output_dir / ".summary.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    csv_tmp.replace(output_dir / "summary.csv")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    records = summarize(
        args.prefix, args.log_root, args.output_dir, tuple(args.arms)
    )
    completed = sum(record["status"] == "completed" for record in records)
    print(f"wrote {completed}/{len(records)} completed arms to {args.output_dir}")


if __name__ == "__main__":
    main()
