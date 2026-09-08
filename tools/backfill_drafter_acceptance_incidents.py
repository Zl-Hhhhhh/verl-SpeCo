#!/usr/bin/env python3
"""Backfill post-update drafter acceptance regressions from validation logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
MEAN_ACCEPTANCE_RE = re.compile(
    r"drafter/spec_decode/mean_acceptance_length:([-+0-9.eE]+)"
)
PUBLISHED_RE = re.compile(r"drafter/published:(?:1(?:\.0+)?)\b")
TRAINED_RE = re.compile(r"drafter/trained:(?:1(?:\.0+)?)\b")


def _strategy_version(log_file: Path) -> str:
    name = log_file.stem
    if "__" in name:
        return name.rsplit("__", 1)[1]
    return "legacy"


def _algorithm(log_file: Path) -> str:
    parent = log_file.parent.name
    return parent.removesuffix("_vllm") or "unknown"


def _historical_incidents(log_root: Path, threshold: float) -> list[dict]:
    incidents: list[dict] = []
    for log_file in sorted(log_root.rglob("*.log")):
        pending: dict | None = None
        for line_number, raw_line in enumerate(
            log_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            line = ANSI_ESCAPE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            acceptance_match = MEAN_ACCEPTANCE_RE.search(line)
            step = int(step_match.group(1)) if step_match else None
            acceptance = (
                float(acceptance_match.group(1)) if acceptance_match else None
            )
            if pending is not None and step is not None and acceptance is not None:
                if step > pending["update_step"]:
                    drop = pending["baseline"] - acceptance
                    if drop >= threshold:
                        relative_log = str(log_file)
                        identity = (
                            f"{relative_log}:{pending['line_number']}:{line_number}:"
                            f"{pending['update_step']}:{step}:{threshold:.12g}"
                        )
                        incidents.append(
                            {
                                "schema_version": 1,
                                "incident_id": hashlib.blake2b(
                                    identity.encode(), digest_size=12
                                ).hexdigest(),
                                "event": "post_update_mean_acceptance_length_regression",
                                "source": "historical_log_backfill",
                                "recorded_at_utc": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "action": "historical_detected",
                                "algorithm": _algorithm(log_file),
                                "strategy_version": _strategy_version(log_file),
                                "experiment_id": log_file.stem,
                                "run_mode": (
                                    "online" if "_online_" in log_file.stem else "unknown"
                                ),
                                "wandb_run_id": "",
                                "training_log": relative_log,
                                "update_step": pending["update_step"],
                                "observed_step": step,
                                "baseline_mean_acceptance_length": pending[
                                    "baseline"
                                ],
                                "observed_mean_acceptance_length": acceptance,
                                "drop": drop,
                                "threshold": threshold,
                                "published_line_number": pending["line_number"],
                                "observed_line_number": line_number,
                                "drafter_trained": pending["drafter_trained"],
                            }
                        )
                    pending = None
                elif step < pending["update_step"]:
                    pending = None
            if (
                step is not None
                and acceptance is not None
                and PUBLISHED_RE.search(line)
            ):
                pending = {
                    "update_step": step,
                    "baseline": acceptance,
                    "line_number": line_number,
                    "drafter_trained": bool(TRAINED_RE.search(line)),
                }
    return incidents


def _existing_ids(ledger: Path) -> set[str]:
    if not ledger.exists():
        return set()
    ids: set[str] = set()
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        incident_id = record.get("incident_id")
        if isinstance(incident_id, str):
            ids.add(incident_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("outputs/validation_logs"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("outputs/drafter_acceptance_incidents.jsonl"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    existing_ids = _existing_ids(args.ledger)
    incidents = _historical_incidents(args.log_root, args.threshold)
    new_incidents = [
        incident for incident in incidents if incident["incident_id"] not in existing_ids
    ]
    if new_incidents:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        with args.ledger.open("a", encoding="utf-8") as output:
            for incident in new_incidents:
                output.write(json.dumps(incident, sort_keys=True) + "\n")
    print(
        f"detected={len(incidents)} appended={len(new_incidents)} ledger={args.ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
