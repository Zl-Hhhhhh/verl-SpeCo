#!/usr/bin/env python3
"""Seed a W&B run with completed steps parsed from a VERL console log."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from plot_training_curves import parse_console_metrics

wandb: Any = importlib.import_module("wandb")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--max-step", required=True, type=int)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group")
    parser.add_argument("--job-type")
    parser.add_argument("--tags", default="")
    args = parser.parse_args()

    records = [
        record
        for record in parse_console_metrics(args.log)
        if record["step"] <= args.max_step
    ]
    if not records or int(records[-1]["step"]) != args.max_step:
        parser.error(f"log does not contain a complete step {args.max_step}")

    run = wandb.init(
        project=args.project,
        id=args.run_id,
        resume="never",
        name=args.name,
        group=args.group,
        job_type=args.job_type,
        tags=[tag for tag in args.tags.split(",") if tag],
        config={
            "seeded_from_log": str(args.log.resolve()),
            "seeded_through_step": args.max_step,
        },
    )
    for record in records:
        step = int(record["step"])
        run.log(
            {key: value for key, value in record.items() if key != "step"}, step=step
        )
    run.finish()
    print(f"seeded {len(records)} steps into {args.project}/{args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
