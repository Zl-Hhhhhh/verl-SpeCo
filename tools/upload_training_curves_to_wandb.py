#!/usr/bin/env python3
"""Upload numeric training metrics parsed from VERL console logs to W&B."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from plot_training_curves import METRICS, parse_console_metrics, parse_run

wandb: Any = importlib.import_module("wandb")


KEY_METRIC_NAMES = (
    "key_metrics/mean_acceptance_length",
    "key_metrics/generation_time_s",
    "key_metrics/step_time_s",
    "key_metrics/critic_reward_mean",
)


def add_key_metric_aliases(record: dict[str, float]) -> dict[str, float]:
    aliases = {
        alias: record[source]
        for alias, (source, _, _) in zip(KEY_METRIC_NAMES, METRICS, strict=True)
        if source in record
    }
    return {**record, **aliases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="NAME=LOG",
        help="W&B display name and console log path; repeat to upload multiple runs",
    )
    parser.add_argument(
        "--entity",
        default="linzzz-hong-kong-university-of-science-and-technology",
    )
    parser.add_argument(
        "--project",
        required=True,
        help=(
            "destination W&B project, normally verl_speco_dapo, "
            "verl_speco_taco, or verl_speco_wildsci"
        ),
    )
    parser.add_argument("--group", default="qwen3_8b_main_env")
    parser.add_argument("--job-type", default="console-log-import")
    parser.add_argument("--tags", default="main_env,npu,vllm-ascend,console-log-import")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()

    base_tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    uploaded = []
    for name, source in args.run:
        records = parse_console_metrics(source)
        if not records:
            parser.error(f"no training metric lines found in {source}")

        algorithm_tags = [
            algorithm for algorithm in ("eagle3", "dflash") if algorithm in name.lower()
        ]
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=name,
            group=args.group,
            job_type=args.job_type,
            tags=[*base_tags, *algorithm_tags],
            mode=args.mode,
            config={
                "source_log": str(Path(source).resolve()),
                "imported_from_console_log": True,
                "imported_step_count": len(records),
            },
        )
        run.define_metric("training/global_step")
        run.define_metric("*", step_metric="training/global_step")
        for record in records:
            record = add_key_metric_aliases(record)
            run.log(
                {key: value for key, value in record.items() if key != "step"},
                step=int(record["step"]),
            )
        run_url = run.url
        run.finish(exit_code=0)
        uploaded.append((name, len(records), run_url))

    wandb.teardown(exit_code=0)
    for name, step_count, run_url in uploaded:
        print(f"{name}: uploaded {step_count} steps -> {run_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
