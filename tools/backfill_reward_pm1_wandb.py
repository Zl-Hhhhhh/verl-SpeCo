#!/usr/bin/env python3
"""Create W&B overlays with historical rewards converted from ``[0, 1]`` to ``[-1, 1]``.

W&B history is append-only, so this deliberately leaves the source runs and
their original ``critic/rewards/mean`` values intact.  Each overlay is created
in the same project and group, carries a link to its source run, and logs the
standard training metrics at the original W&B step:

* ``critic/rewards/mean``
* ``critic/score/mean``
* ``key_metrics/critic_reward_mean``
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any

wandb: Any = importlib.import_module("wandb")

DEFAULT_SOURCE_RUNS = (
    "verl_speco_taco/931233016f8a",
    "verl_speco_taco/8d7654f633fa",
    "verl_speco_wildsci/d0b8f62e1315",
)


@dataclass(frozen=True)
class SourceRun:
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.project}/{self.run_id}"


def parse_source_run(value: str) -> SourceRun:
    project, separator, run_id = value.strip().partition("/")
    if not separator or not project or not run_id or "/" in run_id:
        raise argparse.ArgumentTypeError("source must use PROJECT/RUN_ID")
    return SourceRun(project=project, run_id=run_id)


def collect_rewards(source_run: Any, source_metric: str) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    for row in source_run.scan_history(keys=["_step", source_metric]):
        value = row.get(source_metric)
        step = row.get("_step")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not isinstance(step, int):
            continue
        raw_score = float(value)
        if raw_score < -1e-6 or raw_score > 1.0 + 1e-6:
            raise ValueError(
                f"{source_run.path}: {source_metric} is not a raw [0, 1] reward "
                f"at step {step}: {raw_score}"
            )
        records.append((step, min(1.0, max(0.0, raw_score))))
    return records


def destination_run_id(source: SourceRun) -> str:
    """Use a deterministic W&B id so a second invocation safely skips it.

    The initial ``pm1`` overlay IDs were deleted after their comparison-only
    schema was retired. W&B never permits a deleted ID to be reused, hence the
    stable ``pm1v2`` suffix for the standard-metric overlays.
    """
    return f"{source.run_id}pm1v2"


def get_destination(api: Any, entity: str, source: SourceRun) -> Any | None:
    destination_path = f"{entity}/{source.project}/{destination_run_id(source)}"
    try:
        return api.run(destination_path)
    except Exception as exc:  # W&B has changed this exception type across releases.
        if exc.__class__.__name__ in {"CommError", "UsageError"}:
            return None
        raise


def destination_has_pm1_metrics(destination: Any) -> bool:
    for row in destination.scan_history(
        keys=["critic/rewards/mean", "key_metrics/critic_reward_mean"]
    ):
        if (
            isinstance(row.get("critic/rewards/mean"), (int, float))
            and isinstance(row.get("key_metrics/critic_reward_mean"), (int, float))
        ):
            return True
    return False


def upload_overlay(
    *,
    entity: str,
    source: SourceRun,
    source_run: Any,
    records: list[tuple[int, float]],
    source_metric: str,
) -> str:
    tags = list(source_run.tags or [])
    for tag in ("reward-pm1", "historical-reward-overlay"):
        if tag not in tags:
            tags.append(tag)

    run = wandb.init(
        entity=entity,
        project=source.project,
        id=destination_run_id(source),
        resume="never",
        name=f"{source_run.name} [reward pm1]",
        group=source_run.group,
        job_type="reward-scale-backfill",
        tags=tags,
        config={
            "source_run_path": f"{entity}/{source.path}",
            "source_metric": source_metric,
            "reward_transform": "pm1 = 2 * raw01 - 1",
            "historical_reward_overlay": True,
            "source_step_count": len(records),
        },
    )
    for source_step, raw_score in records:
        pm1_score = 2.0 * raw_score - 1.0
        run.log(
            {
                "training/global_step": source_step,
                "critic/rewards/mean": pm1_score,
                "critic/score/mean": pm1_score,
                "key_metrics/critic_reward_mean": pm1_score,
            },
            step=source_step,
        )
    url = run.url
    run.finish(exit_code=0)
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source_run,
        metavar="PROJECT/RUN_ID",
        help="source W&B run; defaults to all existing non-DAPO reward runs",
    )
    parser.add_argument(
        "--entity", default="linzzz-hong-kong-university-of-science-and-technology"
    )
    parser.add_argument("--source-metric", default="critic/rewards/mean")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = args.source or [parse_source_run(value) for value in DEFAULT_SOURCE_RUNS]
    api = wandb.Api(timeout=60)
    for source in sources:
        source_run = api.run(f"{args.entity}/{source.path}")
        records = collect_rewards(source_run, args.source_metric)
        if not records:
            raise ValueError(f"{source_run.path}: no numeric {args.source_metric} history")
        raw_values = [value for _, value in records]
        destination_id = destination_run_id(source)
        print(
            f"{source_run.path}: {len(records)} points, raw range "
            f"[{min(raw_values):.6g}, {max(raw_values):.6g}], destination={destination_id}"
        )
        if args.dry_run:
            continue
        destination = get_destination(api, args.entity, source)
        if destination is not None and destination_has_pm1_metrics(destination):
            print(f"  skipped: {args.entity}/{source.project}/{destination_id} already has pm1 metrics")
            continue
        if destination is not None:
            raise RuntimeError(
                f"{args.entity}/{source.project}/{destination_id} exists but has an incompatible history; "
                "delete only the overlay run before retrying"
            )
        else:
            url = upload_overlay(
                entity=args.entity,
                source=source,
                source_run=source_run,
                records=records,
                source_metric=args.source_metric,
            )
            print(f"  uploaded: {url}")

    wandb.teardown(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
