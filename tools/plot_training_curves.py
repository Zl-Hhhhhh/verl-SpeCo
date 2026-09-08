#!/usr/bin/env python3
"""Plot the primary SPECO training curves from console log files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_LINE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
METRICS = (
    (
        "drafter/spec_decode/mean_acceptance_length",
        "Mean Acceptance Length",
        "accepted tokens",
    ),
    ("timing_s/gen", "Generation Time", "seconds"),
    ("timing_s/step", "Step Time", "seconds"),
    ("critic/rewards/mean", "Critic Reward", "reward"),
)


@dataclass(frozen=True)
class RunSeries:
    label: str
    source: Path
    steps: tuple[dict[str, float], ...]


def parse_console_metrics(path: Path) -> tuple[dict[str, float], ...]:
    """Extract one metric record per global step from a VERL console log."""
    records: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = ANSI_ESCAPE.sub("", raw_line)
            match = STEP_LINE.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            values: dict[str, float] = {"step": float(step)}
            for item in match.group(2).split(" - "):
                key, separator, raw_value = item.rpartition(":")
                if not separator:
                    continue
                try:
                    value = float(raw_value.strip())
                except ValueError:
                    continue
                if math.isfinite(value):
                    values[key.strip()] = value
            if any(metric in values for metric, _, _ in METRICS):
                records[step] = values
    return tuple(records[step] for step in sorted(records))


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("run must use LABEL=/path/to/training.log")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"log file does not exist: {path}")
    return label.strip(), path


def write_csv(path: Path, runs: list[RunSeries]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "source", "step", *(metric for metric, _, _ in METRICS)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for record in run.steps:
                writer.writerow(
                    {
                        "run": run.label,
                        "source": str(run.source),
                        "step": int(record["step"]),
                        **{metric: record.get(metric, "") for metric, _, _ in METRICS},
                    }
                )


def plot_runs(path: Path, runs: list[RunSeries], title: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    figure.patch.set_facecolor("#f8fafc")
    figure.suptitle(title, fontsize=19, fontweight="bold")

    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    for axis, (metric, metric_title, unit) in zip(axes.flat, METRICS, strict=True):
        axis.set_facecolor("white")
        plotted = False
        for index, run in enumerate(runs):
            points = [
                (int(record["step"]), record[metric])
                for record in run.steps
                if metric in record
            ]
            if not points:
                continue
            plotted = True
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=colors[index % len(colors)],
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=run.label,
            )
        axis.set_title(metric_title, fontsize=14)
        axis.set_xlabel("training step")
        axis.set_ylabel(unit)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        if plotted:
            axis.legend(frameon=False, loc="best")
        else:
            axis.text(
                0.5,
                0.5,
                "metric unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="LABEL=LOG",
        help="label and console log path; repeat to compare runs",
    )
    parser.add_argument("--output", required=True, type=Path, help="output PNG path")
    parser.add_argument(
        "--csv-output", type=Path, help="output CSV path (defaults next to the PNG)"
    )
    parser.add_argument("--title", default="SPECO Full-Flow Training Metrics")
    parser.add_argument(
        "--min-step", type=int, help="only include steps at or above this value"
    )
    parser.add_argument(
        "--max-step", type=int, help="only include steps at or below this value"
    )
    args = parser.parse_args()

    if (
        args.min_step is not None
        and args.max_step is not None
        and args.min_step > args.max_step
    ):
        parser.error("--min-step cannot be greater than --max-step")

    runs = []
    for label, path in args.run:
        records = parse_console_metrics(path)
        records = tuple(
            record
            for record in records
            if (args.min_step is None or record["step"] >= args.min_step)
            and (args.max_step is None or record["step"] <= args.max_step)
        )
        runs.append(RunSeries(label, path, records))
    empty = [run.label for run in runs if not run.steps]
    if empty:
        parser.error(f"no training metric lines found for: {', '.join(empty)}")

    output = args.output.expanduser().resolve()
    csv_output = (
        args.csv_output.expanduser().resolve()
        if args.csv_output is not None
        else output.with_suffix(".csv")
    )
    plot_runs(output, runs, args.title)
    write_csv(csv_output, runs)
    for run in runs:
        print(f"{run.label}: {len(run.steps)} steps from {run.source}")
    print(f"chart: {output}")
    print(f"data: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
