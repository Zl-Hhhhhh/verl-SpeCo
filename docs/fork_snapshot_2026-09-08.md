# Local DSpark experiment snapshot — 2026-09-08

> Historical record: the half-node and P4 wrapper scripts were removed on 2026-09-09. Use `tools/run_dspark_vllm_experiment.sh` with explicit device, dataset and attention settings for new runs; wrapper commands below describe the original experiments.

This branch preserves the code, configurations, tests, launch tools, and notes
used for the local P0–P4 experiments, including the running P4_03 configuration.
It is a snapshot of the working files based on upstream commit `f452ac8`.
Upstream/fork `main` was `88fb24b` when prepared, 21 commits ahead of that base;
those changes have not been merged into this experimental branch.

## Main additions

- Managed NPU experiment launches, serial host locking, recovery and checkpoint
  retention, with shared launchers rather than one script per run.
- DSpark full/SWA context controls and attention experiments.
- Trace-aligned response/rejected training, asymmetric K3, optional dynamic
  stream weights, and first-rejection/suffix weights.
- Student top-16 and student/teacher top-16 union reverse-KL objectives.
- Optional first-rejection correct-token CE, with separate global normalization.
- Dual `best_acceptance` / `best_throughput` drafter checkpoints, raw token
  diagnostics, and optional fixed evaluation trajectory exports.
- Experiment analysis and configuration notes in `docs/`.

Start with [the shared experiment guide](../tools/README_vllm_experiments.md),
[P4 configuration and launch records](p4_top16_full_run_2026-09-05.md), and
[the P4 series analysis](p4_experiment_summary_through_p4_02.md).

P4_03 uses DAPO, Qwen3-8B plus official DSpark block-7, full attention, student
top-16 on both streams, and first-rejection correct-token CE alpha **0.2**.
It collects at most 48 responses per replica and trains six optimizer steps
every five actor steps, for 101 actor steps. Dual best retention and K3 dumps
are enabled. See the launch record for the complete command.

## Environment recorded on the source host

| Package | Installed version |
| --- | --- |
| Python | 3.12 |
| verl-speco | 0.1.0, editable local source |
| verl | 0.8.0.dev0 |
| torch | 2.10.0 |
| torch-npu | 2.10.0.post2 |
| vllm | 0.26.0+empty |
| vllm-ascend | 0.1.dev1+ge592b2e79.d20260811 |
| ray | 2.56.1 |
| transformers | 5.14.1 |
| triton-ascend | 3.2.1 |

These are installed version labels, not a portable lockfile or a claim that
the same package labels reproduce local runtime builds. The experiment launcher
uses `/workspace/main_env_next/bin/python` and the host's CANN 9.0.1 runtime.
Runtime compatibility overlays implemented inside this repository are included;
external environment trees and any changes outside this repository are not.

`tools/run_dspark_p4.sh` delegates to the existing half-node launcher. Its device
mapping is specific to this host: physical cards 4–7 correspond to logical
devices 8–15. Override model/data/runtime paths and verify the device mapping
before running on another machine. The upstream README's broad runtime matrix
does not replace these experiment-specific requirements.

## What is stored separately

`.gitignore` excludes `outputs/`, checkpoints, W&B files, raw dumps, logs,
Parquet datasets, and virtual environments. Model and dataset directories under
`/workspace/models` and `/workspace/data` are also outside this repository.
They must be transferred separately when reproducing or backing up experiments.
Some analysis documents link to these local artifacts; those links will not
resolve in GitHub without a separate artifact copy.

The source checkout's worktree and staging area were preserved while preparing
this branch in an isolated Git worktree. An old staged-only
`docs/experiment_report_2026-08-04.md` was already absent from the working files;
it is not included in this current-working-tree snapshot. Its original staged
content remains in the source index and the local staging backup.

## Snapshot validation

The selected changed/new test files completed with **315 passed, 8 failed,
3 skipped** (470.77 seconds). This is a preservation snapshot, not a claim that
all historical launcher tests are current.

Python compilation and the selected managed-launcher shell syntax checks pass.
All copied changed files were compared byte-for-byte with the source worktree;
this document is the only snapshot-specific addition. Runtime/model settings
have not been altered to prepare this branch.

One known pre-existing test failure is
`tests/integration/test_dspark_attention_ablation_runner.py::test_attention_ablation_runner_is_frozen_dapo_validation`.
It references `tools/run_dspark_attention_ablation.sh`, which was already absent
from the source working directory. The failure was reproduced in the original
checkout. This snapshot retains the test and records the missing historical
helper instead of silently claiming that the entire selected suite passes.

The other seven failures are in `test_vllm_experiment_runner.py`: two DSpark
tests expect the older `p2_06`/cold-start defaults, and five DFlash naming,
checkpoint selection, or W&B-conflict tests expect a `100step` directory while
the launcher now defaults to `101step`. Consequently the conflict fixtures
create a different directory from the one the launcher selects. All copied
test and launcher files match the source checkout; these defaults were not
changed during this snapshot. P4-specific launcher and loss tests passed.

The local JUnit report is stored alongside the isolated worktree at
`outputs/fork_sync_20260908.hUMjJ2/pytest.xml` on the source host and is excluded
from the Git snapshot with the other output artifacts.
