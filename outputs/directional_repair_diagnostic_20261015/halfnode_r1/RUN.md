# Directional repair target diagnostic: physical cards 4–7

Base commit: `13624b06f8b43c75e556423a2ac12d22209d6994`.
Scope: plan item 1 only; actor0 → actor5, zero drafter optimizer steps.

The interpreter and packages are from the existing local `main_env_next`.
Preflight versions and full source hashes are in `environment_check.json`.
All eight permitted logical devices passed a BF16 matrix operation, and both
A3 custom operator registration libraries loaded successfully. The 16 existing
directional math/runtime/analyzer tests passed; two additional topology cases
(8 and 16 ranks) passed. Hydra composition passed; see `resolved_config.yaml`.

Physical cards 4–7 map to logical devices 8–15. TP/SP remain 4, giving two
independent data replicas. Compared with the planned four replicas, per-replica
collection limits, batch size, and holdout sizes are doubled to preserve the
global limits: 40 responses, 32,768 tokens, batch 8, holdout 8 per replica.
The nominal maximum is 64 training responses × 32 positions = 2,048 positions;
actual valid counts must come from the paired artifacts. Window length is 512.
The rollout configuration explicitly uses temperature 1, top-p 1, top-k -1,
and vLLM generation defaults, overriding the model's different HF defaults.

The full original DAPO training source has 1,791,700 rows (17,917 examples
repeated 100 times). It has zero prompt overlap with the original validation
file. `actor_train_disjoint.parquet` is therefore a symlink to the unmodified
original source. An initial attempt to write a filtered Parquet caused a
PyArrow readback segmentation fault; that generated file was removed and is
not used for training. No Python dependency was replaced to handle this.

Repository-external additions made during setup:

- Python 3.12.13 installed by uv under `/root/.local/share/uv/python/`, with
  `/root/.local/bin/python3.12` pointing to it.
- `/usr/local/python3.12.13` points to that Python installation, repairing the
  existing venv's broken interpreter symlink.
- `/workspace` points to `/data1/heyuhang/workspace`, restoring the absolute
  paths embedded in the existing activation scripts, CANN tree and editable package.
- `main_env_next/compat/lib/libstdc++.so.6.0.30` and its SONAME symlink provide
  the `GLIBCXX_3.4.29` required by the copied A3 operators. The library was
  copied from the existing local Docker layer
  `d006c0e25bd150a9bc524f0859af8bbdfea85c258522e4d39f0796a4a6cb3816`.
- `main_env_next/bin/activate-speco-local.sh` sources the original activation
  script and prepends this compatibility library for experiment processes.

System libraries and existing Python packages were not replaced. Other users'
processes and cards 0–3 are excluded. Global `ray stop` is disabled. W&B runs
offline. The detached wrapper automatically analyzes targets after training
and updates `status.txt`; failure stops the sequence.

Entrypoint: `outputs/experiment_setup/directional_repair_diagnostic_20261015/execute_halfnode.sh`.
Session PID: `pid.txt`. Driver/launcher output: `execution.log`.
Training log: `outputs/validation_logs/2026-10-15/dspark_vllm/dspark_dapo_directional_repair_probe_5step_20261015_halfnode_r1.log`.

First startup attempt failed before actor training: vLLM sleep tried to allocate
pinned host weight backups and received ACL error 207001. A standalone 2 GiB
pinned allocation succeeds; the vLLM default CPU binding also binds memory to
one NUMA node, and node 4 has only about 5 GiB free despite more than 400 GiB
available globally. The retry disables `additional_config.enable_cpu_binding`
for these experiment servers, removing that NUMA restriction.

The same vLLM default binding code stopped the host's previously active
`irqbalance` service. The service was restored with `systemctl start irqbalance`
and verified active; CPU binding is disabled on retry so that experiment
workers do not invoke this host service adjustment again. This default runtime
side effect was discovered in the startup log, not during the small matmul
preflight. The failed attempt's output is retained in `execution_attempt1.log`.


Second startup attempt also failed at CaMem pinned weight-backup allocation
before actor step 1, despite CPU binding disabled. CPU binding alone therefore
does not explain or fix the allocation limit. Host available memory was ~423 GiB.
Its output is retained in `execution_attempt2.log`.

Third attempt opts into repository-only `SPECO_VLLM_PAGEABLE_SLEEP=1`, propagated
explicitly to Ray workers. CaMem backups use ordinary CPU tensors with the same
synchronous memcpy, tag selection, and wake-up path. A real 1 MiB NPU tensor
round trip on each logical device 8 through 15 passed exact tensor equality;
backups were confirmed unpinned. Existing packages are unchanged.
Runtime and directional integration tests: 58 passed.

A separate memory watchdog records host available memory, swap usage, and all
8 permitted logical devices' HBM every 10 seconds in `memory_usage.jsonl`.
User-selected threshold: below 32 GiB MemAvailable for two consecutive samples.
On breach, stop only the experiment session and processes carrying its unique
SPECO_DIRECTIONAL_RUN_ID, then write `memory_guard_triggered.json` and status.
The watchdog stop path was verified against an isolated tagged dummy process;
all eight HBM device readings were verified to parse correctly.


Third attempt passed full vLLM sleep and drafter initialization; first rollout
started at approximately 07:25 America/New_York. Runtime remains in progress.
A source snapshot with SHA256s, commit/remotes, run ID, hardware check details,
and memory policy is saved in `reproducibility/manifest.json`. This snapshot
also includes ignored setup scripts so future source changes cannot overwrite
the executed setup. `progress_snapshot.json` records the latest inspected state.

Analyzer coverage audit: added all-candidate sum(m)/sum(TV), explicit
conditional denominators including TV=0 exclusions, target minimum probability
and maximum normalization/TV-identity errors, and fixed-head feature-drift
definitions. Full target checks currently cover selected top-m positions.
The analyzer fails if no paired feature-drift observations are available.
A synthetic end-to-end analysis checks the zero-TV exclusion, both eta values,
feature drift, and output artifacts; see analyzer_selfcheck.log.


Final review completed after actor0-to-actor5 run and offline analysis.
All five actor updates completed; no drafter optimizer updates. Artifacts:
8192 candidate positions, 2048 distinct selected positions, 64 response IDs.
Math checks cover the selected positions; all candidates have scalar scores.
The report is RESULTS.md and docs/directional_repair_diagnostic_results_20261015.md.
Lowest host MemAvailable was 66.58 GiB; peak permitted logical-device HBM was
48.59 GiB; the user-selected 32 GiB guard was not triggered. No processes with
this run ID remained at final review; irqbalance is active.

Artifact readback exposed local Arrow dictionary/bit-packed-boolean problems.
No installed packages were changed. Offline exports now disable dictionary
encoding, store the Parquet selected flag as int8 0/1, and retain full-precision
JSONL. Both Parquet files match JSONL field by field within 1e-12; the all-
candidate two-sided denominator matches original JSONL and the summary report.
Logs: artifact_readback.log, final_checks.log (18 passed),
parquet_regression_test.log (updated readback test passed), report_generation.log.
Repeated q forward noise is zero; actor-feature recapture noise was not measured.
Directions do not establish a benefit over ordinary mass-matched repair.
Stages 2 and 3 remain unstarted.
