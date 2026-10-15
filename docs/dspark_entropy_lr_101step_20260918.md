# Entropy-controlled drafter LR experiment (2026-09-18)

Launch: `bash tools/run_dspark_entropy_lr_101step.sh` inside the experiment container.
Experiment: `dspark_dapo2000_online_entropy_lr_101step_20260918`.
Managed run: `dspark_20260918T102658Z_1578036` (launched 2026-09-18 10:26:58 UTC).
Prelaunch suite: 160 passed; 16 warnings. Managed launcher returned successfully;
this launch record does not imply completion of the first training cycle.

The first run failed at step 5 before its first probe with `KeyError: input_ids`:
the installed RLHFDataset returns raw chat prompts and a dummy tensor. No full
checkpoint was saved before this failure. The repaired rerun uses experiment
`dspark_dapo2000_online_entropy_lr_101step_20260918_r2` and starts from the original
models, preserving the failed run's artifacts.
Rerun managed ID: `dspark_20260918T151735Z_2101234`, launched at 15:17:35 UTC.
Background startup is verified; the step-5 NPU validation is still pending at handoff.

R2 subsequently failed before the first actor step: the new preflight called
`validation_batch` before upstream `RayPPOTrainer.fit` initialized `global_steps`.
R3 separates `prepare_validation_cohort` (no step dependency) from runtime batch
metadata construction. Its regression test enters the real SPECO fit wrapper
without a `global_steps` attribute, then verifies that validation metadata uses
the step set by upstream fit and refreshes on subsequent calls. Real-data preflight
also begins without this attribute. Repair checks: 84 tests passed (17 warnings),
real-data preflight passed, shell syntax and whitespace checks passed. R3 remains
a fresh 101-step run with identical model/LR/loss/offload configuration.
R3 managed run: `dspark_20260918T154326Z_2164034`, launched 15:43:26 UTC.
The managed process is running at handoff; first actor step and online validation
are not yet verified. Failed R1/R2 records are preserved under their original IDs.

The repair deduplicates `raw_prompt`, uses the installed generation-batch adapter,
and checks the cohort before actor training begins. Real-data preflight found
that the validation file's 960 rows contain only 30 unique prompts, each repeated
32 times, with zero prompt overlap with the actor training file. This run uses
all 30; dispatch pads to 32 for eight rollout workers, then removes padding before
trace statistics and throughput-token counting. There is no larger independent
cohort available in this file. The LR rule itself is unchanged.

Repair validation: 82 focused tests passed (17 warnings), real Qwen3 tokenizer /
DAPO validation dataset / installed verl adapter preflight passed, shell syntax
and whitespace checks passed. The tests exercise deduplication across batches,
actual raw-chat collation, and exclusion of padded responses from measurements.

Start fresh from Qwen3-8B and dspark_qwen3_8b_block7 for 101 actor steps,
using the same DAPO 2000 seed-17 data, batch 64 × 5 responses, 16 NPUs,
TP/SP 4, and prompt/response limits 512/8192. Actor parameter offload is true.
All asymmetric K3 flags are false; ordinary uniform anchors and legacy K3
clipping remain. Verification entropy and K3 log probabilities are recorded.
Each fifth actor step collects data and performs five drafter optimizer steps.

## LR and evidence

- Base LR 1e-5, floor 5e-6, cap 1.5e-5, constant scheduler without warmup.
- Certainty C is the share of reached verification positions with actor entropy
  below 0.01. Unreached suffix positions are excluded. EMA beta is 0.3.
- The ceiling is `1e-5 + 5e-6 * clip((EMA(C)-0.70)/0.15, 0, 1)`.
- Mismatch M is first rejections among reached low-entropy positions divided
  by all reached low-entropy positions. It is not the overall rejection rate.
- Following the user's latest instruction, **one** passing validation increases
  the next cycle's LR by 10%, subject to M > 10%, rejected K3 clamp ratio < 50%,
  and the entropy ceiling. No consecutive-success requirement remains.
- A clear regression halves LR, bounded by the floor. Inconclusive evidence
  leaves LR unchanged. Entropy observations may reduce LR to a lower ceiling.

After the actor update, synchronize the actor once and compare old/new drafter
on up to 32 unique validation prompts with greedy generation capped at 512
tokens. Expand to up to 64 if inconclusive and more unique prompts are available
(the actual DAPO file supplies only 30). Accept the candidate only when aggregate
acceptance length improves by at least 0.05, the paired mean difference minus
2.1 standard errors is positive, low-entropy mismatch decreases, and estimated
time savings across five real rollouts exceed training/probe/publication cost.
The interval is a screening rule, not a formal repeated-testing guarantee.
An acceptance interval wholly below -0.05 or latency regression over 15% counts
as clear regression. Other failures to demonstrate improvement are inconclusive.

Rejected candidates restore baseline parameters, optimizer moments, scheduler,
and counters. Candidate resolution precedes checkpoint saving; the upstream
actor synchronization is skipped only if already performed for this validation.
Existing collection, fixed-loss, gradient, and post-update acceptance guards
remain. The precheck permits clamp ratio up to 1 and rejected K3 median up to 10;
the separate 50% saturation threshold prevents LR growth, not candidate evaluation.

Controller decisions are appended to `entropy_lr_decisions.jsonl` in the
experiment checkpoint directory. Each actor checkpoint includes
`entropy_lr_state.json`. Training metrics include applied/next LR, ceiling,
certainty EMA, mismatch, saturation, paired evidence, net-time estimate, and
commit/rollback status.

## Limits and checks

Short greedy validation estimates acceptance and latency; full training rollout
throughput and wall-clock time remain the final efficiency measures. Validation
adds overhead and changes the runtime schedule relative to the fixed-LR baseline.
The previous run's cross-rank log-q divergence was not proven fixed by this change.
Inspect retained K3/entropy diagnostics when comparing the runs.

CPU tests cover one-validation LR growth, caps/floors, reached-position counting,
paired evidence, scheduler persistence, exact Adam rollback, candidate commit and
failure paths, held-out cohort construction, and validation/checkpoint/sync order.
Shell syntax and a dry-run configuration resolution are also checked. These do
not establish NPU stability or positive speedup; those require the actual run.
