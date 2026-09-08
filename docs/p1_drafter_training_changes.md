# SpeCo P1 drafter training

P1 is the managed online DSpark policy identified by the `__p1` experiment,
checkpoint, log, and W&B suffix. Offline runs keep their existing names. The
library defaults remain backward compatible: trace-aligned anchors, rejection
RKL, fixed holdout evaluation, and unique-first sampling are enabled by the
managed online launcher rather than globally.

## 1. Trace-aligned anchors

DSpark no longer samples an arbitrary response position in P1. Every training
anchor is mapped from an actual vLLM verify event:

```text
absolute_anchor = prompt_len - 1 + event.response_start
window_anchor   = absolute_anchor - feature_start - crop_start
```

The anchor is the context token immediately before the verify event's first
draft token. Events outside the collected hidden-state window are discarded.
If an item has no valid event in its window, the item is dropped; P1 never
falls back to a random anchor.

When a response window contains more events than `dspark_num_anchors`, a stable
hash priority selects the retained events, then restores sequence order. This
avoids a front-of-window bias while keeping selection reproducible.

For multi-turn agent generation, `response_start` after the first generation is
local to that generation. Such events are intentionally ignored unless the
producer supplies `trajectory_response_start`; mapping a local offset into the
full trajectory would create a false anchor.

## 2. Accepted and rejected supervision

For a verify event with `a` accepted drafts out of `m` drafts:

- offsets `[0, a)` are the accepted draft prefix;
- offset `a` is the first rejection/replacement position when `a < m`;
- offsets `(a, m)` are an unverified recursive suffix after the branch diverged.

P1 applies exact full-vocabulary chunked CE and distribution L1 to the accepted
prefix and the target replacement position. The unverified suffix is masked
from CE and L1. If every draft was accepted, all `m` draft positions remain
positive supervision.

At the first rejection, let `r` be the draft token rejected by target
verification, `q = P_drafter(r)`, and `p = P_target(r)`. P1 adds a binary reverse
KL hard-negative term:

```text
RKL_reject = q log(q / p) + (1 - q) log((1 - q) / (1 - p))

L = 0.1 * L_CE + 0.9 * L_L1 + 1.0 * L_RKL_reject
```

This directly lowers excess probability on the actual rejected proposal while
retaining positive teacher supervision for the token used to continue the
target trajectory. It does not treat later, unverified drafts as rejected
tokens.

Managed defaults can be overridden with:

```text
DSPARK_TRACE_ALIGNED_ANCHORS=True
DSPARK_FIRST_REJECTION_RKL_ALPHA=1.0
DSPARK_L1_LOSS_ALPHA=0.9
DSPARK_LOSS_MODE=chunked_ce
```

## 3. Reuse policy

The managed online defaults are now:

```text
DRAFT_TRAIN_STEPS=5
DRAFT_COLLECT_INTERVAL_STEPS=3
DRAFT_TRAINING_INTERVAL_STEPS=3
DRAFT_BATCH_SIZE_PER_GPU=4
DRAFT_MAX_COLLECT_SAMPLES_PER_REPLICA=20
DRAFT_TRAIN_UNIQUE_SAMPLES_FIRST=true
DRAFT_LR=5e-6
```

Sampling first excludes sample IDs already used in the current five-step
cycle. Reuse begins only after the unique pool is exhausted. Four samples are
reserved for the fixed validation cohort during its first cycle, leaving 16
training samples for 20 draws (about 20% reuse). In later cycles all 20 newly
collected samples are trainable, so the expected reuse is zero when collection
and trace materialization both reach their caps.

Actual behavior is exposed by `drafter/train_sample_draws`,
`drafter/train_unique_samples`, and `drafter/train_sample_reuse_ratio`.

## 4. Persistent fixed validation

Each replica retains the first four samples whose collected feature window
contains a mappable verify event. The cohort's input IDs, hidden states, masks,
and trace are retained for the trainer lifetime and are never passed to the
optimizer. It is evaluated immediately before and after every five-step
drafter update.

The primary W&B metrics are:

```text
drafter/fixed_cohort/pre_loss
drafter/fixed_cohort/post_loss
drafter/fixed_cohort/delta_loss
drafter/fixed_cohort/pre_first_rejection_rkl
drafter/fixed_cohort/post_first_rejection_rkl
drafter/fixed_cohort/pre_top1_acc
drafter/fixed_cohort/post_top1_acc
```

The pre/post pair uses the same target LM head within one update cycle, so its
delta isolates the drafter update. Across RL cycles the actor and its LM head
can still change; use the paired delta for causal comparison and the absolute
curve as a drift-sensitive diagnostic.

## 5. Trace diagnostics

P1 adds the following DSpark metrics:

```text
dspark/first_rejection_rkl
dspark/first_rejection_q_prob
dspark/first_rejection_p_prob
dspark/first_rejection_log_ratio
dspark/trace_anchor_count
dspark/trace_accepted_position_count
dspark/trace_first_rejection_count
dspark/trace_unverified_suffix_count
dspark/trace_all_accepted_anchor_count
```

Collection-stage metrics remain available under `drafter/spec_trace_*`,
`drafter/spec_trace_stage/*`, and `drafter/oldlogprob_*`. Together these separate
trace transport failures from a loss that is present but ineffective.

## 6. Main files

- `verl_speco/backends/dspark_trainer_backend.py`: trace mapping, masks, and RKL.
- `verl_speco/trainer/base_trainer.py`: unique-first sampling and fixed holdout.
- `verl_speco/workers/speco_worker.py`: fixed-cohort pre/post evaluation.
- `verl_speco/trainer/speco_ray_trainer.py`: trace-stage and accept/reject metrics.
- `tools/speco_vllm_experiment_common.sh`: online P1 defaults and naming.
- `tools/run_dspark_vllm_experiment.sh`: DSpark P1 switches.
