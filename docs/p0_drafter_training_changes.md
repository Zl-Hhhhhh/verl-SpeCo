# P0 Drafter Training Changes

## Scope

This document describes the P0 changes in the working tree relative to the
currently staged legacy implementation. It intentionally excludes the staged
changes, which remain the previous strategy.

P0 has four goals:

1. Preserve real speculative draft/verify outcomes alongside each online
   training sample.
2. Remove response-position and batch-order bias from hidden-state collection.
3. Replace the biased restricted-vocabulary CE default with exact,
   memory-bounded full-vocabulary CE.
4. Keep P0 runs isolated from legacy checkpoints and W&B runs.

Trace recovery additionally instruments every transport boundary and refuses a
drafter update when the collected batch is incomplete.

The change applies to the online old-logprob collection path. Offline rollout
and the legacy staged strategy are not changed by this document.

## Sampling And Collection

### Real speculative trace collection

The vLLM EngineCore scheduler is patched at the actual verification boundary.
For every scheduled speculative verify step, P0 records a request-aligned
event containing:

- response offset before the event;
- proposed draft token ids and their count;
- verifier-emitted token ids;
- accepted prefix token ids and accepted count;
- whether the verifier emitted a replacement token or an all-accepted bonus
  token;
- the first rejected draft token and its verifier replacement, when present;
- token ids eventually emitted by the serving frontend.

With async scheduling, scheduler-visible draft ids are `-1` placeholders.
P0 therefore forces the NPU model runner's existing `[batch, k]` draft-id CPU
copy and attaches those real ids to the corresponding `ModelRunnerOutput`.
The scheduler uses them for the event instead of recording placeholders.

Events are transported from EngineCore to the frontend through the existing
per-output transfer payload. As a fail-closed fallback for vLLM variants that
consume connector metadata before SPECO can read it, EngineCore also writes the
complete request trace once at request completion to a request-scoped file in
`/dev/shm/verl_speco_vllm_traces`; the HTTP server atomically reads and removes
it. Both paths normalize vLLM's internal
`<external_request_id>-<8 hex chars>` id back to the external request id. The
result is placed in `extra_fields["speculative_trace"]`. Agent-loop requests
retain the full multi-turn history and annotate each event with
`generation_index`.

When old-logprob features are collected, the trace is stored with the drafter
sample as `speculative_trace`. This allows later objectives to align a sampled
window with real verifier outcomes rather than reconstructing acceptance from
aggregate metrics.

The EngineCore also attaches request-level stage counters independently of
whether an event was built. They cover scheduled drafts, request/index lookup,
real draft-id availability, event construction, output attachment,
msgpack/frontend receipt, request-completion spooling, HTTP return, and
agent-loop receipt. These counters are carried as
`speculative_trace_diagnostics`, making the first stage that loses trace data
visible in W&B.

### Variable-length windows

Previously, a response had to contain the entire fixed hidden-state window
(`hidden_state_window_tokens_per_sample + 1`, normally 513 rows) or it was not
collected. P0 admits every nonempty response and uses:

```
hidden_rows = min(configured_train_rows + 1, response_length)
```

`hidden_position_mask` marks only the populated rows. This avoids treating
zero-filled padding positions as hidden-state locations, and lets a response
shorter than 513 tokens train the drafter with all of its available response
tokens.

### Stratified response-position selection

The default window mode is now:

```
hidden_state_window_mode: stratified
hidden_state_window_strata_weights: [0.34, 0.33, 0.33]
```

For a response longer than one window, eligible window offsets are divided into
front, middle, and tail thirds. Selected samples are allocated across strata
to track the configured weights, then an offset within the assigned third is
chosen deterministically from the sample hash. This gives near one-third
coverage to each region for a reasonably sized collection instead of always
training from the response prefix.

`front` and `random` remain supported modes. A sequence that fits entirely in
one window is reported as the `full` stratum. OmegaConf list and mapping forms
of the weight configuration are accepted and normalized.

### Collection-cap fairness

Candidates are first assigned a deterministic hash priority. They are then
considered in ascending priority before the per-replica sample/token caps are
applied. This is hash-priority sampling without replacement, equivalent in
intent to reservoir selection but deterministic for a given step and batch.

Consequently, an early batch element no longer wins merely because it appears
before a later element. Existing owner capacity and token-budget checks still
determine whether a candidate can be assigned to a replica.

The collection plan is built from the original rollout batch, before the actor
policy sub-batch is selected for old-logprob computation. Collected trace and
feature metadata therefore keep their original sample indexing.

Sample-aligned hidden-state `ObjectRef` and metadata lists are represented as
`NonTensorStack` values before data-parallel outputs are concatenated. Treating
them as scalar non-tensor data caused concatenation to retain only the first DP
shard. Stable per-cycle sample ids now verify the planned, dispatched, and
consumed cardinalities across this boundary.

## Loss Change

### Previous restricted CE

The previous default `restricted_ce` used a candidate vocabulary `S` composed
of active target ids and ids present in the batch. Its per-token objective was:

```
L_restricted = -z[y] + log(sum(exp(z[v]) for v in S))
```

Tokens outside `S` were excluded from the softmax normalizer and had zero CE
gradient. This is not equivalent to language-model cross entropy over the
target vocabulary `V`.

### P0 exact chunked CE

P0 introduces `chunked_ce` for both DFlash and DSpark. It computes:

```
L_exact = -z[y] + log(sum(exp(z[v]) for v in V))
```

The active token rows are processed in chunks of
`*_ce_token_chunk_size` (default 64). Each chunk materializes full-vocabulary
logits, computes exact `torch.nn.functional.cross_entropy`, and uses
non-reentrant activation checkpointing during backward. It therefore has the
same loss and hidden-state gradient as non-chunked `full_vocab` CE, while not
retaining logits for every active token at once.

For DSpark, the exact logits include the Markov bias, matching the previous
full-vocabulary branch. DFlash uses the target LM head projection directly.
The optional `sampled_ce` mode remains an explicitly approximate alternative;
the `restricted_ce` name is retained only as a deprecated compatibility alias
to `chunked_ce`.

## Distributed Training Safety

Online variable-length collection can leave a single drafter rank without a
trainable batch. The previous readiness gate checked only the data-parallel
dimension, even though FSDP parameter collectives also require every
sequence-parallel peer to participate. A missing SP peer could therefore let
the remaining ranks enter forward/backward and wait until HCCL timed out.

The readiness gate now performs a `MIN` reduction over SP and then DP, so a
missing batch or a recoverable local batch-preparation failure makes the whole
mesh skip that microstep before any FSDP collective. W&B reports
`drafter/mesh_batch_skips` and `drafter/batch_prepare_failures`; both should
remain zero in a healthy run.

Block-drafter diagnostics previously added 18 small HCCL reductions per
microstep (15 scalar metrics and three per-position vectors). They are now
packed into one fixed-length reduction, preserving the existing metric sums
while avoiding interleaving hundreds of extra collectives with FSDP during a
20-step online update.

Managed DFlash and DSpark experiment runners now default to `chunked_ce`.
The library-level YAML default remains `full_vocab`; P0 adds the chunk-size
configuration to the base config but does not silently change direct library
users.

Because exact CE requires all target-LM-head rows, P0 disables
`target_lm_head_row_restricted_sync` in the managed runners. The old
row-selection helper returns no restricted row set for block-drafter CE, so a
partial synchronized LM head cannot accidentally alter the objective.

### DSpark total objective

The DSpark mixture and its default weights are unchanged:

```
L = 0.1 * L_CE + 0.9 * L_L1
```

`L_L1` is the weighted mean L1 distance between the drafter and target
full-vocabulary distributions at each valid block position. It is parallel
per position, not a recursive rollout loss. The P0 CE change changes only the
CE branch from restricted to exact normalization; it does not change the L1
formula, block size, anchor sampling, response mask, or positional decay.

## Metrics

P0 adds these training metrics:

| Metric | Meaning |
| --- | --- |
| `drafter/oldlogprob_candidate_samples` | Nonempty responses eligible before sampling/caps. |
| `drafter/oldlogprob_planned_samples` | Samples assigned to a collection owner. |
| `drafter/oldlogprob_materialized_samples` | Planned samples with usable hidden rows. |
| `drafter/oldlogprob_dispatched_samples` | Materialized samples sent to drafter workers. |
| `drafter/oldlogprob_consumed_samples` | Samples actually stored by drafter workers. |
| `drafter/oldlogprob_{materialization,consumption}_ratio` | Coverage across the two collection transport boundaries. |
| `drafter/oldlogprob_collected_rows` | Actual variable hidden-state rows collected. |
| `drafter/window_samples_{front,middle,tail,full}` | Selected-window distribution. |
| `drafter/materialized_window_samples_{front,middle,tail,full}` | Strata after hidden-state materialization. |
| `drafter/collection_drop/*` | Drop counts by sampling, capacity, position, and worker reason. |
| `drafter/spec_trace_samples` | Collected samples containing at least one verify event. |
| `drafter/spec_trace_verify_steps` | Number of traced speculative verification events. |
| `drafter/spec_trace_draft_tokens` | Proposed draft-token count in the trace. |
| `drafter/spec_trace_accepted_tokens` | Accepted prefix-token count in the trace. |
| `drafter/spec_trace_rejected_tokens` | `draft_tokens - accepted_tokens`. |
| `drafter/spec_trace_sample_coverage` | Trace-bearing collected samples divided by collected samples. |
| `drafter/spec_trace_token_acceptance` | Accepted draft tokens divided by proposed draft tokens. |
| `drafter/spec_trace_stage/*` | EngineCore-to-agent-loop trace boundary counters. |
| `drafter/training_guard_*` | Whether collection integrity allowed the drafter update and a numeric failure reason. |
| `drafter/train_{loss,vloss,ploss,grad_norm}` | Actual optimizer-step scalar diagnostics. |
| `drafter/train_{sample_draws,unique_samples,sample_reuse_ratio}` | Reuse of collected samples across drafter batches. |
| `drafter/fixed_eval/*` | Stable prompt-hash cohort trace quality, tagged with the current actor snapshot step. |

These make it possible to distinguish missing traces, a collection cap,
position imbalance, and genuine acceptance degradation.

By default, a materialization, worker-consumption, or trace coverage below
0.95 skips the drafter update while allowing the actor update to continue.
`collection_integrity_fail_action: raise` turns this into a hard debugging
failure. The fixed evaluation cohort uses a stable prompt hash and seed for
selection; it does not freeze the actor, so `actor_snapshot_step` must be used
when comparing the cohort over time.

## Explicit Non-Goals

P0 records rejected proposals but does not yet use them in the training loss.
In particular, it does not add a reverse-KL term such as
`KL(p_drafter || p_target)` on rejected proposal states, nor does it reweight
CE/L1 by accepted versus rejected status. Trace data is therefore an
observability and future-objective input, not a current RKL implementation.

Hard-example sampling is also intentionally unchanged. The current P0 loss
and window logic use the normal collected samples only.

## Run Identity And Compatibility

Online managed runs append `__p0` by default to auto-generated experiment,
checkpoint, log, and W&B run names. Set `SPECO_ONLINE_STRATEGY_TAG` to a short
safe custom label to identify a P0 variant. Offline run names are unchanged,
and an explicitly supplied `EXPERIMENT_ID` or `WANDB_RUN_NAME` remains
authoritative.

The runner no longer falls back to a legacy checkpoint directory when the P0
directory is empty. This prevents a P0 launch from resuming a checkpoint whose
weights and training semantics belong to the old strategy.

## Validation Added With P0

The working-tree tests cover:

- short responses retaining their available window rows;
- hash-priority selection before a collection cap;
- front/middle/tail allocation and OmegaConf weight parsing;
- equality of chunked and full-vocabulary DSpark CE loss and hidden-state
  gradient;
- the deprecated `restricted_ce` compatibility mapping;
- real vLLM verify-event fields and EngineCore-to-frontend transport;
- real vLLM msgpack encode/decode preservation of trace payloads;
- async NPU worker replacement of scheduler `-1` placeholders with real draft
  token ids;
- internal/external vLLM request-id normalization and one-shot trace-spool
  consumption;
- multi-turn agent-loop trace accumulation;
- four-way DP concatenation preserving every sample-aligned ObjectRef;
- collection guards for trace loss and worker-consumption loss;
- P0 run-name/checkpoint isolation from legacy checkpoints.

The final one-step NPU smoke run
`trace_recovery_smoke_r10_20260817` collected trace for 4/4 selected samples:
146 verify steps, 1,022 proposed draft tokens, 367 accepted tokens, and 655
rejected tokens (`token_acceptance=0.3591`). Collection materialization and
worker consumption were both 1.0, the integrity guard passed, and the drafter
completed one optimizer step (`loss=1.2919`, `grad_norm=2.5787`) before
publishing the update.
