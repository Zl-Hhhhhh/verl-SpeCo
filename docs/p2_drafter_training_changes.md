# SpeCo P2 drafter training

P2 is the managed online DSpark strategy identified by the `__p2` experiment,
checkpoint, log, and W&B suffix. It keeps P1's real verify-event anchors,
unique-first sampling, and persistent fixed cohort, but replaces the training
objective and consumes the entire rejected draft branch.

## Verify trace payload

For every draft token proposed at a real vLLM verify step, the rollout worker
records its token ID and one scalar target-model log-probability:

```text
draft_token_ids[i]
draft_teacher_logprobs[i] = log P_target(draft_token_ids[i])
```

The current managed rollout uses temperature 1. P2 deliberately records the
canonical pre-top-k/top-p target distribution, so sampling truncation cannot
turn a valid teacher probability into zero. The payload is request-aligned
before crossing the engine/frontend boundary. Training is blocked when finite
teacher-logprob coverage is below 95%, rather than silently dropping the
rejected objective.

## Positive target-trajectory stream

For a verify event with `a` accepted tokens, positive supervision covers the
accepted prefix plus the target replacement position. Positions after the
first rejection are excluded from this stream because their draft prefix has
diverged from the target trajectory.

P2 uses the local Bernoulli forward KL on the target trajectory token as the
main term. For target probability `p` and drafter probability `q`:

```text
FKL_pos = p log(p/q) + (1-p) log((1-p)/(1-q))
```

The previous run showed CE increasing from about 2 to 4.34 and distribution L1
increasing from about 0.75 to 1.33 while the rejection term improved. P2
therefore sets CE's optimization weight to zero and retains L1 only as a small
full-distribution shape regularizer:

```text
alpha_FKL = 1.0
alpha_CE  = 0.0
alpha_L1  = 0.1
```

CE is still measured for diagnostics and top-k accuracy is still reported.

## Rejected draft stream

P2 replaces P1's first-rejection Bernoulli reverse KL with Draft-OPD's sampled
K3 estimator. It includes the first rejected token and every later token in the
same proposed draft block. For a sampled draft token `y`, let:

```text
delta = clamp(log P_target(y) - log P_drafter(y), -20, 20)
K3(y) = clamp(exp(delta) - delta - 1, max=10)
```

The expected K3 value under drafter samples estimates `KL(q || p)`. Rejected
offset `j` receives weight `0.8^j`. The rejected stream is normalized by its own
weighted token count before it is combined with the positive stream:

```text
L = 1.0 * mean(FKL_pos)
  + 0.1 * mean(L1_pos)
  + 1.0 * weighted_mean(K3_rejected)
```

The rejected branch uses the actual recursive draft prefix for the DSpark
Markov input: `[anchor, draft_0, ..., draft_(j-1)]`. It does not reuse actor
trajectory tokens after divergence.

## Managed settings

```text
DSPARK_CE_LOSS_ALPHA=0.0
DSPARK_ACCEPTED_FKL_ALPHA=1.0
DSPARK_L1_LOSS_ALPHA=0.1
DSPARK_L1_CHUNK_SIZE=32
DSPARK_REJECTED_K3_ALPHA=1.0
DSPARK_REJECTED_POSITION_DECAY=0.8
DSPARK_REJECTED_LOG_RATIO_CLAMP=20.0
DSPARK_REJECTED_LOSS_MAX_CLAMP=10.0
```

The current cadence is five drafter optimizer steps every three actor steps.

## Primary metrics

```text
dspark/accepted_fkl
dspark/accepted_q_prob
dspark/accepted_p_prob
dspark/accepted_log_ratio
dspark/rejected_k3
dspark/rejected_q_prob
dspark/rejected_p_prob
dspark/rejected_teacher_minus_draft_log_ratio
dspark/rejected_k3_clamp_ratio
dspark/rejected_k3_weighted_count
drafter/spec_trace_teacher_logprob_tokens
drafter/spec_trace_teacher_logprob_coverage
```

The fixed cohort exposes the same loss components under
`drafter/fixed_cohort/pre_*`, `post_*`, and `delta_*`.
