# DSpark top-16 options and half-node recipe

> Historical record: the half-node and P4 wrapper scripts were removed on 2026-09-09. Use `tools/run_dspark_vllm_experiment.sh` with explicit device, dataset and attention settings for new runs; wrapper commands below describe the original experiments.

Prepared on 2026-09-05. No 101-step training job is launched by this change.
Reference: the completed P2_02 true-SWA-512 run, W&B `2a6b95c4b0fc`.
See `p2_02_true_swa512_runtime_topk_assessment_2026-09-05.md` for the timing and K3-tail evidence.

## Independent loss options

The ordinary DSpark launcher keeps K3 as the default on both streams.

| Environment variable | Hydra training key | Default |
| --- | --- | --- |
| `DSPARK_RESPONSE_LOSS` | `dspark_response_loss` | `k3` |
| `DSPARK_REJECTED_LOSS` | `dspark_rejected_loss` | `k3` |
| `DSPARK_TOPK` | `dspark_topk` | `16` |

Both selectors accept `k3` or `topk`. `dspark_loss_mode=chunked_ce` remains the
logit/CE computation strategy; it is NOT the stream objective selector.

For student-selected support S, the new objective is:

```text
log_q_S = log_softmax(student_logits[S])
log_p_S = log_softmax(teacher_logits[S])     # stop-gradient
L_topk  = sum(exp(log_q_S) * (log_q_S - log_p_S))
```

This is the renormalized student-support reverse KL discussed around Eq. 5 of
[the paper](https://arxiv.org/pdf/2604.13016), differentiated directly. It is not
a reproduction of that project's PPO/reward-based optimizer.

- Response: select the CURRENT training student's top 16, including the Markov
  correction, and score the current actor's final hidden state on the same IDs.
  Supervised positions remain accepted responses, the first replacement token,
  and the existing all-accepted bonus continuation anchors.
- Rejected: select the ROLLOUT student's top 16 AFTER recursive Markov correction.
  The unmodified actor verification forward scores those IDs on the actual
  proposed branch. Candidate IDs and teacher scores remain fixed during that
  collection/training cycle; current training-student probabilities are recomputed.
  This is deliberately not described as current-student top-k after every update.
- Rejected positions and their existing absolute block-position decay are unchanged.
  Later rejected suffixes use their actual previous proposal token, not replacement
  tokens from the accepted response trajectory.
- Missing/misaligned candidates, stale proposal IDs, duplicate/out-of-vocabulary
  support, and non-finite candidate scores fail explicitly. There is no silent K3
  fallback. Bonus-only anchors do not need rejected candidate metadata.
- Existing `dspark_accepted_k3_alpha` and `dspark_rejected_k3_alpha` multiply the
  selected stream objective, despite their historical names. Setting one to zero
  disables that stream term. CE/L1/FKL coefficients retain their meanings.
- Stream weights, dynamic accept/reject balancing, joint token normalization,
  and rejected-position decay still apply. K3 clamps/asymmetric switches affect
  K3 terms only, never top-k KL.

The new diagnostics are `dspark/response_topk_loss`, `dspark/rejected_topk_loss`,
`*_topk_student_mass`, `*_topk_teacher_mass`, and `*_topk_enabled`, under the existing
training-metric prefix. Existing K3 diagnostics and optional sampled-token dumps
remain available as diagnostics, even when K3 no longer contributes to the loss.
Top-k loss is NOT substituted into the old K3 metric names.

Coverage matters: q=(.54,.36,.10) and p=(.06,.04,.90) have zero top-2 subset KL but
very different full distributions. A falling top-k loss is insufficient evidence
of improving acceptance when teacher mass on the support is low.

## Rollout scope and cost

Rejected candidate capture is implemented for the installed vLLM-Ascend v1 DSpark
greedy proposer and its padded drafter batch path. It does not change actor
attention, actor verification, sampled output tokens, or DSpark proposals.

Response-only top-k needs no additional rollout candidate capture. Rejected top-k
adds a vocabulary top-k selection, alignment checks, teacher candidate gathering,
and CPU trace transfer. Currently capture follows the rollout trace path, not just
the final 32 selected training anchors. It can increase generation time and host
memory; it is an optimization-objective experiment, not a promised rollout speedup.
Only IDs and 16 teacher log probabilities are retained per draft position, not
full vocabulary logits. No extra actor forward is introduced.

## Physical cards and equivalent budgets

`npu-smi info -m` identifies TWO logical devices per physical card on this host:

```text
physical 4 -> logical 8,9
physical 5 -> logical 10,11
physical 6 -> logical 12,13
physical 7 -> logical 14,15
```

Use `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`, not `4,5,6,7`.
The latter would use physical cards 2 and 3, which are outside the available half.

The prepared profile is `tools/run_dspark_topk_half_node.sh`:

| Quantity | Reference, whole node | Prepared, half node |
| --- | ---: | ---: |
| Logical devices / drafter workers | 16 | 8 |
| Rollout TP / training SP | 4 / 4 | 4 / 4 |
| Rollout replicas | 4 | 2 |
| Global prompts x responses | 64 x 5 | 64 x 5 |
| Actor PPO minibatch (before n=5) | 16 | 16 |
| Actor microbatch per device | 10 | 10 |
| Drafter `batch_size_per_gpu`, per TP replica in this path | 4 | 8 |
| Global distinct-sample batch budget per draft update | 16 | 16 |
| Collection sample cap per replica | 20 | 40 |
| Collection feature-token budget per replica | 16384 | 32768 |
| Global collection caps, samples / feature tokens | 80 / 65536 | 80 / 65536 |
| Fixed evaluation samples per replica | 4 | 8 |
| Global fixed evaluation cohort | 16 | 16 |
| Collection / training interval | 5 / 5 | 5 / 5 |
| Drafter optimizer steps per cycle / LR | 5 / 1e-5 | 5 / 1e-5 |
| Drafter attention window / feature training span | 512 / 1024 | 512 / 1024 |
| Anchors per sample / draft block size | 32 / 7 | 32 / 7 |

The feature budget also counts the extra alignment row. Therefore actual collected
samples remain length-dependent, typically around 60-64 globally for near-1024-token
windows, not automatically 80. Integer per-replica caps and request partitioning can
change the exact count slightly even with an unchanged global budget.

The profile preserves model sizes, maximum response length 8192, actor LR, global
PPO minibatch, drafter optimizer count, and cold-start weights. Actor accumulation
increases automatically with the larger per-replica share. Drafter local batch size
doubles, so its activation memory rises; TP=4 and actor parameter/optimizer offload
remain enabled to retain memory headroom. Do not halve the global prompt batch or
shorten outputs merely to match the old seconds/step: those change the experiment.

This is budget equivalence, not bitwise equality or a guarantee of identical
acceptance. Data partitioning, stochastic trajectories, and reductions differ.
Expect substantially longer wall-clock time, with roughly twice the device-bound
work per device; do not compare raw tokens/second to the whole-node run as a
loss-function effect. Use a half-node K3 control with the same instrumentation.

101 steps, `SAVE_FREQ=10`, `SAVE_OFFSET=9`, dual `best_acceptance` / `best_throughput`,
checkpoint retention, and K3 dump remain enabled. The half-node script disables
global `ray stop --force` at both startup and cleanup and uses a fresh local Ray
instance; it does not bypass the existing host lock or kill other jobs.

## Prepared comparisons

Preview commands only; these do not start Ray or training:

```bash
DRY_RUN=1 DSPARK_TOPK_ARM=k3 bash tools/run_dspark_topk_half_node.sh
DRY_RUN=1 DSPARK_TOPK_ARM=response bash tools/run_dspark_topk_half_node.sh
DRY_RUN=1 DSPARK_TOPK_ARM=rejected bash tools/run_dspark_topk_half_node.sh
DRY_RUN=1 DSPARK_TOPK_ARM=both bash tools/run_dspark_topk_half_node.sh
```

The default arm is `response`. For eventual experiments, first compare the
half-node K3 control and response-only top-16. Then test rejected-only top-16 to
isolate its optimization benefit and collection overhead; combine both only after
the individual comparisons. Maintain the same seed/init, sample budget, dump
settings, and hardware allocation. Monitor acceptance, effective tokens/second,
generation seconds, top-k teacher coverage, gradient norm, and per-block-position
acceptance. Do not infer a gain from reduced training loss alone.

CPU tests cover all four loss combinations, weighted objective reconstruction,
gradient/teacher-detach behavior, candidate capture after Markov correction, request
reordering and prefix trimming, trace/crop/batch transport, and half-node dry runs.
The explicit NPU smoke uses logical device 8 only and checks top-16 forward/backward
and post-Markov capture, plus actual vLLM-Ascend patch installation/idempotence
using the worker's normal runtime initialization order. A full distributed Qwen3 engine/training smoke is still
required before a long run; the tensor smoke is not an 8B end-to-end validation.

Validation completed: focused regression suite 159 passed / 3 skipped (including
the opt-in NPU test); explicit logical-device-8 NPU smoke 1 passed; the added
metric-export assertion also passed. Shell syntax, Python compilation, and
`git diff --check` passed. A broader legacy launcher test invocation encountered
two pre-existing stale expectations (`p2_06` / 100 steps / default cold start,
versus the existing `p3_01` / 101 steps / no default cold start); those unrelated
defaults and tests were left unchanged.
