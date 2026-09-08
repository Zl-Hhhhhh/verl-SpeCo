# True SWA-512 rollout timing and top-k distillation assessment

Analysis date: 2026-09-05 UTC. This is an analysis report; training and runtime code were not changed.

## Runs and measurement scope

- Current run: `qwen3_8b_dspark_online_w16_101step_b64__p2_02_swa512`, W&B `2a6b95c4b0fc`.
- Managed run: `dspark_p2_02_swa512_101step_k3dump_r2_20260904`.
- Completed all 101 actor steps, exit status 0, from 2026-09-04 14:50:31 to 2026-09-05 03:18:44 UTC, including startup and shutdown.
- Current log: `outputs/validation_logs/2026-09-04/dspark_vllm/qwen3_8b_dspark_online_w16_101step_b64__p2_02_swa512.log`.
- Historical full control: `qwen3_8b_dspark_online_w16_100step_b64__p2_02`, combining its 2026-08-25 and 2026-08-26 logs. For repeated step 20, the later record wins.
- K3 evidence: `outputs/drafter_analysis/qwen3_8b_dspark_online_w16_101step_b64__p2_02_swa512/2a6b95c4b0fc/k3_logprobs/` contains 1,600 files. Token-level analysis below reads all 80 files at each of actor steps 5, 50, and 100. These are optimizer observations, not independent evaluation samples.

Comparisons use actor steps 1-100. Metrics are arithmetic step means except generation ms/token, which is total generation seconds divided by total response tokens, multiplied by 1,000. Response token totals are reconstructed from the logged float32 response means and 320 responses/step, so they are approximate.

The full control is historical, has a resume, and does not share a frozen actor or identical generated trajectories with the new run. These measurements establish a regression of the overall configuration; they do not identify the isolated causal effect of SWA.

## Completed-run results

| Metric, steps 1-100 | Historical full | True SWA-512 | Change |
| --- | ---: | ---: | ---: |
| Generation seconds/step | 267.090 | 337.500 | +26.36% |
| Generation ms/output token, aggregate | 0.149723 | 0.190219 | +27.05% |
| Aggregate output tokens/second | 6678.99 | 5257.11 | -21.29% |
| Mean acceptance length, including bonus | 3.7754 | 3.5602 | -5.70% |
| Mean response length | 5574.66 | 5544.61 | -0.54% |
| Total step seconds/step | 362.781 | 439.994 | +21.28% |
| Old-logprob seconds/step | 19.441 | 21.297 | +1.856 seconds |
| Actor-update seconds/step | 67.946 | 69.274 | +1.328 seconds |
| Drafter-stage seconds/step | 2.798 | 6.071 | +3.273 seconds |

The extra generation time is 7,041 seconds, approximately 117 minutes. It accounts for 91.2% of the extra total step time. K3 dump, larger training collection windows, and dual-best saves cannot be the main direct explanation for the generation regression: their work is primarily outside `timing_s/gen`.

| Actor steps | Full acceptance | SWA acceptance | Full aggregate gen ms/token | SWA aggregate gen ms/token |
| --- | ---: | ---: | ---: | ---: |
| 1-20 | 3.4075 | 3.2699 | 0.163940 | 0.188832 |
| 21-40 | 3.6878 | 3.4967 | 0.161077 | 0.203417 |
| 41-60 | 3.9628 | 3.6211 | 0.134335 | 0.197541 |
| 61-80 | 3.9542 | 3.6778 | 0.134294 | 0.190987 |
| 81-100 | 3.8646 | 3.7355 | 0.150880 | 0.167318 |

The final 20 steps improve relative to the middle of the run, but still lag the full control in acceptance and output-normalized generation speed. Neither run provides a frozen-actor estimate of the benefit attributable to online drafter updates.

## Why the rollout is slower

### The attention crop is active

The successful launch logs validate the adapter and show grouped crop execution with `runtime_window=519`, `block_size=128`, and `groups=1`. Public context window 512 excludes the 7 draft query tokens. Page rounding can retain up to 127 extra context tokens. This run is different from the historical SWA overlays that did not actually activate the adapter.

The first four actor steps, before any drafter update, already show acceptance 3.2116 versus 3.3444 and aggregate generation ms/token 0.185916 versus 0.164041. An initial checkpoint/context mismatch and runtime differences are plausible causes; drafter training and K3 dump cannot directly cause this pre-update gap.

### Added metadata work is on the critical path

The installed Ascend runtime currently executes:

1. `llm_base_proposer.py:874`: the generic `SlidingWindowAdapter.apply()` crops the common block table and updates sequence lengths.
2. `verl_speco/integration/vllm_runtime.py:1675`: the grouped DSpark patch independently performs window indexing, gather, masking, and copy for native draft KV groups.
3. `llm_base_proposer.py:2097`: the DSpark metadata builder uses the per-group table, replacing the generic table for each draft group.
4. `attention/attention_v1.py:321`: parallel drafting selects the device sequence-length tensor; line 333 calls `.tolist()`, requiring host visibility of its contents.

Both crop paths contain small eager tensor operations. The metadata synchronization also limits overlap of these operations with subsequent attention execution. The DSpark proposer explicitly forces `use_cuda_graph=False` at `spec_decode/dspark_proposer.py:77`; simply changing the experiment's `enforce_eager` flag is not sufficient to graph this implementation.

This identifies concrete overhead candidates, not a measured attribution of all extra seconds. Removing redundant common-table materialization while keeping the window arithmetic, fusing per-group indexing/masking, and reusing preallocated index grids are reasonable profiling targets. A primary table can only be reused when it addresses the correct KV group.

### Only part of drafting becomes cheaper

The noncausal draft attention kernel still uses `sparse_mode=0`, now on the cropped KV table (`attention_v1.py:1330`). This is full attention over the selected window, preserving bidirectional draft-block attention.

Actor verification still uses its full context. Draft MLPs, projections, vocabulary projection, tensor-parallel communication, and the seven sequential Markov-bias/argmax operations are not shortened by SWA. The native proposal loop is at `llm_base_proposer.py:1133`. Thus a smaller attention matrix does not imply a proportionally faster complete draft/verify cycle.

Acceptance falling from 3.7754 to 3.5602 implies roughly 6% more verification work per output token under a fixed-cost approximation. That is much smaller than the measured 27% output-normalized slowdown. As a diagnostic proxy, estimating request verification counts by `sum(output_tokens_step / acceptance_step)` gives +5.25% total estimated calls, while `generation_time / estimated_calls` rises about 20.06%. These are not engine-measured verification counts or kernel latencies: batching, tails, and aggregated acceptance can change the relation. They nevertheless show why acceptance alone is an insufficient explanation.

### Training has additional mismatches

The current training and collection span is 1,024 tokens, with local attention 512 and the original 16K collection budget. Mean collected windows/cycle fell from 80 to 62.7. The logged training-unique metric fell from 19.6 to 15.75, and mean reuse rose from 2.0% to 23.75%; the unique metric must not be interpreted as a global deduplicated count across all workers.

Training anchors near the beginning of each collected span see less than a full window. The logged mean local context is about 384 tokens, whereas long-prefix rollout queries usually see at least 512 context tokens after page rounding. This is a training/rollout exposure difference. It may limit adaptation, but it does not explain the pre-update runtime slowdown.

## What the K3 dumps show

Current objective uses response K3 weight 1, rejected K3 weight 0.5, rejected block-position decay 0.8, and no asymmetric or dynamic weighting. Its per-token core is:

```text
delta = clamp(log p_actor(token) - log q_drafter(token), -20, 20)
loss = min(exp(delta) - delta - 1, 10)
```

The outer hard cap gives zero K3 gradient above 10. The inner log-ratio clamp also gives zero gradient when the raw delta lies outside [-20, 20]. Removing only the outer cap would not fix the second source of saturation.

| Dump metric | Actor step 5 | Actor step 100 |
| --- | ---: | ---: |
| Response K3 mean | 2.9902 | 1.9790 |
| Rejected K3 mean | 4.3673 | 5.9606 |
| Response hard-cap fraction | 26.22% | 16.56% |
| Rejected hard-cap fraction, weighted | 28.10% | 49.56% |
| First-rejection hard-cap fraction, weighted | 27.36% | 62.59% |
| Rejected raw delta below -20, weighted | 6.84% | 33.13% |

At step 100 specifically:

- Correct target replacements at the first rejection: 53.19% capped; mean actor probability 0.9214 versus drafter probability 0.2192.
- Incorrect proposed tokens at the first rejection: 62.59% capped; mean actor probability 0.03595 versus drafter probability 0.4628.
- Of the full rejected stream, 45.03 percentage points are negative-tail caps and 4.53 points positive-tail caps. About 90.9% of capped rejected weight is on the negative side.
- First-rejection raw delta is below -20 for 43.87% of its weight.
- Response top-1 accuracy is 73.42%, while top-5 is 91.46%. These are training diagnostics on selected response positions, not rollout acceptance or teacher/student top-k overlap.

The useful corrective pair is therefore often inactive: the wrong candidate is not pushed down by its capped K3, while the correct replacement is not pushed up by its capped K3. Other noncapped examples can still affect the same logits through shared parameters and softmax normalization; these observations do not mean the entire model has zero gradients.

## Paper definition and applicability

The requested paper is [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/pdf/2604.13016). Section 2.2, Eq. (5), selects the student's top-k tokens and renormalizes both distributions within that set before computing reverse KL. Its default k is 16. Section 4.2 compares student-top-k, intersection, and symmetric-difference supervision; the reported outcomes concern reasoning benchmarks, not DSpark acceptance or Ascend runtime speed.

Using this repository's notation, p is the actor and q is the drafter:

```text
S = TopK(q, k)
q_bar(v) = q(v) / sum(q(u) for u in S)
p_bar(v) = p(v) / sum(p(u) for u in S)
L_topk = sum(q_bar(v) * (log q_bar(v) - log p_bar(v)) for v in S)
```

This is vocabulary top-k at each supervised position, not the k highest-loss positions and not a new acceptance/rejection weight schedule.

There is also an implementation distinction. The [official actor implementation](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py#L555) forms rewards from raw teacher/student log-probability differences and normalized student weights; its [policy loss](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/trainer/ppo/core_algos.py#L1109) applies a PPO-style surrogate to stored rewards. This is not identical to directly differentiating the renormalized Eq. (5) loss. An experiment must name which objective it implements.

### Why a distribution objective is worth testing

The current accepted calculation already projects actor and drafter states to vocabulary logits, then gathers one target token (`dspark_trainer_backend.py:809`). A multi-candidate loss can use that already available information to supervise the relative ranking of alternatives. A stable log-softmax KL does not require the current `exp(log p - log q)` and hard loss cap.

However, current runtime proposals are greedy: `dspark_proposer.py:56` rejects probabilistic draft sampling, and `llm_base_proposer.py:1151` selects argmax after Markov correction. The teacher rollout is stochastic, but the proposal is not drawn from the drafter softmax being trained. Response and rejection filtering further change the token distribution. Consequently, the sampled-KL unbiasedness argument under samples from the current student's softmax does not apply directly here. Replacing K3 changes the effective optimization objective, not just the variance of an equivalent estimator.

Candidate ranking and actor probability of the actual greedy proposal are especially relevant. A decrease in a softmax KL diagnostic need not change the argmax, and a smaller reasoning-model entropy is not itself a speculative-decoding benefit.

### Why pure renormalized student-top-k can miss the error

Consider a three-token example:

```text
            A       B       C
drafter q   0.54    0.36    0.10
actor p     0.06    0.04    0.90
```

Student top-2 selects A and B. Both restricted distributions become [0.6, 0.4], so the restricted KL is exactly zero despite the actor strongly preferring C. This is a mathematical limitation of the subset-normalized objective, not an observed probability vector from this run.

Measure teacher and drafter mass coverage on the selected set before trusting this approximation. Do not assume the paper's overlap-mass observation holds for a compressed SWA DSpark model. An overlap-only loss is a lower-priority option because it can omit incorrect student-only candidates and correct teacher-only alternatives.

An additional experiment could use the union of teacher-top-16 and student-top-16, plus a single OTHER category containing all omitted probability mass. KL on that coarsened full distribution retains a signal about mass moving into/out of the selected set. It still cannot distinguish individual tokens inside OTHER and still needs empirical validation. This is a proposed adaptation for this system, not the paper's default loss.

### Response and rejected branches have different data requirements

| Branch | Available supervision | Top-k implementation implication |
| --- | --- | --- |
| Response/accepted/replacement | Actor final hidden state and synchronized LM head; full logits can already be reconstructed | First implementation target; reuse chunked logits and preserve Markov previous-token alignment |
| First rejected proposal | Currently only the actor logprob of that proposal is carried in the trace | A general top-k loss needs candidate probabilities or the corresponding actor state; shared-prefix cases may permit carefully checked reuse |
| Later rejected suffix | Teacher scalar logprobs on the actual recursive proposed branch | Must capture/recompute teacher multi-candidate scores on that same proposed prefix; true-response hidden states are not interchangeable |

The scalar capture is at `vllm_runtime.py:2118`; rejected loss consumption is at `dspark_trainer_backend.py:991`. Existing K3 dumps cannot reconstruct teacher top-k, support coverage, entropy, or the missing candidate probabilities. These need new collection/diagnostics.

For runtime-captured student top-k, candidate IDs are tied to the rollout drafter version. Reusing them through five optimizer updates gives a frozen support approximation, not dynamically recomputed current-student top-k. Store provenance and define the intended behavior explicitly. Avoid exporting full-vocabulary teacher logits for every verification event. Top-k capture is additional work on an already slow rollout path and should be limited to selected training events or moved to a separate teacher-scoring stage where practical.

## Recommended order

1. **Separate runtime cost from quality.** Run a short, repeated frozen actor/drafter comparison on the same DAPO cohort and current code: full, true SWA-512, and optionally SWA-1024. Use sufficiently long responses to exercise the window. Record actual verify counts, accepted tokens, response tokens, total generation time, and profiler time for metadata, draft attention, other draft computation, and actor verification. A 128-prompt cohort with repeats is more informative here than another 101-step training run.
2. **Test response distribution supervision first.** At fixed attention settings and a fixed actor snapshot, compare the current dual K3 objective with response student-top-16 reverse KL plus the existing rejected K3. Preserve sampling, stream weights, position decay, and optimizer-step budget. Also compare the proposed union-plus-OTHER variant if selected-set mass coverage is poor. Loss coefficients need scale calibration; the numeric value 1 is not automatically an equivalent gradient budget across objectives.
3. **Expand to rejected distribution supervision after branch alignment is verified.** Add correct-prefix multi-candidate teacher information, starting with first-rejection positions. Compare before extending to all unverified suffix positions. Use actual proposed previous tokens for the Markov head, keep actor-version provenance, and audit selected-set coverage and proposal rank.

Top-k is a credible way to address ineffective capped supervision and merits higher priority than more unstructured accept/reject weight tuning. It cannot directly remove the measured runtime metadata overhead or restore context that SWA has removed. The performance and loss experiments should therefore be evaluated separately, with fresh-rollout acceptance and output-normalized generation time deciding whether a training improvement is useful.
