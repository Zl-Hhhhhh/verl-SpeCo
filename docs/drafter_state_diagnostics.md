# Drafter state diagnostics

Set `SPECO_DRAFTER_STATE_DIAGNOSTICS_DIR` to opt in for a DSpark run. Pass the
same variable through Ray `runtime_env.env_vars`. With the variable unset,
the lifecycle wrappers do not inspect tensors or write files.

This probe addresses the reported DP0 rank 0 versus ranks 1–3 student
log-probability differences starting at actor step 20. The existing dump alone
does not prove identical hidden states or model runtime state, or establish a
cause for the step 70 acceptance collapse. Only DP0 enters the drafter checkpoint
save/export path, so it is a useful comparison point, not a proven cause.

## Evidence

- Per-rank `events.jsonl`: before/after checkpoint save, full export, activation,
  cleanup/offload, activation warmup release, target lm_head synchronization,
  publish snapshot, and every optimizer training forward.
- Exact SHA-256 fingerprints of local model parameter shards and target head
  weights, with dtype, device, shape, strides, placements, and mesh coordinates.
  These calls do not gather full parameters or introduce distributed collectives.
  Parameter and publish fingerprints stream at most 16 MiB per CPU transfer;
  they do not materialize a full CPU copy of the target head. Buffer/input/output
  artifacts still retain complete independent snapshots.
  **Different FSDP shard coordinates are not expected to have equal hashes.**
- Full registered buffers, including nonpersistent RoPE `inv_freq`, `cos_cached`,
  `sin_cached`, plus plain tensor attributes and tensor template caches.
  Buffer values are content-addressed per rank. Module training mode and RoPE
  scalar configuration are recorded too.
- Every optimizer forward's complete prepared crop batch, including hidden
  states, target last hidden states, tokens, absolute positions, teacher
  probabilities, trace/anchor metadata, masks, and available sample metadata.
  Forward outputs include context features, draft hidden states, positions,
  noise embeddings, and accepted/rejected K3 stream records. These are saved
  before optimizer mutation and linked to existing K3 dumps via `forward_id`.
- Publish snapshot hashes and actor sender payload hashes keyed by actor step;
  an `ack` means the rollout update RPC returned successfully. It does not prove
  that every vLLM internal cache matches the sender. This probe does not dump
  the entire vLLM KV cache or replace a checkpoint with a replayable full model.

The probe clones/detaches evidence without changing model state or drawing RNG
values. Exact hashes and full inputs incur CPU transfer, hashing and disk costs;
performance measurements from this run include that overhead. Any diagnostic
I/O/capture failure is explicitly logged with `[drafter state diagnostics]` and
does not silently abort or alter the optimizer path. Check for these failures
before treating missing evidence as agreement.

## Analysis

Run inside the experiment container:

```bash
python tools/analyze_drafter_state_diagnostics.py \
  /workspace/verl-SpeCo/outputs/drafter_analysis/EXPERIMENT/state_diagnostics \
  --compare-logprobs > /workspace/verl-SpeCo/outputs/drafter_analysis/EXPERIMENT/state_report.json
```

The report lists local lifecycle mutations, replicated buffer differences, and
whether compared ranks received identical input tensors. Log-prob comparison
aligns sample/anchor/position/token/previous-token identities and requires teacher
log-probs to agree within 1e-6. A reported gap is not causal attribution; inspect
the linked full inputs, buffer artifacts, head versions, and nearest save/export
events. Cache growth and target head updates can produce expected changes.

The probe captures optimizer training forwards. Fixed-cohort evaluation and
rollout generation do not receive full forward-input dumps in this version.
