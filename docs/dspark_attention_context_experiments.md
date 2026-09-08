# DSpark Attention-Context Experiments

This implementation keeps the existing Qwen3 DSpark behavior as the default and
exposes attention-context changes as explicit experiment options.

## Supported modes

| Mode | Training mask | vLLM serving |
|---|---|---|
| `full` | Full actor prefix plus bidirectional same draft block | Supported |
| `swa` | Recent actor context plus bidirectional same draft block | Supported |
| `hybrid` | Local layers plus the final `N` full-prefix layers | Training only |
| `sparse_global` | Local layers plus final layers that see recent context and selected global tokens | Training only |

The current vLLM DFlash implementation rejects mixed full/sliding layers because
they require multiple KV-cache groups. SPECO therefore fails before engine startup
when `hybrid` or `sparse_global` is requested for serving. It does not silently run
a mismatched mask.

## 1. Full baseline and 2. uniform SWA

Run the same fixed drafter through the deployable `full`, `swa-512`, `swa-1024`,
and `swa-2048` arms:

```bash
DRAFTER_PATH=/workspace/models/dspark_qwen3_8b_block7 \
TOTAL_STEPS=10 \
./tools/run_dspark_attention_ablation.sh
```

The command dispatches the entire four-arm queue as one detached managed run by
default. Set `SPECO_ABLATION_LAUNCH_MODE=foreground` only for interactive
debugging. `SPECO_WAIT_FOR_HOST_LOCK=1` queues it behind an already-running job.

The ablation runner is validation-only: actor and drafter weights remain frozen.
Its default cohort is a deterministic seed-42 sample of 128 rows from the DAPO validation parquet with
greedy decoding and a 2048-token response cap. Override `VAL_MAX_SAMPLES`,
`TRAIN_MAX_SAMPLES`, `MAX_RESPONSE_LENGTH`, `VAL_N`, or pass trailing Hydra overrides when a larger
or sampled evaluation is required. Validation reports aggregate speculative
decode counters under `val-aux/drafter/spec_decode/*`.

After each arm, the runner refreshes `summary.json` and `summary.csv` under
`outputs/drafter_analysis/$ABLATION_PREFIX/`. The summary contains verification
steps, accepted tokens, mean acceptance length, validation generation time, and
effective response tokens for every arm.

`swa` creates a config-only runtime overlay under
`/tmp/verl_speco_dspark_attention`. Original checkpoints and weights are not
modified. `dspark_context_window` counts actor-context tokens, while the
vLLM-Ascend block-table window also contains the bidirectional draft block. For
an actor window of 1024 and a seven-token draft block, SPECO injects:

```json
{
  "dflash_config": {
    "use_swa": true,
    "swa_window_size": 1031,
    "causal": false
  },
  "sliding_window": 1031
}
```

It also sets
`rollout.engine_kwargs.vllm.additional_config.draft_window_size=1031`. This
activates vLLM-Ascend's `SlidingWindowAdapter`, which crops the draft block table
before the eager non-causal attention kernel runs. On native multi-group DSpark,
SPECO applies the same span to every draft-layer KV-cache group. Startup fails if
the model, engine, instantiated adapter, or grouped-table patch do not match.
These values are injected automatically; configure the public actor-context
window only. Because the adapter starts at a paged-KV block boundary, the runtime
may retain up to `block_size - 1` additional old actor tokens; it never removes
any of the seven draft queries.

The NPU smoke run on 2026-09-04 verified the complete path for SWA-512. The
generated vLLM command contained `draft_window_size=519`; all 16 TP workers
validated the adapter, and the first draft step reported grouped block-table
cropping with `runtime_window=519`, `block_size=128`, and one KV-cache group.
The run completed successfully. Its log is
`outputs/validation_logs/2026-09-04/dspark_vllm/dspark_swa_runtime_fix_smoke_v3_20260904.log`.
The smoke metrics are only a runtime health check, not a full-vs-SWA performance
comparison.

## Matched optional training

Training remains `full` unless explicitly enabled. A matched SWA-1024 run needs a
larger collected/cropped sequence so the mask actually removes old context:

```bash
RUN_MODE=online \
DSPARK_VLLM_ATTENTION_MODE=swa \
DSPARK_VLLM_CONTEXT_WINDOW=1024 \
DSPARK_TRAINING_ATTENTION_MODE=swa \
DSPARK_TRAINING_CONTEXT_WINDOW=1024 \
DSPARK_TRAINING_MAX_WINDOW=2048 \
DSPARK_COLLECTION_WINDOW=2048 \
DSPARK_USE_ABSOLUTE_POSITION_IDS=True \
bash tools/run_dspark_vllm_experiment.sh
```

The launcher rejects matched context-window training when either collected window
is not larger than the local window. Absolute position IDs preserve the original
RoPE indices for cropped windows and draft positions.

## 3. Hybrid and 4. sparse global training

Hybrid modes should be trained with the standalone feature-store trainer, because
the current online vLLM worker cannot serve or candidate-gate them. The `4 local +
1 global` ablation is:

```bash
/workspace/main_env_next/bin/python -m verl_speco.draft_train_launcher \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  actor_rollout_ref.rollout.drafter.model_path=/path/to/dspark \
  actor_rollout_ref.rollout.drafter.training.mode=offline \
  actor_rollout_ref.rollout.drafter.training.feature_store.path=/path/to/features \
  actor_rollout_ref.rollout.drafter.training.dspark_attention_mode=hybrid \
  actor_rollout_ref.rollout.drafter.training.dspark_context_window=1024 \
  actor_rollout_ref.rollout.drafter.training.dspark_global_layer_count=1 \
  actor_rollout_ref.rollout.drafter.training.dspark_max_window=2048 \
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=2048 \
  actor_rollout_ref.rollout.drafter.training.dspark_use_absolute_position_ids=True
```

`sparse_global` keeps the same local mask and adds selected tokens only in the
final global layers:

```bash
/workspace/main_env_next/bin/python -m verl_speco.draft_train_launcher \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  actor_rollout_ref.rollout.drafter.model_path=/path/to/dspark \
  actor_rollout_ref.rollout.drafter.training.mode=offline \
  actor_rollout_ref.rollout.drafter.training.feature_store.path=/path/to/features \
  actor_rollout_ref.rollout.drafter.training.dspark_attention_mode=sparse_global \
  actor_rollout_ref.rollout.drafter.training.dspark_context_window=1024 \
  actor_rollout_ref.rollout.drafter.training.dspark_global_layer_count=1 \
  actor_rollout_ref.rollout.drafter.training.dspark_global_prefix_tokens=128 \
  actor_rollout_ref.rollout.drafter.training.dspark_global_prompt_tokens=512 \
  actor_rollout_ref.rollout.drafter.training.dspark_actor_guided_page_size=16 \
  actor_rollout_ref.rollout.drafter.training.dspark_actor_guided_page_budget_tokens=512 \
  actor_rollout_ref.rollout.drafter.training.dspark_max_window=2048 \
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=2048 \
  actor_rollout_ref.rollout.drafter.training.dspark_use_absolute_position_ids=True
```

The optional sample field `actor_selected_context_page_starts` is interpreted as
absolute page-start positions. Prefix tokens approximate BoS/system anchors and
the last prompt tokens approximate the current user prompt; exact chat-role spans
require the data pipeline to provide role boundaries. Do not publish hybrid or
sparse-global checkpoints to the current vLLM path until mixed-cache serving is
implemented.

## Diagnostics

Online training reports the mean absolute anchor position, mean local context
tokens, and anchor counts in `0-511`, `512-1023`, `1024-2047`, and `2048+` buckets,
in addition to existing per-draft-position loss, accuracy, K3, and acceptance
metrics. Compare fixed-checkpoint inference first; train a matched mode only after
it improves acceptance length and end-to-end latency on long-context buckets.
