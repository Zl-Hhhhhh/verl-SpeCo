# Replace rejected K3 with first-rejection CE

This frozen-actor screening run tests both actor50 and actor100. It keeps accepted/response K3 at weight 1, disables
rejected-branch K3, and compares first-rejection replacement-token CE weights
0.05, 0.1, and 0.2. The drafter LR remains 1e-5.
Both `dspark_rejected_k3_alpha` and `dspark_rejected_stream_weight` are zero.

Each arm starts from the same original drafter weights, empty Adam state,
scheduler state, and RNG state. It reuses the exact cached actor100 training
batches from `outputs/frozen_actor_lr_20260921/r5/actor100`, with per-rank hash
checks. Evaluation remains greedy ABBA at 5 and 20 optimizer updates, plus the
320-response load ABBA after 20 updates.

The objective for this run is:

`L = mean(response K3) + alpha * mean(first-rejection replacement CE)`

There is no rejected-stream objective or joint response/rejected token-count
normalization. This intentionally tests the requested new objective, rather
than preserving the previous response K3 coefficient of approximately 0.818.
Asymmetric K3 remains disabled. Actor parameter and optimizer offload remain
enabled, and actor optimization remains disabled.

The two actor checkpoints run serially on the same 16 NPUs, each using its own
cached training batches (`r3/actor50` and `r5/actor100`). actor50 must complete
successfully before actor100 starts.

Managed run: `dspark_frozen_actor50_100_ce_replace_rejected_20260921_r3`.

The run pins the currently installed CANN runtime at
`/usr/local/Ascend/cann-9.0.1` and propagates its Python packages explicitly to
vLLM spawned workers. The former `/workspace/cann901_validation/runtime` mount
is not present in the current container.

Outputs: `outputs/frozen_actor_ce_replace_rejected_20260921/r1/actor50` and
`outputs/frozen_actor_ce_replace_rejected_20260921/r1/actor100`.
