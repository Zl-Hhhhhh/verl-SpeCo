# Fixed-actor rejected-prefix screening

This experiment keeps the actor frozen at saved steps 50 and 100 and compares
five drafter objectives from an identical initial model, empty Adam/scheduler
state, RNG state, and cached per-rank training batches:

- baseline: response K3 + 0.1 independently normalized first-rejection CE;
- the baseline plus 0.1 times independently normalized rejected K3 over
  relative rejected-prefix lengths 1, 2, 3, or 4.

Prefix position zero is the actual first rejection (`draft_position -
accepted_count == 0`), not block position zero. The ordinary jointly normalized
rejected stream remains disabled. Each arm is evaluated after 5 and 20 optimizer
steps with crossed ABBA probes; the 20-step checkpoint also gets a 320-response,
2048-token load probe. Actor updates are disabled and every arm must pass exact
weight, optimizer, scheduler, step, and RNG restoration checks.

Managed run: `dspark_frozen_actor50_100_rejected_prefix_20260922_r1`.

Outputs: `outputs/frozen_actor_rejected_prefix_20260922/r1/actor100` and
`actor50`.
