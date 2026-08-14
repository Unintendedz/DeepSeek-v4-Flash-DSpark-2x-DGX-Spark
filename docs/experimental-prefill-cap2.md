# Experimental two-prefill admission profile

The Anemll `0.1.1` image rejects vLLM's Concurrent Partial Prefill CLI option
before engine initialization. The existing issue #27 runtime patch already
adds a scheduler admission gate; this profile makes that patch-time gate
selectable without passing the rejected CLI flag.

The repository default is conservative:

```env
DSPARK_MAX_INFLIGHT_PREFILLS=1
```

To let two long prompts make prefill progress concurrently:

```env
DSPARK_MAX_INFLIGHT_PREFILLS=2
LONG_PREFILL_TOKEN_THRESHOLD=1024
MAX_NUM_BATCHED_TOKENS=8192
```

Then restart the two-node serve. The hotfix writes the selected value as a
literal into the scheduler in each new container, so changing the env without
a restart has no effect.

## What changes

- Up to two already-admitted requests may consume chunked-prefill budget in a
  scheduler step instead of one.
- The merged issue #43 decode floor still reserves bounded service for active
  decode lanes behind those prefills.
- Model weights, KV dtype, prompt tokens, sampling settings, speculative
  decoding, and decode kernels are unchanged. Output quality should therefore
  be semantically unchanged, although timing-dependent sampling is never a
  bit-for-bit determinism guarantee under concurrency.

This is a targeted admission-cap backport, not a claim that every upstream
Concurrent Partial Prefill feature is present in the old image.

## Expected trade-off

Cap 2 can reduce serialized TTFT/head-of-line waiting when two large cold
prompts arrive together. It can also give each prefill less instantaneous
budget, increase memory pressure, or hurt a lone request. It does not increase
single-stream decode tok/s.

The live two-conversation deployment completed real long-context traffic with
cap 2 and the issue #43 floor, but no x4/x8 benchmark matrix is published for
this alpha. Treat any throughput claim beyond that as unverified.

## Validate and roll back

The real-image patch application test injects cap 2, compiles the patched
scheduler, and checks idempotence:

```bash
python3 tests/test_issue43_patchapply.py
python3 tests/sim/test_issue43_scheduler_sim.py --cap 2
```

Watch scheduler diagnostics during a controlled run:

```env
DSPARK_ISSUE43_SCHED_DIAG=1
```

Rollback is a full restart with:

```env
DSPARK_MAX_INFLIGHT_PREFILLS=1
DSPARK_ISSUE43_SCHED_DIAG=0
```
