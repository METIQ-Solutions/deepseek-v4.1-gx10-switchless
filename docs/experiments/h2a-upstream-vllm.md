# Experiment: a newer upstream vLLM line (evaluated, not adopted)

**Status: not part of the supported runtime and not deployable from this
repository.** Recorded because it produced architectural knowledge worth keeping,
and because omitting it would leave a reader wondering whether a newer vLLM was
ever tried.

## What was evaluated

A candidate runtime built on a newer upstream vLLM release with a much smaller
custom patch set, re-basing the GB10/SM12x geometry patches and the disk-backed
Engram work onto the renamed upstream model package.

## What it achieved

* TP4 startup on the four-node switchless ring;
* the **switchless NCCL design reused unchanged** — the same verified ring-only
  library, proving the transport is not tied to one vLLM revision;
* model identity and ordinary generation.

## Why it was not promoted

The candidate **failed the application-level functional contract** we require —
tool calling. A runtime that cannot satisfy the tool-calling contract is not a
viable replacement for the running service, so the candidate was cleaned up and
the stable runtime was restored and re-verified.

**Final disposition: not currently viable.** No benchmark or scheduler
experiment was run against it; it did not get far enough.

## The one failure that looked fundamental, and was not

Early in the evaluation the candidate failed with:

```
NCCL WARN  Rank 0 has no transport for recv peer 2 on channel 0/0
NCCL error: internal error
```

This reads like a fundamental incompatibility between a modern vLLM and
switchless NCCL. It is not. Batch invariance rewrites the collective algorithm
matrix **in-process** to `NCCL_ALGO=allreduce:tree` before the tensor-parallel
group is built; a Tree all-reduce needs the diagonal pair rank 0 ↔ rank 2, which
a switchless ring does not cable. With Ring preserved, the collective works.

This is now a first-class invariant of this project rather than an experiment
footnote: see [`../nccl-contract.md`](../nccl-contract.md) and
[`../lessons-learned.md`](../lessons-learned.md) §1b. The supported
configuration pins `VLLM_BATCH_INVARIANT=0`.

## What must not be concluded

* **Not** that newer vLLM cannot run DeepSeek V4.1 on this hardware. That was not
  shown; the evidence points the other way.
* **Not** that a newer scheduler was benchmarked. It was not.
* **Not** that the switchless design is superseded. It was reused successfully.
* **Not** that Tree collectives can be made to work on a ring — the diagonal is
  simply not cabled.

## If you want to revisit this

1. Satisfy the application-level functional contract (tool calling) on the
   candidate.
2. Keep the switchless collective contract Ring-only — assert
   `VLLM_BATCH_INVARIANT=0`, or prove a deterministic Ring configuration.
3. Run `scripts/verify-cluster.sh`, then `scripts/verify-1m-context.py`, and
   compare against `benchmarks/results/` under identical settings — including the
   GPU clock-state check, which invalidates otherwise clean numbers.
