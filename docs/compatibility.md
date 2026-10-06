# Compatibility and future work

## What this repository represents

The **validated switchless generation** of the implementation: a ring-only NCCL
transport over a direct four-node ConnectX cycle, with the runtime contract that
was qualified end-to-end on it. It is published because it is reproducible and
because the constraints it documents are not obvious.

It is not a claim that this is the only, or the best, way to serve DeepSeek V4.1
Flash on four GB10/GX10 nodes. It is one working design, with its trade-offs
stated plainly.

## A newer upstream-vLLM candidate was evaluated and not adopted

We built and evaluated a candidate runtime on a newer upstream vLLM release with
a much smaller custom patch set. It is **not** part of what you get here, and the
repository does not depend on any of it.

What was learned, because it is genuinely useful to anyone attempting the same
thing:

* the switchless NCCL design was **reused unchanged** by the newer candidate — the
  library boundary held, and the transport itself was not the obstacle;
* the candidate did reach TP4 startup, patched switchless operation, model
  identity and ordinary generation;
* it was **not promoted**, because it did not satisfy the full application-level
  functional contract we require (tool calling), so it was not a viable
  replacement for the running service;
* one failure on the way there was **not** a fundamental switchless limitation.
  It was batch-invariance behaviour rewriting the NCCL configuration toward Tree
  collectives, which on a ring requires a diagonal rank that is not cabled. That
  mechanism is documented in full in [`nccl-contract.md`](nccl-contract.md), and
  it is the reason this repository states the Ring-only invariant so explicitly.

### What is deliberately *not* claimed

* That modern vLLM fundamentally cannot run DeepSeek V4.1 on this hardware. It
  was not proven, and the evidence points the other way.
* That a newer scheduler was benchmarked. It was not — the candidate never got
  far enough.
* That the switchless approach is obsolete or that a newer runtime supersedes it.
  It is simply a different, unproven-by-us combination so far.

## Future work this repository points at

* **A switched fabric.** A switch removes the diagonal-reachability constraint
  entirely and makes Tree, PAT and Mesh-plugin designs viable. That is a
  different design point with different trade-offs, and it may well supersede
  parts of this one.
* **Batch invariance on a ring.** Making determinism/batch-invariance work here
  would require either a proven-deterministic Ring configuration or NCCL-level
  work to synthesise a tree over a non-fully-connected topology. Both are
  substantial and neither was attempted.
* **Failing fast instead of confusingly.** The ring-only library currently knows
  Tree is unsupported but lets a communicator be constructed anyway, producing a
  mid-collective `internal error`. A check at communicator initialisation that
  refuses an algorithm matrix it cannot serve would turn that into a one-line
  diagnostic.
* **Newer vLLM lineages**, once the remaining application-level contract is
  satisfied and the switchless collective contract is preserved.

## Hardware and software combinations actually tested

Only what is stated in the README and `benchmarks/results/`: four GB10/GX10
nodes, the pinned runtime and library revisions listed there, the pinned model
revision, and the four-node ring cabling. Anything else is untested by us. In
particular we do not claim it works on a switched fabric, on more or fewer than
four nodes, with two-rank pairs, or on a mixed-vendor interconnect.
