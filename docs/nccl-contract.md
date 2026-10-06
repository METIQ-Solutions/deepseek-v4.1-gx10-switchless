# The NCCL collective contract for a switchless ring

This is the single most important document in the repository. If you change one
thing about this deployment and it stops working, the cause is most likely here.

## The contract

**Ring is the only supported collective algorithm.** Tree, PAT, CollNet and NVLS
are not usable on this fabric, because they reference rank pairs that are not
physically cabled.

The ring cables exactly four links:

```
rank 0 -- rank 1
rank 1 -- rank 2
rank 2 -- rank 3
rank 3 -- rank 0
```

Rank 0 has **no** transport to rank 2, and rank 1 has none to rank 3. Those are
the *diagonals*, and they are two hops apart.

A Ring all-reduce only ever exchanges with `prev`/`next` in the ring, so it maps
onto the cabling perfectly. A Tree all-reduce does not.

## How the invariant is enforced, and how it is not

The ring-only NCCL build makes `ncclTransportTreeConnect()` and
`ncclTransportPatConnect()` return success **without creating connectors**. It
does *not* restrict NCCL's algorithm matrix — that is still whatever the
environment and the process ask for.

Concretely:

| Layer | What it does |
| --- | --- |
| `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_SKIP_TREE_CONNECT=1` | Suppress Tree/PAT *transport setup*. |
| `NCCL_ALGO=Ring` | Tells NCCL to use Ring. This must actually survive to the library. |
| `NCCL_IB_MERGE_NICS=0`, exact `NCCL_IB_HCA`, subnet-aware routing | Make the port choice deterministic across the two neighbour-facing RoCE devices. |
| `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4` | Pin the qualified four-channel ring. |

The last piece is the fragile one: **`NCCL_ALGO=Ring` is configuration, not a
patch-enforced property.** Anything that rewrites the process environment after
the launcher has set it can silently reintroduce Tree, and the failure appears
later as a confusing transport error.

## The vLLM landmine: `VLLM_BATCH_INVARIANT`

This is not hypothetical — it is the exact failure that cost us a candidate
runtime.

With `VLLM_BATCH_INVARIANT=1`, vLLM (v0.30.0 at least; the pins exist from at
least 2026-08-24) rewrites NCCL's algorithm matrix **inside the running process**
during worker initialisation, before the tensor-parallel group is built:

```
NCCL_ALGO=allreduce:tree
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
NCCL_NTHREADS=1
NCCL_PROTO=Simple
```

The first collective of the tensor-parallel communicator is therefore a **Tree**
all-reduce, whose tree edge for rank 0 is rank 2 — a diagonal with no connector.
The result is:

```
NCCL WARN  Rank 0 has no transport for recv peer 2 on channel 0/0
NCCL error: internal error
```

### Recognising it

If you see `Rank <r> has no transport for recv peer <p>` after
`ncclCommInitRank … Init COMPLETE`, check for these corroborating fingerprints
of a Tree-forcing override:

* `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` reported as `1` (you configured 4);
* `Invalid NCCL_NTHREADS 1` (you did not set `NCCL_NTHREADS`);
* `1 coll channels` / `1 p2p channels per peer` (the qualified ring uses 4).

Note that the override happens *in-process*, so a launcher-level check of
`NCCL_ALGO` cannot catch it. That is why this repository states the invariant in
the configuration (`VLLM_BATCH_INVARIANT=0` in `config/vllm.env`), checks the
running container's effective environment (`scripts/verify-cluster.sh`), and
documents the failure signature rather than relying on a single guard.

## Other unsupported shapes

Ring-only means more than "the four-rank ring works". Any communicator whose
ring or tree contains a diagonal pair is unsupported, including:

* a two-rank subgroup such as `{0,2}` or `{1,3}`;
* any expert-parallel, data-parallel or other group that spans opposite ranks;
* any group for which NCCL would choose PAT (binomial-tree all-gather or
  reduce-scatter under the default algorithm matrix).

This is a precondition for consumers of the deployment, not a bug to be worked
around. A working Ring collective must not be read as generic diagonal
peer-to-peer reachability.

## What the ring does *not* need

* **All-to-all fabric reachability.** It deliberately has none. Bootstrap and any
  non-tensor traffic use the separate management network, which is why
  `NCCL_SOCKET_IFNAME` and the rendezvous address point at management.
* **A switch.** The whole point is that two directly cabled neighbours exchange
  RDMA traffic with no fabric in between.

## Boundary note: why a Mesh-plugin deployment does not hit this

An independent, publicly documented four-node GB10 deployment of the same model
uses the Mesh net plugin rather than a ring-only NCCL build. Mesh presents a
virtual switch, so a Tree collective is *routable* there rather than impossible,
and that deployment also leaves batch invariance off. Its permissive transport
is a different design choice with a different trade-off; it is not evidence that
Tree would work on a switchless ring. We document this boundary because it is
the clearest available statement of what is and is not being claimed here.
