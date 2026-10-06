# Lessons learned, including negative results

Everything here is a result we paid for. Negative results are included because
they are the reason the supported configuration looks the way it does.

## 1. Stock NCCL cannot drive a switchless ring, and a ping will not tell you

`NCCL_ALGO=Ring` selects the ring *algorithm*; it does not stop NCCL from
setting up Tree and PAT transports. On a switchless ring those paths try to
reach ranks that are not physically adjacent, and the collective either fails or
falls back to a transport the ring cannot carry.

Consequences that shaped this repository:

* the ring-only NCCL build is required, not optional (`NCCL_SWITCHLESS_RING_ONLY=1`,
  `NCCL_SKIP_TREE_CONNECT=1`);
* an adjacent-IP `ping` proves nothing. All four ranks must be shown to complete
  a collective over native IB **and** the byte counters must show traffic on
  every physical edge — two separate proofs (`scripts/verify-cluster.sh`,
  stages B and C);
* the hardened build refuses `NCCL_IB_MERGE_NICS != 0` and requires exactly two
  distinct eligible listener GIDs, so a silent misconfiguration becomes a loud
  error.

## 1b. The collective contract is configuration, not just a patch

The ring-only NCCL build suppresses Tree/PAT *transport setup*. It does not
restrict NCCL's algorithm matrix, so `NCCL_ALGO=Ring` must actually survive from
your configuration to the library. Anything that rewrites the process
environment after the launcher has set it can silently reintroduce Tree.

The concrete example cost us a candidate runtime: vLLM's batch-invariance mode
(`VLLM_BATCH_INVARIANT=1`) rewrites the algorithm matrix **in-process** to
`NCCL_ALGO=allreduce:tree` (one channel, `Simple`, `NCCL_NTHREADS=1`) before the
tensor-parallel group is built. A Tree all-reduce needs a diagonal rank pair —
rank 0 ↔ rank 2 on this cabling — which a switchless ring does not provide, so
the first collective fails with `Rank 0 has no transport for recv peer 2`.

Because the rewrite happens inside the process, a launcher-level check of
`NCCL_ALGO` cannot catch it. Hence: `VLLM_BATCH_INVARIANT=0` is part of the
supported configuration, the running container's effective environment is
checked during verification, and the failure fingerprints are documented.

The general invariant, worth stating by itself:

> Any runtime behaviour that forces Tree or any non-Ring collective is
> incompatible with this physical switchless topology unless the required peer
> connectivity actually exists.

The same applies to any communicator whose ring or tree spans a diagonal pair —
a two-rank `{0,2}` subgroup, an expert- or data-parallel group across opposite
ranks, or anything NCCL would route with PAT. See
[`nccl-contract.md`](nccl-contract.md).

## 2. Concurrent prefill: the budget is aggregate, the threshold is per request

`long_prefill_token_threshold` caps how much of a *single* request may be
scheduled in one step. `max_num_scheduled_tokens`, falling back to
`max_num_batched_tokens`, bounds the *aggregate* step. The scheduler keeps
admitting waiting requests while aggregate budget remains, so:

```
--max-num-batched-tokens 16384
--long-prefill-token-threshold 8192
```

admits two independent ~8K prefills in one step. That is what was wanted, and
it could not be expressed with the wrong flag.

The failure mode is instructive: with `long_prefill_token_threshold=0` the cap
is disabled, so raising `max_num_batched_tokens` from 8192 to 16384 let the
first waiting request consume the whole budget. At concurrency 4 the measured
TTFT staircase was **15.96 / 32.22 / 47.60 / 62.65 s** — each request waiting
for the previous one's full prefill, not for a shared step.

Our supported configuration keeps `--max-num-batched-tokens 8192`. The
experiment above explains the mechanism; it was not adopted, because the
validated configuration is the one that was end-to-end qualified.

## 3. The rejected 32768 token budget

A `max_num_batched_tokens` of 32768 was rejected: it does not admit usefully
more independent prefills on this hardware, it inflates the exported scheduler
budget, and it makes the TTFT behaviour of a single large request worse. The
validator refuses a candidate at or above that value.

## 4. Engram: ~203 GiB of tables in a 128 GiB node

The two Engram n-gram tables are on the order of **203 GiB** in FP8. They cannot
live in a 128 GiB unified-memory node, and the obvious alternative — pinning
them into host memory — is what turns a slow path into a hard failure:

* the pinned-host / `/dev/shm` / transparent-huge-page design
  (`cpu_offload`) is RAM-backed by construction;
* on this class of hardware it previously contributed to catastrophic memory
  exhaustion, including a hard reset;
* it is still under active rework upstream, and at least one path regressed.

Hence disk-backed Engram: sequential `pread` of rows from local NVMe,
rank-offset corrected, staged before the graph-captured forward, with the
sliding-window pieces kept resident. Combined with the host-memory boundary in
`docs/safety.md`, the failure mode becomes a bounded, observable slowdown rather
than a node reset.

Practical consequences:

* Engram DP sharing is refused explicitly (`dp_shared_memory=True` raises) rather
  than silently returning local hashes;
* local NVMe, not the network, is the Engram backing store — an NFS-backed
  Engram turns a memory problem into a latency problem;
* the ~203 GiB concern is a property of the checkpoint, not of the patch.

## 5. GB10 / SM12x sparse-MLA geometry

DeepSeek V4.1's sparse MLA attention does not serve on GB10 out of the box: the
SWA cache is constructed with a 32-token block, the backend reports 128-token
pages off SM90, and the FlashInfer sparse-MLA decode kernel only instantiates
64-token pages while the paged-MQA-logits kernel requires 64 states. The minimum
fix is the pinned SM12x page-size patch set used here, with `--block-size 128`
pinned globally so the per-layer overrides apply instead of racing the preferred
page size.

## 6. Mesh was investigated and abandoned

A Mesh `libnccl-net.so` plugin path was built and considered. It was rejected:

* its NCCL runtime is **2.29.7**, below the 2.30.7 this runtime is pinned to;
* the plugin library alone is ~2.2× the size of the ring-only NCCL build;
* it does not remove the ring-only requirement.

Details and the artifacts that were examined are in
[`docs/experiments/mesh.md`](experiments/mesh.md).

## 7. A newer upstream vLLM line was evaluated and not adopted

An upstream-first candidate (a newer vLLM release plus a re-based patch set) was
researched, built and executed. It reached TP4 startup, patched switchless
operation, model identity and ordinary generation — and was then **not
promoted**, because it did not satisfy the full application-level functional
contract (tool calling). Its final disposition was "not currently viable", the
candidate was cleaned up, and the stable runtime was restored and re-verified.

Two things are worth keeping from it:

* The switchless NCCL design was **reused unchanged**. The transport was never
  the obstacle, which is useful evidence that the design is not tied to one vLLM
  revision. See [`compatibility.md`](compatibility.md).
* One failure on the way there looked like a fundamental switchless limitation
  and was not. It was batch invariance rewriting the collective algorithm to
  Tree. That is now documented as a first-class invariant in
  [`nccl-contract.md`](nccl-contract.md) and in §1b above.

It is deliberately **not** claimed that a newer runtime cannot work here, nor
that any newer scheduler was benchmarked — it was not; the candidate stopped
first. See [`experiments/h2a-upstream-vllm.md`](experiments/h2a-upstream-vllm.md).
Nothing in this repository depends on that candidate.

## 8. Operational lessons

* **Do not let a distributed generation restart itself.** A host-local unit
  cannot prove its peers belong to the same generation, so ranks are started as
  a unit and never reconstructed rank-by-rank after a boot. `restart: "no"` and
  no boot enablement; after a reboot the nodes are compute-idle until a
  coordinated start.
* **Fence the head first.** The rendezvous endpoint must be confirmed gone before
  the workers stop; stopping workers first leaves the head retrying against a
  peer that no longer exists.
* **A healthy `/health` is not usable inference.** The API answers before the
  communicator is usable, so the canary requires an actual generated token.
* **Verify the library that is loaded, not the one that was installed.** Three
  paths inside the image resolve NCCL separately (PyTorch's pip package, DeepEP
  and vLLM's PyNccl), so the pip copies are replaced by symlinks and the loaded
  library is checked by path, SHA-256 and `ncclGetVersion()`.
* **An empty completion can still look "non-empty".** `"role":"assistant"` and
  `"finish_reason":"stop"` satisfy a naive quoted-string match, so the checks
  assert on `usage.completion_tokens`.
* **GPU clock state invalidates benchmarks.** See `docs/benchmarks.md`.
