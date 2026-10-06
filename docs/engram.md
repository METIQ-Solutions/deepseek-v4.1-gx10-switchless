# Engram: disk-backed serving of a ~203 GiB table set

## The problem

DeepSeek V4.1 Flash carries two Engram n-gram tables on the order of **203 GiB**
in FP8. Each node in this deployment has 128 GiB of *unified* memory shared
between the GPU and the host. The tables therefore cannot be resident — not
"are expensive to keep resident", but *do not fit*.

The upstream and day-0 alternatives make this worse rather than better: Engram
offload pins the tables into host memory using anonymous huge-page-backed
storage registered with `cudaHostRegister`, or a `NamedTemporaryFile` under
`/dev/shm`. Both are RAM-backed by construction. On this hardware that path has
contributed to catastrophic memory exhaustion, including a hard reset, and it is
still under active rework upstream.

## The approach used here

`DSV41_ENGRAM_DISK=1` plus the staged patch set makes Engram read its tables
from **local NVMe**:

* rows are read sequentially with `pread`, rank-offset corrected, from a local
  file on NVMe;
* the staging step runs *before* the CUDA-graph-captured forward, so the graph
  capture never sees file I/O;
* the sliding-window pieces stay resident;
* Engram data-parallel sharing is refused explicitly rather than silently
  returning local hashes: with `dp_shared_memory=True` the shim raises
  `NotImplementedError`.

Why local NVMe and not the network: an NFS-backed Engram converts a memory
problem into a latency problem that shows up as TTFT variance under load. The
model itself is also loaded from local storage with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; nothing in the serving path depends on a model host.

## The one local composition step

`scripts/fetch-engram-patches.sh`:

1. downloads seven files from `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`
   at the pinned revision `592540c69853a8ce9285236ebfd6e54dfc83a013`;
2. verifies each against its published SHA-256 and fails closed on a mismatch;
3. appends the local `gather_engram_hashes` shim to `engram.py`, delimited by
   explicit `# BEGIN/END DSV41 GATHER ENGRAM HASHES` markers;
4. syntax-checks the composed module.

The shim is the only non-upstream edit, and it is deliberately the *stricter*
option: it refuses shared-memory Engram DP instead of silently returning local
hashes. This deployment has one Engram data-parallel group (TP4, no Engram DP),
so the branch is unreachable in normal operation — which is exactly why it must
fail loudly rather than guess.

See `docs/provenance.md` for why the composed file's hash is not pinned to a
fixed value.

## Operating notes

* **Watch host memory, not just GPU memory.** Engram I/O errors appear as
  headroom collapse before they appear as inference failures.
* **The first request after a start is slower.** Engram pages are cold in the
  NVMe/OS cache; do not benchmark a cold start.
* **Do not "optimise" this into `/dev/shm` or pinned memory.** That is the
  configuration that failed.
* **A near-1M request stresses this path hardest.** Run
  `scripts/verify-1m-context.py` after any change to Engram staging.
