# Experiment: Mesh transport (investigated, rejected)

**Status: not part of the supported runtime.** Recorded so the decision is not
re-litigated from scratch, and so a future reader knows what was examined.

## What was tried

A Mesh `libnccl-net.so` plugin path (`NCCL_NET=Mesh`, `NCCL_ALGO=Ring`,
`NCCL_RUNTIME_CONNECT=1`) was built into a candidate runtime alongside the
ring-only NCCL build, so the two transports could be compared on the same image.

## What was observed

| Property | Ring-only switchless (adopted) | Mesh plugin (rejected) |
| --- | --- | --- |
| NCCL runtime | 2.30.7 | **2.29.7** |
| Plugin/library footprint | one `libnccl.so.2.30.7` | plugin `libnccl-net.so` ≈2.2× that size, plus a Mesh NCCL 2.29.7 |
| Removes the ring-only requirement | n/a | **no** — `NCCL_ALGO=Ring` was still required |
| Upstream pin compatibility | matches the runtime's pinned NCCL version | below it |

## Why it was rejected

1. **Version conflict.** Adopting Mesh would have put the runtime on a NCCL
   older than the version the rest of the stack is pinned to, for no functional
   gain.
2. **It does not solve the original problem.** The ring-only requirement comes
   from Tree/PAT setup on a switchless topology, and Mesh did not remove it.
3. **Cost.** More moving parts, a larger plugin, and a second NCCL library in the
   same image.

## What remains

The rejected path is preserved only as history. The runtime, the environment and
the verification tooling in this repository use the ring-only build exclusively;
`scripts/verify-cluster.sh` actively fails if `NET/Mesh` or `NET/Socket` appears
as the tensor transport.
