# Provenance and licensing

This repository is an independent, standalone reference implementation. It
contains **no code, no weights and no binaries copied from the private
infrastructure repository it was distilled from**, and it redistributes no
third-party artifacts. Every external input is pinned to an immutable revision
and verified by checksum at fetch time.

## What this repository contains, and where it came from

### Original work in this repository

| Artifact | Nature |
| --- | --- |
| `scripts/` (preflight, converge/start, fence/stop, transport and functional verification, 1M acceptance) | Original. The lifecycle, the fail-closed checks and the topology model were written for this repository. The *design constraints* they encode (ring-only transport, rendezvous on the management plane, checked fence, per-edge byte evidence) come from operating a four-node switchless ring. |
| `scripts/validate-topology.py`, `check-edge-deltas.py`, `discover-roce-gids.py`, `collect-fabric-metrics.py` | Original implementations of checks that previously existed only as cluster-specific automation in the estate this was distilled from. |
| `compose/compose.yaml`, `config/`, `docker/Dockerfile`, `docker/build.sh` | Original files describing an established runtime. `docker/Dockerfile` deliberately mirrors the base image's own layout (including replacing the pip `nvidia/nccl/lib` copies with symlinks) because three different code paths inside the image resolve NCCL separately. |
| `benchmarks/scripts/` | Original benchmark driver and its offline regression tests. |
| `docs/` | Original documentation of our own measurements, negative results and lessons. |

### Third-party work, fetched and verified but not redistributed

| Project | Licence | Pinned revision | Role here |
| --- | --- | --- | --- |
| [yunwei37/dgx-spark-4-ring-no-switch](https://github.com/yunwei37/dgx-spark-4-ring-no-switch) | MIT (© 2026 Yunwei Zhang) | image `sha256:2f8e2a70e73541eacf8c71a8990e907fa18fcd8b2a8d02cd0d54a7c81382a13c` | The public ARM64 four-ring runtime this repository builds from. The four-node switchless topology, the launch pattern and the serving-profile structure follow this work. |
| [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) | Apache-2.0 | `v0.0.1` = `480389e72390f93801ae6f703dd489fe5fd00bea`, archive `sha256:b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f`, library `sha256:78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57` | The ring-only NCCL 2.30.7 build. This is the single change this repository makes to the base image, and the reason a switchless ring works at all. |
| [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | BSD-3-Clause | `73cf112295c33aee2b895f329f592f2a9b4b0f97` (v2.30.7) | The patched parent tree of the above build. |
| [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | MIT (© 2026 Tech2Wild) | `592540c69853a8ce9285236ebfd6e54dfc83a013` | The disk-backed Engram implementation and the GB10/SM12x sparse-MLA geometry patches: seven files, each verified by SHA-256 before use. |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | inside the pinned image | The inference stack, including the DeepSeek V4.1 model implementation and DSpark speculative decoding. |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0 | build commit `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` (0.7.0rc1) | Sparse-MLA SM12x kernels for GB10. |
| [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) | MIT (© 2026 Anemll project contributors) | `47503f8e38dadd4dededca798150db2619594fce` (0.1.1) | DSpark speculative decoding lineage for this model family. |
| NVIDIA CUDA container images | NVIDIA container licence | base layers of the pinned image | Unmodified build/runtime base. |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | MIT | discussion/PR #3 | The published finding that NCCL Tree/PAT setup attempts non-neighbour links even under `NCCL_ALGO=Ring`. Referenced, not vendored. |

All upstream licences (MIT, Apache-2.0, BSD-3-Clause) are permissive and
mutually compatible. This repository is Apache-2.0; the required attributions
are in `NOTICE`.

## The NCCL patches in `docker/patches/switchless/`

These three patches describe the ring-only behaviour of the release binary that
this runtime actually loads:

| Patch | What it changes |
| --- | --- |
| `nccl-2.30.7-skip-tree-pat.patch` | Skips `ncclTransportTreeConnect` / `ncclTransportPatConnect` under `NCCL_SKIP_TREE_CONNECT`. |
| `nccl-2.30.7-advertise-all-listener-gids.patch` | Advertises listener GIDs from all merged devices rather than only the first two. |
| `nccl-2.30.7-hardened-switchless.patch` | Turns the above into real parameters (`NCCL_SWITCHLESS_RING_ONLY`, `NCCL_SKIP_TREE_CONNECT`), refuses `NCCL_IB_MERGE_NICS != 0`, requires exactly two distinct eligible listener GIDs, and rejects duplicates. |

They are shipped as documentation of the behaviour the runtime depends on. They
are **not** required to consume the release: the runtime installs the verified
binary. The parent tree is NVIDIA NCCL (BSD-3-Clause); the producing project is
Apache-2.0.

Two honest caveats:

1. These patches are not applied by `docker/build.sh`; the upstream release
   archive is used directly, and its identity is verified twice (archive and
   installed library).
2. A private source lock recorded checksums for three NCCL patch items, but
   those recorded values do not correspond to the SHA-256 of these patch files.
   They were most likely digests of the patched source files rather than of the
   patch text. This repository therefore does not claim those hashes; the
   authoritative identity of the transport is the release archive and library
   SHA-256 above.

## The base image, and what is and is not reproducible about it

`docker/Dockerfile` builds `FROM ghcr.io/yunwei37/dgx-spark-4-ring-no-switch@sha256:2f8e2a70…`.
That digest is the published, public, source-locked base for the switchless
V4.1 candidate, and the resulting runtime is fully reproducible from public
artifacts by anyone.

For completeness: the deployment that produced the measurements in
`benchmarks/results/` ran a privately re-published copy of the runtime with a
different manifest digest. That copy has no public provenance recorded, and this
repository neither publishes nor references it. If you are comparing against
those numbers, use the base digest above plus the same ring-only NCCL release;
the difference between the two was container-image re-publication, not image
content.

## The Engram composition step

`scripts/fetch-engram-patches.sh` verifies all seven upstream files against
their published SHA-256, then appends one local shim to `engram.py` so that
disk-backed Engram refuses the shared-memory Engram-DP mode instead of silently
returning local hashes. The shim is delimited by explicit markers and the file
is syntax-checked afterwards.

Our private deployment recorded a SHA-256 of the *composed* file. This
repository verifies the upstream source exactly and verifies the composition
behaviourally (source hash, shim present, module parses) rather than pinning the
exact bytes of a composed file, because the exact bytes depend on the append and
on how the file was assembled. That is a deliberate choice: the security-relevant
property is that the third-party file is authentic and the shim is the one you
reviewed.

## What we did not invent

To be explicit, because it matters more than the attribution boilerplate:

* **Switchless NCCL is not ours.** The ring-only build is
  [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl), and
  its own documentation already states the limitation we re-documented here
  ("works when the deployment also forces `NCCL_ALGO=Ring`, but it leaves NCCL's
  internal algorithm matrix claiming that unsupported algorithms exist").
* **The four-ring GB10 runtime recipe is not ours.** It is
  [yunwei37/dgx-spark-4-ring-no-switch](https://github.com/yunwei37/dgx-spark-4-ring-no-switch).
* **DSpark is not ours.** It comes from the model family and
  [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10).
* **DeepSeek V4.1 support in vLLM is not ours.** It is upstream and, in our
  runtime, inherited from the pinned base image.
* **Disk-backed Engram is not ours.** It is
  [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark).
* **The finding that Tree/PAT still reaches non-neighbour links under
  `NCCL_ALGO=Ring` was published by others** (MiaAI-Lab).

What is ours: the specific integration and build recipe in `docker/`, the
operational hardening (host-memory containment, checked fencing,
worker-before-head ordering, canary semantics), the verification tooling
(per-edge byte proof, Ring-only and Tree-override detection, GID discovery,
topology validation, near-1M acceptance) and this documentation of what was
measured and what failed.

## The batch-invariance finding

The `VLLM_BATCH_INVARIANT` interaction documented in
[`nccl-contract.md`](nccl-contract.md) is our own diagnosis, reached from
container logs and pinned upstream sources after a candidate runtime failed. It
is a statement about a specific vLLM behaviour on a specific topology, not a
claim about vLLM in general. It is included because the failure signature is
genuinely confusing and cost us real time.

## Model weights

No model weights are redistributed. `config/topology.env` points at a local
directory you populate yourself from
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`
(88 files, 510,313,353,565 bytes, 48 SafeTensors shards). The runtime never
contacts a model host: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are part
of the validated configuration.
