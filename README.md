# DeepSeek V4.1 Flash on 4× GB10 with a switchless 200 GbE ConnectX ring

A **reproducible, production-tested reference implementation** for serving
DeepSeek V4.1 Flash at 1M context across four GB10 / GX10 nodes using a direct
switchless 200 GbE RoCE ring: TP4, FP8 MLA, disk-backed Engram, DSpark
speculative decoding, and a ring-only NCCL build.

This is a standalone reference implementation, not a copy of anyone's private
infrastructure. It is four shell scripts, one Python acceptance gate and one
benchmark driver over a small, reviewable configuration.

---

## What has actually been demonstrated

| | |
| --- | --- |
| Hardware | 4 × ASUS Ascent GX10 / NVIDIA DGX Spark-class GB10, 128 GiB unified memory per node |
| Model | official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint |
| Parallelism | **TP4**, PP1, one head + three headless workers |
| Interconnect | **switchless ring**, 2 × 200 GbE ConnectX per node, two lanes per direction, `/30` point-to-point, no switch |
| Transport | native NCCL `NET/IB` (RoCE v2) with a **patched, ring-only NCCL 2.30.7** (`NCCL_ALGO=Ring`, Tree/PAT transport setup suppressed, 4 channels over both RoCE ports, no Socket and no Mesh fallback) |
| Engram | **disk-backed** from local NVMe |
| Speculative decoding | **DSpark, k=5** |
| Context | **1,048,576 configured and genuinely exercised** |
| Near-1M | **1,048,283-token request successfully validated**, retrieval code returned |
| Measured | 128K: TTFT 68.454 s, ≈1,910 tok/s effective prefill, ≈35.24 tok/s decode · near-1M: TTFT 949.635 s, ≈1,103.88 tok/s effective prefill |

"Effective prefill" is `prompt_tokens / TTFT` and **includes HTTP, queueing and
transport time — it is not pure GPU prefill speed.** Full configuration, method
and caveats: [`benchmarks/results/`](benchmarks/results/v4.1-gx10-tp4-switchless.md).

We publish the numbers we measured and kept, the tooling to reproduce the rest,
and an explicit list of what we did **not** measure — rather than a full table
padded with estimates.

---

## The one thing to understand before you change anything

**Ring is the only collective algorithm this fabric can carry.** Tree, PAT,
CollNet and NVLS all reference rank pairs that a ring does not cable — rank 0
has no link to rank 2, and rank 1 has none to rank 3.

The ring-only NCCL build suppresses Tree/PAT *transport setup*, but it does
**not** restrict NCCL's algorithm matrix: `NCCL_ALGO=Ring` has to survive all the
way to the library. Anything that rewrites the process environment after the
launcher sets it can silently reintroduce Tree.

The concrete landmine: vLLM's batch-invariance mode rewrites NCCL in-process to
`NCCL_ALGO=allreduce:tree` (one channel, `Simple`, `NCCL_NTHREADS=1`) before the
tensor-parallel group is built. The result is a confusing mid-collective failure:

```
NCCL WARN  Rank 0 has no transport for recv peer 2 on channel 0/0
NCCL error: internal error
```

This is why the supported configuration pins `VLLM_BATCH_INVARIANT=0`, why
verification checks the running container's effective environment, and why the
failure fingerprints are documented.

> **Invariant:** any runtime behaviour that forces Tree or any non-Ring
> collective is incompatible with this physical switchless topology unless the
> required peer connectivity actually exists.

Full detail, including the unsupported subgroup shapes: [`docs/nccl-contract.md`](docs/nccl-contract.md).

---

## Quick start

Assumes four nodes cabled as a ring, a separate management network, Docker, and
a local copy of the checkpoint on each node.

```sh
# 0. read these first
less docs/architecture.md docs/nccl-contract.md docs/network-topology.md docs/runtime.md

# 1. describe your cluster (the only file with host-specific values)
$EDITOR config/topology.env
python3 scripts/validate-topology.py config/topology.env     # static check, no hardware

# 2. build the runtime on an arm64 node, then put it on every rank
docker/build.sh --tag deepseek-v4.1-gx10-switchless:local

# 3. record your hosts so strict host-key checking can succeed
ssh-keyscan -p 22 10.20.0.11 >> ~/.ssh/known_hosts   # ... and the other three
# verify the fingerprints out of band before trusting them

# 4. read-only readiness across all ranks
scripts/preflight.sh

# 5. start (converges each host, fences, starts workers then head, runs canary)
scripts/start.sh

# 6. prove the transport and the model
scripts/verify-cluster.sh

# 7. the hard acceptance gate (slow: a near-1M prefill takes ~15 minutes)
python3 scripts/verify-1m-context.py --endpoint http://10.20.0.11:8889 --out ./reports

# later
scripts/stop.sh
```

### Prerequisites

* Four GB10/GX10 nodes with 128 GiB unified memory each, cabled as described in
  [`docs/network-topology.md`](docs/network-topology.md) — one cycle, two lanes
  per direction, no switch.
* A management network reachable from where you run the scripts, plus SSH access
  with a key. The scripts use `BatchMode=yes` and `StrictHostKeyChecking=yes`;
  they never disable verification and never use a password.
* Docker with the NVIDIA container runtime on every node.
* The checkpoint at `MODEL_ROOT` on every node — 88 files, 510,313,353,565
  bytes, 48 SafeTensors shards. Not redistributed here; obtain it from
  `deepseek-ai/DeepSeek-V4.1-Flash` under its own terms.
* MTU 9000 and the `/30` addresses applied to the fabric interfaces.
  `scripts/preflight.sh` verifies this and tells you what is wrong.

---

## Repository layout

```
README.md  LICENSE  NOTICE

docker/     Dockerfile (pinned public base + verified ring-only NCCL)
            build.sh
            patches/switchless/    the ring-only behaviour, as reviewable patches

config/     topology.env           the only file with host-specific values
            vllm.env               serving configuration, every value explained

compose/    compose.yaml           one rank; same file on every node

scripts/    validate-topology.py   static topology check
            preflight.sh           read-only readiness, all ranks
            host-preflight.sh      the per-host checks
            discover-roce-gids.py  common IPv4 RoCE-v2 GID index
            collect-fabric-metrics.py
            check-edge-deltas.py   per-edge byte proof
            install-switchless-nccl.sh
            fetch-engram-patches.sh
            install-memory-boundary.sh
            start.sh  stop.sh  canary.sh  lib.sh
            nccl-collective.py     model-free four-rank collective
            acceptance.py          functional acceptance (incl. tools, DSpark)
            verify-cluster.sh      the transport + function proof
            verify-1m-context.py   hard near-1M acceptance

benchmarks/ scripts/benchmark.py   TTFT, decode, effective prefill, C1–C8
            scripts/test_benchmark.py
            results/               what we measured, and what we did not

tests/      test_validators.py     offline tests for the static validators

docs/       architecture · nccl-contract · network-topology · runtime · engram
            safety · benchmarks · troubleshooting · provenance
            lessons-learned · compatibility
            experiments/           history only, not the supported runtime
```

---

## Attribution

This project stands on other people's work, and the parts that make a switchless
ring work are **not ours**:

* **[alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)**
  (Apache-2.0) — the ring-only NCCL 2.30.7 build. This is the single change we
  make to the base image, and the reason this topology works at all.
* **[yunwei37/dgx-spark-4-ring-no-switch](https://github.com/yunwei37/dgx-spark-4-ring-no-switch)**
  (MIT) — the public four-ring GB10 runtime recipe and image we build from.
* **[tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)**
  (MIT) — the disk-backed Engram implementation and the GB10/SM12x sparse-MLA
  geometry patches.
* **[NVIDIA/nccl](https://github.com/NVIDIA/nccl)** (BSD-3-Clause) — the patched
  parent tree.
* **[vllm-project/vllm](https://github.com/vllm-project/vllm)** (Apache-2.0) and
  **[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)**
  (Apache-2.0) — the inference stack and the SM12x kernels.
* **[Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)** (MIT)
  — DSpark speculative decoding for this model family.
* **DeepSeek** — the model and the DSpark method.
* **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)**
  — the published finding that Tree/PAT setup still reaches for non-neighbour
  links under `NCCL_ALGO=Ring`.

**We did not invent switchless NCCL, DSpark, DeepSeek V4.1 support, or the
four-ring GB10 recipe.** Our contribution is the integration, the operational
hardening, the verification tooling and the record of what was measured and what
failed. Pinned revisions and licences: `NOTICE` and
[`docs/provenance.md`](docs/provenance.md).

---

## Negative results, kept on purpose

* **Stock NCCL cannot drive a switchless ring**, and an adjacent-IP ping will not
  tell you. Two independent proofs are needed.
* **Batch invariance silently forces Tree**, which this fabric cannot carry. See
  [`docs/nccl-contract.md`](docs/nccl-contract.md).
* **Large independent prefills serialize** at the scheduler's 8,192-token
  aggregate budget; the observed C4 TTFT staircase was
  15.96 / 32.22 / 47.60 / 62.65 s.
* **A 32768 batched-token budget was rejected.**
* **Engram**: ~203 GiB of tables do not fit in 128 GiB, and the pinned-host
  `/dev/shm` path caused memory exhaustion — hence disk-backed Engram.
* **GB10/SM12x sparse-MLA geometry** does not work out of the box; the pinned
  page-size patches are required.
* **Mesh was investigated and abandoned** (older NCCL, no benefit).
* **A newer upstream vLLM line was evaluated and not adopted** — it failed the
  tool-calling contract, not the switchless transport.

Details: [`docs/lessons-learned.md`](docs/lessons-learned.md),
[`docs/experiments/`](docs/experiments/), [`docs/compatibility.md`](docs/compatibility.md).

---

## Status and limitations

* This is a reference implementation, not a supported product. No HA, no rolling
  upgrade, no multi-tenant scheduling.
* It is a **switchless-generation** design. A switched fabric removes the
  diagonal-reachability constraint and may well supersede parts of this; this
  repository remains useful as a reproducible switchless reference.
* Ring-only means more than "the four-rank ring works": any communicator whose
  ring or tree spans a diagonal pair (a `{0,2}` subgroup, an EP/DP group across
  opposite ranks) is unsupported. See
  [`docs/nccl-contract.md`](docs/nccl-contract.md).
* The image is `linux/arm64` only, and the lifecycle assumes one rank per node on
  nodes with on the order of 128 GiB of unified memory.
* The scripts write to `/opt/switchless-nccl`, a systemd slice, and
  `WORKLOAD_ROOT` on each node, and require root there.
* A cold start takes tens of minutes; the near-1M gate takes about fifteen
  minutes of prefill per position.
* Only the hardware, software and model revisions listed in this README and
  `benchmarks/results/` have been tested by us.

## Licence

Apache-2.0 (`LICENSE`). Third-party attributions in `NOTICE`; this repository
redistributes no model weights, no container images and no third-party binaries.
