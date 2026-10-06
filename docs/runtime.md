# Runtime: image, serving configuration and lifecycle

## Image

`docker/Dockerfile` takes the pinned public four-ring GB10 runtime and makes one
change: it installs the verified ring-only NCCL 2.30.7 build and points every
NCCL lookup path at it.

```
ghcr.io/yunwei37/dgx-spark-4-ring-no-switch@sha256:2f8e2a70…   (pinned base)
  + /opt/switchless-nccl/libnccl.so.2.30.7                     (verified archive)
  = this runtime
```

Nothing else is modified. vLLM, PyTorch, CUDA, FlashInfer, the DeepSeek V4.1
model implementation and DSpark all come from the base image unchanged.

Build (on an arm64 node or under emulation):

```sh
docker/build.sh --tag deepseek-v4.1-gx10-switchless:local
```

`build.sh` downloads the NCCL release, verifies the archive SHA-256, verifies
the library inside it, verifies the marker string that identifies the hardened
build, builds with `--network=none`-safe steps, and then re-verifies the built
image by loading the library and asserting `ncclGetVersion() == 23007`.

There is no registry dependency: the resulting image is tagged locally. Put it on
every rank however your environment already moves images (`docker save`/`load`,
your own registry, or a network share).

### Why the pip NCCL copies are replaced by symlinks

Three code paths resolve NCCL separately — PyTorch's `nvidia/nccl/lib` package
copy, DeepEP, and vLLM's PyNccl (`VLLM_NCCL_SO_PATH`). Leaving the pip copies in
place would let one of them load a different NCCL than the ring-only build. The
image replaces them with symlinks to `/opt/switchless-nccl/libnccl.so.2.30.7`,
and `verify-cluster.sh` checks the library actually mapped into the running
process by path, SHA-256 and version rather than trusting the install.

## Serving configuration

`config/vllm.env` is the whole configuration and every value is commented with
its reason. The summary:

| Setting | Value | Why |
| --- | --- | --- |
| `--served-model-name` | `deepseek-v4.1-flash` | Identity used by the API and the canary. |
| `--tensor-parallel-size` | 4 | One rank per node, 4 nodes. |
| `--nnodes` / `--node-rank` | 4 / per host | Multiprocessing backend, head + 3 headless workers. |
| `--max-model-len` | 1048576 | The capability actually exercised, not just configured. |
| `--max-num-seqs` | 32 | |
| `--max-num-batched-tokens` | 8192 | The validated budget; see `lessons-learned.md` for why not 16384/32768. |
| `--max-cudagraph-capture-size` | 32 | |
| `--gpu-memory-utilization` | 0.86 | Chosen against the host-memory boundary; higher values exhausted host memory. |
| `--kv-cache-dtype` | `fp8` | The model's sparse MLA layout on GB10 accepts an FP8 KV cache. |
| `--block-size` | 128 | Pinned globally so per-layer overrides apply instead of racing the preferred page size. |
| `VLLM_KV_CACHE_LAYOUT` | `BLNHC` | |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER_MLA_SPARSE_DSV41` | |
| `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | Full CUDA graphs for decode with DSpark. |
| `--speculative-config` | DSpark, k=5, probabilistic | Adaptive verification is off: this runtime's indexer backend cannot trim verification requests, so enabling it only adds work. |
| `--scheduling-policy` | `priority` | Lower numeric priority first, arrival time as tie-breaker, default priority 0 — so existing callers keep arrival order. |
| `--enable-prefix-caching`, `--async-scheduling`, `--enable-chunked-prefill` | on | |
| `--tool-call-parser` / `--reasoning-parser` / `--tokenizer-mode` | `deepseek_v41` | |
| `DSV41_ENGRAM_DISK` | `1` | See `engram.md`. |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | Serving never contacts a model host. |

Per-rank values (`NODE_RANK`, `MASTER_ADDR`, `VLLM_HOST_IP`, `NCCL_IB_HCA`, the
rendezvous interface, `--headless`) are **not** in `vllm.env`; they are generated
per host into `compose/.env` from `config/topology.env`, so one compose file is
correct everywhere.

## Lifecycle

```
preflight.sh          read-only readiness across all ranks
   │
start.sh              converge hosts, fence, start workers, start head,
   │                  wait for health, wait for containers, run canary
   │
verify-cluster.sh     transport evidence, model-free collective,
   │                  per-edge byte proof, functional acceptance
   │
verify-1m-context.py  hard near-1M acceptance (slow, run deliberately)
   │
stop.sh               checked fence: head first (rendezvous verified free),
                      then workers, then confirm the terminal state
```

Convergence inside `start.sh` is idempotent and installs, per rank: the ring-only
NCCL library, the staged Engram/SM12x patches, the memory boundary, and the
rendered compose files.

Ordering is load-bearing. Workers start first so they are waiting in the
rendezvous path when the head arrives; the head starts last and is fenced first.
There is no automatic restart of a distributed rank: `restart: "no"`, no boot
enablement, and no per-rank unit that could recreate a generation on its own.

## Remote access

The lifecycle scripts use `ssh` with `BatchMode=yes`, `StrictHostKeyChecking=yes`
and your `SSH_OPTS` from the topology file. They never disable host-key
verification, never use a password, and never fall back to another credential
path. Record the hosts in `known_hosts` before the first run; if a key changes,
that is a finding to investigate, not a prompt to accept.

## Expected timeline

A cold start is dominated by loading 510 GB of checkpoint per rank and capturing
CUDA graphs. `start.sh` waits up to `STARTUP_TIMEOUT_SECONDS` (default 3600) for
the head API. A near-1M request takes on the order of 15 minutes of prefill; see
`benchmarks/results/`.
