# Architecture

## Shape of the deployment

Four GB10 nodes serve one DeepSeek V4.1 Flash checkpoint with tensor
parallelism 4. Rank 0 is the head: it owns the API and the rendezvous. Ranks 1–3
are headless workers in the same distributed process group.

```
        client ──HTTP──▶ rank 0 (head) :8889
                            │  TP4, NCCL over the direct ring
              ┌─────────────┼─────────────┬─────────────┐
              ▼             ▼             ▼             ▼
           rank 0        rank 1        rank 2        rank 3
           (GPU)         (GPU)         (GPU)         (GPU)
           Engram/NVMe   Engram/NVMe   Engram/NVMe   Engram/NVMe
              ▲             ▲             ▲             ▲
              └──── switchless ConnectX ring (2×200 GbE) ────┘
                            ▲
                    bootstrap on the 10 GbE management plane
```

## Why it is split this way

* **One rank per node.** Each node's 128 GiB unified memory must hold its share
  of a 510 GB checkpoint plus KV cache. There is no room for a second rank.
* **TP4 rather than PP.** The ring is a low-latency, high-bandwidth neighbour
  fabric: tensor parallelism's all-reduce pattern maps onto it directly, and a
  pipeline would add latency without removing the memory pressure.
* **Two planes.** Bootstrap needs all-to-all reachability; the ring deliberately
  does not provide it. Keeping rendezvous on the management plane is what allows
  tensor traffic to stay on the point-to-point links (see `network-topology.md`).
* **Disk-backed Engram.** ~203 GiB of tables cannot be resident in 128 GiB
  (see `engram.md`).
* **Ring-only NCCL.** Stock NCCL's Tree/PAT setup reaches for non-adjacent ranks
  (see `lessons-learned.md` §1).

## Component map

| Path | Responsibility |
| --- | --- |
| `docker/` | Build the runtime: pinned base + verified ring-only NCCL. |
| `docker/patches/switchless/` | The auditable description of the ring-only behaviour. |
| `config/topology.env` | The single machine-readable description of the cluster: ranks, addresses, edges, image, paths, containment. |
| `config/vllm.env` | The serving configuration that is identical on every rank. |
| `compose/compose.yaml` | One rank: image, mounts, devices, environment, `vllm serve` command line, cgroup parent. |
| `scripts/validate-topology.py` | Rejects a topology that cannot physically work before anything touches hardware. |
| `scripts/preflight.sh` + `host-preflight.sh` | Read-only readiness across all ranks. |
| `scripts/install-*.sh`, `fetch-engram-patches.sh` | Idempotent per-host convergence: ring-only NCCL, staged patches, memory boundary. |
| `scripts/start.sh` / `stop.sh` | Coordinated start (workers first) and checked fence (head first). |
| `scripts/verify-cluster.sh` | Transport proof, model-free collective, per-edge byte proof, functional acceptance. |
| `scripts/verify-1m-context.py` | Hard near-1M acceptance. |
| `scripts/lib.sh` | Shared topology loading, remote execution, health waits, fence. |
| `benchmarks/` | Measurement tooling and measured results. |

## Trust boundaries

* **Model data** is read-only and local. The serving container mounts the model
  root `:ro`, and the runtime is offline (`HF_HUB_OFFLINE=1`). Serving never
  fetches weights.
* **Third-party binaries** enter only through `scripts/install-switchless-nccl.sh`
  (pinned release URL, archive SHA-256, library SHA-256, build marker) and
  `scripts/fetch-engram-patches.sh` (pinned revision, per-file SHA-256).
* **Remote execution** is `ssh` with strict host-key checking and batch mode.
* **No credentials** are stored in this repository or in the rendered compose
  files. The image is expected to be present locally (`pull_policy: never`), so
  the lifecycle never needs registry credentials.
* **The container** runs with `network_mode: host`, `ipc: host`,
  `shm_size: 64gb`, all GPUs, and `/dev/infiniband` — the minimum needed for
  multi-node RDMA inference. It is placed below a systemd memory slice rather
  than given an unbounded host.

## What is intentionally absent

This is a single-deployment reference implementation. It contains no
orchestrator, no multi-tenant scheduling, no model gateway, no monitoring stack
and no CI system. The lifecycle is four shell scripts, one Python acceptance
gate and one benchmark driver, all of which you can read in full.
