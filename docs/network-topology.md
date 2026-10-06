# Network topology

## Two planes, deliberately separate

```
        management 10 GbE (rendezvous, SSH, control)          one reachable network
        ┌───────────────┬───────────────┬───────────────┐
        │               │               │               │
     ┌──┴───┐        ┌──┴───┐        ┌──┴───┐        ┌──┴───┐
     │rank 0│        │rank 1│        │rank 2│        │rank 3│
     │ head │        │worker│        │worker│        │worker│
     └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘
   port0│      port1    │      port1    │      port1    │
        │   ┌───────────┘   ┌───────────┘   ┌───────────┘
        └───┘  port0        └───────────────┘  port0
        ▲                                       ▲
        └───────────── physical cycle ──────────┘
             2 × 200 GbE ConnectX, /30 point-to-point, no switch
```

* **Management plane** — a single network every rank can reach. SSH, the vLLM
  rendezvous (`tcp://<head>:25001`) and the Gloo/TP socket interfaces live here.
  This is what allows bootstrap to be all-to-all even though the fabric is not.
* **Fabric plane** — direct ConnectX links only. `NCCL_ALGO=Ring` with the
  ring-only NCCL build keeps *tensor* traffic here.

The two must not overlap. `scripts/validate-topology.py` fails if a control
address falls inside a fabric `/30`, if a control interface is also a fabric
interface, or if control addresses are duplicated.

## Why a ring, and why two lanes

Each node's port 0 connects to the next node's port 1, forming one physical
cycle 0→1→2→3→0. Every edge is two independent point-to-point physical links
("lanes"), which is why each node declares **four** fabric addresses: two lanes
on each of two ports.

Point-to-point `/30` segments are used because they make the addressing
statement *true*: each side of a link has exactly one usable peer address. A
broader prefix would let traffic reach a non-adjacent rank through a route that
the ring cannot actually carry, producing a fast-looking, wrong result.

There is deliberately **no** all-to-all fabric reachability: rank 2 and rank 3 do
not need to route to rank 0's port 0 address. Do not add multi-hop fabric routes
or enable IP forwarding to make arbitrary fabric addresses reachable.

## Link configuration

* IPv4 `/30` addressing as declared in `config/topology.env`.
* **MTU 9000** on every fabric interface — a formality on modern hardware, but
  silent MTU mismatch shows up as unexplained collective slowness.
* **No default route** on any fabric interface. A fabric lane must never be a
  candidate egress path for ordinary traffic.
* `dhcp4: false`; addresses are static and match the topology file exactly.
  `scripts/host-preflight.sh` verifies the address, the MTU and the absence of a
  default route per interface.

If you need to apply addressing changes with Netplan, note that
`netplan apply` restarts NetworkManager and briefly disconnects **all** ConnectX
interfaces on the target, even when one port changes. Stop the deployment
first, then apply, then re-run `scripts/preflight.sh`.

## NCCL environment, and why each value is there

| Variable | Value | Reason |
| --- | --- | --- |
| `NCCL_NET` | `IB` | The ring is native RoCE; no plugin. |
| `NCCL_ALGO` | `Ring` | The only collective that maps onto a physical cycle. |
| `NCCL_SWITCHLESS_RING_ONLY` / `NCCL_SKIP_TREE_CONNECT` | `1` | Suppress Tree and PAT transport setup, which would otherwise try to reach non-neighbour ranks. Implemented by the ring-only build. |
| `NCCL_IB_SUBNET_AWARE_ROUTING` / `NCCL_IB_SUBNET_PREFIX_LEN` | `1` / `30` | Match peers by their point-to-point subnet. |
| `NCCL_IB_MERGE_NICS` | `0` | Merging NICs would hide which physical lane carries traffic; the hardened build refuses a non-zero value. |
| `NCCL_IB_HCA` | `=rocep1s0f0:1,rocep1s0f1:1` | Exact selection. The `=` prefix means "use these and fail otherwise" — never let NCCL choose. |
| `NCCL_CROSS_NIC` | `1` | Distribute channels across both HCAs. |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `4` / `4` | Pin the channel count so the two-lane wiring is used predictably. |
| `NCCL_P2P_LEVEL` | `SYS` | Direct peer access within the node. |
| `NCCL_PROTO` | `LL,LL128,Simple` | Protocols the ring supports. |
| `NCCL_CUMEM_ENABLE` | `0` | Avoids allocations outside the accounted memory boundary. |
| `NCCL_NVLS_ENABLE` | `0` | No NVLink SHARP on this topology. |
| `NCCL_IB_ADDR_FAMILY` / `NCCL_IB_ROCE_VERSION_NUM` | `AF_INET` / `2` | IPv4 RoCE v2, matching the `/30` plan. |
| `NCCL_SOCKET_FAMILY` / `NCCL_SOCKET_IFNAME` | `AF_INET` / management interface | Bootstrap on the management plane, never on a fabric lane. |

## RoCE GID index

Every rank must use the same IPv4 RoCE-v2 GID index on **both** selected HCAs.
NCCL advertises listener GIDs from both ports and peers match them by subnet, so
if the two HCAs disagree, some lanes silently carry nothing
(`scripts/discover-roce-gids.py`). `scripts/preflight.sh` discovers the common
index and fails if it differs from the value declared in `config/topology.env`,
so a driver or firmware change that moves the index is caught before a start
rather than during a collective.

Linux exposes unpopulated GID slots whose sysfs attributes return `EINVAL`; the
discovery skips those only *after* checking their type and netdev, and still
fails on a selected GID that cannot be read.

## Proving the fabric is used

Two independent proofs, both in `scripts/verify-cluster.sh`:

1. the collective succeeds over native IB — `Using network IB` / `NET/IB` in the
   logs, with `SWITCHLESS/HARDENED` markers, and no `NET/Socket` or `NET/Mesh`;
2. per-HCA byte counters on **both ends of every physical edge** move by at
   least `MIN_EDGE_BYTES` (default 1 MiB) across a real prefill.

Proof 1 alone is insufficient: a collective can succeed over a path the ring
cannot carry. Proof 2 alone is insufficient: counters move even when the wrong
transport is used. Together they say "the ring carried the tensor data".
