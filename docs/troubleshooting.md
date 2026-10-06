# Troubleshooting

Symptom, cause, action. Every entry here came from a real failure.

## Preflight

**`CHECK fabric FAIL ... mtu=1500`**
MTU is not 9000 on a fabric lane. Silent MTU mismatch looks like unexplained
collective slowness rather than an error. Fix the interface MTU and re-run.

**`CHECK fabric FAIL ... address-mismatch`**
The live address does not match `config/topology.env`. Either the topology file
or the host is wrong — do not "fix" this by editing the file to match the host
until you have confirmed which one is intended. `netplan apply` briefly
disconnects **all** ConnectX interfaces on the target, even when one port
changes, so stop the deployment first.

**`CHECK fabric FAIL ... default-route`**
A fabric lane has a default route. Remove it: a point-to-point ring lane must
never be an egress candidate for ordinary traffic.

**`CHECK roce_gid FAIL no unique common IPv4 RoCE-v2 GID index`**
The two selected HCAs do not share a RoCE-v2 IPv4 GID index, so some lanes would
carry nothing while the collective still "works" over the others. Check the
address family, the RoCE version, and that both HCAs are actually the ones you
cabled. Do not guess an index — the discovery output names the affected device.

**`CHECK roce_gid FAIL observed common index 5, topology declares 3`**
A driver or firmware change moved the index. Update `NCCL_GID_INDEX` only after
confirming the new index really is the IPv4 RoCE-v2 GID of the declared
addresses.

**`CHECK headroom FAIL ... need 16 GiB`**
Something else holds memory on that node, or a previous generation did not
fully exit. Check for a stopped-but-present container and for stale lock
files, then retry. Do not lower the gate to make the start proceed.

**`CHECK memory_boundary FAIL ... run install-memory-boundary.sh`**
The slice is not installed or not active. `start.sh` installs it during
convergence; running `preflight.sh` before the first `start.sh` is expected to
report this.

## Start

**The head never becomes healthy within the timeout**
Look at the head's container logs. Common causes, in order of likelihood:
a missing or incomplete model root (check the shard count); Engram staging
failing (`fetch-engram-patches.sh` output); or a rank that never entered the
rendezvous path. Confirm every worker container is running — the head will wait
indefinitely for a peer that never joined.

**`FENCE_FAIL ... belongs to project=… service=…, refusing to stop it`**
A container with the expected name belongs to a different Compose project. This
is the fence doing its job. Investigate what owns it; do not delete it to make
the start proceed.

**`FENCE_FAIL tcp://<head>:25001 is still listening after the fence`**
An unexpected process owns the rendezvous port. The fence deliberately does not
kill it. Identify the process before proceeding.

**The canary fails while `/health` is fine**
The API answers before the communicator is usable. Check the container logs for
whether the ranks actually connected. If `NET/Socket` or `NET/Mesh` appears, the
ring-only transport is not in effect — see below.

## Transport

**Logs contain `via NET/Socket` or `NET/Mesh`**
The tensor transport is not the ring. This is a hard failure. Check that
`/opt/switchless-nccl/libnccl.so.2.30.7` is mounted and that
`/proc/<pid>/maps` in the container shows *only* that library — the three NCCL
lookup paths inside the image must all resolve to it.

**Logs lack `SWITCHLESS/HARDENED` or `Connected all rings`**
The ring-only build was not loaded (the stock NCCL is in use), or the flags were
lost. Verify the library and the environment, then restart.

**The collective times out**
Check, in order: that every fabric address is `/30` and matches the peer's
segment (one physical link = one `/30`); that both lanes of every edge are
correctly paired per `RING_EDGES`; that MTU matches; that no default route
exists on a fabric lane. `scripts/validate-topology.py` catches the first two
before a start.

**`EDGE_DELTA_FAIL` — an edge moved fewer bytes than the threshold**
The collective succeeded without that edge carrying its share. Typically a lane
is down, one port is not cabled as declared, or traffic is being placed on the
other lane exclusively. Verify physical cabling against `RING_EDGES`; verify the
counters move on both ends of the named edge.

## Serving

**Near-1M fails while shorter contexts work**
The measured prompt must be at least 95% of 1,048,576 tokens; a report below
that is rejected rather than passed. If the length is right but the retrieval
code does not come back, suspect KV-cache pressure or an Engram staging problem
before suspecting the model. Re-run after a restart and after a warm period to
distinguish a cold-cache artefact from a real defect.

**TTFT rises sharply under concurrency**
Expected for a single large request consuming the whole per-step budget. See
`lessons-learned.md` §2 for the mechanism; the validated configuration
deliberately keeps `--max-num-batched-tokens 8192`.

**Throughput dropped by ~1.5× with no configuration change**
Check the GPU clock state. GB10 parts can latch into a slow state that moves
results by up to ~1.5× and survives until the node is fully powered off
(unplugged), not merely rebooted. Measure clocks before and after every
benchmark block.

**Peak memory errors, container killed, or the node resets**
Host memory, not GPU memory, is the usual culprit. Confirm the container is
attached to the memory slice, that nothing else is consuming the node, and that
Engram is disk-backed (`DSV41_ENGRAM_DISK=1`) rather than pinned in RAM. Do not
raise the limits to get past this.

**An "independent prefill" result looks too fast**
Check `vllm:prefix_cache_hits_total`. If it moved, a reused prefix produced the
result, not the scheduler.

## Getting more detail

* `docker logs <container>` — NCCL INIT/NET/GRAPH lines are part of the
  validated configuration, not debugging leftovers.
* `python3 scripts/validate-topology.py config/topology.env` — pure static check.
* `python3 -m unittest benchmarks.scripts.test_benchmark -v` — verifies the
  measurement tooling itself, with no server involved.
* `scripts/preflight.sh --rank N` — one node at a time.
