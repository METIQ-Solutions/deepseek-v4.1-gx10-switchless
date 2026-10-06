# Safety: memory containment and failure behaviour

On a unified-memory GB10 node, "out of memory" is not a clean exception. It is a
node-level event: the host can become unresponsive, the container can be killed
mid-collective, and — in the worst observed case — the node resets. The
containment in this repository exists to keep a memory-pressure event bounded,
observable and recoverable.

## Host-memory boundary

Each rank runs inside a systemd slice that the container is placed *below*:

```ini
[Slice]
MemoryHigh=104G
MemoryMax=112G
MemorySwapMax=0
```

Installed by `scripts/install-memory-boundary.sh`; `compose/compose.yaml` sets
`cgroup_parent` to that slice, and `scripts/preflight.sh` fails if the slice is
not active or the container is not attached to it.

Why these choices:

* **The boundary is a parent of the container, not the container's own limit.**
  A container-level limit would not cover the pinned host memory, the tested
  CUDA/unified-memory charges, and the container's other processes together.
* **`MemoryHigh` and `MemoryMax` differ.** `MemoryHigh` is the earlier
  throttling/pressure boundary; `MemoryMax` is the hard ceiling. A single equal
  value removes the early warning and makes the first symptom a hard kill.
* **No swap.** Reclaiming to swap on a unified-memory node converts a slow path
  into a failure mode with a much worse failure signature.
* **The values are derived from ~127.5 GiB usable per node, the lowest observed
  idle `MemAvailable`, and an explicit reserve for the operating system and the
  management plane.**

For reference, an independent four-GB10 deployment that measured host headroom
by hand reported idle `MemAvailable` of 6 GiB (head) / 8 GiB (others) and
benchmark lows of 3 GiB / 4 GiB, and found that `gpu_memory_utilization=0.88`
exhausted host memory. Their approach was host-headroom-based; this one is
cgroup-based and stricter, which is deliberate.

## Startup headroom gate

Before a start, `scripts/preflight.sh` requires `STARTUP_HEADROOM_GIB` (default
16 GiB) of `MemAvailable` on each rank. This is intentionally much higher than
the observed floor of a *running* deployment, because the expensive moment is
model load plus CUDA graph capture, not steady state.

## Refusing to start on top of a running generation

`start.sh` fences first and only then starts. `stop.sh` and the fence inside
`lib.sh` verify the ownership labels of the container they are about to stop
(project and service), resolve the full container ID, and stop only that ID.
A same-named container belonging to another Compose project is **refused**, not
stopped.

## Failure behaviour

* `restart: "no"` and no boot enablement: a rank never independently
  reconstructs a distributed generation. After a reboot the nodes are
  compute-idle until a coordinated `start.sh`.
* The head is fenced first, and its rendezvous endpoint must be confirmed free.
  If an unexpected process still owns the port, the fence **fails** and does not
  kill it — an unknown owner is an investigation, not a cleanup task.
* Distinct interruption classes are never conflated: a Docker/API failure is
  never reported as "the container is absent". Assuming absence on a broken
  daemon would let two generations coexist.
* The canary requires an actually generated token, so "healthy API + usable
  distributed inference" is one pass/fail fact rather than two optimistic ones.

## Things this repository deliberately does not do

* It does not raise a limit to make a run succeed. If a limit must be raised,
  that is a finding about the workload, not a configuration tweak.
* It does not disable TLS or SSH host-key verification.
* It does not fetch and hold credentials: no image pull of a private registry is
  part of the lifecycle. Build and distribute the image by whatever means your
  environment already trusts.
* It does not assume a Hugging Face token. If your access to the checkpoint is
  gated, obtain access through your normal means and populate `MODEL_ROOT`
  yourself.
