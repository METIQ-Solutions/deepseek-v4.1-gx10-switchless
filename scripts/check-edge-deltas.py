#!/usr/bin/env python3
"""Prove tensor traffic crossed every physical ring edge.

A four-rank collective can succeed over a path the ring cannot actually carry
(Socket over management Ethernet, for instance). Byte counters are the evidence
that the data really traversed each point-to-point edge, so this compares
per-HCA counters taken before and after a workload and requires every declared
edge to have moved at least a minimum number of bytes.

Usage:
  python3 check-edge-deltas.py --topology config/topology.env --samples-dir DIR \
      [--minimum-bytes 1048576] [--management-max-bytes 0]

`samples-dir` must contain `before-rank<N>.json` and `after-rank<N>.json`, each
the output of collect-fabric-metrics.py for that rank.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

EDGE = re.compile(r"^(\d+):(\d+)->(\d+):(\d+)$")
LANES_PER_PORT = 2
METRIC_KEYS = ("port_xmit_data", "port_rcv_data")


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("validate_topology", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load topology validator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Single source of truth for topology parsing: the validator next to this file.
validator = load_validator(Path(__file__).with_name("validate-topology.py"))


def edge_lanes(topology: dict[str, str], from_rank: int, from_port: int, to_rank: int, to_port: int):
    """Yield (rank, interface) for the two lanes of one physical edge."""
    for lane in range(LANES_PER_PORT):
        for rank, port in ((from_rank, from_port), (to_rank, to_port)):
            entries = topology[f"RANK_{rank}_FABRIC"].split()
            interface = entries[port * LANES_PER_PORT + lane].split(":")[0]
            yield rank, interface


def counter_total(sample: dict, interface: str) -> int:
    fabric = sample.get("fabric") or {}
    entry = fabric.get(interface)
    if entry is None:
        raise SystemExit(f"sample does not contain counters for {interface}")
    return sum(int(entry.get(key, 0)) for key in METRIC_KEYS)


def run(topology_path: str, samples_dir: str, minimum_bytes: int,
        management_max_bytes: int = 0) -> int:
    """Return 0 when every declared edge carried its minimum traffic."""
    topology = validator.parse(Path(topology_path))
    validator.validate(topology)

    node_count = int(topology["NODE_COUNT"])
    samples = Path(samples_dir)
    before = {}
    after = {}
    for rank in range(node_count):
        for label, target in (("before", before), ("after", after)):
            path = samples / f"{label}-rank{rank}.json"
            if not path.is_file():
                raise SystemExit(f"missing counter sample {path}")
            target[rank] = json.loads(path.read_text(encoding="utf-8"))

    failures = 0
    print("edge                 moved_bytes        threshold  result")
    for raw in topology["RING_EDGES"].split():
        match = EDGE.fullmatch(raw)
        from_rank, from_port, to_rank, to_port = (int(part) for part in match.groups())
        deltas = {}
        for rank, interface in edge_lanes(topology, from_rank, from_port, to_rank, to_port):
            delta = counter_total(after[rank], interface) - counter_total(before[rank], interface)
            if delta < 0:
                raise SystemExit(
                    f"counter for {interface} on rank {rank} decreased; the sample pair is not monotonic"
                )
            deltas[(rank, interface)] = delta
        moved = sum(deltas.values())
        ok = moved >= minimum_bytes
        print(f"{raw:<20} {moved:>16} {minimum_bytes:>18} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            detail = ", ".join(f"rank{r}:{iface}={value}" for (r, iface), value in sorted(deltas.items()))
            print(f"  edge {raw} moved only {moved} bytes across {detail}", file=sys.stderr)

    if management_max_bytes > 0:
        management = 0
        for rank in range(node_count):
            before_mgmt = before[rank]["management"]
            after_mgmt = after[rank]["management"]
            management += (after_mgmt["tx_bytes"] - before_mgmt["tx_bytes"]) + (
                after_mgmt["rx_bytes"] - before_mgmt["rx_bytes"]
            )
        ok = management <= management_max_bytes
        print(f"{'management plane':<20} {management:>16} {management_max_bytes:>18} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            print(
                f"management plane moved {management} bytes, above the {management_max_bytes} limit; "
                "tensor traffic may be falling back to Ethernet",
                file=sys.stderr,
            )

    if failures:
        print(f"EDGE_DELTA_FAIL failures={failures}", file=sys.stderr)
        return 1
    print(f"EDGE_DELTA_OK edges={len(topology['RING_EDGES'].split())} minimum_bytes={minimum_bytes}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="config/topology.env")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--minimum-bytes", type=int, default=1048576)
    parser.add_argument(
        "--management-max-bytes",
        type=int,
        default=0,
        help="fail if management bytes exceed this (0 disables the check)",
    )
    args = parser.parse_args()
    return run(args.topology, args.samples_dir, args.minimum_bytes, args.management_max_bytes)


if __name__ == "__main__":
    sys.exit(main())
