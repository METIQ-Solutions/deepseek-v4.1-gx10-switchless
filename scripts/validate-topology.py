#!/usr/bin/env python3
"""Validate a switchless-ring topology file before anything touches hardware.

The topology file (config/topology.env) is source-able shell, so this validator
parses it as KEY=VALUE rather than importing YAML. It enforces the physical
rules that make a ring usable, so that a wiring or addressing mistake is caught
before a single rank is started:

  1. every declared variable exists for every rank;
  2. every fabric address is an IPv4 /30;
  3. the two ranks sharing a physical link really are the two usable addresses
     of that one /30 segment (both lanes of both ports);
  4. the control plane never overlaps or reuses the fabric plane;
  5. the ring edges form one cycle in which every rank participates exactly
     twice: once as a source port and once as a destination port;
  6. tensor parallelism times pipeline parallelism equals the rank count.

Usage: python3 scripts/validate-topology.py [config/topology.env]
Exit 0 prints a one-line summary; any violation prints TOPOLOGY_INVALID.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
from pathlib import Path

REQUIRED_CLUSTER = (
    "CLUSTER_NAME",
    "CLUSTER_TP_SIZE",
    "CLUSTER_PP_SIZE",
    "NODE_COUNT",
    "HEAD_RANK",
    "API_PORT",
    "MASTER_PORT",
    "IMAGE",
    "WORKLOAD_ROOT",
    "MODEL_ROOT",
    "CACHE_ROOT",
    "NCCL_LIBRARY",
    "NCCL_GID_INDEX",
    "NCCL_IB_HCA",
)

REQUIRED_RANK = ("NAME", "SSH", "CONTROL_IP", "CONTROL_IFACE", "FABRIC")

LANES_PER_PORT = 2
PORTS_PER_NODE = 2
EDGE = re.compile(r"^(\d+):(\d+)->(\d+):(\d+)$")


class Invalid(Exception):
    pass


def parse(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise Invalid(f"topology file not found: {path}")
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise Invalid(f"{path}:{number}: not a KEY=VALUE line: {line!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Z0-9_]+", key):
            raise Invalid(f"{path}:{number}: invalid key {key!r}")
        if key in values:
            raise Invalid(f"{path}:{number}: duplicate key {key}")
        values[key] = value.strip().strip('"')
    return values


def require(values: dict[str, str], key: str, where: str) -> str:
    value = values.get(key)
    if value is None or value == "":
        raise Invalid(f"{where}: missing required value {key}")
    return value


def positive_int(values: dict[str, str], key: str, where: str) -> int:
    raw = require(values, key, where)
    if not raw.isdigit() or int(raw) < 1:
        raise Invalid(f"{where}: {key} must be a positive integer, got {raw!r}")
    return int(raw)


def parse_fabric(rank: int, raw: str) -> dict[str, ipaddress.IPv4Interface]:
    entries = raw.split()
    expected = PORTS_PER_NODE * LANES_PER_PORT
    if len(entries) != expected:
        raise Invalid(
            f"rank {rank}: FABRIC must list {expected} 'interface:address/prefix' "
            f"entries (port0 lane a, port0 lane b, port1 lane a, port1 lane b), got {len(entries)}"
        )
    lanes: dict[str, ipaddress.IPv4Interface] = {}
    for entry in entries:
        interface, _, address = entry.partition(":")
        if not interface or not address:
            raise Invalid(f"rank {rank}: malformed FABRIC entry {entry!r}")
        if interface in lanes:
            raise Invalid(f"rank {rank}: interface {interface} listed twice in FABRIC")
        try:
            parsed = ipaddress.ip_interface(address)
        except ValueError as error:
            raise Invalid(f"rank {rank}: interface {interface} has an invalid address: {error}") from error
        if parsed.version != 4:
            raise Invalid(f"rank {rank}: {interface} must carry an IPv4 address")
        if parsed.network.prefixlen != 30:
            raise Invalid(
                f"rank {rank}: {interface} must be a /30 point-to-point address, "
                f"got /{parsed.network.prefixlen}"
            )
        lanes[interface] = parsed
    return lanes


def lane_address(lanes: dict[str, ipaddress.IPv4Interface], port: int, lane: int):
    return list(lanes.values())[port * LANES_PER_PORT + lane]


def lane_interface(lanes: dict[str, ipaddress.IPv4Interface], port: int, lane: int):
    return list(lanes.keys())[port * LANES_PER_PORT + lane]


def validate(values: dict[str, str]) -> str:
    node_count = positive_int(values, "NODE_COUNT", "cluster")
    tp_size = positive_int(values, "CLUSTER_TP_SIZE", "cluster")
    pp_size = positive_int(values, "CLUSTER_PP_SIZE", "cluster")
    head_rank = values.get("HEAD_RANK", "")
    if not head_rank.isdigit():
        raise Invalid(f"cluster: HEAD_RANK must be a non-negative integer, got {head_rank!r}")
    head_rank = int(head_rank)
    for key in REQUIRED_CLUSTER:
        require(values, key, "cluster")

    if tp_size * pp_size != node_count:
        raise Invalid(
            f"cluster: CLUSTER_TP_SIZE*CLUSTER_PP_SIZE ({tp_size}*{pp_size}) "
            f"must equal NODE_COUNT ({node_count})"
        )
    if head_rank >= node_count:
        raise Invalid(f"cluster: HEAD_RANK {head_rank} is outside 0..{node_count - 1}")

    fabric: dict[int, dict[str, ipaddress.IPv4Interface]] = {}
    control_ip: dict[int, str] = {}
    control_iface: dict[int, str] = {}
    for rank in range(node_count):
        prefix = f"RANK_{rank}_"
        for key in REQUIRED_RANK:
            require(values, f"{prefix}{key}", f"rank {rank}")
        fabric[rank] = parse_fabric(rank, values[f"{prefix}FABRIC"])
        control_ip[rank] = values[f"{prefix}CONTROL_IP"]
        control_iface[rank] = values[f"{prefix}CONTROL_IFACE"]
        try:
            parsed_control = ipaddress.ip_address(control_ip[rank])
        except ValueError as error:
            raise Invalid(f"rank {rank}: CONTROL_IP is not a valid address: {error}") from error
        if parsed_control.version != 4:
            raise Invalid(f"rank {rank}: CONTROL_IP must be IPv4")

    # Rule 4: the control plane must be a different plane from the fabric.
    for rank in range(node_count):
        if control_iface[rank] in fabric[rank]:
            raise Invalid(
                f"rank {rank}: CONTROL_IFACE {control_iface[rank]} is also a fabric interface; "
                "the rendezvous plane must not be a ConnectX ring interface"
            )
        for interface, lane in fabric[rank].items():
            if ipaddress.ip_address(control_ip[rank]) == lane.ip:
                raise Invalid(f"rank {rank}: CONTROL_IP equals the {interface} fabric address")
            if ipaddress.ip_address(control_ip[rank]) in lane.network:
                raise Invalid(
                    f"rank {rank}: CONTROL_IP {control_ip[rank]} falls inside the "
                    f"{lane.network} fabric segment"
                )
    if len({control_ip[rank] for rank in range(node_count)}) != node_count:
        raise Invalid("cluster: CONTROL_IP values must be unique per rank")

    # Rules 3 and 5: the declared edges must pair the lanes correctly and form
    # exactly one cycle through every rank.
    edges = values.get("RING_EDGES", "").split()
    if len(edges) != node_count:
        raise Invalid(f"cluster: RING_EDGES must contain exactly {node_count} edges, got {len(edges)}")
    as_source: set[int] = set()
    as_target: set[int] = set()
    for raw in edges:
        match = EDGE.fullmatch(raw)
        if not match:
            raise Invalid(f"cluster: malformed ring edge {raw!r} (expected fromRank:fromPort->toRank:toPort)")
        from_rank, from_port, to_rank, to_port = (int(part) for part in match.groups())
        for rank, port in ((from_rank, from_port), (to_rank, to_port)):
            if rank >= node_count:
                raise Invalid(f"cluster: ring edge {raw} references rank {rank} beyond NODE_COUNT")
            if port >= PORTS_PER_NODE:
                raise Invalid(f"cluster: ring edge {raw} references port {port} beyond {PORTS_PER_NODE} ports")
        if from_rank == to_rank:
            raise Invalid(f"cluster: ring edge {raw} connects a rank to itself")
        if from_rank in as_source or to_rank in as_target:
            raise Invalid(f"cluster: ring edge {raw} reuses a rank port already bound by another edge")
        as_source.add(from_rank)
        as_target.add(to_rank)
        for lane in range(LANES_PER_PORT):
            local = lane_address(fabric[from_rank], from_port, lane)
            remote = lane_address(fabric[to_rank], to_port, lane)
            if local.network != remote.network:
                raise Invalid(
                    f"cluster: ring edge {raw} lane {lane} is not one physical segment: "
                    f"{lane_interface(fabric[from_rank], from_port, lane)} {local} and "
                    f"{lane_interface(fabric[to_rank], to_port, lane)} {remote} are on different /30s"
                )
            if local.ip == remote.ip:
                raise Invalid(f"cluster: ring edge {raw} lane {lane} uses the same address at both ends")
    if as_source != set(range(node_count)) or as_target != set(range(node_count)):
        raise Invalid(
            "cluster: the ring must use every rank exactly once as a source port and "
            f"once as a destination port (sources={sorted(as_source)}, targets={sorted(as_target)})"
        )

    used = []
    for rank in range(node_count):
        used.extend(str(lane.ip) for lane in fabric[rank].values())
    if len(set(used)) != len(used):
        raise Invalid("cluster: the same fabric address is assigned to more than one interface")

    # Rule 7: a /30 segment may only ever join a declared ring neighbour pair.
    # If two ranks share a segment without being an edge, they have an
    # *unintended* direct path — a diagonal link — which the Ring-only NCCL
    # contract does not expect and which would change what the fabric proves.
    neighbour_pair = set()
    for raw in edges:
        from_rank, _, to_rank, _ = (int(part) for part in EDGE.fullmatch(raw).groups())
        neighbour_pair.add(frozenset((from_rank, to_rank)))
    segment_users: dict[str, set[int]] = {}
    for rank in range(node_count):
        for lane in fabric[rank].values():
            segment_users.setdefault(str(lane.network), set()).add(rank)
    for segment, ranks in sorted(segment_users.items()):
        if len(ranks) != 2 or frozenset(ranks) not in neighbour_pair:
            raise Invalid(
                f"cluster: {segment} is shared by ranks {sorted(ranks)}, which are not a declared "
                "ring neighbour pair — this is an unintended diagonal dependency"
            )

    return (
        f"TOPOLOGY_OK cluster={values['CLUSTER_NAME']} ranks={node_count} "
        f"tp={tp_size} pp={pp_size} head={head_rank} edges={len(edges)}"
    )


def main() -> int:
    args = sys.argv[1:]
    interfaces_rank = None
    if "--interfaces" in args:
        index = args.index("--interfaces")
        try:
            interfaces_rank = int(args[index + 1])
        except (IndexError, ValueError):
            print("TOPOLOGY_INVALID --interfaces requires a rank number", file=sys.stderr)
            return 2
        del args[index:index + 2]

    path = Path(args[0] if args else "config/topology.env")
    try:
        values = parse(path)
        if interfaces_rank is not None:
            # Emitted for scripts/verify-cluster.sh, which needs the fabric
            # interfaces of one rank without re-implementing topology parsing.
            rank_interfaces = {
                rank: [entry.split(":")[0] for entry in values[f"RANK_{rank}_FABRIC"].split()]
                for rank in range(int(values["NODE_COUNT"]))
            }
            if interfaces_rank not in rank_interfaces:
                raise Invalid(f"rank {interfaces_rank} is outside NODE_COUNT")
            print(json.dumps(rank_interfaces[interfaces_rank]))
            return 0
        print(validate(values))
    except Invalid as error:
        print(f"TOPOLOGY_INVALID {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
