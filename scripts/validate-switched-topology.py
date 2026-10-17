#!/usr/bin/env python3
"""Validate the isolated one-HCA switched logical-ring topology."""
from __future__ import annotations
import ipaddress
import re
import sys
from pathlib import Path

EDGE = re.compile(r"^(\d+):0->(\d+):0$")

def parse(path: Path) -> dict[str, str]:
    values = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values

def require(values, key):
    if not values.get(key):
        raise ValueError(f"missing {key}")
    return values[key]

def validate(values):
    if values.get("TOPOLOGY_MODE") != "switched-ring-one-hca":
        raise ValueError("TOPOLOGY_MODE must be switched-ring-one-hca")
    if values.get("NODE_COUNT") != "4" or values.get("CLUSTER_TP_SIZE") != "4" or values.get("CLUSTER_PP_SIZE") != "1":
        raise ValueError("switched mode requires four ranks with TP4/PP1")
    hca = require(values, "NCCL_IB_HCA")
    if not re.fullmatch(r"=roce[^,:]+:1", hca):
        raise ValueError("switched mode requires exactly one selected HCA port")
    addresses = []
    network = None
    for rank in range(4):
        prefix = f"RANK_{rank}_"
        fabric = require(values, prefix + "FABRIC").split()
        if len(fabric) != 1:
            raise ValueError(f"rank {rank} must declare exactly one switched fabric endpoint")
        interface, address = fabric[0].split(":", 1)
        parsed = ipaddress.ip_interface(address)
        if parsed.version != 4 or parsed.network.prefixlen != 24:
            raise ValueError(f"rank {rank} endpoint must be IPv4 /24")
        if network is None:
            network = parsed.network
        if parsed.network != network:
            raise ValueError("all switched endpoints must share one /24")
        addresses.append(parsed.ip)
        control = ipaddress.ip_address(require(values, prefix + "CONTROL_IP"))
        if control in parsed.network:
            raise ValueError(f"rank {rank} control address overlaps switched fabric")
        if require(values, prefix + "CONTROL_IFACE") == interface:
            raise ValueError(f"rank {rank} control interface overlaps fabric interface")
    if len(set(addresses)) != 4:
        raise ValueError("switched fabric addresses must be unique")
    expected = {(0, 1), (1, 2), (2, 3), (3, 0)}
    edges = set()
    for raw in require(values, "RING_EDGES").split():
        match = EDGE.fullmatch(raw)
        if not match:
            raise ValueError(f"invalid switched logical-ring edge {raw}")
        edges.add((int(match.group(1)), int(match.group(2))))
    if edges != expected:
        raise ValueError(f"logical ring must be exactly {sorted(expected)}, got {sorted(edges)}")
    return f"TOPOLOGY_OK mode=switched-ring-one-hca ranks=4 tp=4 pp=1 network={network} hca={hca}"

def main():
    try:
        print(validate(parse(Path(sys.argv[1] if len(sys.argv) > 1 else "config/topology.switched-ring-one-hca.env"))))
    except (OSError, ValueError, IndexError) as error:
        print(f"TOPOLOGY_INVALID {error}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())