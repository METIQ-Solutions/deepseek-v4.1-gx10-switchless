#!/usr/bin/env python3
"""Read this host's fabric and management byte counters.

Used by scripts/verify-cluster.sh to prove that tensor data actually crossed
each physical ring edge, rather than only that the collective succeeded (a
collective can succeed over a path the ring cannot carry, which is exactly the
failure mode the ring-only NCCL build exists to prevent).

Counters are read from sysfs, so this needs no elevated privileges and never
touches the network. Output is JSON on stdout.

Usage:
  python3 collect-fabric-metrics.py '<json {node: [iface, ...], ...}>' management_iface
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SYSFS_IB = Path("/sys/class/infiniband")
SYSFS_NET = Path("/sys/class/net")
# Must match scripts/discover-roce-gids.py: the optional uppercase domain marker
# and the PCI domain digits come before the bus ("enP2p1s0f1np1").
MELLANOX_INTERFACE = re.compile(r"^en(P?)(\d*)p(\d+)s(\d+)f(\d+)np(\d+)$")
COUNTERS = ("port_xmit_data", "port_rcv_data")


class MetricError(Exception):
    pass


def hca_for_interface(interface: str) -> str:
    match = MELLANOX_INTERFACE.fullmatch(interface)
    if not match:
        raise MetricError(f"cannot derive an HCA name from interface {interface!r}")
    domain_marker, domain, bus, slot, function, _port = match.groups()
    return f"roce{domain_marker}{domain}p{bus}s{slot}f{function}"


def read_counter(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except OSError as error:
        raise MetricError(f"cannot read counter {path}: {error}") from error
    except ValueError as error:
        raise MetricError(f"counter {path} is not an integer") from error
    if value < 0:
        raise MetricError(f"counter {path} is negative")
    return value


def fabric(interfaces: list[str]) -> dict:
    result: dict[str, dict] = {}
    for interface in interfaces:
        hca = hca_for_interface(interface)
        port = SYSFS_IB / hca / "ports" / "1" / "counters"
        if not port.is_dir():
            raise MetricError(f"HCA port not found for {interface}: {port}")
        result[interface] = {
            "hca": hca,
            "port": 1,
            **{name: read_counter(port / name) for name in COUNTERS},
        }
    return result


def management(interface: str) -> dict:
    base = SYSFS_NET / interface / "statistics"
    if not base.is_dir():
        raise MetricError(f"management interface not found: {interface}")
    return {
        "interface": interface,
        "tx_bytes": read_counter(base / "tx_bytes"),
        "rx_bytes": read_counter(base / "rx_bytes"),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise MetricError("usage: collect-fabric-metrics.py '<interfaces-json>' <management-interface>")
    interfaces = json.loads(sys.argv[1])
    if not isinstance(interfaces, list) or not interfaces:
        raise MetricError("a non-empty list of fabric interfaces is required")
    print(json.dumps(
        {"fabric": fabric(interfaces), "management": management(sys.argv[2])},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MetricError, json.JSONDecodeError, OSError) as error:
        print(f"METRICS_FAIL {error}", file=sys.stderr)
        sys.exit(1)
