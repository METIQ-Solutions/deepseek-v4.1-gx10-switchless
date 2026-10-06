#!/usr/bin/env python3
"""Find the single IPv4 RoCE-v2 GID index shared by the selected ConnectX HCAs.

The switchless ring relies on every rank using the same GID index on both
selected HCAs, because NCCL advertises the listener GIDs from both ports and
the peer matches them by subnet. If the two HCAs disagree, some lanes silently
never carry RDMA, so this is checked explicitly instead of left to NCCL.

Empty sysfs GID slots on Linux can return EINVAL when read. Those are skipped
only after their type/netdev have been examined; a *selected* GID that cannot
be read is still a hard failure.

Usage:
  python3 discover-roce-gids.py '<json {iface: addr/prefix, ...}>' '<NCCL_IB_HCA>'

NCCL_IB_HCA must be the exact-selection form, e.g. "=rocep1s0f0:1,rocep1s0f1:1".
"""

from __future__ import annotations

import errno
import ipaddress
import json
import re
import sys
from pathlib import Path

SYSFS_ROOT = Path("/sys/class/infiniband")
# Mellanox netdev names encode the PCI domain, bus, slot and function, with an
# optional uppercase domain marker: enp1s0f0np0, enP2p1s0f1np1.
MELLANOX_INTERFACE = re.compile(r"^en(P?)(\d*)p(\d+)s(\d+)f(\d+)np(\d+)$")


class GidError(Exception):
    pass


def hca_for_interface(interface: str) -> str:
    """Map a kernel netdev name to its Mellanox RoCE HCA name.

    enp1s0f0np0    -> rocep1s0f0
    enP2p1s0f1np1  -> roceP2p1s0f1
    """
    match = MELLANOX_INTERFACE.fullmatch(interface)
    if not match:
        raise GidError(
            f"cannot derive an HCA name from interface {interface!r}; "
            "this helper expects Mellanox-style names such as enp1s0f0np0"
        )
    domain_marker, domain, bus, slot, function, _port = match.groups()
    return f"roce{domain_marker}{domain}p{bus}s{slot}f{function}"


def read_attribute(path: Path, allow_unpopulated: bool = False) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError as error:
        if allow_unpopulated and error.errno == errno.EINVAL:
            return None
        raise GidError(f"cannot read GID attribute {path}: {error}") from error


def discover(addresses: dict[str, str], selected_hcas: str, root: Path = SYSFS_ROOT) -> dict:
    if not isinstance(addresses, dict) or not addresses:
        raise GidError("at least one fabric interface address is required")
    if not isinstance(selected_hcas, str) or not selected_hcas.startswith("="):
        raise GidError("NCCL_IB_HCA must select exact devices with a leading '='")

    declared = [item for item in selected_hcas[1:].split(",") if item]
    if len(declared) != 2 or len(set(declared)) != 2:
        raise GidError("exactly two distinct RoCE HCAs must be selected")

    # A declared entry is "device:port" (e.g. "rocep1s0f0:1"); the sysfs path
    # uses the device name, the selection key keeps the port.
    device_of: dict[str, str] = {}
    port_of: dict[str, str] = {}
    for entry in declared:
        device, _, port = entry.partition(":")
        if not device or not port.isdecimal():
            raise GidError(f"NCCL_IB_HCA entry {entry!r} must be 'device:port'")
        device_of[entry] = device
        port_of[entry] = port

    interface_by_hca = {hca_for_interface(interface) + ":1": interface for interface in addresses}
    missing = [hca for hca in declared if hca not in interface_by_hca]
    if missing:
        raise GidError(
            f"selected HCAs do not correspond to the declared fabric interfaces: {missing} "
            f"(derived from {sorted(interface_by_hca)})"
        )

    selected = [interface_by_hca[hca] for hca in declared]
    functions = {re.search(r"f(\d+)np\d+$", interface).group(1) for interface in selected}
    if functions != {"0", "1"}:
        raise GidError("the two selected HCAs must cover both physical ports of the NIC")

    eligible: dict[str, dict] = {}
    for hca in declared:
        interface = interface_by_hca[hca]
        address = ipaddress.ip_interface(addresses[interface])
        if address.version != 4 or address.network.prefixlen != 30:
            raise GidError(f"{interface} must have an IPv4 /30 address, got {address}")
        device = device_of[hca]
        port_number = port_of[hca]
        port = root / device / "ports" / port_number
        if not port.is_dir():
            raise GidError(f"{device} port {port_number} is not present under {root}")
        indices: set[int] = set()
        for gid_type in sorted((port / "gid_attrs" / "types").iterdir(), key=lambda item: item.name):
            if not gid_type.name.isdecimal():
                raise GidError(f"non-numeric GID index on {device}: {gid_type.name}")
            if read_attribute(gid_type, allow_unpopulated=True) != "RoCE v2":
                continue
            if read_attribute(port / "gid_attrs" / "ndevs" / gid_type.name, allow_unpopulated=True) != interface:
                continue
            try:
                gid = ipaddress.IPv6Address(read_attribute(port / "gids" / gid_type.name))
            except (ValueError, GidError) as error:
                raise GidError(f"{device} has an unreadable selected GID at index {gid_type.name}: {error}") from error
            if gid.ipv4_mapped == address.ip:
                indices.add(int(gid_type.name))
        if not indices:
            raise GidError(f"{device} has no IPv4 RoCE-v2 GID for {interface} {address.ip}")
        eligible[hca] = {"interface": interface, "ip": str(address.ip), "indices": indices}

    common = set.intersection(*(device["indices"] for device in eligible.values()))
    if len(common) != 1:
        raise GidError(
            "no unique common IPv4 RoCE-v2 GID index across the selected HCAs: "
            + ", ".join(f"{hca}={sorted(device['indices'])}" for hca, device in sorted(eligible.items()))
        )
    return {
        "index": common.pop(),
        # Keyed by HCA device name so the output reads like the sysfs tree; the
        # configured selection entry is device + ":" + port.
        "devices": {
            device_of[hca]: {
                "interface": device["interface"],
                "ip": device["ip"],
                "port": int(port_of[hca]),
            }
            for hca, device in sorted(eligible.items())
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise GidError("usage: discover-roce-gids.py '<addresses-json>' '<NCCL_IB_HCA>'")
    try:
        addresses = json.loads(sys.argv[1])
    except json.JSONDecodeError as error:
        raise GidError(f"addresses argument is not JSON: {error}") from error
    print(json.dumps(discover(addresses, sys.argv[2]), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GidError, OSError) as error:
        print(f"GID_CHECK_FAIL {error}", file=sys.stderr)
        sys.exit(1)
