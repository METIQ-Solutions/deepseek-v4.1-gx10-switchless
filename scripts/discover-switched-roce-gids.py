#!/usr/bin/env python3
"""Discover the common IPv4 RoCE-v2 GID for one selected switched HCA."""
from __future__ import annotations
import errno
import ipaddress
import json
import re
import sys
from pathlib import Path

ROOT=Path('/sys/class/infiniband')
MELLANOX=re.compile(r'^en(P?)(\d*)p(\d+)s(\d+)f(\d+)np(\d+)$')

def hca_for_interface(interface):
    match=MELLANOX.fullmatch(interface)
    if not match: raise ValueError(f'cannot derive HCA from {interface}')
    marker,domain,bus,slot,function,_=match.groups()
    return f'roce{marker}{domain}p{bus}s{slot}f{function}'

def read(path, allow_empty=False):
    try: return path.read_text(encoding='ascii').strip()
    except OSError as error:
        if allow_empty and error.errno == errno.EINVAL: return None
        raise

def discover(addresses, selected, root=ROOT):
    if not selected.startswith('=') or len(selected[1:].split(',')) != 1:
        raise ValueError('switched mode requires exactly one exact HCA selection')
    entry=selected[1:]
    device, sep, port = entry.partition(':')
    if not sep or not port.isdecimal(): raise ValueError('HCA must be device:port')
    if len(addresses) != 1: raise ValueError('switched mode requires one fabric interface')
    interface, raw = next(iter(addresses.items()))
    address=ipaddress.ip_interface(raw)
    if address.version != 4 or address.network.prefixlen != 24:
        raise ValueError('switched endpoint must be IPv4 /24')
    derived=hca_for_interface(interface)
    if derived != device: raise ValueError(f'{device} does not match {interface} ({derived})')
    port_path=root/device/'ports'/port
    if not port_path.is_dir(): raise ValueError(f'missing HCA port {port_path}')
    indices=[]
    for gid_type in sorted((port_path/'gid_attrs'/'types').iterdir(), key=lambda p:p.name):
        if not gid_type.name.isdecimal(): continue
        if read(gid_type, True) != 'RoCE v2': continue
        if read(port_path/'gid_attrs'/'ndevs'/gid_type.name, True) != interface: continue
        gid=ipaddress.IPv6Address(read(port_path/'gids'/gid_type.name))
        if gid.ipv4_mapped == address.ip: indices.append(int(gid_type.name))
    if len(indices) != 1: raise ValueError(f'expected one common IPv4 RoCE-v2 GID, found {indices}')
    return {'index': indices[0], 'devices': {device: {'interface': interface, 'ip': str(address.ip), 'port': int(port)}}}

def main():
    if len(sys.argv)!=3: raise SystemExit('usage: discover-switched-roce-gids.py addresses-json =device:port')
    try: print(json.dumps(discover(json.loads(sys.argv[1]), sys.argv[2]), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f'GID_CHECK_FAIL {error}', file=sys.stderr); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())