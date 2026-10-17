#!/usr/bin/env python3
"""Offline tests for the one-HCA switched logical-ring adaptation."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def switched_values() -> dict[str, str]:
    values = {
        "TOPOLOGY_MODE": "switched-ring-one-hca",
        "CLUSTER_NAME": "ds-v41-switched-ring",
        "CLUSTER_TP_SIZE": "4",
        "CLUSTER_PP_SIZE": "1",
        "NODE_COUNT": "4",
        "HEAD_RANK": "0",
        "API_PORT": "8889",
        "MASTER_PORT": "25001",
        "IMAGE": "deepseek-v4.1-gx10-switchless-switched:local",
        "WORKLOAD_ROOT": "/srv/deepseek-v4.1-switched",
        "MODEL_ROOT": "/srv/models/deepseek-v4.1-flash/revision",
        "CACHE_ROOT": "/var/cache/ai-vllm/deepseek-v4.1-switched",
        "NCCL_LIBRARY": "/opt/switched-nccl/libnccl.so.2.30.7",
        "NCCL_GID_INDEX": "3",
        "NCCL_IB_HCA": "=rocep1s0f1:1",
        "RING_EDGES": "0:0->1:0 1:0->2:0 2:0->3:0 3:0->0:0",
    }
    for rank in range(4):
        values.update(
            {
                f"RANK_{rank}_NAME": f"dgx-{rank + 1}",
                f"RANK_{rank}_SSH": f"ansible@dgx-{rank + 1}.home",
                f"RANK_{rank}_CONTROL_IP": f"198.51.100.{21 + rank}",
                f"RANK_{rank}_CONTROL_IFACE": "enP7s7",
                f"RANK_{rank}_FABRIC": f"enp1s0f1np1:192.0.2.{21 + rank}/24",
            }
        )
    return values


def test_switched_topology_accepts_one_hca_and_logical_ring():
    validator = load(ROOT / "scripts/validate-switched-topology.py")
    assert validator.validate(switched_values()).startswith("TOPOLOGY_OK")


def test_switched_topology_rejects_two_endpoints_per_rank():
    validator = load(ROOT / "scripts/validate-switched-topology.py")
    values = switched_values()
    values["RANK_0_FABRIC"] += " enP2p1s0f1np1:192.168.161.121/24"
    try:
        validator.validate(values)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("switched mode accepted an unintended second fabric endpoint")


def test_switched_gid_discovery_requires_one_matching_hca():
    discover = load(ROOT / "scripts/discover-switched-roce-gids.py")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        port = root / "rocep1s0f1" / "ports" / "1"
        (port / "gid_attrs/types").mkdir(parents=True)
        (port / "gid_attrs/ndevs").mkdir(parents=True)
        (port / "gids").mkdir(parents=True)
        (port / "gid_attrs/types/3").write_text("RoCE v2\n", encoding="ascii")
        (port / "gid_attrs/ndevs/3").write_text("enp1s0f1np1\n", encoding="ascii")
        (port / "gids/3").write_text("::ffff:c000:0215\n", encoding="ascii")
        result = discover.discover(
            {"enp1s0f1np1": "192.0.2.21/24"},
            "=rocep1s0f1:1",
            root,
        )
        assert result["index"] == 3
        assert result["devices"]["rocep1s0f1"]["interface"] == "enp1s0f1np1"


def test_ring_only_gate_is_preserved_and_switch_gate_is_explicit():
    patch = (ROOT / "docker/patches/switched-one-hca/nccl-2.30.7-switched-one-hca.patch").read_text(encoding="utf-8")
    config = (ROOT / "config/vllm-switched-ring-one-hca.env").read_text(encoding="utf-8")
    assert 'NCCL_SWITCHED_ONE_HCA=1' in config
    assert 'NCCL_SWITCHLESS_RING_ONLY=1' in config
    assert 'NCCL_SKIP_TREE_CONNECT=1' in config
    assert 'NCCL_ALGO=Ring' in config
    assert 'NCCL_NET=IB' in config
    assert 'SWITCHED_ONE_HCA' in patch
    assert 'ncclSwitchedOneHca' in patch
    assert 'Tree' not in patch or 'SKIP_TREE_CONNECT' in patch


def main():
    test_switched_topology_accepts_one_hca_and_logical_ring()
    test_switched_topology_rejects_two_endpoints_per_rank()
    test_switched_gid_discovery_requires_one_matching_hca()
    test_ring_only_gate_is_preserved_and_switch_gate_is_explicit()
    print("PASS switched one-HCA adaptation tests")


if __name__ == "__main__":
    main()