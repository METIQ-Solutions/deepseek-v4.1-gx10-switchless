#!/usr/bin/env python3
"""Offline regression tests for the static validators in scripts/.

These cover the checks that stand between a wiring or addressing mistake and a
started cluster, so they are worth asserting rather than testing by hand:

  * topology validation (a valid file, and each way a file can be wrong);
  * RoCE GID discovery against a synthetic sysfs tree;
  * the per-edge byte-delta proof.

Nothing here touches hardware or the network. Run with:

    python3 -m unittest tests.test_validators -v
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
TOPOLOGY = REPO_ROOT / "config" / "topology.env"

sys.path.insert(0, str(SCRIPTS))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology_validator = load("topology_validator", "validate-topology.py")
gid_discovery = load("gid_discovery", "discover-roce-gids.py")
edge_deltas = load("edge_deltas", "check-edge-deltas.py")


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.base = TOPOLOGY.read_text(encoding="utf-8")

    def check(self, text: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.env"
            path.write_text(text, encoding="utf-8")
            return topology_validator.validate(topology_validator.parse(path))

    def test_example_topology_is_valid(self):
        self.assertIn("TOPOLOGY_OK", self.check(self.base))

    def test_ring_edge_lanes_must_share_one_physical_segment(self):
        broken = self.base.replace("enp1s0f1np1:10.10.0.2/30", "enp1s0f1np1:10.10.0.34/30")
        with self.assertRaisesRegex(topology_validator.Invalid, "not one physical segment"):
            self.check(broken)

    def test_control_plane_must_not_overlap_the_fabric(self):
        with self.assertRaisesRegex(topology_validator.Invalid, "falls inside"):
            self.check(self.base.replace("RANK_2_CONTROL_IP=10.20.0.13", "RANK_2_CONTROL_IP=10.10.0.11"))

    def test_control_interface_must_not_be_a_fabric_interface(self):
        with self.assertRaisesRegex(topology_validator.Invalid, "also a fabric interface"):
            self.check(self.base.replace("RANK_0_CONTROL_IFACE=enp7s7", "RANK_0_CONTROL_IFACE=enp1s0f0np0"))

    def test_fabric_addresses_must_be_slash_30(self):
        with self.assertRaisesRegex(topology_validator.Invalid, "must be a /30"):
            self.check(self.base.replace("enp1s0f0np0:10.10.0.1/30", "enp1s0f0np0:10.10.0.1/24"))

    def test_parallelism_must_match_the_rank_count(self):
        with self.assertRaisesRegex(topology_validator.Invalid, "must equal NODE_COUNT"):
            self.check(self.base.replace("CLUSTER_TP_SIZE=4", "CLUSTER_TP_SIZE=2"))

    def test_every_rank_must_appear_once_as_source_and_once_as_target(self):
        broken = self.base.replace(
            'RING_EDGES="0:0->1:1 1:0->2:1 2:0->3:1 3:0->0:1"',
            'RING_EDGES="0:0->1:1 0:0->2:1 2:0->3:1 3:0->0:1"',
        )
        with self.assertRaisesRegex(topology_validator.Invalid, "already bound|sources="):
            self.check(broken)

    def test_missing_rank_values_are_reported(self):
        with self.assertRaisesRegex(topology_validator.Invalid, "missing required value RANK_3_NAME"):
            self.check(self.base.replace("RANK_3_NAME=gx10-4\n", ""))

    def test_a_diagonal_segment_is_rejected(self):
        """Rank 0 and rank 2 are two hops apart; they must not share a /30."""
        broken = self.base.replace(
            "enp1s0f1np1:10.10.0.10/30", "enp1s0f1np1:10.10.0.26/30",
        )
        with self.assertRaisesRegex(topology_validator.Invalid, "diagonal|different /30"):
            self.check(broken)


class GidDiscoveryTests(unittest.TestCase):
    """Synthetic /sys/class/infiniband trees.

    The properties that matter: the index must be the IPv4 RoCE-v2 GID matching
    the interface address, and both selected HCAs must agree on one index.
    """

    @staticmethod
    def make_tree(root: Path, entries):
        for hca, interface, address, index, gid_type in entries:
            attr = root / hca / "ports" / "1"
            for directory in (attr / "gid_attrs" / "types", attr / "gid_attrs" / "ndevs", attr / "gids"):
                directory.mkdir(parents=True, exist_ok=True)
            (attr / "gid_attrs" / "types" / str(index)).write_text(gid_type + "\n")
            (attr / "gid_attrs" / "ndevs" / str(index)).write_text(interface + "\n")
            (attr / "gids" / str(index)).write_text(f"::ffff:{address}\n")

    def test_common_index_is_found_and_other_types_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tree(root, [
                ("rocep1s0f0", "enp1s0f0np0", "10.10.0.1", 3, "RoCE v2"),
                ("rocep1s0f1", "enp1s0f1np1", "10.10.0.26", 3, "RoCE v2"),
                ("rocep1s0f0", "enp1s0f0np0", "10.10.0.1", 1, "IB/RoCE v1"),
            ])
            result = gid_discovery.discover(
                {"enp1s0f0np0": "10.10.0.1/30", "enp1s0f1np1": "10.10.0.26/30"},
                "=rocep1s0f0:1,rocep1s0f1:1",
                root=root,
            )
            self.assertEqual(result["index"], 3)
            self.assertEqual(sorted(result["devices"]), ["rocep1s0f0", "rocep1s0f1"])

    def test_hcas_must_agree_on_the_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tree(root, [
                ("rocep1s0f0", "enp1s0f0np0", "10.10.0.1", 3, "RoCE v2"),
                ("rocep1s0f1", "enp1s0f1np1", "10.10.0.26", 5, "RoCE v2"),
            ])
            with self.assertRaisesRegex(gid_discovery.GidError, "no unique common"):
                gid_discovery.discover(
                    {"enp1s0f0np0": "10.10.0.1/30", "enp1s0f1np1": "10.10.0.26/30"},
                    "=rocep1s0f0:1,rocep1s0f1:1",
                    root=root,
                )

    def test_selected_hca_must_match_a_declared_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(gid_discovery.GidError, "do not correspond"):
                gid_discovery.discover(
                    {"enp1s0f0np0": "10.10.0.1/30"},
                    "=rocep9s0f0:1,rocep9s0f1:1",
                    root=Path(directory),
                )

    def test_a_gid_on_the_wrong_address_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_tree(root, [
                ("rocep1s0f0", "enp1s0f0np0", "10.10.0.99", 3, "RoCE v2"),
                ("rocep1s0f1", "enp1s0f1np1", "10.10.0.26", 3, "RoCE v2"),
            ])
            with self.assertRaisesRegex(gid_discovery.GidError, "no IPv4 RoCE-v2 GID"):
                gid_discovery.discover(
                    {"enp1s0f0np0": "10.10.0.1/30", "enp1s0f1np1": "10.10.0.26/30"},
                    "=rocep1s0f0:1,rocep1s0f1:1",
                    root=root,
                )

    def test_interface_names_must_be_derivable(self):
        with self.assertRaisesRegex(gid_discovery.GidError, "cannot derive an HCA name"):
            gid_discovery.hca_for_interface("eth0")

    def test_hca_names_are_derived_correctly(self):
        self.assertEqual(gid_discovery.hca_for_interface("enp1s0f0np0"), "rocep1s0f0")
        self.assertEqual(gid_discovery.hca_for_interface("enP2p1s0f1np1"), "roceP2p1s0f1")

    def test_metric_collector_derives_the_same_hca_names(self):
        """The two scripts duplicate this mapping; they must not drift.

        A mismatch would make the byte-delta proof read counters from a device
        the GID check never validated.
        """
        metrics = load("metrics_collector", "collect-fabric-metrics.py")
        for interface in ("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1"):
            self.assertEqual(
                metrics.hca_for_interface(interface),
                gid_discovery.hca_for_interface(interface),
                f"scripts disagree about {interface}",
            )


class EdgeDeltaTests(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, str]:
        return topology_validator.parse(TOPOLOGY)

    @classmethod
    def interfaces_for(cls, rank: int) -> list[str]:
        return [entry.split(":")[0] for entry in cls.topology()[f"RANK_{rank}_FABRIC"].split()]

    @classmethod
    def all_interface_names(cls) -> list[str]:
        names = []
        for rank in range(int(cls.topology()["NODE_COUNT"])):
            names.extend(cls.interfaces_for(rank))
        return names

    def write_samples(self, directory: Path, delta_for) -> None:
        """Write before/after counter samples for every rank.

        `delta_for(rank, interface)` returns the bytes that interface should
        appear to have moved.
        """
        topology = self.topology()
        for rank in range(int(topology["NODE_COUNT"])):
            interfaces = self.interfaces_for(rank)
            for label, multiplier in (("before", 0), ("after", 1)):
                sample = {
                    "fabric": {
                        interface: {
                            "hca": f"roceX{rank}{index}",
                            "port": 1,
                            "port_xmit_data": multiplier * delta_for(rank, interface),
                            "port_rcv_data": 0,
                        }
                        for index, interface in enumerate(interfaces)
                    },
                    "management": {"interface": "mgmt0", "tx_bytes": 0, "rx_bytes": 0},
                }
                (directory / f"{label}-rank{rank}.json").write_text(json.dumps(sample), encoding="utf-8")

    def run_check(self, delta_for, minimum: int = 1_048_576) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.write_samples(path, delta_for)
            return edge_deltas.run(str(TOPOLOGY), str(path), minimum)

    @staticmethod
    def flat(value: int):
        return lambda _rank, _interface: value

    def test_edge_traffic_above_the_threshold_passes(self):
        self.assertEqual(self.run_check(self.flat(2_000_000)), 0)

    def test_a_flat_sample_pair_fails(self):
        self.assertEqual(self.run_check(self.flat(0)), 1)

    def test_starving_one_physical_edge_fails(self):
        """Zero exactly the four lane endpoints of the first declared edge."""
        topology = self.topology()
        edge = topology["RING_EDGES"].split()[0]
        from_rank, from_port, to_rank, to_port = (
            int(part) for pair in edge.split("->") for part in pair.split(":")
        )
        starved: set[tuple[int, str]] = set()
        for lane in range(2):
            for rank, port in ((from_rank, from_port), (to_rank, to_port)):
                starved.add((rank, self.interfaces_for(rank)[port * 2 + lane]))

        def delta_for(rank: int, interface: str) -> int:
            return 0 if (rank, interface) in starved else 2_000_000

        self.assertEqual(self.run_check(delta_for), 1)

    def test_management_traffic_above_the_limit_fails(self):
        """A collective succeeding over Ethernet must be caught."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.write_samples(path, self.flat(2_000_000))
            topology = self.topology()
            for rank in range(int(topology["NODE_COUNT"])):
                sample = json.loads((path / f"after-rank{rank}.json").read_text(encoding="utf-8"))
                sample["management"] = {"interface": "mgmt0", "tx_bytes": 100_000_000, "rx_bytes": 0}
                (path / f"after-rank{rank}.json").write_text(json.dumps(sample), encoding="utf-8")
            self.assertEqual(
                edge_deltas.run(str(TOPOLOGY), str(path), 1_048_576, management_max_bytes=1_000_000),
                1,
            )

    def test_a_shrinking_counter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.write_samples(path, self.flat(2_000_000))
            # Swap before/after for one rank: counters must be monotonic.
            first = path / "before-rank0.json"
            second = path / "after-rank0.json"
            first.write_text(second.read_text(encoding="utf-8"), encoding="utf-8")
            second.write_text(json.dumps({
                "fabric": {name: {"hca": "x", "port": 1, "port_xmit_data": 0, "port_rcv_data": 0}
                           for name in self.all_interface_names()},
                "management": {"interface": "mgmt0", "tx_bytes": 0, "rx_bytes": 0},
            }), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not monotonic"):
                edge_deltas.run(str(TOPOLOGY), str(path), 1_048_576)


if __name__ == "__main__":
    unittest.main()
