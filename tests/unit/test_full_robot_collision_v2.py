"""Actual PyBullet DIRECT self-collision flag regressions, CPU only.

Synthetic geometry verifies software filtering. KaiHand zero-pose observations
are recorded without turning mesh contact into hardware collision ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import pybullet as p


def link(name):
    return f'<link name="{name}"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><sphere radius=".1"/></geometry></collision><visual><geometry><sphere radius=".1"/></geometry></visual></link>'


def joint(name, parent, child, x):
    return f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{x} 0 0"/><axis xyz="0 0 1"/><limit lower="-3.141593" upper="3.141593" effort="1" velocity="1"/></joint>'


class TestFullRobotCollisionV2(unittest.TestCase):
    records = []

    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU-only fixture requires empty CUDA_VISIBLE_DEVICES")
        root = Path(os.environ["TMPDIR"]).resolve(strict=True)
        if "four_stream_algorithm_baseline_v2" not in root.parts or "ai1" not in root.parts:
            raise RuntimeError("fixtures must remain in own ai1 scratch")
        cls.scratch = Path(tempfile.mkdtemp(prefix="collision_flags_v2_", dir=root))
        cls.chain = cls.scratch / "non_adjacent_chain.urdf"
        cls.chain.write_text('<robot name="non_adjacent_chain">' + link("base") + link("middle") + link("tip") + joint("base_middle", "base", "middle", 1) + joint("middle_tip", "middle", "tip", -1) + '</robot>')
        cls.adjacent = cls.scratch / "adjacent_chain.urdf"
        cls.adjacent.write_text('<robot name="adjacent_chain">' + link("base") + link("child") + joint("adjacent", "base", "child", .05) + '</robot>')

    @classmethod
    def tearDownClass(cls):
        target = os.environ.get("COLLISION_V2_REPORT")
        if target:
            dest = Path(target).resolve()
            if dest.parent != cls.scratch.parent.parent:
                raise RuntimeError("report must stay in own ai1 lane")
            record = {"schema": "COLLISION_FLAG_V2_CPU_REGRESSION", "gpu_used": False,
                      "fixture_root": str(cls.scratch), "records": cls.records,
                      "old_flags": 16, "new_flags": 24,
                      "source_data_modified": False, "shared_source_modified": False,
                      "claim_limit": "Synthetic software checks and mesh zero-pose observations only; not full Robot collision-free certification or physical hardware truth"}
            with dest.open("x", encoding="utf-8") as f:
                json.dump(record, f, indent=2, allow_nan=False)

    def setUp(self):
        self.client = p.connect(p.DIRECT)
        self.assertGreaterEqual(self.client, 0)

    def tearDown(self):
        p.disconnect(self.client)

    def load(self, path, flags):
        return p.loadURDF(str(path), useFixedBase=True, flags=flags, physicsClientId=self.client)

    def contacts(self, body):
        p.performCollisionDetection(physicsClientId=self.client)
        return [{"links": [int(row[3]), int(row[4])], "penetration_m": max(0., -float(row[8]))}
                for row in p.getContactPoints(body, body, physicsClientId=self.client) if row[8] < 0]

    def test_flags_are_explicit(self):
        self.assertEqual(p.URDF_USE_SELF_COLLISION, 8)
        self.assertEqual(p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, 16)
        self.assertEqual(p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, 24)

    def test_old_flags_miss_known_nonadjacent_overlap(self):
        body = self.load(self.chain, 16)
        geometry = p.getClosestPoints(body, body, .01, linkIndexA=-1, linkIndexB=1, physicsClientId=self.client)
        contacts = self.contacts(body)
        self.records.append({"case": "old_flags_16", "geometry_closest_distance_m": [float(row[8]) for row in geometry], "contacts": contacts})
        self.assertTrue(any(row[8] < -.15 for row in geometry))
        self.assertEqual(contacts, [])

    def test_new_flags_detect_known_nonadjacent_overlap(self):
        body = self.load(self.chain, 8 | 16)
        contacts = self.contacts(body)
        self.records.append({"case": "new_flags_24_nonadjacent", "contacts": contacts})
        self.assertTrue(any(set(row["links"]) == {-1, 1} and row["penetration_m"] > .15 for row in contacts))
        self.assertFalse(any(set(row["links"]) in ({-1, 0}, {0, 1}) for row in contacts))

    def test_new_flags_separation_negative(self):
        body = self.load(self.chain, 8 | 16)
        # Rotate the parent at x=1: the child origin x=-1 then moves to x=2.
        # Rotating the child's own joint does not translate its origin.
        p.resetJointState(body, 0, math.pi, physicsClientId=self.client)
        position = p.getLinkState(body, 1, computeForwardKinematics=True, physicsClientId=self.client)[4]
        contacts = self.contacts(body)
        self.records.append({"case": "new_flags_24_separated", "tip_position": list(position), "contacts": contacts})
        self.assertAlmostEqual(position[0], 2., places=5)
        self.assertEqual(contacts, [])

    def test_adjacent_overlap_is_excluded_but_positive_control_collides(self):
        body = self.load(self.adjacent, 8 | 16)
        geometry = p.getClosestPoints(body, body, .01, linkIndexA=-1, linkIndexB=0, physicsClientId=self.client)
        excluded = self.contacts(body)
        self.assertTrue(any(row[8] < -.1 for row in geometry))
        self.assertEqual(excluded, [])
        # Bullet load-time parent exclusion is not reversed by pair filtering.
        # Reload this synthetic fixture with explicit INCLUDE_PARENT as control.
        p.removeBody(body, physicsClientId=self.client)
        control = self.load(self.adjacent, p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        enabled = self.contacts(control)
        self.records.append({"case": "adjacent_exclusion", "excluded_contacts": excluded, "explicit_positive_control_contacts": enabled})
        self.assertTrue(any(set(row["links"]) == {-1, 0} for row in enabled))

    def test_actual_kai_zero_mesh_observations_not_quality_gate(self):
        repo = Path(os.environ["CHAOYANG_REPO_ROOT"]).resolve(strict=True)
        pin_path = repo / "assets/robot/ROBOT_ASSET_PIN.json"
        pin = json.loads(pin_path.read_text())
        entries = [entry for entry in pin["urdfs"] if entry["path"].startswith("assets/robot/kaihand/")]
        self.assertEqual(len(entries), 2)
        for entry in entries:
            urdf = (repo / entry["path"]).resolve(strict=True)
            raw = urdf.read_bytes()
            robot = ET.fromstring(raw)
            actual_pairs = []
            for flags in (16, 8 | 16):
                body = self.load(urdf, flags)
                names = {-1: p.getBodyInfo(body, physicsClientId=self.client)[0].decode()}
                parents = {}
                outside_zero_limits = []
                for i in range(p.getNumJoints(body, physicsClientId=self.client)):
                    info = p.getJointInfo(body, i, physicsClientId=self.client)
                    names[i], parents[i] = info[12].decode(), int(info[16])
                    p.resetJointState(body, i, 0., physicsClientId=self.client)
                    if info[3] >= 0 and not float(info[8]) <= 0 <= float(info[9]):
                        outside_zero_limits.append(names[i])
                contacts = self.contacts(body)
                for row in contacts:
                    row["link_names"] = [names[i] for i in row["links"]]
                    a, b = row["links"]
                    row["direct_parent_child"] = parents.get(a) == b or parents.get(b) == a
                actual_pairs.append({"flags": flags, "contact_count": len(contacts), "contacts": contacts, "zero_outside_joint_limits": outside_zero_limits})
                p.removeBody(body, physicsClientId=self.client)
            self.records.append({"case": "kai_zero_pose_observation", "urdf": str(urdf), "urdf_sha256": hashlib.sha256(raw).hexdigest(), "collision_geometry_kind": "URDF mesh / Bullet approximation", "observations": actual_pairs, "quality_asserted": False})
            self.assertEqual(actual_pairs[0]["contact_count"], 0)
            self.assertFalse(any(row["direct_parent_child"] for row in actual_pairs[1]["contacts"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
