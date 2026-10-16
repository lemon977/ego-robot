"""Metadata categories must not masquerade as runtime path consumers."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chaoyang.governance import organize_project_layout_v1 as layout


class ReferenceProofTests(unittest.TestCase):
    def check_payload(self, *, packet=None, active=False, contract=None):
        root = layout.REPO_ROOT
        packet_path = "tasks/current/unit_reference/TASK_PACKET.json"
        index = {"task_packets": [{"task_id": "unit_reference", "packet_path": packet_path,
                                  "execution_allowed": active}]}
        def read(path):
            if path == layout.INDEX:
                return index
            if path == root / packet_path:
                return packet or {}
            if path == root / "docs/governance/ALGORITHM_CONTRACT.json":
                return contract or {}
            return {"tests": [{"group": "governance", "path": "tests/governance/test_example.py"}]}
        with patch.object(layout, "load_json", side_effect=read), patch.object(
            layout.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")
        ):
            return layout.runtime_hits("governance")

    def test_group_and_terminal_cleanup_metadata_are_not_consumers(self):
        hits = self.check_payload(packet={"read_set": ["governance"], "write_set": ["governance"],
                                         "inputs": {"moves": [["governance", "archive/governance"]]}})
        self.assertEqual(hits, [])

    def test_historical_runtime_code_pin_is_protected(self):
        hits = self.check_payload(packet={"runtime_code": [
            {"path": str(layout.REPO_ROOT / "governance/audit.py"), "sha256": "f" * 64, "bytes": 1}]})
        self.assertEqual(hits, ["tasks/current/unit_reference/TASK_PACKET.json"])

    def test_historical_actual_input_pin_is_protected(self):
        hits = self.check_payload(packet={"inputs": {"source": {
            "path": str(layout.REPO_ROOT / "governance/audit.py"), "sha256": "f" * 64, "bytes": 1}}})
        self.assertEqual(hits, ["tasks/current/unit_reference/TASK_PACKET.json"])

    def test_active_read_set_is_protected(self):
        hits = self.check_payload(packet={"read_set": ["governance"]}, active=True)
        self.assertEqual(hits, ["tasks/current/unit_reference/TASK_PACKET.json"])

    def test_current_contract_file_reference_is_protected(self):
        hits = self.check_payload(contract={"code_closure": [{"path": str(
            layout.REPO_ROOT / "governance/audit.py"), "sha256": "f" * 64, "bytes": 1}]})
        self.assertEqual(hits, ["docs/governance/ALGORITHM_CONTRACT.json"])


if __name__ == "__main__":
    unittest.main()
