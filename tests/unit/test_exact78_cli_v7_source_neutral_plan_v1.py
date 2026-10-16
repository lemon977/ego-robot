"""Check-only source-neutral planning and failure boundaries."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli_v7 import source_neutral_plan_v1 as planner
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


SOURCE = {
    "kind": "TRACKER_CLIP_RGB", "session_root": "/root/raw/session_a",
    "session_id": "session_a", "files": {
        role: {"path": "/root/raw/session_a/" + role, "bytes": 1, "sha256": "a" * 64}
        for role in planner.ROLES["TRACKER_CLIP_RGB"]
    },
}
DOMAIN = {
    "physical_eye": "tracker_session_left_RGB", "crop_xywh": None,
    "time_mode": "TRACKER_CLIP_METADATA_ONLY",
    "capture_to_decode_alignment": "CLIP_INDEX_ONLY",
    "declared_frame_count": 12, "fps_nominal": 25.0,
    "calibration_claim": "CLIP_RECTIFIED_K_ONLY_WORLD_POSE_NOT_SOLVED",
}


class SourceNeutralPlanTests(unittest.TestCase):
    def test_stable_check_only_signature_and_explicit_limits(self) -> None:
        evidence = {"source": copy.deepcopy(SOURCE), "domain": copy.deepcopy(DOMAIN)}
        with patch.object(planner, "_source_evidence", return_value=evidence), \
             patch.object(planner, "_writer_state", return_value="READY_FOR_FUTURE_CODE_ACCEPTANCE_ONLY"):
            first = planner.plan_one(Path(SOURCE["session_root"]))
            second = planner.plan_one(Path(SOURCE["session_root"]),
                                      expected_sha=first["submission_input_sha256"])
        self.assertEqual(first, second)
        self.assertFalse(first["registration_started"])
        self.assertFalse(first["decode_verified"])
        self.assertFalse(first["control_ground_truth"])
        self.assertEqual(first["stage_packet_construction"], "NOT_IMPLEMENTED")

    def test_expected_sha_drift_rejected_before_writer_decision(self) -> None:
        with patch.object(planner, "_source_evidence", return_value={"source": SOURCE, "domain": DOMAIN}), \
             patch.object(planner, "_writer_state") as writer:
            with self.assertRaisesRegex(ColdStartInputError, "EXPECTED_INPUT_SHA_DRIFT"):
                planner.plan_one(Path(SOURCE["session_root"]), expected_sha="0" * 64)
        writer.assert_not_called()

    def test_writer_conflict_and_double_writer_reject(self) -> None:
        active = {"task_packets": [{"execution_allowed": True, "task_id": "other"}]}
        state = {"next_task": {"task_id": "other"},
                 "tasks": [{"task_id": "other", "status": "RUNNING"}]}
        with patch.object(planner, "load_json", side_effect=lambda path: state if path == planner.TASK_STATE_PATH else active):
            self.assertEqual(planner._writer_state(), "WAIT_ACTIVE_WRITER")
            state["tasks"].append({"task_id": "second", "status": "RUNNING"})
            with self.assertRaisesRegex(ColdStartInputError, "DOUBLE_WRITER"):
                planner._writer_state()

    def test_unsupported_role_and_hand_motion_role_rejected_before_admission(self) -> None:
        source = copy.deepcopy(SOURCE)
        source["files"]["hand"] = {"path": "/root/raw/session_a/hand", "bytes": 1,
                                   "sha256": "b" * 64}
        with patch.object(planner, "_source_root", return_value=Path(SOURCE["session_root"])), \
             patch.object(planner.cpu, "_precheck_declared_paths"), \
             patch.object(planner, "inspect_session", return_value=source), \
             patch.object(planner, "_admit_known_domain") as admit:
            with self.assertRaisesRegex(ColdStartInputError, "SOURCE_ROLE_SET_UNSUPPORTED"):
                planner._source_evidence(Path(SOURCE["session_root"]), Path("/root/raw"))
        admit.assert_not_called()

    def test_manifest_requires_pinned_rows_and_rejects_duplicates(self) -> None:
        path = Path(__file__)
        row = {"path": SOURCE["session_root"], "expected_input_sha256": "a" * 64}
        value = {"schema_version": planner.SCHEMA, "sessions": [row, copy.deepcopy(row)]}
        with patch.object(planner, "load_json", return_value=value), \
             patch.object(planner, "file_ref", return_value={"path": str(path), "sha256": "c" * 64}), \
             patch.object(planner, "plan_one") as one:
            with self.assertRaisesRegex(ColdStartInputError, "MANIFEST_DUPLICATE_SESSION"):
                planner.plan_manifest(path, expected_sha="c" * 64)
        one.assert_not_called()

    def test_manifest_sha_drift_rejected_before_rows(self) -> None:
        path = Path(__file__)
        with patch.object(planner, "file_ref", return_value={"path": str(path), "sha256": "c" * 64}), \
             patch.object(planner, "load_json") as load:
            with self.assertRaisesRegex(ColdStartInputError, "EXPECTED_MANIFEST_SHA_DRIFT"):
                planner.plan_manifest(path, expected_sha="d" * 64)
        load.assert_not_called()

    def test_cli_exposes_check_only_only(self) -> None:
        from chaoyang.cli_v7 import main
        with self.assertRaises(SystemExit) as error:
            main(["submit", "exact78-raw-to-robot", "--session", "unused", "--register-and-run"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
