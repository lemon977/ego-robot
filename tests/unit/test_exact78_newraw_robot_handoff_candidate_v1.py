"""Synthetic, CPU-only negative tests for terminal new-raw Robot handoff."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref
from chaoyang.governance import exact78_newraw_robot_handoff_candidate_v1 as gate


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class NewrawRobotHandoffCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="newraw-robot-handoff-",
                                           dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.session = "play_cards_0902_001"
        self.descriptor_sha = "a" * 64
        self.producer = "exact78_newraw_play_cards_0902_001_aaaaaaaaaaaaaaaa_v1"
        self.task = "exact78_newraw_hawor_play_cards_0902_001_aaaaaaaaaaaaaaaa_v4"
        self.attempt = self.root / "_run/current" / self.task / "attempts/attempt_0001"
        self.inner_root = self.attempt / "candidate/hawor"
        self.inner_root.mkdir(parents=True)
        self.raw = self.root / "raw/play_cards_0902_001.mp4"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"synthetic raw RGB")
        self.raw_ref = file_ref(self.raw)
        cpu_path = self.root / "_run/current" / self.producer / "attempts/attempt_0001/RESULT.json"
        write_json(cpu_path, {"status": "PASSED", "source_session": self.session,
                              "descriptor_sha256": self.descriptor_sha,
                              "source_files": {"video": self.raw_ref}})
        self.cpu_refs = {
            "upstream_admission_descriptor": {"path": "synthetic-descriptor", "bytes": 1,
                                               "sha256": "1" * 64},
            "upstream_admission_frame_time_map": {"path": "synthetic-map", "bytes": 1,
                                                   "sha256": "2" * 64},
            "upstream_admission_seal": {"path": "synthetic-seal", "bytes": 1,
                                        "sha256": "3" * 64},
            "upstream_admission_result": file_ref(cpu_path),
            "upstream_admission_receipt": {"path": "synthetic-receipt", "bytes": 1,
                                           "sha256": "4" * 64},
        }
        self.rows = [{"decoded_display_index": i, "pts": i,
                      "best_effort_timestamp": i, "qpc_raw_tick": None,
                      "capture_timestamp_raw": None,
                      "capture_to_decode_alignment": "CLIP_FRAME_INDEX_ONLY",
                      "crop_xywh": [0, 0, 1280, 960]} for i in range(3)]
        self.admission = {
            "session_id": self.session, "source_kind": "tracker", "frame_count": 3,
            "fps_nominal": 30.0, "physical_eye": "left",
            "time_mode": "CLIP_FRAME_INDEX_ONLY", "source_video": self.raw,
            "frame_rows_sealed": self.rows,
            "descriptor_ref": self.cpu_refs["upstream_admission_descriptor"],
            "frame_map_ref": self.cpu_refs["upstream_admission_frame_time_map"],
            "admission_seal_ref": self.cpu_refs["upstream_admission_seal"],
            "producer_result_ref": self.cpu_refs["upstream_admission_result"],
            "producer_receipt_ref": self.cpu_refs["upstream_admission_receipt"],
        }
        self.packet = {
            "task_id": self.task, "phase": "EXACT78_NEWRAW_HAWOR_GPU_CANDIDATE_V4",
            "execution_mode": "EXACT78_CANDIDATE", "run_signature": "synthetic-signature",
            "runtime_code": [], "config": {"frame_count": 3, "source_kind": "tracker"},
            "inputs": {
                **self.cpu_refs, "source_session": self.session,
                "source_files": {"video": self.raw_ref}, "raw_video": self.raw_ref,
                "model_assets": {}, "model_python": {"path": "synthetic-model"},
                "native_tracer": {"path": "synthetic-strace"},
            },
        }
        self.packet_path = self.root / "tasks/current" / self.task / "TASK_PACKET.json"
        write_json(self.packet_path, self.packet)
        npz = self.inner_root / "HAWOR_CAMERA_PREDICTIONS.npz"
        np.savez(npz, prediction_valid=np.array([[1, 1, 1], [0, 0, 0]], dtype=bool),
                 joints_3d_camera=np.zeros((2, 3, 21, 3)),
                 joints_2d=np.zeros((2, 3, 21, 2)),
                 root_orient_camera=np.zeros((2, 3, 3, 3)),
                 hand_pose_rotmat=np.zeros((2, 3, 15, 3, 3)),
                 betas=np.zeros((2, 3, 10)), intrinsics=np.zeros((3, 3, 3)),
                 original_frame_indices=np.arange(3), fps_nominal=np.array(30.0))
        provenance = self.inner_root / "FRAME_PROVENANCE.json"
        frame_rows = [{**row, "model_input_index": i, "output_index": i,
                       "sbs_physical_eye": "left", "time_mode": "CLIP_FRAME_INDEX_ONLY"}
                      for i, row in enumerate(self.rows)]
        write_json(provenance, {"schema_version": "exact78-hawor-frame-provenance-v3",
                                "session_id": self.session, "frame_count": 3,
                                "physical_eye": "left", "time_mode": "CLIP_FRAME_INDEX_ONLY",
                                "frame_rows": frame_rows})
        audit_path = self.inner_root / "INPUT_AUDIT.json"
        write_json(audit_path, {"schema_version": "exact78-hawor-input-audit-v3",
                                "raw_source_ref": self.raw_ref,
                                "source_role": "new_session_candidate",
                                "source_split_claim": "NOT_ASSIGNED",
                                "infiller_invoked": False, "world_pose_emitted": False})
        inner_path = self.inner_root / "RESULT.json"
        write_json(inner_path, {
            "schema_version": "exact78-hawor-pure-rgb-result-v3",
            "status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
            "session_id": self.session, "source_kind": "tracker",
            "source_frames_decoded": 3, "model_input_frames": 3,
            "world_pose_valid": False, "training_eligible": False,
            "model_prediction_frames": {"left": 3, "right": 0},
            "npz": file_ref(npz), "frame_provenance": file_ref(provenance),
            "input_audit": file_ref(audit_path),
        })
        native_path = self.attempt / "candidate/NATIVE_OPEN_AUDIT.json"
        trace_path = self.attempt / "candidate/native_open_trace.log"
        trace_path.write_text("synthetic native open trace\n", encoding="utf-8")
        write_json(native_path, {
            "schema_version": "exact78-newraw-hawor-native-open-audit-v4",
            "status": "PASS_NATIVE_OPEN_SCOPE", "source_video_open_observed": True,
            "forbidden_paths": [], "permitted_source_refs": {"video": self.raw_ref},
            "opened_external_paths": [str(self.raw)], "trace": file_ref(trace_path),
            "failed_external_write_open_attempts": [],
            "unresolved_external_write_open_returns": [],
            "successful_persistent_external_writable_opens": [],
            "successful_transient_kernel_ipc_writable_opens": [],
            "successful_runtime_device_writable_opens": [],
            "write_open_counts": {
                "failed_external_attempts": 0, "unresolved_external_returns": 0,
                "successful_persistent_external": 0,
                "successful_transient_kernel_ipc": 0,
                "successful_runtime_device": 0,
            },
        })
        seal_path = self.attempt / "candidate/HAWOR_STAGE_SEAL.json"
        write_json(seal_path, {
            "schema_version": "exact78-newraw-hawor-stage-seal-v4",
            "status": "SEALED_CAMERA_PREDICTIONS", "task_id": self.task,
            "session_id": self.session, "downstream_use": "SEALED_STAGE_ONLY",
            "signature": gate._hawor_signature(self.packet),
            "result": file_ref(inner_path), "native_open_audit": file_ref(native_path),
        })
        self.outer_path = self.attempt / "RESULT.json"
        self.outer = {
            "schema_version": "exact78-newraw-hawor-gpu-execution-result-v4",
            "task_id": self.task, "status": "PASSED", "failure_code": None,
            "algorithm_attempts": 1, "external_write_open_observed": False,
            "transient_kernel_ipc_writable_opens": 0,
            "stage_status": "SEALED_CAMERA_PREDICTIONS",
            "source_sha_stable_at_terminal": True,
            "source_session": self.session, "source_frames_expected": 3,
            "producer_result": self.cpu_refs["upstream_admission_result"],
            "visual_qualified": False, "training_eligible": False,
            "stage": {"result": file_ref(inner_path),
                      "native_open_audit": file_ref(native_path),
                      "prediction_counts": {"left": 3, "right": 0},
                      "native_write_open_counts": {
                          "failed_external_attempts": 0,
                          "unresolved_external_returns": 0,
                          "successful_persistent_external": 0,
                          "successful_transient_kernel_ipc": 0,
                          "successful_runtime_device": 0,
                      },
                      "seal": file_ref(seal_path)},
        }
        write_json(self.outer_path, self.outer)
        receipt_path = self.root / "tasks/receipts" / f"{self.task.upper()}.json"
        write_json(receipt_path, {"schema_version": "exact78-newraw-hawor-gpu-receipt-v4",
                                  "task_id": self.task, "status": "PASSED",
                                  "result": file_ref(self.outer_path)})
        self.state_path = self.root / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        write_json(self.state_path, {"tasks": [{
            "task_id": self.task, "status": "PASSED",
            "task_packet": file_ref(self.packet_path),
            "result": file_ref(self.outer_path),
            "terminal_receipt": file_ref(receipt_path),
            "run_signature": self.packet["run_signature"],
        }]})

    def verify(self) -> dict:
        return gate.verify_handoff(outer_result=self.outer_path,
                                   admission=self.admission, repo_root=self.root)

    def _mock_json(self, target: Path, mutate) -> object:
        original = gate._object

        def overridden(path: Path) -> dict:
            value = original(path)
            if path == target:
                mutate(value)
            return value

        return patch.object(gate, "_object", side_effect=overridden)

    def test_terminal_same_session_handoff_passes(self) -> None:
        value = self.verify()
        self.assertEqual(value["frame_count"], 3)
        self.assertEqual(value["source_kind"], "tracker")
        self.assertEqual(value["hawor_npz"]["path"],
                         str(self.inner_root / "HAWOR_CAMERA_PREDICTIONS.npz"))

    def test_outer_failed_or_partial_never_consumed(self) -> None:
        with self._mock_json(self.outer_path, lambda value: value.update(status="FAILED_RUNTIME_FINAL")):
            with self.assertRaisesRegex(gate.HandoffError, "OUTER_NOT_TERMINAL_PASSED"):
                self.verify()
        with self._mock_json(self.outer_path, lambda value: value.update(stage={})):
            with self.assertRaisesRegex(gate.HandoffError, "STAGE_FIVE_FIELDS_REQUIRED"):
                self.verify()

    def test_old_v1_v2_v3_schema_cannot_be_mistaken_for_v4_successor(self) -> None:
        for version in ("v1", "v2", "v3"):
            with self._mock_json(self.outer_path, lambda value: value.update(
                    schema_version=f"exact78-newraw-hawor-gpu-execution-result-{version}")):
                with self.assertRaisesRegex(gate.HandoffError, "OUTER_NOT_TERMINAL_PASSED"):
                    self.verify()
        old = self.root / "_run/current" / self.task.replace("_v4", "_v3")
        old = old / "attempts/attempt_0001/RESULT.json"
        with self.assertRaisesRegex(gate.HandoffError, "OUTER_ATTEMPT_DRIFT"):
            gate.verify_handoff(outer_result=old, admission=self.admission,
                                repo_root=self.root)

    def test_source_sha_drift_rejected(self) -> None:
        self.raw.write_bytes(b"mutated")
        with self.assertRaisesRegex(gate.HandoffError, "REF_SHA_OR_PATH_DRIFT:raw:video"):
            self.verify()

    def test_old_derived_result_path_rejected(self) -> None:
        def mutate(value: dict) -> None:
            value["stage"]["result"]["path"] = str(self.root / "_run/current/old/RESULT.json")

        with self._mock_json(self.outer_path, mutate):
            with self.assertRaisesRegex(gate.HandoffError, "REF_SHA_OR_PATH_DRIFT:inner"):
                self.verify()

    def test_native_open_forbidden_path_rejected(self) -> None:
        path = self.attempt / "candidate/NATIVE_OPEN_AUDIT.json"
        with self._mock_json(path, lambda value: value.update(forbidden_paths=["OLD_ACTION_READ:x"])):
            with self.assertRaisesRegex(gate.HandoffError, "NATIVE_OPEN_SCOPE_DRIFT"):
                self.verify()

    def test_native_persistent_write_or_count_drift_rejected(self) -> None:
        path = self.attempt / "candidate/NATIVE_OPEN_AUDIT.json"
        with self._mock_json(path, lambda value: value.update(
                successful_persistent_external_writable_opens=[str(self.raw)])):
            with self.assertRaisesRegex(gate.HandoffError, "NATIVE_OPEN_SCOPE_DRIFT"):
                self.verify()

    def test_outer_native_write_counts_must_match_native_audit(self) -> None:
        with self._mock_json(self.outer_path, lambda value: value["stage"]["native_write_open_counts"].update(
                successful_persistent_external=1)):
            with self.assertRaisesRegex(gate.HandoffError, "NATIVE_OPEN_SCOPE_DRIFT"):
                self.verify()

    def test_transient_kernel_ipc_is_disclosed_but_not_falsely_forbidden(self) -> None:
        path = self.attempt / "candidate/NATIVE_OPEN_AUDIT.json"
        with self._mock_json(path, lambda value: (
                value["successful_transient_kernel_ipc_writable_opens"].append("/dev/shm/sem.synthetic"),
                value["write_open_counts"].update(successful_transient_kernel_ipc=1))), \
                self._mock_json(self.outer_path, lambda value: (
                    value.update(transient_kernel_ipc_writable_opens=1),
                    value["stage"]["native_write_open_counts"].update(
                        successful_transient_kernel_ipc=1))):
            self.assertEqual(self.verify()["frame_count"], 3)
        with self._mock_json(path, lambda value: value["write_open_counts"].update(
                successful_transient_kernel_ipc=1)):
            with self.assertRaisesRegex(gate.HandoffError, "NATIVE_OPEN_SCOPE_DRIFT"):
                self.verify()

    def test_frame_map_mismatch_rejected(self) -> None:
        path = self.inner_root / "FRAME_PROVENANCE.json"
        with self._mock_json(path, lambda value: value["frame_rows"][1].update(pts=123)):
            with self.assertRaisesRegex(gate.HandoffError, "FRAME_MAP_DRIFT:1"):
                self.verify()

    def test_cross_session_packet_and_unsealed_ledger_rejected(self) -> None:
        with self._mock_json(self.packet_path,
                             lambda value: value["inputs"].update(source_session="other")):
            with self.assertRaisesRegex(gate.HandoffError, "PACKET_CPU_BINDING_DRIFT"):
                self.verify()
        with self._mock_json(self.state_path,
                             lambda value: value["tasks"][0].update(status="RUNNING")):
            with self.assertRaisesRegex(gate.HandoffError, "NOT_TERMINAL_IN_LEDGER"):
                self.verify()

    def test_seal_signature_drift_rejected(self) -> None:
        path = self.attempt / "candidate/HAWOR_STAGE_SEAL.json"
        with self._mock_json(path, lambda value: value.update(signature="0" * 64)):
            with self.assertRaisesRegex(gate.HandoffError, "STAGE_SEAL_DRIFT"):
                self.verify()


if __name__ == "__main__":
    unittest.main()
