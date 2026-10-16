"""Negative input and frame-evidence checks before any model import."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.cli import exact78_submit
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    ColdStartInputError, PROJECT_ROOT, file_ref, inspect_session, validate_manifest,
    validate_output_path, validate_producer_reads,
)
from chaoyang.ops.run_exact78_raw_to_robot_v1 import candidate_plan
from chaoyang.pipeline.exact78_motion_state_v1 import (
    classify_frame, mark_rendered, short_gap_limit_frames,
)


def write(path: Path, value: bytes | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode("utf-8"))


class InputGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_run/current")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tracker(self, *, extrinsics: bool = True) -> Path:
        session = self.root / "tracker/play_cards_0902_042"
        name = session.name
        write(session / f"CameraRecord_{name}.mp4", b"rgb")
        write(session / "source_stereo" / f"CameraRecord_{name}_stereo.mp4", b"stereo")
        write(session / "camera_params.json", {"extrinsics": {"left": [1], "right": [1]}}
              if extrinsics else {"intrinsics": {"left": [1]}})
        write(session / f"slam_trajectory_{name}.jsonl", b"pose\n")
        write(session / "clip_manifest.json", {
            "clip_name": name,
            "files": {"video": f"CameraRecord_{name}.mp4", "tracking": "trackingData.txt",
                      "source_stereo_video": f"source_stereo/CameraRecord_{name}_stereo.mp4",
                      "camera_params": "camera_params.json",
                      "slam": f"slam_trajectory_{name}.jsonl"},
            "video": {"frame_count": 171, "fps": 30.0},
        })
        return session

    def test_tracker_reads_raw_roles_and_corrected_calibration_shape(self) -> None:
        source = inspect_session(self.tracker(), allowed_root=self.root)
        self.assertEqual(source["metric_calibration"], "SOURCE_PRESENT_QUALITY_UNVERIFIED")
        self.assertEqual(set(source["files"]), {"video", "source_stereo_video", "camera_params", "slam", "clip_manifest"})
        source_absent = inspect_session(self.tracker(extrinsics=False), allowed_root=self.root)
        self.assertEqual(source_absent["metric_calibration"], "ABSENT")

    def test_historical_arrays_masks_and_placement_rejected_before_model_import(self) -> None:
        session = self.tracker()
        source = inspect_session(session, allowed_root=self.root)
        for name in ("HAWOR_RAW_MANO21.npz", "role_mask.npy", "placement.json"):
            forbidden = session / name
            write(forbidden, b"historical")
            with self.assertRaisesRegex(ColdStartInputError, "FORBIDDEN_PRODUCER_READ"):
                validate_producer_reads(source, [forbidden])
        self.assertNotIn("chaoyang.ops.run_exact78_hawor_raw_v1", sys.modules)

    def test_forged_source_ref_cannot_relabel_historical_array_as_rgb(self) -> None:
        session = self.tracker()
        source = inspect_session(session, allowed_root=self.root)
        forbidden = session / "HAWOR_RAW_MANO21.npz"
        write(forbidden, b"historical")
        source["files"]["video"]["path"] = str(forbidden)
        with self.assertRaisesRegex(ColdStartInputError, "SOURCE_ROLE_PATH_DRIFT"):
            validate_producer_reads(source, [forbidden])

    def test_manifest_rejects_old_product_fields(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "MANIFEST_KEYS_NOT_RAW_ONLY"):
            validate_manifest({"schema_version": "exact78-raw-submit-v1",
                               "sessions": ["/raw/a"], "canonical_raw_hawor": "/old/a.npz"})

    def test_submit_request_is_stable_and_candidate_only(self) -> None:
        session = self.tracker()
        first = exact78_submit.build_request(session=session, allowed_root=self.root)
        second = exact78_submit.build_request(session=session, allowed_root=self.root)
        self.assertEqual(first["submission_input_sha256"], second["submission_input_sha256"])
        plan = candidate_plan(first)
        self.assertEqual(plan["sessions"][0]["stages"]["hawor"], "NOT_EVALUATED")
        self.assertFalse(plan["execution_complete"])
        self.assertFalse(plan["training_eligible"])
        self.assertNotIn("chaoyang.ops.run_exact78_hawor_raw_v1", sys.modules)

    def test_same_signature_attaches_and_different_signature_rejects(self) -> None:
        request = exact78_submit.build_request(session=self.tracker(), allowed_root=self.root)
        packet_path = self.root / "tasks/current/exact78_candidate/TASK_PACKET.json"
        pid = os.getpid()
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        writer = {"pid": pid, "proc_start_ticks": ticks, "executor_epoch": "test",
                  "fencing_token": "test-token", "scope": "candidate"}
        attempt = self.root / "_run/current/exact78_candidate/attempts/attempt_0001"
        write(attempt / "WRITER.json", writer)
        write(packet_path, {"task_id": "exact78_candidate",
                            "execution_mode": "EXACT78_CANDIDATE",
                            "run_signature": "test-signature", "writer": writer,
                            "write_set": [str(attempt)],
                            "inputs": {"submission_input_sha256": request["submission_input_sha256"]}})
        packet_sha = file_ref(packet_path)["sha256"]
        write(self.root / "tasks/current/INDEX.json", {"task_packets": [
            {"task_id": "exact78_candidate", "packet_path": "tasks/current/exact78_candidate/TASK_PACKET.json",
             "packet_sha256": packet_sha, "execution_allowed": True}]})
        state_path = self.root / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        state = {"next_task": {"task_id": "exact78_candidate"}, "tasks": [
            {"task_id": "exact78_candidate", "status": "RUNNING", "run_signature": "test-signature",
             "pid": pid, "proc_start_ticks": ticks, "task_packet": {"sha256": packet_sha}}]}
        write(state_path, state)
        with patch.object(exact78_submit, "PROJECT_ROOT", self.root):
            self.assertEqual(exact78_submit._attach_existing(request)["status"],
                             "ATTACHED_TO_ACTIVE_CANDIDATE")
            changed = {**request, "submission_input_sha256": "0" * 64}
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_PACKET_SIGNATURE_CONFLICT"):
                exact78_submit._attach_existing(changed)
            writer["proc_start_ticks"] = ticks + 1
            write(attempt / "WRITER.json", writer)
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_WRITER_FENCE_DRIFT"):
                exact78_submit._attach_existing(request)
            writer["proc_start_ticks"] = ticks
            write(attempt / "WRITER.json", writer)
            state["tasks"][0]["status"] = "PASSED"
            write(state_path, state)
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_TASK_STATE_DRIFT"):
                exact78_submit._attach_existing(request)

    def test_0915_left_eye_and_timebase_are_explicit(self) -> None:
        session = self.root / "cards_120_0915/001"
        for name in ("vst.h264", "vst.qpc.ts.jsonl", "vst.ts.jsonl"):
            write(session / "raw" / name, b"raw")
        write(session / "raw/camera_params.meta.json", {"extrinsic_convention": "head_to_camera_3x4_row_major"})
        camera = {"width": 2048, "height": 1536,
                  "left": {"sourceIndex": 1, "extrinsic": [1]},
                  "right": {"sourceIndex": 0, "extrinsic": [1]}}
        write(session / "raw/camera_params.json", camera)
        source = inspect_session(session, allowed_root=self.root)
        self.assertEqual(source["crop_xywh"], [2048, 0, 2048, 1536])
        self.assertEqual(source["session_id"], "play_cards_0915_001")
        self.assertEqual(source["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertNotIn("raw/pico.jsonl", source["files"])
        camera["left"]["sourceIndex"] = 0
        write(session / "raw/camera_params.json", camera)
        with self.assertRaisesRegex(ColdStartInputError, "PHYSICAL_EYE_IDENTITY_UNVERIFIED"):
            inspect_session(session, allowed_root=self.root)

    def test_attempt_write_set_excludes_external_and_other_attempt(self) -> None:
        attempt = PROJECT_ROOT / "_run/current/exact78_t1a_test/attempts/attempt_0001"
        self.assertEqual(validate_output_path(attempt / "stage/RESULT.json", attempt_root=attempt),
                         attempt / "stage/RESULT.json")
        for target in (Path("/path/to/private_data/egodata/forbidden.json"),
                       PROJECT_ROOT / "_run/current/other/attempts/attempt_0001/RESULT.json"):
            with self.assertRaisesRegex(ColdStartInputError, "OUTPUT_OUTSIDE_PROJECT_ATTEMPT"):
                validate_output_path(target, attempt_root=attempt)


class MotionStateTests(unittest.TestCase):
    def state(self, source: str, **changes):
        arguments = {"frame": 3, "side": "right", "roi_source": "MODEL_DETECTOR",
                     "source": source, "visibility": "VISIBLE", "time_mode": "CAPTURE_SYNC_VERIFIED",
                     "wrist_position_valid": True, "wrist_orientation_valid": True,
                     "local_hand_shape_valid": True, "coordinate_frame": "WORLD",
                     "world_pose_aligned": True, "model_input_valid": True}
        arguments.update(changes)
        return classify_frame(**arguments)

    def test_short_gap_is_time_bounded_and_requires_two_matching_anchors(self) -> None:
        self.assertEqual(short_gap_limit_frames(30), 10)
        self.assertEqual(short_gap_limit_frames(25), 8)
        good = self.state("DETERMINISTIC_INTERPOLATED", fps=25, gap_frames=8,
                          same_identity_anchors=True, same_coordinate_anchors=True)
        self.assertTrue(good.motion_evidence_eligible)
        self.assertFalse(self.state("DETERMINISTIC_INTERPOLATED", fps=25, gap_frames=9,
                                    same_identity_anchors=True, same_coordinate_anchors=True).motion_evidence_eligible)
        self.assertFalse(self.state("DETERMINISTIC_INTERPOLATED", fps=30, gap_frames=10,
                                    same_identity_anchors=False, same_coordinate_anchors=True).motion_evidence_eligible)

    def test_hold_is_only_short_prefix_or_suffix_in_world_or_base(self) -> None:
        self.assertTrue(self.state("EDGE_HELD", fps=30, gap_frames=10,
                                   edge_gap_kind="PREFIX").motion_evidence_eligible)
        self.assertFalse(self.state("EDGE_HELD", fps=30, gap_frames=10,
                                    edge_gap_kind="INTERIOR").motion_evidence_eligible)
        self.assertFalse(self.state("EDGE_HELD", fps=30, gap_frames=11,
                                    edge_gap_kind="SUFFIX").motion_evidence_eligible)
        self.assertFalse(self.state("EDGE_HELD", fps=30, gap_frames=3,
                                    edge_gap_kind="SUFFIX", coordinate_frame="CAMERA").motion_evidence_eligible)

    def test_prompted_prediction_requires_current_rgb_role_admission(self) -> None:
        self.assertFalse(self.state("RGB_HAWOR_MASK_PROMPTED", roi_source="CURRENT_RGB_ROLE_MASK",
                                    mask_prompt_admitted=False).motion_evidence_eligible)
        self.assertTrue(self.state("RGB_HAWOR_MASK_PROMPTED", roi_source="CURRENT_RGB_ROLE_MASK",
                                   mask_prompt_admitted=True).motion_evidence_eligible)
        self.assertFalse(self.state("RGB_HAWOR_MASK_PROMPTED", roi_source="MODEL_DETECTOR",
                                    mask_prompt_admitted=True).motion_evidence_eligible)

    def test_drawable_diagnostic_is_never_training_or_automatically_visible(self) -> None:
        state = self.state("DIAGNOSTIC_LONG_GAP")
        self.assertTrue(state.robot_drawable)
        self.assertFalse(state.motion_evidence_eligible)
        self.assertFalse(state.training_eligible)
        self.assertFalse(state.final_visible)
        self.assertTrue(mark_rendered(state, robot_pixels_visible=True).final_visible)

    def test_unverified_time_or_world_alignment_blocks_motion_evidence(self) -> None:
        self.assertFalse(self.state("RGB_HAWOR_PREDICTED", time_mode="TIMEBASE_UNVERIFIED").motion_evidence_eligible)
        self.assertFalse(self.state("RGB_HAWOR_PREDICTED", world_pose_aligned=False).motion_evidence_eligible)


if __name__ == "__main__":
    unittest.main()
