"""CPU-only interface tests for the fresh HaWoR -> Robot diagnostic helper."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref, PROJECT_ROOT
from chaoyang.pipeline.exact78_fresh_robot_stage_v1 import (
    FreshRobotError, MANO_NAMES, display_tracks, load_fresh_inputs,
    normalize_roi_source, normalize_time_mode,
)


class DisplayTrackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.joints = np.full((2, 30, 21, 3), np.nan, dtype=np.float32)
        self.valid = np.zeros((2, 30), dtype=bool)

    def anchor(self, side: int, frame: int, x: float) -> None:
        self.joints[side, frame] = x
        self.valid[side, frame] = True

    def test_ambiguous_hawor_tracking_roi_is_not_admitted_as_detector(self) -> None:
        self.assertEqual(normalize_roi_source("HAWOR_DETECTION_OR_TRACKED_RGB"), "DIAGNOSTIC")
        self.assertEqual(normalize_time_mode("CLIP_FRAME_INDEX_ONLY"), "TRACKER_CLIP_METADATA_ONLY")
        with self.assertRaisesRegex(FreshRobotError, "ROI_SOURCE_ENUM_DRIFT"):
            normalize_roi_source("OLD_ROLE_MASK")


    def test_short_gap_30fps_and_long_gap_remain_distinct(self) -> None:
        self.anchor(0, 0, 0.0)
        self.anchor(0, 11, 11.0)  # 10 missing frames, inclusive first-round limit.
        self.anchor(0, 29, 29.0)  # 17 missing frames, diagnostic only.
        tracks, source, short, edge = display_tracks(self.joints, self.valid, 30.0)
        self.assertEqual(source[0, 5], "DETERMINISTIC_INTERPOLATED")
        self.assertTrue(short[0, 5])
        self.assertAlmostEqual(tracks[0, 5, 0, 0], 5.0)
        self.assertEqual(source[0, 20], "DIAGNOSTIC_LONG_GAP")
        self.assertFalse(short[0, 20])
        self.assertEqual(source[1, 20], "DIAGNOSTIC_NEUTRAL")
        self.assertEqual(edge[1, 20], "NONE")
        self.assertTrue(np.isnan(tracks[1]).all())

    def test_native_25fps_uses_eight_frame_limit_and_edges_only(self) -> None:
        self.anchor(0, 9, 9.0)
        self.anchor(0, 18, 18.0)
        _, source, short, edge = display_tracks(self.joints, self.valid, 25.0)
        self.assertEqual(source[0, 2], "DIAGNOSTIC_LONG_GAP")  # 9 > 8.
        self.assertEqual(edge[0, 2], "PREFIX")
        self.assertFalse(short[0, 2])
        self.assertEqual(source[0, 10], "DETERMINISTIC_INTERPOLATED")
        self.assertTrue(short[0, 10])
        self.assertEqual(source[0, 28], "DIAGNOSTIC_LONG_GAP")
        self.assertEqual(edge[0, 28], "SUFFIX")


class FreshInputTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_run/current")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / "raw.mp4"
        self.raw.write_bytes(b"fresh-rgb")
        self.npz = self.root / "HAWOR_CAMERA_PREDICTIONS.npz"
        self.provenance = self.root / "FRAME_PROVENANCE.json"
        self.audit = self.root / "INPUT_AUDIT.json"
        self.count = 3

    def write_fixture(self, *, old_hand_path: str | None = None,
                      side_order: tuple[str, str] = ("left", "right")) -> None:
        values = {"joints_3d_camera": np.full((2, self.count, 21, 3), np.nan, np.float32),
                  "joints_2d": np.full((2, self.count, 21, 2), np.nan, np.float32),
                  "root_orient_camera": np.full((2, self.count, 3, 3), np.nan, np.float32),
                  "hand_pose_rotmat": np.full((2, self.count, 15, 3, 3), np.nan, np.float32),
                  "betas": np.full((2, self.count, 10), np.nan, np.float32),
                  "prediction_valid": np.zeros((2, self.count), bool),
                  "intrinsics": np.tile(np.eye(3, dtype=np.float32)[None], (self.count, 1, 1)),
                  "original_frame_indices": np.arange(self.count, dtype=np.int32),
                  "fps_nominal": np.asarray(30.0, dtype=np.float32),
                  "mano_joint_names": np.asarray(MANO_NAMES),
                  "anatomical_side_names": np.asarray(side_order)}
        np.savez_compressed(self.npz, **values)
        self.provenance.write_text(json.dumps({
            "schema_version": "exact78-hawor-frame-provenance-v3", "session_id": "test_session",
            "intrinsics_image_size": [1280, 960], "frame_rows": [
                {"decoded_display_index": i, "output_index": i,
                 "hands": {"left": {}, "right": {}}, "time_mode": "TIMEBASE_UNVERIFIED"}
                for i in range(self.count)]}), encoding="utf-8")
        audit = {"session_id": "test_session", "source_kind": "tracker",
                 "raw_source_ref": file_ref(self.raw)}
        if old_hand_path:
            audit["old_hawor"] = old_hand_path
        self.audit.write_text(json.dumps(audit), encoding="utf-8")

    def test_exact_fresh_v3_contract_accepts_empty_predictions(self) -> None:
        self.write_fixture()
        values, provenance, audit = load_fresh_inputs(
            self.npz, self.provenance, self.audit, self.raw, "test_session")
        self.assertEqual(values["joints_3d_camera"].shape, (2, 3, 21, 3))
        self.assertEqual(len(provenance["frame_rows"]), 3)
        self.assertEqual(audit["raw_source_ref"], file_ref(self.raw))

    def test_side_order_and_historical_audit_member_rejected(self) -> None:
        self.write_fixture(side_order=("right", "left"))
        with self.assertRaisesRegex(FreshRobotError, "HAWOR_SIDE_ORDER_DRIFT"):
            load_fresh_inputs(self.npz, self.provenance, self.audit, self.raw, "test_session")
        self.write_fixture(old_hand_path="/old/HAWOR_RAW_MANO21.npz")
        with self.assertRaisesRegex(FreshRobotError, "FORBIDDEN_AUDIT_INPUT_ROLE"):
            load_fresh_inputs(self.npz, self.provenance, self.audit, self.raw, "test_session")

    def test_raw_source_sha_and_frame_mapping_rejected(self) -> None:
        self.write_fixture()
        self.raw.write_bytes(b"changed")
        with self.assertRaisesRegex(FreshRobotError, "RAW_VIDEO_SOURCE_SHA_DRIFT"):
            load_fresh_inputs(self.npz, self.provenance, self.audit, self.raw, "test_session")
        self.raw.write_bytes(b"fresh-rgb")
        value = json.loads(self.provenance.read_text(encoding="utf-8"))
        value["frame_rows"][1]["decoded_display_index"] = 2
        self.provenance.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(FreshRobotError, "FRESH_HAWOR_FRAME_MAP_DRIFT"):
            load_fresh_inputs(self.npz, self.provenance, self.audit, self.raw, "test_session")


if __name__ == "__main__":
    unittest.main()
