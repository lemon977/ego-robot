"""CPU-only new-session admission never registers or runs a model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chaoyang.cli import exact78_first_submit_bridge_candidate as bridge
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError, PROJECT_ROOT


class FirstSubmitBridgeCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-first-submit-test-", dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_root = self.root / "source_data"
        self.data_root.mkdir()

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _tracker(self, name: str = "play_cards_0915_003", *, eye: str = "left") -> Path:
        session = self.data_root / "tracker" / name
        self._write(session / "rgb.mp4", "synthetic encoded bytes")
        self._write(session / "source_stereo/stereo.mp4", "synthetic stereo bytes")
        self._write(session / "camera_params.json", "{}")
        self._write(session / f"slam_trajectory_{name}.jsonl", "{}\n")
        self._write(session / "clip_manifest.json", json.dumps({
            "clip_name": name,
            "files": {"video": "rgb.mp4", "source_stereo_video": "source_stereo/stereo.mp4",
                      "camera_params": "camera_params.json",
                      "slam": f"slam_trajectory_{name}.jsonl"},
            "video": {"frame_count": 2, "fps": 25, "width": 1280, "height": 960},
            "video_rectification": {"eye": eye, "rectified": True,
                                    "resolution": [1280, 960],
                                    "intrinsics": [[800, 0, 640], [0, 800, 480], [0, 0, 1]]},
        }))
        return session

    def _vst(self) -> Path:
        session = self.data_root / "cards_test_0915" / "003"
        self._write(session / "raw/vst.h264", "synthetic encoded bytes")
        self._write(session / "raw/vst.qpc.ts.jsonl", "100\n200\n")
        self._write(session / "raw/vst.ts.jsonl", "1000\n2000\n")
        self._write(session / "raw/camera_params.meta.json", "{}")
        self._write(session / "raw/camera_params.json", json.dumps({
            "width": 640, "height": 480, "outputWidth": 1280,
            "outputHeight": 480, "fps": 30,
            "left": {"sourceIndex": 1, "fx": 400, "fy": 400},
            "right": {"sourceIndex": 0},
        }))
        return session

    def test_tracker_first_submit_is_descriptor_only(self) -> None:
        session = self._tracker()
        fake_frames = [{"width": 1280, "height": 960, "pts": i,
                        "best_effort_timestamp": i, "pict_type": "P"} for i in range(2)]
        with patch.object(bridge, "_decode_frame_rows", return_value=fake_frames):
            result = bridge.plan_single(session, allowed_root=self.data_root)
        self.assertEqual(result["status"], "BLOCKED_RUNNER_NOT_GENERALIZED")
        self.assertEqual(result["descriptor"]["decoded_frames"], 2)
        self.assertEqual(result["descriptor"]["time_mode"], "TRACKER_CLIP_METADATA_ONLY")
        self.assertEqual(result["descriptor"]["independent_source_group"], "UNVERIFIED")
        self.assertFalse(result["registered"])
        self.assertEqual(result["task_packet_draft"]["commands"], [])
        self.assertFalse((PROJECT_ROOT / result["target_draft"]["packet_path"]).exists())
        serialized = json.dumps(result).lower()
        self.assertNotIn("accepted_", serialized)
        self.assertNotIn("input_freeze", serialized)
        self.assertNotIn("cards042_template", serialized)

    def test_new_session_identity_is_source_bound(self) -> None:
        one = self._tracker("play_cards_0915_003")
        two = self._tracker("play_cards_0915_004")
        fake_frames = [{"width": 1280, "height": 960} for _ in range(2)]
        with patch.object(bridge, "_decode_frame_rows", return_value=fake_frames):
            first = bridge.plan_single(one, allowed_root=self.data_root)
            second = bridge.plan_single(two, allowed_root=self.data_root)
        self.assertNotEqual(first["descriptor_sha256"], second["descriptor_sha256"])
        self.assertNotEqual(first["target_draft"]["task_id"], second["target_draft"]["task_id"])

    def test_unsupported_eye_is_rejected(self) -> None:
        session = self._tracker(eye="right")
        with patch.object(bridge, "_decode_frame_rows") as probe:
            with self.assertRaisesRegex(ColdStartInputError, "TRACKER_RECTIFIED_LEFT_RGB_UNVERIFIED"):
                bridge.plan_single(session, allowed_root=self.data_root)
        probe.assert_not_called()

    def test_frame_count_and_dimensions_are_hard_admission_gates(self) -> None:
        session = self._tracker()
        with patch.object(bridge, "_decode_frame_rows", return_value=[{"width": 1280, "height": 960}]):
            with self.assertRaisesRegex(ColdStartInputError, "RAW_DECODE_FRAME_COUNT_DRIFT"):
                bridge.plan_single(session, allowed_root=self.data_root)
        with patch.object(bridge, "_decode_frame_rows", return_value=[
                {"width": 1280, "height": 960}, {"width": 640, "height": 480}]):
            with self.assertRaisesRegex(ColdStartInputError, "RAW_DECODE_DIMENSION_DRIFT"):
                bridge.plan_single(session, allowed_root=self.data_root)

    def test_unknown_domain_is_rejected_before_decode(self) -> None:
        with patch.object(bridge, "build_request", return_value={"sources": [{"kind": "UNKNOWN"}]}), \
                patch.object(bridge, "_decode_frame_rows") as probe:
            with self.assertRaisesRegex(ColdStartInputError, "UNSUPPORTED_SOURCE_DOMAIN"):
                bridge.plan_single(self.data_root, allowed_root=self.data_root)
        probe.assert_not_called()

    def test_vst_diagnostic_clock_does_not_upgrade_world(self) -> None:
        session = self._vst()
        fake_frames = [{"width": 1280, "height": 480, "pts": None} for _ in range(2)]
        with patch.object(bridge, "_decode_frame_rows", return_value=fake_frames):
            result = bridge.plan_single(session, allowed_root=self.data_root)
        descriptor = result["descriptor"]
        self.assertEqual(descriptor["frame_rows"][0]["qpc_raw_tick"], 100)
        self.assertEqual(descriptor["capture_to_decode_alignment"], "INDEX_ALIGNMENT_UNVERIFIED")
        self.assertFalse(descriptor["world_pose_authority"])
        self.assertFalse(descriptor["metric_authority"])


if __name__ == "__main__":
    unittest.main()
