"""CPU-only source-domain and stage-contract tests for the Exact78 frontend."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_exact78_fresh_frontend_v2 import (
    MAP_GENERATOR, MASK_MISSING_SLEEVE_CABLE, _prompts, _timing_rows, authorize_frontend, map_stage,
    propainter_command, raw, role_status, seal_stage, select_primary_frame, validate_spec,
    verify_internal_mp4,
)


class FreshFrontendV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=REPO_ROOT / "_run/maintenance")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "raw").mkdir()
        (self.root / "source_stereo").mkdir()
        (self.root / "raw/vst.h264").write_bytes(b"test")
        (self.root / "source_stereo/CameraRecord_stereo.mp4").write_bytes(b"test")
        (self.root / "raw/camera_params.json").write_bytes(b"{}")
        (self.root / "camera_params.json").write_bytes(b"{}")

    def spec(self, family="VST_0915"):
        video = (self.root / "raw/vst.h264" if family == "VST_0915" else
                 self.root / "source_stereo/CameraRecord_stereo.mp4")
        return {
            "task_id": "candidate", "session_id": "session", "task": "poker",
            "source_family": family, "source_root": str(self.root),
            "source_video": {"path": str(video)},
            "camera_params": {"path": str(self.root / ("raw/camera_params.json" if family == "VST_0915" else "camera_params.json"))},
            "map_rule": "NO_MAP_VST_ENCODED" if family == "VST_0915" else "GENERATE_FROM_CAPTURE_CAMERA_PARAMS",
            "map_generator_code": "ABSENT" if family == "VST_0915" else artifact_ref(MAP_GENERATOR),
            "source_index": 1 if family == "VST_0915" else 0,
            "source_eye_size": [2048, 1536],
            "frame_count": 2, "source_fps": 25 if family == "TRACKER_0901" else 30,
            "image_domain": "VST_ENCODED_ALREADY_UNDISTORTED" if family == "VST_0915" else "TRACKER_RECTIFIED_PRIMARY",
            "transform": "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY" if family == "VST_0915" else "SAME_SESSION_EQUIDIS62_TO_PINHOLE",
            "frame_timing": "ABSENT", "sam_checkpoint": {}, "propainter_entry": {},
            "propainter_weights": {name: {} for name in (
                "raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")},
            "output_root": str(self.root / "output"),
        }

    def test_0915_capture_source_and_left_eye_are_required(self):
        value = self.spec()
        validate_spec(value)
        value["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "VST_PHYSICAL_LEFT"):
            validate_spec(value)
        value["source_index"] = 1
        value["source_video"] = {"path": str(self.root / "source_stereo/CameraRecord_stereo.mp4")}
        with self.assertRaisesRegex(ValueError, "VST_SOURCE_MUST_BE_CAPTURE_RAW"):
            validate_spec(value)
        value = self.spec()
        value["transform"] = "SAME_SESSION_EQUIDIS62_TO_PINHOLE"
        with self.assertRaises(ValueError):
            validate_spec(value)

    def test_0915_crop_resize_never_uses_remap(self):
        sbs = np.empty((1536, 4096, 3), np.uint8)
        sbs[:, :2048] = (10, 20, 30)
        sbs[:, 2048:] = (40, 50, 60)
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.cv2.remap",
                   side_effect=AssertionError("lens remap used")):
            image, crop = select_primary_frame(sbs, self.spec(), None)
        self.assertEqual(crop, (2048, 0, 2048, 1536))
        self.assertEqual(image.shape, (960, 1280, 3))
        self.assertTrue(np.all(image == (40, 50, 60)))
        with self.assertRaisesRegex(ValueError, "VST_REMAP_FORBIDDEN"):
            select_primary_frame(sbs, self.spec(), (np.zeros((960, 1280)),) * 2)

    def test_0915_camera_identity_must_agree_with_raw_crop(self):
        value = self.spec()
        output = self.root / "output"
        output.mkdir()
        (output / "MAP.json").write_text("{}")
        map_record = {"session_id": value["session_id"], "source_family": value["source_family"],
                      "camera_params": value["camera_params"], "source_video": value["source_video"],
                      "source_index": value["source_index"], "map_rule": value["map_rule"],
                      "camera_maps": "ABSENT"}
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.checked",
                   side_effect=lambda ref: Path(ref.get("path", "unused"))), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v2.load_json",
                   side_effect=[map_record, {"left": {"sourceIndex": 0}, "right": {"sourceIndex": 1}}]):
            with self.assertRaisesRegex(RuntimeError, "VST_CALIBRATION_PHYSICAL_EYE_DRIFT"):
                raw(value, output)

    def test_tracker_map_is_same_session_deterministic_and_needs_no_extrinsics(self):
        value = self.spec("TRACKER_0901")
        camera_path = Path(value["camera_params"]["path"])
        camera_path.write_text(json.dumps({
            "reference_resolution": {"w": 2048, "h": 1536},
            "left": {"intrinsics": {"fx": 936.0, "fy": 926.0, "cx": 1020.0, "cy": 775.0},
                     "distortion": {"model": "equiDis62", "coeffs": [0.0] * 8}},
        }))
        value["camera_params"] = artifact_ref(camera_path)
        output_a = self.root / "map_a"
        output_b = self.root / "map_b"
        output_a.mkdir()
        output_b.mkdir()
        map_stage(value, output_a)
        map_stage(value, output_b)
        result = load_json(output_a / "MAP.json")
        self.assertEqual(result["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertFalse(result["metric_authority"])
        self.assertEqual(result["camera_maps"]["sha256"],
                         load_json(output_b / "MAP.json")["camera_maps"]["sha256"])
        with np.load(result["camera_maps"]["path"], allow_pickle=False) as maps:
            self.assertEqual(maps["rectify_to_native_x"].shape, (960, 1280))
            self.assertEqual(maps["rectify_to_native_y"].shape, (960, 1280))
        with self.assertRaisesRegex(RuntimeError, "MAP_RESULT_ALREADY_EXISTS"):
            map_stage(value, output_a)
        seal_stage(value, output_a, "map", artifact_ref(camera_path))
        self.assertEqual(load_json(output_a / "SEAL_MAP.json")["result"]["sha256"],
                         artifact_ref(output_a / "MAP.json")["sha256"])
        with self.assertRaisesRegex(RuntimeError, "STAGE_SEAL_TARGET_INVALID"):
            seal_stage(value, output_a, "map", artifact_ref(camera_path))

    def test_old_map_input_is_rejected_and_vst_seals_absent_map(self):
        tracker = self.spec("TRACKER_0902")
        tracker["camera_maps"] = {"path": "old_session_map.npz"}
        with self.assertRaisesRegex(ValueError, "TRACKER_MAP_MUST_BE_GENERATED_IN_STAGE"):
            validate_spec(tracker)
        vst = self.spec()
        camera_path = Path(vst["camera_params"]["path"])
        camera_path.write_text(json.dumps({"left": {"sourceIndex": 1},
                                           "right": {"sourceIndex": 0}}))
        vst["camera_params"] = artifact_ref(camera_path)
        output = self.root / "vst_map"
        output.mkdir()
        map_stage(vst, output)
        self.assertEqual(load_json(output / "MAP.json")["camera_maps"], "ABSENT")
        self.assertFalse((output / "CAMERA_MAPS.npz").exists())

    def test_frontend_packet_writer_gate_rejects_sha_and_fence_drift(self):
        task_id = "exact78_test_frontend"
        task_root = self.root / "_run/current" / task_id / "attempts/attempt_0001"
        packet_path = self.root / "tasks/current" / task_id / "TASK_PACKET.json"
        index_path = self.root / "tasks/current/INDEX.json"
        task_root.mkdir(parents=True)
        packet_path.parent.mkdir(parents=True)
        writer = {"scope": task_id, "pid": 123, "proc_start_ticks": 456,
                  "fencing_token": "frozen"}
        (task_root / "WRITER.json").write_text(json.dumps(writer))
        packet_path.write_text(json.dumps({"task_id": task_id, "writer": writer,
                                           "write_set": [str(task_root)]}))
        index_path.write_text(json.dumps({"task_packets": [{
            "task_id": task_id,
            "packet_path": str(packet_path.relative_to(self.root)),
            "packet_sha256": artifact_ref(packet_path)["sha256"],
            "execution_allowed": True,
        }]}))
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.REPO_ROOT", self.root), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v2.process_identity",
                   return_value={"alive": True, "start_ticks": 456}):
            self.assertEqual(authorize_frontend(task_id)[1], task_root)
        route = load_json(index_path)
        route["task_packets"][0]["packet_sha256"] = "0" * 64
        index_path.write_text(json.dumps(route))
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.REPO_ROOT", self.root):
            with self.assertRaisesRegex(RuntimeError, "FRONTEND_PACKET_SHA_DRIFT"):
                authorize_frontend(task_id)

    def test_tracker_25_fps_is_preserved_without_vst_rule(self):
        value = self.spec("TRACKER_0901")
        validate_spec(value)
        sbs = np.empty((1536, 4096, 3), np.uint8)
        sbs[:, :2048] = (1, 2, 3)
        sbs[:, 2048:] = (7, 8, 9)
        yy, xx = np.indices((960, 1280), dtype=np.float32)
        image, crop = select_primary_frame(sbs, value, (xx, yy))
        self.assertEqual(crop, (0, 0, 2048, 1536))
        self.assertTrue(np.all(image == (1, 2, 3)))
        command = propainter_command(value, self.root, Path("vendor/ProPainter/inference_propainter.py"))
        self.assertEqual(command[command.index("--save_fps") + 1], "25")
        self.assertIn("--save_frames", command)

    def test_timing_mapping_rejects_duplicate_and_reorder(self):
        value = self.spec()
        rows, mode = _timing_rows(value)
        self.assertIsNone(rows)
        self.assertEqual(mode, "TIMEBASE_UNVERIFIED")
        value["frame_timing"] = {"path": "unused"}
        record = {"time_mode": "SOURCE_TIMEBASE_VERIFIED", "frames": [
            {"decode_display_index": 0, "capture_frame_id": 7},
            {"decode_display_index": 1, "capture_frame_id": 7},
        ]}
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.checked", return_value=Path("unused")), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v2.load_json", return_value=record):
            with self.assertRaisesRegex(ValueError, "CAPTURE_FRAME_ID"):
                _timing_rows(value)
            record["frames"][1]["capture_frame_id"] = 8
            record["frames"][1]["decode_display_index"] = 0
            with self.assertRaisesRegex(ValueError, "FRAME_TIMING_DUPLICATE"):
                _timing_rows(value)
            record["frames"][1]["decode_display_index"] = 1
            with self.assertRaisesRegex(ValueError, "VST_QPC_TIMEBASE_NOT_PROVEN"):
                _timing_rows(value)

    def test_t0_0915_projection_keeps_unknown_capture_id_and_raw_qpc(self):
        value = self.spec()
        value["frame_timing"] = {"path": "frozen_t0"}
        record = {"sessions": {"session": {
            "decoded_frames": 2, "nominal_fps_from_capture_metadata": 30,
            "time_mode": "TIMEBASE_UNVERIFIED",
            "frame_rows": [
                {"decoded_display_index": 0, "capture_frame_id": None,
                 "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
                 "qpc_raw_tick": 100, "pts": None, "capture_timestamp_raw": 1000},
                {"decoded_display_index": 1, "capture_frame_id": None,
                 "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
                 "qpc_raw_tick": 200, "pts": None, "capture_timestamp_raw": 2000},
            ],
        }}}
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.checked", return_value=Path("frozen_t0")), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v2.load_json", return_value=record):
            rows, mode = _timing_rows(value)
            self.assertEqual(mode, "TIMEBASE_UNVERIFIED")
            self.assertEqual([row["capture_frame_id"] for row in rows], [None, None])
            self.assertEqual([row["qpc_ticks"] for row in rows], [100, 200])
            self.assertEqual([row["pts"] for row in rows], [None, None])
            self.assertEqual([row["decode_display_index"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["capture_to_decode_alignment"], "INDEX_ALIGNMENT_UNVERIFIED")
            record["sessions"]["session"]["nominal_fps_from_capture_metadata"] = 25
            with self.assertRaisesRegex(ValueError, "SOURCE_FPS_CAPTURE_METADATA_DRIFT"):
                _timing_rows(value)
            record["sessions"]["session"]["nominal_fps_from_capture_metadata"] = 30
            record["sessions"]["session"]["time_mode"] = "SOURCE_TIMEBASE_VERIFIED"
            with self.assertRaisesRegex(ValueError, "VST_FROZEN_TIMEBASE_DRIFT"):
                _timing_rows(value)

    def test_t0_tracker_clip_metadata_does_not_become_verified_time(self):
        value = self.spec("TRACKER_0901")
        value["frame_timing"] = {"path": "frozen_t0"}
        record = {"sessions": {"session": {
            "decoded_frames": 2, "nominal_fps_from_capture_metadata": 25,
            "time_mode": "TRACKER_CLIP_METADATA_ONLY",
            "frame_rows": [
                {"decode_display_index": 0, "capture_frame_id": None, "pts": None},
                {"decode_display_index": 1, "capture_frame_id": None, "pts": None},
            ],
        }}}
        with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.checked", return_value=Path("frozen_t0")), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v2.load_json", return_value=record):
            rows, mode = _timing_rows(value)
        self.assertEqual(len(rows), 2)
        self.assertEqual(mode, "TIMEBASE_UNVERIFIED")

    def test_0915_unverified_sleeves_cables_are_explicit(self):
        prompts = _prompts(self.spec())
        self.assertIn("finger_sleeve", prompts)
        self.assertIn("yellow_cable", prompts)
        self.assertEqual(role_status("VST_0915"), MASK_MISSING_SLEEVE_CABLE)
        self.assertEqual(role_status("TRACKER_0902"), "MODEL_OUTPUT_UNREVIEWED")

    def test_propainter_internal_mp4_full_decode_fps_duration_and_sha(self):
        path = self.root / "internal.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 720))
        self.assertTrue(writer.isOpened())
        for index in range(3):
            writer.write(np.full((720, 960, 3), index * 40, np.uint8))
        writer.release()
        evidence = verify_internal_mp4(path, 3, 25)
        self.assertEqual(evidence["artifact"], artifact_ref(path))
        self.assertEqual((evidence["decoded_frames"], evidence["fps"], evidence["duration_seconds"]),
                         (3, 25, 0.12))
        with self.assertRaisesRegex(RuntimeError, "PROPAINTER_VIDEO_TIME_OR_DOMAIN"):
            verify_internal_mp4(path, 4, 25)
        with self.assertRaisesRegex(RuntimeError, "PROPAINTER_VIDEO_TIME_OR_DOMAIN"):
            verify_internal_mp4(path, 3, 30)
        with self.assertRaisesRegex(RuntimeError, "PROPAINTER_VIDEO_MISSING_OR_LINK"):
            verify_internal_mp4(self.root / "missing.mp4", 3, 25)


if __name__ == "__main__":
    unittest.main()
