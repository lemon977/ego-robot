"""CPU-only tests for the sealed-HaWoR Robot encoder successor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref
from chaoyang.ops import run_exact78_robot_from_sealed_hawor_v1 as operation
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_fresh_robot_stage_v2 as successor


class RobotFFmpeg44SuccessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="exact78_robot_ffmpeg44_test_",
                                                dir=PROJECT_ROOT / "_run/current")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_ffmpeg44_encoder_has_no_fps_mode_and_preserves_five_frames(self) -> None:
        render = self.root / "render"
        render.mkdir()
        temp = render / "tmp"
        temp.mkdir()
        video = render / "five.mp4"
        process = successor.ffmpeg44_encoder(video, 30.0, 32, 24, temp)
        assert process.stdin is not None
        process.stdin.write(np.zeros((5, 24, 32, 3), dtype=np.uint8).tobytes())
        process.stdin.close()
        self.assertEqual(process.wait(timeout=30), 0)
        checked = legacy._verify_video(video, 5, (32, 24), 30.0)
        self.assertEqual(checked["decoded_frames"], 5)
        self.assertAlmostEqual(checked["encoded_duration_seconds"], 5 / 30, delta=0.05)

    def test_override_restored_after_exception(self) -> None:
        original = legacy._encoder
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            with successor._ffmpeg44_override():
                self.assertIs(legacy._encoder, successor.ffmpeg44_encoder)
                raise RuntimeError("stopped")
        self.assertIs(legacy._encoder, original)

    def test_wrapper_failure_has_no_result(self) -> None:
        npz = self.root / "HAWOR_CAMERA_PREDICTIONS.npz"
        np.savez_compressed(npz, original_frame_indices=np.arange(2), fps_nominal=np.asarray(30.0))
        output = self.root / "robot"
        with mock.patch.object(legacy, "run_robot_stage", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                successor.run_robot_stage_ffmpeg44(
                    hawor_npz=npz, frame_provenance=self.root / "FRAME_PROVENANCE.json",
                    input_audit=self.root / "INPUT_AUDIT.json", raw_video=self.root / "raw.mp4",
                    source_kind="tracker", session_id="test", output_root=output)
        self.assertFalse((output / "RESULT.json").exists())

    def test_sealed_input_rejects_sha_drift_before_algorithm(self) -> None:
        root = self.root / "candidate/hawor"
        root.mkdir(parents=True)
        for name in ("RESULT.json", "HAWOR_CAMERA_PREDICTIONS.npz",
                     "FRAME_PROVENANCE.json", "INPUT_AUDIT.json"):
            (root / name).write_text("{}", encoding="utf-8")
        (root.parent / "HAWOR_STAGE_SEAL.json").write_text("{}", encoding="utf-8")
        (root.parent / "NATIVE_OPEN_AUDIT.json").write_text("{}", encoding="utf-8")
        packet = {"inputs": {
            "sealed_hawor": {name: file_ref(root / name) for name in
                             ("RESULT.json", "HAWOR_CAMERA_PREDICTIONS.npz",
                              "FRAME_PROVENANCE.json", "INPUT_AUDIT.json")},
            "hawor_seal": file_ref(root.parent / "HAWOR_STAGE_SEAL.json"),
            "native_open_audit": file_ref(root.parent / "NATIVE_OPEN_AUDIT.json"),
        }}
        (root / "INPUT_AUDIT.json").write_text("{\"changed\":true}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SEALED_HAWOR_SHA_OR_PATH_DRIFT"):
            operation._sealed_hawor(root, packet, "test", "tracker")

    def test_sealed_input_accepts_only_full_stage_lineage(self) -> None:
        root = self.root / "candidate/hawor"
        root.mkdir(parents=True)
        for name in ("HAWOR_CAMERA_PREDICTIONS.npz", "FRAME_PROVENANCE.json",
                     "INPUT_AUDIT.json"):
            (root / name).write_bytes(name.encode("utf-8"))
        native_path = root.parent / "NATIVE_OPEN_AUDIT.json"
        native_path.write_text(json.dumps({"status": "PASS_NATIVE_OPEN_SCOPE"}), encoding="utf-8")
        result_path = root / "RESULT.json"
        result_path.write_text(json.dumps({
            "schema_version": "exact78-hawor-pure-rgb-result-v3",
            "status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
            "session_id": "test", "source_kind": "tracker",
            "npz": file_ref(root / "HAWOR_CAMERA_PREDICTIONS.npz"),
            "frame_provenance": file_ref(root / "FRAME_PROVENANCE.json"),
            "input_audit": file_ref(root / "INPUT_AUDIT.json"),
        }), encoding="utf-8")
        seal_path = root.parent / "HAWOR_STAGE_SEAL.json"
        seal_path.write_text(json.dumps({
            "schema_version": "exact78-raw-hawor-stage-seal-v1", "session_id": "test",
            "result": file_ref(result_path), "native_open_audit": file_ref(native_path),
        }), encoding="utf-8")
        packet = {"inputs": {
            "sealed_hawor": {name: file_ref(root / name) for name in
                             ("RESULT.json", "HAWOR_CAMERA_PREDICTIONS.npz",
                              "FRAME_PROVENANCE.json", "INPUT_AUDIT.json")},
            "hawor_seal": file_ref(seal_path), "native_open_audit": file_ref(native_path),
        }}
        actual = operation._sealed_hawor(root, packet, "test", "tracker")
        self.assertEqual(actual, (root / "HAWOR_CAMERA_PREDICTIONS.npz",
                                  root / "FRAME_PROVENANCE.json", root / "INPUT_AUDIT.json"))
        packet["inputs"]["native_open_audit"] = file_ref(native_path)
        native_path.write_text(json.dumps({"status": "FAILED_NATIVE_OPEN_SCOPE"}), encoding="utf-8")
        packet["inputs"]["native_open_audit"] = file_ref(native_path)
        with self.assertRaisesRegex(ValueError, "SEALED_HAWOR_STAGE_SEAL_DRIFT"):
            operation._sealed_hawor(root, packet, "test", "tracker")

    def test_operation_gate_precedes_any_candidate_work(self) -> None:
        with mock.patch.object(operation, "authorize_exact78_operation",
                               side_effect=SystemExit("denied")) as gate:
            with mock.patch.object(operation, "run_candidate") as candidate:
                with mock.patch("sys.argv", [operation.OPERATION, "--session", "unused"]):
                    with self.assertRaisesRegex(SystemExit, "denied"):
                        operation.main()
                gate.assert_called_once()
                candidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
