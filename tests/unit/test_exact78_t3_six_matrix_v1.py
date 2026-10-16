"""CPU-only fail-closed tests for the frozen six-session T3 snapshot."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chaoyang.governance import build_exact78_t3_six_matrix_v1 as matrix


class SixMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-t3-test-", dir=matrix.REPO_ROOT / "_run/current")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        selected, mapped = {}, {}
        for index in range(6):
            session_id = f"session_{index}"
            video = {"path": f"/path/to/private_data/egodata/test/{session_id}.mp4",
                     "bytes": 10, "sha256": "a" * 64}
            selected[session_id] = {"source_path": f"/path/to/private_data/egodata/test/{session_id}",
                                    "role": "development" if index < 4 else "holdout",
                                    "source_group": f"group_{index}", "files": {"video": video}}
            mapped[session_id] = {"role": selected[session_id]["role"],
                                  "decoded_frames": 2, "source_video": video,
                                  "frame_rows": [{"sbs_physical_eye": "left"}] * 2,
                                  "nominal_fps_from_capture_metadata": 25,
                                  "time_mode": "TIMEBASE_UNVERIFIED",
                                  "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
                                  "head_pose_clock_domain": "UNVERIFIED"}
        self._write(matrix.FREEZE_REL, {"selected": selected})
        self._write(matrix.FRAME_MAP_REL, {"sessions": mapped})
        self.state = {"tasks": []}
        self._write(matrix.STATE_REL, self.state)
        self.sha_patches = [patch.object(matrix, "FREEZE_SHA", matrix._sha(self.root / matrix.FREEZE_REL)),
                            patch.object(matrix, "FRAME_MAP_SHA", matrix._sha(self.root / matrix.FRAME_MAP_REL))]
        for item in self.sha_patches:
            item.start()
            self.addCleanup(item.stop)

    def _write(self, relative: Path, value: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def _task(self, status: str) -> dict:
        return {"task_id": "exact78_test_s0", "phase": matrix.T2_PHASE,
                "session": "session_0", "status": status}

    def _terminal(self, *, complete: bool = True) -> dict:
        task = self._task("PASSED")
        attempt = Path("_run/current/exact78_test_s0/attempts/attempt_0001")
        candidate = attempt / "candidate"
        hawor_ref = matrix._ref(self._write(candidate / "hawor/RESULT.json", {
            "session_id": "session_0", "status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
            "source_frames_decoded": 2, "model_input_frames": 2,
            "model_prediction_frames": {"left": 2, "right": 1},
            "model_infill_frames": 0, "deterministic_fill_frames": 0}))
        hawor_seal_ref = matrix._ref(self._write(candidate / "HAWOR_STAGE_SEAL.json", {
            "session_id": "session_0", "result": hawor_ref}))
        numeric_ref = matrix._ref(self._write(
            candidate / "robot/v1_stage_with_ffmpeg44_encoder/RESULT.json", {
                "session_id": "session_0", "frame_count": 2,
                "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
                "counts": {"arm_status": {"PASS_NUMERIC_CAMERA_DIAGNOSTIC": 1},
                           "hand_status": {"HOLD_REACH": 3}, "display_drawable_rows": 4,
                           "final_visible_frames": 2}}))
        videos = {name: {"decoded_frames": 2, "video": {"sha256": "b" * 64}}
                  for name in ("ego_robot", "external_full_robot", "raw_motion_source")}
        robot_ref = matrix._ref(self._write(candidate / "robot/RESULT.json", {
            "session_id": "session_0", "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
            "frame_count": 2, "stage_result": numeric_ref, "verified_videos": videos}))
        robot_seal_ref = matrix._ref(self._write(candidate / "ROBOT_STAGE_SEAL.json", {
            "session_id": "session_0", "result": robot_ref,
            "input_hawor_seal": hawor_seal_ref}))
        child_ref = matrix._ref(self._write(candidate / "RESULT.json", {
            "session_id": "session_0", "status": "CAMERA_DIAGNOSTIC_STAGES_COMPLETE",
            "execution_complete": True, "stages": {"hawor": hawor_ref,
            "hawor_seal": hawor_seal_ref, "robot": robot_ref, "robot_seal": robot_seal_ref},
            "stage_status": {"hawor": "SEALED_CAMERA_PREDICTIONS",
                             "robot": "SEALED_CAMERA_DIAGNOSTIC",
                             "clean": "NOT_EVALUATED", "depth": "NOT_EVALUATED",
                             "object6d": "NOT_EVALUATED", "composite": "NOT_EVALUATED"}}))
        result_ref = matrix._ref(self._write(attempt / "RESULT.json", {
            "task_id": task["task_id"], "status": "PASSED", "source_session": "session_0",
            "source_frames_expected": 2, "execution_complete": complete,
            "child_result": child_ref, "robot_video_complete": True,
            "videos": videos, "stage_status": {
                "hawor": "SEALED_CAMERA_PREDICTIONS", "robot": "SEALED_CAMERA_DIAGNOSTIC",
                "clean": "NOT_EVALUATED", "depth": "NOT_EVALUATED",
                "object6d": "NOT_EVALUATED", "composite": "NOT_EVALUATED"},
            "visual_qualified": False, "product_qualified": False,
            "training_eligible": False, "user_review": "PENDING"}))
        receipt_ref = matrix._ref(self._write(Path("tasks/receipts/test_s0.json"), {
            "task_id": task["task_id"], "status": "PASSED", "result": result_ref}))
        task.update(result=result_ref, terminal_receipt=receipt_ref)
        return task

    def test_missing_and_active_never_read_partial(self) -> None:
        first = matrix.build_matrix(self.root)
        self.assertEqual(first["execution_counts"], {"NOT_EVALUATED": 6})
        self.state["tasks"] = [self._task("RUNNING")]
        self._write(matrix.STATE_REL, self.state)
        with patch.object(matrix, "_fill_terminal", side_effect=AssertionError("partial read")):
            result = matrix.build_matrix(self.root)
        self.assertEqual(result["sessions"]["session_0"]["execution"]["status"], "IN_PROGRESS")
        self.assertEqual(result["sessions"]["session_0"]["hawor"]["status"], "NOT_EVALUATED")

    def test_complete_chain_keeps_qualifications_separate(self) -> None:
        self.state["tasks"] = [self._terminal()]
        self._write(matrix.STATE_REL, self.state)
        row = matrix.build_matrix(self.root)["sessions"]["session_0"]
        self.assertEqual(row["execution"]["status"], "PASSED")
        self.assertEqual(row["hawor"]["model_prediction_frames"], {"left": 2, "right": 1})
        self.assertEqual(row["robot"]["arm_status_counts"]["PASS_NUMERIC_CAMERA_DIAGNOSTIC"], 1)
        self.assertTrue(row["robot"]["video_complete_sealed_claim"])
        self.assertEqual(row["stage_qualification"]["occlusion"], "NOT_EVALUATED")
        self.assertFalse(row["visual_qualified"])
        self.assertFalse(row["training_eligible"])

    def test_broken_terminal_chain_is_not_consumed(self) -> None:
        task = self._terminal()
        task["result"]["sha256"] = "0" * 64
        self.state["tasks"] = [task]
        self._write(matrix.STATE_REL, self.state)
        row = matrix.build_matrix(self.root)["sessions"]["session_0"]
        self.assertEqual(row["execution"]["status"], "BLOCKED_TERMINAL_PROOF")
        self.assertEqual(row["hawor"]["status"], "NOT_EVALUATED")
        self.assertEqual(row["evidence"], {})

    def test_ambiguous_task_is_not_latest_selected(self) -> None:
        self.state["tasks"] = [self._task("RUNNING"), self._task("PASSED")]
        self._write(matrix.STATE_REL, self.state)
        row = matrix.build_matrix(self.root)["sessions"]["session_0"]
        self.assertEqual(row["execution"]["status"], "AMBIGUOUS_TASK_SELECTION")


if __name__ == "__main__":
    unittest.main()
