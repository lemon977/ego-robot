"""CPU-only checks for the read-only, fail-closed six-session scene join."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from chaoyang.governance import build_exact78_t3_six_scene_matrix_v2 as candidate


class T3SixSceneMatrixV2Tests(unittest.TestCase):
    def test_qualification_is_monotonic_and_never_self_approves(self) -> None:
        for action_claim in (False, None, True):
            for scene_claim in (False, None, True):
                value = candidate.qualification(action_claim, scene_claim)
                self.assertNotEqual(value, "QUALIFIED")
                if action_claim is False or scene_claim is False:
                    self.assertEqual(value, "NOT_QUALIFIED")
                elif action_claim is True and scene_claim is True:
                    self.assertEqual(value, "NOT_EVALUATED_INDEPENDENT_REVIEW_REQUIRED")
                else:
                    self.assertEqual(value, "NOT_EVALUATED")

    def test_absent_scene_has_complete_nonqualifying_shape(self) -> None:
        row = candidate._safe_scene({"status": "FAILED_RUNTIME_FINAL"}, "task")
        self.assertEqual(row["task_id"], "task")
        self.assertEqual(row["mask"]["status"], "NOT_EVALUATED")
        self.assertEqual(row["clean"]["status"], "NOT_EVALUATED")
        self.assertIsNone(row["visual_qualified"])
        self.assertIsNone(row["product_qualified"])
        self.assertIsNone(row["training_eligible"])

    def test_sealed_six_have_distinct_source_sha_mapping_and_gates(self) -> None:
        matrix = candidate.build_matrix()
        self.assertEqual(matrix["status"], "READ_ONLY_SUCCESSOR_NOT_T3_SEAL")
        self.assertEqual(len(matrix["selection"]["development"]), 4)
        self.assertEqual(len(matrix["selection"]["holdout"]), 2)
        self.assertEqual(matrix["selection"]["source_holdout_gap"]["status"],
                         "SPLIT_NOT_ESTABLISHED")
        expected = {
            "get_potato_chips_0901_013": (870, 25, 870, 0, 610),
            "get_potato_chips_0902_103": (284, 30, 284, 0, 180),
            "play_cards_0901_001": (645, 30, 645, 0, 645),
            "play_cards_0902_042": (171, 30, 171, 0, 171),
            "play_cards_0915_001": (150, 30, 150, 0, 150),
            "play_cards_0915_002": (131, 30, 131, 0, 129),
        }
        self.assertEqual(set(matrix["sessions"]), set(expected))
        for sid, (frames, fps, human, equipment, obj) in expected.items():
            with self.subTest(session=sid):
                row = matrix["sessions"][sid]
                self.assertEqual((row["raw"]["frames"], row["raw"]["nominal_fps"]),
                                 (frames, fps))
                self.assertEqual(row["execution"]["action_status"], "PASSED")
                self.assertEqual(row["execution"]["scene_frontend_status"],
                                 "PASSED_EXECUTION_ONLY")
                self.assertEqual(row["coverage"]["mask_frames"], frames)
                self.assertEqual(row["coverage"]["clean_frames"], frames)
                counts = row["coverage"]["sam_instance_nonempty_frames"]
                self.assertEqual((counts["human"], counts["equipment"], counts["object"]),
                                 (human, equipment, obj))
                self.assertEqual(row["cross_lane_frame_mapping"]
                                 ["scene_raw_support_counts"]["model_input_rows"], frames)
                self.assertEqual(row["visual_qualification"], "NOT_QUALIFIED")
                self.assertEqual(row["full_product_qualification"], "NOT_QUALIFIED")
                self.assertEqual(row["training_label_qualification"], "NOT_QUALIFIED")
                self.assertEqual(row["human_review"], "PENDING")
                self.assertEqual(row["ai_review"], "NOT_EVALUATED")
                self.assertEqual(row["geometry_prequalification"]["trusted_occlusion"]
                                 ["geometry_decision_coverage"], "NOT_EVALUATED")
                for name in ("depth", "object6d"):
                    self.assertEqual(row["geometry_prequalification"][name]
                                     ["fresh_result"], "NOT_EVALUATED")
                evidence = row["evidence"]
                for key in ("scene_raw_source", "scene_camera_params", "scene_frontend_spec",
                            "scene_map_seal", "scene_map_result", "scene_raw_seal",
                            "scene_raw_result", "action_hawor_input_audit"):
                    self.assertEqual(len(evidence[key]["sha256"]), 64, key)
                self.assertEqual(set(evidence["scene_clean_internal_videos"]),
                                 {"inpaint_out.mp4", "masked_in.mp4"})
                for video in evidence["scene_clean_internal_videos"].values():
                    self.assertEqual(video["decoded_frames"], frames)
                    self.assertEqual(len(video["artifact"]["sha256"]), 64)
                for video in row["robot"]["videos"].values():
                    self.assertEqual(video["decoded_frames"], frames)
                    self.assertEqual(len(video["video"]["sha256"]), 64)
                if row["source"] == "0915":
                    self.assertEqual(row["cross_lane_frame_mapping"]["status"],
                                     "SAME_RAW_VST_BYTES_INDEX_ONLY_TIMEBASE_UNVERIFIED")
                    self.assertTrue(row["cross_lane_frame_mapping"]["same_source_file_sha"])
                    self.assertEqual(row["cross_lane_frame_mapping"]["time_mode"],
                                     "TIMEBASE_UNVERIFIED")
                    self.assertEqual(row["cross_lane_frame_mapping"]
                                     ["scene_physical_source_index"], 1)
                    self.assertEqual(row["cross_lane_frame_mapping"]
                                     ["scene_raw_support_counts"]["qpc_nonnull"], frames)
                else:
                    self.assertEqual(row["cross_lane_frame_mapping"]["status"],
                                     "NOT_PROVEN_MONO_STEREO_FRAME_MAPPING")
                    self.assertFalse(row["cross_lane_frame_mapping"]["same_source_file_sha"])
                    self.assertEqual(row["cross_lane_frame_mapping"]
                                     ["scene_physical_source_index"], 0)
                    self.assertEqual(row["cross_lane_frame_mapping"]
                                     ["scene_raw_support_counts"]["qpc_nonnull"], 0)

    def test_failed_scene_branch_does_not_crash_or_promote_qualification(self) -> None:
        original = candidate.scene_v1._terminal_frontend
        target = "play_cards_0902_042"

        def failed_one(root, sid, task, frames):
            if sid == target:
                return {"status": "FAILED_RUNTIME_FINAL",
                        "task_id": candidate.scene_v1.FRONTENDS[sid][0]}
            return original(root, sid, task, frames)

        with mock.patch.object(candidate.scene_v1, "_terminal_frontend", side_effect=failed_one):
            row = candidate.build_matrix()["sessions"][target]
        self.assertEqual(row["execution"]["scene_frontend_status"], "FAILED_RUNTIME_FINAL")
        self.assertEqual(row["coverage"]["status"], "NOT_EVALUATED")
        self.assertEqual(row["visual_qualification"], "NOT_QUALIFIED")
        self.assertEqual(row["full_product_qualification"], "NOT_QUALIFIED")
        self.assertEqual(row["training_label_qualification"], "NOT_QUALIFIED")

    def test_action_source_sha_drift_fails_closed(self) -> None:
        sid = "play_cards_0902_042"
        base = candidate.action.build_matrix(candidate.ROOT)["sessions"][sid]
        freeze = candidate.action._json(candidate.ROOT / candidate.action.FREEZE_REL)["selected"]
        state = candidate.action._json(candidate.ROOT / candidate.scene_v1.STATE_REL)
        task_id = candidate.scene_v1.FRONTENDS[sid][0]
        task = next(row for row in state["tasks"] if row["task_id"] == task_id)
        scene = candidate._safe_scene(candidate.scene_v1._terminal_frontend(
            candidate.ROOT, sid, task, base["frames"]), task_id)
        wrong = copy.deepcopy(base["raw_source"])
        wrong["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "HAWOR_ACTION_SOURCE_NOT_T0"):
            candidate._scene_source_evidence(candidate.ROOT, sid, base["frames"],
                                             base["nominal_fps"], wrong, freeze[sid], scene, base)

    def test_pinned_legacy_join_sha_drift_fails_before_reading_results(self) -> None:
        with mock.patch.object(candidate, "PINNED_SCENE_V1_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "PINNED_SCENE_V1_BUILDER_SHA_DRIFT"):
                candidate.build_matrix()


if __name__ == "__main__":
    unittest.main()
