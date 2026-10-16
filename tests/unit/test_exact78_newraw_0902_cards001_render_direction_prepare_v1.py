"""CPU/read-only checks for the 201-frame new Raw camera-direction blueprint."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import unittest
from unittest import mock

from chaoyang.governance import prepare_exact78_newraw_0902_cards001_render_direction_v1 as candidate


class NewRawCards001DirectionPrepareTests(unittest.TestCase):
    def test_fixed_stage_spec_is_201_frames_without_recomputing_robot(self) -> None:
        spec, states, intrinsics = candidate.frozen_spec()
        self.assertEqual(spec["session_id"], "play_cards_0902_001")
        self.assertEqual(spec["source_kind"], "TRACKER_CLIP_RGB")
        self.assertEqual(spec["inputs"]["stage_result"]["sha256"], candidate.STAGE_SHA)
        self.assertEqual(spec["inputs"]["states"]["sha256"], candidate.STATES_SHA)
        self.assertEqual(spec["inputs"]["hawor_npz"]["sha256"], candidate.HAWOR_SHA)
        self.assertEqual(spec["inputs"]["raw_video"]["sha256"], candidate.RAW_SHA)
        self.assertEqual(states["q_arm"].shape, (201, 2, 7))
        self.assertEqual(states["q_hand"].shape, (201, 2, 22))
        self.assertEqual(intrinsics.shape, (201, 3, 3))
        self.assertFalse(candidate.OUT.exists())

    def test_projection_preflight_keeps_hold_and_model_truth_separate(self) -> None:
        spec, states, intrinsics = candidate.frozen_spec()
        rows = candidate.projection_observations(spec, states, intrinsics)
        self.assertEqual([(row["frame"], row["side"]) for row in rows],
                         [(0, "left"), (0, "right"), (100, "left"), (100, "right"),
                          (200, "left"), (200, "right")])
        for row in rows:
            self.assertEqual(row["comparison_scope"],
                             "INTERNAL_MODEL_TARGET_ONLY_NOT_VISUAL_TRUTH")
            self.assertEqual(row["arm_hold_not_action_evidence"],
                             not row["arm_status"].startswith("PASS"))
        # F0 is numerically anchored on both sides; this is a projection
        # consistency check, not evidence of true 3-D hand position.
        for row in rows[:2]:
            self.assertFalse(row["arm_hold_not_action_evidence"])
            target = row["hawor_model_wrist_px"]
            old = row["old_inverse_camera_wrist_px"]
            new = row["candidate_base_to_camera_wrist_px"]
            old_error = sum((a - b) ** 2 for a, b in zip(target, old)) ** .5
            new_error = sum((a - b) ** 2 for a, b in zip(target, new)) ** .5
            self.assertLess(new_error, old_error)
        self.assertTrue(rows[-1]["arm_hold_not_action_evidence"])

    def test_blueprint_is_unregistered_and_single_variable(self) -> None:
        spec, _, _ = candidate.frozen_spec()
        packet = candidate.packet_blueprint(spec)
        self.assertTrue(packet["blueprint_only"])
        self.assertFalse(packet["registration_allowed"])
        self.assertIsNone(packet["writer"])
        self.assertEqual(packet["commands"], [candidate.COMMAND])
        self.assertEqual(packet["config"]["change_only"],
                         "EGO_CAMERA_INV_T_CAMERA_BASE_TO_T_CAMERA_BASE")
        self.assertEqual(packet["config"]["candidate_contract_key"], candidate.CONTRACT_KEY)
        self.assertIn("NEW_SESSION_CANDIDATE_ROUTE_STAGED_WITH_PACKET",
                      packet["prerequisites"])
        for key in ("flange_parent_changed", "q_base_mount_recomputed",
                    "numeric_gates_changed"):
            self.assertFalse(packet["config"][key])
        self.assertEqual(packet["config"]["ik_calls"], 0)
        self.assertEqual(packet["config"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["write_set"][0], str(candidate.OUT.relative_to(candidate.REPO_ROOT)))
        self.assertEqual(set(packet["required_outputs"][2:5]),
                         {f"candidate/render/{name}" for name in candidate.VIDEO_FILES.values()})
        self.assertEqual(packet["config"]["observation_plan"]["projection_frames"],
                         [0, 100, 200])
        self.assertEqual(packet["config"]["observation_plan"]["old_video_ab"],
                         "EVALUATOR_ONLY_AFTER_FRESH_SEAL")
        self.assertFalse(candidate.OUT.exists())

    def test_read_only_check_never_opens_old_robot_videos_or_writes(self) -> None:
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            self.assertNotIn(path.name, candidate.VIDEO_FILES.values())
            return original_open(path, *args, **kwargs)

        before = tuple(path.exists() for path in (candidate.OUT, candidate.PACKET,
                                                  candidate.RECEIPT))
        with mock.patch.object(Path, "open", guarded_open):
            dry = candidate.check_only()
        self.assertEqual(dry["status"], "DRY_RUN_EVIDENCE_READY_NOT_ROUTABLE")
        self.assertIn(dry["route_status"], {"NEW_SESSION_CANDIDATE_ROUTE_NOT_PUBLISHED",
                                             "PUBLISHED_NEW_SESSION_CANDIDATE_ROUTE"})
        self.assertFalse(dry["registered"])
        self.assertFalse(dry["rendered"])
        self.assertEqual(dry["frames"], 201)
        self.assertEqual(tuple(path.exists() for path in (candidate.OUT, candidate.PACKET,
                                                          candidate.RECEIPT)), before)

    def test_sha_drift_and_h24_cutoff_fail_closed(self) -> None:
        with mock.patch.object(candidate, "STAGE_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "SEALED_STAGE_SHA_DRIFT"):
                candidate.frozen_spec()
        spec, _, _ = candidate.frozen_spec()
        with self.assertRaisesRegex(RuntimeError, "CAMPAIGN_HARD_CUTOFF_PASSED"):
            candidate.packet_blueprint(spec, now=candidate.CAMPAIGN_END)
        packet = candidate.packet_blueprint(
            spec, now=candidate.CAMPAIGN_END - timedelta(seconds=30))
        self.assertEqual(packet["deadline_at"], candidate.CAMPAIGN_END.isoformat())


if __name__ == "__main__":
    unittest.main()
