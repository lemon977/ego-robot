"""No-write and fail-closed tests for the fixed T3 Action+Scene sealer."""

from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path
import unittest
from unittest import mock

from chaoyang.governance import seal_exact78_t3_six_scene_matrix_v2 as seal


class T3SixSceneSealV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        # An unrelated active packet may advance task-state heartbeat at any
        # moment, so each test must take its own read-only ledger snapshot.
        self.report = seal.candidate_report()

    def test_six_fixed_sessions_2251_and_zero_quality_passes(self) -> None:
        ready, blockers = seal.readiness(self.report)
        self.assertTrue(ready, blockers)
        self.assertEqual(set(self.report["sessions"]), set(seal.EXPECTED_FRAMES))
        self.assertEqual(sum(seal.EXPECTED_FRAMES.values()), 2251)
        self.assertEqual(self.report["cohort"]["development"], 4)
        self.assertEqual(self.report["cohort"]["pre_frozen_cross_group_holdout"], 2)
        self.assertEqual(self.report["cohort"]["0902_independent_holdout"],
                         "SPLIT_NOT_ESTABLISHED")
        self.assertEqual(self.report["execution_counts"], {
            "action_passed": 6, "scene_passed_execution_only": 6,
            "visual_qualified": 0, "product_qualified": 0, "training_eligible": 0,
        })

    def test_packet_is_bounded_and_only_owns_new_write_set(self) -> None:
        packet = seal.packet_value(self.report)
        self.assertEqual(packet["task_id"], seal.TASK_ID)
        self.assertEqual(packet["commands"], [seal.COMMAND])
        self.assertEqual(packet["required_outputs"],
                         ["MATRIX.json", "RESULT.json", "terminal_receipt"])
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["write_set"], list(seal.WRITE_SET))
        self.assertEqual(packet["inputs"]["pre_register_matrix_sha256"],
                         seal._sha(self.report))
        self.assertGreaterEqual(len(packet["inputs"]["terminal_source_output_refs"]), 100)
        self.assertNotIn(str(seal.TASK_STATE_PATH),
                         {ref["path"] for ref in packet["inputs"]["terminal_source_output_refs"]})
        self.assertEqual(len(packet["runtime_code"]), len(seal.RUNTIME_CODE))

    def test_source_video_refs_and_result_keep_exact_sha(self) -> None:
        packet = seal.packet_value(self.report)
        matrix_ref = {"path": "/synthetic/MATRIX.json", "bytes": 1, "sha256": "a" * 64}
        result = seal.result_value(self.report, packet, matrix_ref)
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["report_integrity"], "SEALED_ACTION_SCENE_SHA_CHAINS")
        for key in ("visual_qualified", "product_qualified", "training_eligible",
                    "geometry_qualified"):
            self.assertFalse(result[key])
        for sid, row in self.report["sessions"].items():
            actual = result["source_output_sha"][sid]
            self.assertEqual(actual["action_rgb_source"], row["evidence"]["action_rgb_source"])
            self.assertEqual(actual["scene_raw_source"], row["evidence"]["scene_raw_source"])
            self.assertEqual(actual["clean_result"], row["evidence"]["scene_clean_result"])
            self.assertEqual(set(actual["clean_internal_videos"]),
                             {"inpaint_out.mp4", "masked_in.mp4"})
            self.assertEqual(set(actual["robot_videos"]),
                             {"ego_robot", "external_full_robot", "raw_motion_source"})
            self.assertTrue(all(len(ref["sha256"]) == 64 for ref in
                                (*actual["clean_internal_videos"].values(),
                                 *actual["robot_videos"].values())))

    def test_action_or_scene_execution_failure_blocks_seal(self) -> None:
        sid = "play_cards_0902_042"
        for lane, status in (("action_status", "FAILED_RUNTIME_FINAL"),
                             ("scene_frontend_status", "FAILED_RUNTIME_FINAL")):
            report = copy.deepcopy(self.report)
            report["sessions"][sid]["execution"][lane] = status
            with self.subTest(lane=lane):
                ready, blockers = seal.readiness(report)
                self.assertFalse(ready)
                self.assertTrue(any(sid in text for text in blockers))
                with self.assertRaisesRegex(seal.T3SceneNotReady, "T3_SCENE_SIX_NOT_READY"):
                    seal.packet_value(report)

    def test_bad_sha_frame_eye_time_or_quality_is_not_sealable(self) -> None:
        sid = "play_cards_0915_001"
        mutations = (
            lambda row: row["evidence"]["scene_raw_source"].update(sha256="bad"),
            lambda row: row["robot"]["videos"]["ego_robot"].update(decoded_frames=149),
            lambda row: row["cross_lane_frame_mapping"].update(scene_physical_source_index=0),
            lambda row: row["cross_lane_frame_mapping"].update(time_mode="SYNCHRONIZED"),
            lambda row: row.update(visual_qualification="QUALIFIED"),
        )
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report["sessions"][sid])
            with self.subTest(mutation=mutate.__code__.co_firstlineno):
                self.assertFalse(seal.readiness(report)[0])

    def test_check_only_never_registers_or_writes(self) -> None:
        before = tuple(path.exists() for path in (seal.ATTEMPT, seal.PACKET, seal.RECEIPT))
        with mock.patch.object(seal, "register", side_effect=AssertionError("registered")), \
                mock.patch.object(seal, "finish", side_effect=AssertionError("finished")):
            dry = seal.dry_run()
        self.assertIn(dry["status"], {"DRY_RUN_READY", "DRY_RUN_EVIDENCE_READY_PUBLISHER_BUSY"})
        self.assertFalse(dry["registered"])
        self.assertEqual(dry["cohort"]["frames"], 2251)
        self.assertEqual(tuple(path.exists() for path in (seal.ATTEMPT, seal.PACKET, seal.RECEIPT)),
                         before)

    def test_h24_readiness_and_check_only_fail_without_registration(self) -> None:
        before = seal.CAMPAIGN_END - timedelta(microseconds=1)
        at = seal.CAMPAIGN_END
        after = at + timedelta(seconds=1)
        self.assertTrue(seal.readiness(self.report, now=before)[0])
        for stamp in (at, after):
            with self.subTest(stamp=stamp):
                ready, blockers = seal.readiness(self.report, now=stamp)
                self.assertFalse(ready)
                self.assertIn("CAMPAIGN_HARD_CUTOFF_PASSED", blockers)
                with self.assertRaisesRegex(seal.T3SceneNotReady,
                                            "CAMPAIGN_HARD_CUTOFF_PASSED"):
                    seal.packet_value(self.report, now=stamp)
                with mock.patch.object(seal, "register", side_effect=AssertionError("registered")):
                    dry = seal.dry_run(now=stamp)
                self.assertEqual(dry["status"], "DRY_RUN_H24_CUTOFF")
                self.assertFalse(dry["registered"])

    def test_packet_deadline_is_minimum_of_one_hour_and_fixed_h24(self) -> None:
        early = seal.CAMPAIGN_END - timedelta(hours=2)
        late = seal.CAMPAIGN_END - timedelta(seconds=30)
        for stamp, expected in ((early, early + timedelta(hours=1)),
                                (late, seal.CAMPAIGN_END)):
            with self.subTest(stamp=stamp):
                packet = seal.packet_value(self.report, now=stamp)
                self.assertEqual(packet["deadline_at"], expected.isoformat())
                self.assertEqual(packet["campaign_deadline_at"],
                                 "2026-10-01T18:47:25.116287+08:00")

    def test_register_refuses_after_h24_before_any_write(self) -> None:
        packet = seal.packet_value(self.report, now=seal.CAMPAIGN_END - timedelta(hours=1))
        before = tuple(path.exists() for path in (seal.ATTEMPT, seal.PACKET, seal.RECEIPT))
        with mock.patch.object(seal, "datetime") as clock:
            clock.now.return_value = seal.CAMPAIGN_END + timedelta(seconds=1)
            with self.assertRaisesRegex(RuntimeError, "CAMPAIGN_HARD_CUTOFF_PASSED"):
                seal.register(self.report, packet)
        self.assertEqual(tuple(path.exists() for path in (seal.ATTEMPT, seal.PACKET, seal.RECEIPT)),
                         before)

    def test_cli_check_only_exits_nonzero_after_h24(self) -> None:
        with mock.patch.object(seal, "dry_run", return_value={
                "status": "DRY_RUN_H24_CUTOFF", "registered": False}), \
                mock.patch("sys.argv", ["sealer", "--check-only"]), \
                mock.patch.object(seal, "register", side_effect=AssertionError("registered")):
            self.assertEqual(seal.main(), 2)

    def test_active_publisher_is_separate_from_evidence_readiness(self) -> None:
        blockers = seal.publisher_blockers(
            {"next_task": {"task_id": "other"}, "tasks": [{"status": "RUNNING"}]},
            {"task_packets": [{"task_id": "other", "execution_allowed": True}]})
        self.assertEqual(len(blockers), 3)
        self.assertTrue(seal.readiness(self.report)[0])

    def test_no_legacy_sealer_or_active_successor_is_read(self) -> None:
        original_open = Path.open
        forbidden = ("seal_exact78_t3_six_matrix_v1.py",
                     "exact78_white_device_v5", "exact78_cards042_first_anchor")

        def guarded_open(path: Path, *args, **kwargs):
            self.assertFalse(any(token in str(path) for token in forbidden), str(path))
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            seal.candidate_report()


if __name__ == "__main__":
    unittest.main()
