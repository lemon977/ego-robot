"""No-registration tests for the source-neutral stage mapping prep."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli_v7.source_neutral_plan_v1 import plan_one
from chaoyang.cli_v8 import source_neutral_stage_packet_plan_v1 as stage
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


TRACKER = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_002")
TRACKER_0901 = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards/play_cards_0901_001")
VST = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/004")


class StagePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.accepted = stage._accepted_code()
        cls.tracker = plan_one(TRACKER)
        cls.tracker_0901 = plan_one(TRACKER_0901)
        cls.vst = plan_one(VST)

    def test_real_source_examples_remain_blocked_plans(self) -> None:
        for value in (self.tracker, self.tracker_0901, self.vst):
            with self.subTest(kind=value["source"]["source_kind"]):
                plan = stage.plan_source(value["source"], value["submission_input_sha256"],
                                         accepted=self.accepted)
                self.assertEqual(plan["status"], "PREP_ONLY_BLOCKED_NO_PACKET")
                self.assertEqual([row["node"] for row in plan["stages"]],
                                 ["CPU_ADMISSION", "HAWOR_V4", "ROBOT_V1", "SCENE"])
                self.assertTrue(all(row["run_signature"] is None and not row["execution_allowed"]
                                    for row in plan["stages"]))
                self.assertFalse(plan["execution_allowed"])

    def test_tracker_scene_selects_stereo_and_vst_uses_same_raw_bytes(self) -> None:
        tracker = stage.plan_source(self.tracker["source"], self.tracker["submission_input_sha256"],
                                    accepted=self.accepted)
        vst = stage.plan_source(self.vst["source"], self.vst["submission_input_sha256"],
                                accepted=self.accepted)
        self.assertEqual(tracker["stages"][3]["binding"]["source_roles"],
                         ["source_stereo_video", "camera_params"])
        self.assertEqual(vst["stages"][3]["binding"]["source_roles"],
                         ["raw/vst.h264", "raw/camera_params.json"])
        self.assertEqual(vst["time_mode"], "TIMEBASE_UNVERIFIED")

    def test_signature_drift_rejected(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "DESCRIPTOR_OR_SIGNATURE_DRIFT"):
            stage.plan_source(self.tracker["source"], "0" * 64, accepted=self.accepted)

    def test_old_derived_hand_role_injection_rejected(self) -> None:
        source = copy.deepcopy(self.tracker["source"])
        source["source_files"]["hand"] = {"path": str(TRACKER / "processed/hand.json"),
                                          "bytes": 1, "sha256": "0" * 64}
        with self.assertRaisesRegex(ColdStartInputError, "DESCRIPTOR_OR_SIGNATURE_DRIFT"):
            stage.plan_source(source, stage._digest(source), accepted=self.accepted)

    def test_wrong_physical_eye_or_promoted_time_rejected(self) -> None:
        source = copy.deepcopy(self.vst["source"])
        source["physical_eye"] = "left_sourceIndex_0"
        with self.assertRaisesRegex(ColdStartInputError, "VST_EYE_OR_TIME_CLAIM_DRIFT"):
            stage.plan_source(source, stage._digest(source), accepted=self.accepted)
        source = copy.deepcopy(self.vst["source"])
        source["time_mode"] = "QPC_HEAD_SYNC_VERIFIED"
        with self.assertRaisesRegex(ColdStartInputError, "VST_EYE_OR_TIME_CLAIM_DRIFT"):
            stage.plan_source(source, stage._digest(source), accepted=self.accepted)

    def test_cross_domain_descriptor_rejected(self) -> None:
        source = copy.deepcopy(self.tracker["source"])
        source["session_root"] = str(VST)
        with self.assertRaisesRegex(ColdStartInputError, "SOURCE_DOMAIN_UNVERIFIED"):
            stage.plan_source(source, stage._digest(source), accepted=self.accepted)

    def test_code_acceptance_absence_rejected(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "V7_CODE_ACCEPTANCE_REQUIRED"):
            stage.plan_source(self.tracker["source"], self.tracker["submission_input_sha256"],
                              accepted={})

    def test_sealed_acceptance_result_sha_drift_rejected(self) -> None:
        with patch.object(stage, "validate_artifact_ref", return_value=["sha drift"]):
            with self.assertRaisesRegex(ColdStartInputError, "RESULT_SHA_DRIFT"):
                stage._accepted_code()

    def test_dependency_fingerprint_is_stable_without_packet_or_execution(self) -> None:
        first = stage.plan_source(self.tracker["source"], self.tracker["submission_input_sha256"],
                                  accepted=self.accepted)
        second = stage.plan_source(self.tracker["source"], self.tracker["submission_input_sha256"],
                                   accepted=self.accepted)
        self.assertEqual([row["dependency_fingerprint"] for row in first["stages"]],
                         [row["dependency_fingerprint"] for row in second["stages"]])
        self.assertNotIn("task_packet", first)
        self.assertEqual(first["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
