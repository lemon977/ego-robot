"""Check-only v9 bridge tests over two fresh, known-format raw sessions."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli_v9 import cpu_admission_packet_v1 as bridge
from chaoyang.governance import execute_exact78_cli_v9_cpu_admission_candidate_v1 as executor
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


TRACKER = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_005")
VST = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/005")


class V9BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tracker = bridge.prepare(TRACKER)
        cls.vst = bridge.prepare(VST)

    def test_fresh_domains_decode_and_bind_v7_to_cpu_identity(self) -> None:
        for prepared, kind, count in ((self.tracker, "TRACKER_CLIP_RGB", 183),
                                      (self.vst, "0915_RAW_VST", 149)):
            with self.subTest(kind=kind):
                descriptor = prepared["cpu"]["descriptor"]
                self.assertEqual(descriptor["source_kind"], kind)
                self.assertEqual(descriptor["decoded_frames"], count)
                self.assertEqual(len(descriptor["frame_rows"]), count)
                self.assertEqual(descriptor["source_files"], prepared["v7"]["source"]["source_files"])
                self.assertFalse(any("pico" in role or "manus" in role or "tactile" in role
                                     or "processed" in role for role in descriptor["source_files"]))

    def test_packet_reuses_published_cpu_operation_and_finite_budget(self) -> None:
        for prepared in (self.tracker, self.vst):
            packet = bridge.packet_value(prepared)
            self.assertEqual(packet["phase"], "EXACT78_NEWRAW_CPU_ADMISSION")
            self.assertIn(executor.cpu.OPERATION, packet["commands"][0])
            self.assertEqual(executor.cpu._assert_cli_admission_ready(packet),
                             "PUBLISHED_CPU_CANDIDATE_ROUTE")
            self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
            self.assertEqual(packet["budgets"]["gpu_count"], 0)
            self.assertEqual(packet["budgets"]["model_calls"], 0)
            self.assertEqual(packet["budgets"]["wall_seconds_max"], 1800)
            self.assertEqual(packet["inputs"]["v7_source_only_sha256"],
                             prepared["v7"]["submission_input_sha256"])
        self.assertNotEqual(bridge.packet_value(self.tracker)["task_id"],
                            bridge.packet_value(self.vst)["task_id"])

    def test_v7_source_sha_drift_rejected_before_packet(self) -> None:
        bad = copy.deepcopy(self.tracker["v7"])
        bad["source"]["source_files"]["video"]["sha256"] = "0" * 64
        with patch.object(bridge, "plan_one", return_value=bad):
            with self.assertRaisesRegex(ColdStartInputError, "V7_SOURCE_TO_DECODED_DESCRIPTOR_DRIFT"):
                bridge.prepare(TRACKER)

    def test_unpermitted_motion_role_rejected_before_packet(self) -> None:
        bad = copy.deepcopy(self.tracker["v7"])
        bad["source"]["source_files"]["pico_hand_motion"] = bad["source"]["source_files"]["video"]
        with patch.object(bridge, "plan_one", return_value=bad):
            with self.assertRaisesRegex(ColdStartInputError, "V7_SOURCE_TO_DECODED_DESCRIPTOR_DRIFT"):
                bridge.prepare(TRACKER)

    def test_unaccepted_v7_code_blocks_before_decode(self) -> None:
        with patch.object(bridge, "_accepted_code", side_effect=ColdStartInputError("V7_NOT_ACCEPTED")), \
             patch.object(bridge.cpu, "preflight") as preflight:
            with self.assertRaisesRegex(ColdStartInputError, "V7_NOT_ACCEPTED"):
                bridge.prepare(TRACKER)
        preflight.assert_not_called()

    def test_vst_eye_or_clock_upgrade_rejected(self) -> None:
        bad = copy.deepcopy(self.vst["v7"])
        bad["source"]["physical_eye"] = "left_sourceIndex_0"
        with patch.object(bridge, "plan_one", return_value=bad):
            with self.assertRaisesRegex(ColdStartInputError, "V7_SOURCE_TO_DECODED_DESCRIPTOR_DRIFT"):
                bridge.prepare(VST)

    def test_published_route_drift_rejected(self) -> None:
        with patch.object(bridge.cpu, "_assert_cli_admission_ready",
                          side_effect=RuntimeError("PUBLISHED_ROUTE_DRIFT")):
            with self.assertRaisesRegex(RuntimeError, "PUBLISHED_ROUTE_DRIFT"):
                bridge.packet_value(self.tracker)

    def test_check_only_does_not_register_or_run(self) -> None:
        with patch.object(executor.cpu, "register") as register, \
             patch.object(executor.cpu, "run_registered_child") as run:
            value = executor.check_only(TRACKER)
        self.assertEqual(value["status"], "READY_NOT_REGISTERED")
        self.assertEqual(value["route"], "PUBLISHED_CPU_CANDIDATE_ROUTE")
        self.assertFalse(value["registration_started"])
        register.assert_not_called()
        run.assert_not_called()

    def test_existing_task_id_or_writer_fails_closed(self) -> None:
        packet = bridge.packet_value(self.tracker)
        state = {"tasks": [{"task_id": packet["task_id"], "status": "PASSED"}], "next_task": None}
        index = {"task_packets": []}
        with patch.object(executor, "load_json", side_effect=lambda p: state if p == executor.cpu.TASK_STATE_PATH else index):
            value = executor.check_only(TRACKER)
        self.assertEqual(value["status"], "EXISTING_TASK_ID_REQUIRES_SEALED_ATTACHMENT")
        state["next_task"] = {"task_id": "other"}
        with patch.object(executor, "load_json", side_effect=lambda p: state if p == executor.cpu.TASK_STATE_PATH else index):
            value = executor.check_only(TRACKER)
        self.assertEqual(value["status"], "WAIT_ACTIVE_WRITER")


if __name__ == "__main__":
    unittest.main()
