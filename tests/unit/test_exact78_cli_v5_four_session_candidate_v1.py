"""Read-only identity and terminal-chain tests for the four-session v5 prep."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli_v5 import exact78_four_session_manifest_candidate_v1 as v5
from chaoyang.cli_v5 import exact78_newraw_cpu_submit_manifest_v1 as cpu_adapter
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


class FourSessionAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.manifest_ref = v5.frozen_manifest()
        cls.rows = {row["session_id"]: row for row in cls.manifest["sessions"]}
        cls.old = cls.rows["play_cards_0915_003"]
        cls.fresh = cls.rows["play_cards_0915_004"]

    def test_four_exact_sessions_and_fresh_source_integrity(self) -> None:
        self.assertEqual(list(self.rows), ["get_potato_chips_0901_001",
                                          "play_cards_0902_002",
                                          "play_cards_0915_003",
                                          "play_cards_0915_004"])
        self.assertEqual(self.fresh["frame_count_decoded"], 152)
        self.assertEqual(self.fresh["crop_xywh"], [2048, 0, 2048, 1536])
        self.assertEqual(self.fresh["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(self.fresh["independent_source_group"], "UNVERIFIED")
        self.assertEqual(set(self.fresh["source_files"]), v5.RAW_ROLES)
        self.assertNotIn("sealed_robot_terminal", self.fresh)

    def test_manifest_sha_drift_rejects_before_planning(self) -> None:
        with patch.object(v5, "MANIFEST_SHA", "0" * 64), \
             patch.object(v5.phase_exact, "plan_one") as planner:
            with self.assertRaisesRegex(ColdStartInputError, "CLI_V5_MANIFEST_SHA_DRIFT"):
                v5.plan_one(Path(self.fresh["source_root"]),
                            self.fresh["submission_input_sha256"])
        planner.assert_not_called()

    def test_unlisted_or_wrong_sha_rejects_before_planning(self) -> None:
        with patch.object(v5.phase_exact, "plan_one") as planner:
            for source, digest in ((Path(self.fresh["source_root"]), "0" * 64),
                                   (Path(v5.VST_ROOT) / "005", "0" * 64)):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ColdStartInputError,
                                                "CLI_V5_SESSION_NOT_IN_FROZEN_MANIFEST"):
                        v5._entry(source, digest)
        planner.assert_not_called()

    def test_eye_clock_and_role_mutations_are_rejected(self) -> None:
        original = v5.load_json
        for mutate in (
            lambda data: data["sessions"][3].update(crop_xywh=[0, 0, 2048, 1536]),
            lambda data: data["sessions"][3].update(time_mode="QPC_NS_VERIFIED"),
            lambda data: data["sessions"][3]["source_files"].pop("raw/vst.ts.jsonl"),
            lambda data: data["sessions"][3]["source_files"]["raw/vst.h264"].update(sha256="0" * 64),
        ):
            with self.subTest(mutate=mutate):
                def changed(path):
                    value = original(path)
                    if path == v5.MANIFEST:
                        value = copy.deepcopy(value)
                        mutate(value)
                    return value
                with patch.object(v5, "load_json", side_effect=changed):
                    with self.assertRaises(ColdStartInputError):
                        v5.frozen_manifest()

    def test_sealed_003_attaches_without_rebuilding_historical_packet(self) -> None:
        with patch.object(v5.phase_exact, "plan_one") as planner:
            decision = v5.plan_one(Path(self.old["source_root"]),
                                   self.old["submission_input_sha256"])
        planner.assert_not_called()
        self.assertTrue(decision["sealed_attachment"])
        self.assertEqual(decision["node"], "ROBOT_V1")
        self.assertFalse(decision["visual_qualified"])

    def test_sealed_003_robot_pin_or_hawor_chain_drift_rejects(self) -> None:
        changed = copy.deepcopy(self.old)
        changed["sealed_robot_terminal"]["result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ColdStartInputError, "CLI_V5_003_FROZEN_ROBOT_TRIPLE_DRIFT"):
            v5._sealed_003(changed)
        original = v5._sealed_raw_stage
        def corrupt(task_id, *args, **kwargs):
            packet, result, refs = original(task_id, *args, **kwargs)
            if "_hawor_" in task_id:
                result = copy.deepcopy(result)
                result["producer_result"]["sha256"] = "0" * 64
            return packet, result, refs
        with patch.object(v5, "_sealed_raw_stage", side_effect=corrupt):
            with self.assertRaisesRegex(ColdStartInputError, "CLI_V5_003_SEALED_CHAIN_DRIFT"):
                v5._sealed_003(self.old)

    def test_fresh_004_cpu_ready_is_only_read_only_preview(self) -> None:
        prepared = v5.phase_exact.producer.preflight(Path(self.fresh["source_root"]))
        packet = {"inputs": {"source_files": self.fresh["source_files"]}}
        with patch.object(v5.phase_exact, "plan_one", return_value=(
                {"status": "READY", "node": "CPU_ADMISSION"}, packet)), \
             patch.object(v5.phase_exact.producer, "preflight", return_value=prepared):
            decision = v5.plan_one(Path(self.fresh["source_root"]),
                                   self.fresh["submission_input_sha256"])
        self.assertEqual(decision["status"], "READY")
        self.assertFalse(decision["registration_started"])
        self.assertEqual(decision["model_calls"], 0)

    def test_fresh_004_frame_pair_or_eye_drift_rejects(self) -> None:
        prepared = v5.phase_exact.producer.preflight(Path(self.fresh["source_root"]))
        packet = {"inputs": {"source_files": self.fresh["source_files"]}}
        for change in (lambda d: d["frame_rows"][1].update(qpc_raw_tick=d["frame_rows"][0]["qpc_raw_tick"]),
                       lambda d: d["frame_rows"][0].update(sbs_physical_eye="right_sourceIndex_0"),
                       lambda d: d["frame_rows"][0].update(pts=0)):
            with self.subTest(change=change):
                corrupt = copy.deepcopy(prepared)
                change(corrupt["descriptor"])
                with patch.object(v5.phase_exact, "plan_one", return_value=(
                        {"status": "READY", "node": "CPU_ADMISSION"}, packet)), \
                     patch.object(v5.phase_exact.producer, "preflight", return_value=corrupt):
                    with self.assertRaisesRegex(ColdStartInputError,
                                                "CLI_V5_004_FRAME_TIME_MAP_DRIFT"):
                        v5.plan_one(Path(self.fresh["source_root"]),
                                    self.fresh["submission_input_sha256"])

    def test_cpu_adapter_rejects_sealed_003_and_has_no_register_mode(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "CLI_V5_CPU_FRESH_004_ONLY"):
            cpu_adapter.check_one(Path(self.old["source_root"]),
                                  self.old["submission_input_sha256"])
        with self.assertRaises(SystemExit) as caught:
            cpu_adapter.main(["--session", self.fresh["source_root"],
                              "--expected-input-sha256", self.fresh["submission_input_sha256"],
                              "--register-and-run"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
