"""CPU-only negative tests for the unregistered sealed-to-HaWoR-v3 seam."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError, file_ref
from chaoyang.governance.exact78_hawor_v3_sealed_handoff_candidate import (
    ADMISSION_ADAPTER, HANDOFF_CODE, V3_MODEL_CORE, prepare_v3_handoff,
)
from tests.unit.test_exact78_hawor_descriptor_adapter_candidate import (
    HaWoRDescriptorAdapterCandidateTest,
)


class HaWoRV3SealedHandoffCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = HaWoRDescriptorAdapterCandidateTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _items(self, *, vst: bool = False, old_derived: bool = False) -> tuple[dict, ...]:
        source = (self.fixture._vst() if vst else
                  self.fixture._tracker(old_derived=old_derived))
        items = self.fixture._sealed(source, vst=vst)
        packet = copy.deepcopy(items[0])
        packet["runtime_code"] = [file_ref(path) for path in
                                  (V3_MODEL_CORE, ADMISSION_ADAPTER, HANDOFF_CODE)]
        packet["read_set"] = ["AGENTS.md", "producer terminal receipt"]
        return packet, *items[1:]

    def _prepare(self, items: tuple[dict, ...]) -> dict:
        return prepare_v3_handoff(packet=items[0], repo_root=self.fixture.repo,
                                  allowed_root=self.fixture.data)

    def test_non_t0_tracker_handoff_reuses_only_v3_core_contract(self) -> None:
        handoff = self._prepare(self._items())
        spec = handoff["v3_core_spec"]
        self.assertEqual(spec["session_id"], "play_cards_0915_009")
        self.assertEqual(spec["frame_count"], 2)
        self.assertEqual(spec["frame_rows_sealed"][1]["decoded_display_index"], 1)
        self.assertEqual(handoff["permitted_core_functions"],
                         ["extract_model_frames", "run_hawor", "frame_status"])
        self.assertEqual(handoff["forbidden_v3_functions"], ["build_spec", "main"])
        self.assertFalse(handoff["model_invoked"])
        self.assertFalse(handoff["gpu_lease_verified"])

    def test_vst_timebase_remains_unverified(self) -> None:
        handoff = self._prepare(self._items(vst=True))
        self.assertEqual(handoff["v3_core_spec"]["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(handoff["v3_core_spec"]["frame_rows_sealed"][0]["qpc_raw_tick"], 100)
        self.assertFalse(handoff["v3_core_spec"]["world_pose_authority"])

    def test_old_t0_or_motion_input_rejects_before_admission_read(self) -> None:
        items = self._items()
        for key in ("t0_input_freeze", "canonical_motion", "accepted_template"):
            with self.subTest(key=key):
                packet = copy.deepcopy(items[0])
                packet["inputs"][key] = "forbidden"
                with self.assertRaisesRegex(ColdStartInputError, "OLD_MOTION_OR_T0_INPUT"):
                    self._prepare((packet, *items[1:]))

    def test_old_t0_read_set_and_derived_ref_reject(self) -> None:
        items = self._items()
        packet = copy.deepcopy(items[0])
        packet["read_set"].append(
            "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/FRAME_TIME_MAP.json")
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_READ_SET"):
            self._prepare((packet, *items[1:]))
        packet = copy.deepcopy(items[0])
        packet["inputs"]["extra_ref"] = {"path": "/old/preprocess/all_data/hand_motion.npz"}
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_REFERENCE"):
            self._prepare((packet, *items[1:]))
        packet = copy.deepcopy(items[0])
        packet["commands"] = ["chaoyang run old --input _run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930"]
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_READ_SET"):
            self._prepare((packet, *items[1:]))

    def test_v3_core_code_sha_must_be_pinned_and_current(self) -> None:
        items = self._items()
        packet = copy.deepcopy(items[0])
        packet["runtime_code"] = []
        with self.assertRaisesRegex(ColdStartInputError, "V3_MODEL_CORE_SHA_PIN_REQUIRED"):
            self._prepare((packet, *items[1:]))
        packet = copy.deepcopy(items[0])
        packet["runtime_code"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ColdStartInputError, "V3_MODEL_CORE_SHA_DRIFT"):
            self._prepare((packet, *items[1:]))
        packet = copy.deepcopy(items[0])
        packet["runtime_code"].pop()
        with self.assertRaisesRegex(ColdStartInputError, "HANDOFF_CODE_SHA_PIN_REQUIRED"):
            self._prepare((packet, *items[1:]))

    def test_source_sha_and_old_derived_video_reject(self) -> None:
        items = self._items()
        source = self.fixture.data / "tracker/play_cards_0915_009/rgb.mp4"
        source.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ColdStartInputError, "SOURCE_SHA_DRIFT"):
            self._prepare(items)
        self.fixture.doCleanups()
        self.fixture = HaWoRDescriptorAdapterCandidateTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_SOURCE_FORBIDDEN"):
            self._prepare(self._items(old_derived=True))

    def test_running_producer_partial_and_cross_session_reject(self) -> None:
        items = self._items()
        state_path = self.fixture.repo / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["tasks"][0]["status"] = "RUNNING"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(ColdStartInputError, "UPSTREAM_NOT_TERMINAL"):
            self._prepare(items)
        state["tasks"][0]["status"] = "PASSED"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        packet = copy.deepcopy(items[0])
        packet["inputs"]["source_session"] = "play_cards_0915_010"
        with self.assertRaisesRegex(ColdStartInputError, "CONSUMER_SOURCE_SESSION_DRIFT"):
            self._prepare((packet, *items[1:]))


if __name__ == "__main__":
    unittest.main()
