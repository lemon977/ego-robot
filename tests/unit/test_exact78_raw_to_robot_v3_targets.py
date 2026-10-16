"""Model-free six-target T0 identity and negative-source tests for v3."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, inspect_session
from chaoyang.ops import run_exact78_raw_to_robot_v2 as prior
from chaoyang.ops import run_exact78_raw_to_robot_v3 as candidate


CHIPS013 = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips/get_potato_chips_0901_013")
CARDS001 = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/001")
SIX_SESSIONS = {
    "get_potato_chips_0901_013": CHIPS013,
    "get_potato_chips_0902_103": Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/potato_chips/get_potato_chips_0902_103"),
    "play_cards_0901_001": Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards/play_cards_0901_001"),
    "play_cards_0902_042": Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_042"),
    "play_cards_0915_001": CARDS001,
    "play_cards_0915_002": Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/002"),
}


class Exact78TwoTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_0901 = inspect_session(CHIPS013)
        cls.source_0915 = inspect_session(CARDS001)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="exact78_v3_targets_test_",
                                                dir=PROJECT_ROOT / "_run/current")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _validate(self, source: dict, frame_map: Path = prior.T0_MAP) -> dict:
        return candidate._validate_t0_target(source, freeze=prior.T0_FREEZE,
                                             frame_map=frame_map)

    def test_frozen_0901_native_clip_identity(self) -> None:
        prior._assert_t0_sha()
        target = self._validate(self.source_0901)
        self.assertEqual(target["frame_count"], 870)
        self.assertEqual(target["time_mode"], "TRACKER_CLIP_METADATA_ONLY")
        self.assertEqual(target["physical_eye"], "tracker_session_left_RGB")

    def test_frozen_0915_raw_physical_left_and_unverified_clock(self) -> None:
        prior._assert_t0_sha()
        target = self._validate(self.source_0915)
        self.assertEqual(target["frame_count"], 150)
        self.assertEqual(target["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(target["physical_eye"], "left_sourceIndex_1")

    def test_same_static_rule_pre_admits_all_six_t0_captures(self) -> None:
        self.assertEqual(set(candidate.TARGETS), set(SIX_SESSIONS))
        expected_roles = {"get_potato_chips_0901_013": "development",
                          "get_potato_chips_0902_103": "development",
                          "play_cards_0901_001": "holdout",
                          "play_cards_0902_042": "development",
                          "play_cards_0915_001": "development",
                          "play_cards_0915_002": "holdout"}
        for session_id, path in SIX_SESSIONS.items():
            with self.subTest(session_id=session_id):
                source = inspect_session(path)
                target = self._validate(source)
                self.assertEqual(target["session_id"], session_id)
                self.assertEqual(target["source_role"], expected_roles[session_id])
                self.assertEqual(target["frame_count"], candidate.TARGETS[session_id]["frames"])

    def test_holdout_role_cannot_be_relabelled_as_development(self) -> None:
        source = inspect_session(SIX_SESSIONS["play_cards_0915_002"])
        frozen = json.loads(prior.T0_FREEZE.read_text(encoding="utf-8"))
        frozen["selected"]["play_cards_0915_002"]["role"] = "development"
        changed = self.root / "wrong_holdout_role.json"
        changed.write_text(json.dumps(frozen), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "T0_TARGET_IDENTITY_OR_TIME_MODE_DRIFT"):
            candidate._validate_t0_target(source, freeze=changed, frame_map=prior.T0_MAP)

    def test_0901_historical_or_replaced_video_ref_rejected(self) -> None:
        source = copy.deepcopy(self.source_0901)
        source["files"]["video"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "T0_RAW_VIDEO_SHA_DRIFT"):
            self._validate(source)

    def test_0915_timebase_promotion_without_evidence_rejected(self) -> None:
        frame_map = json.loads(prior.T0_MAP.read_text(encoding="utf-8"))
        frame_map["sessions"]["play_cards_0915_001"]["qpc_frequency"] = 1_000_000_000
        changed = self.root / "untrusted_frame_map.json"
        changed.write_text(json.dumps(frame_map), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "0915_TIMEBASE_OR_PHYSICAL_EYE_CLAIM_DRIFT"):
            self._validate(self.source_0915, changed)

    def test_0915_right_eye_spoof_rejected(self) -> None:
        source = copy.deepcopy(self.source_0915)
        source["physical_eye"] = "right_sourceIndex_0"
        with self.assertRaisesRegex(ValueError, "T0_TARGET_IDENTITY_OR_TIME_MODE_DRIFT"):
            self._validate(source)

    def test_0915_t0_frame_sidecar_index_gap_rejected(self) -> None:
        frame_map = json.loads(prior.T0_MAP.read_text(encoding="utf-8"))
        frame_map["sessions"]["play_cards_0915_001"]["frame_rows"][31]["decoded_display_index"] = 30
        changed = self.root / "wrong_frame_map.json"
        changed.write_text(json.dumps(frame_map), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "T0_FRAME_INDEX_MAP_DRIFT"):
            self._validate(self.source_0915, changed)

    def test_unlisted_session_rejected_before_raw_read(self) -> None:
        source = copy.deepcopy(self.source_0901)
        source["session_id"] = "unlisted_new_session"
        with self.assertRaisesRegex(ValueError, "T0_FROZEN_TARGET_NOT_AUTHORIZED"):
            self._validate(source)

    def test_t0_map_sha_drift_rejected(self) -> None:
        changed = self.root / "changed_t0_map.json"
        changed.write_text(prior.T0_MAP.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with mock.patch.object(prior, "T0_MAP", changed):
            with self.assertRaisesRegex(ValueError, "T0_FRAME_TIME_MAP_SHA_DRIFT"):
                prior._assert_t0_sha()

    def test_fresh_hawor_cannot_upgrade_0915_time_or_reorder_frames(self) -> None:
        target = {"session_id": "play_cards_0915_001", "frame_count": 2,
                  "physical_eye": "left_sourceIndex_1",
                  "raw_video": self.source_0915["files"]["raw/vst.h264"]}
        t0_rows = [{"crop_xywh": [2048, 0, 2048, 1536], "qpc_raw_tick": 100 + i,
                    "capture_timestamp_raw": 200 + i, "pts": None,
                    "best_effort_timestamp": None,
                    "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED"}
                   for i in range(2)]
        rows = [{**row, "decoded_display_index": i, "model_input_index": i,
                 "output_index": i, "sbs_physical_eye": "left_sourceIndex_1",
                 "time_mode": "TIMEBASE_UNVERIFIED"}
                for i, row in enumerate(t0_rows)]
        provenance = {"schema_version": "exact78-hawor-frame-provenance-v3",
                      "session_id": target["session_id"],
                      "physical_eye": "left_sourceIndex_1",
                      "time_mode": "TIMEBASE_UNVERIFIED", "frame_rows": rows}
        audit = {"schema_version": "exact78-hawor-input-audit-v3",
                 "session_id": target["session_id"], "source_kind": "vst0915",
                 "raw_source_ref": target["raw_video"], "world_pose_emitted": False}
        candidate._validate_hawor_stage_claims(provenance, audit, t0_rows,
                                               target, "vst0915")
        provenance["frame_rows"][1]["qpc_raw_tick"] = 100
        with self.assertRaisesRegex(ValueError, "HAWOR_TO_T0_FRAME_PROVENANCE_DRIFT"):
            candidate._validate_hawor_stage_claims(provenance, audit, t0_rows,
                                                   target, "vst0915")
        provenance["frame_rows"][1]["qpc_raw_tick"] = 101
        provenance["time_mode"] = "CAPTURE_SYNC_VERIFIED"
        with self.assertRaisesRegex(ValueError, "HAWOR_SOURCE_FRAME_TIME_CLAIM_DRIFT"):
            candidate._validate_hawor_stage_claims(provenance, audit, t0_rows,
                                                   target, "vst0915")

    def test_gate_precedes_candidate_and_model_import(self) -> None:
        with mock.patch.object(candidate, "authorize_exact78_operation",
                               side_effect=SystemExit("denied")) as gate:
            with mock.patch.object(candidate, "run_candidate") as runner:
                with mock.patch("sys.argv", [candidate.OPERATION, "--session", "unused"]):
                    with self.assertRaisesRegex(SystemExit, "denied"):
                        candidate.main()
                gate.assert_called_once()
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
