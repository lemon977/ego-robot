"""Sealed same-session admission for a future, unregistered HaWoR runner."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chaoyang.cli import exact78_first_submit_bridge_candidate as bridge
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    ColdStartInputError,
    PROJECT_ROOT,
    file_ref,
)
from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import build_spec_from_sealed


class HaWoRDescriptorAdapterCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-hawor-adapter-test-", dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.data = self.root / "source_data"
        self.repo.mkdir()
        self.data.mkdir()

    def _write(self, path: Path, value: str | dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value,
                        encoding="utf-8")

    def _tracker(self, *, old_derived: bool = False) -> Path:
        name = "play_cards_0915_009"
        session = self.data / "tracker" / name
        video_role = "preprocess/all_data.mp4" if old_derived else "rgb.mp4"
        self._write(session / video_role, "synthetic RGB")
        self._write(session / "source_stereo/stereo.mp4", "synthetic stereo")
        self._write(session / "camera_params.json", "{}")
        self._write(session / f"slam_trajectory_{name}.jsonl", "{}\n")
        self._write(session / "clip_manifest.json", {
            "clip_name": name,
            "files": {"video": video_role, "source_stereo_video": "source_stereo/stereo.mp4",
                      "camera_params": "camera_params.json",
                      "slam": f"slam_trajectory_{name}.jsonl"},
            "video": {"frame_count": 2, "fps": 25, "width": 1280, "height": 960},
            "video_rectification": {"eye": "left", "rectified": True,
                                    "resolution": [1280, 960],
                                    "intrinsics": [[800, 0, 640], [0, 800, 480], [0, 0, 1]]},
        })
        return session

    def _vst(self) -> Path:
        session = self.data / "cards_unit_0915" / "009"
        self._write(session / "raw/vst.h264", "synthetic SBS")
        self._write(session / "raw/vst.qpc.ts.jsonl", "100\n200\n")
        self._write(session / "raw/vst.ts.jsonl", "1000\n2000\n")
        self._write(session / "raw/camera_params.meta.json", "{}")
        self._write(session / "raw/camera_params.json", {
            "width": 640, "height": 480, "outputWidth": 1280, "outputHeight": 480,
            "fps": 30, "left": {"sourceIndex": 1, "fx": 400, "fy": 400,
                                  "cx": 320, "cy": 240}, "right": {"sourceIndex": 0},
        })
        return session

    def _sealed(self, session: Path, *, vst: bool = False) -> tuple[dict, ...]:
        dimensions = (1280, 480) if vst else (1280, 960)
        frames = [{"width": dimensions[0], "height": dimensions[1], "pts": index,
                   "best_effort_timestamp": index, "pict_type": "P"} for index in range(2)]
        with patch.object(bridge, "_decode_frame_rows", return_value=frames):
            prepared = bridge.plan_single(session, allowed_root=self.data)
        descriptor = prepared["descriptor"]
        task_id = prepared["target_draft"]["task_id"]
        admission = self.repo / "_run/current" / task_id / "attempts/attempt_0001/admission"
        descriptor_path = admission / "DESCRIPTOR.json"
        map_path = admission / "FRAME_TIME_MAP.json"
        seal_path = admission / "SEAL.json"
        mapping = {
            "schema_version": "exact78-first-submit-frame-map-v1",
            **{key: descriptor[key] for key in (
                "session_id", "source_kind", "physical_eye", "crop_xywh", "time_mode",
                "capture_to_decode_alignment", "decoded_frames", "raw_video", "frame_rows")},
        }
        self._write(descriptor_path, descriptor)
        self._write(map_path, mapping)
        descriptor_ref, map_ref = file_ref(descriptor_path), file_ref(map_path)
        seal = {
            "schema_version": "exact78-first-submit-admission-seal-v1",
            "status": "SEALED_CPU_ADMISSION", "task_id": task_id,
            "descriptor": descriptor_ref, "frame_time_map": map_ref,
            "descriptor_sha256": prepared["descriptor_sha256"],
        }
        self._write(seal_path, seal)
        seal_ref = file_ref(seal_path)
        return self._terminal_refs(task_id, descriptor, descriptor_ref, map_ref, seal_ref)

    def _terminal_refs(self, task_id: str, descriptor: dict, descriptor_ref: dict,
                       map_ref: dict, seal_ref: dict) -> tuple[dict, ...]:
        import hashlib

        descriptor_sha = hashlib.sha256(bridge._canonical(descriptor)).hexdigest()
        producer_packet_path = self.repo / "tasks/current" / task_id / "TASK_PACKET.json"
        self._write(producer_packet_path, {
            "task_id": task_id, "execution_mode": "EXACT78_CANDIDATE",
            "run_signature": "producer-signature",
            "inputs": {"descriptor_sha256": descriptor_sha,
                       "submission_input_sha256": descriptor["submission_input_sha256"],
                       "source_files": descriptor["source_files"]},
        })
        producer_packet_ref = file_ref(producer_packet_path)
        attempt = Path(seal_ref["path"]).parent.parent
        result_path = attempt / "RESULT.json"
        self._write(result_path, {
            "schema_version": "exact78-first-submit-cpu-admission-result-v1",
            "task_id": task_id, "status": "PASSED", "run_signature": "producer-signature",
            "source_session": descriptor["session_id"], "source_root": descriptor["session_root"],
            "source_files": descriptor["source_files"],
            "submission_input_sha256": descriptor["submission_input_sha256"],
            "descriptor_sha256": descriptor_sha,
            "admission": {"descriptor": descriptor_ref, "frame_time_map": map_ref,
                          "seal": seal_ref},
        })
        result_ref = file_ref(result_path)
        receipt_path = self.repo / "tasks/receipts" / f"{task_id.upper()}.json"
        self._write(receipt_path, {"task_id": task_id, "status": "PASSED", "result": result_ref})
        receipt_ref = file_ref(receipt_path)
        self._write(self.repo / "docs/governance/LONG_HORIZON_TASK_STATE.json", {
            "tasks": [{"task_id": task_id, "session": descriptor["session_id"],
                       "status": "PASSED", "task_packet": producer_packet_ref,
                       "result": result_ref, "terminal_receipt": receipt_ref}],
            "next_task": {"task_id": "exact78_hawor_consumer"},
        })
        consumer_packet = {
            "task_id": "exact78_hawor_consumer", "execution_mode": "EXACT78_CANDIDATE",
            "run_signature": "consumer-signature",
            "inputs": {"upstream_admission_descriptor": descriptor_ref,
                       "upstream_admission_frame_time_map": map_ref,
                       "upstream_admission_seal": seal_ref,
                       "upstream_admission_result": result_ref,
                       "upstream_admission_receipt": receipt_ref,
                       "upstream_admission_descriptor_sha256": descriptor_sha,
                       "submission_input_sha256": descriptor["submission_input_sha256"],
                       "source_session": descriptor["session_id"],
                       "source_root": descriptor["session_root"]},
        }
        return consumer_packet, descriptor_ref, map_ref, seal_ref, result_ref, receipt_ref

    def _run(self, items: tuple[dict, ...]) -> dict:
        packet, descriptor_ref, map_ref, seal_ref, result_ref, receipt_ref = items
        return build_spec_from_sealed(
            packet=packet, descriptor_ref=descriptor_ref, frame_map_ref=map_ref,
            seal_ref=seal_ref, producer_result_ref=result_ref,
            producer_receipt_ref=receipt_ref, repo_root=self.repo, allowed_root=self.data)

    def _reseal(self, items: tuple[dict, ...], *, descriptor: dict | None = None,
                mapping: dict | None = None) -> tuple[dict, ...]:
        packet, descriptor_ref, map_ref, seal_ref, _, _ = items
        descriptor_path, map_path, seal_path = (Path(x["path"]) for x in
                                                (descriptor_ref, map_ref, seal_ref))
        if descriptor is not None:
            self._write(descriptor_path, descriptor)
        if mapping is not None:
            self._write(map_path, mapping)
        descriptor_ref, map_ref = file_ref(descriptor_path), file_ref(map_path)
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["descriptor"], seal["frame_time_map"] = descriptor_ref, map_ref
        if descriptor is not None:
            import hashlib
            seal["descriptor_sha256"] = hashlib.sha256(bridge._canonical(descriptor)).hexdigest()
        self._write(seal_path, seal)
        seal_ref = file_ref(seal_path)
        if descriptor is None:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        return self._terminal_refs(seal["task_id"], descriptor, descriptor_ref, map_ref, seal_ref)

    def test_tracker_spec_uses_sealed_same_session_inputs(self) -> None:
        spec = self._run(self._sealed(self._tracker()))
        self.assertEqual(spec["source_kind"], "tracker")
        self.assertEqual(spec["frame_count"], 2)
        self.assertEqual(spec["intrinsics"][0][0], 800)
        self.assertEqual(spec["time_mode"], "CLIP_FRAME_INDEX_ONLY")
        self.assertTrue(spec["native_open_audit_required"])
        self.assertNotIn("frame_rows_t0", spec)

    def test_source_sha_drift_rejects(self) -> None:
        session = self._tracker()
        items = self._sealed(session)
        (session / "rgb.mp4").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ColdStartInputError, "SOURCE_SHA_DRIFT"):
            self._run(items)

    def test_eye_ambiguity_rejects_even_when_newly_resealed(self) -> None:
        items = self._sealed(self._tracker())
        descriptor = json.loads(Path(items[1]["path"]).read_text(encoding="utf-8"))
        descriptor["physical_eye"] = "UNKNOWN"
        mapping = json.loads(Path(items[2]["path"]).read_text(encoding="utf-8"))
        mapping["physical_eye"] = "UNKNOWN"
        items = self._reseal(items, descriptor=descriptor, mapping=mapping)
        with self.assertRaisesRegex(ColdStartInputError, "TRACKER_SOURCE_CLAIM_DRIFT"):
            self._run(items)

    def test_frame_map_mismatch_rejects(self) -> None:
        items = self._sealed(self._tracker())
        mapping = json.loads(Path(items[2]["path"]).read_text(encoding="utf-8"))
        mapping["frame_rows"][1]["decoded_display_index"] = 7
        items = self._reseal(items, mapping=mapping)
        with self.assertRaisesRegex(ColdStartInputError, "FRAME_MAP_COUNT_OR_SOURCE_DRIFT"):
            self._run(items)

    def test_old_derived_video_path_rejects(self) -> None:
        items = self._sealed(self._tracker(old_derived=True))
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_SOURCE_FORBIDDEN"):
            self._run(items)

    def test_vst_timebase_remains_unverified(self) -> None:
        items = self._sealed(self._vst(), vst=True)
        spec = self._run(items)
        self.assertEqual(spec["source_kind"], "vst0915")
        self.assertEqual(spec["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(spec["frame_rows_sealed"][0]["qpc_raw_tick"], 100)
        self.assertFalse(spec["world_pose_authority"])
        self.assertFalse(spec["metric_authority"])

    def test_timebase_upgrade_rejects_even_when_newly_resealed(self) -> None:
        items = self._sealed(self._vst(), vst=True)
        descriptor = json.loads(Path(items[1]["path"]).read_text(encoding="utf-8"))
        descriptor["time_mode"] = "QPC_VERIFIED"
        mapping = json.loads(Path(items[2]["path"]).read_text(encoding="utf-8"))
        mapping["time_mode"] = "QPC_VERIFIED"
        items = self._reseal(items, descriptor=descriptor, mapping=mapping)
        with self.assertRaisesRegex(ColdStartInputError, "VST0915_SOURCE_CLAIM_DRIFT"):
            self._run(items)

    def test_packet_binding_or_missing_seal_rejects(self) -> None:
        items = self._sealed(self._tracker())
        packet = copy.deepcopy(items[0])
        packet["inputs"]["upstream_admission_seal"] = None
        with self.assertRaisesRegex(ColdStartInputError, "PACKET_SEALED_INPUT_BINDING_DRIFT"):
            self._run((packet, *items[1:]))
        Path(items[3]["path"]).unlink()
        with self.assertRaisesRegex(ColdStartInputError, "SEALED_REF_MISSING"):
            self._run(items)

    def test_running_producer_partial_is_not_consumed(self) -> None:
        items = self._sealed(self._tracker())
        state_path = self.repo / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["tasks"][0]["status"] = "RUNNING"
        self._write(state_path, state)
        with self.assertRaisesRegex(ColdStartInputError, "UPSTREAM_NOT_TERMINAL_IN_LEDGER"):
            self._run(items)

    def test_cross_session_consumer_binding_rejects(self) -> None:
        items = self._sealed(self._tracker())
        packet = copy.deepcopy(items[0])
        packet["inputs"]["source_session"] = "play_cards_0915_010"
        with self.assertRaisesRegex(ColdStartInputError, "CONSUMER_SOURCE_SESSION_DRIFT"):
            self._run((packet, *items[1:]))


if __name__ == "__main__":
    unittest.main()
