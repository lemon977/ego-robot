#!/usr/bin/env python3
"""CPU-only fixtures for the V2.1 D1_CLEAN_PREP preparation contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "d1_clean_prep_cpu", HERE / "run_d1_clean_prep_cpu_orchestrator.py"
)
assert SPEC and SPEC.loader
d1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d1)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "d1_clean_prep_validator", HERE / "validate_d1_clean_prep_prepared_output.py"
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8), mode="RGB").save(path)


def write_mask(path: Path, points: tuple[tuple[int, int], ...] = ()) -> None:
    value = np.zeros((960, 1280), dtype=np.uint8)
    for y, x in points:
        value[y, x] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="L").save(path)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frame_count = 2
        authority_payloads = {
            "packages/B1R/RESULT.json": {"status": "COMPLETED_WITH_QUALITY_REJECTION"},
            "packages/B1R/BATCH_RESULT.json": {
                "status": "COMPLETED_WITH_QUALITY_REJECTION",
                "capability": "HAND_ONLY",
                "session_id": "play_cards_0915_044",
            },
            "packages/B1R/B1R_HAND_ADMISSION.json": {
                "status": "COMPLETED_WITH_QUALITY_REJECTION",
                "sides": [
                    {"role": "left_hand", "b1r_status": "REJECTED_DIRECT_OBSERVED_ADMISSION", "consumer_allowed": False},
                    {"role": "right_hand", "b1r_status": "REJECTED_TRACKER_DIRECTION_INCOMPLETE", "consumer_allowed": False},
                ],
            },
            "packages/B1/RESULT.json": {"status": "FAILED_RUNTIME_FINAL"},
            "packages/B1/BATCH_RESULT.json": {"status": "FAILED_RUNTIME_FINAL"},
        }
        self.authority_paths = {}
        for relative, payload in authority_payloads.items():
            path = root / "upstream" / relative
            write_json(path, payload)
            self.authority_paths[relative] = path
        d1.PINNED_UPSTREAM_SHA256 = {
            relative: d1.file_ref(path)["sha256"] for relative, path in self.authority_paths.items()
        }
        self.terminal = self.authority_paths["packages/B1R/RESULT.json"]
        self.session_files: dict[str, dict] = {}
        for session_id, task in d1.TASK_BY_SESSION.items():
            raw_frames = []
            for frame_id in range(self.frame_count):
                rgb = np.full((960, 1280, 3), 17 + frame_id, dtype=np.uint8)
                path = root / session_id / "raw" / f"{frame_id:06d}.png"
                write_rgb(path, rgb)
                raw_frames.append({"frame_id": frame_id, "rgb": d1.file_ref(path)})
            raw_manifest = root / session_id / "RAW_MANIFEST.json"
            write_json(raw_manifest, {
                "schema_version": d1.RAW_SCHEMA,
                "session_id": session_id,
                "task": task,
                "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "image_size": [1280, 960],
                "lossless_frames": True,
                "frame_count": self.frame_count,
                "frames": raw_frames,
            })

            hand_paths = {}
            for side in d1.SIDES:
                frame_rows = []
                for frame_id in range(self.frame_count):
                    points = ()
                    if task == "playing_cards" and side == "left":
                        points = ((0, 0), (1, 1))
                    elif task == "playing_cards" and side == "right":
                        points = ((0, 1),)
                    elif task == "potato_chips" and side == "right":
                        points = ((2, 2), (2, 3))
                    path = root / session_id / "hand" / side / f"{frame_id:06d}.png"
                    write_mask(path, points)
                    frame_rows.append({
                        "frame_id": frame_id,
                        "observed": bool(points),
                        "mask": d1.file_ref(path),
                    })
                manifest_path = root / session_id / f"HAND_{side.upper()}.json"
                write_json(manifest_path, {
                    "schema_version": d1.HAND_SCHEMA,
                    "producer_package": "B1",
                    "session_id": session_id,
                    "side": side,
                    "consumer_status": d1.ADMITTED_SIDE,
                    "frame_count": self.frame_count,
                    "whole_session_hand_terminal_required": False,
                    "frames": frame_rows,
                })
                hand_paths[side] = manifest_path

            if task == "playing_cards":
                instance_ids = ["playing_card_00"]
                identity_authority = "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY"
            else:
                instance_ids = ["potato_chip_00", "potato_chip_01", "potato_chip_02"]
                identity_authority = "THREE_SEPARATE_PHYSICAL_INSTANCE_SLOTS"
            object_frames = []
            for frame_id in range(self.frame_count):
                instances = []
                for instance_id in instance_ids:
                    points = ()
                    if task == "playing_cards" and instance_id == "playing_card_00":
                        points = ((1, 1),)
                    elif task == "potato_chips" and instance_id == "potato_chip_00":
                        points = ((2, 2),)
                    elif task == "potato_chips" and instance_id == "potato_chip_01":
                        points = ((3, 0),)
                    path = root / session_id / "object" / instance_id / f"{frame_id:06d}.png"
                    write_mask(path, points)
                    instances.append({
                        "instance_id": instance_id,
                        "status": "ADMITTED_VISIBLE_PROTECTION" if points else "UNKNOWN_NOT_VISIBLE",
                        "observed": bool(points),
                        "mask": d1.file_ref(path),
                    })
                object_frames.append({"frame_id": frame_id, "instances": instances})
            object_manifest = root / session_id / "OBJECT_MANIFEST.json"
            write_json(object_manifest, {
                "schema_version": d1.OBJECT_SCHEMA,
                "producer_package": "B1",
                "session_id": session_id,
                "task": task,
                "frame_count": self.frame_count,
                "depends_on_hand_terminal": False,
                "union_mask_created": False,
                "visible_surface_only": True,
                "hidden_shape_inferred": False,
                "identity_authority": identity_authority,
                "instance_ids": instance_ids,
                "frames": object_frames,
            })
            self.session_files[session_id] = {
                "task": task,
                "raw": raw_manifest,
                "hands": hand_paths,
                "object": object_manifest,
            }
        self.composite = root / "B1_B1R_D1_COMPOSITE.json"
        self.handoff = root / "B1_D1_HANDOFF.json"
        self.input = root / "D1_INPUT.json"
        self.rebind()

    def rebind(self, handoff_mutator=None, input_mutator=None) -> None:
        sessions = []
        for session_id in d1.FIXED_SESSIONS:
            files = self.session_files[session_id]
            hand_sides = {
                side: {
                    "status": d1.ADMITTED_SIDE,
                    "mask_manifest": d1.file_ref(files["hands"][side]),
                }
                for side in d1.SIDES
            }
            if session_id == "play_cards_0915_044":
                hand_sides = {
                    "left": {
                        "status": "UNKNOWN_B1R_REJECTED_DIRECT_OBSERVED_ADMISSION",
                        "mask_manifest": None,
                    },
                    "right": {
                        "status": "UNKNOWN_B1R_REJECTED_TRACKER_DIRECTION_INCOMPLETE",
                        "mask_manifest": None,
                    },
                }
            if session_id == "get_potato_chips_0915_097":
                hand_sides["left"] = {
                    "status": "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR",
                    "mask_manifest": None,
                }
            sessions.append({
                "session_id": session_id,
                "task": files["task"],
                "frame_count": self.frame_count,
                "hand_sides": hand_sides,
                "task_object": {
                    "depends_on_hand_terminal": False,
                    "manifest": d1.file_ref(files["object"]),
                },
            })
        handoff = {
            "schema_version": d1.B1_HANDOFF_SCHEMA,
            "terminal": True,
            "status": "COMPLETED_WITH_QUALITY_REJECTION",
            "authority_mode": d1.COMPOSITE_AUTHORITY_MODE,
            "aggregate_gate_relaxed": False,
            "capability_gate_relaxed": False,
            "terminal_receipt": d1.file_ref(self.terminal),
            "sessions": sessions,
        }
        if handoff_mutator:
            handoff_mutator(handoff)
        closure_sessions = []
        for row in handoff["sessions"]:
            closure_sessions.append({
                "session_id": row["session_id"],
                "hand": row["hand_sides"],
                "task_object": {"manifest": row["task_object"]["manifest"]},
            })
        composite = {
            "schema_version": d1.COMPOSITE_SCHEMA,
            "status": "COMPOSITE_CLOSED_WITH_QUALITY_REJECTION",
            "terminal": True,
            "fixed_sessions": list(d1.FIXED_SESSIONS),
            "aggregate_gate_relaxed": False,
            "capability_gate_relaxed": False,
            "b1r_hand_capability_terminal": {
                "status": "COMPLETED_WITH_QUALITY_REJECTION",
                "result": d1.file_ref(self.authority_paths["packages/B1R/RESULT.json"]),
            },
            "original_b1": {
                "status": "FAILED_RUNTIME_FINAL",
                "original_b1_result_mutated": False,
                "used_as_aggregate_terminal": False,
                "result": d1.file_ref(self.authority_paths["packages/B1/RESULT.json"]),
            },
            "pinned_upstream_artifacts": {
                relative: d1.file_ref(path) for relative, path in self.authority_paths.items()
            },
            "sessions": closure_sessions,
        }
        write_json(self.composite, composite)
        handoff["composite_closure"] = d1.file_ref(self.composite)
        write_json(self.handoff, handoff)
        manifest = {
            "schema_version": d1.INPUT_SCHEMA,
            "execution_mode": "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU",
            "fixed_sessions": list(d1.FIXED_SESSIONS),
            "b1_handoff": d1.file_ref(self.handoff),
            "sessions": [
                {
                    "session_id": session_id,
                    "task": self.session_files[session_id]["task"],
                    "frame_count": self.frame_count,
                    "raw_frame_manifest": d1.file_ref(self.session_files[session_id]["raw"]),
                }
                for session_id in d1.FIXED_SESSIONS
            ],
        }
        if input_mutator:
            input_mutator(manifest)
        write_json(self.input, manifest)


class D1CleanPrepCPUFixtures(unittest.TestCase):
    def test_prepare_outputs_strict_domains_unknown_and_source_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            output = Path(directory) / "output"
            result = d1.prepare(fixture.input, output)
            self.assertEqual(result["status"], "BLOCKED_PREREQ_FRESH_INPAINTING_OR_UNKNOWN")
            self.assertFalse(result["clean_terminal"])
            self.assertFalse(result["model_execution_performed"])
            chips = json.loads((output / "sessions/get_potato_chips_0915_097/FRAME_MANIFEST.json").read_text())
            row = chips["frames"][0]
            raw = np.asarray(Image.open(row["raw_rgb"]["path"]))
            candidate = np.asarray(Image.open(row["candidate_rgb"]["path"]))
            m_remove = np.asarray(Image.open(row["M_remove"]["path"])) > 0
            m_flow = np.asarray(Image.open(row["M_flow"]["path"])) > 0
            m_write = np.asarray(Image.open(row["M_write"]["path"])) > 0
            unknown = np.asarray(Image.open(row["UNKNOWN"]["path"])) > 0
            source = np.asarray(Image.open(row["source_map"]["path"]))
            self.assertTrue(np.array_equal(raw, candidate))
            self.assertEqual(int(m_remove.sum()), 1)
            self.assertEqual(int(m_write.sum()), 1)
            self.assertGreater(int(m_flow.sum()), int(m_write.sum()))
            self.assertTrue(np.all(m_flow[m_write]))
            self.assertTrue(np.array_equal(unknown, m_write))
            self.assertEqual(int(source[2, 2]), int(d1.SOURCE_CHIP_00_PROTECTED_RAW))
            self.assertTrue(np.all(source[unknown] == d1.SOURCE_UNKNOWN_UNWRITTEN))
            self.assertEqual(row["changed_outside_m_write_pixels"], 0)

    def test_object_protection_runs_when_one_hand_side_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            output = Path(directory) / "output"
            d1.prepare(fixture.input, output)
            result = json.loads((output / "sessions/get_potato_chips_0915_097/RESULT.json").read_text())
            self.assertEqual(result["unknown_hand_sides"], ["left"])
            self.assertFalse(result["object_lane_depends_on_hand_terminal"])
            self.assertGreater(result["totals"]["protected_pixels"], 0)
            self.assertFalse(result["chips_instances_unioned"])

    def test_failed_runtime_final_b1_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            write_json(fixture.terminal, {"status": "FAILED_RUNTIME_FINAL"})
            fixture.rebind(lambda handoff: handoff.update({"status": "FAILED_RUNTIME_FINAL"}))
            with self.assertRaisesRegex(d1.ContractError, "usable terminal"):
                d1.validate_contract(fixture.input)

    def test_unknown_side_cannot_smuggle_a_mask_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))

            def mutate(handoff):
                chips = handoff["sessions"][1]
                chips["hand_sides"]["left"]["mask_manifest"] = d1.file_ref(
                    fixture.session_files["get_potato_chips_0915_097"]["hands"]["left"]
                )

            fixture.rebind(mutate)
            with self.assertRaisesRegex(d1.ContractError, "unadmitted masks"):
                d1.validate_contract(fixture.input)

    def test_chips_requires_three_instances_and_no_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.session_files["get_potato_chips_0915_097"]["object"]
            payload = json.loads(path.read_text())
            payload["union_mask_created"] = True
            write_json(path, payload)
            fixture.rebind()
            with self.assertRaisesRegex(d1.ContractError, "independence/visibility contract drift"):
                d1.validate_contract(fixture.input)

    def test_overlapping_chips_instances_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            path = fixture.session_files["get_potato_chips_0915_097"]["object"]
            payload = json.loads(path.read_text())
            for frame in payload["frames"]:
                chip0 = frame["instances"][0]["mask"]
                frame["instances"][1].update({
                    "status": "ADMITTED_VISIBLE_PROTECTION",
                    "observed": True,
                    "mask": chip0,
                })
            write_json(path, payload)
            fixture.rebind()
            with self.assertRaisesRegex(d1.ContractError, "overlap/union"):
                d1.prepare(fixture.input, Path(directory) / "output")

    def test_future_donor_or_old_clean_authority_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.rebind(input_mutator=lambda manifest: manifest.update({"future_donor": {"frame": 1}}))
            with self.assertRaisesRegex(d1.ContractError, "forbidden authority"):
                d1.validate_contract(fixture.input)

    def test_byte_change_outside_m_write_fails_audit(self) -> None:
        raw = np.zeros((3, 4, 3), dtype=np.uint8)
        candidate = raw.copy()
        candidate[0, 0] = 1
        empty = np.zeros((3, 4), dtype=bool)
        with self.assertRaisesRegex(d1.ContractError, "outside M_write"):
            d1.audit_domains(raw, candidate, empty, empty, empty, empty, empty)

    def test_poker_identity_without_independent_evidence_stays_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            output = Path(directory) / "output"
            d1.prepare(fixture.input, output)
            result = json.loads((output / "sessions/play_cards_0915_044/RESULT.json").read_text())
            self.assertEqual(result["poker_identity"], "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY")
            self.assertFalse(result["hidden_ground_truth_consumed"])

    def test_independent_validator_recomputes_zero_write_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            output = Path(directory) / "output"
            d1.prepare(fixture.input, output)
            receipt = validator.validate(output)
            self.assertEqual(receipt["status"], "PASSED_D1_CLEAN_PREP_CPU_PREFILL_DEEP_VALIDATION")
            self.assertEqual(receipt["frames_validated"], 4)
            self.assertGreater(receipt["aggregate"]["m_write_pixels"], 0)
            self.assertEqual(receipt["aggregate"]["changed_outside_m_write_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
