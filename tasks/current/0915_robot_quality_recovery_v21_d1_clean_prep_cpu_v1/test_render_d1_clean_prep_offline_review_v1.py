#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import render_d1_clean_prep_offline_review_v1 as review


def save(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return review.file_ref(path)


class ReviewTests(unittest.TestCase):
    def build(self, root: Path, mutate_candidate: bool = False) -> tuple[Path, dict[str, int], tuple[int, int]]:
        prepared = root / "prepared"
        prepared.mkdir()
        counts = {"play_cards_0915_044": 2, "get_potato_chips_0915_097": 2}
        shape = (60, 80)
        summaries = []
        for session_id, count in counts.items():
            task = review.EXPECTED_TASKS[session_id]
            frames = []
            totals = {"m_remove_pixels": 0, "m_write_pixels": 0, "m_flow_pixels": 0,
                      "unknown_pixels": 0, "protected_pixels": 0}
            for frame_id in range(count):
                raw = np.zeros((*shape, 3), dtype=np.uint8)
                raw[..., 0] = 25 + frame_id * 30
                raw[..., 1] = np.arange(shape[1], dtype=np.uint8)
                candidate = raw.copy()
                m_remove = np.zeros(shape, dtype=bool)
                m_write = np.zeros(shape, dtype=bool)
                m_flow = np.zeros(shape, dtype=bool)
                unknown = np.zeros(shape, dtype=bool)
                source = np.zeros(shape, dtype=np.uint8)
                if task == "playing_cards":
                    source[20:35, 30:55] = review.SOURCE_POKER_UNKNOWN
                    statuses = {"left": "UNKNOWN_L", "right": "UNKNOWN_R"}
                else:
                    m_remove[18:38, 20:45] = True
                    m_write[:] = m_remove
                    unknown[:] = m_write
                    m_flow[15:41, 17:48] = True
                    source[unknown] = review.SOURCE_UNKNOWN
                    statuses = {"left": "UNKNOWN_L", "right": "ADMITTED_B1_PER_SIDE_HAND_MASK"}
                if mutate_candidate and session_id.startswith("play_cards") and frame_id == 1:
                    candidate[0, 0, 0] ^= 1
                frame_dir = prepared / session_id / f"{frame_id:06d}"
                raw_ref = save(frame_dir / "raw.png", raw)
                candidate_ref = save(frame_dir / "candidate.png", candidate)
                m_remove_ref = save(frame_dir / "remove.png", m_remove.astype(np.uint8) * 255)
                m_flow_ref = save(frame_dir / "flow.png", m_flow.astype(np.uint8) * 255)
                m_write_ref = save(frame_dir / "write.png", m_write.astype(np.uint8) * 255)
                unknown_ref = save(frame_dir / "unknown.png", unknown.astype(np.uint8) * 255)
                source_ref = save(frame_dir / "source.png", source)
                protected = np.isin(source, list(review.OBJECT_SOURCE_CODES))
                totals["m_remove_pixels"] += int(m_remove.sum())
                totals["m_write_pixels"] += int(m_write.sum())
                totals["m_flow_pixels"] += int(m_flow.sum())
                totals["unknown_pixels"] += int(unknown.sum())
                totals["protected_pixels"] += int(protected.sum())
                frames.append({
                    "frame_id": frame_id,
                    "raw_rgb": raw_ref,
                    "candidate_rgb": candidate_ref,
                    "M_remove": m_remove_ref,
                    "M_flow": m_flow_ref,
                    "M_write": m_write_ref,
                    "UNKNOWN": unknown_ref,
                    "source_map": source_ref,
                    "raw_decoded_sha256": review.decoded_sha256(raw),
                    "candidate_decoded_sha256": review.decoded_sha256(candidate),
                    "hand_side_status": statuses,
                    "m_remove_pixels": int(m_remove.sum()),
                    "m_flow_pixels": int(m_flow.sum()),
                    "m_write_pixels": int(m_write.sum()),
                    "unknown_pixels": int(unknown.sum()),
                    "protected_pixels": int(protected.sum()),
                    "changed_pixels": int(np.any(raw != candidate, axis=2).sum()),
                })
            manifest = {
                "schema_version": review.FRAME_SCHEMA,
                "session_id": session_id,
                "task": task,
                "frame_count": count,
                "unknown_hand_sides": ["left", "right"] if task == "playing_cards" else ["left"],
                "poker_identity": "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY" if task == "playing_cards" else None,
                "chips_instance_mode": "THREE_SEPARATE_NO_UNION" if task == "potato_chips" else None,
                "frames": frames,
            }
            manifest_path = prepared / session_id / "FRAME_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            session_result_path = prepared / session_id / "RESULT.json"
            session_result_path.write_text(json.dumps({
                "session_id": session_id,
                "task": task,
                "clean_terminal": False,
                "fresh_inpainting_executed": False,
                "materialized_changed_pixels": 0,
                "totals": totals,
            }), encoding="utf-8")
            summaries.append({
                "session_id": session_id,
                "task": task,
                "result": review.file_ref(session_result_path),
                "frame_manifest": review.file_ref(manifest_path),
                "totals": totals,
            })
        result = {
            "schema_version": review.PRODUCER_SCHEMA,
            "execution_mode": "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU",
            "gpu_used": False,
            "model_execution_performed": False,
            "clean_terminal": False,
            "training_eligible": False,
            "sessions": summaries,
        }
        (prepared / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
        return prepared, counts, shape

    def test_complete_render_and_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, counts, shape = self.build(root)
            receipt = review.render_all(prepared, root / "visuals", counts, shape)
            self.assertEqual(receipt["status"], "PASSED_FULL_DECODE_OFFLINE_REVIEW")
            self.assertEqual([row["decoded_frames"] for row in receipt["sessions"]], [2, 2])
            self.assertTrue((root / "visuals" / "D1_CLEAN_PREP_VISUAL_REVIEW_RECEIPT.json").is_file())

    def test_changed_candidate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, counts, shape = self.build(root, mutate_candidate=True)
            with self.assertRaisesRegex(review.ReviewError, "candidate is not decoded-byte-identical"):
                review.render_all(prepared, root / "visuals", counts, shape)
            self.assertFalse((root / "visuals").exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, counts, shape = self.build(root)
            (root / "visuals").mkdir()
            with self.assertRaisesRegex(review.ReviewError, "fresh output root"):
                review.render_all(prepared, root / "visuals", counts, shape)


if __name__ == "__main__":
    unittest.main()
