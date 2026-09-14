#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cv2


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "tools/run_poker_task_object_frontier_successor_v2.py"
SPEC = importlib.util.spec_from_file_location("poker_frontier_v2", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PokerFrontierSuccessorV2Test(unittest.TestCase):
    def test_scope_is_exact_frontier_eight(self) -> None:
        self.assertEqual(len(module.SELECTED), 8)
        self.assertTrue(all(value.startswith("play_cards_0902_") for value in module.SELECTED))

    def test_each_session_has_five_early_geometry_rgb_observations(self) -> None:
        dataset = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards")
        for session in sorted(module.SELECTED):
            video = dataset / session / f"CameraRecord_{session}.mp4"
            capture = cv2.VideoCapture(str(video))
            self.assertTrue(capture.isOpened(), session)
            observed = []
            for frame_id in range(31):
                ok, frame = capture.read()
                self.assertTrue(ok, f"{session}:{frame_id}")
                item = module.poker_rightmost_slot_evidence(frame)
                if item is not None:
                    observed.append(item)
            capture.release()
            self.assertGreaterEqual(len(observed), 5, session)
            self.assertTrue(all(item["blue_fraction"] >= 0.25 for item in observed), session)
            self.assertTrue(all(item["area"] > 80 for item in observed), session)

    def test_no_role_mask_dependency(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("raw_role_masks", source)
        self.assertNotIn("hand_tracker_union_masks", source)


if __name__ == "__main__":
    unittest.main()
