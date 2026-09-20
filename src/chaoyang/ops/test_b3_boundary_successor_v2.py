from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "b3_boundary_v2",
    HERE / "run_b3_sam31_independent_hand_equipment_canary_v2.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PoisonBoundaryStream:
    """Mimic the pinned SAM3 iterator: desired frames, then a bad boundary."""

    def __init__(self, start: int, end_exclusive: int) -> None:
        self.start = start
        self.end_exclusive = end_exclusive
        self.requested = 0
        self.boundary_requested = False
        self.closed = False

    def __iter__(self):
        try:
            for frame in range(self.start, self.end_exclusive):
                self.requested += 1
                yield frame, {"frame": frame}
            self.boundary_requested = True
            raise IndexError("simulated empty SAM3 batched detector boundary")
        finally:
            self.closed = True


class FakeState(dict):
    def __init__(self) -> None:
        super().__init__()
        self.was_cleared = False

    def clear(self) -> None:
        self.was_cleared = True
        super().clear()


class FakeModel:
    def __init__(self, start: int, end_exclusive: int, height: int, width: int) -> None:
        self.start = start
        self.end_exclusive = end_exclusive
        self.height = height
        self.width = width
        self.state = FakeState()
        self.stream: PoisonBoundaryStream | None = None
        self.propagate_kwargs = None

    def init_state(self, **_kwargs):
        return self.state

    def _output(self):
        mask = np.zeros((1, self.height, self.width), dtype=bool)
        mask[0, 2:6, 3:7] = True
        return {
            "out_binary_masks": mask,
            "out_probs": np.asarray([0.9]),
            "out_obj_ids": np.asarray([7]),
        }

    def add_prompt(self, **_kwargs):
        return None, self._output()

    def propagate_in_video(self, **kwargs):
        self.propagate_kwargs = kwargs
        self.stream = PoisonBoundaryStream(self.start, self.end_exclusive)

        def with_outputs():
            for frame, _ in self.stream:
                yield frame, self._output()

        return with_outputs()


class B3BoundarySuccessorV2Test(unittest.TestCase):
    def test_exact_64_frame_prefix_never_requests_poison_boundary(self) -> None:
        raw = PoisonBoundaryStream(0, 64)
        stream = iter(raw)
        rows = list(MODULE.take_exact_frozen_window(stream, start=0, end_exclusive=64))
        self.assertEqual([frame for frame, _ in rows], list(range(64)))
        self.assertEqual(raw.requested, 64)
        self.assertFalse(raw.boundary_requested)
        self.assertTrue(raw.closed)

    def test_exact_nonzero_chunk_never_requests_next_chunk_seed(self) -> None:
        raw = PoisonBoundaryStream(64, 128)
        rows = list(MODULE.take_exact_frozen_window(iter(raw), start=64, end_exclusive=128))
        self.assertEqual(rows[0][0], 64)
        self.assertEqual(rows[-1][0], 127)
        self.assertFalse(raw.boundary_requested)

    def test_premature_stop_fails_closed(self) -> None:
        def short():
            yield 0, {}
            yield 1, {}

        with self.assertRaisesRegex(RuntimeError, "ended before"):
            list(MODULE.take_exact_frozen_window(short(), start=0, end_exclusive=3))

    def test_out_of_order_frame_fails_closed(self) -> None:
        def out_of_order():
            yield 10, {}
            yield 12, {}

        with self.assertRaisesRegex(RuntimeError, "sequential frame order"):
            list(MODULE.take_exact_frozen_window(out_of_order(), start=10, end_exclusive=12))

    def test_process_prompt_window_integrates_exact_boundary(self) -> None:
        model = FakeModel(start=64, end_exclusive=128, height=12, width=16)
        def normalize_without_torch(outputs, _height, _width):
            return (
                np.asarray(outputs["out_binary_masks"], dtype=bool),
                np.asarray(outputs["out_probs"], dtype=float),
                np.asarray(outputs["out_obj_ids"], dtype=np.int64),
            )

        with mock.patch.object(MODULE, "normalize", side_effect=normalize_without_torch):
            stream, ledger = MODULE.process_prompt_window(
                model,
                Path("unused"),
                prompt_spec={
                    "text": "hand",
                    "minimum_area_pixels": 1,
                    "maximum_area_fraction": 0.5,
                    "maximum_components": 2,
                    "maximum_candidates": 1,
                },
                start=64,
                end_exclusive=128,
                height=12,
                width=16,
            )
        self.assertEqual(model.propagate_kwargs["max_frame_num_to_track"], 64)
        self.assertEqual(ledger["frames"][0]["frame_index"], 64)
        self.assertEqual(ledger["frames"][-1]["frame_index"], 127)
        self.assertEqual(stream["7"].shape[0], 64)
        self.assertIsNotNone(model.stream)
        self.assertEqual(model.stream.requested, 64)
        self.assertFalse(model.stream.boundary_requested)
        self.assertTrue(model.stream.closed)
        self.assertTrue(model.state.was_cleared)


if __name__ == "__main__":
    unittest.main()
