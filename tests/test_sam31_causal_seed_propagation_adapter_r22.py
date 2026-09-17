from __future__ import annotations

import numpy as np

from chaoyang.pipeline.sam31_causal_seed_propagation_adapter_r22 import (
    CausalSeedContract,
    collect_causal_forward,
    gate_anatomical_side_nearest,
    gate_offscreen_empty,
    gate_reentry_latency,
    gate_single_instance,
)


def row(frame: int, present: bool, centroid=(2.0, 2.0), components=1, area=4, unexpected=None):
    return {"source_frame": frame, "present": present, "centroid_xy": list(centroid) if present else None, "connected_components": components if present else 0, "area_pixels": area if present else 0, "unexpected_object_ids": unexpected or []}


def test_offscreen_empty_gate() -> None:
    rows = [row(0, True), row(1, False), row(2, False), row(3, True)]
    evidence = gate_offscreen_empty(rows, {0: True, 1: False, 2: False, 3: True}, 1.0)
    assert evidence["pass"] and evidence["empty_fraction"] == 1.0


def test_reentry_latency_gate() -> None:
    rows = [row(0, True), row(1, False), row(2, False), row(3, False), row(4, True)]
    evidence = gate_reentry_latency(rows, {0: True, 1: False, 2: False, 3: True, 4: True}, 1)
    assert evidence["pass"] and evidence["reentries"][1]["latency_frames"] == 1


def test_anatomical_side_nearest_gate() -> None:
    left = [row(0, True, (1, 1)), row(1, True, (2, 1))]
    right = [row(0, True, (9, 1)), row(1, True, (8, 1))]
    evidence = gate_anatomical_side_nearest(left, right, {0: [0, 0], 1: [1, 0]}, {0: [10, 0], 1: [9, 0]}, 1.0)
    assert evidence["pass"] and evidence["checks"] == 4


def test_single_instance_gate() -> None:
    assert gate_single_instance([row(0, True, components=1, area=8)], 8, 1, 2.0)["pass"]
    assert not gate_single_instance([row(0, True, components=2, area=8)], 8, 1, 2.0)["pass"]


def test_collect_forces_forward_and_persists_missing_frames() -> None:
    class Fake:
        request = None
        def handle_stream_request(self, request):
            self.request = request
            mask = np.zeros((1, 6, 8), bool); mask[0, 1:3, 2:4] = True
            yield {"frame_index": 0, "outputs": {"masks": mask, "ids": np.array([7])}}
            yield {"frame_index": 2, "outputs": {"masks": mask, "ids": np.array([7])}}
    def normalize(outputs, height, width):
        return outputs["masks"], np.ones(len(outputs["ids"])), outputs["ids"]
    fake = Fake()
    rows, masks = collect_causal_forward(fake, CausalSeedContract("s", 7, 0, 3, 10, 6, 8), normalize)
    assert fake.request["propagation_direction"] == "forward"
    assert [item["source_frame"] for item in rows] == [10, 11, 12]
    assert rows[1]["reason"] == "NO_STREAM_OUTPUT" and not rows[1]["present"]
    assert sorted(masks) == [10, 11, 12]
