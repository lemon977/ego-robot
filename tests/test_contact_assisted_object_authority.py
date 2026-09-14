import numpy as np

from pipeline.contact_assisted_object_authority import (
    evaluate_cross_instance_identity,
    evaluate_instance_rows,
    frame_mask_metrics,
)
from tools.run_contact_assisted_object_instance_authority import propagate_segment


FRAME_GATE = {
    "present_fraction_min": 0.8,
    "longest_missing_run_frames_max": 1,
    "spatial_valid_fraction_min": 0.8,
    "image_edge_contact_pixels_max": 0,
    "connected_components_over_16px_max": 2,
    "largest_component_area_fraction_min": 0.8,
    "adjacent_area_ratio_min": 0.2,
    "adjacent_area_ratio_max": 5.0,
}
IDENTITY_GATE = {
    "contact_reseed_point_to_mask_px_max": 24.0,
}


def square(x0: int) -> np.ndarray:
    value = np.zeros((30, 40), bool)
    value[10:15, x0:x0 + 6] = True
    return value


def test_full_instance_timeline_and_contact_gate_pass():
    rows = []
    for frame in range(5):
        row = frame_mask_metrics(
            square(5 + frame),
            raw_id=101,
            area_fraction_min=0.01,
            area_fraction_max=0.1,
            gate=FRAME_GATE,
        )
        rows.append({"frame_index": frame, **row})
    result = evaluate_instance_rows(
        rows,
        frame_count=5,
        raw_id=101,
        contact_frame=2,
        contact_reseed_distance_px=0.0,
        frame_gate=FRAME_GATE,
        identity_gate=IDENTITY_GATE,
    )
    assert result["pass"] is True


def test_identity_collapse_is_a_gate_only_and_never_edits_masks():
    left = [square(5), square(6)]
    right = [square(25), square(6)]
    result = evaluate_cross_instance_identity(
        {"object_0": left, "object_1": right}, duplicate_iou_max=0.85
    )
    assert result["pass"] is False
    assert result["identity_collapse_frame_count"] == 1
    assert result["masks_modified_or_mutually_excluded"] is False


def test_missing_end_never_counts_as_reappearance():
    rows = []
    for frame in range(5):
        mask = square(5) if frame < 4 else np.zeros((30, 40), bool)
        rows.append({
            "frame_index": frame,
            **frame_mask_metrics(
                mask,
                raw_id=7,
                area_fraction_min=0.01,
                area_fraction_max=0.1,
                gate=FRAME_GATE,
            ),
        })
    result = evaluate_instance_rows(
        rows,
        frame_count=5,
        raw_id=7,
        contact_frame=None,
        contact_reseed_distance_px=None,
        frame_gate=FRAME_GATE,
        identity_gate=IDENTITY_GATE,
    )
    assert result["pass"] is False
    assert result["gates"]["last_frame_present"] is False


def test_propagation_bound_is_inclusive_delta_without_contact_future_leak(tmp_path):
    class FakeModel:
        def __init__(self):
            self.arguments = None

        def propagate_in_video(self, **kwargs):
            self.arguments = kwargs
            start = kwargs["start_frame_idx"]
            maximum = kwargs["max_frame_num_to_track"]
            for frame in range(start, start + maximum + 1):
                yield frame, {
                    "out_binary_masks": np.ones((1, 1, 4, 5), dtype=bool),
                    "out_obj_ids": np.asarray([17]),
                }

    model = FakeModel()
    emitted = propagate_segment(
        model,
        {},
        raw_id=17,
        start_frame=2,
        end_frame=5,
        width=5,
        height=4,
        probability_threshold=0.5,
        mask_root=tmp_path,
    )
    assert model.arguments["max_frame_num_to_track"] == 3
    assert emitted == {2, 3, 4, 5}
    assert not (tmp_path / "00006.png").exists()
