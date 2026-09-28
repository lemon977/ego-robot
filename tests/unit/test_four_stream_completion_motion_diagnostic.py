import json
import numpy as np
import pytest
from chaoyang.ops.run_four_stream_completion_motion_diagnostic import analyze


def fixture():
    pose = np.zeros((2, 149, 21, 3), dtype=np.float32)
    pixel = np.zeros((2, 149, 21, 2), dtype=np.float64)
    pixel[..., 0] = 500
    pixel[..., 1] = 500
    observed = np.zeros((2, 149), bool)
    inferred = np.zeros_like(observed)
    valid = np.zeros_like(observed)
    inferred[1, :102] = True
    valid[1, :102] = True
    pose[1, :, 0, 2] = 1.0
    pose[1, 48:, 0, 0] = 1.25
    source = {"joints_3d_camera": pose, "joints_2d": pixel, "observed": observed, "inferred": inferred,
              "predicted_valid": valid, "original_frame_indices": np.arange(149),
              "timestamp_ns": np.arange(149) + 1, "anatomical_side_names": np.array(["left", "right"])}
    boxes = np.zeros((149, 2, 4))
    boxes[:, :, 2:] = [100, 100]
    boxes[:, 1] = [719, 948, 741, 960]
    roi = {"boxes_xyxy": boxes, "roi_valid": np.ones((149, 2), bool),
           "anatomical_side_names": np.array(["left", "right"])}
    return source, roi, {i: (960, 960) for i in range(46, 61)}


def test_reports_crop_support_and_source_jump_without_repair():
    source, roi, sizes = fixture()
    before = {key: value.copy() for key, value in source.items()}
    out = analyze(source, roi, sizes, list(range(46, 61)))
    row = next(item for item in out["frame_rows"] if item["frame_id"] == 47 and item["side"] == "right")
    assert row["touches_bottom"] and row["roi_width_px"] == 22 and row["roi_height_px"] == 12
    right = next(item for item in out["full_timeline"] if item["side"] == "right")
    assert right["largest_delta_from_frame"] == 47 and right["largest_delta_to_frame"] == 48
    assert right["largest_consecutive_root_delta_m"] == pytest.approx(1.25)
    json.dumps(out, allow_nan=False)
    for key, value in before.items():
        np.testing.assert_array_equal(source[key], value)


def test_preserves_absent_side_and_inferred_not_observed():
    source, roi, sizes = fixture()
    out = analyze(source, roi, sizes, list(range(46, 61)))
    left = next(item for item in out["full_timeline"] if item["side"] == "left")
    right = next(item for item in out["full_timeline"] if item["side"] == "right")
    assert left["observed_frames"] == left["inferred_frames"] == left["predicted_valid_frames"] == 0
    assert right["observed_frames"] == 0 and right["inferred_frames"] == 102 and right["predicted_valid_frames"] == 102


def test_rejects_side_time_and_schema_drift():
    source, roi, sizes = fixture()
    source["anatomical_side_names"] = np.array(["right", "left"])
    with pytest.raises(ValueError, match="SIDE"):
        analyze(source, roi, sizes, list(range(46, 61)))
    source, roi, sizes = fixture()
    source["timestamp_ns"][3] = source["timestamp_ns"][2]
    with pytest.raises(ValueError, match="TIMESTAMP"):
        analyze(source, roi, sizes, list(range(46, 61)))
    source, roi, sizes = fixture()
    del source["observed"]
    with pytest.raises(ValueError, match="SCHEMA"):
        analyze(source, roi, sizes, list(range(46, 61)))


def test_requires_all_pinned_image_sizes():
    source, roi, sizes = fixture()
    del sizes[50]
    with pytest.raises(ValueError, match="MISSING"):
        analyze(source, roi, sizes, list(range(46, 61)))
