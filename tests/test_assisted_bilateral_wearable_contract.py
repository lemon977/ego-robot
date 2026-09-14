import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_assisted_bilateral_wearable_temporal as temporal


PROJECT = Path("/mnt/workspace/code/chaoyang")


def test_frozen_contract_is_bounded_assisted_not_session_patch():
    protocol = json.loads(temporal.PROTOCOL.read_text())
    assert protocol["status"] == "FROZEN_BEFORE_TEMPORAL_CANARY"
    assert protocol["temporal_contract"]["maximum_anchor_frames_per_role_per_video"] == 4
    assert len(protocol["temporal_contract"]["canary_normalized_positions"]) == 12
    assert protocol["temporal_contract"]["additional_anchor_choice"].startswith("the failing canary")
    assert "SESSION_SPECIFIC_THRESHOLD" in protocol["forbidden"]


def test_text_selector_accepts_near_anchor_without_changing_pixels():
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, 2:7, 2:7] = True
    masks[1, 12:18, 12:18] = True
    masks[2, 0:2, 18:20] = True
    outputs = {"out_binary_masks": masks, "out_obj_ids": np.asarray([7, 9, 11])}
    roles = [
        {"name": "upper_human_core", "positive": [[8, 4]]},
        {"name": "lower_human_core", "positive": [[14, 14]]},
    ]
    selected, evidence = temporal.select_text_ids(outputs, roles, 20, 20)
    assert selected == {"upper_human_core": 7, "lower_human_core": 9}
    assert evidence["upper_human_core"]["candidates"][0]["anchor_distance_px"] == 2.0
    assert np.array_equal(masks[0], outputs["out_binary_masks"][0])


def test_registered_roles_and_object_points_are_valid_display_coordinates():
    anchors = json.loads(temporal.ANCHORS.read_text())
    for case in ("chips", "poker"):
        for x, y in anchors[case]["object_protection_points"]:
            assert 0 <= x < 540 and 0 <= y < 960
        assert anchors[case]["human_core"]["upper"]["positive"]
        assert anchors[case]["human_core"]["lower"]["positive"]
        assert anchors[case]["tracker_wearable"]["upper"]["positive"]
        assert anchors[case]["tracker_wearable"]["lower"]["positive"]


def test_runner_contains_no_forbidden_pixel_repair_calls():
    source = Path(temporal.__file__).read_text()
    for forbidden in (
        "cv2.dilate(",
        "cv2.erode(",
        "cv2.morphologyEx(",
        "cv2.inRange(",
        "binary_fill_holes",
        "connectedComponents",
    ):
        assert forbidden not in source


def test_object_raw_is_overlay_only_and_never_subtracted():
    protocol = json.loads(temporal.PROTOCOL.read_text())
    assert protocol["object_priority"]["object_raw_pixels_modified"] is False
    assert "no subtraction" in protocol["object_priority"]["raw_human_object_overlap_action"]
    source = Path(temporal.__file__).read_text()
    assert "union & ~object_mask" not in source
    assert "np.logical_and(union, object_mask)" in source


def test_priming_id_is_explicitly_discarded():
    protocol = json.loads(temporal.PROTOCOL.read_text())
    runtime = protocol["pinned_runtime_adapter"]
    assert runtime["fixed_id_order"] == ["DISCARDED_PRIMING", "TARGET_WEARABLE"]
    assert "discard" in runtime["priming_output_policy"]
    source = Path(temporal.__file__).read_text()
    assert "priming_output_consumed\": False" in source
