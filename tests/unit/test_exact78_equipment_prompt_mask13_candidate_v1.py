"""CPU-only negative gates for the unregistered Mask13 successor."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from chaoyang.governance import exact78_equipment_prompt_mask13_candidate_v1 as design
from chaoyang.governance import execute_exact78_equipment_prompt_mask13_candidate_v1 as executor
from chaoyang.ops import run_exact78_equipment_prompt_mask13_candidate_v1 as runner


def _config() -> dict:
    return {
        "old_human_prompt": design.OLD_PROMPT,
        "new_human_prompt": design.NEW_PROMPT,
        "equipment_prompt_added": design.EQUIPMENT_PROMPT_ADDED,
        "other_role_prompts": {
            "tracker_equipment": "a wrist worn tracking device",
            "vst_equipment": "a tactile glove",
            "vst_finger_sleeve": "a finger sleeve",
            "vst_yellow_cable": "a yellow cable",
            "poker_object": "a playing card",
            "chips_object": "a potato chip",
        },
    }


@pytest.mark.parametrize("family,task", [("TRACKER_0902", "poker"),
                                         ("TRACKER_0901", "chips"),
                                         ("VST_0915", "poker")])
def test_only_added_equipment_query_changes(family: str, task: str) -> None:
    config = _config()
    original = {"human": config["old_human_prompt"],
                "object": config["other_role_prompts"][f"{task}_object"],
                "equipment": (config["other_role_prompts"]["vst_equipment"]
                              if family == "VST_0915" else
                              config["other_role_prompts"]["tracker_equipment"])}
    if family == "VST_0915":
        original.update(finger_sleeve="a finger sleeve", yellow_cable="a yellow cable")
    actual = runner.build_prompts(original, config, family)
    assert actual == executor.expected_prompts({"source_family": family, "task": task}, config)
    assert actual["human"] == design.NEW_PROMPT
    assert actual["equipment_original"] == original["equipment"]
    assert actual["equipment_added"] == design.EQUIPMENT_PROMPT_ADDED
    wrong = copy.deepcopy(original)
    wrong["human"] = "a person's hand and forearm, changed again"
    with pytest.raises(RuntimeError, match="BASE_OR_ADDED_PROMPT_DRIFT"):
        runner.build_prompts(wrong, config, family)
    bad_config = copy.deepcopy(config)
    bad_config["equipment_prompt_added"] = "a wristband"
    with pytest.raises(RuntimeError, match="BASE_OR_ADDED_PROMPT_DRIFT"):
        runner.build_prompts(original, bad_config, family)
    bad_config = copy.deepcopy(config)
    bad_config["new_human_prompt"] = config["old_human_prompt"]
    with pytest.raises(RuntimeError, match="BASE_OR_ADDED_PROMPT_DRIFT"):
        runner.build_prompts(original, bad_config, family)
    with pytest.raises(RuntimeError, match="UNSUPPORTED_SOURCE"):
        runner.build_prompts(original, config, "HOLDOUT_0901")


def test_pixel_source_and_protection_negative_gates() -> None:
    shape = (2, 3)
    empty = np.zeros(shape, dtype=bool)
    original = empty.copy()
    original[0, 0] = True
    added = empty.copy()
    added[0, 1] = True
    human = empty.copy()
    human[1, 0] = True
    obj = empty.copy()
    obj[1, 1] = True
    combined = human | original | added
    images = {"human": human, "object": obj,
              "equipment_original": original, "equipment_added": added,
              "equipment": original | added,
              "human_equipment": combined,
              "unknown": combined & obj,
              "protected": obj & ~combined,
              "removal": combined & ~obj}
    assert executor.check_mask_algebra(images, "TRACKER_0902", "scene", 0) == 1
    bad = dict(images, equipment=original)
    with pytest.raises(RuntimeError, match="EQUIPMENT_PIXEL_SOURCE_FAIL"):
        executor.check_mask_algebra(bad, "TRACKER_0902", "scene", 0)
    bad = dict(images, removal=combined)
    bad["removal"] = combined | obj
    with pytest.raises(RuntimeError, match="PROTECTION_ALGEBRA_FAIL"):
        executor.check_mask_algebra(bad, "TRACKER_0902", "scene", 0)


def test_source_specific_clean_stop() -> None:
    tracker = {"source_family": "TRACKER_0902", "equipment_empty_frames": [85],
               "object_empty_frames": [0], "finger_sleeve_empty_frames": [],
               "yellow_cable_empty_frames": []}
    assert executor.source_quality_failures(tracker) == [
        "MASK_DEVICE_UNVERIFIED", "MASK_VISIBLE_OBJECT_UNPROTECTED"]
    vst = dict(tracker, source_family="VST_0915", equipment_empty_frames=[],
               object_empty_frames=[], finger_sleeve_empty_frames=[0],
               yellow_cable_empty_frames=[75])
    assert executor.source_quality_failures(vst) == ["MASK_SLEEVE_CABLE_UNVERIFIED"]
    assert executor.source_quality_failures(dict(tracker, equipment_empty_frames=[],
                                                 object_empty_frames=[])) == []
