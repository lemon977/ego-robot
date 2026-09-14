from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_formal_object6d_constrained_mask as runner


def test_schema_and_method_contract_forbid_flow_and_predecessor_object():
    schema = json.loads(runner.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert runner.METHOD_CONTRACT["object_optical_flow_used"] is False
    assert runner.METHOD_CONTRACT["predecessor_object_mask_consumed"] is False
    assert runner.METHOD_CONTRACT["object_mask_source"] == "FORMAL_OBJECT6D_SELECTED_RGB_VISIBLE_MASK"
    assert set(schema["properties"]["upstream_object6d_scope"]["enum"]) == {
        "CANDIDATE_VISUAL_ONLY", "NUMERIC_AUTHORITY_MASK_CANDIDATE",
        "FORMAL_CONSUMPTION"
    }


def test_decode_object_masks_respects_shape_validity_and_bitorder(tmp_path):
    masks = np.zeros((3, 5, 9), dtype=np.uint8)
    masks[0, 1:3, 2:7] = 1
    masks[2, 0:2, 0:4] = 1
    path = tmp_path / "object.npz"
    np.savez_compressed(
        path,
        visible_mask_selected_rgb_packbits=np.packbits(masks, axis=2, bitorder="little"),
        mask_packbits_bitorder=np.asarray("little"),
        valid=np.asarray([True, False, True]),
        # Frame 1 is deliberately marked visible despite an invalid pose. The
        # fail-closed decoder must reject its packed pixels.
        visibility=np.asarray([True, True, True]),
    )
    decoded, authoritative, diagnostics = runner.decode_object_masks(path, 3, [9, 5])
    assert np.array_equal(decoded, masks.astype(bool))
    assert np.array_equal(authoritative, [True, False, True])
    assert diagnostics["ignored_nonvisible_mask_frames"] == 0


def test_tracker_colour_recall_counts_all_candidates_as_covered():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[45:56, 62:77] = 255
    image[56:65, 62:77] = 110
    joints = np.zeros((21, 2), dtype=np.float64)
    joints[0] = [70, 54]
    joints[[5, 9, 13, 17]] = [70, 72]
    refined = np.ones((96, 128), dtype=bool)
    result = runner.tracker_colour_recall(image, joints, refined)
    assert result["white"]["candidate_pixels"] > 0
    assert result["grey"]["candidate_pixels"] > 0
    assert result["white"]["recall"] == 1.0
    assert result["grey"]["recall"] == 1.0


def test_poker_face_tight_v3_candidate_is_exact_and_task_scoped():
    candidate = {
        "schema_version": "poker-face-tight-physical-instance-object6d-candidate-v3",
        "status": "HOLD_FOR_EXPLICIT_HUMAN_OBJECT6D_REVIEW",
        "consumption_authorized": False,
        "physical_instance_count": 1,
        "physical_identity": "card_ace_diamonds_001",
        "dimensions_m": [0.088, 0.063, 0.001],
        "gates": {
            "full_decode_pass": True,
            "packed_mask_zero_when_visibility_false": True,
            "identity_frozen_single_A_diamond": True,
            "task_geometry_exact_88x63x1mm": True,
            "face_interval_has_high_confidence_frames": True,
        },
        "mask_consumption_contract": (
            "Only packed_mask & valid & visibility. No geometry projection, temporal fill, "
            "or raw V2 packed mask may protect pixels when visibility=false."
        ),
    }
    assert runner.candidate_object6d_result_is_accepted(candidate, "poker")
    assert not runner.candidate_object6d_result_is_accepted(candidate, "chips")
    candidate["gates"]["packed_mask_zero_when_visibility_false"] = False
    assert not runner.candidate_object6d_result_is_accepted(candidate, "poker")
