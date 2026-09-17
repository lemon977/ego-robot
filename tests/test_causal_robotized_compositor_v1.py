from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.pipeline.causal_robotized_compositor_v1 import (  # noqa: E402
    CausalCompositorError,
    TemporalObjectDonor,
    UNKNOWN_SENTINEL_RGB,
    VisualInputMode,
    assert_causal_frame_contract,
    compose_robotized_frame,
)
from chaoyang.pipeline.occlusion_compositor_v1 import ObjectPixelSource, Ownership  # noqa: E402


def _base(shape: tuple[int, int] = (1, 3)) -> dict[str, np.ndarray]:
    owner = np.full(shape, int(Ownership.OBJECT_FRONT), dtype=np.uint8)
    return {
        "clean_background_rgb": np.full((*shape, 3), 7, dtype=np.uint8),
        "ownership": owner,
        "ownership_training_valid_mask": np.ones(shape, dtype=np.bool_),
        "current_raw_rgb": np.full((*shape, 3), 10, dtype=np.uint8),
        "current_raw_object_visible_mask": np.zeros(shape, dtype=np.bool_),
        "robot_rgb": np.full((*shape, 3), 80, dtype=np.uint8),
        "robot_valid_mask": np.ones(shape, dtype=np.bool_),
    }


def test_future_donor_content_cannot_change_current_causal_rgb_sha() -> None:
    inputs = _base()
    past = TemporalObjectDonor(
        frame_id=4,
        rgb=np.full((1, 3, 3), 20, dtype=np.uint8),
        valid_mask=np.ones((1, 3), dtype=np.bool_),
    )
    future_a = TemporalObjectDonor(
        frame_id=6,
        rgb=np.full((1, 3, 3), 30, dtype=np.uint8),
        valid_mask=np.ones((1, 3), dtype=np.bool_),
    )
    # Deliberately mutate both bytes and shape of a future frame.  Causal mode
    # must not inspect that array after rejecting it by frame metadata.
    future_b = TemporalObjectDonor(
        frame_id=6,
        rgb=np.full((7, 9, 3), 255, dtype=np.uint8),
        valid_mask=np.zeros((7, 9), dtype=np.bool_),
    )
    first = compose_robotized_frame(
        target_frame_id=5,
        mode=VisualInputMode.CAUSAL_TRAINING_INPUT,
        temporal_object_donors=[past, future_a],
        **inputs,
    )
    second = compose_robotized_frame(
        target_frame_id=5,
        mode=VisualInputMode.CAUSAL_TRAINING_INPUT,
        temporal_object_donors=[past, future_b],
        **inputs,
    )
    assert first.rgb_sha256 == second.rgb_sha256
    assert first.training_valid_mask_sha256 == second.training_valid_mask_sha256
    assert first.referenced_donor_frame_ids == (4,)
    assert second.referenced_donor_frame_ids == (4,)
    assert np.all(first.object_donor_frame_id == 4)
    assert np.array_equal(first.rgb, np.full((1, 3, 3), 20, dtype=np.uint8))
    assert_causal_frame_contract(first)


def test_causal_output_never_references_donor_after_target() -> None:
    result = compose_robotized_frame(
        target_frame_id=5,
        mode="CAUSAL_TRAINING_INPUT",
        temporal_object_donors=[
            TemporalObjectDonor(3, np.full((1, 3, 3), 20, np.uint8), np.ones((1, 3), bool)),
            TemporalObjectDonor(8, np.full((1, 3, 3), 30, np.uint8), np.ones((1, 3), bool)),
        ],
        **_base(),
    )
    assert result.training_authorized
    assert all(frame_id <= result.target_frame_id for frame_id in result.referenced_donor_frame_ids)
    assert 8 not in result.object_donor_frame_id


def test_offline_bidirectional_may_use_future_but_is_never_training_authorized() -> None:
    result = compose_robotized_frame(
        target_frame_id=5,
        mode="OFFLINE_BIDIRECTIONAL_VISUALIZATION",
        temporal_object_donors=[
            TemporalObjectDonor(6, np.full((1, 3, 3), 30, np.uint8), np.ones((1, 3), bool))
        ],
        **_base(),
    )
    assert result.referenced_donor_frame_ids == (6,)
    assert not result.training_authorized
    assert not np.any(result.training_valid_mask)
    assert_causal_frame_contract(result)


def test_unknown_is_invalid_and_clean_table_pixel_never_impersonates_object() -> None:
    inputs = _base((1, 1))
    result = compose_robotized_frame(
        target_frame_id=5,
        mode="CAUSAL_TRAINING_INPUT",
        temporal_object_donors=[],
        **inputs,
    )
    assert result.effective_ownership[0, 0] == Ownership.TIE_UNKNOWN
    assert not result.training_valid_mask[0, 0]
    assert result.object_pixel_source[0, 0] == ObjectPixelSource.NONE_UNKNOWN
    assert np.array_equal(result.rgb[0, 0], UNKNOWN_SENTINEL_RGB)
    assert not np.array_equal(result.rgb[0, 0], inputs["clean_background_rgb"][0, 0])
    assert not result.clean_pixels_may_supply_object
    assert_causal_frame_contract(result)


def test_robot_front_without_render_evidence_is_unknown_not_clean_fallback() -> None:
    inputs = _base((1, 1))
    inputs["ownership"][:] = int(Ownership.ROBOT_FRONT)
    inputs["robot_valid_mask"][:] = False
    result = compose_robotized_frame(
        target_frame_id=5,
        mode="CAUSAL_TRAINING_INPUT",
        **inputs,
    )
    assert result.effective_ownership[0, 0] == Ownership.TIE_UNKNOWN
    assert np.array_equal(result.rgb[0, 0], UNKNOWN_SENTINEL_RGB)
    assert not result.training_valid_mask[0, 0]


def test_duplicate_donor_frame_id_is_rejected() -> None:
    donor = TemporalObjectDonor(
        4, np.full((1, 3, 3), 20, np.uint8), np.ones((1, 3), dtype=np.bool_)
    )
    with pytest.raises(CausalCompositorError, match="unique"):
        compose_robotized_frame(
            target_frame_id=5,
            mode="CAUSAL_TRAINING_INPUT",
            temporal_object_donors=[donor, donor],
            **_base(),
        )


def test_causal_ledger_schema_separates_training_and_offline_modes() -> None:
    schema = json.loads(
        (PROJECT / "contracts/robotized_compositor_causal_v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    base = {
        "schema_version": "ROBOTIZED_COMPOSITOR_CAUSAL_V1",
        "session_id": "fixture",
        "artifact_revision": "R7_0",
        "target_frame_id": 5,
        "mode": "CAUSAL_TRAINING_INPUT",
        "training_authorized": True,
        "referenced_donor_frame_ids": [4],
        "causal_proof": {
            "all_donor_frame_ids_lte_target": True,
            "future_mutation_invariant_test": True,
        },
        "unknown_training_policy": "TRAINING_VALID_MASK_FALSE",
        "clean_pixels_may_supply_object": False,
        "control_ground_truth": False,
        "rgb_sha256": "0" * 64,
        "training_valid_mask_sha256": "1" * 64,
    }
    validator.validate(base)
    invalid_offline = dict(base)
    invalid_offline["mode"] = "OFFLINE_BIDIRECTIONAL_VISUALIZATION"
    assert any("False was expected" in error.message for error in validator.iter_errors(invalid_offline))
    invalid_clean = dict(base)
    invalid_clean["clean_pixels_may_supply_object"] = True
    assert any("False was expected" in error.message for error in validator.iter_errors(invalid_clean))
