from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from chaoyang.pipeline.object6d_planar_observability_v2 import (
    COMPONENT_NAMES,
    DIRECT_VISIBILITY,
    OBJECT_INSTANCE_IDS,
    PlanarFrameInput,
    PlanarObservabilityError,
    build_observability_document,
    estimate_planar_frame,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "contracts/object6d_planar_observability_v2.schema.json"


def _frame_input(
    *,
    height: int = 60,
    width: int = 80,
    row_slice: slice = slice(20, 40),
    column_slice: slice = slice(10, 70),
) -> PlanarFrameInput:
    mask = np.zeros((height, width), bool)
    mask[row_slice, column_slice] = True
    depth = np.full((height, width), 0.75, np.float64)
    yy, xx = np.indices((height, width), dtype=np.float64)
    return PlanarFrameInput(
        frame_index=0,
        mask=mask,
        mask_state="tracked",
        visibility_state=DIRECT_VISIBILITY,
        depth_m=depth,
        depth_valid=np.ones_like(depth, bool),
        depth_intrinsics=np.asarray([
            [100.0, 0.0, (width - 1) / 2],
            [0.0, 100.0, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ]),
        depth_to_mask_xy=np.stack((xx, yy), axis=-1),
        registration_valid=np.ones_like(depth, bool),
    )


def _artifact(name: str) -> dict[str, object]:
    return {"path": f"/tmp/{name}", "bytes": 1, "sha256": "0" * 64}


def _inputs() -> dict[str, object]:
    return {
        "depth_result": _artifact("result"),
        "depth_adapter_contract": _artifact("adapter"),
        "depth_rgb_alignment_qa": _artifact("rgb-alignment"),
        "depth_contract": _artifact("depth-contract"),
        "depth_summary": _artifact("depth-summary"),
        "depth_frame_manifest": _artifact("depth-frames"),
        "depth_to_sam_mapping": {
            "type": "ANALYTIC_PIXEL_CENTER_2X",
            "formula_x": "x_sam = 2*x_depth + 0.5",
            "formula_y": "y_sam = 2*y_depth + 0.5",
            "source": "ORIGINAL_PHYSICAL_LEFT_640x480",
            "target": "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960",
        },
        "sam_role_manifest": _artifact("manifest"),
        "sam_temporal_state_ledger": _artifact("temporal"),
        "sam_masks": {
            instance_id: _artifact(instance_id)
            for instance_id in OBJECT_INSTANCE_IDS
        },
    }


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


def test_v2_frame_has_four_independent_components_and_no_single_valid() -> None:
    frame = estimate_planar_frame(_frame_input())
    assert all(name in frame for name in COMPONENT_NAMES)
    assert "translation" not in frame
    assert "in_plane_rotation" not in frame
    assert "valid" not in frame
    assert frame["center_xyz"]["observability"] == (
        "OBSERVABLE_VISIBLE_SURFACE_CENTROID"
    )
    assert frame["center_xyz"]["semantics"] == (
        "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_CENTER"
    )
    assert frame["plane_normal"]["observability"] == (
        "OBSERVABLE_DIRECT_VISIBLE_PLANE"
    )
    assert frame["inplane_rotation"]["observability"] == (
        "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS"
    )
    assert frame["full_extent"] == {
        "observability": "UNOBSERVABLE",
        "estimate": None,
        "residual": None,
        "reason": "FULL_OBJECT_BOUNDARY_VISIBILITY_UNPROVEN",
        "semantics": (
            "FULL_PHYSICAL_OBJECT_EXTENT_REQUIRES_COMPLETE_BOUNDARY_EVIDENCE"
        ),
    }


def test_v2_components_fail_closed_independently() -> None:
    frame = estimate_planar_frame(_frame_input(
        height=20,
        width=20,
        row_slice=slice(5, 11),
        column_slice=slice(5, 11),
    ))
    assert frame["center_xyz"]["observability"] != "UNOBSERVABLE"
    assert frame["plane_normal"]["observability"] == "UNOBSERVABLE"
    assert frame["inplane_rotation"]["observability"] == "UNOBSERVABLE"
    assert frame["full_extent"]["observability"] == "UNOBSERVABLE"


def test_v2_document_keeps_cards_tray_and_group_semantically_separate() -> None:
    frame = estimate_planar_frame(_frame_input())
    document = build_observability_document(
        session_id="play_cards_0915_001",
        object_frames={instance_id: [frame] for instance_id in OBJECT_INSTANCE_IDS},
        inputs=_inputs(),
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(document)

    assert [row["instance_id"] for row in document["objects"]] == list(
        OBJECT_INSTANCE_IDS
    )
    tray = document["support_entities"][0]
    assert tray == {
        "instance_id": "black_card_tray",
        "entity_role": "SUPPORT_ENTITY",
        "mask_evidence": "ABSENT",
        "observability": "UNKNOWN",
        "geometry_authority": "NONE",
        "reason": "INDEPENDENT_SUPPORT_MASK_NOT_AVAILABLE",
    }
    group = document["semantic_groups"][0]
    assert group["group_id"] == "card_set"
    assert group["member_instance_ids"] == list(OBJECT_INSTANCE_IDS)
    assert group["mask_authority"] == "NONE"
    assert group["geometry_authority"] == "NONE"
    assert group["pose_authority"] == "NONE"
    assert not _contains_key(document, "valid")


def test_v2_rejects_missing_or_reordered_physical_card_instances() -> None:
    frame = estimate_planar_frame(_frame_input())
    with pytest.raises(PlanarObservabilityError, match="requires playing_card"):
        build_observability_document(
            session_id="play_cards_0915_001",
            object_frames={
                "playing_card_00": [frame],
                "playing_card_01": [frame],
            },
            inputs=_inputs(),
        )
    with pytest.raises(PlanarObservabilityError, match="requires playing_card"):
        build_observability_document(
            session_id="play_cards_0915_001",
            object_frames={
                "playing_card_01": [frame],
                "playing_card_00": [frame],
                "playing_card_02": [frame],
            },
            inputs=_inputs(),
        )


def test_v2_schema_forbids_geometry_on_semantic_group_or_tray_mask_claim() -> None:
    frame = estimate_planar_frame(_frame_input())
    document = build_observability_document(
        session_id="play_cards_0915_001",
        object_frames={instance_id: [frame] for instance_id in OBJECT_INSTANCE_IDS},
        inputs=_inputs(),
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    bad_group = copy.deepcopy(document)
    bad_group["semantic_groups"][0]["center_xyz"] = [0.0, 0.0, 0.0]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad_group)

    bad_tray = copy.deepcopy(document)
    bad_tray["support_entities"][0]["mask_evidence"] = "PRESENT"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad_tray)
