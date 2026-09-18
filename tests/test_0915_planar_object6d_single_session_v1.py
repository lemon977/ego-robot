from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.ops import run_0915_planar_object6d_single_session_canary_v1 as runner
from chaoyang.pipeline.object6d_planar_observability_v1 import (
    DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    PlanarFrameInput,
    PlanarObservabilityError,
    build_observability_document,
    estimate_planar_frame,
)


def fixture(
    *, height: int = 60, width: int = 80,
    row_slice: slice = slice(20, 40), column_slice: slice = slice(10, 70),
) -> PlanarFrameInput:
    mask = np.zeros((height, width), bool)
    mask[row_slice, column_slice] = True
    depth = np.full((height, width), 0.75, np.float64)
    valid = np.ones((height, width), bool)
    yy, xx = np.indices((height, width), dtype=np.float64)
    mapping = np.stack((xx, yy), axis=-1)
    intrinsics = np.asarray([
        [100.0, 0.0, (width - 1) / 2],
        [0.0, 100.0, (height - 1) / 2],
        [0.0, 0.0, 1.0],
    ])
    return PlanarFrameInput(
        frame_index=0,
        mask=mask,
        mask_state="tracked",
        visibility_state=DIRECT_VISIBILITY,
        depth_m=depth,
        depth_valid=valid,
        depth_intrinsics=intrinsics,
        depth_to_mask_xy=mapping,
        registration_valid=np.ones((height, width), bool),
    )


def artifact(name: str) -> dict[str, object]:
    return {"path": f"/tmp/{name}", "bytes": 1, "sha256": "0" * 64}


def contract_inputs() -> dict[str, object]:
    return {
        "depth_result": artifact("result"),
        "depth_contract": artifact("depth-contract"),
        "depth_registration": artifact("registration"),
        "depth_registration_maps": artifact("maps"),
        "depth_frame_manifest": artifact("depth-frames"),
        "sam_role_manifest": artifact("manifest"),
        "sam_temporal_state_ledger": artifact("temporal"),
        "sam_masks": {
            "playing_card_00": artifact("card00"),
            "playing_card_01": artifact("card01"),
        },
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_visible_rectangle_has_three_independent_observable_quantities() -> None:
    result = estimate_planar_frame(fixture())
    assert result["translation"]["observability"] == (
        "OBSERVABLE_VISIBLE_SURFACE_CENTROID"
    )
    assert result["translation"]["semantics"] == (
        "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_CENTER"
    )
    assert result["plane_normal"]["observability"] == (
        "OBSERVABLE_DIRECT_VISIBLE_PLANE"
    )
    normal = np.asarray(result["plane_normal"]["estimate"]["unit_xyz"])
    assert np.allclose(normal, [0.0, 0.0, -1.0], atol=1e-10)
    assert result["plane_normal"]["residual"]["p90_plane_distance_m"] < 1e-10
    assert result["in_plane_rotation"]["observability"] == (
        "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS"
    )
    assert result["in_plane_rotation"]["estimate"]["symmetry_period_rad"] == pytest.approx(
        np.pi
    )
    assert result["hidden_geometry_inferred"] is False


@pytest.mark.parametrize(
    ("mask_state", "visibility_state", "reason"),
    [
        ("unknown", "UNKNOWN", "MASK_STATE_UNKNOWN"),
        ("tracked", "OCCLUDED", "VISIBILITY_OCCLUDED"),
    ],
)
def test_unknown_or_occluded_input_fails_closed(
    mask_state: str, visibility_state: str, reason: str,
) -> None:
    base = fixture()
    value = PlanarFrameInput(
        **{
            **base.__dict__,
            "mask_state": mask_state,
            "visibility_state": visibility_state,
        }
    )
    result = estimate_planar_frame(value)
    for component in ("translation", "plane_normal", "in_plane_rotation"):
        assert result[component] == {
            "observability": "UNOBSERVABLE",
            "estimate": None,
            "residual": None,
            "reason": reason,
            "semantics": result[component]["semantics"],
        }
    assert result["hidden_geometry_inferred"] is False


def test_observability_is_independent_when_plane_support_is_sparse() -> None:
    result = estimate_planar_frame(fixture(
        height=20, width=20, row_slice=slice(5, 11), column_slice=slice(5, 11),
    ))
    assert result["registered_valid_depth_count"] == 36
    assert result["translation"]["observability"] != "UNOBSERVABLE"
    assert result["plane_normal"]["observability"] == "UNOBSERVABLE"
    assert result["plane_normal"]["reason"] == "INSUFFICIENT_POINTS_FOR_PLANE"
    assert result["in_plane_rotation"]["observability"] == "UNOBSERVABLE"


def test_square_visible_support_does_not_invent_in_plane_rotation() -> None:
    result = estimate_planar_frame(fixture(
        row_slice=slice(10, 50), column_slice=slice(20, 60),
    ))
    assert result["translation"]["observability"] != "UNOBSERVABLE"
    assert result["plane_normal"]["observability"] != "UNOBSERVABLE"
    assert result["in_plane_rotation"]["observability"] == "UNOBSERVABLE"
    assert result["in_plane_rotation"]["reason"] == "VISIBLE_SUPPORT_AXIS_AMBIGUOUS"


def test_explicit_nonidentity_pixel_map_is_consumed() -> None:
    base = fixture(height=30, width=40, row_slice=slice(10, 20), column_slice=slice(15, 25))
    shifted = np.asarray(base.depth_to_mask_xy).copy()
    shifted[..., 0] += 1000.0
    value = PlanarFrameInput(**{**base.__dict__, "depth_to_mask_xy": shifted})
    result = estimate_planar_frame(value)
    assert result["registered_valid_depth_count"] == 0
    assert result["translation"]["reason"] == "INSUFFICIENT_REGISTERED_VISIBLE_DEPTH"


def test_unproven_registration_authority_is_rejected() -> None:
    base = fixture()
    value = PlanarFrameInput(
        **{**base.__dict__, "registration_authority": "ASSUMED_IDENTITY"}
    )
    with pytest.raises(PlanarObservabilityError, match="proven explicit pixel map"):
        estimate_planar_frame(value)
    assert base.depth_reference == DEPTH_REFERENCE
    assert base.registration_authority == MASK_REGISTRATION_AUTHORITY


def test_document_validates_and_has_no_unified_confidence_value() -> None:
    frame = estimate_planar_frame(fixture())
    document = build_observability_document(
        session_id="play_cards_0915_001",
        object_frames={
            "playing_card_00": [frame],
            "playing_card_01": [frame],
        },
        inputs=contract_inputs(),
    )
    schema = json.loads(runner.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(document)
    assert document["unified_confidence_emitted"] is False
    for obj in document["objects"]:
        for row in obj["frames"]:
            assert "confidence" not in row
            assert "pose" not in row


def test_runner_is_cpu_only_route_gated_and_uses_only_two_stable_cards() -> None:
    assert runner.INSTANCE_IDS == ("playing_card_00", "playing_card_01")
    assert runner.OUTPUT_NAMESPACE.name == runner.TASK_ID
    runner.validate_output_namespace(
        runner.OUTPUT_NAMESPACE / "attempts/attempt_0001"
    )
    with pytest.raises(RuntimeError, match="task namespace"):
        runner.validate_output_namespace(
            runner.OUTPUT_NAMESPACE.parent / "another_task/attempt_0001"
        )
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "validate_route()" in source
    assert "GPU_COMMAND_RECEIPT" not in source
    assert "torch" not in source
    assert '"weights": "ABSENT"' in source
    assert "identity_assumed" in source
    assert "IDENTITY_JOIN_FORBIDDEN" in source


def test_runner_keeps_masks_packed_until_each_frame_is_consumed(tmp_path: Path) -> None:
    masks = np.zeros((2, 3, 5), bool)
    masks[0, 1, 2] = True
    masks[1, 2, 4] = True
    path = tmp_path / "mask.npz"
    np.savez_compressed(
        path,
        packed=np.packbits(masks.reshape(2, -1), axis=1, bitorder="big"),
        frame_count=np.int32(2), height=np.int32(3), width=np.int32(5),
        bitorder=np.asarray("big"),
    )
    packed = runner._load_packed(path)
    assert packed.shape == masks.shape
    assert packed.packed.nbytes < masks.nbytes
    assert np.array_equal(packed.unpack(0), masks[0])
    assert np.array_equal(packed.unpack(1), masks[1])


def test_depth_upstream_requires_sha_bound_nonlinear_sam_join(tmp_path: Path) -> None:
    write_json(tmp_path / "RESULT.json", {
        "status": "PASSED", "depth_admission": "PASS",
        "external_accuracy": "UNVERIFIED",
    })
    write_json(tmp_path / "DEPTH_CONTRACT.json", {
        "external_accuracy": "UNVERIFIED",
        "frame_arrays": {"depth_reference": DEPTH_REFERENCE},
    })
    np.savez_compressed(
        tmp_path / "REGISTRATION_MAPS.npz",
        depth_to_sam_resize_xy=np.zeros((2, 3, 2), np.float32),
        sam_resize_in_bounds=np.ones((2, 3), bool),
    )
    maps_ref = runner.ref(tmp_path / "REGISTRATION_MAPS.npz")
    registration = {
        "source_domain": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
        "nonlinear_registration_maps": maps_ref,
        "depth_to_sam_resize_map": {
            "array": "depth_to_sam_resize_xy",
            "source": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
            "target": (
                "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960_PIXEL_CENTERS"
            ),
            "identity_assumed": False,
        },
        "mask_join_policy": (
            "ONLY_VIA_DEPTH_TO_SAM_RESIZE_XY_AND_SAM_RESIZE_IN_BOUNDS; "
            "IDENTITY_JOIN_FORBIDDEN"
        ),
    }
    write_json(tmp_path / "REGISTRATION.json", registration)
    _registration, inputs = runner._validate_depth_upstream(tmp_path)
    assert inputs["depth_registration_maps"] == maps_ref

    registration["depth_to_sam_resize_map"]["identity_assumed"] = True
    write_json(tmp_path / "REGISTRATION.json", registration)
    with pytest.raises(RuntimeError, match="explicit nonlinear map"):
        runner._validate_depth_upstream(tmp_path)
