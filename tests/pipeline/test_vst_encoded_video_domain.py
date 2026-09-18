from __future__ import annotations

import numpy as np
import pytest

from chaoyang.pipeline.vst_encoded_video_domain import (
    ADMITTED_TRANSFORM,
    ENCODED_DOMAIN,
    VSTEncodedDomainError,
    split_resize_physical_eyes,
    validate_encoded_video_contract,
)


def _contract() -> dict[str, object]:
    return {
        "encoded_video_domain": ENCODED_DOMAIN,
        "transform": ADMITTED_TRANSFORM,
        "operation": ADMITTED_TRANSFORM,
        "lens_undistortion_applied": False,
        "distortion_coefficients_consumed": False,
    }


def test_encoded_vst_contract_allows_only_crop_then_resize() -> None:
    validate_encoded_video_contract(_contract())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lens_undistortion_applied", True),
        ("distortion_coefficients_consumed", True),
        ("operation", "EQUIDIS62_TO_PINHOLE"),
        ("transform", "STEREO_RECTIFICATION_REMAP"),
    ],
)
def test_encoded_vst_contract_rejects_lens_or_geometric_remapping(
    field: str,
    value: object,
) -> None:
    contract = _contract()
    contract[field] = value
    with pytest.raises(VSTEncodedDomainError):
        validate_encoded_video_contract(contract)


def test_split_resize_routes_physical_source_indices_without_remap() -> None:
    left_storage = np.full((4, 6, 3), 17, np.uint8)
    right_storage = np.full((4, 6, 3), 231, np.uint8)
    sbs = np.hstack((left_storage, right_storage))

    physical_left, physical_right = split_resize_physical_eyes(
        sbs,
        eye_width=6,
        source_indices=(1, 0),
        output_size=(3, 2),
    )

    assert physical_left.shape == (2, 3, 3)
    assert physical_right.shape == (2, 3, 3)
    assert np.all(physical_left == 231)
    assert np.all(physical_right == 17)


def test_split_resize_rejects_invalid_eye_routing() -> None:
    sbs = np.zeros((4, 12, 3), np.uint8)
    with pytest.raises(VSTEncodedDomainError, match="permutation"):
        split_resize_physical_eyes(sbs, eye_width=6, source_indices=(1, 1))

