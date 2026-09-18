"""Hard boundary for decoded VST video pixels.

The project owner has confirmed that the encoded VST video frames are already
undistorted.  Camera calibration files may retain capture-side distortion
coefficients for provenance, but those coefficients must never be applied to
decoded video pixels.  This module provides the only admitted image extraction
operation until a separate encoded-domain stereo calibration is published.
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


ENCODED_DOMAIN = "VST_ENCODED_ALREADY_UNDISTORTED"
ADMITTED_TRANSFORM = "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY"
PROHIBITED_LENS_OPERATIONS = frozenset({
    "EQUIDIS62_UNDISTORT",
    "EQUIDIS62_TO_PINHOLE",
    "LENS_UNDISTORTION_REMAP",
})


class VSTEncodedDomainError(ValueError):
    """Raised when decoded VST pixels would cross the image-domain boundary."""


def validate_encoded_video_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed unless a consumer preserves the confirmed encoded domain."""

    if contract.get("encoded_video_domain") != ENCODED_DOMAIN:
        raise VSTEncodedDomainError("VST encoded video domain is not pinned")
    if contract.get("transform") != ADMITTED_TRANSFORM:
        raise VSTEncodedDomainError("decoded VST video permits crop plus resize only")
    if contract.get("lens_undistortion_applied") is not False:
        raise VSTEncodedDomainError("decoded VST video must not be lens-undistorted")
    if contract.get("distortion_coefficients_consumed") is not False:
        raise VSTEncodedDomainError(
            "camera distortion coefficients are not encoded-video transforms"
        )
    operation = contract.get("operation")
    if operation in PROHIBITED_LENS_OPERATIONS:
        raise VSTEncodedDomainError(f"prohibited decoded-video operation: {operation}")


def split_resize_physical_eyes(
    sbs: np.ndarray,
    *,
    eye_width: int,
    source_indices: tuple[int, int],
    output_size: tuple[int, int] = (1280, 960),
) -> tuple[np.ndarray, np.ndarray]:
    """Crop physical left/right eyes and resize without geometric remapping."""

    frame = np.asarray(sbs)
    if frame.ndim not in {2, 3}:
        raise VSTEncodedDomainError("SBS frame must be HxW or HxWxC")
    if eye_width <= 0 or frame.shape[1] != eye_width * 2:
        raise VSTEncodedDomainError("SBS width does not equal two encoded eyes")
    if tuple(sorted(source_indices)) != (0, 1):
        raise VSTEncodedDomainError("physical eye source indices must be a permutation")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise VSTEncodedDomainError("output size must be positive width and height")

    halves = (frame[:, :eye_width], frame[:, eye_width:])
    interpolation = (
        cv2.INTER_AREA
        if output_size[0] <= eye_width and output_size[1] <= frame.shape[0]
        else cv2.INTER_LINEAR
    )
    return tuple(
        cv2.resize(halves[index], output_size, interpolation=interpolation)
        for index in source_indices
    )

