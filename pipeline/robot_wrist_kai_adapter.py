"""Cross-session human-wrist to KaiHand-root adapter primitives.

Joint arrays are identity-bearing.  The final-v3 NPZ stores MANO21 order with
``wrist`` at index 0; the separately exported HumanEgo adapter JSON stores a
different 21-point order with ``wrist`` at index 5.  Shape alone cannot
distinguish them.  Callers must supply the names belonging to the exact source
artifact and resolve joints by name.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class WristKaiAdapterError(ValueError):
    """Raised when an identity or rigid-transform contract is ambiguous."""


FINAL_V3_MANO_21_JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_fingertip",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_fingertip",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_fingertip",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_fingertip",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "pinky_fingertip",
)


# The final-v3 producer writes the standard MANO21 tree.  Keep the indices next
# to the names because downstream contact and retarget code must never infer a
# semantic from an array's shape.  In particular, ``points[:5]`` is the thumb
# chain, not the five fingertips.
FINAL_V3_MANO_FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

FINAL_V3_MANO_MCP_INDICES: tuple[int, ...] = (2, 5, 9, 13, 17)
FINAL_V3_MANO_TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


HUMANEGO_21_JOINT_NAMES: tuple[str, ...] = (
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
    "wrist",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "palm_center",
)


def named_joint_index(joint_names: Sequence[str], name: str) -> int:
    """Return an unambiguous named index; never guess or silently fall back."""

    names = tuple(str(value) for value in joint_names)
    matches = tuple(index for index, value in enumerate(names) if value == name)
    if len(matches) != 1:
        raise WristKaiAdapterError(
            f"expected exactly one joint named {name!r}, found {len(matches)}"
        )
    return matches[0]


def named_joint(
    points: np.ndarray,
    joint_names: Sequence[str],
    name: str,
) -> np.ndarray:
    """Select a joint from an ``(..., joint, xyz)`` array by identity."""

    values = np.asarray(points, dtype=np.float64)
    names = tuple(str(value) for value in joint_names)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise WristKaiAdapterError("points must have shape (..., joint, 3)")
    if values.shape[-2] != len(names):
        raise WristKaiAdapterError("joint-name count differs from point array")
    result = values[..., named_joint_index(names, name), :]
    if not np.isfinite(result).all():
        raise WristKaiAdapterError(f"named joint {name!r} contains NaN/Inf")
    return result


def anatomical_basis_from_landmarks(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    pinky_mcp: np.ndarray,
    *,
    handedness: str,
) -> np.ndarray:
    """Construct a right-handed anatomical palm basis from four landmarks.

    Basis columns are ``(anatomical lateral, palm forward, palm normal)``.
    To place both chiralities in one semantic convention, lateral is
    pinky-to-index for a left hand and index-to-pinky for a right hand.  The
    exact same convention is applied to HumanEgo and the official Kai CAD;
    no image-side or array-index guess is permitted.  The raw axes are
    projected to the nearest proper orthonormal basis by SVD.
    """

    if handedness not in {"left", "right"}:
        raise WristKaiAdapterError("handedness must be 'left' or 'right'")
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (wrist, index_mcp, middle_mcp, pinky_mcp)
    )
    if any(value.shape != arrays[0].shape for value in arrays):
        raise WristKaiAdapterError("anatomical landmarks must share one shape")
    if arrays[0].ndim < 1 or arrays[0].shape[-1] != 3:
        raise WristKaiAdapterError("anatomical landmarks must have shape (...,3)")
    if not all(np.isfinite(value).all() for value in arrays):
        raise WristKaiAdapterError("anatomical landmarks contain NaN/Inf")

    wrist_value, index_value, middle_value, pinky_value = arrays
    forward = 0.5 * (index_value + middle_value) - wrist_value
    lateral_sign = 1.0 if handedness == "left" else -1.0
    lateral = lateral_sign * (index_value - pinky_value)
    forward_norm = np.linalg.norm(forward, axis=-1)
    lateral_norm = np.linalg.norm(lateral, axis=-1)
    if np.any(forward_norm <= 1e-9) or np.any(lateral_norm <= 1e-9):
        raise WristKaiAdapterError("degenerate MCP landmarks cannot define a palm basis")
    forward_unit = forward / forward_norm[..., None]
    lateral_unit = lateral / lateral_norm[..., None]
    normal = np.cross(lateral_unit, forward_unit)
    normal_norm = np.linalg.norm(normal, axis=-1)
    if np.any(normal_norm <= 1e-5):
        raise WristKaiAdapterError("lateral and forward palm axes are nearly collinear")
    normal_unit = normal / normal_norm[..., None]
    raw = np.stack((lateral_unit, forward_unit, normal_unit), axis=-1)

    flat = raw.reshape((-1, 3, 3))
    projected = np.empty_like(flat)
    for index, value in enumerate(flat):
        left, _, right_t = np.linalg.svd(value)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right_t)
        projected[index] = left @ correction @ right_t
    result = projected.reshape(raw.shape)
    if not np.isfinite(result).all():
        raise WristKaiAdapterError("anatomical basis projection is non-finite")
    return result


def anatomical_wrist_frame(
    points: np.ndarray,
    joint_names: Sequence[str],
    *,
    handedness: str,
) -> np.ndarray:
    """Construct an anatomical wrist frame using only identity-bearing joints."""

    return anatomical_basis_from_landmarks(
        named_joint(points, joint_names, "wrist"),
        named_joint(points, joint_names, "index_proximal"),
        named_joint(points, joint_names, "middle_proximal"),
        named_joint(points, joint_names, "pinky_proximal"),
        handedness=handedness,
    )


def final_v3_mano_palm_basis(
    points: np.ndarray,
    *,
    handedness: str,
) -> np.ndarray:
    """Return the MANO21 palm basis from wrist0 and MCPs 5/9/17.

    This entry point intentionally accepts no alternate joint-name list.  It
    is for the final-v3 NPZ authority only, and therefore makes an accidental
    HumanEgo21 interpretation fail at the call site instead of silently using
    HumanEgo wrist5 as a MANO index-MCP.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.shape[-2:] != (21, 3):
        raise WristKaiAdapterError("final-v3 MANO points must have shape (...,21,3)")
    if not np.isfinite(values).all():
        raise WristKaiAdapterError("final-v3 MANO points contain NaN/Inf")
    return anatomical_basis_from_landmarks(
        values[..., 0, :],
        values[..., 5, :],
        values[..., 9, :],
        values[..., 17, :],
        handedness=handedness,
    )


def unit_bone_directions(points: np.ndarray) -> np.ndarray:
    """Return unit directions for a non-degenerate polyline.

    Length is deliberately discarded.  This is the primitive used by the
    MANO-to-Kai retargeter so human/robot hand-size differences cannot pull a
    robot fingertip through a task object merely to match a metric endpoint.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3 or values.shape[-2] < 2:
        raise WristKaiAdapterError("polyline points must have shape (...,joint>=2,3)")
    if not np.isfinite(values).all():
        raise WristKaiAdapterError("polyline points contain NaN/Inf")
    vectors = np.diff(values, axis=-2)
    lengths = np.linalg.norm(vectors, axis=-1)
    if np.any(lengths <= 1e-9):
        raise WristKaiAdapterError("zero-length bone cannot define a direction")
    return vectors / lengths[..., None]


def final_v3_mano_unit_bones_local(
    points: np.ndarray,
    *,
    handedness: str,
) -> dict[str, np.ndarray]:
    """Return four unit bone directions per MANO finger in palm coordinates."""

    values = np.asarray(points, dtype=np.float64)
    basis = final_v3_mano_palm_basis(values, handedness=handedness)
    if basis.ndim != 2:
        raise WristKaiAdapterError("one MANO21 hand is required")
    output: dict[str, np.ndarray] = {}
    for finger, indices in FINAL_V3_MANO_FINGER_CHAINS.items():
        camera_directions = unit_bone_directions(values[np.asarray(indices)])
        output[finger] = camera_directions @ basis
    return output


def resample_polyline_unit_directions(
    points: np.ndarray,
    *,
    bone_count: int = 4,
) -> np.ndarray:
    """Arc-length resample a chain, returning a fixed count of unit bones.

    KaiHand has a different number and placement of mechanical pivots from
    MANO.  Resampling by *normalized* arc length gives four comparable angular
    bones without ever matching metric fingertip positions.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise WristKaiAdapterError("one polyline with shape (joint>=2,3) is required")
    if not isinstance(bone_count, int) or bone_count <= 0:
        raise WristKaiAdapterError("bone_count must be a positive integer")
    directions = unit_bone_directions(values)
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], bone_count + 1)
    sampled = np.empty((bone_count + 1, 3), dtype=np.float64)
    for offset, target in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        segment = max(segment, 0)
        fraction = (target - cumulative[segment]) / lengths[segment]
        sampled[offset] = (
            values[segment] * (1.0 - fraction)
            + values[segment + 1] * fraction
        )
    # ``directions`` above performs the explicit degeneracy validation before
    # interpolation; retain the binding so static checkers see the intent.
    del directions
    return unit_bone_directions(sampled)


def _validate_rotation(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise WristKaiAdapterError(f"{name} must be one finite 3x3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=2e-6, rtol=0):
        raise WristKaiAdapterError(f"{name} is not orthonormal")
    if abs(float(np.linalg.det(value)) - 1.0) > 2e-6:
        raise WristKaiAdapterError(f"{name} is not right-handed")
    return value


def fit_fixed_wrist_to_kai_rotation(
    human_local_vectors: np.ndarray,
    kai_local_vectors: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit one proper rotation mapping Kai-local rays to human-wrist rays.

    Both inputs are ``(sample, 3)`` vectors from the respective anatomical
    wrist/root to a *same-named* fingertip or pad.  Their lengths are ignored,
    so the fit cannot introduce scale or a task/object-specific offset.
    """

    from scipy.spatial.transform import Rotation

    human = np.asarray(human_local_vectors, dtype=np.float64)
    kai = np.asarray(kai_local_vectors, dtype=np.float64)
    if human.shape != kai.shape or human.ndim != 2 or human.shape[1] != 3:
        raise WristKaiAdapterError("vector arrays must share shape (sample, 3)")
    if len(human) < 3 or not np.isfinite(human).all() or not np.isfinite(kai).all():
        raise WristKaiAdapterError("at least three finite vector pairs are required")
    human_norm = np.linalg.norm(human, axis=1)
    kai_norm = np.linalg.norm(kai, axis=1)
    if np.any(human_norm <= 1e-9) or np.any(kai_norm <= 1e-9):
        raise WristKaiAdapterError("zero-length rays cannot define an orientation")
    human_unit = human / human_norm[:, None]
    kai_unit = kai / kai_norm[:, None]
    rotation, _ = Rotation.align_vectors(human_unit, kai_unit)
    matrix = _validate_rotation(rotation.as_matrix(), name="fitted rotation")
    aligned = kai_unit @ matrix.T
    cosine = np.clip(np.sum(aligned * human_unit, axis=1), -1.0, 1.0)
    rms_angle_deg = float(np.sqrt(np.mean(np.rad2deg(np.arccos(cosine)) ** 2)))
    return matrix, rms_angle_deg


def build_kai_root_transform(
    wrist_camera: np.ndarray,
    human_root_rotation_camera: np.ndarray,
    wrist_to_kai_rotation: np.ndarray,
) -> np.ndarray:
    """Compose ``T_camera_kai`` with exact named-wrist translation.

    The fixed calibration maps Kai-root local axes into the HumanEgo local
    wrist frame.  It is therefore post-multiplied onto HumanEgo's local-to-
    camera rotation.
    """

    wrist = np.asarray(wrist_camera, dtype=np.float64)
    if wrist.shape != (3,) or not np.isfinite(wrist).all():
        raise WristKaiAdapterError("wrist_camera must be one finite 3-vector")
    human_rotation = _validate_rotation(
        human_root_rotation_camera, name="human root rotation"
    )
    adapter_rotation = _validate_rotation(
        wrist_to_kai_rotation, name="wrist-to-Kai rotation"
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = human_rotation @ adapter_rotation
    result[:3, 3] = wrist
    return result


def chordal_rotation_mean(rotations: np.ndarray) -> np.ndarray:
    """Return the projected chordal mean of right-handed rotation matrices."""

    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise WristKaiAdapterError("rotations must have non-empty shape (sample,3,3)")
    if not np.isfinite(values).all():
        raise WristKaiAdapterError("rotations contain NaN/Inf")
    for offset, value in enumerate(values):
        _validate_rotation(value, name=f"rotation[{offset}]")
    left, _, right_t = np.linalg.svd(np.mean(values, axis=0))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    return _validate_rotation(left @ correction @ right_t, name="chordal mean")


def aggregate_world_roots(world_roots: np.ndarray) -> np.ndarray:
    """Aggregate ``(sample, side, 4, 4)`` roots without choosing one frame."""

    values = np.asarray(world_roots, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (2, 4, 4) or not len(values):
        raise WristKaiAdapterError("world_roots must have non-empty shape (sample,2,4,4)")
    result = np.repeat(np.eye(4)[None], 2, axis=0)
    for side in (0, 1):
        if not np.isfinite(values[:, side]).all():
            raise WristKaiAdapterError("world roots contain NaN/Inf")
        if not np.allclose(values[:, side, 3], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise WristKaiAdapterError("world roots are not homogeneous transforms")
        result[side, :3, 3] = np.median(values[:, side, :3, 3], axis=0)
        result[side, :3, :3] = chordal_rotation_mean(values[:, side, :3, :3])
    return result


def fit_world_rig_from_bilateral_roots(
    natural_roots_rig: np.ndarray,
    target_roots_world: np.ndarray,
    *,
    baseline_weight: float = 4.0,
) -> np.ndarray:
    """Fit one session rigid transform from two Natural-q Kai root pairs.

    Both endpoint orientations contribute to the proper-rotation SVD.  The
    bilateral baseline supplies the positional orientation constraint, while
    translation aligns the endpoint midpoint.  No per-side transform exists.
    """

    source = np.asarray(natural_roots_rig, dtype=np.float64)
    target = np.asarray(target_roots_world, dtype=np.float64)
    if source.shape != (2, 4, 4) or target.shape != (2, 4, 4):
        raise WristKaiAdapterError("bilateral root arrays must have shape (2,4,4)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise WristKaiAdapterError("bilateral roots contain NaN/Inf")
    if not np.isfinite(baseline_weight) or baseline_weight <= 0:
        raise WristKaiAdapterError("baseline_weight must be positive")
    source_direction = source[0, :3, 3] - source[1, :3, 3]
    target_direction = target[0, :3, 3] - target[1, :3, 3]
    source_length, target_length = np.linalg.norm(source_direction), np.linalg.norm(target_direction)
    if source_length <= 1e-9 or target_length <= 1e-9:
        raise WristKaiAdapterError("bilateral roots must have distinct positions")
    source_direction /= source_length
    target_direction /= target_length
    covariance = baseline_weight * np.outer(target_direction, source_direction)
    for source_root, target_root in zip(source, target, strict=True):
        covariance += target_root[:3, :3] @ source_root[:3, :3].T
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    source_midpoint = np.mean(source[:, :3, 3], axis=0)
    target_midpoint = np.mean(target[:, :3, 3], axis=0)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_midpoint - rotation @ source_midpoint
    return result
