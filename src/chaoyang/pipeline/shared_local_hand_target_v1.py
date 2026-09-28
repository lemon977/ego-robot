"""Morphology-aware local hand targets shared by MANUS25 and HaWoR21.

This development-only adapter intentionally carries no wrist-position or wrist-
orientation authority.  It compares fifteen intra-finger bone directions and a
thumb-tip to index-tip pinch vector normalized by a frozen, per-source-side palm
width.  Human and robot bone lengths are never required to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
import trimesh

from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    FINGER_CHAINS,
    HandModel,
    HuroHandOnlyError,
)
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.pico_manus_motion_v2 import MANUS_NAMES, MAP_25_TO_21


METHOD_ID = "SHARED_LOCAL_HAND_DIRECTION_PINCH_V2"
AUTHORITY = "DEVELOPMENT_ONLY_LOCAL_Q22_NO_WRIST_OR_PHYSICAL_AUTHORITY"
# Both producers expose an explicit anatomical (left, right) axis and the two
# pinned URDFs are physical (left, right).  The historical [1, 0] mapping came
# from a camera/render chirality path and is not a local-hand label contract.
SOURCE_TO_PHYSICAL = np.asarray([0, 1], dtype=np.int32)
HAWOR21_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
INTRA_FINGER_EDGES = tuple(
    (chain[index], chain[index + 1])
    for chain in FINGER_CHAINS
    for index in range(3)
)
PINCH_EDGE = (4, 8)

# The first link-to-link offset in the exported KaiHand is a mechanical joint
# placement, not an anatomical phalanx.  These landmarks are the existing,
# evidence-backed convention used by the anatomical feasibility diagnostic:
# three articulated joint centres followed by the distal mesh tip.
ROBOT_ANATOMICAL_LINKS = {
    "thumb": (3, 5, 6),
    "index": (2, 3, 4),
    "middle": (2, 3, 4),
    "ring": (2, 3, 4),
    "pinky": (2, 3, 4),
}
_TIP_LOCAL_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


@dataclass(frozen=True)
class LocalHandTarget:
    directions: np.ndarray
    direction_valid: np.ndarray
    normalized_pinch: np.ndarray
    pinch_valid: bool
    source_reference_width_m: float
    robot_reference_width_m: float
    source_to_robot_rotation: np.ndarray


@dataclass(frozen=True)
class LocalSolveDiagnostics:
    success: bool
    evaluations: int
    cost: float
    max_direction_angle_deg: float
    normalized_pinch_error: float
    temporal_delta_rms_rad: float


def adapt_source_points(
    points: np.ndarray,
    *,
    source_kind: str,
    joint_names: Sequence[str],
    units: str = "m",
) -> np.ndarray:
    """Return an explicit MANO21-layout array without changing coordinates."""
    value = np.asarray(points, dtype=np.float64)
    if units != "m":
        raise HuroHandOnlyError(f"source units must be metres, got {units!r}")
    if source_kind == "MANUS25":
        if tuple(joint_names) != MANUS_NAMES or value.shape[-2:] != (25, 3):
            raise HuroHandOnlyError("MANUS25 name/order or shape mismatch")
        return value[..., MAP_25_TO_21, :].copy()
    if source_kind == "HAWOR21":
        if tuple(joint_names) != HAWOR21_NAMES or value.shape[-2:] != (21, 3):
            raise HuroHandOnlyError("HaWoR21 name/order or shape mismatch")
        return value.copy()
    raise HuroHandOnlyError(f"unsupported local-hand source: {source_kind!r}")


def _palm_basis(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.shape != (21, 3) or not np.isfinite(value).all():
        raise HuroHandOnlyError("palm frame needs finite (21,3) points")
    x = value[5] - value[17]
    x_norm = float(np.linalg.norm(x))
    if not np.isfinite(x_norm) or x_norm < 1e-8:
        raise HuroHandOnlyError("degenerate palm width")
    x /= x_norm
    y = value[9] - value[0]
    y -= x * float(np.dot(x, y))
    y_norm = float(np.linalg.norm(y))
    if not np.isfinite(y_norm) or y_norm < 1e-8:
        raise HuroHandOnlyError("degenerate palm longitudinal axis")
    y /= y_norm
    basis = np.column_stack((x, y, np.cross(x, y)))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8, rtol=0):
        raise HuroHandOnlyError("non-orthogonal palm frame")
    if float(np.linalg.det(basis)) < 0.999999:
        raise HuroHandOnlyError("improper palm frame")
    return basis


def _valid_unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    valid = bool(np.isfinite(value).all() and np.isfinite(norm) and norm >= 1e-8)
    if not valid:
        return np.full(3, np.nan, dtype=np.float64), False
    return value / norm, True


def palm_width(points: np.ndarray) -> float:
    value = np.asarray(points, dtype=np.float64)
    if value.shape != (21, 3):
        raise HuroHandOnlyError("palm width needs (21,3) points")
    width = float(np.linalg.norm(value[5] - value[17]))
    if not np.isfinite(width) or width < 1e-8:
        raise HuroHandOnlyError("degenerate palm width")
    return width


def _terminal_tip_local(hand: HandModel, finger: str) -> np.ndarray:
    """Return the frozen distal mesh-tip proxy in the terminal-link frame."""
    prefix = "hand_l" if hand.side == "left" else "hand_r"
    terminal = ROBOT_ANATOMICAL_LINKS[finger][-1]
    link = f"{prefix}_{finger}_link{terminal}"
    key = (str(hand.urdf_path), hand.side, finger)
    cached = _TIP_LOCAL_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    matches = [visual for visual in hand.model.visuals if visual.link == link]
    if len(matches) != 1:
        raise HuroHandOnlyError(f"cannot uniquely resolve terminal visual {link}")
    visual = matches[0]
    mesh = trimesh.load(visual.mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise HuroHandOnlyError(f"invalid terminal mesh {visual.mesh_path}")
    if finger == "thumb":
        extreme = float(np.max(vertices[:, 0]))
        selected = vertices[vertices[:, 0] >= extreme - 0.0005]
    else:
        extreme = float(np.min(vertices[:, 2]))
        selected = vertices[vertices[:, 2] <= extreme + 0.0005]
    if not len(selected):
        raise HuroHandOnlyError(f"empty terminal-tip selection {link}")
    local = np.asarray(visual.origin, dtype=np.float64) @ np.r_[selected.mean(axis=0), 1.0]
    _TIP_LOCAL_CACHE[key] = local[:3].copy()
    return local[:3].copy()


def anatomical_keypoints_from_q(hand: HandModel, q: np.ndarray) -> np.ndarray:
    """Extract 21 semantic landmarks without consuming fixed link offsets."""
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (22,) or not np.isfinite(value).all():
        raise HuroHandOnlyError("q must be finite with shape (22,)")
    transforms = forward_kinematics(
        hand.model, dict(zip(hand.joint_names, value.tolist(), strict=True))
    )
    prefix = "hand_l" if hand.side == "left" else "hand_r"
    root = transforms[f"{prefix}_base_link"][:3, 3]
    points = [root]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        numbers = ROBOT_ANATOMICAL_LINKS[finger]
        terminal_link = f"{prefix}_{finger}_link{numbers[-1]}"
        points.extend(
            transforms[f"{prefix}_{finger}_link{number}"][:3, 3]
            for number in numbers
        )
        tip = transforms[terminal_link] @ np.r_[_terminal_tip_local(hand, finger), 1.0]
        points.append(tip[:3])
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (21, 3) or not np.isfinite(result).all():
        raise HuroHandOnlyError("anatomical FK keypoint extraction failed")
    return result


def frozen_source_widths(points: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Freeze one median palm width for each anatomical source side."""
    value = np.asarray(points, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if value.ndim != 4 or value.shape[0] != 2 or value.shape[2:] != (21, 3):
        raise HuroHandOnlyError("source points must be anatomical (2,T,21,3)")
    if mask.shape != value.shape[:2]:
        raise HuroHandOnlyError("observed must be anatomical (2,T)")
    output = np.full(2, np.nan, dtype=np.float64)
    for anatomical in range(2):
        widths = []
        for frame in np.flatnonzero(mask[anatomical]):
            try:
                widths.append(palm_width(value[anatomical, frame]))
            except HuroHandOnlyError:
                continue
        if not widths:
            raise HuroHandOnlyError(f"no valid palm width for anatomical side {anatomical}")
        output[anatomical] = float(np.median(widths))
    return output


def build_local_target(
    source_points: np.ndarray,
    robot_neutral_points: np.ndarray,
    *,
    source_reference_width_m: float,
    robot_reference_width_m: float,
) -> LocalHandTarget:
    source = np.asarray(source_points, dtype=np.float64)
    robot = np.asarray(robot_neutral_points, dtype=np.float64)
    if source.shape != (21, 3) or robot.shape != (21, 3):
        raise HuroHandOnlyError("local target points must be (21,3)")
    if not np.isfinite(source_reference_width_m) or source_reference_width_m < 1e-8:
        raise HuroHandOnlyError("invalid frozen source palm width")
    if not np.isfinite(robot_reference_width_m) or robot_reference_width_m < 1e-8:
        raise HuroHandOnlyError("invalid frozen robot palm width")
    source_basis = _palm_basis(source)
    robot_basis = _palm_basis(robot)
    rotation = robot_basis @ source_basis.T
    directions = np.full((len(INTRA_FINGER_EDGES), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(INTRA_FINGER_EDGES), dtype=bool)
    for index, (start, end) in enumerate(INTRA_FINGER_EDGES):
        unit, okay = _valid_unit((source[end] - source[start]) @ rotation.T)
        directions[index], valid[index] = unit, okay
    pinch = ((source[PINCH_EDGE[1]] - source[PINCH_EDGE[0]]) @ rotation.T
             / float(source_reference_width_m))
    pinch_valid = bool(np.isfinite(pinch).all())
    if not pinch_valid:
        pinch = np.full(3, np.nan, dtype=np.float64)
    return LocalHandTarget(
        directions=directions,
        direction_valid=valid,
        normalized_pinch=pinch,
        pinch_valid=pinch_valid,
        source_reference_width_m=float(source_reference_width_m),
        robot_reference_width_m=float(robot_reference_width_m),
        source_to_robot_rotation=rotation,
    )


def _robot_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.full((len(INTRA_FINGER_EDGES), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(INTRA_FINGER_EDGES), dtype=bool)
    for index, (start, end) in enumerate(INTRA_FINGER_EDGES):
        directions[index], valid[index] = _valid_unit(points[end] - points[start])
    return directions, valid


def _direction_angles_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    dots = np.sum(actual * target, axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def solve_local_frame(
    hand: HandModel,
    target: LocalHandTarget,
    *,
    previous_q: np.ndarray | None = None,
    max_evaluations: int = 80,
) -> tuple[np.ndarray, np.ndarray, LocalSolveDiagnostics]:
    if not target.direction_valid.all() or not target.pinch_valid:
        raise HuroHandOnlyError("complete valid local direction/pinch target required")
    neutral_q = 0.5 * (hand.lower + hand.upper)
    initial = neutral_q if previous_q is None else np.asarray(previous_q, dtype=np.float64)
    if initial.shape != (22,) or not np.isfinite(initial).all():
        raise HuroHandOnlyError("previous q must be finite (22,)")
    initial = np.clip(initial, hand.lower, hand.upper)
    previous = initial.copy()
    direction_weight = 20.0
    pinch_weight = 8.0
    temporal_weight = np.sqrt(10.0)
    rest_weight = np.sqrt(0.15)

    def residual(q: np.ndarray) -> np.ndarray:
        points = anatomical_keypoints_from_q(hand, q)
        directions, valid = _robot_directions(points)
        if not valid.all():
            raise HuroHandOnlyError("robot FK produced degenerate bone")
        pinch = ((points[PINCH_EDGE[1]] - points[PINCH_EDGE[0]])
                 / target.robot_reference_width_m)
        return np.concatenate((
            direction_weight * (directions - target.directions).reshape(-1),
            pinch_weight * (pinch - target.normalized_pinch),
            temporal_weight * (q - previous),
            rest_weight * (q - neutral_q),
        ))

    solved = least_squares(
        residual,
        initial,
        bounds=(hand.lower, hand.upper),
        max_nfev=max_evaluations,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    q = np.asarray(solved.x, dtype=np.float64)
    points = anatomical_keypoints_from_q(hand, q)
    directions, valid = _robot_directions(points)
    if not valid.all():
        raise HuroHandOnlyError("solved robot FK produced degenerate bone")
    angles = _direction_angles_deg(directions, target.directions)
    pinch = ((points[PINCH_EDGE[1]] - points[PINCH_EDGE[0]])
             / target.robot_reference_width_m)
    diagnostics = LocalSolveDiagnostics(
        success=bool(solved.success and np.isfinite(q).all()),
        evaluations=int(solved.nfev),
        cost=float(solved.cost),
        max_direction_angle_deg=float(np.max(angles)),
        normalized_pinch_error=float(np.linalg.norm(pinch - target.normalized_pinch)),
        temporal_delta_rms_rad=float(np.sqrt(np.mean((q - previous) ** 2))),
    )
    return q, points, diagnostics


def solve_local_sequence(
    hands: tuple[HandModel, HandModel],
    source_points: np.ndarray,
    observed: np.ndarray,
    frame_ids: np.ndarray,
    *,
    source_kind: str,
    joint_names: Sequence[str],
    anatomical_side_names: Sequence[str] = ("left", "right"),
    units: str = "m",
    max_evaluations: int = 80,
    progress=None,
) -> dict[str, np.ndarray]:
    adapted = adapt_source_points(
        source_points, source_kind=source_kind, joint_names=joint_names, units=units
    )
    mask = np.asarray(observed, dtype=bool)
    frames = np.asarray(frame_ids)
    if tuple(anatomical_side_names) != ("left", "right"):
        raise HuroHandOnlyError("anatomical side axis must be explicit left,right")
    if adapted.ndim != 4 or adapted.shape[0] != 2:
        raise HuroHandOnlyError("adapted source must be anatomical (2,T,J,3)")
    count = adapted.shape[1]
    if mask.shape != (2, count) or frames.shape != (count,) or np.any(np.diff(frames) <= 0):
        raise HuroHandOnlyError("invalid observed/frame axis")
    source_width = frozen_source_widths(adapted, mask)
    neutral = tuple(
        anatomical_keypoints_from_q(hand, 0.5 * (hand.lower + hand.upper))
        for hand in hands
    )
    robot_width = np.asarray([palm_width(points) for points in neutral], dtype=np.float64)
    shape = (count, 2)
    data = {
        "q22": np.full((*shape, 22), np.nan),
        "fk21_root_relative": np.full((*shape, 21, 3), np.nan),
        "target_directions": np.full((*shape, len(INTRA_FINGER_EDGES), 3), np.nan),
        "target_direction_valid": np.zeros((*shape, len(INTRA_FINGER_EDGES)), bool),
        "target_normalized_pinch": np.full((*shape, 3), np.nan),
        "target_pinch_valid": np.zeros(shape, bool),
        "source_to_robot_rotation": np.full((*shape, 3, 3), np.nan),
        "solver_success": np.zeros(shape, bool),
        "solver_evaluations": np.zeros(shape, np.int32),
        "solver_cost": np.full(shape, np.nan),
        "max_direction_angle_deg": np.full(shape, np.nan),
        "normalized_pinch_error": np.full(shape, np.nan),
        "temporal_delta_rms_rad": np.full(shape, np.nan),
        "failure_reason": np.full(shape, "UNOBSERVED", dtype="U96"),
        "source_observed_physical": mask.T.copy(),
        "source_frame_id": frames.copy(),
        "human_to_physical": SOURCE_TO_PHYSICAL.copy(),
        "source_reference_width_m": source_width.copy(),
        "robot_reference_width_m": robot_width.copy(),
        "joint_names": np.asarray([hand.joint_names for hand in hands]),
        "method_id": np.asarray(METHOD_ID),
        "authority": np.asarray(AUTHORITY),
        "source_kind": np.asarray(source_kind),
        "anatomical_side_names": np.asarray(anatomical_side_names),
        "physical_robot_side_names": np.asarray(("left", "right")),
        "intra_finger_edges": np.asarray(INTRA_FINGER_EDGES, dtype=np.int32),
    }
    previous: list[np.ndarray | None] = [None, None]
    for frame in range(count):
        if frame and frames[frame] != frames[frame - 1] + 1:
            previous = [None, None]
        for anatomical, physical_value in enumerate(SOURCE_TO_PHYSICAL):
            physical = int(physical_value)
            if not mask[anatomical, frame]:
                previous[physical] = None
                continue
            try:
                target = build_local_target(
                    adapted[anatomical, frame],
                    neutral[physical],
                    source_reference_width_m=source_width[anatomical],
                    robot_reference_width_m=robot_width[physical],
                )
                data["target_directions"][frame, physical] = target.directions
                data["target_direction_valid"][frame, physical] = target.direction_valid
                data["target_normalized_pinch"][frame, physical] = target.normalized_pinch
                data["target_pinch_valid"][frame, physical] = target.pinch_valid
                data["source_to_robot_rotation"][frame, physical] = target.source_to_robot_rotation
                q, fk, diagnostic = solve_local_frame(
                    hands[physical], target,
                    previous_q=previous[physical],
                    max_evaluations=max_evaluations,
                )
                data["q22"][frame, physical] = q
                data["fk21_root_relative"][frame, physical] = fk - fk[0]
                data["solver_success"][frame, physical] = diagnostic.success
                data["solver_evaluations"][frame, physical] = diagnostic.evaluations
                data["solver_cost"][frame, physical] = diagnostic.cost
                data["max_direction_angle_deg"][frame, physical] = diagnostic.max_direction_angle_deg
                data["normalized_pinch_error"][frame, physical] = diagnostic.normalized_pinch_error
                data["temporal_delta_rms_rad"][frame, physical] = diagnostic.temporal_delta_rms_rad
                data["failure_reason"][frame, physical] = (
                    "NONE" if diagnostic.success else "SOLVER_NOT_CONVERGED"
                )
                previous[physical] = q.copy() if diagnostic.success else None
            except HuroHandOnlyError as error:
                data["failure_reason"][frame, physical] = str(error)[:96]
                previous[physical] = None
        if progress is not None and (frame % 25 == 0 or frame == count - 1):
            progress(frame + 1, count)
    return data


def evaluate_local_quality(
    hands: tuple[HandModel, HandModel],
    states: dict[str, np.ndarray],
    *,
    collision_checker=None,
    direction_gate_deg: float = 15.0,
    pinch_gate: float = 0.1,
) -> dict[str, np.ndarray]:
    success = np.asarray(states["solver_success"], dtype=bool)
    shape = success.shape
    limit_pass = np.zeros(shape, dtype=bool)
    collision_known = np.zeros(shape, dtype=bool)
    collision_pass = np.zeros(shape, dtype=bool)
    collision_count = np.full(shape, -1, dtype=np.int32)
    max_penetration_m = np.full(shape, np.nan)
    independent_fk_max_abs_m = np.full(shape, np.nan)
    for frame, physical in np.argwhere(success):
        frame, physical = int(frame), int(physical)
        hand = hands[physical]
        q = np.asarray(states["q22"][frame, physical], dtype=np.float64)
        limit_pass[frame, physical] = bool(
            np.isfinite(q).all()
            and np.all(q >= hand.lower - 1e-9)
            and np.all(q <= hand.upper + 1e-9)
        )
        fk = anatomical_keypoints_from_q(hand, q)
        fk -= fk[0]
        independent_fk_max_abs_m[frame, physical] = float(np.max(np.abs(
            fk - states["fk21_root_relative"][frame, physical]
        )))
        if collision_checker is not None:
            diagnostic = collision_checker.check(physical, q)
            collision_known[frame, physical] = bool(diagnostic.known)
            collision_count[frame, physical] = int(diagnostic.illegal_contact_count)
            max_penetration_m[frame, physical] = float(diagnostic.max_penetration_m)
            collision_pass[frame, physical] = bool(
                diagnostic.known and diagnostic.illegal_contact_count == 0
            )
    complete_target = (
        np.asarray(states["target_direction_valid"], dtype=bool).all(axis=-1)
        & np.asarray(states["target_pinch_valid"], dtype=bool)
    )
    direction_pass = (
        np.isfinite(states["max_direction_angle_deg"])
        & (states["max_direction_angle_deg"] <= direction_gate_deg)
    )
    pinch_pass = (
        np.isfinite(states["normalized_pinch_error"])
        & (states["normalized_pinch_error"] <= pinch_gate)
    )
    fk_pass = np.isfinite(independent_fk_max_abs_m) & (independent_fk_max_abs_m <= 1e-8)
    numeric_local_gate_pass = (
        success & complete_target & direction_pass & pinch_pass & limit_pass
        & collision_pass & fk_pass
    )
    return {
        "complete_target": complete_target,
        "direction_pass": direction_pass,
        "pinch_pass": pinch_pass,
        "limit_pass": limit_pass,
        "collision_known": collision_known,
        "collision_pass": collision_pass,
        "collision_count": collision_count,
        "max_penetration_m": max_penetration_m,
        "independent_fk_max_abs_m": independent_fk_max_abs_m,
        # This is the complete machine-checkable gate.  Independent visual
        # review remains a separate session-level requirement before a label
        # can be admitted for training.
        "numeric_local_gate_pass": numeric_local_gate_pass,
    }


def enumerate_h50(
    quality_pass: np.ndarray,
    frame_ids: np.ndarray,
    timestamp_ns: np.ndarray,
    *,
    source_frames: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    quality = np.asarray(quality_pass, dtype=bool)
    frames = np.asarray(frame_ids)
    timestamps = np.asarray(timestamp_ns)
    if quality.ndim != 2 or quality.shape[1] != 2:
        raise HuroHandOnlyError("quality pass must have shape (T,2)")
    if frames.shape != (len(quality),) or timestamps.shape != frames.shape:
        raise HuroHandOnlyError("H50 timeline mismatch")
    if source_frames != 51:
        raise HuroHandOnlyError("H50 requires exactly 51 source frames")
    anchors: list[int] = []
    sides: list[int] = []
    dt = np.diff(timestamps)
    positive = dt[dt > 0]
    if not len(positive):
        return np.asarray(anchors, np.int64), np.asarray(sides, np.int32)
    contiguous_edge = (
        (np.diff(frames) == 1)
        & (dt > 0)
        & (dt <= 2.5 * np.median(positive))
    )
    for start in range(len(quality) - source_frames + 1):
        stop = start + source_frames
        if not contiguous_edge[start:stop - 1].all():
            continue
        for side in range(2):
            if quality[start:stop, side].all():
                anchors.append(start)
                sides.append(side)
    return np.asarray(anchors, np.int64), np.asarray(sides, np.int32)
