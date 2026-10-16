"""Per-side frame evidence for an offline, RGB-derived Robot display.

HaWoR outputs are predictions.  The fields below keep drawing a Robot apart
from evidence for visual qualification and from real-Robot training labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math


SHORT_GAP_SECONDS = 10.0 / 30.0
MODEL_SOURCES = frozenset({"RGB_HAWOR_PREDICTED", "RGB_HAWOR_MASK_PROMPTED",
                           "HAWOR_MODEL_INFILLED"})
FILL_SOURCES = frozenset({"DETERMINISTIC_INTERPOLATED", "EDGE_HELD"})
DIAGNOSTIC_SOURCES = frozenset({"DIAGNOSTIC_LONG_GAP", "DIAGNOSTIC_NEUTRAL"})
ALL_SOURCES = MODEL_SOURCES | FILL_SOURCES | DIAGNOSTIC_SOURCES | {"MISSING"}
TIME_MODES = frozenset({"CAPTURE_SYNC_VERIFIED", "TRACKER_CLIP_METADATA_ONLY",
                        "TIMEBASE_UNVERIFIED", "CAMERA_ONLY_DIAGNOSTIC"})
VISIBILITY = frozenset({"VISIBLE", "OCCLUDED", "OUT_OF_VIEW", "UNKNOWN"})
ROI_SOURCES = frozenset({"MODEL_DETECTOR", "CURRENT_RGB_ROLE_MASK", "NONE", "DIAGNOSTIC"})
EDGE_GAP_KINDS = frozenset({"PREFIX", "SUFFIX", "INTERIOR", "NONE"})


def short_gap_limit_frames(fps: float) -> int:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("INVALID_FPS")
    return int(math.floor(fps * SHORT_GAP_SECONDS + 1e-12))


@dataclass(frozen=True)
class FrameMotionState:
    source_frame: int
    side: str
    roi_source: str
    motion_source: str
    visibility: str
    time_mode: str
    wrist_position_valid: bool
    wrist_orientation_valid: bool
    local_hand_shape_valid: bool
    robot_drawable: bool
    motion_evidence_eligible: bool
    training_eligible: bool
    coordinate_frame: str
    world_pose_aligned: bool
    model_input_valid: bool
    model_prediction_valid: bool
    model_completion_valid: bool
    deterministic_fill: bool
    final_visible: bool

    def as_json(self) -> dict:
        return asdict(self)


def classify_frame(*, frame: int, side: str, roi_source: str, source: str,
                   visibility: str, time_mode: str, wrist_position_valid: bool,
                   wrist_orientation_valid: bool, local_hand_shape_valid: bool,
                   coordinate_frame: str, model_input_valid: bool = False,
                   gap_frames: int | None = None, fps: float | None = None,
                   edge_gap_kind: str = "NONE", world_pose_aligned: bool = False,
                   same_identity_anchors: bool = False,
                   same_coordinate_anchors: bool = False,
                   model_admitted: bool = False,
                   mask_prompt_admitted: bool = False) -> FrameMotionState:
    if frame < 0 or side not in {"left", "right"} or source not in ALL_SOURCES:
        raise ValueError("FRAME_IDENTITY_OR_SOURCE")
    if visibility not in VISIBILITY or time_mode not in TIME_MODES:
        raise ValueError("VISIBILITY_OR_TIME_MODE")
    if roi_source not in ROI_SOURCES or edge_gap_kind not in EDGE_GAP_KINDS:
        raise ValueError("ROI_SOURCE_OR_EDGE_KIND")
    if coordinate_frame not in {"WORLD", "BASE", "CAMERA", "UNKNOWN"}:
        raise ValueError("COORDINATE_FRAME")

    pose_valid = wrist_position_valid and wrist_orientation_valid and local_hand_shape_valid
    short_bounded = (gap_frames is not None and fps is not None
                     and 0 < gap_frames <= short_gap_limit_frames(fps))
    if source == "DETERMINISTIC_INTERPOLATED":
        credible = (pose_valid and short_bounded and same_identity_anchors
                    and same_coordinate_anchors and coordinate_frame in {"WORLD", "BASE"})
    elif source == "EDGE_HELD":
        credible = (pose_valid and short_bounded and edge_gap_kind in {"PREFIX", "SUFFIX"}
                    and coordinate_frame in {"WORLD", "BASE"})
    elif source == "HAWOR_MODEL_INFILLED":
        credible = pose_valid and model_admitted and coordinate_frame in {"WORLD", "BASE"}
    elif source == "RGB_HAWOR_PREDICTED":
        credible = (pose_valid and model_input_valid and roi_source == "MODEL_DETECTOR"
                    and coordinate_frame in {"WORLD", "BASE"})
    elif source == "RGB_HAWOR_MASK_PROMPTED":
        credible = (pose_valid and model_input_valid and roi_source == "CURRENT_RGB_ROLE_MASK"
                    and mask_prompt_admitted and coordinate_frame in {"WORLD", "BASE"})
    else:
        credible = False

    # An unverified 0915 camera/head clock cannot certify a world trajectory.
    if not world_pose_aligned or time_mode in {"TIMEBASE_UNVERIFIED", "CAMERA_ONLY_DIAGNOSTIC"}:
        credible = False
    drawable = source != "MISSING"
    return FrameMotionState(
        source_frame=frame, side=side, roi_source=roi_source,
        motion_source=source, visibility=visibility, time_mode=time_mode,
        wrist_position_valid=wrist_position_valid,
        wrist_orientation_valid=wrist_orientation_valid,
        local_hand_shape_valid=local_hand_shape_valid,
        robot_drawable=drawable, motion_evidence_eligible=bool(credible),
        training_eligible=False, coordinate_frame=coordinate_frame,
        world_pose_aligned=world_pose_aligned,
        model_input_valid=model_input_valid,
        model_prediction_valid=source in {"RGB_HAWOR_PREDICTED", "RGB_HAWOR_MASK_PROMPTED"}
        and pose_valid, model_completion_valid=source == "HAWOR_MODEL_INFILLED" and model_admitted,
        deterministic_fill=source in FILL_SOURCES | {"DIAGNOSTIC_LONG_GAP"},
        final_visible=False,
    )


def mark_rendered(state: FrameMotionState, *, robot_pixels_visible: bool) -> FrameMotionState:
    """Only the compositor can assert that a drawable Robot was visible."""
    if robot_pixels_visible and not state.robot_drawable:
        raise ValueError("VISIBLE_ROBOT_WITHOUT_DRAWABLE_STATE")
    return replace(state, final_visible=robot_pixels_visible)
