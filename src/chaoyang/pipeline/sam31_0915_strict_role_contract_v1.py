"""Frozen role prompts and temporal-state rules for the 0915 SAM3.1 canary.

The boxes are development annotations for one user-authorized session.  They
are prompts, not pixel truth.  The pinned multiplex runtime represents the
first box as a geometric prompt; no separate exemplar encoder is claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


SESSION_ID = "play_cards_0915_001"
IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP"
WIDTH = 1280
HEIGHT = 960
FRAME_COUNT = 150
TEMPORAL_STATES = ("seeded", "tracked", "reseeded", "unknown")
ROLE_NAMES = (
    "left_hand", "right_hand", "left_forearm", "right_forearm",
    "left_finger_sleeve", "right_finger_sleeve",
    "left_cable", "right_cable", "task_object",
)
REMOVAL_ROLES = ROLE_NAMES[:-1]
FORBIDDEN_IDENTITY_TOKENS = ("tracker", "controller", "pico")


class StrictRoleContractError(ValueError):
    pass


@dataclass(frozen=True)
class Seed:
    frame_index: int
    box_xywh: tuple[float, float, float, float]
    positive_points_xy: tuple[tuple[float, float], ...]
    negative_points_xy: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class InstancePrompt:
    instance_id: str
    role: str
    text: str
    primary: Seed
    fallback: Seed | None
    minimum_area: int
    maximum_area_fraction: float
    area_ratio_range: tuple[float, float]
    maximum_components: int
    maximum_centroid_jump_px: float
    physical_identity_policy: str


def _seed(frame: int, box: tuple[int, int, int, int], point: tuple[int, int],
          negatives: tuple[tuple[int, int], ...] = ()) -> Seed:
    return Seed(
        frame_index=frame,
        box_xywh=tuple(float(value) for value in box),
        positive_points_xy=((float(point[0]), float(point[1])),),
        negative_points_xy=tuple((float(x), float(y)) for x, y in negatives),
    )


# The boxes were reviewed on the resize-only frames 30 and 60.  They are kept
# here rather than inferred from a remapped legacy frame.  One box initializes
# one physical/role instance, so the three cards can never collapse into a
# class-level union mask.
FROZEN_INSTANCE_PROMPTS: tuple[InstancePrompt, ...] = (
    InstancePrompt("left_hand_00", "left_hand", "hand",
        _seed(60, (276, 730, 305, 230), (455, 805), ((700, 735),)),
        _seed(30, (302, 700, 285, 260), (470, 790), ((720, 910),)),
        2500, 0.22, (0.18, 5.0), 6, 175.0, "SIDE_LOCKED_HUMAN_ROLE"),
    InstancePrompt("right_hand_00", "right_hand", "hand",
        _seed(60, (585, 600, 245, 335), (700, 750), ((460, 805),)),
        _seed(30, (635, 835, 220, 125), (730, 915), ((470, 790),)),
        2200, 0.22, (0.18, 5.0), 6, 175.0, "SIDE_LOCKED_HUMAN_ROLE"),
    InstancePrompt("left_forearm_00", "left_forearm", "forearm",
        _seed(60, (150, 855, 335, 105), (285, 930), ((475, 795),)),
        _seed(30, (260, 840, 235, 120), (335, 925), ((500, 790),)),
        900, 0.20, (0.12, 7.0), 4, 190.0, "SIDE_LOCKED_HUMAN_ROLE"),
    InstancePrompt("right_forearm_00", "right_forearm", "forearm",
        _seed(60, (655, 825, 235, 135), (760, 930), ((700, 735),)),
        _seed(30, (650, 865, 230, 95), (760, 930), ((720, 900),)),
        900, 0.20, (0.12, 7.0), 4, 190.0, "SIDE_LOCKED_HUMAN_ROLE"),
    InstancePrompt("left_finger_sleeve_cluster_00", "left_finger_sleeve", "finger sleeve",
        _seed(60, (410, 680, 175, 270), (515, 760)),
        _seed(30, (430, 690, 160, 250), (525, 765)),
        180, 0.10, (0.08, 10.0), 12, 220.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT"),
    InstancePrompt("right_finger_sleeve_cluster_00", "right_finger_sleeve", "finger sleeve",
        _seed(60, (575, 590, 245, 285), (695, 680)),
        _seed(30, (640, 835, 210, 125), (720, 885)),
        180, 0.10, (0.08, 10.0), 12, 220.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT"),
    InstancePrompt("left_cable_00", "left_cable", "cable",
        _seed(60, (345, 690, 245, 255), (410, 820)),
        _seed(30, (375, 720, 210, 220), (420, 820)),
        45, 0.08, (0.03, 16.0), 20, 260.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT"),
    InstancePrompt("right_cable_00", "right_cable", "cable",
        _seed(60, (565, 620, 285, 330), (805, 780)),
        _seed(30, (775, 865, 100, 95), (815, 910)),
        45, 0.08, (0.03, 16.0), 20, 260.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT"),
    InstancePrompt("playing_card_00", "task_object", "playing card",
        _seed(30, (482, 564, 108, 125), (536, 625)), None,
        1000, 0.05, (0.35, 2.8), 3, 90.0, "SEPARATE_VISIBLE_PHYSICAL_OBJECT"),
    InstancePrompt("playing_card_01", "task_object", "playing card",
        _seed(30, (586, 570, 100, 132), (636, 635)), None,
        1000, 0.05, (0.35, 2.8), 3, 90.0, "SEPARATE_VISIBLE_PHYSICAL_OBJECT"),
    InstancePrompt("playing_card_02", "task_object", "playing card",
        _seed(30, (678, 582, 105, 138), (731, 648)), None,
        1000, 0.05, (0.30, 3.0), 3, 110.0, "SEPARATE_VISIBLE_PHYSICAL_OBJECT"),
)


def validate_seed(seed: Seed) -> None:
    if not 0 <= seed.frame_index < FRAME_COUNT:
        raise StrictRoleContractError("seed frame is outside the fixed canary")
    x, y, w, h = seed.box_xywh
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT:
        raise StrictRoleContractError(f"invalid seed box: {seed.box_xywh}")
    for px, py in (*seed.positive_points_xy, *seed.negative_points_xy):
        if not 0 <= px < WIDTH or not 0 <= py < HEIGHT:
            raise StrictRoleContractError("seed point is outside the resize-only image")


def validate_instance_prompts(prompts: tuple[InstancePrompt, ...] = FROZEN_INSTANCE_PROMPTS) -> None:
    ids: set[str] = set()
    card_ids = []
    for prompt in prompts:
        if prompt.instance_id in ids:
            raise StrictRoleContractError(f"duplicate instance: {prompt.instance_id}")
        ids.add(prompt.instance_id)
        if prompt.role not in ROLE_NAMES:
            raise StrictRoleContractError(f"unknown role: {prompt.role}")
        if any(token in prompt.instance_id.lower() for token in FORBIDDEN_IDENTITY_TOKENS):
            raise StrictRoleContractError("forbidden telemetry identity in visual role")
        if " or " in prompt.text.lower() or len(prompt.text.split()) > 3:
            raise StrictRoleContractError("role prompts must be short noun phrases")
        validate_seed(prompt.primary)
        if prompt.fallback is not None:
            validate_seed(prompt.fallback)
        if prompt.role == "task_object":
            card_ids.append(prompt.instance_id)
    if len(card_ids) != 3 or len(set(card_ids)) != 3:
        raise StrictRoleContractError("the three visible cards must stay independent")


def build_prompt_plan(video_sha256: str, hawor_sha256: str) -> dict[str, Any]:
    validate_instance_prompts()
    return {
        "schema_version": "sam31-0915-strict-role-prompt-plan-v1",
        "session_id": SESSION_ID,
        "model_identity": "SAM3.1_ONLY_USER_LOCKED",
        "image_domain": {
            "identity": IMAGE_DOMAIN, "width": WIDTH, "height": HEIGHT,
            "rectified": False, "remap_applied": False,
        },
        "prompt_api": {
            "contract_name": "initial_box_prompt",
            "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
            "runtime_semantics": "MULTIPLEX_GEOMETRIC_BOX",
            "separate_exemplar_model": False,
            "source_note": (
                "The pinned multiplex class overrides the base video API visual-prompt "
                "branch; the first box is recorded as a geometric role initializer."
            ),
        },
        "point_policy": {
            "usage": "SEED_SELECTION_AND_QUALITY_EVIDENCE_ONLY",
            "refinement_status": "HELD_AFTER_V3_PARTIAL_ROUTE_CONTINUITY_FAILURE",
            "claim_limit": (
                "Points do not alter mask pixels in this canary. The pinned multiplex "
                "point-refinement path switches from full semantic propagation to a "
                "partial SAM2 route and requires a separate continuity repair."
            ),
        },
        "instances": [asdict(prompt) for prompt in FROZEN_INSTANCE_PROMPTS],
        "temporal_states": list(TEMPORAL_STATES),
        "reseed_policy": {
            "type": "QUALITY_TRIGGERED_NOT_PERIODIC",
            "maximum_reseeds_per_instance": 1,
            "minimum_unknown_run_to_trigger": 3,
            "minimum_unknown_fraction_to_trigger": 0.08,
            "merge": "PRIMARY_IF_VALID_ELSE_FALLBACK_IF_VALID_ELSE_UNKNOWN",
        },
        "derived_clean_union": list(REMOVAL_ROLES),
        "inputs": {"video_sha256": video_sha256, "hawor_sha256": hawor_sha256},
        "tracker_role_created": False,
        "controller_role_created": False,
        "pico26_consumed": False,
        "claim_limit": (
            "Frozen one-session development prompts, not pixel truth. Initial boxes "
            "establish role instances and do not create an exemplar subsystem."
        ),
    }


def longest_false_run(valid: np.ndarray) -> int:
    array = np.asarray(valid, bool).reshape(-1)
    best = current = 0
    for value in array:
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def should_reseed(valid: np.ndarray, *, has_fallback: bool) -> tuple[bool, list[str]]:
    array = np.asarray(valid, bool).reshape(-1)
    reasons = []
    if not has_fallback:
        return False, reasons
    unknown_fraction = float(np.mean(~array)) if len(array) else 1.0
    if longest_false_run(array) >= 3:
        reasons.append("UNKNOWN_RUN_GE_3")
    if unknown_fraction >= 0.08:
        reasons.append("UNKNOWN_FRACTION_GE_0_08")
    return bool(reasons), reasons
