"""Frozen weak-role prompt contract for ``play_cards_0915_001``.

This successor deliberately does not rerun the two hand instances or the first
two playing-card instances.  Those four masks are immutable regression inputs
from the completed v5 canary.  The only model targets here are the roles that
were weak in v5: two forearms, individually boxed visible finger sleeves, two
visible yellow-cable segments, and ``playing_card_02``.

Boxes are development prompts in the confirmed resize-only physical-left
image domain.  They are not annotations or pixel truth.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from chaoyang.pipeline.sam31_0915_strict_role_contract_v1 import (
    FRAME_COUNT,
    HEIGHT,
    IMAGE_DOMAIN,
    SESSION_ID,
    TEMPORAL_STATES,
    WIDTH,
    InstancePrompt,
    Seed,
    StrictRoleContractError,
    validate_seed,
)


TASK_ID = "0915_sam31_weak_role_canary_v1"
PHASE = "0915_SAM31_WEAK_ROLE_SINGLE_SESSION_CANARY_V1"
MODEL_WEIGHT = "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
WEIGHTS = (MODEL_WEIGHT,)
RESEED_POLICY = "QUALITY_TRIGGERED_NOT_PERIODIC"
REGRESSION_INPUT_POLICY = "READ_ONLY_SHA_GUARDED_NO_RECOMPUTE"

TARGET_ROLES = (
    "left_forearm",
    "right_forearm",
    "left_finger_sleeve",
    "right_finger_sleeve",
    "left_cable",
    "right_cable",
    "task_object",
)
REGRESSION_INSTANCE_IDS = (
    "left_hand_00",
    "right_hand_00",
    "playing_card_00",
    "playing_card_01",
)
TARGET_INSTANCE_IDS = (
    "left_forearm_00",
    "right_forearm_00",
    "left_finger_sleeve_visible_00",
    "left_finger_sleeve_visible_01",
    "left_finger_sleeve_visible_02",
    "left_finger_sleeve_visible_03",
    "right_finger_sleeve_visible_00",
    "right_finger_sleeve_visible_01",
    "right_finger_sleeve_visible_02",
    "right_finger_sleeve_visible_03",
    "left_yellow_cable_visible_00",
    "right_yellow_cable_visible_00",
    "playing_card_02",
)


def _seed(
    frame: int,
    box: tuple[int, int, int, int],
    point: tuple[int, int],
    negatives: tuple[tuple[int, int], ...] = (),
) -> Seed:
    return Seed(
        frame_index=frame,
        box_xywh=tuple(float(value) for value in box),
        positive_points_xy=((float(point[0]), float(point[1])),),
        negative_points_xy=tuple((float(x), float(y)) for x, y in negatives),
    )


# Sleeve IDs are intentionally ordinal, not anatomical finger labels: the
# image supports separate visible instances but does not by itself establish a
# reliable digit identity.  Each box encloses one sleeve, never a whole hand.
FROZEN_WEAK_ROLE_PROMPTS: tuple[InstancePrompt, ...] = (
    InstancePrompt(
        "left_forearm_00", "left_forearm", "forearm",
        _seed(30, (150, 850, 245, 110), (260, 920), ((475, 790),)),
        _seed(60, (155, 850, 230, 110), (270, 920), ((500, 790),)),
        700, 0.12, (0.12, 7.0), 4, 190.0, "SIDE_LOCKED_HUMAN_ROLE",
    ),
    InstancePrompt(
        "right_forearm_00", "right_forearm", "forearm",
        _seed(60, (735, 850, 125, 110), (790, 925), ((680, 750),)),
        _seed(90, (705, 855, 130, 105), (765, 920), ((670, 760),)),
        700, 0.15, (0.12, 7.0), 4, 190.0, "SIDE_LOCKED_HUMAN_ROLE",
    ),
    InstancePrompt(
        "left_finger_sleeve_visible_00", "left_finger_sleeve", "finger sleeve",
        _seed(60, (438, 742, 72, 82), (475, 773), ((430, 835),)),
        _seed(30, (463, 704, 78, 82), (505, 740), ((450, 805),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "left_finger_sleeve_visible_01", "left_finger_sleeve", "finger sleeve",
        _seed(60, (472, 779, 72, 80), (510, 812), ((455, 870),)),
        _seed(30, (491, 749, 66, 84), (525, 785), ((472, 845),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "left_finger_sleeve_visible_02", "left_finger_sleeve", "finger sleeve",
        _seed(60, (485, 824, 74, 82), (522, 858), ((465, 920),)),
        _seed(30, (494, 794, 68, 82), (528, 828), ((470, 890),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "left_finger_sleeve_visible_03", "left_finger_sleeve", "finger sleeve",
        _seed(60, (491, 868, 82, 88), (532, 906), ((455, 930),)),
        _seed(30, (474, 850, 78, 100), (515, 895), ((450, 820),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "right_finger_sleeve_visible_00", "right_finger_sleeve", "finger sleeve",
        _seed(60, (607, 735, 70, 88), (642, 778), ((700, 790),)),
        _seed(90, (641, 770, 70, 91), (675, 813), ((730, 830),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "right_finger_sleeve_visible_01", "right_finger_sleeve", "finger sleeve",
        _seed(60, (678, 624, 64, 84), (710, 665), ((670, 730),)),
        _seed(90, (650, 674, 68, 91), (684, 716), ((635, 790),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "right_finger_sleeve_visible_02", "right_finger_sleeve", "finger sleeve",
        _seed(60, (717, 640, 64, 90), (749, 682), ((705, 760),)),
        _seed(90, (690, 682, 66, 94), (723, 727), ((680, 805),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "right_finger_sleeve_visible_03", "right_finger_sleeve", "finger sleeve",
        _seed(60, (751, 676, 66, 91), (784, 719), ((740, 790),)),
        _seed(90, (726, 697, 68, 96), (760, 744), ((715, 820),)),
        120, 0.012, (0.06, 12.0), 4, 95.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "left_yellow_cable_visible_00", "left_cable", "yellow cable",
        _seed(60, (365, 778, 58, 132), (395, 841), ((440, 835),)),
        _seed(30, (410, 736, 58, 140), (440, 810), ((485, 805),)),
        35, 0.012, (0.02, 20.0), 12, 180.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "right_yellow_cable_visible_00", "right_cable", "yellow cable",
        _seed(60, (782, 746, 54, 142), (809, 818), ((755, 820),)),
        _seed(90, (785, 708, 56, 145), (814, 780), ((760, 805),)),
        35, 0.012, (0.02, 20.0), 12, 180.0, "ROLE_INSTANCE_UNKNOWN_AFTER_EXIT",
    ),
    InstancePrompt(
        "playing_card_02", "task_object", "playing card",
        _seed(30, (678, 582, 105, 138), (731, 648), ((810, 650),)),
        _seed(90, (638, 620, 102, 158), (683, 653), ((760, 720),)),
        650, 0.05, (0.20, 4.0), 3, 125.0, "SEPARATE_VISIBLE_PHYSICAL_OBJECT",
    ),
)


def validate_weak_role_prompts(
    prompts: tuple[InstancePrompt, ...] = FROZEN_WEAK_ROLE_PROMPTS,
) -> None:
    ids = tuple(prompt.instance_id for prompt in prompts)
    if ids != TARGET_INSTANCE_IDS:
        raise StrictRoleContractError("weak-role target list or ordering drift")
    if len(ids) != len(set(ids)):
        raise StrictRoleContractError("duplicate weak-role instance")
    for prompt in prompts:
        if prompt.role not in TARGET_ROLES:
            raise StrictRoleContractError(f"unexpected weak role: {prompt.role}")
        if prompt.instance_id in REGRESSION_INSTANCE_IDS:
            raise StrictRoleContractError("regression instance must not be recomputed")
        lowered = f"{prompt.instance_id} {prompt.role} {prompt.text}".lower()
        if any(token in lowered for token in ("tracker", "controller", "pico")):
            raise StrictRoleContractError("telemetry identity cannot become a visual role")
        if len(prompt.text.split()) > 2 or " or " in prompt.text.lower():
            raise StrictRoleContractError("weak-role text must be a short noun phrase")
        validate_seed(prompt.primary)
        if prompt.fallback is None:
            raise StrictRoleContractError("every weak target needs one bounded fallback")
        validate_seed(prompt.fallback)
        if "finger_sleeve" in prompt.role:
            for seed in (prompt.primary, prompt.fallback):
                _x, _y, width, height = seed.box_xywh
                if width > 86 or height > 105 or width * height > 8_000:
                    raise StrictRoleContractError("sleeve box can swallow more than one finger")
        if prompt.role.endswith("_forearm"):
            for seed in (prompt.primary, prompt.fallback):
                _x, _y, width, height = seed.box_xywh
                if width > 260 or height > 140 or width * height > 30_000:
                    raise StrictRoleContractError("forearm box can swallow the whole hand")
        if prompt.role.endswith("_cable"):
            for seed in (prompt.primary, prompt.fallback):
                _x, _y, width, height = seed.box_xywh
                if width > 60 or width * height > 8_500:
                    raise StrictRoleContractError("cable box is not a tight visible-segment box")


def regression_paths(regression_root: Path) -> tuple[Path, ...]:
    """Return the complete immutable v5 read set used by this successor."""

    return (
        regression_root / "ROLE_MANIFEST.json",
        regression_root / "TEMPORAL_STATE_LEDGER.json",
        *(regression_root / "masks" / f"{name}.npz"
          for name in REGRESSION_INSTANCE_IDS),
    )


def build_prompt_plan(
    *, video_sha256: str, hawor_sha256: str,
    regression_sha256: dict[str, str],
) -> dict[str, Any]:
    validate_weak_role_prompts()
    return {
        "schema_version": "sam31-0915-weak-role-prompt-plan-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "model_identity": "SAM3.1_ONLY_USER_LOCKED",
        "weights": list(WEIGHTS),
        "image_domain": {
            "identity": IMAGE_DOMAIN,
            "width": WIDTH,
            "height": HEIGHT,
            "rectified": False,
            "remap_applied": False,
        },
        "prompt_api": {
            "contract_name": "initial_visual_box",
            "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
            "runtime_semantics": "MULTIPLEX_GEOMETRIC_BOX",
            "separate_exemplar_model": False,
        },
        "target_instances": [asdict(prompt) for prompt in FROZEN_WEAK_ROLE_PROMPTS],
        "regression_instances": list(REGRESSION_INSTANCE_IDS),
        "regression_input_policy": REGRESSION_INPUT_POLICY,
        "temporal_states": list(TEMPORAL_STATES),
        "reseed_policy": {
            "type": RESEED_POLICY,
            "maximum_reseeds_per_instance": 1,
            "minimum_unknown_run_to_trigger": 3,
            "minimum_unknown_fraction_to_trigger": 0.08,
            "merge": "PRIMARY_IF_VALID_ELSE_FALLBACK_IF_VALID_ELSE_UNKNOWN",
        },
        "inputs": {
            "video_sha256": video_sha256,
            "hawor_sha256": hawor_sha256,
            "regression_sha256": regression_sha256,
        },
        "tracker_role_created": False,
        "controller_role_created": False,
        "pico26_consumed": False,
        "claim_limit": (
            "One-session development prompts for weak-role review. Ordinal sleeve "
            "instances are separate visible regions, not anatomical digit truth. "
            "Unknown is not absence; no batch, contact, or deployment authority."
        ),
    }
