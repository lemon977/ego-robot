from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops.run_0915_sam31_strict_role_canary_v1 import _collect_bidirectional
from chaoyang.pipeline.sam31_0915_strict_role_contract_v1 import (
    FRAME_COUNT,
    FROZEN_INSTANCE_PROMPTS,
    IMAGE_DOMAIN,
    REMOVAL_ROLES,
    ROLE_NAMES,
    build_prompt_plan,
    should_reseed,
    validate_instance_prompts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_plan_is_resize_only_and_matches_pinned_multiplex_semantics() -> None:
    plan = build_prompt_plan("0" * 64, "1" * 64)
    assert plan["image_domain"] == {
        "identity": IMAGE_DOMAIN,
        "width": 1280,
        "height": 960,
        "rectified": False,
        "remap_applied": False,
    }
    assert plan["prompt_api"]["contract_name"] == "initial_box_prompt"
    assert plan["prompt_api"]["runtime_semantics"] == "MULTIPLEX_GEOMETRIC_BOX"
    assert plan["prompt_api"]["separate_exemplar_model"] is False


def test_roles_are_split_and_task_objects_stay_independent() -> None:
    validate_instance_prompts()
    roles = {prompt.role for prompt in FROZEN_INSTANCE_PROMPTS}
    assert roles == set(ROLE_NAMES)
    assert set(REMOVAL_ROLES) == roles - {"task_object"}
    cards = [prompt for prompt in FROZEN_INSTANCE_PROMPTS
             if prompt.role == "task_object"]
    assert [prompt.instance_id for prompt in cards] == [
        "playing_card_00", "playing_card_01", "playing_card_02"
    ]
    assert all(" or " not in prompt.text for prompt in FROZEN_INSTANCE_PROMPTS)
    assert all(
        token not in prompt.instance_id.lower()
        for prompt in FROZEN_INSTANCE_PROMPTS
        for token in ("tracker", "controller", "pico")
    )


def test_reseed_is_quality_triggered_and_bounded_not_periodic() -> None:
    valid = np.ones(FRAME_COUNT, bool)
    assert should_reseed(valid, has_fallback=True) == (False, [])
    valid[70:73] = False
    triggered, reasons = should_reseed(valid, has_fallback=True)
    assert triggered is True
    assert "UNKNOWN_RUN_GE_3" in reasons
    assert should_reseed(valid, has_fallback=False) == (False, [])


def test_direction_no_points_exception_becomes_unknown_evidence() -> None:
    class Model:
        def propagate_in_video(self, **kwargs):
            if kwargs["reverse"]:
                raise RuntimeError("No points are provided; please add points first")
            mask = np.zeros((1, 960, 1280), bool)
            mask[0, 10:20, 10:20] = True
            yield 60, {
                "out_binary_masks": mask,
                "out_probs": np.asarray([0.9]),
                "out_obj_ids": np.asarray([7]),
            }

    masks, evidence = _collect_bidirectional(Model(), {}, anchor=60, raw_id=7)
    assert masks[60].sum() == 100
    assert evidence == [
        {"direction": "forward", "status": "COMPLETE", "frames_yielded": 1},
        {
            "direction": "backward",
            "status": "UNKNOWN_DIRECTION_TRACKER_HAS_NO_CONFIRMED_INSTANCE",
            "frames_yielded_before_hold": 0,
            "reason": "No points are provided; please add points first",
        },
    ]


def test_visual_role_mask_v2_accepts_only_explicit_temporal_counts() -> None:
    schema = json.loads(
        (ROOT / "contracts/visual_role_mask_v2.schema.json").read_text(encoding="utf-8")
    )
    instances = []
    for prompt in FROZEN_INSTANCE_PROMPTS:
        instances.append({
            "instance_id": prompt.instance_id,
            "role": prompt.role,
            "physical_identity_policy": prompt.physical_identity_policy,
            "mask_archive": f"masks/{prompt.instance_id}.npz",
            "state_counts": {
                "seeded": 1, "tracked": FRAME_COUNT - 1,
                "reseeded": 0, "unknown": 0,
            },
            "initial_box_prompt": {
                "frame_index": prompt.primary.frame_index,
                "box_xywh": list(prompt.primary.box_xywh),
            },
            "reseed_count": 0,
        })
    manifest = {
        "schema_version": "visual-role-mask-v2",
        "session_id": "play_cards_0915_001", "frame_count": FRAME_COUNT,
        "image_domain": IMAGE_DOMAIN,
        "model": {
            "identity": "SAM3.1", "weight_sha256": "0" * 64,
            "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
            "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
        },
        "roles": list(ROLE_NAMES), "instances": instances,
        "derived_masks": {
            "human_equipment_union": {
                "mask_archive": "masks/HUMAN_EQUIPMENT_UNION.npz",
                "source_roles": list(REMOVAL_ROLES),
                "direction": "DERIVED_FROM_ROLE_MASKS_ONLY",
            }
        },
        "temporal_state_ledger": "TEMPORAL_STATE_LEDGER.json",
        "tracker_role_created": False, "controller_role_created": False,
        "pico26_consumed": False,
    }
    jsonschema.Draft202012Validator(schema).validate(manifest)


def test_current_task_packet_is_single_session_one_weight_and_exact_read_set() -> None:
    packet = build_packet("0915_sam31_strict_role_canary_v5")
    assert packet["weights"] == [
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
    ]
    assert len(packet["read_set"]) == 8
    encoded = json.dumps(packet, ensure_ascii=False)
    assert "play_cards_0915_001" in encoded
    assert "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz" in encoded
    assert "no_batch_expansion" in encoded
    assert "multiplex_semantic_full_propagation" in encoded
    assert "direction_level_fail_closed_unknown" in encoded
