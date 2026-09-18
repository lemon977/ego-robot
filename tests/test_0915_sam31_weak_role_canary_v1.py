from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops.run_0915_sam31_weak_role_canary_v1 import (
    CENTRAL_GPU_LEASE,
    OUTPUT_NAMESPACE,
    assert_regression_inputs_unchanged,
    snapshot_regression_inputs,
    validate_output_namespace,
)
from chaoyang.pipeline.sam31_0915_weak_role_contract_v1 import (
    FROZEN_WEAK_ROLE_PROMPTS,
    IMAGE_DOMAIN,
    MODEL_WEIGHT,
    REGRESSION_INPUT_POLICY,
    REGRESSION_INSTANCE_IDS,
    RESEED_POLICY,
    TARGET_INSTANCE_IDS,
    TASK_ID,
    WEIGHTS,
    build_prompt_plan,
    regression_paths,
    validate_weak_role_prompts,
)


def test_target_roles_are_exactly_weak_roles_and_do_not_recompute_regressions() -> None:
    validate_weak_role_prompts()
    ids = tuple(prompt.instance_id for prompt in FROZEN_WEAK_ROLE_PROMPTS)
    assert ids == TARGET_INSTANCE_IDS
    assert set(ids).isdisjoint(REGRESSION_INSTANCE_IDS)
    assert REGRESSION_INSTANCE_IDS == (
        "left_hand_00", "right_hand_00", "playing_card_00", "playing_card_01",
    )
    assert sum("left_finger_sleeve_visible" in name for name in ids) == 4
    assert sum("right_finger_sleeve_visible" in name for name in ids) == 4
    assert all("cluster" not in name for name in ids)
    assert ids[-1] == "playing_card_02"


def test_per_finger_and_cable_boxes_are_tight_and_text_is_short() -> None:
    for prompt in FROZEN_WEAK_ROLE_PROMPTS:
        assert len(prompt.text.split()) <= 2
        assert " or " not in prompt.text
        for seed in (prompt.primary, prompt.fallback):
            assert seed is not None
            _x, _y, width, height = seed.box_xywh
            if "finger_sleeve" in prompt.role:
                assert width <= 86
                assert height <= 105
                assert width * height <= 8_000
            if prompt.role.endswith("_forearm"):
                assert width <= 260
                assert height <= 140
                assert width * height <= 30_000
            if prompt.role.endswith("_cable"):
                assert prompt.text == "yellow cable"
                assert width <= 60
                assert width * height <= 8_500


def test_plan_is_sam31_only_resize_only_and_has_no_periodic_reseed() -> None:
    plan = build_prompt_plan(
        video_sha256="0" * 64,
        hawor_sha256="1" * 64,
        regression_sha256={"ROLE_MANIFEST.json": "2" * 64},
    )
    assert TASK_ID == "0915_sam31_weak_role_canary_v1"
    assert WEIGHTS == (MODEL_WEIGHT,)
    assert len(plan["weights"]) == 1
    assert plan["model_identity"] == "SAM3.1_ONLY_USER_LOCKED"
    assert plan["image_domain"] == {
        "identity": IMAGE_DOMAIN,
        "width": 1280,
        "height": 960,
        "rectified": False,
        "remap_applied": False,
    }
    assert plan["prompt_api"]["contract_name"] == "initial_visual_box"
    assert plan["prompt_api"]["separate_exemplar_model"] is False
    assert plan["reseed_policy"]["type"] == RESEED_POLICY
    assert plan["reseed_policy"]["maximum_reseeds_per_instance"] == 1
    assert not any("period" in key.lower() for key in plan["reseed_policy"])
    assert plan["temporal_states"] == ["seeded", "tracked", "reseeded", "unknown"]


def test_plan_creates_no_tracker_controller_or_pico_role() -> None:
    plan = build_prompt_plan(
        video_sha256="0" * 64,
        hawor_sha256="1" * 64,
        regression_sha256={},
    )
    encoded_targets = json.dumps(plan["target_instances"], ensure_ascii=False).lower()
    assert "tracker" not in encoded_targets
    assert "controller" not in encoded_targets
    assert "pico" not in encoded_targets
    assert plan["tracker_role_created"] is False
    assert plan["controller_role_created"] is False
    assert plan["pico26_consumed"] is False


def test_regression_read_set_is_sha_guarded_and_mutation_is_detected(tmp_path: Path) -> None:
    for path in regression_paths(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"immutable:{path.name}".encode())
    before = snapshot_regression_inputs(tmp_path)
    assert set(before) == {
        "ROLE_MANIFEST.json",
        "TEMPORAL_STATE_LEDGER.json",
        "masks/left_hand_00.npz",
        "masks/right_hand_00.npz",
        "masks/playing_card_00.npz",
        "masks/playing_card_01.npz",
    }
    assert REGRESSION_INPUT_POLICY == "READ_ONLY_SHA_GUARDED_NO_RECOMPUTE"
    assert_regression_inputs_unchanged(before, tmp_path)
    (tmp_path / "masks/left_hand_00.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="read-only v5 regression evidence changed"):
        assert_regression_inputs_unchanged(before, tmp_path)


def test_runner_has_own_output_namespace_and_central_gpu_lease() -> None:
    assert OUTPUT_NAMESPACE.name == TASK_ID
    validate_output_namespace(OUTPUT_NAMESPACE / "attempts/attempt_0001")
    with pytest.raises(RuntimeError, match="task namespace"):
        validate_output_namespace(OUTPUT_NAMESPACE.parent / "another_task/attempt_0001")
    assert CENTRAL_GPU_LEASE.name == "run_gpu_command_with_v71_lease.py"
