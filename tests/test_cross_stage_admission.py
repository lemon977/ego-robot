from copy import deepcopy
import json
from pathlib import Path

import jsonschema

from pipeline.cross_stage_admission import (
    evaluate_cross_stage,
    validate_cross_stage_input,
    verify_evidence_files,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/cross_stage_admission_v1"


def gate(status="PASS", authority="PRODUCER", *reasons):
    return {"status": status, "authority": authority, "reason_codes": list(reasons)}


def provider(name, status="PASS", authority="INDEPENDENT_QA", *reasons):
    return {"provider": name, "eligibility": gate(status, authority, *reasons)}


def stage(gates, *, scope="FULL_SESSION", authorized=None, **extra):
    result = {
        "scope": scope,
        "reported_status": "PASS",
        "claim_limit": "Unit-test full-session fixture.",
        "evidence": [{"path": "/declared/evidence.json", "sha256": "0" * 64}],
        "gate_contract": {
            "id": "frozen-test-contract-v1",
            "path": "/declared/gate_contract.json",
            "sha256": "1" * 64,
            "frozen_before_evaluation": True,
        },
        "gates": gates,
        **extra,
    }
    if authorized is not None:
        result["consumption_authorized"] = authorized
    return result


def passing_bundle(profile="ROBOT_RGB_COMPOSITE_KAI22"):
    variant = {
        "RAW_RGB_KAI22": "RAW",
        "CLEAN_RGB_KAI22": "CLEAN",
        "ROBOT_RGB_COMPOSITE_KAI22": "ROBOT_COMPOSITE",
    }[profile]
    capture = {
        name: gate(authority="CAPTURE")
        for name in (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "camera_world",
            "pico21_diagnostic",
            "pico21_left_provider",
            "pico21_right_provider",
            "task_interval_defined",
            "segment_contract",
            "clean_donor_authority",
            "static_setup_anchor_authority",
            "object6d_authority",
            "session_static_base",
        )
    }
    hawor = {
        name: gate(
            authority=(
                "INDEPENDENT_QA"
                if name in {"anatomical_identity", "independent_contour"}
                else "PRODUCER"
            )
        )
        for name in (
            "execution",
            "mano21_structure",
            "numeric_mask_seed",
            "numeric_robot_seed",
            "anatomical_identity",
            "independent_contour",
            "direct_provenance",
            "contact_direct_observation",
        )
    }
    mask = {
        name: gate(authority="HUMAN_REVIEW" if name == "manual_review" else "PRODUCER")
        for name in (
            "input_authority",
            "semantic_roles",
            "moving_task_object_identity",
            "static_setup_anchor_scope",
            "prompt_provider_lineage",
            "temporal_identity",
            "full_session_coverage",
            "object_protection",
            "lineage_integrity",
            "manual_review",
        )
    }
    clean = {
        name: gate(authority="HUMAN_REVIEW" if name == "manual_review" else "PRODUCER")
        for name in (
            "input_mask_authority",
            "donor_authority",
            "donor_purity",
            "source_map",
            "spatial_residual",
            "illumination",
            "shadow",
            "seam",
            "temporal",
            "object_protection",
            "codec",
            "manual_review",
        )
    }
    robot = {
        name: gate(
            authority=(
                "INDEPENDENT_QA"
                if name in {"retarget_independent_qa", "functional_retarget_authority"}
                else "HUMAN_REVIEW"
                if name == "manual_review"
                else "PRODUCER"
            )
        )
        for name in (
            "input_hawor_authority",
            "input_clean_authority",
            "mount_calibration",
            "session_static_base",
            "object6d",
            "retarget_schema",
            "mano_reorder",
            "pose",
            "joint_limits",
            "temporal",
            "hand_morphology",
            "retarget_independent_qa",
            "functional_retarget_authority",
            "contact",
            "nonpenetration",
            "self_collision",
            "depth_occlusion",
            "manual_review",
        )
    }
    training = {
        name: gate(authority="INDEPENDENT_QA" if name == "eval_disjoint" else "PRODUCER")
        for name in (
            "split_frozen",
            "eval_disjoint",
            "task_isolated",
            "image_variant_frozen",
            "upstream_lineage_frozen",
            "selector_reproducible",
        )
    }
    return {
        "schema_version": "chaoyang-cross-stage-evidence-v1",
        "session_id": "fixture_001",
        "task_id": "generic_task",
        "training_profile": profile,
        "image_variant": variant,
        "stages": {
            "capture": stage(capture),
            "hawor": stage(hawor),
            "mask": stage(
                mask,
                authorized=True,
                seed_route="HAWOR",
                provider_routes={
                    "human_core": {
                        "left": provider("HAWOR_MANO21"),
                        "right": provider("HAWOR_MANO21"),
                    },
                    "wearable_tracker": {
                        "left": provider("VISUAL_INSTANCE"),
                        "right": provider("VISUAL_INSTANCE"),
                    },
                    "moving_task_object": {
                        "global": provider("VISUAL_INSTANCE")
                    },
                    "static_setup_anchor": {
                        "global": provider("CAPTURE_STATIC_SETUP")
                    },
                },
            ),
            "clean": stage(clean, authorized=True),
            "robot": stage(robot, authorized=True, sidecar_consumption_authorized=True),
            "training": stage(training),
        },
    }


def test_full_formal_composite_passes():
    result = evaluate_cross_stage(passing_bundle())
    assert result["input_contract"]["valid"] is True
    assert result["routes"]["humanego_train_eligible"] is True
    assert result["blame"]["primary"] == []


def test_old_data_regression_fixtures_remain_fail_closed():
    expectations = json.loads((FIXTURES / "REGRESSION_EXPECTATIONS.json").read_text())
    for name, expected in expectations.items():
        bundle = json.loads((FIXTURES / name).read_text())
        assert validate_cross_stage_input(bundle) == []
        assert verify_evidence_files(bundle, evidence_root=ROOT) == []
        result = evaluate_cross_stage(bundle)
        assert result["routes"]["humanego_train_eligible"] is expected["humanego_train_eligible"]
        if "primary_category" in expected:
            assert result["blame"]["primary"][0]["category"] == expected["primary_category"]
        for key, value in expected.items():
            if key in {"humanego_train_eligible", "primary_category"}:
                continue
            if key in result["routes"]:
                assert result["routes"][key] is value
            else:
                assert result["categories"][key]["status"] == value


def test_input_and_output_json_schemas_accept_regression_fixtures():
    input_schema = json.loads(
        (ROOT / "contracts/quality/cross_stage_admission_input_v1.schema.json").read_text()
    )
    output_schema = json.loads(
        (ROOT / "contracts/quality/cross_stage_admission_output_v1.schema.json").read_text()
    )
    for path in FIXTURES.glob("*.json"):
        if path.name.startswith("REGRESSION_"):
            continue
        bundle = json.loads(path.read_text())
        jsonschema.validate(bundle, input_schema)
        jsonschema.validate(evaluate_cross_stage(bundle), output_schema)


def test_pico_failure_is_advisory_for_rgb_hawor_and_manual_mask():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["pico21_diagnostic"] = gate(
        "FAIL", "CAPTURE", "PICO_HAND_JUMP"
    )
    bundle["stages"]["mask"]["seed_route"] = "MANUAL"
    bundle["stages"]["mask"]["provider_routes"]["human_core"] = {
        "left": provider("MANUAL"),
        "right": provider("MANUAL"),
    }
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["hawor_rgb_execution_allowed"] is True
    assert result["routes"]["mask_execution_allowed"] is True
    assert result["categories"]["mask_semantic"]["status"] == "PASS"


def test_pico_failure_blocks_only_explicit_pico_raw_point_seed():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["pico21_left_provider"] = gate(
        "FAIL", "CAPTURE", "PICO_LEFT_WRIST_STEP_313MM"
    )
    bundle["stages"]["mask"]["seed_route"] = "PICO_RAW_POINT"
    bundle["stages"]["mask"]["provider_routes"]["human_core"] = {
        "left": provider("PICO_OPENXR21"),
        "right": provider("PICO_OPENXR21"),
    }
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["mask_semantic"]["status"] == "HOLD"
    assert result["routes"]["mask_formal_ready"] is False


def test_camera_world_hold_does_not_block_image_mask():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["camera_world"] = gate(
        "HOLD", "CAPTURE", "WORLD_EPOCH_UNPROVEN"
    )
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["mask_execution_allowed"] is True
    assert result["routes"]["robot_motion_execution_allowed"] is False


def test_canary_pass_and_consumption_false_cannot_promote_mask():
    bundle = passing_bundle()
    bundle["stages"]["mask"]["scope"] = "CANARY"
    bundle["stages"]["mask"]["consumption_authorized"] = False
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["mask_semantic"]["status"] == "HOLD"
    assert result["categories"]["mask_temporal"]["status"] == "HOLD"
    assert result["routes"]["mask_formal_ready"] is False


def test_producer_cannot_self_approve_independent_hawor_identity():
    bundle = passing_bundle()
    bundle["stages"]["hawor"]["gates"]["anatomical_identity"]["authority"] = "PRODUCER"
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["hawor_algorithm"]["status"] == "HOLD"
    assert result["routes"]["humanego_train_eligible"] is False


def test_raw_profile_does_not_require_mask_clean_or_object_contact():
    bundle = passing_bundle("RAW_RGB_KAI22")
    del bundle["stages"]["mask"]
    del bundle["stages"]["clean"]
    for name in (
        "object6d_authority",
        "segment_contract",
        "clean_donor_authority",
    ):
        bundle["stages"]["capture"]["gates"][name] = gate(
            "HOLD", "CAPTURE", "NOT_REQUIRED_BY_RAW_PROFILE"
        )
    robot = bundle["stages"]["robot"]
    robot["consumption_authorized"] = False
    for name in (
        "input_clean_authority",
        "mount_calibration",
        "object6d",
        "contact",
        "nonpenetration",
        "self_collision",
        "depth_occlusion",
        "manual_review",
    ):
        robot["gates"][name] = gate("HOLD", "PRODUCER", "NOT_REQUIRED_BY_RAW_SIDECAR_PROFILE")
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["robot_sidecar_ready"] is True
    assert result["routes"]["robot_formal_ready"] is False
    assert result["routes"]["humanego_train_eligible"] is True


def test_same_partial_products_do_not_pass_clean_profile():
    bundle = passing_bundle("RAW_RGB_KAI22")
    del bundle["stages"]["mask"]
    del bundle["stages"]["clean"]
    bundle["training_profile"] = "CLEAN_RGB_KAI22"
    bundle["image_variant"] = "CLEAN"
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["humanego_train_eligible"] is False


def test_downstream_failure_is_visible_but_not_primary_before_upstream_pass():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["object6d_authority"] = gate(
        "HOLD", "CAPTURE", "OBJECT6D_MISSING"
    )
    bundle["stages"]["robot"]["gates"]["hand_morphology"] = gate(
        "FAIL", "PRODUCER", "HAND_SHAPE_FAILED"
    )
    result = evaluate_cross_stage(bundle)
    assert "robot_geometry" in result["blame"]["observed_failures"]
    assert result["blame"]["primary"][0]["category"] == "upstream_capture"
    assert {
        (row["category"], row["route"])
        for row in result["blame"]["primary"]
    } >= {
        ("upstream_capture", "robot_composite_capture_input"),
        ("robot_geometry", "motion_sidecar"),
    }
    assert any(
        row["category"] == "robot_geometry"
        for row in result["blame"]["suppressed_downstream_failures"]
    )


def test_parallel_object6d_hold_does_not_hide_independent_mask_failure():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["object6d_authority"] = gate(
        "HOLD", "CAPTURE", "OBJECT6D_MISSING"
    )
    bundle["stages"]["mask"]["gates"]["semantic_roles"] = gate(
        "FAIL", "INDEPENDENT_QA", "TRACKER_LEAK"
    )
    result = evaluate_cross_stage(bundle)
    primary = {(row["category"], row["route"]) for row in result["blame"]["primary"]}
    assert ("upstream_capture", "robot_composite_capture_input") in primary
    assert ("mask_semantic", "formal_semantic") in primary
    assert not any(
        row["category"] == "mask_semantic"
        for row in result["blame"]["suppressed_downstream_failures"]
    )


def test_gate_contract_must_be_frozen_before_evaluation():
    bundle = passing_bundle()
    bundle["stages"]["mask"]["gate_contract"]["frozen_before_evaluation"] = False
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["mask_semantic"]["status"] == "HOLD"
    assert result["routes"]["mask_formal_ready"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_missing_claim_metadata_is_invalid_and_fail_closed():
    bundle = passing_bundle()
    del bundle["stages"]["robot"]["claim_limit"]
    result = evaluate_cross_stage(bundle)
    assert result["input_contract"]["valid"] is False
    assert any("stages.robot.claim_limit" in error for error in result["input_contract"]["schema_errors"])
    assert result["routes"]["humanego_train_eligible"] is False


def test_explicit_training_failure_remains_visible_with_other_upstream_holds():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["gates"]["object6d_authority"] = gate(
        "HOLD", "CAPTURE", "OBJECT6D_MISSING"
    )
    bundle["stages"]["training"]["gates"]["split_frozen"] = gate(
        "FAIL", "INDEPENDENT_QA", "SPLIT_LEAK"
    )
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["training_eligibility"]["status"] == "FAIL"
    assert "training_eligibility" in result["blame"]["observed_failures"]
    assert any(
        row["category"] == "training_eligibility" for row in result["blame"]["primary"]
    )
    assert result["routes"]["humanego_train_eligible"] is False


def test_capture_canary_allows_diagnostics_but_never_formal_training():
    bundle = passing_bundle()
    bundle["stages"]["capture"]["scope"] = "CANARY"
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["hawor_rgb_execution_allowed"] is True
    assert result["categories"]["upstream_capture"]["status"] == "HOLD"
    assert result["routes"]["robot_formal_ready"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_raw_training_requires_explicit_robot_sidecar_consumption_authority():
    bundle = passing_bundle("RAW_RGB_KAI22")
    bundle["stages"]["robot"]["sidecar_consumption_authorized"] = False
    result = evaluate_cross_stage(bundle)
    assert result["routes"]["robot_sidecar_ready"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_static_setup_anchor_cannot_substitute_for_moving_object_identity():
    bundle = passing_bundle()
    bundle["stages"]["mask"]["gates"]["moving_task_object_identity"] = gate(
        "FAIL", "INDEPENDENT_QA", "MOVED_OBJECT_INSTANCE_LOST"
    )
    # The static setup route and its authority remain PASS, but they have a
    # different semantic role and therefore cannot rescue the moving object.
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["mask_semantic"]["status"] == "FAIL"
    assert result["categories"]["mask_temporal"]["status"] == "FAIL"
    assert result["routes"]["mask_formal_ready"] is False


def test_hawor_numeric_pass_is_not_mask_provider_without_identity_and_contour():
    bundle = passing_bundle()
    bundle["stages"]["hawor"]["gates"]["anatomical_identity"] = gate(
        "NOT_EVALUATED", "NONE", "INDEPENDENT_IDENTITY_REVIEW_MISSING"
    )
    bundle["stages"]["hawor"]["gates"]["independent_contour"] = gate(
        "NOT_EVALUATED", "NONE", "FROZEN_CONTOUR_LABELS_MISSING"
    )
    result = evaluate_cross_stage(bundle)
    providers = result["categories"]["mask_semantic"]["provider_admission"]
    assert providers["human_core.left"]["status"] == "HOLD"
    assert providers["human_core.right"]["status"] == "HOLD"
    assert result["routes"]["mask_formal_ready"] is False


def test_pico_provider_eligibility_is_per_side():
    bundle = passing_bundle()
    mask = bundle["stages"]["mask"]
    mask["seed_route"] = "MIXED"
    mask["provider_routes"]["human_core"] = {
        "left": provider("PICO_OPENXR21"),
        "right": provider("MANUAL"),
    }
    bundle["stages"]["capture"]["gates"]["pico21_left_provider"] = gate(
        "FAIL", "CAPTURE", "PICO_LEFT_WRIST_STEP_313MM"
    )
    result = evaluate_cross_stage(bundle)
    providers = result["categories"]["mask_semantic"]["provider_admission"]
    assert providers["human_core.left"]["status"] == "HOLD"
    assert providers["human_core.right"]["status"] == "PASS"
    assert result["routes"]["mask_formal_ready"] is False


def test_hawor_failure_does_not_block_mask_when_no_mask_role_declares_hawor():
    bundle = passing_bundle()
    mask = bundle["stages"]["mask"]
    mask["seed_route"] = "MANUAL"
    mask["provider_routes"]["human_core"] = {
        "left": provider("MANUAL"),
        "right": provider("MANUAL"),
    }
    bundle["stages"]["hawor"]["gates"]["anatomical_identity"] = gate(
        "FAIL", "INDEPENDENT_QA", "HAWOR_WRONG_HAND_IDENTITY"
    )
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["hawor_algorithm"]["status"] == "FAIL"
    assert result["categories"]["mask_semantic"]["status"] == "PASS"
    assert result["categories"]["mask_temporal"]["status"] == "PASS"


def test_functional_retarget_requires_independent_authority():
    bundle = passing_bundle("RAW_RGB_KAI22")
    bundle["stages"]["robot"]["gates"]["functional_retarget_authority"] = gate(
        "PASS", "PRODUCER"
    )
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["robot_authority"]["status"] == "HOLD"
    assert result["routes"]["robot_sidecar_ready"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_evidence_hash_verification_detects_drift(tmp_path):
    artifact = tmp_path / "evidence.json"
    artifact.write_text("before", encoding="utf-8")
    bundle = passing_bundle()
    capture = bundle["stages"]["capture"]
    capture["evidence"] = [{"path": str(artifact), "sha256": "f" * 64}]
    errors = verify_evidence_files({**bundle, "stages": {"capture": capture}})
    assert any("stages.capture.evidence[0]: sha256 mismatch" in error for error in errors)
    result = evaluate_cross_stage(bundle, evidence_verification_errors=errors)
    assert result["input_contract"]["valid"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_unknown_gate_is_invalid_and_fail_closed():
    bundle = passing_bundle()
    bundle["stages"]["mask"]["gates"]["session_specific_magic"] = gate()
    result = evaluate_cross_stage(bundle)
    assert result["input_contract"]["valid"] is False
    assert result["routes"]["humanego_train_eligible"] is False


def test_profile_variant_mismatch_blocks_training():
    bundle = deepcopy(passing_bundle("RAW_RGB_KAI22"))
    bundle["image_variant"] = "CLEAN"
    result = evaluate_cross_stage(bundle)
    assert result["categories"]["training_eligibility"]["status"] == "HOLD"
    assert result["routes"]["humanego_train_eligible"] is False
