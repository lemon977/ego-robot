from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.pipeline.contact_occlusion_contracts_r3 import (
    ContactOcclusionContractError,
    audit_zbuffer_front,
    validate_contact_hypothesis,
    validate_occlusion_silver,
)
from chaoyang.pipeline.depth_mask_clean_contracts_r3 import (
    PipelineContractError,
    audit_clean_domains,
    audit_mask_visibility,
    evaluate_clean_successor_gate,
    evaluate_depth_quality,
    fuse_controller_wrist_depth,
    validate_depth_source_audit,
    validate_donor_atlas_record,
    validate_mask_contract,
)
from chaoyang.ops.validate_pipeline_contracts_r3 import (
    contact_fixture,
    depth00_fixture,
    donor_fixture,
    mask_fixture,
    occlusion_fixture,
)


def test_all_r3_json_schemas_are_valid_and_accept_fixtures() -> None:
    fixtures = {
        "depth_source_audit_r3.schema.json": depth00_fixture(),
        "depth_quality_evidence_r3.schema.json": {
            "schema_version": "DEPTH_QUALITY_EVIDENCE_R3", "stage": "DEPTH-10", "valid_coverage": 1.0,
            "formula_mae_m": 0.0, "formula_p95_m": 0.0, "lr_consistency_mae_px": None,
            "registration_p90_px": None, "temporal_static_std_m": None, "boundary_uncertain_ratio": 0.0,
            "temporal_static_std_p95_m": None, "temporal_frame_median_std_m": None,
            "low_texture_ratio": 0.0, "reflective_ratio": 0.0, "motion_blur_ratio": 0.0,
            "depth_confidence_present": False, "external_metric_accuracy": "UNKNOWN",
            "claim_limit": "Internal consistency only."
        },
        "wrist_fusion_r3.schema.json": {
            "schema_version": "WRIST_FUSION_R3", "stage": "DEPTH-20", "primary_anchor": "CONTROLLER_SE3",
            "finger_source": "MANUS25", "stereo_measurement": "VISIBLE_SURFACE_OPTICAL_Z",
            "configuration": {"causal_window": 15, "innovation_limit_m": 0.03, "stereo_weight": 0.15, "ema_alpha": 0.15, "correction_cap_m": 0.005},
            "frames": [], "external_metric_accuracy": "UNKNOWN", "claim_limit": "Internal bounded fusion only."
        },
        "mask_role_object_r3.schema.json": mask_fixture(),
        "clean_write_domain_r3.schema.json": {
            "schema_version": "CLEAN_WRITE_DOMAIN_R3", "stage": "CLEAN-20",
            "mask_domains": {"remove": "M_remove", "flow": "M_flow", "write": "M_write", "relations": ["M_write_SUBSET_M_remove", "M_remove_SUBSET_M_flow", "M_write_DISJOINT_VISIBLE_OBJECT"]},
            "pixel_source_precedence": ["RAW_VISIBLE_OBJECT", "CAUSAL_TEMPORAL_DONOR", "POSE_VERIFIED_OBJECT_ATLAS", "PROPainter_REMAINDER", "UNKNOWN"],
            "causal_training_input": True, "fresh_propainter_required": True,
            "quality_gates": {"contact_added_removal_reduction": 0.5, "human_residual_no_regression": True, "nonhuman_leakage_no_regression": True, "visible_object_retention": 0.999, "byte_exact_outside_write": True},
            "claim_limit": "Visual Clean only."
        },
        "causal_donor_atlas_r3.schema.json": donor_fixture(),
        "contact_hypothesis_r3.schema.json": contact_fixture(),
        "occlusion_silver_r3.schema.json": occlusion_fixture(),
    }
    for name, fixture in fixtures.items():
        schema = json.loads((PROJECT / "contracts" / name).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(fixture)


def test_depth_source_rejects_native_confidence_or_borrowed_calibration() -> None:
    record = depth00_fixture()
    validate_depth_source_audit(record)
    bad = copy.deepcopy(record); bad["depth_confidence_present"] = True
    with pytest.raises(PipelineContractError, match="confidence"):
        validate_depth_source_audit(bad)
    bad = copy.deepcopy(record); bad["calibration"]["identity_status"] = "NEARBY_SESSION"
    with pytest.raises(PipelineContractError, match="verified"):
        validate_depth_source_audit(bad)


def test_depth_quality_formula_is_internal_and_does_not_invent_confidence() -> None:
    disparity = np.asarray([[10.0, 20.0], [5.0, np.nan]])
    depth = np.asarray([[1.0, 0.5], [2.0, np.nan]])
    result = evaluate_depth_quality(disparity_px=disparity, depth_m=depth, valid=np.isfinite(depth), focal_px=100, baseline_m=0.1)
    assert result.formula_mae_m == pytest.approx(0.0)
    assert result.depth_confidence_present is False
    assert result.external_metric_accuracy == "UNKNOWN"


def test_depth_temporal_drift_does_not_measure_static_plane_slope() -> None:
    disparity = np.asarray([[[10.0, 5.0], [4.0, 2.0]]] * 3)
    depth = 10.0 / disparity
    result = evaluate_depth_quality(
        disparity_px=disparity,
        depth_m=depth,
        valid=np.ones_like(depth, dtype=bool),
        focal_px=100.0,
        baseline_m=0.1,
        static_region_mask=np.ones((2, 2), dtype=bool),
        lr_residual_px=np.zeros_like(depth),
    )
    assert result.temporal_static_std_m == pytest.approx(0.0)
    assert result.temporal_static_std_p95_m == pytest.approx(0.0)
    assert result.temporal_frame_median_std_m == pytest.approx(0.0)
    with pytest.raises(PipelineContractError, match="LR residual shape"):
        evaluate_depth_quality(
            disparity_px=disparity,
            depth_m=depth,
            valid=np.ones_like(depth, dtype=bool),
            focal_px=100.0,
            baseline_m=0.1,
            lr_residual_px=np.zeros((2, 2)),
        )


def test_wrist_fusion_keeps_controller_on_invalid_and_rejects_large_innovation() -> None:
    controller = np.asarray([[1.0, 2.0, 0.5], [1.1, 2.1, 0.5], [1.2, 2.2, 0.5]])
    result = fuse_controller_wrist_depth(controller, np.asarray([0.51, np.nan, 0.60]), np.asarray([True, False, True]))
    assert result.accepted_stereo.tolist() == [True, False, False]
    assert np.array_equal(result.fused_wrist_m[1:], controller[1:])
    assert np.all(np.abs(result.correction_m) <= 0.005)
    assert result.source_flags[2] == "CONTROLLER_ONLY_INNOVATION_REJECTED"


def test_mask_baseline_h4_semantics_and_offscreen_empty() -> None:
    record = mask_fixture(); validate_mask_contract(record)
    bad = copy.deepcopy(record); bad["algorithm_baseline"] = "SAM2.1"
    with pytest.raises(PipelineContractError, match="SAM3.1"):
        validate_mask_contract(bad)
    bad = copy.deepcopy(record); bad["qa_status"] = "FAILED_QUALITY_C"
    with pytest.raises(PipelineContractError, match="formal H4"):
        validate_mask_contract(bad)
    masks = {"left": np.zeros((2, 2, 2), dtype=bool)}
    audit_mask_visibility(masks, {"left": [False, True]})
    masks["left"][0, 0, 0] = True
    with pytest.raises(PipelineContractError, match="offscreen"):
        audit_mask_visibility(masks, {"left": [False, True]})


def test_mask_schema_rejects_wrong_formal_h4_and_unknown_fields() -> None:
    schema = json.loads((PROJECT / "contracts/mask_role_object_r3.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    bad = mask_fixture()
    bad["qa_status"] = "FAILED_QUALITY_C"
    assert any(error.validator == "const" for error in validator.iter_errors(bad))
    bad = mask_fixture()
    bad["invented_authority"] = True
    assert any(error.validator == "additionalProperties" for error in validator.iter_errors(bad))


def test_chips_objects_are_three_independent_instances() -> None:
    record = mask_fixture()
    record.update(mask_kind="OBJECT", scope="EXACT78", task="chips")
    record["instances"] = [
        {"instance_id": f"chips_{item}", "role": "task_object", "offscreen_policy": "EMPTY_MASK", "reentry_policy": "RESTORE_SAME_IDENTITY"}
        for item in ("left", "center", "right")
    ]
    validate_mask_contract(record)
    record["instances"][0]["instance_id"] = "chips_union"
    with pytest.raises(PipelineContractError, match="union"):
        validate_mask_contract(record)


def test_clean_domains_protect_object_and_keep_outside_write_byte_exact() -> None:
    remove = np.asarray([[True, False], [False, False]])
    flow = np.asarray([[True, True], [False, False]])
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    output = source.copy(); output[0, 0] = 10
    result = audit_clean_domains(m_remove=remove, m_flow=flow, m_write=remove, protected_object_mask=np.zeros_like(remove), input_rgb=source, output_rgb=output)
    assert result.changed_outside_write_pixels == 0
    with pytest.raises(PipelineContractError, match="protected"):
        audit_clean_domains(m_remove=remove, m_flow=flow, m_write=remove, protected_object_mask=remove)
    output[1, 1] = 20
    with pytest.raises(PipelineContractError, match="byte-exact"):
        audit_clean_domains(m_remove=remove, m_flow=flow, m_write=remove, protected_object_mask=np.zeros_like(remove), input_rgb=source, output_rgb=output)


def test_clean_gate_cannot_pass_by_leaving_more_human_pixels() -> None:
    result = evaluate_clean_successor_gate(baseline_contact_added_removal=10, candidate_contact_added_removal=4, baseline_human_residual=2, candidate_human_residual=3, baseline_nonhuman_leakage=2, candidate_nonhuman_leakage=1, visible_object_retention=1.0)
    assert result["contact_band_reduction_pass"] is True
    assert result["human_residual_no_regression"] is False
    assert result["passed"] is False


def test_atlas_must_be_causal_identity_and_pose_verified() -> None:
    record = donor_fixture(); validate_donor_atlas_record(record)
    bad = copy.deepcopy(record); bad["source_frame_ids"] = [11]
    with pytest.raises(PipelineContractError, match="future donor"):
        validate_donor_atlas_record(bad)
    bad = copy.deepcopy(record); bad["identity_verified"] = False
    with pytest.raises(PipelineContractError, match="identity"):
        validate_donor_atlas_record(bad)


def test_contact_unknown_and_attachment_cannot_self_prove() -> None:
    record = contact_fixture(); validate_contact_hypothesis(record)
    bad = copy.deepcopy(record)
    bad["hypotheses"][0].update(state="UNKNOWN", valid=True, surface_distance_mm=None, relative_velocity=None, slip_score=None)
    with pytest.raises(ContactOcclusionContractError, match="UNKNOWN contact"):
        validate_contact_hypothesis(bad)
    bad = copy.deepcopy(record)
    bad["evidence_nodes"][2]["may_support_contact_authority"] = True
    with pytest.raises(ContactOcclusionContractError, match="capability escalation"):
        validate_contact_hypothesis(bad)


def test_occlusion_silver_rejects_accuracy_future_donor_and_unknown_training() -> None:
    record = occlusion_fixture(); validate_occlusion_silver(record)
    bad = copy.deepcopy(record); bad["known_accuracy"] = 0.99
    with pytest.raises(ContactOcclusionContractError, match="Gold-only"):
        validate_occlusion_silver(bad)
    bad = copy.deepcopy(record); bad["pixel_records"][0]["donor_frame_id"] = 5
    with pytest.raises(ContactOcclusionContractError, match="future donor"):
        validate_occlusion_silver(bad)
    bad = copy.deepcopy(record); bad["pixel_records"][1]["training_valid"] = True
    with pytest.raises(ContactOcclusionContractError, match="UNKNOWN"):
        validate_occlusion_silver(bad)


def test_occlusion_silver_ready_is_derived_and_rejects_all_unknown() -> None:
    bad = occlusion_fixture()
    bad.update(
        known_decision_coverage=0.0,
        unknown_pixel_ratio=1.0,
        unknown_contact_frame_ratio=1.0,
        max_unknown_run=100,
        protected_retention=0.0,
        pixel_provenance_coverage=0.0,
        temporal_consistency_pass=False,
        zbuffer_consistency_pass=False,
        byte_exact_outside_authorized_band=False,
        causal_donor_pass=False,
    )
    bad["pixel_records"] = [{
        "frame_id": 0,
        "ownership": "TIE_UNKNOWN",
        "pixel_source": "UNKNOWN",
        "training_valid": False,
        "donor_frame_id": None,
    }]
    with pytest.raises(ContactOcclusionContractError, match="disagrees"):
        validate_occlusion_silver(bad)
    bad["status"] = "FAILED_QUALITY_C"
    validate_occlusion_silver(bad)


def test_zbuffer_is_depth_order_not_draw_order() -> None:
    assert audit_zbuffer_front(robot_depth_m=[0.8, 1.1, np.nan], object_depth_m=[1.0, 0.9, 0.7]) == (
        "ROBOT_FRONT", "OBJECT_FRONT", "OBJECT_FRONT"
    )


@pytest.mark.parametrize("mode", ["depth-00", "depth-10", "depth-20", "mask-role", "mask-object", "atlas", "donor", "clean", "contact-10", "occlusion-silver"])
def test_cli_dry_run_works_outside_repository(mode: str, tmp_path: Path) -> None:
    command = [sys.executable, str(PROJECT / "src/chaoyang/ops/validate_pipeline_contracts_r3.py"), "--mode", mode, "--dry-run"]
    completed = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASSED_DEVELOPMENT_DRY_RUN"


def test_materialized_task_packets_are_bounded_and_schema_valid() -> None:
    root = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3"
    index = json.loads((root / "TASK_PACKET_INDEX.json").read_text())
    assert index["status"] == "DEVELOPMENT_PACKETS_MATERIALIZED_NOT_CURRENT"
    assert index["authority_promoted"] is False
    assert len(index["packets"]) == 10
    packet_schema = json.loads((PROJECT / "docs/governance/schemas/task_packet.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(packet_schema)
    for entry in index["packets"]:
        packet_path = Path(entry["path"].replace("/mnt/workspace/code/chaoyang/tasks/", "/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/content/history/tasks/"))
        assert packet_path.is_absolute()
        packet = json.loads(packet_path.read_text())
        validator.validate(packet)
        assert len(packet["read_set"]) <= 8
        assert packet["authority_promoted"] is False
        assert packet["prerequisites"][:2] == ["G0_CORE_GOVERNANCE=PASS", "governance_freshness=FRESH"]
