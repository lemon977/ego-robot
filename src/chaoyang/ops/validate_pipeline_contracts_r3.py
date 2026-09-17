#!/usr/bin/env python3
"""验证 V7.1-R3 Depth/Mask/Clean/Contact/Occlusion 开发合同。

适用模式：``--dry-run`` 使用内置合成 fixture；``--input`` 验证一份 JSON。
本工具不运行模型、不写 current governance、不晋升 authority。若指定 ``--output``，
只写调用方明确给出的 immutable attempt 路径。
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.pipeline.contact_occlusion_contracts_r3 import (  # noqa: E402
    audit_zbuffer_front,
    validate_contact_hypothesis,
    validate_occlusion_silver,
)
from chaoyang.pipeline.depth_mask_clean_contracts_r3 import (  # noqa: E402
    audit_clean_domains,
    audit_mask_visibility,
    evaluate_clean_successor_gate,
    evaluate_depth_quality,
    fuse_controller_wrist_depth,
    validate_depth_source_audit,
    validate_donor_atlas_record,
    validate_mask_contract,
)


SHA0 = "0" * 64


def depth00_fixture() -> dict[str, Any]:
    return {
        "schema_version": "DEPTH_SOURCE_AUDIT_R3", "stage": "DEPTH-00", "session_id": "fixture",
        "depth_convention": "OPTICAL_Z", "depth_unit": "m", "formula": "Z=fB/d",
        "depth_confidence_present": False, "external_metric_accuracy": "UNKNOWN",
        "calibration": {"K": [[100.0, 0, 1], [0, 100.0, 1], [0, 0, 1]], "baseline_m": 0.1, "identity_status": "VERIFIED_SAME_SESSION"},
        "image_domains": {key: {} for key in ("raw_fisheye", "rectified_left", "selected_left", "visual_mp4")},
        "coordinate_adapters": [{"source_domain": "rectified_left", "target_domain": "selected_left", "implementation_sha256": SHA0}],
        "claim_limit": "Internal coordinate and formula audit only; not external metric accuracy."
    }


def mask_fixture() -> dict[str, Any]:
    return {
        "schema_version": "MASK_ROLE_OBJECT_R3", "mask_kind": "ROLE", "scope": "SENSOR_H4_FORMAL",
        "input_domain": "RAW_RGB", "algorithm_baseline": "SAM3.1", "task": "none",
        "instances": [
            {"instance_id": "left_glove", "role": "left_glove", "offscreen_policy": "EMPTY_MASK", "reentry_policy": "RESTORE_SAME_IDENTITY"},
            {"instance_id": "right_glove", "role": "right_glove", "offscreen_policy": "EMPTY_MASK", "reentry_policy": "RESTORE_SAME_IDENTITY"}
        ],
        "execution_status": "BLOCKED_RESOURCE", "qa_status": "NOT_EVALUATED", "policy_status": "POLICY_DEFERRED",
        "pixel_mask_authority": False, "authority_promoted": False,
        "claim_limit": "Formal H4 has not run pixel inference."
    }


def donor_fixture() -> dict[str, Any]:
    return {
        "schema_version": "CAUSAL_DONOR_ATLAS_R3", "artifact_kind": "OBJECT_ATLAS", "session_id": "fixture",
        "object_instance_id": "card_0", "target_frame_id": 10, "source_frame_ids": [2, 7, 10],
        "causal_training_eligible": True, "identity_verified": True, "pose_verified": True,
        "claim_limit": "Causal visible texture evidence only."
    }


def contact_fixture() -> dict[str, Any]:
    direct = {
        "evidence_id": "direct", "evidence_type": "DIRECT_OBJECT6D", "parent_evidence_ids": [], "evidence_depth": 0,
        "may_support_contact_authority": True, "may_support_object6d_authority": True,
        "may_support_gold_contact_accuracy": False, "may_upgrade_tactile_supported_contact": False,
    }
    seed = {
        "evidence_id": "seed", "evidence_type": "CONTACT_SEED", "parent_evidence_ids": ["direct"], "evidence_depth": 1,
        "may_support_contact_authority": True, "may_support_object6d_authority": False,
        "may_support_gold_contact_accuracy": False, "may_upgrade_tactile_supported_contact": False,
    }
    attachment = {
        "evidence_id": "attachment", "evidence_type": "HAND_OBJECT_ATTACHMENT", "parent_evidence_ids": ["seed"], "evidence_depth": 2,
        "may_support_contact_authority": False, "may_support_object6d_authority": False,
        "may_support_gold_contact_accuracy": False, "may_upgrade_tactile_supported_contact": False,
    }
    return {
        "schema_version": "CONTACT_HYPOTHESIS_R3", "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN",
        "evidence_nodes": [direct, seed, attachment],
        "hypotheses": [{
            "frame_id": 0, "hand_side": "right", "finger_id": "index", "pad_link": "right_index_pad",
            "object_instance_id": "card_0", "state": "TOUCH_CANDIDATE", "surface_distance_mm": 1.0,
            "relative_velocity": 0.0, "slip_score": 0.1, "uncertainty": 0.2, "valid": True,
            "evidence_id": "seed", "parent_evidence_ids": ["direct"]
        }],
        "claim_limit": "Hypothesis only; no external contact accuracy."
    }


def occlusion_fixture() -> dict[str, Any]:
    return {
        "schema_version": "OCCLUSION_SILVER_R3", "authority_level": "SILVER", "status": "VISUAL_OCCLUSION_SILVER_READY",
        "known_decision_coverage": 0.8, "unknown_pixel_ratio": 0.2, "unknown_contact_frame_ratio": 0.1,
        "max_unknown_run": 2, "protected_retention": 1.0, "pixel_provenance_coverage": 0.8,
        "temporal_consistency_pass": True, "zbuffer_consistency_pass": True,
        "byte_exact_outside_authorized_band": True, "causal_donor_pass": True,
        "accuracy_reported": False, "external_accuracy": "UNKNOWN",
        "pixel_records": [
            {"frame_id": 4, "ownership": "OBJECT_FRONT", "pixel_source": "CAUSAL_TEMPORAL_DONOR", "training_valid": True, "donor_frame_id": 2},
            {"frame_id": 4, "ownership": "TIE_UNKNOWN", "pixel_source": "UNKNOWN", "training_valid": False, "donor_frame_id": None}
        ],
        "claim_limit": "Silver internal consistency only; no Gold accuracy."
    }


def run_mode(mode: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if mode == "depth-00":
        record = payload or depth00_fixture(); validate_depth_source_audit(record)
        return {"validated": True, "external_metric_accuracy": "UNKNOWN", "depth_confidence_present": False}
    if mode == "depth-10":
        disparity = np.asarray([[10.0, 20.0], [5.0, np.nan]])
        depth = np.asarray([[1.0, 0.5], [2.0, np.nan]])
        result = evaluate_depth_quality(disparity_px=disparity, depth_m=depth, valid=np.isfinite(depth), focal_px=100.0, baseline_m=0.1)
        return asdict(result)
    if mode == "depth-20":
        controller = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
        result = fuse_controller_wrist_depth(controller, np.asarray([0.51, np.nan, 0.6]), np.asarray([True, False, True]))
        return {"fused_wrist_m": result.fused_wrist_m.tolist(), "correction_m": result.correction_m.tolist(), "accepted_stereo": result.accepted_stereo.tolist(), "source_flags": list(result.source_flags), "external_metric_accuracy": "UNKNOWN"}
    if mode in {"mask-role", "mask-object"}:
        record = payload or mask_fixture()
        if mode == "mask-object" and payload is None:
            record = {**record, "mask_kind": "OBJECT", "scope": "EXACT78", "task": "chips", "instances": [
                {"instance_id": f"chips_{name}", "role": "task_object", "offscreen_policy": "EMPTY_MASK", "reentry_policy": "RESTORE_SAME_IDENTITY"}
                for name in ("left", "center", "right")
            ]}
        validate_mask_contract(record)
        masks = {item["instance_id"]: np.zeros((2, 2, 2), dtype=np.bool_) for item in record["instances"]}
        visibility = {key: [False, False] for key in masks}
        return {"contract_validated": True, **audit_mask_visibility(masks, visibility)}
    if mode in {"atlas", "donor"}:
        record = payload or donor_fixture()
        if mode == "donor" and payload is None:
            record = {**record, "artifact_kind": "BACKGROUND_DONOR", "object_instance_id": None, "identity_verified": False, "pose_verified": False}
        validate_donor_atlas_record(record); return {"validated": True, "causal_training_eligible": record["causal_training_eligible"]}
    if mode == "clean":
        mask = np.asarray([[True, False], [False, False]])
        flow = np.asarray([[True, True], [False, False]])
        audit = audit_clean_domains(m_remove=mask, m_flow=flow, m_write=mask, protected_object_mask=np.zeros_like(mask))
        gates = evaluate_clean_successor_gate(baseline_contact_added_removal=10, candidate_contact_added_removal=5, baseline_human_residual=2, candidate_human_residual=2, baseline_nonhuman_leakage=1, candidate_nonhuman_leakage=1, visible_object_retention=1.0)
        return {"domain_audit": asdict(audit), "quality_gates": gates}
    if mode == "contact-10":
        validate_contact_hypothesis(payload or contact_fixture()); return {"validated": True, "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN"}
    if mode == "occlusion-silver":
        validate_occlusion_silver(payload or occlusion_fixture())
        return {"validated": True, "front_order": audit_zbuffer_front(robot_depth_m=[0.8, 1.0, np.nan], object_depth_m=[1.0, 0.8, 1.0]), "accuracy_reported": False, "external_accuracy": "UNKNOWN"}
    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("depth-00", "depth-10", "depth-20", "mask-role", "mask-object", "atlas", "donor", "clean", "contact-10", "occlusion-silver"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.input is None:
        parser.error("use --dry-run or provide --input")
    payload = json.loads(args.input.read_text()) if args.input else None
    result = {"schema_version": "PIPELINE_CONTRACT_R3_DRY_RUN_RESULT", "mode": args.mode, "status": "PASSED_DEVELOPMENT_DRY_RUN", "authority_promoted": False, "result": run_mode(args.mode, payload)}
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
