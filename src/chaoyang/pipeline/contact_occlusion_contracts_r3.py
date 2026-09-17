"""V7.1-R3 Contact-10 与 Occlusion Silver fail-closed 合同。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from chaoyang.pipeline.object_contact_evidence_v1 import EvidenceGraphError, validate_evidence_dag


class ContactOcclusionContractError(ValueError):
    """接触或遮挡记录违反合同。"""


FINGERS = {"thumb", "index", "middle", "ring", "little"}
CONTACT_STATES = {"APPROACH", "TOUCH_CANDIDATE", "SLIDE_CANDIDATE", "RELEASE", "UNKNOWN"}
OWNERSHIP = {"BACKGROUND", "HUMAN_FRONT", "OBJECT_FRONT", "ROBOT_FRONT", "TIE_UNKNOWN"}
PIXEL_SOURCES = {
    "RAW_VISIBLE_OBJECT",
    "CAUSAL_TEMPORAL_DONOR",
    "POSE_VERIFIED_OBJECT_ATLAS",
    "TEXTURED_OBJECT_RENDERER",
    "UNKNOWN",
}

# Silver 不报告人工真值 accuracy，但仍必须满足可用性/来源闭包门。阈值来自
# 当前 compositor 合同；不能由调用方仅把 status 写成 READY 来绕过。
SILVER_GATES = {
    "known_decision_coverage_min": 0.70,
    "unknown_pixel_ratio_max": 0.30,
    "unknown_contact_frame_ratio_max": 0.20,
    "max_unknown_run_max": 5,
    "protected_retention_min": 0.99,
    "pixel_provenance_coverage_min": 0.70,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContactOcclusionContractError(message)


def validate_contact_hypothesis(record: Mapping[str, Any]) -> None:
    """验证逐指 Contact-10 输出并复用 evidence DAG 环检测。"""

    _require(record.get("schema_version") == "CONTACT_HYPOTHESIS_R3", "wrong contact schema")
    _require(record.get("claim_status") == "HYPOTHESIS_ONLY", "contact has no external truth")
    _require(record.get("external_accuracy") == "UNKNOWN", "external contact accuracy is unknown")
    evidence_nodes = record.get("evidence_nodes")
    _require(isinstance(evidence_nodes, list), "evidence_nodes are required")
    try:
        graph = validate_evidence_dag(evidence_nodes)
    except EvidenceGraphError as error:
        raise ContactOcclusionContractError(str(error)) from error

    hypotheses = record.get("hypotheses")
    _require(isinstance(hypotheses, list) and hypotheses, "contact hypotheses are required")
    seen: set[tuple[int, str, str]] = set()
    for row in hypotheses:
        _require(isinstance(row, Mapping), "contact hypothesis must be an object")
        hand = row.get("hand_side")
        finger = row.get("finger_id")
        _require(hand in {"left", "right"}, "invalid hand_side")
        _require(finger in FINGERS, "invalid finger_id")
        frame_id = row.get("frame_id")
        _require(type(frame_id) is int and frame_id >= 0, "invalid frame_id")
        key = (frame_id, hand, finger)
        _require(key not in seen, "duplicate per-frame finger hypothesis")
        seen.add(key)
        state = row.get("state")
        _require(state in CONTACT_STATES, "invalid contact state")
        evidence_id = row.get("evidence_id")
        _require(evidence_id in graph.nodes_by_id, "contact row references missing evidence")
        valid = row.get("valid")
        _require(type(valid) is bool, "valid must be boolean")
        uncertainty = row.get("uncertainty")
        _require(isinstance(uncertainty, (int, float)) and 0 <= uncertainty <= 1, "uncertainty must be [0,1]")
        if state == "UNKNOWN":
            _require(valid is False, "UNKNOWN contact must be invalid")
            _require(row.get("surface_distance_mm") is None, "UNKNOWN distance must be null")
            _require(row.get("relative_velocity") is None, "UNKNOWN velocity must be null")
            _require(row.get("slip_score") is None, "UNKNOWN slip must be null")
        else:
            _require(valid is True, "known contact hypothesis must be valid")
            _require(row.get("object_instance_id"), "known contact requires object identity")
            distance = row.get("surface_distance_mm")
            _require(isinstance(distance, (int, float)) and np.isfinite(distance), "distance must be finite")


def validate_occlusion_silver(record: Mapping[str, Any]) -> None:
    """验证 Silver 指标与像素来源；Silver 永远不得报告 accuracy。"""

    _require(record.get("schema_version") == "OCCLUSION_SILVER_R3", "wrong occlusion schema")
    _require(record.get("authority_level") == "SILVER", "this validator only accepts Silver")
    _require(record.get("accuracy_reported") is False, "Silver must not report accuracy")
    _require(record.get("external_accuracy") == "UNKNOWN", "Silver external accuracy is unknown")
    forbidden = {"known_accuracy", "precision", "recall", "f1"}
    _require(not forbidden.intersection(record), "Gold-only accuracy metric found in Silver")
    for key in (
        "known_decision_coverage",
        "unknown_pixel_ratio",
        "unknown_contact_frame_ratio",
        "protected_retention",
        "pixel_provenance_coverage",
    ):
        value = record.get(key)
        _require(isinstance(value, (int, float)) and 0 <= value <= 1, f"{key} must be [0,1]")
    _require(type(record.get("max_unknown_run")) is int and record["max_unknown_run"] >= 0, "invalid max_unknown_run")
    for key in (
        "temporal_consistency_pass",
        "zbuffer_consistency_pass",
        "byte_exact_outside_authorized_band",
        "causal_donor_pass",
    ):
        _require(type(record.get(key)) is bool, f"{key} must be boolean")

    pixels = record.get("pixel_records")
    _require(isinstance(pixels, list) and pixels, "pixel_records must be a non-empty array")
    for pixel in pixels:
        _require(pixel.get("ownership") in OWNERSHIP, "invalid ownership")
        _require(pixel.get("pixel_source") in PIXEL_SOURCES, "invalid pixel source")
        training_valid = pixel.get("training_valid")
        _require(type(training_valid) is bool, "training_valid must be boolean")
        if pixel.get("ownership") == "TIE_UNKNOWN" or pixel.get("pixel_source") == "UNKNOWN":
            _require(training_valid is False, "UNKNOWN pixels must be invalid for training")
        donor_frame = pixel.get("donor_frame_id")
        target_frame = pixel.get("frame_id")
        if pixel.get("pixel_source") == "CAUSAL_TEMPORAL_DONOR":
            _require(type(donor_frame) is int and type(target_frame) is int, "donor frame IDs required")
            _require(donor_frame <= target_frame, "future donor is forbidden")

    gate_results = {
        "known_decision_coverage": record["known_decision_coverage"]
        >= SILVER_GATES["known_decision_coverage_min"],
        "unknown_pixel_ratio": record["unknown_pixel_ratio"] <= SILVER_GATES["unknown_pixel_ratio_max"],
        "unknown_contact_frame_ratio": record["unknown_contact_frame_ratio"]
        <= SILVER_GATES["unknown_contact_frame_ratio_max"],
        "max_unknown_run": record["max_unknown_run"] <= SILVER_GATES["max_unknown_run_max"],
        "protected_retention": record["protected_retention"] >= SILVER_GATES["protected_retention_min"],
        "pixel_provenance_coverage": record["pixel_provenance_coverage"]
        >= SILVER_GATES["pixel_provenance_coverage_min"],
        "temporal_consistency": record["temporal_consistency_pass"],
        "zbuffer_consistency": record["zbuffer_consistency_pass"],
        "byte_exact_outside_authorized_band": record["byte_exact_outside_authorized_band"],
        "causal_donor": record["causal_donor_pass"],
    }
    computed_ready = all(gate_results.values())
    declared_ready = record.get("status") == "VISUAL_OCCLUSION_SILVER_READY"
    _require(
        declared_ready == computed_ready,
        "Silver status disagrees with computed coverage/provenance/consistency gates",
    )


def audit_zbuffer_front(
    *, robot_depth_m: Sequence[float], object_depth_m: Sequence[float], tolerance_m: float = 1e-4
) -> tuple[str, ...]:
    """独立于绘制顺序的两层 z-buffer 判定。"""

    robot = np.asarray(robot_depth_m, dtype=np.float64)
    obj = np.asarray(object_depth_m, dtype=np.float64)
    _require(robot.shape == obj.shape, "depth arrays must share shape")
    result: list[str] = []
    for robot_z, object_z in zip(robot.reshape(-1), obj.reshape(-1)):
        if not np.isfinite(robot_z) and not np.isfinite(object_z):
            result.append("TIE_UNKNOWN")
        elif not np.isfinite(robot_z):
            result.append("OBJECT_FRONT")
        elif not np.isfinite(object_z):
            result.append("ROBOT_FRONT")
        elif abs(robot_z - object_z) <= tolerance_m:
            result.append("TIE_UNKNOWN")
        elif robot_z < object_z:
            result.append("ROBOT_FRONT")
        else:
            result.append("OBJECT_FRONT")
    return tuple(result)
