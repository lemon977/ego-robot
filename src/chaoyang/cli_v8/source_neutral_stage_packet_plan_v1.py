"""Map an accepted v7 raw source into blocked, non-runnable stage plans.

No Task Packet is built here. Existing stage packet_value interfaces require
sealed upstream evidence, and the Scene frontend has no generic operation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chaoyang.cli_v7.source_neutral_plan_v1 import plan_one
from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, validate_artifact_ref
from chaoyang.governance.exact78_cold_start_guard_v1 import DATA_ROOT, ColdStartInputError


ACCEPTANCE = REPO_ROOT / "tasks/receipts/EXACT78_GENERIC_SUBMIT_CLI_V7_CODE_ACCEPTANCE_V1.json"
ROLES = {
    "TRACKER_CLIP_RGB": frozenset({"video", "source_stereo_video", "camera_params",
                                   "slam", "clip_manifest"}),
    "0915_RAW_VST": frozenset({"raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                               "raw/camera_params.json", "raw/camera_params.meta.json"}),
}
FORBIDDEN = frozenset({"preprocess", "all_data", "trackingdata", "processed", "clean",
                       "robot", "pico", "manus", "tactile", "hand", "controller"})
BUILDERS = {
    "CPU_ADMISSION": "src/chaoyang/governance/execute_exact78_first_submit_cpu_candidate_v1.py",
    "HAWOR_V4": "src/chaoyang/governance/execute_exact78_newraw_hawor_gpu_candidate_v4.py",
    "ROBOT_V1": "src/chaoyang/governance/execute_exact78_newraw_robot_cpu_candidate_v1.py",
    "SCENE": "src/chaoyang/governance/build_exact78_t3_six_scene_matrix_v2.py",
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _accepted_code() -> dict[str, Any]:
    receipt_ref = artifact_ref(ACCEPTANCE)
    receipt = load_json(ACCEPTANCE)
    if (receipt.get("schema_version") != "exact78-generic-submit-cli-v7-acceptance-receipt-v1"
            or receipt.get("status") != "PASSED"
            or receipt.get("task_id") != "exact78_generic_submit_cli_v7_code_acceptance_v1"):
        raise ColdStartInputError("V7_CODE_ACCEPTANCE_NOT_PASSED")
    result_ref = receipt.get("result")
    if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
        raise ColdStartInputError("V7_CODE_ACCEPTANCE_RESULT_SHA_DRIFT")
    packet_ref = receipt.get("task_packet")
    if not isinstance(packet_ref, dict) or validate_artifact_ref(packet_ref):
        raise ColdStartInputError("V7_CODE_ACCEPTANCE_PACKET_SHA_DRIFT")
    result = load_json(Path(result_ref["path"]))
    if (result.get("status") != "PASSED"
            or result.get("schema_version") != "exact78-generic-submit-cli-v7-acceptance-result-v1"
            or result.get("run_signature") != receipt.get("run_signature")
            or result.get("task_packet") != receipt.get("task_packet")):
        raise ColdStartInputError("V7_CODE_ACCEPTANCE_CHAIN_DRIFT")
    code = result.get("code")
    if not isinstance(code, dict) or any(validate_artifact_ref(ref) for ref in code.values()):
        raise ColdStartInputError("V7_ACCEPTED_CODE_SHA_DRIFT")
    return {"receipt": receipt_ref, "result": result_ref, "code": code}


def _source(source: dict[str, Any], signature: str) -> dict[str, Any]:
    if (source.get("schema_version") != "exact78-source-neutral-check-plan-v1"
            or source.get("source_kind") not in ROLES
            or set(source.get("source_files") or {}) != ROLES[source["source_kind"]]
            or not isinstance(signature, str) or len(signature) != 64
            or _digest(source) != signature):
        raise ColdStartInputError("V7_SOURCE_DESCRIPTOR_OR_SIGNATURE_DRIFT")
    root = Path(source.get("session_root", ""))
    allowed = DATA_ROOT.resolve(strict=True)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or not root.resolve(strict=True).is_relative_to(allowed)):
        raise ColdStartInputError("SOURCE_ROOT_INVALID")
    domain = root.resolve(strict=True).relative_to(allowed).parts[0]
    expected_domains = ({"chips_cards_tracker_0901", "chips_cards_tracker_0902"}
                        if source["source_kind"] == "TRACKER_CLIP_RGB" else
                        {"chips_cards_hands_0915"})
    if domain not in expected_domains:
        raise ColdStartInputError("SOURCE_DOMAIN_UNVERIFIED")
    for role, ref in source["source_files"].items():
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise ColdStartInputError("SOURCE_REF_INVALID:" + role)
        path = Path(ref["path"])
        if (not path.is_relative_to(root)
                or any(part.lower() in FORBIDDEN for part in path.relative_to(root).parts)
                or validate_artifact_ref(ref)):
            raise ColdStartInputError("FORBIDDEN_OR_DRIFTED_SOURCE_ROLE:" + role)
    if source["source_kind"] == "TRACKER_CLIP_RGB":
        if (source.get("physical_eye") != "tracker_session_left_RGB"
                or source.get("time_mode") != "TRACKER_CLIP_METADATA_ONLY"
                or source.get("capture_to_decode_alignment") != "CLIP_INDEX_ONLY"):
            raise ColdStartInputError("TRACKER_EYE_OR_TIME_CLAIM_DRIFT")
    elif (source.get("physical_eye") != "left_sourceIndex_1"
          or source.get("time_mode") != "TIMEBASE_UNVERIFIED"
          or source.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"):
        raise ColdStartInputError("VST_EYE_OR_TIME_CLAIM_DRIFT")
    return source


def _node(name: str, source_signature: str, builder_ref: dict[str, Any],
          required: list[str], missing: list[str], source_roles: list[str],
          interface: str, blocker: str) -> dict[str, Any]:
    binding = {"node": name, "source_signature": source_signature,
               "builder": builder_ref, "required": required,
               "source_roles": source_roles, "interface": interface}
    return {"node": name, "status": "BLOCKED_MISSING_SEALED_INPUT_OR_ROUTE",
            "packet_construction": "NOT_IMPLEMENTED",
            "dependency_fingerprint": _digest(binding), "binding": binding,
            "missing": missing, "blocker": blocker,
            "run_signature": None, "execution_allowed": False}


def plan_source(source: dict[str, Any], signature: str,
                *, accepted: dict[str, Any] | None = None) -> dict[str, Any]:
    source = _source(source, signature)
    sealed = _accepted_code()
    if accepted is not None and accepted != sealed:
        raise ColdStartInputError("V7_CODE_ACCEPTANCE_REQUIRED")
    accepted = sealed
    builders = {name: artifact_ref(REPO_ROOT / path) for name, path in BUILDERS.items()}
    kind = source["source_kind"]
    roles = source["source_files"]
    rgb = "video" if kind == "TRACKER_CLIP_RGB" else "raw/vst.h264"
    scene_rgb = "source_stereo_video" if kind == "TRACKER_CLIP_RGB" else rgb
    camera = "camera_params" if kind == "TRACKER_CLIP_RGB" else "raw/camera_params.json"
    cpu = _node("CPU_ADMISSION", signature, builders["CPU_ADMISSION"],
        ["source_descriptor", "decoded_frame_rows", "FRAME_TIME_MAP", "admission_seal"],
        ["decoded_frame_rows", "FRAME_TIME_MAP", "admission_seal",
         "dynamic_task_packet_builder_and_candidate_route"],
        sorted(roles), "preflight(session)->packet_value(prepared)",
        "V7_DESCRIPTOR_IS_NOT_DECODED_CPU_ADMISSION_RESULT")
    hawor = _node("HAWOR_V4", signature, builders["HAWOR_V4"],
        ["terminal_cpu_RESULT", "descriptor", "FRAME_TIME_MAP", "admission_seal",
         "cpu_receipt", "HaWoR_model_asset_refs", "GPU_lease"],
        ["terminal_cpu_RESULT", "descriptor", "FRAME_TIME_MAP", "admission_seal",
         "cpu_receipt", "model_asset_and_runtime_refs", "GPU_lease"],
        [rgb, camera], "packet_value(cpu_result_path)",
        "REQUIRES_SEALED_CPU_RESULT_AND_NATIVE_OPEN_AUDIT")
    robot = _node("ROBOT_V1", signature, builders["ROBOT_V1"],
        ["terminal_cpu_RESULT", "terminal_HaWoR_v4_RESULT", "HaWoR_receipt",
         "HaWoR_stage_seal", "native_open_audit", "robot_asset_pin"],
        ["terminal_cpu_RESULT", "terminal_HaWoR_v4_RESULT", "HaWoR_receipt",
         "HaWoR_stage_seal", "native_open_audit", "robot_asset_pin"],
        [rgb, camera], "packet_value(hawor_outer_result_path)",
        "REQUIRES_SEALED_HAWOR_HANDOFF_AND_ROBOT_ASSETS")
    scene = _node("SCENE", signature, builders["SCENE"],
        ["same_source_frontend_spec", "scene_frame_map", "mask_result_and_seal",
         "clean_result_and_seal", "object_trust_evidence", "scene_route"],
        ["generic_scene_frontend_packet_builder", "same_source_frontend_spec",
         "scene_frame_map", "mask_result_and_seal", "clean_result_and_seal",
         "object_trust_evidence", "scene_route"],
        [scene_rgb, camera], "T3_six_session_join_only; no generic packet_value",
        "FIXED_T0_SIX_SESSION_SCENE_JOIN_NOT_GENERIC_RUNNER")
    return {"schema_version": "exact78-source-neutral-stage-packet-plan-prep-v1",
            "status": "PREP_ONLY_BLOCKED_NO_PACKET", "source_kind": kind,
            "session_root": source["session_root"], "source_session": source["session_id"],
            "submission_input_sha256": signature,
            "code_acceptance": {"receipt": accepted["receipt"],
                                "result": accepted.get("result")},
            "source_file_roles": {name: roles[name] for name in sorted(roles)},
            "time_mode": source["time_mode"],
            "capture_to_decode_alignment": source["capture_to_decode_alignment"],
            "physical_eye": source["physical_eye"],
            "stages": [cpu, hawor, robot, scene],
            "execution_allowed": False, "model_calls": 0,
            "visual_qualified": False, "training_eligible": False,
            "control_ground_truth": False}


def plan_raw(session: Path) -> dict[str, Any]:
    decision = plan_one(session)
    return plan_source(decision["source"], decision["submission_input_sha256"])


__all__ = ["plan_raw", "plan_source"]
