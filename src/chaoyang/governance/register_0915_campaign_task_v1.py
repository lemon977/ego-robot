#!/usr/bin/env python3
"""CAS-register exactly one finite 0915 campaign task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.campaign_0915_task_specs_v1 import (
    PLAN_REVISION,
    TASK_ORDER,
    build_packet,
    predecessor_task,
)
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
FOUNDATION_TASK_ID = "0915_foundationstereo_single_session_canary_v1"
FOUNDATION_SESSION_ID = "play_cards_0915_001"
FOUNDATION_WEIGHT = (
    "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
)
FOUNDATION_OBJECT6D_SCOPE = "VISUAL_OBJECT6D_CANDIDATE_INPUT"
FOUNDATION_RUNTIME_CLOSURE = (
    REPO_ROOT / "tasks/receipts/FOUNDATIONSTEREO_RUNTIME_CLOSURE_V1.json"
)
VST_ENCODED_DOMAIN_CONFIRMATION = (
    REPO_ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
)
FOUNDATION_OBJECT6D_USER_CONFIRMATION = (
    REPO_ROOT
    / "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json"
)
ENCODED_FOUNDATION_TASK_ID = "0915_foundationstereo_encoded_domain_canary_v1"
ENCODED_DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound_file(
    owner: dict[str, Any], field: str, root: Path, filename: str,
) -> tuple[Path, dict[str, Any]]:
    reference = owner.get(field)
    if not isinstance(reference, dict):
        raise RuntimeError(f"Object6D depth admission lacks bound {field}")
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError(
            f"Object6D depth admission {field} artifact drift: " + "; ".join(errors)
        )
    path = Path(str(reference["path"]))
    expected = (root / filename).resolve()
    if path.resolve() != expected:
        raise RuntimeError(
            f"Object6D depth admission {field} is outside the depth attempt"
        )
    return path, reference


def validate_foundationstereo_object6d_admission(
    result_path: Path,
) -> dict[str, Any]:
    """Validate the complete immutable FoundationStereo evidence closure.

    This is deliberately stricter than a terminal-status check.  A passing depth
    result is consumable by the visual Object6D canary only when its model,
    runtime, writer fence, GPU execution, exact T0 calibration, quality summary,
    depth semantics and nonlinear registration map all agree.
    """

    result_path = result_path.resolve(strict=True)
    root = result_path.parent
    result = load_json(result_path)
    if (
        result.get("schema_version")
        != "0915-foundationstereo-single-session-canary-result-v1"
        or result.get("task_id") != FOUNDATION_TASK_ID
        or result.get("session_id") != FOUNDATION_SESSION_ID
        or result.get("status") != "PASSED"
        or result.get("depth_admission") != "PASS"
        or result.get("first_blocker") is not None
        or result.get("external_accuracy") != "UNVERIFIED"
        or result.get("consumption_authorized") is not True
        or result.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
        or result.get("weights") != [FOUNDATION_WEIGHT]
        or result.get("source_mutated") is not False
    ):
        raise RuntimeError("Object6D depth admission terminal authority mismatch")

    paths: dict[str, Path] = {}
    refs: dict[str, dict[str, Any]] = {}
    for field, filename in {
        "calibration": "CALIBRATION.json",
        "registration": "REGISTRATION.json",
        "registration_maps": "REGISTRATION_MAPS.npz",
        "depth_contract": "DEPTH_CONTRACT.json",
        "depth_summary": "DEPTH_SUMMARY.json",
        "gpu_command_receipt": "GPU_COMMAND_RECEIPT.json",
        "run_signature": "RUN_SIGNATURE.json",
        "writer_claim": "CLAIM.json",
    }.items():
        paths[field], refs[field] = _bound_file(result, field, root, filename)

    gpu = load_json(paths["gpu_command_receipt"])
    if (
        gpu.get("schema_version") != "v71-gpu-command-receipt-v1"
        or gpu.get("status") != "PASSED"
        or gpu.get("task_id") != FOUNDATION_TASK_ID
        or gpu.get("attempt_id") != root.name
        or gpu.get("returncode") != 0
        or gpu.get("error") is not None
    ):
        raise RuntimeError("Object6D depth admission GPU receipt mismatch")

    signature = load_json(paths["run_signature"])
    signature_digest = signature.get("run_signature_sha256")
    unsigned_signature = {
        key: value for key, value in signature.items()
        if key != "run_signature_sha256"
    }
    if (
        signature.get("schema_version") != "0915-foundationstereo-run-signature-v1"
        or signature.get("task_id") != FOUNDATION_TASK_ID
        or signature.get("session_id") != FOUNDATION_SESSION_ID
        or signature.get("weights") != [FOUNDATION_WEIGHT]
        or not isinstance(signature.get("executor_epoch"), int)
        or signature.get("executor_epoch", 0) < 1
        or signature_digest != _canonical_sha256(unsigned_signature)
    ):
        raise RuntimeError("Object6D depth admission run signature mismatch")

    runtime_ref = signature.get("runtime", {}).get("runtime_closure_receipt")
    expected_runtime_ref = artifact_ref(FOUNDATION_RUNTIME_CLOSURE)
    if runtime_ref != expected_runtime_ref or validate_artifact_ref(runtime_ref):
        raise RuntimeError("Object6D depth admission runtime closure is not pinned")
    runtime = load_json(FOUNDATION_RUNTIME_CLOSURE)
    if (
        runtime.get("schema_version") != "foundationstereo-runtime-closure-v1"
        or runtime.get("status") != "PASS_BOUNDED_CANARY_RUNTIME_IDENTITY_CLOSED"
        or runtime.get("task_id") != FOUNDATION_TASK_ID
        or runtime.get("execution_performed") is not False
        or runtime.get("gpu_used") is not False
        or runtime.get("batch_authorized") is not False
        or runtime.get("external_accuracy") != "UNVERIFIED"
        or runtime.get("consumer_policy") != {
            "quality_pass_required": True,
            "authorized_scopes": [FOUNDATION_OBJECT6D_SCOPE],
            "all_other_consumers_authorized": False,
        }
        or runtime.get("output_schema_identities") != signature.get("schema_identity")
    ):
        raise RuntimeError("Object6D depth admission runtime authority mismatch")

    claim = load_json(paths["writer_claim"])
    task_packet_ref = signature.get("input_manifest", {}).get("task_packet")
    if not isinstance(task_packet_ref, dict) or validate_artifact_ref(task_packet_ref):
        raise RuntimeError("Object6D depth admission task packet is not pinned")
    if (
        claim.get("schema_version") != "0915-foundationstereo-writer-claim-v1"
        or claim.get("task_id") != FOUNDATION_TASK_ID
        or claim.get("session_id") != FOUNDATION_SESSION_ID
        or claim.get("attempt_id") != root.name
        or claim.get("weights") != [FOUNDATION_WEIGHT]
        or claim.get("status") != "CLAIMED"
        or claim.get("executor_epoch") != signature.get("executor_epoch")
        or claim.get("run_signature_sha256") != signature_digest
        or claim.get("unique_write_root") != str(root)
        or claim.get("task_packet") != task_packet_ref
        or not isinstance(claim.get("pid"), int)
        or not isinstance(claim.get("proc_start_ticks"), int)
        or not isinstance(claim.get("fencing_token_sha256"), str)
        or len(claim.get("fencing_token_sha256", "")) != 64
    ):
        raise RuntimeError("Object6D depth admission writer claim/signature mismatch")

    summary = load_json(paths["depth_summary"])
    frames = summary.get("frames")
    quality = summary.get("quality")
    quality_gates = quality.get("gates") if isinstance(quality, dict) else None
    review_decode = summary.get("review_decode")
    if (
        summary.get("schema_version") != "0915-foundationstereo-depth-summary-v1"
        or summary.get("task_id") != FOUNDATION_TASK_ID
        or summary.get("session_id") != FOUNDATION_SESSION_ID
        or summary.get("status") != "PASS"
        or summary.get("depth_admission") != "PASS"
        or summary.get("external_accuracy") != "UNVERIFIED"
        or summary.get("consumption_authorized") is not True
        or summary.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
        or summary.get("frame_count") != 150
        or summary.get("full_sbs_decode") is not True
        or summary.get("model_load_count") != 1
        or summary.get("model_inference_count") != 300
        or not isinstance(frames, list)
        or len(frames) != 150
        or [row.get("frame") for row in frames] != list(range(150))
        or not isinstance(quality, dict)
        or quality.get("passed") is not True
        or not isinstance(quality_gates, dict)
        or set(quality_gates) != {
            "full_decode",
            "formula_recompute",
            "geometric_validity",
            "lr_testable_coverage",
            "lr_consistency",
            "final_validity",
            "temporal_distribution_stability",
            "rgb_edge_support",
        }
        or not all(value is True for value in quality_gates.values())
        or not isinstance(review_decode, dict)
        or review_decode.get("full_decode") is not True
        or review_decode.get("frame_count") != 150
        or summary.get("source_mutated") is not False
    ):
        raise RuntimeError("Object6D depth admission summary/quality mismatch")

    t0_ref = signature.get("input_manifest", {}).get("t0_stereo_preflight")
    if not isinstance(t0_ref, dict) or validate_artifact_ref(t0_ref):
        raise RuntimeError("Object6D depth admission T0 preflight is not pinned")
    t0 = load_json(Path(t0_ref["path"]))
    calibrated = t0.get("candidates", {}).get("calibrated_rectified", {})
    selected = "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED"
    calibration = load_json(paths["calibration"])
    calibration_identity = signature.get("calibration_identity")
    if (
        t0.get("status") != "PASS_GPU_DEPTH_ADMISSION"
        or t0.get("session_id") != FOUNDATION_SESSION_ID
        or t0.get("decision", {}).get("gpu_depth_allowed") is not True
        or t0.get("decision", {}).get("selected_candidate") != selected
        or calibrated.get("gpu_eligible") is not True
        or calibrated.get("quality", {}).get("passed") is not True
        or not isinstance(calibrated.get("calibration"), dict)
        or calibration.get("schema_version")
        != "0915-foundationstereo-frozen-calibration-v1"
        or calibration.get("source") != t0_ref
        or calibration.get("consumption")
        != "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION"
        or calibration.get("selected_domain") != selected
        or calibration.get("calibration") != calibrated.get("calibration")
        or calibration_identity != {
            "selected_domain": selected,
            "calibration_sha256": _canonical_sha256(calibrated.get("calibration")),
            "consumption": "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION",
        }
    ):
        raise RuntimeError("Object6D depth admission exact T0 calibration mismatch")

    contract = load_json(paths["depth_contract"])
    if (
        contract.get("schema_version") != "0915-foundationstereo-depth-contract-v1"
        or contract.get("task_id") != FOUNDATION_TASK_ID
        or contract.get("session_id") != FOUNDATION_SESSION_ID
        or contract.get("frame_count") != 150
        or contract.get("frame_geometry") != [640, 480]
        or contract.get("frame_arrays", {}).get("depth_reference")
        != "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"
        or contract.get("native_model_confidence") != "ABSENT_NOT_FABRICATED"
        or contract.get("occluded_or_hidden_geometry") != "INVALID_NOT_COMPLETED"
        or contract.get("external_accuracy") != "UNVERIFIED"
        or contract.get("consumption_authorized") is not True
        or contract.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
    ):
        raise RuntimeError("Object6D depth admission depth contract mismatch")

    registration = load_json(paths["registration"])
    join = registration.get("depth_to_sam_resize_map")
    if (
        registration.get("schema_version") != "0915-foundationstereo-registration-v1"
        or registration.get("source_domain")
        != "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS"
        or registration.get("depth_reference")
        != "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"
        or registration.get("nonlinear_registration_maps") != refs["registration_maps"]
        or not isinstance(join, dict)
        or join.get("array") != "depth_to_sam_resize_xy"
        or join.get("source")
        != "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS"
        or join.get("target")
        != "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960_PIXEL_CENTERS"
        or join.get("identity_assumed") is not False
        or "IDENTITY_JOIN_FORBIDDEN" not in str(registration.get("mask_join_policy"))
    ):
        raise RuntimeError("Object6D depth admission registration-map mismatch")

    worker_path = root / "DEPTH_WORKER_RESULT.json"
    if not worker_path.is_file():
        raise RuntimeError("Object6D depth admission lacks DEPTH_WORKER_RESULT.json")
    worker = load_json(worker_path)
    if (
        worker.get("schema_version") != "0915-foundationstereo-worker-result-v1"
        or worker.get("status") != "COMPLETED"
        or worker.get("task_id") != FOUNDATION_TASK_ID
        or worker.get("session_id") != FOUNDATION_SESSION_ID
        or worker.get("external_accuracy") != "UNVERIFIED"
        or worker.get("consumption_authorized") is not True
        or worker.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
        or worker.get("run_signature") != refs["run_signature"]
        or worker.get("writer_claim") != refs["writer_claim"]
        or worker.get("calibration") != refs["calibration"]
        or worker.get("registration") != refs["registration"]
        or worker.get("registration_maps") != refs["registration_maps"]
        or worker.get("depth_contract") != refs["depth_contract"]
        or worker.get("depth_summary") != refs["depth_summary"]
        or worker.get("access_contract", {}).get("source_mutated") is not False
    ):
        raise RuntimeError("Object6D depth admission worker/result closure mismatch")
    return {
        "result": result,
        "summary": summary,
        "contract": contract,
        "registration": registration,
        "calibration": calibration,
        "run_signature": signature,
        "writer_claim": claim,
        "gpu_receipt": gpu,
        "references": refs,
    }


def validate_encoded_foundationstereo_object6d_admission(
    result_path: Path,
) -> dict[str, Any]:
    """Admit only the user-confirmed encoded-domain Depth closure to Object6D."""

    result_path = result_path.resolve(strict=True)
    root = result_path.parent
    result = load_json(result_path)
    if (
        result.get("schema_version")
        != "0915-foundationstereo-encoded-canary-result-v1"
        or result.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or result.get("session_id") != FOUNDATION_SESSION_ID
        or result.get("status") != "PASSED"
        or result.get("depth_admission") != "PASS"
        or result.get("first_blocker") is not None
        or result.get("external_accuracy") != "UNVERIFIED"
        or result.get("consumption_authorized") is not True
        or result.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
        or result.get("horizontal_reflection_for_disparity_sign") is not True
        or result.get("lens_undistortion_applied") is not False
        or result.get("output_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or result.get("weights") != [FOUNDATION_WEIGHT]
        or result.get("source_mutated") is not False
    ):
        raise RuntimeError("encoded Object6D depth admission terminal mismatch")

    paths: dict[str, Path] = {}
    refs: dict[str, dict[str, Any]] = {}
    for field, filename in {
        "adapter_contract": "ADAPTER_CONTRACT.json",
        "rgb_alignment_qa": "RGB_ALIGNMENT_QA.json",
        "depth_contract": "DEPTH_CONTRACT.json",
        "depth_summary": "DEPTH_SUMMARY.json",
        "depth_worker_result": "DEPTH_WORKER_RESULT.json",
        "gpu_command_receipt": "GPU_COMMAND_RECEIPT.json",
        "run_signature": "RUN_SIGNATURE.json",
        "writer_claim": "CLAIM.json",
    }.items():
        paths[field], refs[field] = _bound_file(result, field, root, filename)

    adapter = load_json(paths["adapter_contract"])
    if (
        adapter.get("schema_version")
        != "0915-foundationstereo-encoded-adapter-contract-v1"
        or adapter.get("horizontal_reflection_for_disparity_sign") is not True
        or adapter.get("both_eyes_reflected") is not True
        or adapter.get("camera_swap") is not False
        or adapter.get("physical_source_indices") != {"left": 1, "right": 0}
        or adapter.get("lens_undistortion_applied") is not False
        or adapter.get("lens_remap_applied") is not False
        or adapter.get("output_spatial_unflip") is not True
        or adapter.get("output_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or adapter.get("mirrored_principal_point_rule")
        != "cx_mirrored_px = width_px - 1 - cx_physical_px"
    ):
        raise RuntimeError("encoded Object6D adapter contract mismatch")

    alignment = load_json(paths["rgb_alignment_qa"])
    if (
        alignment.get("schema_version")
        != "0915-foundationstereo-pixelwise-rgb-alignment-v1"
        or alignment.get("frame_count") != 150
        or alignment.get("depth_grid_domain")
        != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or alignment.get("maximum_absolute_channel_error") != 0
        or alignment.get("mismatched_pixels") != 0
        or alignment.get("coordinate_roundtrip_max_abs_error_px") != 0.0
        or alignment.get("first_frame_physical_left_depth_rgb_sha256")
        != alignment.get("first_frame_unflipped_model_left_sha256")
    ):
        raise RuntimeError("encoded Object6D pixel-alignment proof mismatch")

    contract = load_json(paths["depth_contract"])
    physical_k = contract.get("physical_left_intrinsics")
    mirrored_k = contract.get("model_mirrored_intrinsics")
    if (
        contract.get("schema_version")
        != "0915-foundationstereo-encoded-depth-contract-v1"
        or contract.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or contract.get("session_id") != FOUNDATION_SESSION_ID
        or contract.get("depth_reference") != ENCODED_DEPTH_REFERENCE
        or contract.get("frame_count") != 150
        or contract.get("frame_geometry") != [640, 480]
        or contract.get("depth_to_physical_left_rgb")
        != "IDENTITY_640x480_AFTER_OUTPUT_UNFLIP"
        or contract.get("depth_to_sam_resize_homography")
        != [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]
        or not isinstance(physical_k, list)
        or not isinstance(mirrored_k, list)
        or abs(float(mirrored_k[0][2]) - (639.0 - float(physical_k[0][2])))
        > 1e-9
        or contract.get("native_model_confidence") != "ABSENT_NOT_FABRICATED"
        or contract.get("occluded_or_hidden_geometry")
        != "INVALID_NOT_COMPLETED"
        or contract.get("external_accuracy") != "UNVERIFIED"
        or contract.get("consumption_authorized") is not True
        or contract.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
    ):
        raise RuntimeError("encoded Object6D depth contract mismatch")

    summary = load_json(paths["depth_summary"])
    quality = summary.get("quality", {})
    gates = quality.get("gates", {}) if isinstance(quality, dict) else {}
    if (
        summary.get("schema_version")
        != "0915-foundationstereo-encoded-depth-summary-v1"
        or summary.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or summary.get("status") != "PASS"
        or summary.get("frame_count") != 150
        or summary.get("model_load_count") != 1
        or summary.get("model_inference_count") != 300
        or quality.get("passed") is not True
        or not gates
        or not all(value is True for value in gates.values())
        or summary.get("source_mutated") is not False
    ):
        raise RuntimeError("encoded Object6D depth summary mismatch")

    signature = load_json(paths["run_signature"])
    signature_digest = signature.get("run_signature_sha256")
    unsigned_signature = {
        key: value for key, value in signature.items()
        if key != "run_signature_sha256"
    }
    if (
        signature.get("schema_version")
        != "0915-foundationstereo-encoded-run-signature-v1"
        or signature.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or signature.get("session_id") != FOUNDATION_SESSION_ID
        or signature.get("weights") != [FOUNDATION_WEIGHT]
        or signature.get("adapter", {}).get("model_adapter")
        != "SIMULTANEOUS_HORIZONTAL_REFLECTION_NO_CAMERA_SWAP"
        or signature.get("adapter", {}).get("output_transform")
        != "HORIZONTAL_UNFLIP_TO_PHYSICAL_LEFT"
        or signature.get("pixel_domain", {}).get("lens_undistortion_applied")
        is not False
        or signature_digest != _canonical_sha256(unsigned_signature)
    ):
        raise RuntimeError("encoded Object6D run signature mismatch")

    claim = load_json(paths["writer_claim"])
    if (
        claim.get("schema_version")
        != "0915-foundationstereo-encoded-writer-claim-v1"
        or claim.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or claim.get("status") != "CLAIMED"
        or claim.get("run_signature_sha256") != signature_digest
        or claim.get("unique_write_root") != str(root)
        or claim.get("weights") != [FOUNDATION_WEIGHT]
    ):
        raise RuntimeError("encoded Object6D writer claim mismatch")

    gpu = load_json(paths["gpu_command_receipt"])
    worker = load_json(paths["depth_worker_result"])
    if (
        gpu.get("schema_version") != "v71-gpu-command-receipt-v1"
        or gpu.get("status") != "PASSED"
        or gpu.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or gpu.get("returncode") != 0
        or gpu.get("error") is not None
        or worker.get("schema_version")
        != "0915-foundationstereo-encoded-worker-result-v1"
        or worker.get("task_id") != ENCODED_FOUNDATION_TASK_ID
        or worker.get("status") != "COMPLETED"
        or worker.get("consumption_authorized") is not True
        or worker.get("authorized_scopes") != [FOUNDATION_OBJECT6D_SCOPE]
        or worker.get("source_mutated") is not False
        or worker.get("adapter_contract") != refs["adapter_contract"]
        or worker.get("rgb_alignment_qa") != refs["rgb_alignment_qa"]
        or worker.get("depth_contract") != refs["depth_contract"]
        or worker.get("depth_summary") != refs["depth_summary"]
    ):
        raise RuntimeError("encoded Object6D GPU/worker closure mismatch")

    frames = sorted((root / "frames").glob("*.npz"))
    if len(frames) != 150 or [path.stem for path in frames] != [
        f"{index:06d}" for index in range(150)
    ]:
        raise RuntimeError("encoded Object6D immutable depth frame axis mismatch")
    return {
        "result": result,
        "adapter": adapter,
        "alignment": alignment,
        "contract": contract,
        "summary": summary,
        "run_signature": signature,
        "writer_claim": claim,
        "gpu_receipt": gpu,
        "worker": worker,
        "references": refs,
        "frames": frames,
    }


def _validate_materialized_read_closure(
    packet: dict[str, Any], *, repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Reject a sole executable route whose declared inputs are not present."""
    errors: list[str] = []
    for raw in packet.get("read_set", []):
        path = Path(str(raw))
        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.exists():
            errors.append(f"read_set path is not materialized: {raw}")
    weights = packet.get("weights")
    if isinstance(weights, list):
        for raw in weights:
            path = Path(str(raw))
            resolved = path if path.is_absolute() else repo_root / path
            if not resolved.is_file():
                errors.append(f"weight is not materialized: {raw}")
    return errors


def _validate_predecessor(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    if state.get("next_task") is not None:
        raise RuntimeError("registration requires no current next_task")
    live = [row for row in state.get("tasks", []) if row.get("status") in LIVE]
    if live:
        raise RuntimeError("registration requires no active or pending task")
    if any(row.get("task_id") == task_id for row in state.get("tasks", [])):
        raise RuntimeError(f"task already registered: {task_id}")
    predecessor_id = predecessor_task(task_id)
    predecessor = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == predecessor_id),
        None,
    )
    if predecessor is None:
        raise RuntimeError(f"predecessor task is absent: {predecessor_id}")
    if task_id == TASK_ORDER[0]:
        if predecessor.get("status") not in {"PASSED", "FAILED_RUNTIME_FINAL"}:
            raise RuntimeError("input preparation predecessor is not terminal")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("input preparation predecessor result is not bound")
        result = load_json(Path(result_ref["path"]))
        counts = result.get("counts", {})
        if counts.get("0916_sessions") != 240 or counts.get("0916_failed") != 0:
            raise RuntimeError("0916 cleaning did not close 240 sessions with zero runtime failure")
    elif task_id == "0915_input_prepare_cad_v2":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("corrective v2 requires terminal v1 CLI routing failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("corrective v2 predecessor result is not bound")
        result_path = Path(result_ref["path"])
        result = load_json(result_path)
        audit_log = result_path.parent / "AUDIT.log"
        expected_error = (
            "operation is not maintained by the current algorithm contract: "
            "audit_0915_processed_self_containment_v2"
        )
        if (
            result.get("returncodes") != {"audit": 1}
            or result.get("self_containment") is not None
            or result.get("prepared_manifest") is not None
            or not audit_log.is_file()
            or audit_log.read_text(encoding="utf-8").strip() != expected_error
        ):
            raise RuntimeError("v1 failure is not the exact pre-execution CLI route blocker")
    elif task_id == "0915_vst_image_domain_ab_v1":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("VST image-domain research requires the HaWoR task to be cancelled")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("cancelled HaWoR result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != "0915_hawor_full_v1"
            or result.get("status") != "CANCELLED"
            or result.get("first_blocker")
            != "VST_IMAGE_DOMAIN_UNRESOLVED_POSSIBLE_REDUNDANT_UNDISTORTION"
        ):
            raise RuntimeError("VST research predecessor is not the exact user image-domain hold")
    elif task_id == "0915_hawor_resize_only_canary_v1":
        if predecessor.get("status") != "BLOCKED_EXTERNAL":
            raise RuntimeError("HaWoR resize-only canary requires the closed A/B review")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("A/B result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("conclusion", {}).get("candidate_mono_domain")
            != "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY"
            or result.get("conclusion", {}).get("current_remap_baseline_admissible") is not False
        ):
            raise RuntimeError("A/B result does not bind the confirmed candidate")
        confirmation_path = REPO_ROOT / "tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json"
        confirmation = load_json(confirmation_path)
        if (
            confirmation.get("status") != "CONFIRMED"
            or confirmation.get("confirmed_candidate")
            != "A_PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY"
            or confirmation.get("authorized_next_scope") != "ONE_SESSION_HAWOR_CANARY_ONLY"
        ):
            raise RuntimeError("user confirmation does not authorize this bounded canary")
    elif task_id == "0915_sam31_mask_full_v1":
        raise RuntimeError(
            "SAM3.1 full batch remains blocked pending separate batch authorization"
        )
    elif task_id == "0915_sam31_strict_role_canary_v2":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("strict-role v2 requires the terminal read-set correction")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v1 cancellation result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("execution_started") is not False
            or result.get("gpu_started") is not False
            or result.get("first_blocker")
            != "TASK_PACKET_READ_SET_OMITS_ACCEPTED_BOUNDED_HAWOR_SUCCESSOR"
        ):
            raise RuntimeError("strict-role v1 was not the exact pre-execution read-set correction")
    elif task_id == "0915_sam31_strict_role_canary_v3":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("strict-role v3 requires the terminal v2 import-path failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v2 failure result is not bound")
        result = load_json(Path(result_ref["path"]))
        gpu_ref = result.get("gpu_command_receipt")
        if not isinstance(gpu_ref, dict) or validate_artifact_ref(gpu_ref):
            raise RuntimeError("strict-role v2 GPU failure receipt is not bound")
        gpu = load_json(Path(gpu_ref["path"]))
        if (
            result.get("session_admission") != "NOT_PRODUCED"
            or gpu.get("returncode") != 1
            or "ModuleNotFoundError: No module named 'sam3'"
            not in str(gpu.get("stderr_tail"))
        ):
            raise RuntimeError("strict-role v2 was not the exact pre-model vendor import failure")
    elif task_id == "0915_sam31_strict_role_canary_v4":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("strict-role v4 requires the terminal v3 early diagnostic")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v3 cancellation result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("completed_instances") != 2
            or result.get("session_admission") != "NOT_PRODUCED"
            or result.get("first_blocker")
            != "POINT_REFINEMENT_PROPAGATION_ID_OR_ROUTE_NOT_CONTINUOUS"
        ):
            raise RuntimeError("strict-role v3 was not the exact two-hand early diagnostic")
    elif task_id == "0915_sam31_strict_role_canary_v5":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("strict-role v5 requires the terminal v4 direction failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v4 failure result is not bound")
        result = load_json(Path(result_ref["path"]))
        gpu_ref = result.get("gpu_command_receipt")
        if not isinstance(gpu_ref, dict) or validate_artifact_ref(gpu_ref):
            raise RuntimeError("strict-role v4 GPU failure receipt is not bound")
        gpu = load_json(Path(gpu_ref["path"]))
        if (
            result.get("session_admission") != "NOT_PRODUCED"
            or "RuntimeError: No points are provided; please add points first"
            not in str(gpu.get("stderr_tail"))
            or '"completed_instances": 2' not in str(gpu.get("stdout_tail"))
        ):
            raise RuntimeError("strict-role v4 was not the exact direction-level exception")
    elif task_id == "0915_stereo_interaction_cpu_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("CPU evidence canary requires the completed strict-role v5")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_PARALLEL_CANARIES_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_session") != "play_cards_0915_001"
            or "STEREO_DOMAIN_CPU_PREFLIGHT"
            not in authorization.get("authorized_scope", [])
            or "INTERACTION_V0A_IMAGE_2D_ONLY"
            not in authorization.get("authorized_scope", [])
        ):
            raise RuntimeError("bounded CPU evidence lacks exact user authorization")
    elif task_id == "0915_sam31_weak_role_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("weak-role canary requires terminal CPU evidence routing")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("CPU evidence predecessor result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("lane_fences_valid") is not True
            or result.get("gpu_used") is not False
            or result.get("session_id") != "play_cards_0915_001"
        ):
            raise RuntimeError("CPU evidence predecessor did not close both fenced lanes")
    elif task_id == "0915_stereo_encoded_domain_preflight_v1":
        if predecessor.get("status") != "REJECTED_QUALITY":
            raise RuntimeError(
                "encoded-domain preflight requires the historical wrong-domain "
                "FoundationStereo terminal"
            )
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("historical FoundationStereo result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != FOUNDATION_TASK_ID
            or result.get("status") != "REJECTED_QUALITY"
            or result.get("consumption_authorized") is not False
        ):
            raise RuntimeError("historical FoundationStereo terminal identity mismatch")
        confirmation = load_json(VST_ENCODED_DOMAIN_CONFIRMATION)
        if (
            confirmation.get("status")
            != "CONFIRMED_ENCODED_VIDEO_ALREADY_UNDISTORTED"
            or confirmation.get("admitted_visual_transform")
            != "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY"
            or "EQUIDIS62_TO_PINHOLE"
            not in confirmation.get("prohibited_operations", [])
        ):
            raise RuntimeError("VST encoded-video domain confirmation is invalid")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_CPU_NEXT_TASKS_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_session") != FOUNDATION_SESSION_ID
            or authorization.get("authorized_tasks_in_order", [None])[0] != task_id
            or authorization.get("execution_policy", {}).get("gpu_allowed") is not False
            or authorization.get("execution_policy", {}).get("lens_undistortion_allowed")
            is not False
        ):
            raise RuntimeError("encoded-domain preflight lacks exact user authorization")
    elif task_id == "0915_removal_envelope_v2_real_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Removal V2 real canary requires terminal CPU routing")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("encoded-domain preflight result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != "0915_stereo_encoded_domain_preflight_v1"
            or result.get("status") != "PASSED"
            or result.get("gpu_used") is not False
        ):
            raise RuntimeError("Removal V2 routing predecessor did not close safely")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_CPU_NEXT_TASKS_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_session") != FOUNDATION_SESSION_ID
            or authorization.get("authorized_tasks_in_order", [None, None])[1]
            != task_id
            or authorization.get("execution_policy", {}).get("sam_rerun_allowed")
            is not False
            or authorization.get("execution_policy", {}).get("inpaint_allowed")
            is not False
        ):
            raise RuntimeError("Removal V2 real canary lacks exact user authorization")
    elif task_id == "0915_foundationstereo_encoded_domain_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError(
                "encoded-domain FoundationStereo requires the passed encoded-domain "
                "stereo preflight"
            )
        predecessor_ref = predecessor.get("result")
        if not isinstance(predecessor_ref, dict) or validate_artifact_ref(predecessor_ref):
            raise RuntimeError("encoded-domain stereo preflight result is not bound")
        stereo_result = load_json(Path(predecessor_ref["path"]))
        if (
            stereo_result.get("task_id")
            != "0915_stereo_encoded_domain_preflight_v1"
            or stereo_result.get("status") != "PASSED"
            or stereo_result.get("preflight_decision")
            != "PASS_DIRECT_FOUNDATION_INPUT"
            or stereo_result.get("gpu_successor_authorized") is not True
            or stereo_result.get("source_mutated") is not False
        ):
            raise RuntimeError("encoded-domain stereo preflight did not authorize GPU Depth")

        confirmation = load_json(FOUNDATION_OBJECT6D_USER_CONFIRMATION)
        adapter = confirmation.get("foundationstereo_adapter", {})
        if (
            confirmation.get("status") != "CONFIRMED"
            or confirmation.get("authorized_session") != FOUNDATION_SESSION_ID
            or confirmation.get("authorized_tasks_in_order", [None])[0] != task_id
            or adapter.get("physical_left_source_index") != 1
            or adapter.get("physical_right_source_index") != 0
            or adapter.get("swap_physical_cameras") is not False
            or adapter.get("horizontal_reflection_for_disparity_sign") is not True
            or adapter.get("reflect_both_eyes") is not True
            or adapter.get("output_spatial_unflip_to_physical_left") is not True
            or adapter.get("final_rgb_depth_pixel_alignment_required") is not True
            or adapter.get("lens_undistortion_allowed") is not False
            or adapter.get("lens_remap_allowed") is not False
        ):
            raise RuntimeError(
                "encoded-domain FoundationStereo lacks the exact user-confirmed adapter"
            )
    elif task_id == "0915_planar_object6d_observability_canary_v2":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError(
                "Object6D observability v2 requires the encoded Depth terminal"
            )
        depth_ref = predecessor.get("result")
        if not isinstance(depth_ref, dict) or validate_artifact_ref(depth_ref):
            raise RuntimeError("encoded Depth result is not bound")
        validate_encoded_foundationstereo_object6d_admission(
            Path(depth_ref["path"])
        )
        confirmation = load_json(FOUNDATION_OBJECT6D_USER_CONFIRMATION)
        entities = confirmation.get("object6d_entities", {})
        if (
            confirmation.get("status") != "CONFIRMED"
            or confirmation.get("authorized_session") != FOUNDATION_SESSION_ID
            or confirmation.get("authorized_tasks_in_order", [None, None])[1]
            != task_id
            or entities.get("task_objects")
            != ["playing_card_00", "playing_card_01", "playing_card_02"]
            or entities.get("support_entities") != ["black_card_tray"]
            or entities.get("card_and_tray_single_rigid_body") is not False
            or entities.get("semantic_groups", {}).get("card_set", {}).get(
                "rigid_pose_authority"
            ) != "NONE"
            or entities.get("independent_observability_fields")
            != ["center_xyz", "plane_normal", "inplane_rotation", "full_extent"]
            or entities.get("unknown_must_not_be_completed") is not True
            or entities.get("measured_card_dimensions")
            != "ABSENT_PENDING_USER_MEASUREMENT"
        ):
            raise RuntimeError(
                "Object6D observability v2 lacks the exact entity/observability lock"
            )
    elif task_id == "0915_interaction_contact_robot_dev_v2":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("Interaction/Contact/Robot V2 requires the V1 enum-only early stop")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("Interaction/Contact/Robot V1 failure is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("first_blocker")
            != "HAWOR_PROVENANCE_ENUM_ASSUMPTION_MISMATCH_BEFORE_R0"
            or result.get("execution_scope_reached") != "INPUT_VALIDATION_ONLY"
            or result.get("algorithm_outputs_produced") is not False
            or result.get("model_rerun_performed") is not False
        ):
            raise RuntimeError("V1 was not the exact pre-R0 provenance-enum failure")
    elif task_id == "0915_interaction_contact_robot_dev_v3":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("Interaction/Contact/Robot V3 requires the V2 schema-only stop")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("Interaction/Contact/Robot V2 failure is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("first_blocker")
            != "CARD_DIMENSION_SCHEMA_OMITTED_BOUNDED_BUDGET_FIELDS_AFTER_R0"
            or result.get("execution_scope_reached")
            != "R0_COMPLETED_OBJECT6D_QA_COMPUTED"
            or result.get("r0_output_produced") is not True
            or result.get("model_rerun_performed") is not False
        ):
            raise RuntimeError("V2 was not the exact post-R0 sidecar-schema failure")
    elif task_id == "0915_human_stereo_surface_association_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("surface association requires the terminal V3 development canary")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("Interaction/Contact/Robot V3 result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != "0915_interaction_contact_robot_dev_v3"
            or result.get("development_terminal")
            != "R0_COMPLETE_R1_BLOCKED_LOCAL_EVIDENCE"
            or result.get("r0_status") != "COMPLETED_DEVELOPMENT_BASELINE"
            or result.get("r1_status") != "BLOCKED_LOCAL_EVIDENCE"
            or result.get("model_rerun_performed") is not False
            or result.get("removal_or_clean_consumed") is not False
        ):
            raise RuntimeError("V3 is not the exact frozen R0-complete/R1-blocked predecessor")
    elif task_id == "0915_human_stereo_alignment_bound_audit_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("alignment bound audit requires the terminal surface canary")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("surface-association result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != "0915_human_stereo_surface_association_canary_v1"
            or result.get("alignment_status") != "PASS_DEVELOPMENT_ALIGNMENT"
            or result.get("contact_window_count") != 0
            or result.get("r1_status") != "BLOCKED_LOCAL_EVIDENCE"
            or result.get("model_rerun_performed") is not False
            or result.get("removal_or_clean_consumed") is not False
        ):
            raise RuntimeError("surface canary is not the exact immutable audit predecessor")
    elif task_id == "0915_foundationstereo_single_session_canary_v1":
        confirmation = load_json(VST_ENCODED_DOMAIN_CONFIRMATION)
        if (
            confirmation.get("status")
            != "CONFIRMED_ENCODED_VIDEO_ALREADY_UNDISTORTED"
            or confirmation.get("admitted_visual_transform")
            != "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY"
            or "EQUIDIS62_TO_PINHOLE"
            not in confirmation.get("prohibited_operations", [])
        ):
            raise RuntimeError("VST encoded-video domain confirmation is invalid")
        raise RuntimeError(
            "wrong-domain FoundationStereo task is terminal; register a fresh "
            "encoded-video-domain successor with zero lens-undistortion"
        )
    elif task_id == "0915_removal_envelope_single_session_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Removal Envelope canary requires terminal weak-role execution")
        weak_ref = predecessor.get("result")
        if not isinstance(weak_ref, dict) or validate_artifact_ref(weak_ref):
            raise RuntimeError("Removal Envelope weak-role predecessor is not bound")
        weak_result = load_json(Path(weak_ref["path"]))
        if (
            weak_result.get("session_id") != "play_cards_0915_001"
            or weak_result.get("session_admission") != "AWAITING_USER_VISUAL_REVIEW"
        ):
            raise RuntimeError("Removal Envelope predecessor is not the bounded weak-role run")
        review = load_json(
            REPO_ROOT / "tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json"
        )
        if review.get("status") != "REJECTED_QUALITY_AS_CLEAN_BASELINE":
            raise RuntimeError("Removal Envelope requires the recorded weak-role visual rejection")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_task") != task_id
            or authorization.get("authorized_session") != "play_cards_0915_001"
            or authorization.get("weights") != "ABSENT"
        ):
            raise RuntimeError("Removal Envelope lacks exact user authorization")
    elif task_id == "0915_planar_object6d_single_session_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Object6D canary requires the Depth execution terminal")
        depth_ref = predecessor.get("result")
        if not isinstance(depth_ref, dict) or validate_artifact_ref(depth_ref):
            raise RuntimeError("Depth canary result is not bound")
        validate_foundationstereo_object6d_admission(Path(depth_ref["path"]))
    elif predecessor.get("status") != "PASSED":
        raise RuntimeError(f"predecessor did not pass: {predecessor_id}")
    return predecessor


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_ORDER)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))

    state = load_json(TASK_STATE_PATH)
    predecessor = _validate_predecessor(state, args.task_id)
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    predecessor_index = REPO_ROOT / pointer["index_path"]
    if predecessor_index.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current pointer does not resolve to tasks/current/INDEX.json")
    current_ref = artifact_ref(CURRENT_INDEX)
    if (
        pointer.get("index_bytes") != current_ref["bytes"]
        or pointer.get("index_sha256") != current_ref["sha256"]
    ):
        raise RuntimeError("current task index pointer mismatch")

    packet = build_packet(args.task_id)
    errors = _validate_packet(packet)
    errors.extend(_validate_materialized_read_closure(packet))
    if errors:
        raise RuntimeError("invalid campaign packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / "tasks/current" / args.task_id / "TASK_PACKET.json"
    if packet_path.parent.exists() or packet_path.parent.is_symlink():
        raise RuntimeError(f"fresh task packet directory required: {packet_path.parent}")

    output.mkdir(parents=True)
    frozen = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen)
    _write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": "0915_FULL_FUNNEL_V1",
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen),
        "task_packets": [{
            "task_id": args.task_id,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": packet["weights"],
        }],
        "claim_limit": "Exactly one finite 0915 campaign task is routable; algorithm tasks bind exactly one logical weight.",
    }
    created = now_iso()
    state["tasks"].append({
        "task_id": args.task_id,
        "phase": packet["phase"],
        "plan_execution_revision": "0915_FULL_FUNNEL_V1",
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
    })
    state["next_task"] = {
        "task_id": args.task_id,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "All fixed-denominator inputs reach a recorded terminal without changing quality gates.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": args.task_id,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered one finite 0915 campaign task; execution not started.",
        "task_packet": packet_ref,
        "predecessor": predecessor.get("task_id"),
    }])[-100:]

    result_path = output / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "register-0915-campaign-task-result-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "weights": packet["weights"],
        "execution_started": False,
        "mask_policy": "SAM3.1_ONLY_USER_LOCKED",
        "claim_limit": "Governance registration only; no algorithm or dataset result.",
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{args.task_id.upper()}_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor_index,
    )
    _write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "register-0915-campaign-task-receipt-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    })
    print(json.dumps({
        "status": "PASSED", "task_id": args.task_id,
        "weights": packet["weights"],
        "governance_revision": published["governance_revision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
