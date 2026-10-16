"""Read-only handoff gate for a terminal new-raw HaWoR -> Robot candidate.

The caller must first validate CPU admission with build_spec_from_sealed().
This module never registers a task, reads a previous Robot result, or renders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import (
    PROJECT_ROOT, file_ref, stage_signature,
)


class HandoffError(ValueError):
    """A sealed upstream identity or frame contract is not established."""


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffError(f"JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _ref(ref: Any, expected: Path, label: str) -> Path:
    try:
        valid = (isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}
                 and ref.get("path") == str(expected)
                 and not expected.is_symlink()
                 and expected.resolve(strict=True) == expected
                 and file_ref(expected) == ref)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise HandoffError(f"REF_SHA_OR_PATH_DRIFT:{label}")
    return expected


def _admission_refs(admission: dict[str, Any]) -> dict[str, Any]:
    names = {
        "upstream_admission_descriptor": "descriptor_ref",
        "upstream_admission_frame_time_map": "frame_map_ref",
        "upstream_admission_seal": "admission_seal_ref",
        "upstream_admission_result": "producer_result_ref",
        "upstream_admission_receipt": "producer_receipt_ref",
    }
    refs = {key: admission.get(value) for key, value in names.items()}
    if any(not isinstance(value, dict) for value in refs.values()):
        raise HandoffError("VALIDATED_CPU_ADMISSION_REFS_REQUIRED")
    return refs


def _hawor_signature(packet: dict[str, Any]) -> str:
    inputs = packet["inputs"]
    required = (
        "upstream_admission_descriptor", "upstream_admission_frame_time_map",
        "upstream_admission_seal", "upstream_admission_result",
        "upstream_admission_receipt", "model_python", "native_tracer",
    )
    try:
        return stage_signature(
            inputs=[*(inputs[name] for name in required),
                    *(inputs["source_files"][name]
                      for name in sorted(inputs["source_files"]))],
            code=packet["runtime_code"], config=packet["config"],
            weights=[inputs["model_assets"][name]
                     for name in sorted(inputs["model_assets"])],
            calibration=(inputs["source_files"].get("camera_params")
                         or inputs["source_files"].get("raw/camera_params.json")
                         or "ABSENT"),
            schema_version="exact78-newraw-hawor-stage-v4",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HandoffError("HAWOR_PACKET_SIGNATURE_FIELDS_DRIFT") from exc


def verify_handoff(*, outer_result: Path, admission: dict[str, Any],
                   repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify terminal outer/receipt/seal/native/inner and exact sealed frames.

    Returns only immutable refs and the original raw video path for a separate
    fenced CPU Robot Task Packet. No provisional HaWoR partial is admissible.
    """
    root = repo_root.resolve(strict=True)
    cpu_refs = _admission_refs(admission)
    cpu_result_path = Path(cpu_refs["upstream_admission_result"]["path"])
    _ref(cpu_refs["upstream_admission_result"], cpu_result_path, "cpu_result")
    cpu = _object(cpu_result_path)
    session = admission.get("session_id")
    if (not isinstance(session, str) or cpu.get("source_session") != session
            or cpu.get("status") != "PASSED"
            or not isinstance(cpu.get("descriptor_sha256"), str)):
        raise HandoffError("CPU_ADMISSION_IDENTITY_DRIFT")
    kind = admission.get("source_kind")
    if kind not in {"tracker", "vst0915"}:
        raise HandoffError("UNSUPPORTED_SOURCE_KIND")
    expected_id = f"exact78_newraw_hawor_{session}_{cpu['descriptor_sha256'][:16]}_v4"
    attempt = root / "_run/current" / expected_id / "attempts/attempt_0001"
    outer_path = attempt / "RESULT.json"
    if outer_result != outer_path or outer_path.is_symlink():
        raise HandoffError("HAWOR_OUTER_ATTEMPT_DRIFT")
    outer_ref = file_ref(outer_path)
    outer = _object(outer_path)
    if (outer.get("schema_version") != "exact78-newraw-hawor-gpu-execution-result-v4"
            or outer.get("task_id") != expected_id
            or outer.get("status") != "PASSED"
            or outer.get("failure_code") is not None
            or outer.get("algorithm_attempts") != 1
            or outer.get("stage_status") != "SEALED_CAMERA_PREDICTIONS"
            or outer.get("source_sha_stable_at_terminal") is not True
            or outer.get("external_write_open_observed") is not False
            or outer.get("source_session") != session
            or outer.get("source_frames_expected") != admission.get("frame_count")
            or outer.get("producer_result") != cpu_refs["upstream_admission_result"]
            or outer.get("visual_qualified") is not False
            or outer.get("training_eligible") is not False):
        raise HandoffError("HAWOR_OUTER_NOT_TERMINAL_PASSED")

    receipt_path = root / "tasks/receipts" / f"{expected_id.upper()}.json"
    receipt_ref = file_ref(receipt_path)
    receipt = _object(receipt_path)
    if (receipt.get("schema_version") != "exact78-newraw-hawor-gpu-receipt-v4"
            or receipt.get("task_id") != expected_id or receipt.get("status") != "PASSED"
            or receipt.get("result") != outer_ref):
        raise HandoffError("HAWOR_RECEIPT_NOT_PASSED")
    packet_path = root / "tasks/current" / expected_id / "TASK_PACKET.json"
    packet_ref = file_ref(packet_path)
    packet = _object(packet_path)
    if (packet.get("task_id") != expected_id
            or packet.get("phase") != "EXACT78_NEWRAW_HAWOR_GPU_CANDIDATE_V4"
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise HandoffError("HAWOR_PACKET_IDENTITY_DRIFT")
    inputs = packet.get("inputs")
    source_files = cpu.get("source_files")
    video_role = "video" if kind == "tracker" else "raw/vst.h264"
    if (not isinstance(source_files, dict) or video_role not in source_files
            or not isinstance(inputs, dict)
            or any(inputs.get(key) != ref for key, ref in cpu_refs.items())
            or inputs.get("source_session") != session
            or inputs.get("source_files") != source_files
            or inputs.get("raw_video") != source_files[video_role]
            or inputs.get("raw_video", {}).get("path") != str(admission.get("source_video"))
            or packet.get("config", {}).get("frame_count") != admission.get("frame_count")
            or packet.get("config", {}).get("source_kind") != admission.get("source_kind")):
        raise HandoffError("HAWOR_PACKET_CPU_BINDING_DRIFT")
    for role, ref in inputs.get("source_files", {}).items():
        path = Path(ref["path"])
        _ref(ref, path, f"raw:{role}")
    state = _object(root / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == expected_id]
    if (len(rows) != 1 or rows[0].get("status") != "PASSED"
            or rows[0].get("task_packet") != packet_ref
            or rows[0].get("result") != outer_ref
            or rows[0].get("terminal_receipt") != receipt_ref
            or rows[0].get("run_signature") != packet.get("run_signature")):
        raise HandoffError("HAWOR_NOT_TERMINAL_IN_LEDGER")

    stage = outer.get("stage")
    if not isinstance(stage, dict) or set(stage) != {
            "result", "native_open_audit", "prediction_counts",
            "native_write_open_counts", "seal"}:
        raise HandoffError("HAWOR_STAGE_FIVE_FIELDS_REQUIRED")
    inner_path = attempt / "candidate/hawor/RESULT.json"
    native_path = attempt / "candidate/NATIVE_OPEN_AUDIT.json"
    seal_path = attempt / "candidate/HAWOR_STAGE_SEAL.json"
    _ref(stage["result"], inner_path, "inner")
    _ref(stage["native_open_audit"], native_path, "native_open")
    _ref(stage["seal"], seal_path, "stage_seal")
    native, seal, inner = _object(native_path), _object(seal_path), _object(inner_path)
    _ref(native.get("trace"), attempt / "candidate/native_open_trace.log", "native_trace")
    native_lists = (
        "failed_external_write_open_attempts",
        "unresolved_external_write_open_returns",
        "successful_persistent_external_writable_opens",
        "successful_transient_kernel_ipc_writable_opens",
        "successful_runtime_device_writable_opens",
    )
    counts = native.get("write_open_counts", {})
    if (native.get("schema_version") != "exact78-newraw-hawor-native-open-audit-v4"
            or native.get("status") != "PASS_NATIVE_OPEN_SCOPE"
            or native.get("source_video_open_observed") is not True
            or native.get("forbidden_paths") != []
            or native.get("successful_persistent_external_writable_opens") != []
            or native.get("permitted_source_refs") != inputs["source_files"]
            or str(admission["source_video"]) not in native.get("opened_external_paths", [])
            or any(not isinstance(native.get(name), list) for name in native_lists)
            or counts != {
                "failed_external_attempts": len(native.get(native_lists[0], [])),
                "unresolved_external_returns": len(native.get(native_lists[1], [])),
                "successful_persistent_external": len(native.get(native_lists[2], [])),
                "successful_transient_kernel_ipc": len(native.get(native_lists[3], [])),
                "successful_runtime_device": len(native.get(native_lists[4], [])),
            }
            or stage["native_write_open_counts"] != counts
            or outer.get("transient_kernel_ipc_writable_opens")
               != len(native["successful_transient_kernel_ipc_writable_opens"])):
        raise HandoffError("HAWOR_NATIVE_OPEN_SCOPE_DRIFT")
    if (seal.get("schema_version") != "exact78-newraw-hawor-stage-seal-v4"
            or seal.get("status") != "SEALED_CAMERA_PREDICTIONS"
            or seal.get("task_id") != expected_id
            or seal.get("session_id") != session
            or seal.get("downstream_use") != "SEALED_STAGE_ONLY"
            or seal.get("signature") != _hawor_signature(packet)
            or seal.get("result") != stage["result"]
            or seal.get("native_open_audit") != stage["native_open_audit"]):
        raise HandoffError("HAWOR_STAGE_SEAL_DRIFT")
    if (inner.get("schema_version") != "exact78-hawor-pure-rgb-result-v3"
            or inner.get("status") != "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS"
            or inner.get("session_id") != session
            or inner.get("source_kind") != admission.get("source_kind")
            or inner.get("source_frames_decoded") != admission.get("frame_count")
            or inner.get("model_input_frames") != admission.get("frame_count")
            or inner.get("world_pose_valid") is not False
            or inner.get("training_eligible") is not False):
        raise HandoffError("HAWOR_INNER_IDENTITY_DRIFT")
    inner_root = inner_path.parent
    outputs = {
        "npz": inner_root / "HAWOR_CAMERA_PREDICTIONS.npz",
        "frame_provenance": inner_root / "FRAME_PROVENANCE.json",
        "input_audit": inner_root / "INPUT_AUDIT.json",
    }
    for name, path in outputs.items():
        _ref(inner.get(name), path, name)
    provenance = _object(outputs["frame_provenance"])
    audit = _object(outputs["input_audit"])
    rows = provenance.get("frame_rows")
    expected_rows = admission.get("frame_rows_sealed")
    count = admission["frame_count"]
    if (provenance.get("schema_version") != "exact78-hawor-frame-provenance-v3"
            or provenance.get("session_id") != session
            or provenance.get("frame_count") != count
            or provenance.get("physical_eye") != admission.get("physical_eye")
            or provenance.get("time_mode") != admission.get("time_mode")
            or not isinstance(rows, list) or len(rows) != count
            or not isinstance(expected_rows, list) or len(expected_rows) != count):
        raise HandoffError("HAWOR_PROVENANCE_IDENTITY_DRIFT")
    for index, (row, original) in enumerate(zip(rows, expected_rows)):
        if (row.get("decoded_display_index") != index
                or row.get("model_input_index") != index
                or row.get("output_index") != index
                or row.get("sbs_physical_eye") != admission["physical_eye"]
                or row.get("crop_xywh") != original.get("crop_xywh")
                or row.get("time_mode") != admission["time_mode"]
                or any(row.get(key) != original.get(key) for key in (
                    "pts", "best_effort_timestamp", "qpc_raw_tick",
                    "capture_timestamp_raw", "capture_to_decode_alignment"))):
            raise HandoffError(f"HAWOR_FRAME_MAP_DRIFT:{index}")
    if (audit.get("schema_version") != "exact78-hawor-input-audit-v3"
            or audit.get("raw_source_ref") != inputs["raw_video"]
            or audit.get("source_role") != "new_session_candidate"
            or audit.get("source_split_claim") != "NOT_ASSIGNED"
            or audit.get("infiller_invoked") is not False
            or audit.get("world_pose_emitted") is not False):
        raise HandoffError("HAWOR_INPUT_AUDIT_DRIFT")
    with np.load(outputs["npz"], allow_pickle=False) as data:
        shapes = {
            "prediction_valid": (2, count), "joints_3d_camera": (2, count, 21, 3),
            "joints_2d": (2, count, 21, 2), "root_orient_camera": (2, count, 3, 3),
            "hand_pose_rotmat": (2, count, 15, 3, 3), "betas": (2, count, 10),
            "intrinsics": (count, 3, 3),
        }
        if (any(name not in data or data[name].shape != shape
                for name, shape in shapes.items())
                or not np.array_equal(data["original_frame_indices"], np.arange(count))
                or float(data["fps_nominal"]) != admission.get("fps_nominal")):
            raise HandoffError("HAWOR_NPZ_SHAPE_OR_TIMELINE_DRIFT")
        prediction_counts = {side: int(data["prediction_valid"][index].sum())
                             for index, side in enumerate(("left", "right"))}
    if (stage["prediction_counts"] != prediction_counts
            or inner.get("model_prediction_frames") != prediction_counts):
        raise HandoffError("HAWOR_PREDICTION_COUNT_DRIFT")
    return {
        "session_id": session, "source_kind": admission["source_kind"],
        "frame_count": count, "fps_nominal": admission["fps_nominal"],
        "raw_video": inputs["raw_video"], "source_files": inputs["source_files"],
        "admission_refs": cpu_refs, "hawor_outer_result": outer_ref,
        "hawor_receipt": receipt_ref, "hawor_packet": packet_ref,
        "hawor_stage_seal": stage["seal"],
        "hawor_native_open_audit": stage["native_open_audit"],
        "hawor_inner_result": stage["result"],
        "hawor_npz": inner["npz"],
        "hawor_frame_provenance": inner["frame_provenance"],
        "hawor_input_audit": inner["input_audit"],
        "frame_rows_sealed": expected_rows,
        "time_mode": admission["time_mode"],
        "physical_eye": admission["physical_eye"],
    }


__all__ = ["HandoffError", "verify_handoff"]
