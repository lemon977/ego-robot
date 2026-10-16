"""Pure, unregistered HaWoR spec adapter for sealed same-session raw evidence.

This does not replace or invoke the T0-only HaWoR CLI. A future fenced runner
must verify its own active Task Packet and native-open audit before model use.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from chaoyang.cli.exact78_first_submit_bridge_candidate import _admit_known_domain
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT,
    PROJECT_ROOT,
    ColdStartInputError,
    file_ref,
    validate_producer_reads,
)


FRAME_FIELDS = frozenset({
    "decoded_display_index", "pts", "best_effort_timestamp", "pict_type",
    "qpc_raw_tick", "capture_timestamp_raw", "capture_to_decode_alignment",
    "sbs_physical_eye", "crop_xywh",
})
DESCRIPTOR_FIELDS = frozenset({
    "schema_version", "session_id", "session_root", "source_kind", "source_files",
    "raw_video", "submission_input_sha256", "decoded_frames", "fps_nominal",
    "physical_eye", "crop_xywh", "time_mode", "capture_to_decode_alignment",
    "calibration_claim", "independent_source_group", "world_pose_authority",
    "metric_authority", "frame_rows",
})
FRAME_MAP_FIELDS = frozenset({
    "schema_version", "session_id", "source_kind", "physical_eye", "crop_xywh",
    "time_mode", "capture_to_decode_alignment", "decoded_frames", "raw_video", "frame_rows",
})
FORBIDDEN_SOURCE_COMPONENTS = frozenset({
    "preprocess", "all_data", "trackingdata", "processed", "robot", "clean", "pico", "manus",
})
TARGET_SIZE = (1280, 960)  # identical to frozen HaWoR v3 model preprocessing


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ColdStartInputError("SEALED_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise ColdStartInputError("SEALED_JSON_OBJECT_REQUIRED")
    return value


def _project_ref(ref: Any, root: Path) -> tuple[Path, dict[str, Any]]:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ColdStartInputError("SEALED_REF_REQUIRED")
    raw = Path(ref["path"])
    if raw.is_symlink():
        raise ColdStartInputError("SEALED_REF_SYMLINK_FORBIDDEN")
    try:
        path = raw.resolve(strict=True)
    except OSError as exc:
        raise ColdStartInputError("SEALED_REF_MISSING") from exc
    if root not in path.parents or file_ref(path) != ref:
        raise ColdStartInputError("SEALED_REF_SHA_OR_SCOPE_DRIFT")
    return path, _object(path)


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ColdStartInputError(f"{label}_INVALID")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ColdStartInputError(f"{label}_INVALID")
    return number


def _source_from_descriptor(descriptor: dict[str, Any], allowed_root: Path) -> dict[str, Any]:
    if (set(descriptor) != DESCRIPTOR_FIELDS
            or descriptor.get("schema_version") != "exact78-first-submit-descriptor-candidate-v1"):
        raise ColdStartInputError("DESCRIPTOR_SCHEMA_DRIFT")
    raw_root = descriptor.get("session_root")
    if not isinstance(raw_root, str):
        raise ColdStartInputError("SESSION_ROOT_ABSENT")
    session = Path(raw_root)
    if session.is_symlink():
        raise ColdStartInputError("SYMLINK_SESSION_FORBIDDEN")
    session = session.resolve(strict=True)
    if not session.is_dir() or not session.is_relative_to(allowed_root):
        raise ColdStartInputError("SOURCE_OUTSIDE_ALLOWED_ROOT")
    if any(part.lower() in FORBIDDEN_SOURCE_COMPONENTS
           for part in session.relative_to(allowed_root).parts):
        raise ColdStartInputError("OLD_DERIVED_SESSION_FORBIDDEN")
    source = {
        "kind": descriptor.get("source_kind"), "session_id": descriptor.get("session_id"),
        "session_root": str(session), "files": descriptor.get("source_files"),
        "physical_eye": descriptor.get("physical_eye"),
        "crop_xywh": descriptor.get("crop_xywh"), "time_mode": descriptor.get("time_mode"),
    }
    if source["kind"] == "TRACKER_CLIP_RGB":
        expected_id = session.name
    elif source["kind"] == "0915_RAW_VST":
        parent = session.parent.name
        if parent.startswith("cards_") and parent.endswith("_0915"):
            expected_id = f"play_cards_0915_{session.name}"
        elif parent.startswith("chips_") and parent.endswith("_0915"):
            expected_id = f"get_potato_chips_0915_{session.name}"
        else:
            raise ColdStartInputError("VST0915_SOURCE_IDENTITY_UNVERIFIED")
    else:
        raise ColdStartInputError("UNSUPPORTED_SOURCE_DOMAIN")
    if source["session_id"] != expected_id:
        raise ColdStartInputError("SESSION_ID_SOURCE_PATH_DRIFT")
    if not isinstance(source["files"], dict):
        raise ColdStartInputError("SOURCE_ROLE_SET_DRIFT")
    for role, ref in source["files"].items():
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise ColdStartInputError("SOURCE_REF_INVALID")
        path = Path(ref["path"])
        if path.is_symlink():
            raise ColdStartInputError("SYMLINK_SOURCE_FORBIDDEN")
        path = path.resolve(strict=True)
        if not path.is_relative_to(session):
            raise ColdStartInputError(f"SOURCE_ROLE_OUTSIDE_SESSION:{role}")
        relative = path.relative_to(session)
        if any(part.lower() in FORBIDDEN_SOURCE_COMPONENTS for part in relative.parts):
            raise ColdStartInputError(f"OLD_DERIVED_SOURCE_FORBIDDEN:{role}")
        if file_ref(path) != ref:
            raise ColdStartInputError(f"SOURCE_SHA_DRIFT:{role}")
    validate_producer_reads(source, [Path(ref["path"]) for ref in source["files"].values()])
    return source


def _frame_rows(descriptor: dict[str, Any], mapping: dict[str, Any],
                domain: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    if (set(mapping) != FRAME_MAP_FIELDS
            or mapping.get("schema_version") != "exact78-first-submit-frame-map-v1"):
        raise ColdStartInputError("FRAME_MAP_SCHEMA_DRIFT")
    for key in ("session_id", "source_kind", "physical_eye", "crop_xywh", "time_mode",
                "capture_to_decode_alignment", "decoded_frames", "raw_video"):
        expected = (source["kind"] if key == "source_kind" else
                    source["session_id"] if key == "session_id" else
                    descriptor.get(key))
        if mapping.get(key) != expected:
            raise ColdStartInputError(f"FRAME_MAP_IDENTITY_DRIFT:{key}")
    count = descriptor.get("decoded_frames")
    rows = mapping.get("frame_rows")
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or count != domain["declared_frame_count"]
            or not isinstance(rows, list) or len(rows) != count
            or descriptor.get("frame_rows") != rows):
        raise ColdStartInputError("FRAME_MAP_COUNT_OR_SOURCE_DRIFT")
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != FRAME_FIELDS
                or type(row.get("decoded_display_index")) is not int
                or row.get("decoded_display_index") != index
                or row.get("sbs_physical_eye") != domain["physical_eye"]
                or row.get("crop_xywh") != domain["crop_xywh"]
                or row.get("capture_to_decode_alignment") != domain["capture_to_decode_alignment"]):
            raise ColdStartInputError(f"FRAME_MAP_ROW_DRIFT:{index}")
        if any(value is not None and type(value) is not int for value in
               (row.get("pts"), row.get("best_effort_timestamp"))):
            raise ColdStartInputError(f"FRAME_MAP_TIMESTAMP_TYPE_DRIFT:{index}")
        if source["kind"] == "0915_RAW_VST":
            if (type(row.get("qpc_raw_tick")) is not int
                    or type(row.get("capture_timestamp_raw")) is not int
                    or row.get("qpc_raw_tick") != int(domain["qpc_ticks"][index])
                    or row.get("capture_timestamp_raw") != int(domain["capture_timestamps"][index])):
                raise ColdStartInputError(f"VST0915_INDEX_ALIGNMENT_DRIFT:{index}")
        elif row.get("qpc_raw_tick") is not None or row.get("capture_timestamp_raw") is not None:
            raise ColdStartInputError(f"TRACKER_TIMESTAMP_AUTHORITY_DRIFT:{index}")
    return rows


def build_spec_from_sealed(
    *, packet: dict[str, Any], descriptor_ref: dict[str, Any],
    frame_map_ref: dict[str, Any], seal_ref: dict[str, Any],
    producer_result_ref: dict[str, Any], producer_receipt_ref: dict[str, Any],
    repo_root: Path = PROJECT_ROOT, allowed_root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Validate a terminal upstream producer and return diagnostic model spec.

    The three admission files must share one producer attempt. The consumer
    has its own Task Packet/signature. Its future executor must separately
    enforce the live writer, GPU lease and native-open gates.
    """
    root = repo_root.resolve(strict=True)
    allowed = allowed_root.resolve(strict=True)
    if packet.get("execution_mode") != "EXACT78_CANDIDATE":
        raise ColdStartInputError("CANDIDATE_PACKET_REQUIRED")
    consumer_id = packet.get("task_id")
    if not isinstance(consumer_id, str) or not consumer_id.startswith("exact78_"):
        raise ColdStartInputError("CONSUMER_TASK_ID_REQUIRED")
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict) or any(inputs.get(key) != ref for key, ref in (
        ("upstream_admission_descriptor", descriptor_ref),
        ("upstream_admission_frame_time_map", frame_map_ref),
        ("upstream_admission_seal", seal_ref),
        ("upstream_admission_result", producer_result_ref),
        ("upstream_admission_receipt", producer_receipt_ref),
    )):
        raise ColdStartInputError("PACKET_SEALED_INPUT_BINDING_DRIFT")
    seal_path, seal = _project_ref(seal_ref, root)
    attempt = seal_path.parent.parent
    producer_id = attempt.parent.parent.name
    if (seal_path.parent.name != "admission" or not attempt.name.startswith("attempt_")
            or not producer_id.startswith("exact78_newraw_")
            or producer_id == consumer_id
            or attempt.parent != root / "_run/current" / producer_id / "attempts"):
        raise ColdStartInputError("SEALED_ATTEMPT_SCOPE_DRIFT")
    descriptor_path, descriptor = _project_ref(descriptor_ref, root)
    map_path, mapping = _project_ref(frame_map_ref, root)
    if descriptor_path.parent != seal_path.parent or map_path.parent != seal_path.parent:
        raise ColdStartInputError("SEALED_INPUT_NOT_SAME_ATTEMPT")
    if (seal.get("schema_version") != "exact78-first-submit-admission-seal-v1"
            or seal.get("status") != "SEALED_CPU_ADMISSION"
            or seal.get("task_id") != producer_id
            or seal.get("descriptor") != descriptor_ref
            or seal.get("frame_time_map") != frame_map_ref
            or seal.get("descriptor_sha256") != hashlib.sha256(_canonical(descriptor)).hexdigest()
            or inputs.get("upstream_admission_descriptor_sha256") != seal.get("descriptor_sha256")
            or inputs.get("submission_input_sha256") != descriptor.get("submission_input_sha256")):
        raise ColdStartInputError("ADMISSION_SEAL_BINDING_DRIFT")
    result_path, producer_result = _project_ref(producer_result_ref, root)
    receipt_path, producer_receipt = _project_ref(producer_receipt_ref, root)
    if (result_path != attempt / "RESULT.json"
            or receipt_path != root / "tasks/receipts" / f"{producer_id.upper()}.json"
            or producer_result.get("schema_version") != "exact78-first-submit-cpu-admission-result-v1"
            or producer_result.get("task_id") != producer_id
            or producer_result.get("status") != "PASSED"
            or producer_result.get("admission") != {
                "descriptor": descriptor_ref, "frame_time_map": frame_map_ref, "seal": seal_ref}
            or producer_result.get("source_session") != descriptor.get("session_id")
            or producer_result.get("source_root") != descriptor.get("session_root")
            or producer_result.get("source_files") != descriptor.get("source_files")
            or producer_result.get("submission_input_sha256") != descriptor.get("submission_input_sha256")
            or producer_result.get("descriptor_sha256") != seal.get("descriptor_sha256")
            or producer_receipt.get("task_id") != producer_id
            or producer_receipt.get("status") != "PASSED"
            or producer_receipt.get("result") != producer_result_ref):
        raise ColdStartInputError("UPSTREAM_TERMINAL_RESULT_DRIFT")
    state = _object(root / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    terminal = [item for item in state.get("tasks", []) if item.get("task_id") == producer_id]
    if (len(terminal) != 1 or terminal[0].get("status") != "PASSED"
            or terminal[0].get("result") != producer_result_ref
            or terminal[0].get("terminal_receipt") != producer_receipt_ref
            or terminal[0].get("session") != descriptor.get("session_id")):
        raise ColdStartInputError("UPSTREAM_NOT_TERMINAL_IN_LEDGER")
    producer_packet_path = root / "tasks/current" / producer_id / "TASK_PACKET.json"
    producer_packet_ref = terminal[0].get("task_packet")
    if (not isinstance(producer_packet_ref, dict)
            or producer_packet_ref.get("path") != str(producer_packet_path)):
        raise ColdStartInputError("UPSTREAM_PACKET_REF_ABSENT")
    _, producer_packet = _project_ref(producer_packet_ref, root)
    if (producer_packet.get("task_id") != producer_id
            or producer_packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or producer_packet.get("run_signature") != producer_result.get("run_signature")
            or producer_packet.get("inputs", {}).get("descriptor_sha256") != seal.get("descriptor_sha256")
            or producer_packet.get("inputs", {}).get("source_files") != descriptor.get("source_files")
            or producer_packet.get("inputs", {}).get("submission_input_sha256")
               != descriptor.get("submission_input_sha256")
            or not isinstance(packet.get("run_signature"), str)
            or packet["run_signature"] == producer_packet.get("run_signature")):
        raise ColdStartInputError("UPSTREAM_PACKET_OR_CONSUMER_SIGNATURE_DRIFT")
    if (inputs.get("source_session") != descriptor.get("session_id")
            or inputs.get("source_root") != descriptor.get("session_root")):
        raise ColdStartInputError("CONSUMER_SOURCE_SESSION_DRIFT")
    source = _source_from_descriptor(descriptor, allowed)
    domain = _admit_known_domain(source)
    raw_role = domain["raw_video_role"]
    if (descriptor.get("raw_video") != source["files"][raw_role]
            or descriptor.get("decoded_frames") != domain["declared_frame_count"]
            or descriptor.get("fps_nominal") != domain["fps_nominal"]
            or descriptor.get("physical_eye") != domain["physical_eye"]
            or descriptor.get("crop_xywh") != domain["crop_xywh"]
            or descriptor.get("time_mode") != domain["time_mode"]
            or descriptor.get("capture_to_decode_alignment") != domain["capture_to_decode_alignment"]
            or descriptor.get("calibration_claim") != domain["calibration_claim"]
            or descriptor.get("world_pose_authority") is not False
            or descriptor.get("metric_authority") is not False
            or descriptor.get("independent_source_group") != "UNVERIFIED"):
        raise ColdStartInputError("DESCRIPTOR_CAPABILITY_DRIFT")
    rows = _frame_rows(descriptor, mapping, domain, source)
    session = Path(source["session_root"])
    if source["kind"] == "TRACKER_CLIP_RGB":
        manifest = _object(Path(source["files"]["clip_manifest"]["path"]))
        intrinsics = manifest["video_rectification"]["intrinsics"]
        for axis in (0, 1):
            _finite(intrinsics[axis][axis], "TRACKER_FOCAL", positive=True)
        allowed_refs = {key: source["files"][key] for key in
                        ("clip_manifest", "video", "camera_params", "slam")}
        kind = "tracker"
    else:
        camera = _object(Path(source["files"]["raw/camera_params.json"]["path"]))
        left = camera["left"]
        width, height = camera["width"], camera["height"]
        sx, sy = TARGET_SIZE[0] / width, TARGET_SIZE[1] / height
        intrinsics = [[_finite(left.get("fx"), "VST_FX", positive=True) * sx, 0.0,
                       _finite(left.get("cx"), "VST_CX") * sx],
                      [0.0, _finite(left.get("fy"), "VST_FY", positive=True) * sy,
                       _finite(left.get("cy"), "VST_CY") * sy], [0.0, 0.0, 1.0]]
        allowed_refs = {key.removeprefix("raw/"): value for key, value in source["files"].items()}
        kind = "vst0915"
    return {
        "source_video": Path(source["files"][raw_role]["path"]),
        "allowed_refs": allowed_refs,
        "intrinsics": intrinsics, "fps_nominal": domain["fps_nominal"],
        "frame_count": len(rows), "crop_xywh": domain["crop_xywh"],
        "source_dimensions": domain["expected_dimensions"],
        "time_mode": ("CLIP_FRAME_INDEX_ONLY" if kind == "tracker" else "TIMEBASE_UNVERIFIED"),
        "physical_eye": domain["physical_eye"], "calibration_claim": domain["calibration_claim"],
        "session_id": source["session_id"], "session": session,
        "frame_rows_sealed": rows, "source_kind": kind,
        "input_role": "new_session_candidate", "split_claim": "NOT_ASSIGNED",
        "model_input_size": list(TARGET_SIZE), "infiller_invoked": False,
        "world_pose_authority": False, "metric_authority": False,
        "native_open_audit_required": True,
        "forbidden_reads": "NO_HARDWARE_HAND_NO_ALL_DATA_NO_OLD_HAWOR_NO_MASK_CLEAN_ROBOT",
        "descriptor_ref": descriptor_ref, "frame_map_ref": frame_map_ref,
        "admission_seal_ref": seal_ref,
        "producer_result_ref": producer_result_ref,
        "producer_receipt_ref": producer_receipt_ref,
    }


__all__ = ["build_spec_from_sealed"]
