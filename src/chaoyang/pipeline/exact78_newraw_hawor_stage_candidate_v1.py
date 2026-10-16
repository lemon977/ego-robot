"""HaWoR-only stage from a terminal, same-session new-raw CPU admission.

This is an unregistered candidate. The fenced GPU executor must native-open
audit this provisional stage and publish its SHA seal before downstream use.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from chaoyang.governance.exact78_cold_start_guard_v1 import (
    PROJECT_ROOT, file_ref, validate_output_path,
)
from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import (
    build_spec_from_sealed,
)


class NewRawHaWoRRejected(RuntimeError):
    """The sealed CPU admission or frozen model-stage claim did not match."""


def _new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")


def _input_refs(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = packet.get("inputs", {})
    names = (
        "upstream_admission_descriptor", "upstream_admission_frame_time_map",
        "upstream_admission_seal", "upstream_admission_result",
        "upstream_admission_receipt",
    )
    refs = {name: inputs.get(name) for name in names}
    if any(not isinstance(ref, dict) for ref in refs.values()):
        raise NewRawHaWoRRejected("FIVE_SEALED_ADMISSION_REFS_REQUIRED")
    return refs


def checked_spec(packet: dict[str, Any]) -> dict[str, Any]:
    refs = _input_refs(packet)
    return build_spec_from_sealed(
        packet=packet,
        descriptor_ref=refs["upstream_admission_descriptor"],
        frame_map_ref=refs["upstream_admission_frame_time_map"],
        seal_ref=refs["upstream_admission_seal"],
        producer_result_ref=refs["upstream_admission_result"],
        producer_receipt_ref=refs["upstream_admission_receipt"],
    )


def _attempt(packet: dict[str, Any]) -> Path:
    task_id = packet.get("task_id")
    if not isinstance(task_id, str) or not task_id.startswith("exact78_newraw_hawor_"):
        raise NewRawHaWoRRejected("CONSUMER_TASK_ID_DRIFT")
    attempt = PROJECT_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    permitted = [(PROJECT_ROOT / raw).resolve() for raw in packet.get("write_set", [])
                 if isinstance(raw, str) and not Path(raw).is_absolute()]
    if attempt.resolve() not in permitted:
        raise NewRawHaWoRRejected("ATTEMPT_WRITE_SET_ABSENT")
    return attempt


def run_stage(packet: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Write a provisional HaWoR result; native-open audit/seal is external."""
    # Importing this generic v3 module is allowed only after the operation gate.
    from chaoyang.ops import run_exact78_hawor_raw_v3 as v3
    import numpy as np

    attempt = _attempt(packet)
    output = validate_output_path(output_root, attempt_root=attempt)
    if output != attempt / "candidate/hawor" or output.exists():
        raise NewRawHaWoRRejected("FRESH_HAWOR_OUTPUT_REQUIRED")
    spec = checked_spec(packet)
    if (spec["input_role"] != "new_session_candidate"
            or spec["split_claim"] != "NOT_ASSIGNED"
            or spec["world_pose_authority"] is not False
            or spec["metric_authority"] is not False
            or spec["infiller_invoked"] is not False
            or spec["native_open_audit_required"] is not True):
        raise NewRawHaWoRRejected("SPEC_QUALIFICATION_DRIFT")
    assets = v3.model_asset_refs()
    if assets != packet.get("inputs", {}).get("model_assets"):
        raise NewRawHaWoRRejected("MODEL_ASSET_SHA_DRIFT")
    if (v3.TARGET_SIZE != (1280, 960)
            or spec["model_input_size"] != [1280, 960]):
        raise NewRawHaWoRRejected("FROZEN_MODEL_INPUT_SIZE_DRIFT")

    started = time.monotonic()
    output.mkdir(parents=True)
    jpeg_rows, jpeg_digest = v3.extract_model_frames(
        spec, output / "model_input/extracted_images")
    if len(jpeg_rows) != spec["frame_count"]:
        raise NewRawHaWoRRejected("MODEL_INPUT_FRAME_COUNT_DRIFT")
    arrays = v3.run_hawor(spec, output / "model_input/extracted_images")
    valid = arrays.get("prediction_valid")
    count = spec["frame_count"]
    shapes = {
        "joints_3d_camera": (2, count, 21, 3),
        "joints_2d": (2, count, 21, 2),
        "root_orient_camera": (2, count, 3, 3),
        "hand_pose_rotmat": (2, count, 15, 3, 3),
        "betas": (2, count, 10),
        "prediction_valid": (2, count),
        "intrinsics": (count, 3, 3),
    }
    if (not isinstance(valid, np.ndarray)
            or any(not isinstance(arrays.get(name), np.ndarray)
                   or arrays[name].shape != shape for name, shape in shapes.items())):
        raise NewRawHaWoRRejected("PREDICTION_VALID_SHAPE_DRIFT")
    if (not np.array_equal(arrays.get("original_frame_indices"),
                           np.arange(count, dtype=np.int32))
            or not np.isfinite(arrays["joints_3d_camera"][valid]).all()
            or not np.isfinite(arrays["joints_2d"][valid]).all()):
        raise NewRawHaWoRRejected("HAWOR_ARRAY_FRAME_OR_SIDE_DRIFT")
    npz = output / "HAWOR_CAMERA_PREDICTIONS.npz"
    np.savez_compressed(npz, **arrays)
    frame_rows = [v3.frame_status(row, index, spec, bool(valid[0, index]),
                                  bool(valid[1, index]))
                  for index, row in enumerate(spec["frame_rows_sealed"])]
    provenance = output / "FRAME_PROVENANCE.json"
    _new_json(provenance, {
        "schema_version": "exact78-hawor-frame-provenance-v3",
        "session_id": spec["session_id"], "frame_count": spec["frame_count"],
        "fps_nominal": spec["fps_nominal"], "fps_is_timestamp_truth": False,
        "model_input_width": 1280, "model_input_height": 960,
        "intrinsics_image_size": [1280, 960],
        "physical_eye": spec["physical_eye"], "time_mode": spec["time_mode"],
        "frame_rows": frame_rows,
    })
    refs = _input_refs(packet)
    audit = output / "INPUT_AUDIT.json"
    raw_key = "video" if spec["source_kind"] == "tracker" else "vst.h264"
    _new_json(audit, {
        "schema_version": "exact78-hawor-input-audit-v3",
        "session_id": spec["session_id"], "source_kind": spec["source_kind"],
        "source_role": spec["input_role"], "source_split_claim": spec["split_claim"],
        "raw_source_ref": spec["allowed_refs"][raw_key],
        # The adapter rehashes every admitted raw role for identity, while
        # only raw_source_ref is decoded as HaWoR model RGB.
        "read_refs": {**packet["inputs"]["source_files"], **refs},
        "model_input_jpeg_refs": jpeg_rows,
        "model_input_jpeg_sha256_aggregate": jpeg_digest,
        "runtime_code": file_ref(Path(__file__)),
        "model_assets": assets,
        "vendor_model_code": {
            "detect_track_video": file_ref(v3.HAWOR / "scripts/scripts_test_video/detect_track_video.py"),
            "hawor_video": file_ref(v3.HAWOR / "scripts/scripts_test_video/hawor_video.py"),
            "pipeline_tools": file_ref(v3.HAWOR / "lib/pipeline/tools.py"),
            "mano_process": file_ref(v3.HAWOR / "hawor/utils/process.py"),
        },
        "calibration_claim": spec["calibration_claim"],
        "same_session_slam_usage": (
            "SHA_ONLY_CAMERA_WORLD_ALIGNMENT_NOT_ESTABLISHED"
            if spec["source_kind"] == "tracker" else "NOT_APPLICABLE"),
        "forbidden_reads": spec["forbidden_reads"],
        "infiller_invoked": False, "world_pose_emitted": False,
    })
    # Recheck raw and sealed inputs after model execution before handing them
    # to the executor's native-open audit and immutable stage seal.
    checked_spec(packet)
    if v3.model_asset_refs() != assets:
        raise NewRawHaWoRRejected("POST_MODEL_ASSET_SHA_DRIFT")
    result = {
        "schema_version": "exact78-hawor-pure-rgb-result-v3",
        "status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
        "session_id": spec["session_id"], "source_kind": spec["source_kind"],
        "source_frames_decoded": spec["frame_count"],
        "model_input_frames": len(jpeg_rows),
        "model_prediction_frames": {
            side: int(valid[index].sum()) for index, side in enumerate(v3.SIDES)},
        "model_infill_frames": 0, "deterministic_fill_frames": 0,
        "npz": file_ref(npz), "frame_provenance": file_ref(provenance),
        "input_audit": file_ref(audit), "time_mode": spec["time_mode"],
        "world_pose_valid": False, "control_ground_truth": False,
        "training_eligible": False, "visual_qualified": False,
        "product_qualified": False, "wall_seconds": round(time.monotonic() - started, 3),
        "claim_limit": "Fresh RGB-derived camera predictions only; not direct 3D observation, world pose, control truth or quality PASS.",
    }
    _new_json(output / "RESULT.json", result)
    return result


__all__ = ["NewRawHaWoRRejected", "checked_spec", "run_stage"]
