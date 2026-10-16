"""Read-only, bounded audit of the original pick/place exports.

This audit deliberately does not decode RGB, run a model, or qualify training.
Nearest-neighbour sensor alignment and offline rendering remain distinguishable
from causal inputs and synchronized real Revo2 controls.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RAW_FIELDS = (
    "head_pose", "left_controller_pose", "right_controller_pose",
    "left_wrist_pose", "right_wrist_pose", "left_hand_joints",
    "right_hand_joints", "left_hand_valid", "right_hand_valid",
    "recv_qpc_ns", "segment_id", "source_row_idx", "video_frame_idx",
)
BASE_BLOCKERS = (
    "BLOCKED_ROBOTIZED_OFFLINE_RGB_NOT_CAUSAL_QUALIFIED",
    "BLOCKED_REAL_REVO2_SYNCHRONIZED_CONTROL_ABSENT",
    "BLOCKED_INDEPENDENT_SOURCE_GROUP_EVIDENCE_ABSENT",
    "BLOCKED_CAMERA_CALIBRATION_AND_EYE_IDENTITY_UNQUALIFIED",
)


def _reference(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"not an object: {path}:{number}")
                yield value


def audit_nas_roots(
    raw_root: str | Path,
    robotized_root: str | Path,
    expected_sessions: int = 199,
) -> dict[str, Any]:
    """Inspect every numbered session without reading the large RGB NPZ files.

    Returned status is always BLOCKED: original exports are not reviewed real
    Revo2 training manifests. Missing files and SHA mismatches are explicit and
    never silently remove a session from the denominator.
    """
    import h5py

    raw_root, robotized_root = Path(raw_root).resolve(), Path(robotized_root).resolve()
    expected_ids = [f"{index:03d}" for index in range(1, expected_sessions + 1)]
    raw_ids = sorted(path.name for path in raw_root.iterdir()
                     if path.is_dir() and path.name.isdigit())
    processed_sessions = robotized_root / "sessions"
    processed_ids = sorted(path.name for path in processed_sessions.iterdir()
                           if path.is_dir() and path.name.isdigit())
    export_path = robotized_root / "EXPORT_MANIFEST_V1.json"
    export = _json(export_path) if export_path.is_file() else {}
    expected_refs = {
        row["destination_relative"]: row for row in export.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("destination_relative"), str)
    }
    totals: Counter = Counter()
    outcomes: Counter = Counter()
    hand_outcomes: Counter = Counter()
    rows = []
    errors = []
    for session_id in expected_ids:
        raw = raw_root / session_id
        processed = processed_sessions / session_id
        row: dict[str, Any] = {
            "session_id": session_id, "blockers": list(BASE_BLOCKERS),
            "control_ground_truth": False, "training_eligible": False,
        }
        try:
            manifest_path = raw / "manifest.json"
            manifest = _json(manifest_path)
            row["raw_manifest"] = _reference(manifest_path)
            row["raw_declared_schema"] = manifest.get("dataset_schema")
            with h5py.File(raw / "dataset.hdf5", "r") as hdf:
                fields = {key: {"shape": list(hdf[key].shape), "dtype": str(hdf[key].dtype)}
                          for key in RAW_FIELDS if key in hdf}
                row["raw_sensor_fields"] = fields
                row["missing_raw_sensor_fields"] = [key for key in RAW_FIELDS if key not in hdf]
                row["raw_hdf_rows"] = len(hdf["head_pose"])
                row["raw_video_frames_represented"] = len(np.unique(hdf["video_frame_idx"][...]))
                row["raw_segment_count"] = len(np.unique(hdf["segment_id"][...]))
                row["raw_hdf_schema"] = str(hdf.attrs.get("schema", "UNKNOWN"))
                row["raw_pose_layout"] = str(hdf.attrs.get("pose_layout", "UNKNOWN"))
                row["raw_coordinate_frame"] = str(hdf.attrs.get("coord", "UNKNOWN"))
                row["tactile_included"] = bool(hdf.attrs.get("tactile_included", False))
                row["grasp_label_present"] = any("grasp" in key.lower() for key in hdf)
                row["done_label_present"] = any("done" in key.lower() for key in hdf)
                row["raw_revo2_control_field_present"] = any("revo2" in key.lower() for key in hdf)
            session = _json(processed / "SESSION.json")
            row["original_status"] = session.get("status", "UNKNOWN")
            row["original_visual_quality"] = session.get("visual_quality", "UNKNOWN")
            row["original_ai_review"] = session.get("ai_review", "UNKNOWN")
            row["original_human_review"] = session.get("human_review", "UNKNOWN")
            row["original_training_eligible"] = session.get("training_eligible", False)
            row["original_control_ground_truth"] = session.get("control_ground_truth", False)
            row["original_claim_limit"] = session.get("claim_limit")
            row["declared_frame_count"] = int(session["frame_count"])
            camera = session.get("camera", {})
            row["camera_qualification"] = {
                key: camera.get(key, "UNKNOWN") for key in (
                    "camera_K_authority", "camera_route_status",
                    "physical_eye_identity_verified", "encoded_eye_identity_qualification",
                )
            }
            source_input = _json(processed / "source" / "SESSION_INPUT.json")
            calibration = source_input.get("calibration", {})
            row["wrist_observation_authority"] = calibration.get("wrist_observation_authority", "UNKNOWN")
            row["actual_motion_source"] = calibration.get("actual_motion_source", "UNKNOWN")
            refs = {}
            for relative in ("source/source.npz", "frame_mapping.jsonl", "frame_states.jsonl", "actions.npz"):
                path = processed / relative
                reference = _reference(path)
                refs[relative] = reference
                expected = expected_refs.get(f"sessions/{session_id}/{relative}")
                if expected is not None and any(reference[key] != expected.get(key) for key in ("bytes", "sha256")):
                    raise ValueError(f"export SHA mismatch: {relative}")
            source_reference = source_input.get("arrays", {})
            if any(refs["source/source.npz"][key] != source_reference.get(key) for key in ("bytes", "sha256")):
                raise ValueError("source array SHA differs from SESSION_INPUT")
            row["file_references"] = refs
            with np.load(processed / "source" / "source.npz", allow_pickle=False) as source:
                qpc = source["qpc_ns"]
                frame_count = len(qpc)
                if frame_count != row["declared_frame_count"]:
                    raise ValueError("source length differs from declared frame count")
                if np.any(np.diff(qpc) <= 0):
                    raise ValueError("non-increasing frame QPC")
                row["frame_count"] = frame_count
                row["future_conditioning_counts"] = {}
                for route, key in (
                    ("pico", "mapping__pico_source_qpc_ns"),
                    ("manus_left", "mapping__manus_left_source_qpc_ns"),
                    ("manus_right", "mapping__manus_right_source_qpc_ns"),
                ):
                    skew = source[key] - qpc
                    count = int(np.count_nonzero(skew > 0))
                    row["future_conditioning_counts"][route] = count
                    totals[f"future_{route}_frames"] += count
                    row.setdefault("source_skew_ns", {})[route] = {
                        "minimum": int(skew.min()), "maximum": int(skew.max()),
                    }
                row["native_sensor_valid_counts"] = {}
                for key in ("native__head_valid", "native__controller_valid",
                            "native__manus_valid", "native__complete_valid"):
                    value = source[key]
                    row["native_sensor_valid_counts"][key] = {
                        "true": int(np.count_nonzero(value)), "total": int(value.size),
                    }
                wrist = source["native__wrist"]
                row["native_wrist_shape"] = list(wrist.shape)
                row["native_wrist_finite"] = bool(np.isfinite(wrist).all())
                if not row["native_wrist_finite"]:
                    row["blockers"].append("BLOCKED_NONFINITE_RAW_WRIST")
            with np.load(processed / "actions.npz", allow_pickle=False) as actions:
                row["digital_action_fields"] = {
                    key: {"shape": list(actions[key].shape), "dtype": str(actions[key].dtype)}
                    for key in ("q_arm", "q_hand", "qpc_ns", "complete_valid") if key in actions
                }
                row["digital_actions_are_real_revo2_controls"] = False
            mapping = list(_jsonl(processed / "frame_mapping.jsonl"))
            if len(mapping) != frame_count or [record.get("frame_id") for record in mapping] != list(range(frame_count)):
                raise ValueError("frame mapping is not one-to-one native frame order")
            states = list(_jsonl(processed / "frame_states.jsonl"))
            if len(states) != frame_count:
                raise ValueError("frame state count mismatch")
            state_counts: Counter = Counter()
            for state in states:
                state_counts[f"occlusion_order:{state.get('occlusion_order', 'UNKNOWN')}"] += 1
                state_counts[f"physical_contact:{state.get('physical_contact', 'UNKNOWN')}"] += 1
                for side, hand in state.get("hands", {}).items():
                    for flag in ("position_violation", "rotation_violation", "qpc_velocity_violation", "qpc_acceleration_violation"):
                        if hand.get(flag) is True:
                            state_counts[f"{side}:{flag}"] += 1
                    state_counts[f"{side}:source:{hand.get('source', 'UNKNOWN')}"] += 1
            row["original_frame_state_counts"] = dict(state_counts)
            hand_outcomes.update(state_counts)
            outcomes[str(row["original_status"])] += 1
            totals["raw_hdf_rows"] += row["raw_hdf_rows"]
            totals["native_video_frames"] += frame_count
            totals["audited_sessions"] += 1
            if any(row["future_conditioning_counts"].values()):
                row["blockers"].append("BLOCKED_NEAREST_SENSOR_INPUT_USES_FUTURE_SAMPLES")
            row["audit_status"] = "STRUCTURE_READ_ONLY_AUDITED_NOT_TRAINING_QUALIFIED"
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["audit_status"] = "BLOCKED_AUDIT_ERROR"
            row["blockers"].append("BLOCKED_SOURCE_AUDIT_ERROR")
            row["error"] = str(error)
            errors.append({"session_id": session_id, "error": str(error)})
        rows.append(row)
    return {
        "schema": "pick-place-humanego-ict-revo2-readiness-audit-v1",
        "status": "BLOCKED", "training_start_allowed": False,
        "formal_training_started": False, "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "raw_root": str(raw_root), "robotized_root": str(robotized_root),
        "expected_sessions": expected_sessions, "raw_session_count": len(raw_ids),
        "processed_session_count": len(processed_ids),
        "missing_raw_sessions": sorted(set(expected_ids) - set(raw_ids)),
        "missing_processed_sessions": sorted(set(expected_ids) - set(processed_ids)),
        "unexpected_raw_sessions": sorted(set(raw_ids) - set(expected_ids)),
        "unexpected_processed_sessions": sorted(set(processed_ids) - set(expected_ids)),
        "totals": dict(totals), "original_status_counts": dict(outcomes),
        "original_frame_state_totals": dict(hand_outcomes),
        "blockers": list(BASE_BLOCKERS) + (["BLOCKED_SOURCE_AUDIT_ERROR"] if errors else []),
        "errors": errors, "sessions": rows,
        "export_manifest": _reference(export_path) if export_path.is_file() else None,
        "export_claims": {key: export.get(key, "UNKNOWN") for key in (
            "status", "sessions", "original_native_frames", "training_eligible",
            "all199_visual_accepted", "human_review_status", "control_ground_truth",
        )},
        "rgb_arrays_loaded": False, "models_loaded": False,
        "claim_limit": "Read-only engineering audit; original FAIL/UNKNOWN remain unchanged; no training or physical authority.",
    }
