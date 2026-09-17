#!/usr/bin/env python3
"""Content-gated handle converter with tactile preservation and 0915 support.

This is an additive wrapper around ``convert_handle_egodex_to_tracker_session``.
It leaves the established image/metadata format intact, but closes two gaps:

* acquisition-aligned HDF5 tactile arrays are materialized in every frame; and
* raw-only captures are aligned without inventing a missing MANUS modality.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import numpy as np

try:
    from . import convert_handle_egodex_to_tracker_session as base
    from .tactile_quality_gate_v1 import analyze_session
except ImportError:  # Direct file execution by the bounded batch runner.
    import convert_handle_egodex_to_tracker_session as base
    from tactile_quality_gate_v1 import analyze_session


SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GATE_NS = 40_000_000
RAW_REQUIRED = (
    "raw/vst.h264", "raw/vst.ts.jsonl", "raw/vst.qpc.ts.jsonl",
    "raw/camera_params.json", "raw/camera_params.meta.json",
    "raw/pico.jsonl", "raw/tactile.jsonl", "raw/tactile.meta.json",
)
_QUALITY: dict[str, Any] | None = None
_PROFILE = ""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _args_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    raise base.ContractError(
                        f"invalid JSON {path}:{line_number}: {exc!r}"
                    ) from exc


def _nearest(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    after = np.searchsorted(source_ns, target_ns, side="left")
    after = np.clip(after, 0, len(source_ns) - 1)
    before = np.clip(after - 1, 0, len(source_ns) - 1)
    choose_before = np.abs(source_ns[before] - target_ns) <= np.abs(
        source_ns[after] - target_ns
    )
    index = np.where(choose_before, before, after)
    return index.astype(np.int64), source_ns[index] - target_ns


def _valid_pose(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        return False
    array = np.asarray(value, dtype=np.float64)
    return bool(np.isfinite(array).all() and np.linalg.norm(array[3:]) > 0.5)


def _raw_pose(value: dict[str, Any]) -> list[float]:
    pose = list(value.get("pos") or []) + list(value.get("quat") or [])
    if not _valid_pose(pose):
        raise base.ContractError("PICO pose is missing, non-finite, or degenerate")
    return [float(item) for item in pose]


_R_PICO_TO_REP103 = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _pico_pose_to_rep103(pose: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(pose), dtype=np.float64)
    position = np.asarray([value[0], value[1], -value[2]])
    quaternion = np.asarray([value[3], value[4], -value[5], -value[6]])
    rotation = base.quat_to_rot(quaternion)
    result_rotation = _R_PICO_TO_REP103 @ rotation @ _R_PICO_TO_REP103.T
    return np.concatenate([_R_PICO_TO_REP103 @ position,
                           base.rot_to_quat(result_rotation)])


def _pico_hands(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    output: dict[str, np.ndarray] = {}
    valid: dict[str, bool] = {}
    hand_root = data.get("Hand") if isinstance(data.get("Hand"), dict) else {}
    for side in SIDES:
        hand = hand_root.get(f"{side}Hand")
        locations = hand.get("HandJointLocations") if isinstance(hand, dict) else None
        good = (isinstance(hand, dict) and int(hand.get("isActive", 0)) == 1 and
                int(hand.get("count", 0)) == 26 and isinstance(locations, list) and
                len(locations) == 26)
        poses: list[np.ndarray] = []
        if good:
            for location in locations:
                try:
                    raw = [float(item) for item in str(location.get("p", "")).split(",")]
                except (TypeError, ValueError):
                    good = False
                    break
                if not _valid_pose(raw):
                    good = False
                    break
                poses.append(_pico_pose_to_rep103(raw))
        valid[side] = bool(good)
        if good:
            output[side] = np.stack(poses)
        else:
            absent = np.zeros((26, 7), dtype=np.float64)
            absent[:, 6] = 1.0
            output[side] = absent
    return output, valid


def _decode_coordinates(wire_layout: dict[str, Any]) -> list[tuple[int, int]]:
    rows, cols = int(wire_layout["rows"]), int(wire_layout["cols"])
    blob = bytes.fromhex(wire_layout["cellmap_hex"])
    if len(blob) != rows * 4:
        raise base.ContractError("tactile cellmap byte count mismatch")
    coordinates: list[tuple[int, int]] = []
    for row in range(rows):
        mask = int.from_bytes(blob[row * 4:(row + 1) * 4], "little")
        coordinates.extend((row, column) for column in range(cols)
                           if mask & (1 << column))
    if len(coordinates) != int(wire_layout["active_count"]):
        raise base.ContractError("tactile cellmap active count mismatch")
    return coordinates


def _finger_arrays(values: np.ndarray, side: str,
                   side_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _decode_coordinates(side_meta["wire_layout"])
    grid = np.zeros((24, 16), dtype=np.int16)
    for index, (row, column) in enumerate(coordinates):
        grid[row, column] = values[index]
    interpretation = side_meta["physical_interpretation"]
    mapping = interpretation.get("finger_mapping", interpretation)
    regions = mapping.get("finger_regions_in_wire_grid_by_side", {}).get(
        side, mapping["finger_regions_in_wire_grid"]
    )
    result = np.zeros((5, 4, 8), dtype=np.int16)
    active = np.ones((5, 4, 8), dtype=bool)
    for finger_index, finger in enumerate(FINGERS):
        r0, r1, c0, c1 = [int(item) for item in regions[finger]]
        result[finger_index] = grid[r0:r1, c0:c1].T
        if finger != "thumb":
            active[finger_index, :, 7] = False
    return result, active


def _load_raw_only(source: Path) -> dict[str, Any]:
    calibration = json.loads((source / "raw/camera_params.json").read_text())
    base.validate_calibration(calibration)
    camera_meta = json.loads((source / "raw/camera_params.meta.json").read_text())
    wall_ts = np.asarray([int(line) for line in
                          (source / "raw/vst.ts.jsonl").read_text().splitlines()
                          if line.strip()], dtype=np.int64)
    qpc_ts = np.asarray([int(line) for line in
                         (source / "raw/vst.qpc.ts.jsonl").read_text().splitlines()
                         if line.strip()], dtype=np.int64)
    if (wall_ts.shape != qpc_ts.shape or len(wall_ts) < 2 or
            np.any(np.diff(wall_ts) <= 0) or np.any(np.diff(qpc_ts) <= 0)):
        raise base.ContractError("VST timestamp sidecars are invalid")

    pico_rows: list[dict[str, Any]] = []
    for record in _json_lines(source / "raw/pico.jsonl"):
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        controller = data.get("Controller") if isinstance(data.get("Controller"), dict) else {}
        try:
            head = _raw_pose(data["Head"])
            left = _raw_pose(controller["left"])
            right = _raw_pose(controller["right"])
            qpc = int(record["recv_qpc_ns"])
            wall = int(record["recv_wall_ns"])
            sensor = int(data["timeStampNs"])
            hands, hand_valid = _pico_hands(data)
        except (KeyError, TypeError, ValueError, base.ContractError):
            continue
        pico_rows.append({"qpc": qpc, "wall": wall, "sensor": sensor,
                          "head": _pico_pose_to_rep103(head),
                          "left": _pico_pose_to_rep103(left),
                          "right": _pico_pose_to_rep103(right),
                          "hands": hands, "hand_valid": hand_valid})
    if len(pico_rows) < 2:
        raise base.ContractError("not enough valid PICO rows")
    pico_qpc = np.asarray([row["qpc"] for row in pico_rows], dtype=np.int64)
    if np.any(np.diff(pico_qpc) <= 0):
        raise base.ContractError("PICO qpc timeline is not strictly increasing")

    tactile_meta = json.loads((source / "raw/tactile.meta.json").read_text())
    tactile: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for record in _json_lines(source / "raw/tactile.jsonl"):
        side = record.get("side")
        if side in SIDES:
            values = np.asarray(record.get("wire_values"), dtype=np.int16)
            if values.shape != (369,):
                raise base.ContractError("tactile wire value shape mismatch")
            tactile[side].append({"qpc": int(record["recv_qpc_ns"]),
                                  "record_seq": int(record["record_seq"]),
                                  "stream_seq": int(record["stream_seq"]),
                                  "values": values})
    tactile_qpc = {side: np.asarray([row["qpc"] for row in tactile[side]],
                                    dtype=np.int64) for side in SIDES}
    for side in SIDES:
        if len(tactile_qpc[side]) < 2 or np.any(np.diff(tactile_qpc[side]) <= 0):
            raise base.ContractError(f"{side} tactile qpc timeline is invalid")

    begin = max(int(pico_qpc[0]), int(qpc_ts[0]),
                *(int(tactile_qpc[side][0]) for side in SIDES))
    end = min(int(pico_qpc[-1]), int(qpc_ts[-1]),
              *(int(tactile_qpc[side][-1]) for side in SIDES))
    step = int(round(1_000_000_000 / 30.0))
    target_qpc = np.arange(begin, end + 1, step, dtype=np.int64)
    if len(target_qpc) < 2:
        raise base.ContractError("raw-only common aligned interval is empty")
    pico_index, pico_delta = _nearest(pico_qpc, target_qpc)
    video_index, video_delta = _nearest(qpc_ts, target_qpc)
    tactile_index: dict[str, np.ndarray] = {}
    tactile_delta: dict[str, np.ndarray] = {}
    keep = (np.abs(pico_delta) <= GATE_NS) & (np.abs(video_delta) <= GATE_NS)
    for side in SIDES:
        tactile_index[side], tactile_delta[side] = _nearest(
            tactile_qpc[side], target_qpc
        )
        keep &= np.abs(tactile_delta[side]) <= GATE_NS
    target_qpc = target_qpc[keep]
    pico_index, video_index = pico_index[keep], video_index[keep]
    video_delta = video_delta[keep]
    for side in SIDES:
        tactile_index[side] = tactile_index[side][keep]
        tactile_delta[side] = tactile_delta[side][keep]
    n = len(target_qpc)
    if n < 2:
        raise base.ContractError("raw-only alignment has fewer than two complete rows")

    selected = [pico_rows[int(index)] for index in pico_index]
    arrays: dict[str, Any] = {
        "timestamp_ns": np.asarray([row["sensor"] for row in selected], dtype=np.int64),
        "recv_wall_ns": np.asarray([row["wall"] for row in selected], dtype=np.int64),
        "recv_qpc_ns": target_qpc,
        "segment_id": np.zeros(n, dtype=np.int32),
        "source_row_idx": pico_index,
        "video_frame_idx": video_index.astype(np.int32),
        "video_offset_ms": (video_delta / 1_000_000).astype(np.float32),
        "head_pose": np.stack([row["head"] for row in selected]),
        "left_controller_pose": np.stack([row["left"] for row in selected]),
        "right_controller_pose": np.stack([row["right"] for row in selected]),
    }
    for side in SIDES:
        hand_pose = np.stack([row["hands"][side] for row in selected])
        hand_valid = np.asarray([row["hand_valid"][side] for row in selected], dtype=bool)
        arrays[f"{side}_pico26_pose"] = hand_pose
        arrays[f"{side}_hand_valid"] = hand_valid
        arrays[f"{side}_wrist_pose"] = hand_pose[:, 0]
        arrays[f"{side}_hand_joints"] = np.zeros((n, 25, 3), dtype=np.float64)
        rows = [tactile[side][int(index)] for index in tactile_index[side]]
        values = np.stack([row["values"] for row in rows])
        fingers_and_mask = [_finger_arrays(row, side, tactile_meta["streams"][side])
                            for row in values]
        fingers = np.stack([item[0] for item in fingers_and_mask])
        active = fingers_and_mask[0][1]
        valid = np.ones(n, dtype=bool)
        arrays[f"{side}_tactile_values"] = values
        arrays[f"{side}_tactile_wire_active_mask"] = np.ones(369, dtype=bool)
        arrays[f"{side}_tactile_wire_valid_mask"] = np.repeat(
            valid[:, None], 369, axis=1
        )
        arrays[f"{side}_tactile_fingers"] = fingers
        arrays[f"{side}_tactile_fingers_active_mask"] = active
        arrays[f"{side}_tactile_fingers_valid_mask"] = np.repeat(
            active[None, ...], n, axis=0
        )
        arrays[f"{side}_tactile_offline_source_index"] = tactile_index[side]
        arrays[f"{side}_tactile_offline_source_valid"] = valid
        arrays[f"{side}_tactile_offline_offset_ms"] = (
            tactile_delta[side] / 1_000_000
        ).astype(np.float32)
        arrays[f"{side}_tactile_causal_source_index"] = np.full(n, -1, dtype=np.int64)
        arrays[f"{side}_tactile_causal_source_valid"] = np.zeros(n, dtype=bool)
        arrays[f"{side}_tactile_sequence"] = np.asarray(
            [row["stream_seq"] for row in rows], dtype=np.int64
        )
        arrays[f"{side}_tactile_causal_age_ms"] = np.zeros(n, dtype=np.float32)

    if np.any(np.diff(arrays["timestamp_ns"]) <= 0):
        raise base.ContractError("aligned PICO sensor timeline is not strictly increasing")
    attrs = {
        "schema": "egodex_v1", "schema_revision": "raw_without_manus_v3",
        "pose_layout": "pos(xyz) + quat(x,y,z,w)", "fps": 30.0,
        "all_exported_frames_complete": True, "complete_frame_coverage": 1.0,
        "joint_names": json.dumps([f"MANUS_ABSENT_{index:02d}" for index in range(25)]),
        "max_skew_ms": 40.0, "video_cam": "{}",
        "source_modality_profile": "raw_without_manus_v1",
        "manus_modality_status": "ABSENT_NOT_CAPTURED",
        "pico26_modality_status": "PRESENT_RAW_ALIGNED",
    }
    manifest = {
        "schema_version": "raw_without_manus_aligned_v1", "authoritative": True,
        "source_level": "RAW_SOURCE_ARCHIVE", "clock_mapping_count": 1,
        "resample_count": 1, "timeline_policy": "HOST_QPC_RGB30",
        "timeline_rate_hz": 30, "modality_absence": {"manus": "ABSENT_NOT_CAPTURED"},
    }
    return {"calibration": calibration, "camera_meta": camera_meta,
            "wall_ts": wall_ts, "qpc_ts": qpc_ts, "arrays": arrays,
            "attrs": attrs, "bundle_manifest": manifest, "clean_bundle": None}


_ORIGINAL_SOURCE_SNAPSHOT = base.source_snapshot
_ORIGINAL_LOAD_SOURCE = base.load_source
_ORIGINAL_GEOMETRY = base.camera_and_hand_geometry
_ORIGINAL_VISIBILITY = base.semantic_visibility
_ORIGINAL_TRAINING = base.write_training_record
_ORIGINAL_METADATA = base.write_metadata


def source_snapshot(source: Path) -> dict[str, dict[str, Any]]:
    if (source / "dataset.hdf5").is_file():
        return _ORIGINAL_SOURCE_SNAPSHOT(source)
    result: dict[str, dict[str, Any]] = {}
    for name in RAW_REQUIRED:
        path = source / name
        if not path.is_file():
            raise base.ContractError(f"required raw-only source file missing: {path}")
        stat = path.stat()
        result[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "sha256": base.sha256(path)}
    return result


def _enhance_acquisition_tactile(source: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    arrays = bundle["arrays"]
    n = len(arrays["timestamp_ns"])
    with h5py.File(source / "dataset.hdf5", "r") as handle:
        for side in SIDES:
            for name in ("tactile_values", "tactile_fingers",
                         "tactile_fingers_active_mask", "tactile_valid",
                         "tactile_offset_ms", "tactile_record_seq",
                         "tactile_stream_seq"):
                key = f"{side}_{name}"
                if key not in handle:
                    raise base.ContractError(f"acquisition tactile array missing: {key}")
                arrays[key] = handle[key][:]
            valid = np.asarray(arrays[f"{side}_tactile_valid"], dtype=bool)
            active = np.asarray(arrays[f"{side}_tactile_fingers_active_mask"], dtype=bool)
            arrays[f"{side}_tactile_wire_active_mask"] = np.ones(369, dtype=bool)
            arrays[f"{side}_tactile_wire_valid_mask"] = np.repeat(
                valid[:, None], 369, axis=1
            )
            arrays[f"{side}_tactile_fingers_valid_mask"] = (
                valid[:, None, None, None] & active[None, ...]
            )
            arrays[f"{side}_tactile_offline_source_index"] = np.asarray(
                arrays[f"{side}_tactile_record_seq"], dtype=np.int64
            )
            arrays[f"{side}_tactile_offline_source_valid"] = valid
            arrays[f"{side}_tactile_offline_offset_ms"] = np.asarray(
                arrays[f"{side}_tactile_offset_ms"], dtype=np.float32
            )
            arrays[f"{side}_tactile_causal_source_index"] = np.full(n, -1, dtype=np.int64)
            arrays[f"{side}_tactile_causal_source_valid"] = np.zeros(n, dtype=bool)
            arrays[f"{side}_tactile_sequence"] = np.asarray(
                arrays[f"{side}_tactile_stream_seq"], dtype=np.int64
            )
            arrays[f"{side}_tactile_causal_age_ms"] = np.zeros(n, dtype=np.float32)
    bundle["attrs"]["source_modality_profile"] = "acquisition_aligned_with_manus_v1"
    bundle["attrs"]["manus_modality_status"] = "PRESENT"
    return bundle


def load_source(source: Path, clean_bundle: Path | None = None) -> dict[str, Any]:
    global _PROFILE
    if (source / "dataset.hdf5").is_file():
        _PROFILE = "acquisition_aligned_with_manus_v1"
        return _enhance_acquisition_tactile(
            source, _ORIGINAL_LOAD_SOURCE(source, clean_bundle)
        )
    if clean_bundle is not None:
        raise base.ContractError("raw-only source does not accept --clean-bundle")
    _PROFILE = "raw_without_manus_v1"
    return _load_raw_only(source)


def geometry(bundle: dict[str, Any], selected_calibration_eye: str = "left") -> dict[str, Any]:
    result = _ORIGINAL_GEOMETRY(bundle, selected_calibration_eye)
    result["source_modality_profile"] = bundle["attrs"].get("source_modality_profile")
    result["bundle_arrays"] = bundle["arrays"]
    return result


def visibility(geometry_value: dict[str, Any], width: int, height: int,
               horizontal_fov_deg: float = 90.0,
               camera_k: np.ndarray | None = None) -> dict[str, Any]:
    if geometry_value.get("source_modality_profile") == "raw_without_manus_v1":
        result = {
            "status": "SESSION_CONTENT_ADMISSIBLE_WITH_DECLARED_MANUS_ABSENCE",
            "fatal_issues": [],
            "policy": "MANUS was not captured; absence is declared and PICO26 is preserved",
            "manus": "ABSENT_NOT_CAPTURED",
            "pico26": "PRESENT_RAW_ALIGNED",
            "horizontal_fov_deg": horizontal_fov_deg,
        }
    else:
        result = _ORIGINAL_VISIBILITY(
            geometry_value, width, height, horizontal_fov_deg, camera_k
        )
    result["tactile_quality"] = _QUALITY
    return result


def _pico26_payload(side: str, row: int, geometry_value: dict[str, Any]) -> dict[str, Any]:
    arrays = geometry_value["bundle_arrays"]
    poses = arrays[f"{side}_pico26_pose"][row]
    world_camera = geometry_value["world_cameras"][row]
    inv_first = np.linalg.inv(geometry_value["world_cameras"][0])
    inv_current = np.linalg.inv(world_camera)
    source_matrices = [base.pose_to_matrix(pose) for pose in poses]
    return {
        "schema_version": "pico26-raw-aligned-v1",
        "status": "PRESENT" if bool(arrays[f"{side}_hand_valid"][row]) else "INACTIVE",
        "joint_count": 26,
        "T_joint_to_source_world_rep103": [matrix.tolist() for matrix in source_matrices],
        "T_joint_to_world": [(inv_first @ matrix).tolist() for matrix in source_matrices],
        "T_joint_to_camera": [(inv_current @ matrix).tolist() for matrix in source_matrices],
        "source": "raw/pico.jsonl HandJointLocations",
    }


def write_training(frame_dir: Path, target: Path, session: str, row: int,
                   bundle: dict[str, Any], geometry_value: dict[str, Any],
                   width: int, height: int, target_k: np.ndarray,
                   image_domain_mode: str) -> None:
    _ORIGINAL_TRAINING(frame_dir, target, session, row, bundle, geometry_value,
                       width, height, target_k, image_domain_mode)
    path = frame_dir / "training_data.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for side in SIDES:
        tactile = record["entities"]["tactile"][side]
        tactile.update({
            "schema_version": "tactile-acquisition-aligned-frame-v2",
            "materialized_view": "nearest_host_qpc",
            "offline_payload": "source/raw/tactile.jsonl",
            "causal_source_index": -1,
            "causal_source_valid": False,
            "causal_sample_age_ms": 0.0,
            "quality_gate": "tactile-content-quality-policy-v1",
        })
    if bundle["attrs"].get("source_modality_profile") == "raw_without_manus_v1":
        hands: dict[str, Any] = {}
        for side in SIDES:
            original = record["entities"]["hands"][side]
            hands[side] = {
                "manus": {"status": "ABSENT_NOT_CAPTURED", "confidence": 0.0},
                "pico26": _pico26_payload(side, row, geometry_value),
                "controller6d": original["controller6d"],
                "pose_source": "PICO26_RAW_ALIGNED; MANUS_ABSENT_NOT_CAPTURED",
            }
        record["entities"]["hands"] = hands
        record["metadata"]["source_alignment"]["manus_modality"] = "ABSENT_NOT_CAPTURED"
        record["metadata"]["source_alignment"]["pico26_modality"] = "PRESENT_RAW_ALIGNED"
    base.json_dump(path, record, compact=True)


def _rewrite_json_lines(path: Path, transform: Any) -> None:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            rows.append(transform(json.loads(line), line_number))
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_metadata(source: Path, staging: Path, session: str,
                   bundle: dict[str, Any], geometry_value: dict[str, Any],
                   visibility_value: dict[str, Any], width: int, height: int,
                   horizontal_fov_deg: float, image_domain_mode: str,
                   target_k: np.ndarray) -> None:
    _ORIGINAL_METADATA(source, staging, session, bundle, geometry_value,
                       visibility_value, width, height, horizontal_fov_deg,
                       image_domain_mode, target_k)
    modality = {
        "profile": bundle["attrs"].get("source_modality_profile"),
        "tactile": "PRESENT_QUALITY_GATED",
        "manus": bundle["attrs"].get("manus_modality_status", "PRESENT"),
        "pico26": bundle["attrs"].get("pico26_modality_status", "NOT_APPLICABLE"),
        "missing_modality_policy": "DECLARED_ABSENT_NOT_FABRICATED",
    }
    for relative in ("clip_manifest.json", "preprocess/pico_humanego_manifest.json"):
        path = staging / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["modality_contract"] = modality
        value["tactile_quality"] = _QUALITY
        if modality["manus"] == "ABSENT_NOT_CAPTURED":
            value.pop("clean_bundle_contract", None)
            if isinstance(value.get("files"), dict):
                value["files"].pop("clean_bundle_manifest", None)
                value["files"].pop("canonical_tactile", None)
            value["raw_only_alignment_contract"] = bundle["bundle_manifest"]
            if relative.endswith("pico_humanego_manifest.json"):
                value["hand_model"] = {
                    "name": "PICO26 raw HandJointLocations",
                    "source": "raw/pico.jsonl",
                    "manus_status": "ABSENT_NOT_CAPTURED",
                    "fabrication_status": "NO_FABRICATION",
                }
        base.json_dump(path, value)

    if modality["manus"] != "ABSENT_NOT_CAPTURED":
        return
    arrays = bundle["arrays"]
    tracking = staging / f"trackingData_{session}.txt"

    def fix_tracking(row: dict[str, Any], line_number: int) -> dict[str, Any]:
        if line_number == 0:
            row["modalityContract"] = modality
        else:
            index = line_number - 1
            row["Hand"] = {
                side: {"isActive": int(bool(arrays[f"{side}_hand_valid"][index])),
                       "source": "PICO26_RAW_ALIGNED",
                       "manusStatus": "ABSENT_NOT_CAPTURED"}
                for side in SIDES
            }
        return row

    _rewrite_json_lines(tracking, fix_tracking)
    controller = staging / f"controller_poses_{session}.jsonl"

    def fix_controller(row: dict[str, Any], line_number: int) -> dict[str, Any]:
        for side in SIDES:
            hand = row["hands"][side]
            hand.pop("T_wrist_to_world", None)
            hand.pop("T_wrist_to_camera", None)
            hand.pop("wrist_derivation", None)
            pico = _pico26_payload(side, line_number, geometry_value)
            hand["pico26_joint0_status"] = pico["status"]
            hand["T_pico26_joint0_to_world"] = pico["T_joint_to_world"][0]
            hand["T_pico26_joint0_to_camera"] = pico["T_joint_to_camera"][0]
            hand["manus_status"] = "ABSENT_NOT_CAPTURED"
        return row

    _rewrite_json_lines(controller, fix_controller)


def write_projection(staging: Path, geometry_value: dict[str, Any], width: int,
                     height: int, camera_k: np.ndarray) -> Path:
    if geometry_value.get("source_modality_profile") != "raw_without_manus_v1":
        return base._V3_ORIGINAL_PROJECTION(staging, geometry_value, width, height,
                                            camera_k)
    path = staging / "review/MODALITY_ABSENCE.json"
    base.json_dump(path, {
        "status": "NOT_APPLICABLE",
        "manus": "ABSENT_NOT_CAPTURED",
        "pico26": "PRESENT_RAW_ALIGNED",
        "reason": "MANUS projection probe is not run when MANUS was not captured",
    })
    return path


def install_patches() -> None:
    base._V3_ORIGINAL_PROJECTION = base.write_projection_probe
    base.source_snapshot = source_snapshot
    base.load_source = load_source
    base.camera_and_hand_geometry = geometry
    base.semantic_visibility = visibility
    base.write_training_record = write_training
    base.write_metadata = write_metadata
    base.write_projection_probe = write_projection


def main() -> int:
    global _QUALITY
    if "--help" in sys.argv or "-h" in sys.argv:
        return base.main()
    source_text = _args_value("--source")
    if source_text is None:
        raise base.ContractError("--source is required")
    source = Path(source_text).resolve(strict=True)
    _QUALITY = analyze_session(source)
    if _QUALITY["status"] != "PASSED_TACTILE_QUALITY":
        report = {
            "schema_version": "handle-egodex-v3-audit-v1",
            "source": str(source),
            "status": "REJECTED_TACTILE_QUALITY",
            "tactile_quality": _QUALITY,
        }
        report_text = _args_value("--report")
        if report_text:
            _atomic_json(Path(report_text), report)
        if "--audit-only" in sys.argv:
            if not report_text:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        raise base.ContractError(
            "tactile quality gate rejected source: " +
            ",".join(item["code"] for item in _QUALITY["failures"])
        )
    install_patches()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
