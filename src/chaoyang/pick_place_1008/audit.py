"""Read-only source audit and structural delivery audit for pick_place_1008.

Production entrypoints are registered governance operations. This module never
reads HDF5 action datasets and never infers camera image-domain qualification.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_POLICY = {
    "clock": "host_qpc_ns",
    "source_clock_precision": "recorded_nanoseconds",
    "timestamp_semantics": "software_stream_receipt_not_exposure_sync",
    "max_skew_ms": 40.0,
    "p95_skew_ms": 30.0,
    "output_pts_tolerance_ns": 1000,
    "min_complete_coverage": 0.95,
    "nearest_tie_policy": "earlier_source_timestamp",
}
RAW_NAMES = {
    "raw_vst": "raw/vst.h264",
    "video_qpc": "raw/vst.qpc.ts.jsonl",
    "video_wall": "raw/vst.ts.jsonl",
    "pico": "raw/pico.jsonl",
    "manus": "raw/manus.jsonl",
    "camera_params": "raw/camera_params.json",
    "camera_meta": "raw/camera_params.meta.json",
    "selection": "raw/capture.selection.json",
    "workflow": "raw/capture.workflow.json",
    "manus_meta": "raw/manus.meta.json",
    "source_manifest": "manifest.json",
    "hdf5_prior_only": "dataset.hdf5",
}


class AuditError(RuntimeError):
    """A source, output-boundary, or immutable-result audit rejection."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: str | Path, known_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "sha256": known_sha or sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AuditError(f"NOT_JSON_OBJECT:{path}")
    return value


def _output_path(path: str | Path) -> Path:
    target = Path(path).resolve()
    if not target.is_relative_to(PROJECT_ROOT.resolve()):
        raise AuditError(f"WRITE_OUTSIDE_PROJECT:{target}")
    return target


def _write_json(path: Path, value: Any) -> None:
    path = _output_path(path)
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AuditError(f"IMMUTABLE_OUTPUT_EXISTS:{path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = _output_path(path)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 allow_nan=False) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AuditError(f"IMMUTABLE_OUTPUT_EXISTS:{path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def parse_checksum_manifest(source_root: str | Path, manifest_path: str | Path
                            ) -> list[tuple[str, Path, str]]:
    root = Path(source_root).resolve(strict=True)
    rows: list[tuple[str, Path, str]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(Path(manifest_path).read_text(
            encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", raw)
        if not match:
            raise AuditError(f"BAD_CHECKSUM_ROW:{line_no}")
        relative = match.group(2).replace("\\", "/")
        relpath = Path(relative)
        if relpath.is_absolute() or ".." in relpath.parts:
            raise AuditError(f"CHECKSUM_PATH_ESCAPE:{relative}")
        target = (root / relpath).resolve(strict=True)
        if not target.is_relative_to(root):
            raise AuditError(f"CHECKSUM_PATH_ESCAPE:{relative}")
        if relative in seen:
            raise AuditError(f"DUPLICATE_CHECKSUM_PATH:{relative}")
        seen.add(relative)
        rows.append((relative, target, match.group(1).lower()))
    return rows


def verify_checksums(source_root: str | Path, manifest_path: str | Path,
                     workers: int = 2, progress_callback: Callable | None = None
                     ) -> dict[str, Any]:
    if not 1 <= workers <= 2:
        raise AuditError("HASH_WORKERS_EXCEED_TWO")
    rows = parse_checksum_manifest(source_root, manifest_path)
    checked: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    def check(row: tuple[str, Path, str]) -> tuple[str, str, str]:
        relative, path, expected = row
        return relative, expected, sha256_file(path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, row) for row in rows]
        for future in as_completed(futures):
            relative, expected, actual = future.result()
            checked[relative] = actual
            if actual != expected:
                failures.append({"path": relative, "expected": expected, "actual": actual})
            if progress_callback:
                progress_callback("hash", len(checked), len(rows))
    return {
        "status": "PASSED" if not failures else "FAILED",
        "entries_expected": len(rows), "entries_verified": len(checked),
        "failures": sorted(failures, key=lambda row: row["path"]),
        "sha_by_relative_path": checked,
    }


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuditError(f"NONINTEGER_TIMESTAMP:{label}")
    return value


def read_integer_jsonl(path: str | Path) -> list[int]:
    values = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                raise AuditError(f"EMPTY_TIMESTAMP_ROW:{path}:{row_index + 1}")
            values.append(_integer(json.loads(line), str(path)))
    return values


def clock_stats(values: list[int]) -> dict[str, Any]:
    deltas = [right - left for left, right in zip(values, values[1:])]
    return {
        "count": len(values), "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "duration_s": (values[-1] - values[0]) / 1e9 if values else 0.0,
        "nonpositive_timestamps": sum(value <= 0 for value in values),
        "duplicates": sum(delta == 0 for delta in deltas),
        "backward": sum(delta < 0 for delta in deltas),
        "strictly_monotonic": bool(values) and all(value > 0 for value in values)
            and all(delta > 0 for delta in deltas),
        "median_gap_ms": float(np.median(deltas) / 1e6) if deltas else None,
        "p95_gap_ms": float(np.percentile(deltas, 95) / 1e6) if deltas else None,
        "max_gap_ms": max(deltas) / 1e6 if deltas else None,
        "gaps_over_100ms": sum(delta > 100_000_000 for delta in deltas),
    }


def nearest_mapping(targets: list[int], sources: list[int],
                    valid: list[bool] | None = None) -> dict[str, Any]:
    if not sources or not clock_stats(sources)["strictly_monotonic"]:
        raise AuditError("MISSING_OR_NONMONOTONIC_SOURCE_CLOCK")
    if valid is None:
        valid = [True] * len(sources)
    if len(valid) != len(sources):
        raise AuditError("SOURCE_VALIDITY_LENGTH_MISMATCH")
    indices, offsets, validity = [], [], []
    for target in targets:
        index = bisect.bisect_left(sources, target)
        candidates = [candidate for candidate in [index - 1, index]
                      if 0 <= candidate < len(sources)]
        selected = min(candidates, key=lambda item: (abs(sources[item] - target), sources[item]))
        offset = sources[selected] - target
        indices.append(selected)
        offsets.append(offset)
        validity.append(bool(valid[selected] and abs(offset) <= 40_000_000))
    absolute_ms = [abs(offset) / 1e6 for offset in offsets]
    p95 = float(np.percentile(absolute_ms, 95)) if absolute_ms else None
    maximum = max(absolute_ms) if absolute_ms else None
    coverage = sum(validity) / len(targets) if targets else 0.0
    passed = bool(targets) and maximum <= 40.0 and p95 <= 30.0 and coverage >= 0.95
    return {"indices": indices, "offsets_ns": offsets, "valid": validity,
            "stats": {"count": len(targets), "coverage": coverage, "p95_ms": p95,
                      "max_ms": maximum, "over_40ms": sum(x > 40 for x in absolute_ms),
                      "status": "PASSED" if passed else "FAILED"}}


def _pose_valid(pose: dict[str, Any]) -> bool:
    try:
        position = np.asarray(pose["pos"], dtype=np.float64)
        quaternion = np.asarray(pose["quat"], dtype=np.float64)
        return bool(position.shape == (3,) and quaternion.shape == (4,)
                    and np.isfinite(position).all() and np.isfinite(quaternion).all()
                    and np.linalg.norm(quaternion) > 1e-6)
    except (KeyError, TypeError, ValueError):
        return False


def _read_sensor_streams(pico_path: Path, manus_path: Path) -> dict[str, Any]:
    pico_times, pico_rows, pico_valid, devices = [], [], [], set()
    components = {"head": [], "controller_left": [], "controller_right": []}
    with pico_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            pico_times.append(_integer(row.get("recv_qpc_ns"), "PICO"))
            pico_rows.append(row_index)
            devices.add(str(row.get("dev", "")))
            data = row.get("data", {})
            head = _pose_valid(data.get("Head", {}))
            left = _pose_valid(data.get("Controller", {}).get("left", {}))
            right = _pose_valid(data.get("Controller", {}).get("right", {}))
            components["head"].append(head)
            components["controller_left"].append(left)
            components["controller_right"].append(right)
            pico_valid.append(head and left and right)
    manus = {side: {"times": [], "rows": [], "valid": [], "glove_ids": set()}
             for side in ["left", "right"]}
    with manus_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            side = row.get("side")
            if side not in manus:
                raise AuditError(f"UNKNOWN_MANUS_SIDE:{side}")
            target = manus[side]
            target["times"].append(_integer(row.get("recv_qpc_ns"), "MANUS"))
            target["rows"].append(row_index)
            target["glove_ids"].add(row.get("glove_id"))
            nodes = np.asarray(row.get("nodes", []), dtype=np.float64)
            node_ok = bool(nodes.shape == (25, 7) and np.isfinite(nodes).all()
                           and row.get("node_ids") == list(range(25))
                           and row.get("node_count") == 25
                           and np.all(np.linalg.norm(nodes[:, 3:], axis=1) > 1e-6))
            target["valid"].append(bool(row.get("calibration_applied") and node_ok))
    return {"pico": {"times": pico_times, "rows": pico_rows, "valid": pico_valid,
                     "components": components, "devices": sorted(devices)},
            "manus": manus}


def camera_snapshot(camera: dict[str, Any], metadata: dict[str, Any]
                    ) -> tuple[dict[str, Any], list[str]]:
    nested = "reference_resolution" in camera
    reference = camera.get("reference_resolution", {})
    size = [reference.get("w"), reference.get("h")] if nested else [
        camera.get("width"), camera.get("height")]
    if not all(isinstance(value, int) and value > 0 for value in size):
        raise AuditError("INVALID_CAMERA_REFERENCE_SIZE")
    eyes = {}
    warnings = []
    for side in ["left", "right"]:
        original = camera.get(side, {})
        intrinsic = original.get("intrinsics", original)
        values = [float(intrinsic[key]) for key in ["fx", "fy", "cx", "cy"]]
        if not all(math.isfinite(value) for value in values) or min(values[:2]) <= 0:
            raise AuditError(f"INVALID_INTRINSICS:{side}")
        fx, fy, cx, cy = values
        extrinsic = (camera.get("extrinsics", {}).get(side) if nested
                     else original.get("extrinsic"))
        array = np.asarray(extrinsic, dtype=np.float64)
        if array.shape == (12,):
            array = array.reshape(3, 4)
        if array.shape == (3, 4):
            array = np.vstack([array, [0, 0, 0, 1]])
        if array.shape != (4, 4) or not np.isfinite(array).all():
            raise AuditError(f"MISSING_OR_INVALID_EXTRINSIC:{side}")
        if not np.allclose(array[3], [0, 0, 0, 1], atol=1e-6):
            raise AuditError(f"INVALID_EXTRINSIC_LAST_ROW:{side}")
        if not np.allclose(array[:3, :3].T @ array[:3, :3], np.eye(3), atol=1e-3):
            raise AuditError(f"NONRIGID_CAMERA_EXTRINSIC:{side}")
        if np.linalg.det(array[:3, :3]) < 0.99:
            raise AuditError(f"INVALID_CAMERA_EXTRINSIC_HANDEDNESS:{side}")
        eyes[side] = {"K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                      "extrinsic_original": extrinsic, "extrinsic_4x4": array.tolist(),
                      "distortion_original": original.get("distortion"),
                      "sourceIndex": original.get("sourceIndex")}
    if not metadata.get("extrinsics_included", False):
        warnings.append("CAMERA_META_EXTRINSICS_FALSE_BUT_JSON_PRESENT")
    if any(eye["sourceIndex"] is None for eye in eyes.values()):
        warnings.append("PHYSICAL_EYE_SOURCE_INDEX_ABSENT")
    warnings.append("ENCODED_IMAGE_DOMAIN_REQUIRES_PROJECTION_EVIDENCE")
    return {
        "schema": "nested_4x4" if nested else "flat_3x4",
        "reference_eye_size": size, "left": eyes["left"], "right": eyes["right"],
        "extrinsic_convention_metadata": metadata.get("extrinsic_convention"),
        "extrinsic_direction_verified": False, "physical_eye_verified": False,
        "image_domain_verified": False,
        "image_domain_raw_metadata": {key: camera[key] for key in [
            "format", "recordingApi", "cameraApi", "distortionModel", "video_info"
        ] if key in camera},
    }, warnings


def decode_video(path: str | Path, threads: int = 2) -> dict[str, Any]:
    if not 1 <= threads <= 2:
        raise AuditError("DECODE_THREADS_EXCEED_TWO")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    stream = {}
    if probe.returncode == 0:
        streams = json.loads(probe.stdout).get("streams", [])
        stream = streams[0] if streams else {}
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
               "-threads", str(threads), "-filter_threads", "1", "-i", str(path),
               "-map", "0:v:0", "-vsync", "0", "-progress", "pipe:1",
               "-nostats", "-f", "null", "-"]
    decoded = subprocess.run(command, capture_output=True, text=True, check=False)
    frames = [int(line.partition("=")[2]) for line in decoded.stdout.splitlines()
              if line.startswith("frame=")]
    return {"status": "PASSED" if probe.returncode == 0 and decoded.returncode == 0
            and not decoded.stderr.strip() and frames else "FAILED",
            "decoded_frames": max(frames, default=0), "stream": stream,
            "decode_exit_code": decoded.returncode, "probe_exit_code": probe.returncode,
            "decode_errors": decoded.stderr[-8000:], "probe_errors": probe.stderr[-2000:],
            "decoder": "ffmpeg_full_decode_xerror", "threads": threads}


def _audit_session(root: Path, session_id: str, sha_by_path: dict[str, str],
                   output_root: Path, decode_threads: int) -> dict[str, Any]:
    session = root / session_id
    errors: list[str] = []
    warnings: list[str] = []
    refs = {}
    for role, relative in RAW_NAMES.items():
        path = session / relative
        if not path.is_file():
            errors.append(f"MISSING_SOURCE:{role}")
            continue
        key = f"{session_id}/{relative}"
        if key not in sha_by_path:
            errors.append(f"SOURCE_NOT_TRANSFER_BOUND:{role}")
        refs[role] = file_ref(path, sha_by_path.get(key))
    result: dict[str, Any] = {"session_id": session_id, "source_refs": refs,
                              "errors": errors, "warnings": warnings,
                              "training_eligible": False, "control_ground_truth": False}
    try:
        if errors:
            raise AuditError("REQUIRED_SOURCE_MISSING_OR_NOT_TRANSFER_BOUND")
        for role in ["source_manifest", "selection", "workflow", "manus_meta"]:
            data = _read_json(Path(refs[role]["path"]))
            if data.get("session") != f"pick_place_1008_{session_id}":
                raise AuditError(f"SESSION_IDENTITY_MISMATCH:{role}")
        metadata = _read_json(Path(refs["camera_meta"]["path"]))
        camera_raw = _read_json(Path(refs["camera_params"]["path"]))
        if metadata.get("sha256") != refs["camera_params"]["sha256"]:
            raise AuditError("CAMERA_SNAPSHOT_SHA_MISMATCH")
        camera, camera_warnings = camera_snapshot(camera_raw, metadata)
        warnings.extend(camera_warnings)
        result.update(device_serial=metadata.get("device_serial"),
                      camera_schema=camera["schema"],
                      camera_snapshot_sha256=refs["camera_params"]["sha256"],
                      camera=camera, reference_eye_size=camera["reference_eye_size"])
        streams = _read_sensor_streams(Path(refs["pico"]["path"]), Path(refs["manus"]["path"]))
        if streams["pico"]["devices"] != [result["device_serial"]]:
            raise AuditError("CAMERA_DEVICE_PICO_IDENTITY_MISMATCH")
        video_times = read_integer_jsonl(refs["video_qpc"]["path"])
        wall_times = read_integer_jsonl(refs["video_wall"]["path"])
        clocks = {"video": clock_stats(video_times), "video_wall": clock_stats(wall_times),
                  "pico": clock_stats(streams["pico"]["times"])}
        for side in ["left", "right"]:
            clocks[f"manus_{side}"] = clock_stats(streams["manus"][side]["times"])
        result["time_stats"] = clocks
        if any(not clocks[key]["strictly_monotonic"]
               for key in ["video", "pico", "manus_left", "manus_right"]):
            raise AuditError("NONMONOTONIC_OR_EMPTY_RAW_CLOCK")
        if len(video_times) != len(wall_times):
            raise AuditError("VIDEO_QPC_WALL_COUNT_MISMATCH")
        result["raw_frame_count"] = len(video_times)
        result["actual_fps_median"] = 1000 / clocks["video"]["median_gap_ms"] \
            if clocks["video"]["median_gap_ms"] else None
        decoded = decode_video(refs["raw_vst"]["path"], decode_threads)
        result["decode"] = decoded
        result["decoded_frame_count"] = decoded["decoded_frames"]
        result["encoded_sbs_size"] = [decoded["stream"].get("width"),
                                      decoded["stream"].get("height")]
        if decoded["status"] != "PASSED":
            errors.append("RAW_FULL_DECODE_FAILED")
        if decoded["decoded_frames"] != len(video_times):
            errors.append("RAW_DECODE_QPC_COUNT_MISMATCH")
        width, height = camera["reference_eye_size"]
        if result["encoded_sbs_size"] != [width * 2, height]:
            errors.append("SBS_REFERENCE_DIMENSION_MISMATCH")
        mappings = {"pico": nearest_mapping(video_times, streams["pico"]["times"],
                                            streams["pico"]["valid"])}
        for side in ["left", "right"]:
            sensor = streams["manus"][side]
            mappings[f"manus_{side}"] = nearest_mapping(video_times, sensor["times"],
                                                        sensor["valid"])
        result["time_stats"]["mapping"] = {
            key: mapping["stats"] for key, mapping in mappings.items()}
        complete = [all(mapping["valid"][index] for mapping in mappings.values())
                    for index in range(len(video_times))]
        complete_coverage = sum(complete) / len(complete)
        result["time_stats"]["complete"] = {
            "count": sum(complete), "total": len(complete), "coverage": complete_coverage,
            "status": "PASSED" if complete_coverage >= 0.95 else "FAILED"}
        if any(mapping["stats"]["status"] != "PASSED" for mapping in mappings.values()):
            errors.append("RAW_VIDEO_SENSOR_TIME_GATE_FAILED")
        if complete_coverage < 0.95:
            errors.append("RAW_VIDEO_COMPLETE_COVERAGE_FAILED")
        rows = []
        for index, timestamp in enumerate(video_times):
            row = {"session_id": session_id, "raw_frame_index": index,
                   "video_qpc_ns": timestamp, "video_wall_ns": wall_times[index],
                   "relative_qpc_ns": timestamp - video_times[0],
                   "time_origin_qpc_ns": video_times[0], "complete_valid": complete[index],
                   "timestamp_semantics": TIME_POLICY["timestamp_semantics"]}
            for key, mapping in mappings.items():
                selected = mapping["indices"][index]
                sensor = streams["pico"] if key == "pico" else streams["manus"][key[6:]]
                row[f"{key}_source_row"] = sensor["rows"][selected]
                row[f"{key}_source_qpc_ns"] = sensor["times"][selected]
                row[f"{key}_offset_ns"] = mapping["offsets_ns"][index]
                row[f"{key}_valid"] = mapping["valid"][index]
            rows.append(row)
        mapping_path = output_root / "sessions" / session_id / "raw_frame_mapping.jsonl"
        _write_jsonl(mapping_path, rows)
        result["raw_frame_mapping"] = file_ref(mapping_path)
        result["raw_counts"] = {"video_qpc": len(video_times), "video_wall": len(wall_times),
                               "pico": len(streams["pico"]["times"]),
                               "manus_left": len(streams["manus"]["left"]["times"]),
                               "manus_right": len(streams["manus"]["right"]["times"])}
        result["raw_metadata"] = {
            "pico_serials": streams["pico"]["devices"],
            "manus_glove_ids": {side: sorted(streams["manus"][side]["glove_ids"])
                                for side in ["left", "right"]},
            "pico_invalid_components": {key: sum(not x for x in values)
                                        for key, values in streams["pico"]["components"].items()},
            "hdf5_consumption": "NONE_HASH_AND_PRIOR_REFERENCE_ONLY",
        }
    except (AuditError, KeyError, ValueError, OSError, TypeError) as exc:
        errors.append(str(exc))
    result["input_status"] = "READY_RAW_AUDIT" if not errors else "FAILED_RAW_AUDIT"
    _write_json(output_root / "sessions" / session_id / "AUDIT.json", result)
    return result


def run_p0_audit(source_root: str | Path, output_root: str | Path,
                 manifest_path: str | Path | None = None, hash_workers: int = 2,
                 decode_workers: int = 2, decode_threads: int = 2,
                 expected_sessions: int = 199, expected_frames: int = 29322,
                 progress_callback: Callable | None = None) -> dict[str, Any]:
    if not 1 <= decode_workers <= 2 or not 1 <= decode_threads <= 2:
        raise AuditError("DECODE_RESOURCE_LIMIT_EXCEEDED")
    root = Path(source_root).resolve(strict=True)
    output = _output_path(output_root)
    manifest_out = _output_path(manifest_path or output / "INPUT_MANIFEST.json")
    output.mkdir(parents=True, exist_ok=True)
    # Admission uses a real bounded write; statvfs capacity is not a hard gate.
    started = time.monotonic()
    transfer_path = root / "_inspection/transfer_receipt.json"
    checksum_path = root / "_inspection/SHA256SUMS"
    transfer = _read_json(transfer_path)
    checksum = verify_checksums(root, checksum_path, hash_workers, progress_callback)
    _write_json(output / "TRANSFER_SHA_AUDIT.json", {
        key: value for key, value in checksum.items() if key != "sha_by_relative_path"})
    if checksum["status"] != "PASSED":
        raise AuditError("SOURCE_DRIFT_TRANSFER_SHA_FAILED")
    if checksum["entries_verified"] != transfer.get("checksum_entries_verified"):
        raise AuditError("TRANSFER_CHECKSUM_COUNT_MISMATCH")
    if transfer.get("sessions") != expected_sessions:
        raise AuditError("TRANSFER_SESSION_COUNT_MISMATCH")
    session_ids = [f"{index:03}" for index in range(1, expected_sessions + 1)]
    actual_ids = sorted(path.name for path in root.iterdir()
                        if path.is_dir() and re.fullmatch(r"\d{3}", path.name))
    if actual_ids != session_ids:
        raise AuditError("FROZEN_SESSION_SET_MISMATCH")
    sessions = []
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        futures = {pool.submit(_audit_session, root, sid, checksum["sha_by_relative_path"],
                               output, decode_threads): sid for sid in session_ids}
        for future in as_completed(futures):
            sessions.append(future.result())
            if progress_callback:
                progress_callback("session", len(sessions), expected_sessions)
    sessions.sort(key=lambda row: row["session_id"])
    total_raw = sum(row.get("raw_frame_count", 0) for row in sessions)
    total_decoded = sum(row.get("decoded_frame_count", 0) for row in sessions)
    failed = [row["session_id"] for row in sessions if row["errors"]]
    frame_pass = total_raw == expected_frames and total_decoded == expected_frames
    manifest = {
        "schema": "pick_place_1008_raw_manifest_v1", "batch_id": "pick_place_1008",
        "source_root": str(root), "expected_sessions": expected_sessions,
        "expected_raw_frames": expected_frames, "time_policy": TIME_POLICY,
        "transfer": {"receipt": file_ref(transfer_path),
                     "checksum_manifest": file_ref(checksum_path),
                     "entries_verified": checksum["entries_verified"]},
        "sessions": sessions, "training_eligible": False, "control_ground_truth": False,
    }
    _write_json(manifest_out, manifest)
    devices = {}
    for row in sessions:
        serial = row.get("device_serial", "UNRESOLVED")
        devices[serial] = devices.get(serial, 0) + 1
    audit = {
        "schema": "pick_place_1008_p0_audit_v1",
        "status": "PASSED" if not failed and frame_pass else "FAILED",
        "sessions": len(sessions), "expected_sessions": expected_sessions,
        "raw_frames": total_raw, "decoded_frames": total_decoded,
        "expected_raw_frames": expected_frames, "failed_sessions": failed,
        "device_counts": devices, "checksum_entries_verified": checksum["entries_verified"],
        "camera_projection_qualification": "P1_REQUIRED_NOT_CLAIMED",
        "input_manifest": file_ref(manifest_out), "elapsed_seconds": time.monotonic() - started,
        "training_eligible": False, "control_ground_truth": False,
    }
    _write_json(output / "AUDIT.json", audit)
    result = {**audit, "operation": "run_exact78_pick_place_audit_candidate_v1",
              "audit": file_ref(output / "AUDIT.json"),
              "source_data_modified": False, "algorithm_models_called": 0}
    _write_json(output / "RESULT.json", result)
    return result


def read_video_pts_ns(path: str | Path) -> list[int]:
    """Return decoded display timestamps in nanoseconds, using the real time base."""
    from fractions import Fraction
    inspected = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=time_base:frame=pts,best_effort_timestamp",
         "-show_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if inspected.returncode != 0 or inspected.stderr.strip():
        raise AuditError(f"VIDEO_PTS_INSPECTION_FAILED:{inspected.stderr[-2000:]}")
    data = json.loads(inspected.stdout)
    streams = data.get("streams", [])
    if len(streams) != 1 or "time_base" not in streams[0]:
        raise AuditError("VIDEO_TIME_BASE_MISSING")
    time_base = Fraction(streams[0]["time_base"])
    values = []
    for row in data.get("frames", []):
        timestamp = row.get("pts", row.get("best_effort_timestamp"))
        if timestamp is None:
            raise AuditError("VIDEO_FRAME_PTS_MISSING")
        values.append(round(int(timestamp) * time_base * 1_000_000_000))
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise AuditError("VIDEO_PTS_NOT_STRICTLY_INCREASING")
    return values


def validate_video_timeline(pts_ns: list[int], raw_qpc_ns: list[int],
                            tolerance_ns: int = 1000) -> list[str]:
    if len(pts_ns) != len(raw_qpc_ns) or not raw_qpc_ns:
        return ["DELIVERY_PTS_COUNT_MISMATCH"]
    if any(right <= left for left, right in zip(pts_ns, pts_ns[1:])):
        return ["DELIVERY_PTS_NOT_STRICTLY_INCREASING"]
    expected = [value - raw_qpc_ns[0] for value in raw_qpc_ns]
    actual = [value - pts_ns[0] for value in pts_ns]
    if any(abs(a - b) > tolerance_ns for a, b in zip(actual, expected)):
        return ["DELIVERY_PTS_QPC_MISMATCH"]
    return []

def _verify_exact_ref(reference: dict[str, Any], project_only: bool = False
                      ) -> dict[str, Any]:
    path = Path(reference["path"]).resolve(strict=True)
    if project_only and not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise AuditError("EVIDENCE_REFERENCE_OUTSIDE_PROJECT")
    actual = file_ref(path)
    if actual != reference:
        raise AuditError(f"EVIDENCE_REFERENCE_DRIFT:{path}")
    return actual


def _verify_p0_manifest_evidence(manifest_path: Path, manifest: dict[str, Any]
                                 ) -> dict[str, Any]:
    manifest_ref = file_ref(manifest_path)
    stage = manifest_path.parent
    stage_result_path = stage / "RESULT.json"
    audit_path = stage / "AUDIT.json"
    sha_audit_path = stage / "TRANSFER_SHA_AUDIT.json"
    outer_path = stage.parent / "RESULT.json"
    result = _read_json(stage_result_path)
    audit = _read_json(audit_path)
    outer = _read_json(outer_path)
    expected_outputs = {
        "candidate/INPUT_MANIFEST.json": manifest_ref,
        "candidate/RESULT.json": file_ref(stage_result_path),
        "candidate/AUDIT.json": file_ref(audit_path),
        "candidate/TRANSFER_SHA_AUDIT.json": file_ref(sha_audit_path),
    }
    if (outer.get("status") not in ["PASSED", "FAILED_RUNTIME_FINAL"]
            or outer.get("validation_errors") != []
            or outer.get("stage_result") != expected_outputs["candidate/RESULT.json"]
            or any(outer.get("required_outputs", {}).get(key) != ref
                   for key, ref in expected_outputs.items())
            or result.get("input_manifest") != manifest_ref
            or audit.get("input_manifest") != manifest_ref
            or result.get("audit") != expected_outputs["candidate/AUDIT.json"]
            or result.get("operation") != "run_exact78_pick_place_audit_candidate_v1"):
        raise AuditError("P0_SEALED_RESULT_MANIFEST_BINDING_MISMATCH")
    _verify_exact_ref(outer["task_packet"], project_only=True)
    packet = _read_json(Path(outer["task_packet"]["path"]))
    if (packet.get("task_id") != outer.get("task_id")
            or packet.get("run_signature") != outer.get("run_signature")):
        raise AuditError("P0_TASK_SIGNATURE_BINDING_MISMATCH")
    receipt_path = PROJECT_ROOT / "tasks/receipts" / f"{outer['task_id'].upper()}.json"
    receipt = _read_json(receipt_path)
    if receipt.get("result") != file_ref(outer_path) or receipt.get("status") != outer["status"]:
        raise AuditError("P0_TERMINAL_RECEIPT_BINDING_MISMATCH")
    for key in ["receipt", "checksum_manifest"]:
        _verify_exact_ref(manifest["transfer"][key])
    transfer = _read_json(Path(manifest["transfer"]["receipt"]["path"]))
    sha_audit = _read_json(sha_audit_path)
    expected_entries = manifest["transfer"]["entries_verified"]
    if (sha_audit.get("status") != "PASSED" or sha_audit.get("failures") != []
            or sha_audit.get("entries_verified") != expected_entries
            or sha_audit.get("entries_expected") != expected_entries
            or transfer.get("checksum_entries_verified") != expected_entries
            or result.get("checksum_entries_verified") != expected_entries
            or result.get("raw_frames") != manifest["expected_raw_frames"]
            or result.get("decoded_frames") != manifest["expected_raw_frames"]
            or result.get("sessions") != manifest["expected_sessions"]):
        raise AuditError("P0_SOURCE_COMPLETENESS_PROOF_MISMATCH")
    for key in ["max_skew_ms", "p95_skew_ms", "min_complete_coverage"]:
        if manifest.get("time_policy", {}).get(key) != TIME_POLICY[key]:
            raise AuditError("TIME_POLICY_NUMERIC_GATE_DRIFT")
    return {"input_manifest": manifest_ref, "stage_result": file_ref(stage_result_path),
            "audit": file_ref(audit_path), "transfer_sha_audit": file_ref(sha_audit_path),
            "executor_result": file_ref(outer_path), "terminal_receipt": file_ref(receipt_path),
            "task_packet": outer["task_packet"], "transfer": manifest["transfer"]}


def _accepted_mapping_stats(mapping: dict[str, Any]) -> dict[str, Any]:
    accepted = [abs(offset) / 1e6 for offset, valid in zip(
        mapping["offsets_ns"], mapping["valid"]) if valid]
    total = len(mapping["valid"])
    coverage = len(accepted) / total if total else 0.0
    maximum = max(accepted) if accepted else None
    p95 = float(np.percentile(accepted, 95)) if accepted else None
    return {"count": len(accepted), "total": total, "coverage": coverage,
            "max_ms": maximum, "p95_ms": p95,
            "status": "PASSED" if accepted and maximum <= 40 and p95 <= 30
                      and coverage >= .95 else "FAILED",
            "scope": "ACCEPTED_OBSERVED_RAW_SOURCE_MATCHES_ONLY",
            "invalid_matches_retained_in_raw_nearest_metrics": True}


def _short_gap_records(targets: list[int], sensor: dict[str, Any],
                        mapping: dict[str, Any], stream: str) -> list[dict[str, Any]]:
    invalid = [index for index, valid in enumerate(mapping["valid"]) if not valid]
    if not invalid:
        return []
    for index in invalid:
        if not sensor["valid"][mapping["indices"][index]]:
            raise AuditError("INVALID_RAW_SENSOR_CONTENT_NOT_TIME_BOUNDARY")
    anchor_indices = [index for index, valid in enumerate(sensor["valid"]) if valid]
    anchor_times = [sensor["times"][index] for index in anchor_indices]
    groups = []
    for index in invalid:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    records = []
    for group in groups:
        start, end = group[0], group[-1]
        before_index = bisect.bisect_right(anchor_times, targets[start]) - 1
        after_index = bisect.bisect_left(anchor_times, targets[end])
        before = anchor_indices[before_index] if before_index >= 0 else None
        after = anchor_indices[after_index] if after_index < len(anchor_indices) else None
        if before is None and after is None:
            raise AuditError("SHORT_GAP_VALID_ANCHOR_ABSENT")
        bounds = [targets[start], targets[end]]
        if before is not None:
            bounds.append(sensor["times"][before])
        if after is not None:
            bounds.append(sensor["times"][after])
        duration = max(bounds) - min(bounds)
        if duration > 500_000_000:
            raise AuditError("INVALID_TIME_GAP_EXCEEDS_0_5_SECONDS")
        if before is None:
            policy, alias = "HOLD_FIRST_VALID_SHORT_GAP", "EDGE_HOLD_SHORT"
        elif after is None:
            policy, alias = "HOLD_LAST_VALID_SHORT_GAP", "EDGE_HOLD_SHORT"
        else:
            policy, alias = "INTERPOLATE_VALID_ANCHORS_SHORT_GAP", "INTERP_SHORT"
        records.append({
            "stream": stream, "start_frame": start, "end_frame_inclusive": end,
            "first_qpc_ns": targets[start], "last_qpc_ns": targets[end],
            "gap_duration_ns": duration, "display_policy": policy,
            "producer_source_alias": alias, "raw_observed": False,
            "anchor_source_row_before": sensor["rows"][before] if before is not None else None,
            "anchor_source_row_after": sensor["rows"][after] if after is not None else None,
            "qualification": "OFFLINE_DISPLAY_ONLY_INVALID_RAW_TIME_MATCH",
        })
    return records


def resolve_time_boundaries(manifest_path: str | Path, output_root: str | Path
                             ) -> dict[str, Any]:
    """Admit documented <=0.5s display gaps without rewriting any raw time gate."""
    import copy
    path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(path)
    evidence = _verify_p0_manifest_evidence(path, manifest)
    original_manifest = copy.deepcopy(manifest)
    output = _output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    affected = {}
    for row in manifest["sessions"]:
        if row.get("input_status") == "READY_RAW_AUDIT":
            continue
        sid = row["session_id"]
        if (row.get("input_status") != "FAILED_RAW_AUDIT"
                or row.get("errors") != ["RAW_VIDEO_SENSOR_TIME_GATE_FAILED"]
                or row.get("decoded_frame_count") != row.get("raw_frame_count")
                or row.get("time_stats", {}).get("complete", {}).get("coverage", 0) < .95):
            raise AuditError(f"TIME_BOUNDARY_NOT_SINGLE_RESOLVABLE_FAILURE:{sid}")
        for reference in row["source_refs"].values():
            _verify_exact_ref(reference)
        _verify_exact_ref(row["raw_frame_mapping"], project_only=True)
        refs = row["source_refs"]
        targets = read_integer_jsonl(refs["video_qpc"]["path"])
        wall = read_integer_jsonl(refs["video_wall"]["path"])
        sensors = _read_sensor_streams(Path(refs["pico"]["path"]), Path(refs["manus"]["path"]))
        if sensors["pico"]["devices"] != [row["device_serial"]]:
            raise AuditError("TIME_RESOLUTION_DEVICE_IDENTITY_MISMATCH")
        streams = {"pico": sensors["pico"], **{
            f"manus_{side}": sensors["manus"][side] for side in ["left", "right"]}}
        mappings = {name: nearest_mapping(targets, stream["times"], stream["valid"])
                    for name, stream in streams.items()}
        original_rows = [json.loads(line) for line in Path(row["raw_frame_mapping"]["path"])
                         .read_text(encoding="utf-8-sig").splitlines()]
        if len(original_rows) != len(targets) or len(targets) != row["raw_frame_count"]:
            raise AuditError("TIME_RESOLUTION_RAW_FRAME_COUNT_MISMATCH")
        for index, recorded in enumerate(original_rows):
            if (recorded.get("raw_frame_index") != index
                    or recorded.get("video_qpc_ns") != targets[index]
                    or recorded.get("video_wall_ns") != wall[index]
                    or recorded.get("relative_qpc_ns") != targets[index] - targets[0]):
                raise AuditError("TIME_RESOLUTION_RAW_TIMELINE_DRIFT")
            for name, mapping in mappings.items():
                selected = mapping["indices"][index]
                expected = {f"{name}_source_row": streams[name]["rows"][selected],
                            f"{name}_source_qpc_ns": streams[name]["times"][selected],
                            f"{name}_offset_ns": mapping["offsets_ns"][index],
                            f"{name}_valid": mapping["valid"][index]}
                if any(recorded.get(key) != value for key, value in expected.items()):
                    raise AuditError("TIME_RESOLUTION_RAW_NEAREST_MAPPING_DRIFT")
        accepted = {name: _accepted_mapping_stats(mapping) for name, mapping in mappings.items()}
        complete = [all(mapping["valid"][index] for mapping in mappings.values())
                    for index in range(len(targets))]
        if any(stats["status"] != "PASSED" for stats in accepted.values()) \
                or sum(complete) / len(complete) < .95:
            raise AuditError("TIME_RESOLUTION_ACCEPTED_MATCH_GATE_FAILED")
        gaps = [gap for name, mapping in mappings.items()
                for gap in _short_gap_records(targets, streams[name], mapping, name)]
        if not gaps:
            raise AuditError("TIME_RESOLUTION_NO_CONFIRMED_INVALID_GAP")
        derived = copy.deepcopy(original_rows)
        for gap in gaps:
            name = gap["stream"]
            for index in range(gap["start_frame"], gap["end_frame_inclusive"] + 1):
                derived[index][f"{name}_observation_status"] = "INVALID_TIME_GAP"
                derived[index][f"{name}_display_policy"] = gap["display_policy"]
                derived[index][f"{name}_producer_source_alias"] = gap["producer_source_alias"]
                derived[index][f"{name}_gap_duration_ns"] = gap["gap_duration_ns"]
                derived[index][f"{name}_raw_observed"] = False
                derived[index][f"{name}_display_fill_allowed"] = True
        derived_path = output / "sessions" / sid / "resolved_raw_frame_mapping.jsonl"
        _write_jsonl(derived_path, derived)
        original_row = next(item for item in original_manifest["sessions"] if item["session_id"] == sid)
        affected[sid] = {
            "original_input_status": original_row["input_status"],
            "original_errors": original_row["errors"],
            "source_refs": original_row["source_refs"],
            "original_mapping": original_row["raw_frame_mapping"],
            "derived_mapping": file_ref(derived_path), "gaps": gaps,
            "raw_nearest_mapping_statistics": original_row["time_stats"]["mapping"],
            "accepted_mapping_statistics": accepted,
            "complete_coverage": sum(complete) / len(complete),
            "raw_frame_count": len(targets), "qpc_modified": False,
            "numeric_gates_modified": False, "training_eligible": False,
            "control_ground_truth": False,
        }
        row["raw_audit_status"] = original_row["input_status"]
        row["input_status"] = "READY_RAW_WITH_INVALID_SHORT_GAP"
        row["original_raw_frame_mapping"] = original_row["raw_frame_mapping"]
        row["raw_frame_mapping"] = file_ref(derived_path)
        row["time_stats"]["accepted_mapping"] = accepted
        row["warnings"].append("INVALID_SHORT_GAP_OFFLINE_DISPLAY_SOURCE_RECORDED")
    resolution = {
        "schema": "pick_place_1008_time_boundary_resolution_v1",
        "status": "PASSED", "source_evidence": evidence, "affected_sessions": affected,
        "ready_unchanged_sessions": [row["session_id"] for row in original_manifest["sessions"]
                                      if row["input_status"] == "READY_RAW_AUDIT"],
        "expected_sessions": manifest["expected_sessions"],
        "expected_raw_frames": manifest["expected_raw_frames"],
        "qpc_modified": False, "numeric_gates_modified": False,
        "display_gap_max_ns": 500_000_000, "time_policy": TIME_POLICY,
        "training_eligible": False, "control_ground_truth": False,
    }
    resolution_path = output / "RESOLUTION.json"
    _write_json(resolution_path, resolution)
    resolution_ref = file_ref(resolution_path)
    manifest["time_boundary_resolution"] = resolution_ref
    manifest["original_input_manifest"] = evidence["input_manifest"]
    for row in manifest["sessions"]:
        if row["session_id"] in affected:
            row["time_boundary_resolution"] = resolution_ref
    resolved_path = output / "RESOLVED_INPUT_MANIFEST.json"
    _write_json(resolved_path, manifest)
    result = {
        "schema": "pick_place_1008_time_resolution_result_v1", "status": "PASSED",
        "resolution": resolution_ref, "input_manifest": file_ref(resolved_path),
        "original_input_manifest": evidence["input_manifest"],
        "resolved_sessions": sorted(affected), "ready_sessions": len(manifest["sessions"]),
        "raw_frames": manifest["expected_raw_frames"], "source_data_modified": False,
        "qpc_modified": False, "numeric_gates_modified": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    _write_json(output / "RESULT.json", result)
    return result


def _verify_resolution_producer(reference: dict[str, Any]) -> None:
    stage = Path(reference["path"]).parent
    result_ref = file_ref(stage / "RESULT.json")
    result = _read_json(stage / "RESULT.json")
    outer_path = stage.parent / "RESULT.json"
    outer = _read_json(outer_path)
    if (result.get("status") != "PASSED" or result.get("resolution") != reference
            or outer.get("status") != "PASSED" or outer.get("stage_result") != result_ref
            or outer.get("required_outputs", {}).get("candidate/RESULT.json") != result_ref
            or outer.get("required_outputs", {}).get("candidate/RESOLUTION.json") != reference):
        raise AuditError("TIME_RESOLUTION_PRODUCER_NOT_SEALED_OR_UNBOUND")
    _verify_exact_ref(result["input_manifest"], project_only=True)
    packet_ref = _verify_exact_ref(outer["task_packet"], project_only=True)
    packet = _read_json(Path(packet_ref["path"]))
    if (packet.get("task_id") != outer.get("task_id")
            or packet.get("run_signature") != outer.get("run_signature")):
        raise AuditError("TIME_RESOLUTION_PRODUCER_SIGNATURE_MISMATCH")
    receipt = _read_json(PROJECT_ROOT / "tasks/receipts" / f"{outer['task_id'].upper()}.json")
    if receipt.get("result") != file_ref(outer_path) or receipt.get("status") != "PASSED":
        raise AuditError("TIME_RESOLUTION_PRODUCER_RECEIPT_MISMATCH")


def validate_resolved_frame_states(source: dict[str, Any], manifest: dict[str, Any],
                                   frame_rows: list[dict[str, Any]],
                                   state_rows: list[dict[str, Any]]) -> list[str]:
    if source.get("input_status") == "READY_RAW_AUDIT":
        return []
    if source.get("input_status") != "READY_RAW_WITH_INVALID_SHORT_GAP":
        return ["SOURCE_RAW_AUDIT_UNREADY"]
    errors = []
    try:
        reference = source.get("time_boundary_resolution")
        if reference != manifest.get("time_boundary_resolution") or not isinstance(reference, dict):
            raise AuditError("TIME_RESOLUTION_REFERENCE_UNBOUND")
        _verify_exact_ref(reference, project_only=True)
        _verify_resolution_producer(reference)
        resolution = _read_json(Path(reference["path"]))
        approved = resolution["affected_sessions"][source["session_id"]]
        if (resolution.get("status") != "PASSED" or resolution.get("qpc_modified") is not False
                or resolution.get("numeric_gates_modified") is not False
                or source.get("raw_audit_status") != "FAILED_RAW_AUDIT"
                or approved["derived_mapping"] != source["raw_frame_mapping"]
                or source.get("source_refs") != approved["source_refs"]):
            raise AuditError("TIME_RESOLUTION_SESSION_BINDING_MISMATCH")
        _verify_exact_ref(approved["derived_mapping"], project_only=True)
        for gap in approved["gaps"]:
            name = gap["stream"]
            if gap["gap_duration_ns"] > 500_000_000 or gap.get("raw_observed") is not False:
                raise AuditError("TIME_RESOLUTION_GAP_QUALIFICATION_INVALID")
            for index in range(gap["start_frame"], gap["end_frame_inclusive"] + 1):
                mapping, state = frame_rows[index], state_rows[index]
                if name.startswith("manus_"):
                    hand = state["hands"][name[6:]]
                    if (hand.get("raw_observed") is not False
                            or hand.get("source") != gap["producer_source_alias"]
                            or hand.get("display_valid") is not True
                            or hand.get("handshape_valid") is not False):
                        errors.append(f"RESOLVED_DISPLAY_GAP_SOURCE_NOT_RECORDED:{name}:{index}")
                    offset = mapping.get(f"{name}_skew_ns", mapping.get(f"{name}_offset_ns"))
                    derived_row = json.loads(Path(approved["derived_mapping"]["path"]).read_text(
                        encoding="utf-8-sig").splitlines()[index])
                    if offset != derived_row[f"{name}_offset_ns"]:
                        errors.append(f"RESOLVED_RAW_NEAREST_OFFSET_CHANGED:{name}:{index}")
                elif state.get("camera_raw_valid") is not False:
                    errors.append(f"RESOLVED_CAMERA_RAW_VALIDITY_FALSE_CLAIM:{index}")
    except (AuditError, KeyError, ValueError, OSError, IndexError, TypeError) as exc:
        errors.append(f"TIME_RESOLUTION_DELIVERY_INVALID:{exc}")
    return errors


def validate_action_arrays(path: str | Path, frame_count: int) -> list[str]:
    errors = []
    try:
        with np.load(path, allow_pickle=False) as arrays:
            for name, shape in {"q_arm": (frame_count, 2, 7),
                                "q_hand": (frame_count, 2, 22),
                                "T_world_base": (4, 4),
                                "c2w": (frame_count, 4, 4)}.items():
                if name not in arrays:
                    errors.append(f"MISSING_ACTION_ARRAY:{name}")
                elif arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                    errors.append(f"INVALID_ACTION_ARRAY:{name}")
            if "T_world_base" in arrays and arrays["T_world_base"].shape == (4, 4):
                if not np.allclose(arrays["T_world_base"][3], [0, 0, 0, 1]):
                    errors.append("INVALID_BASE_LAST_ROW")
    except (OSError, ValueError) as exc:
        errors.append(f"ACTION_ARRAY_UNREADABLE:{exc}")
    return errors


def _validate_review_requirements(review: dict[str, Any], sealed: dict[str, Any],
                                  session_ref: dict[str, Any], frame_count: int
                                  ) -> list[str]:
    errors = []
    requirements = sealed.get("ai_review")
    if not isinstance(requirements, dict):
        return ["SEALED_AI_REVIEW_REQUIREMENTS_MISSING"]
    inspected = review.get("ai_inspected_raw_frame_indices", [])
    inspected_set = set(inspected) if isinstance(inspected, list) and all(
        type(frame) is int for frame in inspected) else set()
    expected_fixed = sorted({0, frame_count - 1, *[
        int(round((frame_count - 1) * fraction)) for fraction in (.25, .5, .75)]})
    fixed = requirements.get("required_fixed_frames")
    if fixed != expected_fixed:
        errors.append("SEALED_FIXED_FRAME_REQUIREMENTS_INVALID")
    elif not set(fixed).issubset(inspected_set):
        errors.append("REQUIRED_FIXED_FRAMES_NOT_INSPECTED")
    abnormal = requirements.get("abnormal_raw_frame_indices")
    if (not isinstance(abnormal, list)
            or any(type(frame) is not int or not 0 <= frame < frame_count for frame in abnormal)
            or len(set(abnormal)) != len(abnormal)):
        errors.append("SEALED_ABNORMAL_FRAME_REQUIREMENTS_MISSING_OR_INVALID")
    elif isinstance(requirements.get("required_abnormal_windows"), list):
        windows = requirements["required_abnormal_windows"]
        if any(not any(isinstance(window, dict) and type(window.get("start_frame")) is int
                       and type(window.get("end_frame_inclusive")) is int
                       and window["start_frame"] <= frame <= window["end_frame_inclusive"]
                       for window in windows) for frame in abnormal):
            errors.append("SEALED_ABNORMAL_FRAMES_WINDOW_COVERAGE_MISMATCH")
    for required_key, coverage_key in [
        ("required_candidate_windows", "candidate_windows_coverage"),
        ("required_abnormal_windows", "abnormal_windows_coverage"),
    ]:
        expected = requirements.get(required_key)
        actual = review.get(coverage_key)
        if not isinstance(expected, list):
            errors.append(f"SEALED_REVIEW_WINDOWS_MISSING:{required_key}")
            continue
        if (not isinstance(actual, dict) or actual.get("source") != session_ref
                or actual.get("total") != len(expected)
                or actual.get("reviewed") != len(expected)
                or actual.get("all_covered") is not True
                or not isinstance(actual.get("windows"), list)
                or len(actual["windows"]) != len(expected)):
            errors.append(f"REVIEW_WINDOWS_SOURCE_COUNT_BINDING_MISMATCH:{coverage_key}")
            continue
        for index, (window, checked_window) in enumerate(zip(expected, actual["windows"])):
            if not isinstance(window, dict) or not isinstance(checked_window, dict):
                errors.append(f"REVIEW_WINDOW_INVALID:{coverage_key}:{index}")
                continue
            declared = {key: value for key, value in checked_window.items()
                        if key != "checked_raw_frame_indices"}
            checked = checked_window.get("checked_raw_frame_indices")
            start = window.get("start_frame")
            end = window.get("end_frame_inclusive")
            if (declared != window or type(start) is not int or type(end) is not int
                    or not 0 <= start <= end < frame_count
                    or not isinstance(checked, list) or not checked
                    or any(type(frame) is not int or not start <= frame <= end
                           or frame not in inspected_set for frame in checked)):
                errors.append(f"REVIEW_WINDOW_RANGE_OR_INSPECTION_MISMATCH:{coverage_key}:{index}")
    return errors


def validate_quality_review(delivery: dict[str, Any], session_ref: dict[str, Any],
                            session_id: str, frame_count: int
                            ) -> tuple[list[str], dict[str, Any] | None]:
    errors = []
    review_ref = delivery.get("quality_review")
    if not isinstance(review_ref, dict):
        return ["MISSING_INDEPENDENT_QUALITY_REVIEW"], None
    try:
        path = Path(review_ref["path"]).resolve(strict=True)
        if not path.is_relative_to(PROJECT_ROOT.resolve()):
            raise AuditError("QUALITY_REVIEW_OUTSIDE_PROJECT")
        if sha256_file(path) != review_ref["sha256"]:
            return ["QUALITY_REVIEW_SHA_MISMATCH"], None
        review = _read_json(path)
        if review.get("session_id") != session_id:
            errors.append("QUALITY_REVIEW_SESSION_IDENTITY_MISMATCH")
        actual_session_ref = file_ref(session_ref["path"])
        if review.get("session_result") != actual_session_ref:
            errors.append("QUALITY_REVIEW_SESSION_RESULT_BINDING_MISMATCH")
        if review.get("visual_status") not in ["PASSED", "ACCEPTED_WITH_DEFECTS"]:
            errors.append("VISUAL_QUALITY_UNACCEPTED")
        issues = review.get("severe_issues_open")
        if not ((type(issues) is int and issues == 0)
                or (isinstance(issues, list) and not issues)):
            errors.append("SEVERE_VISUAL_ISSUES_OPEN_OR_UNRECORDED")
        inspected = review.get("ai_inspected_raw_frame_indices")
        if (not isinstance(inspected, list) or not inspected
                or any(type(frame) is not int or not 0 <= frame < frame_count for frame in inspected)
                or len(set(inspected)) != len(inspected)):
            errors.append("AI_INSPECTION_FRAME_EVIDENCE_MISSING_OR_INVALID")
        for key in ["candidate_windows_coverage", "abnormal_windows_coverage"]:
            coverage = review.get(key)
            if (not isinstance(coverage, dict)
                    or type(coverage.get("total")) is not int
                    or type(coverage.get("reviewed")) is not int
                    or coverage["total"] < 0
                    or coverage["reviewed"] != coverage["total"]
                    or coverage.get("all_covered") is not True):
                errors.append(f"REVIEW_WINDOW_COVERAGE_MISSING_OR_INCOMPLETE:{key}")
        sealed = _read_json(Path(session_ref["path"]))
        errors.extend(_validate_review_requirements(review, sealed, actual_session_ref, frame_count))
        if not isinstance(review.get("accepted_defects"), list):
            errors.append("QUALITY_REVIEW_ACCEPTED_DEFECTS_UNRECORDED")
        return errors, review
    except (KeyError, OSError, ValueError, AuditError) as exc:
        return [f"MISSING_OR_INVALID_QUALITY_REVIEW:{exc}"], None


def audit_deliveries(manifest_path: str | Path, delivery_index_path: str | Path,
                     output_root: str | Path) -> dict[str, Any]:
    """Check delivery bytes/structure without claiming visual or human inspection.

    Index rows have session_id and files(final, review, actions, frame_mapping,
    frame_states, session_result), each a path/sha256 reference. Quality acceptance
    is explicitly supplied by separately sealed, session-bound AI_REVIEW artifacts.
    """
    manifest = _read_json(Path(manifest_path))
    index = _read_json(Path(delivery_index_path))
    expected = {row["session_id"]: row for row in manifest["sessions"]}
    deliveries = index.get("sessions", [])
    by_id = {row["session_id"]: row for row in deliveries}
    global_errors = []
    if len(by_id) != len(deliveries):
        global_errors.append("DUPLICATE_DELIVERY_SESSION")
    if set(by_id) != set(expected):
        global_errors.append("DELIVERY_SESSION_SET_MISMATCH")
    rows = []
    for sid, source in sorted(expected.items()):
        delivery = by_id.get(sid)
        errors = []
        frame_count = source.get("raw_frame_count", 0)
        if delivery is None:
            rows.append({"session_id": sid, "status": "INCOMPLETE", "errors": ["MISSING_DELIVERY"]})
            continue
        files = delivery.get("files", {})
        review = None
        if source.get("input_status") not in ["READY_RAW_AUDIT", "READY_RAW_WITH_INVALID_SHORT_GAP"]:
            errors.append("SOURCE_RAW_AUDIT_UNREADY")
        for role in ["final", "review", "actions", "frame_mapping", "frame_states", "session_result"]:
            ref = files.get(role, {})
            try:
                path = Path(ref["path"]).resolve(strict=True)
                if not path.is_relative_to(PROJECT_ROOT.resolve()):
                    raise AuditError(f"DELIVERY_OUTSIDE_PROJECT:{role}")
                if sha256_file(path) != ref["sha256"]:
                    errors.append(f"DELIVERY_SHA_MISMATCH:{role}")
            except (KeyError, OSError, AuditError) as exc:
                errors.append(f"MISSING_OR_INVALID_DELIVERY:{role}:{exc}")
        if not errors:
            try:
                errors.extend(validate_action_arrays(files["actions"]["path"], frame_count))
                frame_rows = [json.loads(line) for line in Path(files["frame_mapping"]["path"])
                              .read_text(encoding="utf-8-sig").splitlines()]
                state_rows = [json.loads(line) for line in Path(files["frame_states"]["path"])
                              .read_text(encoding="utf-8-sig").splitlines()]
                errors.extend(validate_resolved_frame_states(source, manifest, frame_rows, state_rows))
                original = read_integer_jsonl(source["source_refs"]["video_qpc"]["path"])
                if len(frame_rows) != frame_count or len(state_rows) != frame_count:
                    errors.append("DELIVERY_FRAME_STATE_COUNT_MISMATCH")
                if [row.get("raw_frame_index") for row in frame_rows] != list(range(frame_count)):
                    errors.append("DELIVERY_RAW_FRAME_ORDER_MISMATCH")
                if [row.get("video_qpc_ns") for row in frame_rows] != original:
                    errors.append("DELIVERY_RAW_QPC_MAPPING_MISMATCH")
                for role in ["final", "review"]:
                    decoded = decode_video(files[role]["path"])
                    if decoded["status"] != "PASSED" or decoded["decoded_frames"] != frame_count:
                        errors.append(f"DELIVERY_VIDEO_DECODE_MISMATCH:{role}")
                    try:
                        pts_errors = validate_video_timeline(read_video_pts_ns(files[role]["path"]), original)
                        errors.extend(f"{error}:{role}" for error in pts_errors)
                    except (AuditError, ValueError, OSError) as exc:
                        errors.append(f"DELIVERY_PTS_INVALID:{role}:{exc}")
                sealed = _read_json(Path(files["session_result"]["path"]))
                if sealed.get("session_id") != sid:
                    errors.append("DELIVERY_RESULT_IDENTITY_MISMATCH")
                if sealed.get("training_eligible") is not False \
                        or sealed.get("control_ground_truth") is not False:
                    errors.append("DELIVERY_QUALIFICATION_NOT_FALSE")
                quality_errors, review = validate_quality_review(delivery, files["session_result"], sid, frame_count)
                errors.extend(quality_errors)
            except (AuditError, OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"DELIVERY_CONTENT_AUDIT_FAILED:{exc}")
        rows.append({"session_id": sid, "status": "PASSED" if not errors else "FAILED",
                     "errors": errors, "frame_count": frame_count,
                     "visual_status": review.get("visual_status") if review else "UNREVIEWED",
                     "human_review_status": review.get("human_review_status", "PENDING") if review else "PENDING",
                     "accepted_defects": review.get("accepted_defects", []) if review else [],
                     "quality_review": delivery.get("quality_review")})
    passed = sum(row["status"] == "PASSED" for row in rows)
    result = {
        "schema": "pick_place_1008_delivery_audit_v1",
        "status": "PASSED" if passed == manifest["expected_sessions"] and not global_errors
            and sum(row.get("frame_count", 0) for row in rows) == manifest["expected_raw_frames"]
            else "FAILED",
        "delivered_passed": passed, "expected_sessions": manifest["expected_sessions"],
        "expected_raw_frames": manifest["expected_raw_frames"],
        "errors": global_errors, "sessions": rows,
        "training_eligible": False, "control_ground_truth": False,
        "visual_inspection_performed_by_this_audit": False,
    }
    _write_json(_output_path(output_root) / "DELIVERY_AUDIT.json", result)
    return result


def main() -> None:
    raise AuditError("REGISTERED_OPERATION_REQUIRED:use chaoyang run")


if __name__ == "__main__":
    main()
