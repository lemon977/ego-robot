#!/usr/bin/env python3
"""Publish handle sessions from the acquisition-aligned HDF5 contract.

Unlike rgb30_v1, this batch does not resample MANUS/tactile/pose streams a
second time and does not reject an entire session because a second-pass repair
ratio exceeds one percent.  The acquisition ``dataset.hdf5`` is authoritative.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any

import h5py
import numpy as np


TASK_PREFIX = {"potato_chips": "get_potato_chips", "playing_cards": "play_cards"}
ADMISSION_POLICY = (
    "ACQUISITION_ALIGNED_COMPLETE_ROWS; NO_SECOND_SENSOR_RESAMPLING; "
    "VIDEO_CADENCE_REPORTED_SEPARATELY"
)
REQUIRED_ARRAYS = (
    "timestamp_ns", "recv_qpc_ns", "video_frame_idx", "head_pose",
    "left_controller_pose", "right_controller_pose", "left_wrist_pose",
    "right_wrist_pose", "left_hand_joints", "right_hand_joints",
    "left_hand_valid", "right_hand_valid", "left_tactile_values",
    "right_tactile_values",
)
FINITE_ARRAYS = (
    "head_pose", "left_controller_pose", "right_controller_pose",
    "left_wrist_pose", "right_wrist_pose", "left_hand_joints",
    "right_hand_joints",
)


class BatchError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_start_ticks(pid: int) -> int:
    # Fields after the final ')' begin with proc stat field 3; starttime is 22.
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return int(fields[19])


def acquire_batch_lock(target: Path) -> tuple[int, dict[str, Any]]:
    """Hold a non-blocking dataset lock for the lifetime of this process."""
    path = target / ".batch_convert_acquisition_aligned.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = os.read(fd, 8192).decode("utf-8", errors="replace").strip()
        except OSError:
            owner = "<owner receipt unavailable>"
        os.close(fd)
        raise BatchError(f"target already has a live batch owner: {owner}") from exc
    owner = {
        "schema_version": "handle-acquisition-aligned-batch-lock-v1",
        "pid": os.getpid(),
        "start_ticks": process_start_ticks(os.getpid()),
        "acquired_unix": time.time(),
        "target": str(target),
    }
    payload = (json.dumps(owner, ensure_ascii=False, sort_keys=True) + "\n").encode()
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload)
    os.fsync(fd)
    return fd, owner


def parse_task_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise BatchError("--task-source must be TASK=/absolute/path")
    task, raw = value.split("=", 1)
    if task not in TASK_PREFIX:
        raise BatchError(f"unsupported task: {task}")
    return task, Path(raw).resolve(strict=True)


def sessions(root: Path) -> list[str]:
    values = sorted(p.name for p in root.iterdir()
                    if p.is_dir() and len(p.name) == 3 and p.name.isdigit())
    if not values:
        raise BatchError(f"no NNN sessions below {root}")
    return values


def validate_acquisition(source: Path) -> dict[str, Any]:
    required_files = (
        "dataset.hdf5", "raw/vst.h264", "raw/vst.ts.jsonl",
        "raw/vst.qpc.ts.jsonl", "raw/camera_params.json",
        "raw/camera_params.meta.json", "raw/pico.jsonl", "raw/manus.jsonl",
        "raw/tactile.jsonl",
    )
    missing = [name for name in required_files if not (source / name).is_file()]
    if missing:
        raise BatchError(f"required acquisition files missing: {missing}")
    with h5py.File(source / "dataset.hdf5", "r") as handle:
        missing_arrays = [name for name in REQUIRED_ARRAYS if name not in handle]
        if missing_arrays:
            raise BatchError(f"required HDF5 arrays missing: {missing_arrays}")
        attrs = dict(handle.attrs)
        if attrs.get("schema") != "egodex_v1":
            raise BatchError(f"unsupported HDF5 schema: {attrs.get('schema')}")
        if not bool(attrs.get("all_exported_frames_complete")):
            raise BatchError("all_exported_frames_complete is not true")
        coverage = float(attrs.get("complete_frame_coverage", 0.0))
        if not math.isfinite(coverage) or coverage < 0.95:
            raise BatchError(f"complete_frame_coverage below 0.95: {coverage}")
        timestamp = np.asarray(handle["timestamp_ns"][:], dtype=np.int64)
        if len(timestamp) < 2 or np.any(np.diff(timestamp) <= 0):
            raise BatchError("acquisition timeline is not strictly increasing")
        n = len(timestamp)
        length_errors = [name for name in REQUIRED_ARRAYS
                         if len(handle[name]) != n]
        if length_errors:
            raise BatchError(f"HDF5 first dimension mismatch: {length_errors}")
        nonfinite = [name for name in FINITE_ARRAYS
                     if not np.isfinite(handle[name][:]).all()]
        if nonfinite:
            raise BatchError(f"non-finite acquisition arrays: {nonfinite}")
        video_idx = np.asarray(handle["video_frame_idx"][:], dtype=np.int64)
        if np.any(video_idx < 0):
            raise BatchError("negative video_frame_idx")
        diff = np.diff(video_idx)
        return {
            "status": "PASS_ACQUISITION_ALIGNED",
            "frame_count": n,
            "complete_frame_coverage": coverage,
            "dropped_incomplete_frame_count": int(
                attrs.get("dropped_incomplete_frame_count", 0)
            ),
            "schema_revision": str(attrs.get("schema_revision", "")),
            "gate_ms": float(attrs.get("gate_ms", 0.0)),
            "video_cadence": {
                "unique_source_frames": int(len(np.unique(video_idx))),
                "repeated_transitions": int(np.count_nonzero(diff == 0)),
                "skipped_transitions": int(np.count_nonzero(diff > 1)),
                "negative_transitions": int(np.count_nonzero(diff < 0)),
            },
        }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BatchError(f"atomic receipt mismatch for {label}: {actual!r} != {expected!r}")


def validate_atomic_converter_receipt(
    result: dict[str, Any], target: Path, source: Path, preflight: dict[str, Any]
) -> None:
    """Validate a converter-published target before adopting it into batch state.

    This path is used only when the converter completed its atomic publish but
    the former batch worker was stopped before it could add acquisition policy
    metadata and report the result to the parent STATE.json.
    """
    frame_count = int(preflight["frame_count"])
    expected_session = target.name
    _require_equal(result.get("status"), "PASS_FORMAT_COMPATIBLE", "status")
    _require_equal(result.get("source"), str(source), "source")
    _require_equal(result.get("target"), str(target), "target")
    _require_equal(result.get("session_id"), expected_session, "session_id")
    _require_equal(
        result.get("image_domain_mode"),
        "passthrough_scaled_source_domain",
        "image_domain_mode",
    )
    _require_equal(
        result.get("selection_convention"), "legacy_tracker_left", "selection_convention"
    )
    _require_equal(result.get("frame_count"), frame_count, "frame_count")

    validation = result.get("validation")
    if not isinstance(validation, dict):
        raise BatchError("atomic receipt has no validation object")
    for key in ("frame_count", "all_data_count", "controller_lines", "slam_lines"):
        _require_equal(validation.get(key), frame_count, f"validation.{key}")
    _require_equal(
        validation.get("status"), "PASS_FORMAT_COMPATIBLE", "validation.status"
    )
    for probe_name in ("mono_probe", "stereo_probe"):
        probe = validation.get(probe_name)
        if not isinstance(probe, dict):
            raise BatchError(f"atomic receipt has no validation.{probe_name}")
        try:
            decoded_frames = int(probe.get("nb_read_frames"))
        except (TypeError, ValueError) as exc:
            raise BatchError(
                f"invalid validation.{probe_name}.nb_read_frames: "
                f"{probe.get('nb_read_frames')!r}"
            ) from exc
        _require_equal(decoded_frames, frame_count, f"validation.{probe_name}.nb_read_frames")

    expected_outputs = (
        target / f"CameraRecord_{expected_session}.mp4",
        target / "source_stereo" / f"CameraRecord_{expected_session}_stereo.mp4",
        target / "clip_manifest.json",
        target / "preprocess" / "pico_humanego_manifest.json",
    )
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file()]
    if missing_outputs:
        raise BatchError(f"atomic target missing required outputs: {missing_outputs}")
    all_data = target / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise BatchError(f"atomic target missing all_data: {all_data}")

    snapshot = result.get("source_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise BatchError("atomic receipt has no source_snapshot")
    for relative, expected in snapshot.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise BatchError("malformed source_snapshot entry")
        path = source / relative
        if not path.is_file():
            raise BatchError(f"source_snapshot file missing: {path}")
        stat = path.stat()
        _require_equal(stat.st_size, expected.get("bytes"), f"source_snapshot.{relative}.bytes")
        _require_equal(stat.st_mtime_ns, expected.get("mtime_ns"), f"source_snapshot.{relative}.mtime_ns")
        _require_equal(sha256(path), expected.get("sha256"), f"source_snapshot.{relative}.sha256")


def validate_existing(
    target: Path, source: Path
) -> tuple[dict[str, Any], str] | None:
    result_path = target / "CONVERSION_RESULT.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS_FORMAT_COMPATIBLE":
        return None
    contract = result.get("acquisition_preflight")
    if isinstance(contract, dict) and contract.get("source") == str(source):
        return result, "RESUMED"

    preflight = validate_acquisition(source)
    validate_atomic_converter_receipt(result, target, source, preflight)
    adopted_unix = time.time()
    result["acquisition_preflight"] = {**preflight, "source": str(source)}
    result["admission_policy"] = ADMISSION_POLICY
    result["batch_adoption"] = {
        "status": "ADOPTED_ATOMIC_CONVERTER_RECEIPT",
        "adopted_unix": adopted_unix,
        "reason": (
            "converter atomically published a fully validated target before "
            "the former batch parent recorded it"
        ),
    }
    write_json(result_path, result)
    return result, "ADOPTED_ATOMIC_RECEIPT"


def validate_existing_rejection(rejected: Path, source: Path, task: str,
                                sid: str) -> dict[str, Any] | None:
    result_path = rejected / "RESULT.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = {
        "task": task,
        "session_id": sid,
        "source": str(source),
        "status": "REJECTED_GENUINE_PREFLIGHT_OR_CONVERSION_FAILURE",
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return None
    return result


def process_one(item: dict[str, str], converter: str, target_root: str) -> dict[str, Any]:
    task, sid = item["task"], item["session_id"]
    source = Path(item["source"])
    target = (Path(target_root) / "cleaned" / task /
              f"{TASK_PREFIX[task]}_{item['date_tag']}_{sid}")
    rejected = Path(target_root) / "rejected" / task / sid
    try:
        existing = validate_existing(target, source)
        if existing is not None:
            existing_result, disposition = existing
            return {"task": task, "session_id": sid, "status": disposition,
                    "classification": "CLEANED", "target": str(target),
                    "frames": existing_result["validation"]["frame_count"]}
        existing_rejection = validate_existing_rejection(rejected, source, task, sid)
        if existing_rejection is not None:
            return {"task": task, "session_id": sid, "status": "RESUMED",
                    "classification": "REJECTED", "target": str(rejected),
                    "error": existing_rejection.get("error")}
        if target.exists():
            raise BatchError(f"target exists without valid receipt: {target}")
        preflight = validate_acquisition(source)
        command = [
            sys.executable, converter, "--source", str(source), "--target", str(target),
            "--session-id", target.name, "--selection-convention", "legacy_tracker_left",
            "--image-domain-mode", "passthrough_scaled_source_domain",
        ]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            raise BatchError(done.stderr[-8000:] or done.stdout[-8000:])
        result_path = target / "CONVERSION_RESULT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["acquisition_preflight"] = {**preflight, "source": str(source)}
        result["admission_policy"] = ADMISSION_POLICY
        write_json(result_path, result)
        return {"task": task, "session_id": sid, "status": "COMMITTED",
                "classification": "CLEANED", "target": str(target),
                "frames": preflight["frame_count"],
                "content_status": result.get("content_status")}
    except Exception as exc:
        if rejected.exists():
            raise
        receipt = {
            "schema_version": "handle-acquisition-aligned-rejection-v2",
            "task": task, "session_id": sid, "source": str(source),
            "status": "REJECTED_GENUINE_PREFLIGHT_OR_CONVERSION_FAILURE",
            "error": repr(exc), "traceback": traceback.format_exc()[-12000:],
        }
        write_json(rejected / "RESULT.json", receipt)
        return {"task": task, "session_id": sid, "status": "COMMITTED",
                "classification": "REJECTED", "target": str(rejected),
                "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-source", action="append", required=True)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    target = args.target_root.absolute()
    target.mkdir(parents=True, exist_ok=True)
    lock_fd, lock_owner = acquire_batch_lock(target)
    terminal = target / "DATASET_RESULT.json"
    if terminal.is_file():
        prior = json.loads(terminal.read_text(encoding="utf-8"))
        if prior.get("state") == "COMMITTED":
            raise BatchError(f"immutable completed target: {target}")
    sources: dict[str, Path] = {}
    for raw in args.task_source:
        task, root = parse_task_source(raw)
        if task in sources:
            raise BatchError(f"duplicate task source: {task}")
        sources[task] = root
    items = [
        {"task": task, "session_id": sid, "source": str(root / sid),
         "date_tag": args.date_tag}
        for task, root in sorted(sources.items()) for sid in sessions(root)
    ]
    (target / "cleaned").mkdir(exist_ok=True)
    (target / "rejected").mkdir(exist_ok=True)
    converter = str(Path(__file__).resolve().parent /
                    "convert_handle_egodex_to_tracker_session.py")
    started = time.time()
    state = {
        "schema_version": "handle-acquisition-aligned-batch-state-v2",
        "dataset_id": args.dataset_id, "date_tag": args.date_tag,
        "state": "RUNNING", "session_count": len(items),
        "completed": 0, "cleaned": 0, "rejected": 0,
        "sources": {task: str(root) for task, root in sources.items()},
        "converter": converter, "pid": os.getpid(), "started_unix": started,
        "batch_lock": lock_owner,
        "results": [],
    }
    write_json(target / "STATE.json", state)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_one, item, converter, str(target)): item
                   for item in items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            state.update({
                "completed": len(results),
                "cleaned": sum(x["classification"] == "CLEANED" for x in results),
                "rejected": sum(x["classification"] == "REJECTED" for x in results),
                "last_result": result, "updated_unix": time.time(),
                "results": sorted(results, key=lambda x: (x["task"], x["session_id"])),
            })
            write_json(target / "STATE.json", state)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda x: (x["task"], x["session_id"]))
    final = {**state, "state": "COMMITTED", "completed": len(results),
             "cleaned": sum(x["classification"] == "CLEANED" for x in results),
             "rejected": sum(x["classification"] == "REJECTED" for x in results),
             "finished_unix": time.time(), "results": results}
    write_json(target / "STATE.json", final)
    write_json(terminal, final)
    print(json.dumps({key: final[key] for key in
                      ("state", "session_count", "cleaned", "rejected")},
                     ensure_ascii=False, indent=2))
    os.close(lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
