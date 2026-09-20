#!/usr/bin/env python3
"""Verify all user-facing evidence in the three-stream handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


OLD_TASK_PREFIX = "/mnt/workspace/code/chaoyang/tasks/"
ARCHIVE_TASK_PREFIX = (
    "/mnt/workspace/code/chaoyang/archive/baseline-20260917-0aa69e9/"
    "content/history/tasks/"
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_declared(path: Path, declared: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing: {path}"]
    if path.stat().st_size != declared.get("bytes"):
        errors.append(f"bytes mismatch: {path}")
    if sha256(path) != declared.get("sha256"):
        errors.append(f"sha mismatch: {path}")
    return errors


def current_robot30_path(historical: str) -> Path:
    if not historical.startswith(OLD_TASK_PREFIX):
        raise RuntimeError(f"Robot30 path has unexpected historical prefix: {historical}")
    return Path(ARCHIVE_TASK_PREFIX + historical[len(OLD_TASK_PREFIX):])


def decode_video(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    probe_value: dict[str, Any] = {}
    if probe.returncode:
        errors.append(f"ffprobe failed: {path}: {probe.stderr[-400:]}")
    else:
        try:
            streams = json.loads(probe.stdout).get("streams", [])
            probe_value = streams[0] if streams else {}
        except (json.JSONDecodeError, IndexError) as exc:
            errors.append(f"ffprobe output invalid: {path}: {exc}")
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if decode.returncode:
        errors.append(f"full decode failed: {path}: {decode.stderr[-400:]}")
    return {
        "path": str(path.resolve()),
        "probe": probe_value,
        "full_decode_returncode": decode.returncode,
    }, errors


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot30-matrix", required=True, type=Path)
    parser.add_argument("--cohort-ledger", required=True, type=Path)
    parser.add_argument("--exact78-wrist-video", required=True, type=Path)
    parser.add_argument("--wiyh-video", required=True, type=Path)
    parser.add_argument("--ai2-visual-root", required=True, type=Path)
    parser.add_argument("--summary-document", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    matrix = load(args.robot30_matrix.resolve(strict=True))
    matrix_rows = matrix.get("rows", [])
    if not isinstance(matrix_rows, list) or len(matrix_rows) != 60:
        errors.append("Robot30 source matrix is not the frozen 60-row cohort")
        matrix_rows = []
    robot_videos: list[dict[str, Any]] = []
    absent_rows: list[dict[str, Any]] = []
    for row in matrix_rows:
        declared = row.get("verified_video")
        if not isinstance(declared, dict):
            absent_rows.append({
                "task": row.get("task"),
                "rank": row.get("rank"),
                "session_id": row.get("session_id"),
            })
            continue
        try:
            path = current_robot30_path(str(declared.get("path", "")))
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        errors.extend(validate_declared(path, declared))
        decoded, decode_errors = decode_video(path)
        errors.extend(decode_errors)
        expected_frames = row.get("verified_frame_count")
        actual_frames = decoded.get("probe", {}).get("nb_read_frames")
        if str(expected_frames) != str(actual_frames):
            errors.append(
                f"Robot30 frame mismatch: {row.get('session_id')}: "
                f"expected={expected_frames} actual={actual_frames}"
            )
        robot_videos.append({
            "task": row.get("task"),
            "rank": row.get("rank"),
            "session_id": row.get("session_id"),
            "hard_geometry_pass": row.get("robot30_hard_geometry_pass"),
            **decoded,
        })
    if len(robot_videos) != 47 or len(absent_rows) != 13:
        errors.append(
            f"Robot30 inventory changed: present={len(robot_videos)} absent={len(absent_rows)}"
        )

    primary_videos = []
    for label, path_arg, expected_frames in (
        ("EXACT78_WRIST_PICO_REFERENCE", args.exact78_wrist_video, 420),
        ("WIYH_SESSION101_BLIND", args.wiyh_video, 122),
    ):
        path = path_arg.resolve(strict=True)
        decoded, decode_errors = decode_video(path)
        errors.extend(decode_errors)
        frames = decoded.get("probe", {}).get("nb_read_frames")
        if str(frames) != str(expected_frames):
            errors.append(f"{label} frame mismatch: expected={expected_frames} actual={frames}")
        primary_videos.append({"label": label, **decoded, "sha256": sha256(path)})

    ai2_root = args.ai2_visual_root.resolve(strict=True)
    ai2_paths = sorted(ai2_root.rglob("*.mp4"))
    if not ai2_paths:
        errors.append("AI2 shallow visual directory has no MP4")
    ai2_videos = []
    for path in ai2_paths:
        decoded, decode_errors = decode_video(path)
        errors.extend(decode_errors)
        ai2_videos.append(decoded)

    ledger = load(args.cohort_ledger.resolve(strict=True))
    ledger_rows = ledger.get("sessions", [])
    if not isinstance(ledger_rows, list) or len(ledger_rows) != 16:
        errors.append("0915 frozen ledger is not 16 rows")
        ledger_rows = []
    source_rows = []
    for row in ledger_rows:
        declared = row.get("source_stereo")
        if not isinstance(declared, dict):
            errors.append(f"source_stereo ref absent: {row.get('session_id')}")
            continue
        path = Path(str(declared.get("path", "")))
        before = len(errors)
        errors.extend(validate_declared(path, declared))
        source_rows.append({
            "session_id": row.get("session_id"),
            "cohort_role": row.get("cohort_role"),
            "status": "PASS" if len(errors) == before else "FAIL",
            "access_mode": "INTEGRITY_ONLY_READ_NO_DECODE_NO_ALGORITHM_CONSUMPTION",
            "source_stereo": declared,
        })

    summary = args.summary_document.resolve(strict=True)
    summary_text = summary.read_text(encoding="utf-8")
    for required in (
        "旧 exact78",
        "AI1：0916 手套数据",
        "AI2：0915 裸手全链",
        "REJECTED_NO_RECOVERY",
    ):
        if required not in summary_text:
            errors.append(f"summary document missing required conclusion: {required}")

    output = {
        "schema_version": "chaoyang-three-stream-handoff-verification-v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "summary_document": artifact(summary),
        "exact78_robot30": {
            "source_matrix": artifact(args.robot30_matrix),
            "total_rows": len(matrix_rows),
            "present_full_decode_pass": len(robot_videos) - sum(
                row["full_decode_returncode"] != 0 for row in robot_videos
            ),
            "absent_in_source_matrix": len(absent_rows),
            "videos": robot_videos,
            "absent_rows": absent_rows,
        },
        "primary_videos": primary_videos,
        "ai2_shallow": {"count": len(ai2_videos), "videos": ai2_videos},
        "frozen_0915_sources": {
            "count": len(source_rows),
            "integrity_pass": sum(row["status"] == "PASS" for row in source_rows),
            "rows": source_rows,
            "claim_limit": "SHA integrity-only reads do not open sealed cohorts for algorithm evaluation.",
        },
    }
    write_new(args.output.resolve(), output)
    print(json.dumps({"status": output["status"], "errors": errors}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
