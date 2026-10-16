"""Freeze the 0916 Robot100 V8.1 engineering baseline.

This finalizer intentionally performs a logical archive.  Historical result
directories stay in place because V8.1 receipts refer to immutable predecessor
paths.  They receive explicit SUPERSEDED markers and are excluded from the
single machine-readable CURRENT pointer.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


ATTEMPT = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001"
CURRENT = ATTEMPT / "corrections/anatomical_side_identity_rear_view_v8_1"
SELECTION = ATTEMPT / "SELECTION.json"
CURRENT_POINTER = ATTEMPT / "CURRENT_0916_ROBOT100_BASELINE.json"
ARCHIVE = ATTEMPT / "archive/0916_robot100_pre_v8_1"
HISTORICAL = (
    "inward_forearm_reference_v2",
    "inward_forearm_reference_v3",
    "inward_forearm_reference_v4",
    "inward_forearm_reference_v5",
    "inward_forearm_posture_v6",
    "inward_natural_posture_v7",
    "inward_natural_posture_v7_1",
    "anatomical_side_identity_v8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_ref(ref: dict, allowed_root: Path) -> tuple[Path, int]:
    path = Path(ref["path"]).resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if path != root and root not in path.parents:
        raise RuntimeError(f"ARTIFACT_PATH_ESCAPE:{path}")
    stat = path.stat()
    if stat.st_size != int(ref["bytes"]):
        raise RuntimeError(f"ARTIFACT_BYTES_DRIFT:{path}")
    if sha256(path) != ref["sha256"]:
        raise RuntimeError(f"ARTIFACT_SHA_DRIFT:{path}")
    return path, stat.st_size


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> int:
    selection = load_json(SELECTION)
    session_ids = [row["session_id"] for row in selection["rows"]]
    if len(session_ids) != 100 or len(set(session_ids)) != 100:
        raise RuntimeError("SELECTION_NOT_EXACTLY_100_UNIQUE_SESSIONS")
    if any("0916" not in session_id for session_id in session_ids):
        raise RuntimeError("NON_0916_SESSION_IN_SELECTION")

    root_result = load_json(CURRENT / "RESULT.json")
    if root_result.get("sessions") != session_ids:
        raise RuntimeError("ROOT_RESULT_SESSION_ORDER_OR_COMPLETENESS_DRIFT")

    total_bytes = 0
    maximum_step = 0.0
    projected = [0, 0]
    valid = [0, 0]
    tracking_pass = [0, 0]
    session_results: list[dict] = []
    for session_id in session_ids:
        result_path = CURRENT / "canaries" / session_id / "RESULT.json"
        result = load_json(result_path)
        if result.get("session_id") != session_id:
            raise RuntimeError(f"SESSION_ID_DRIFT:{session_id}")
        if result.get("status") != "CANARY_COMPLETE_PENDING_HUMAN_REVIEW":
            raise RuntimeError(f"SESSION_STATUS_DRIFT:{session_id}")
        side = result.get("side_mapping", {})
        if side.get("corrected_human_to_physical") != [0, 1]:
            raise RuntimeError(f"SIDE_MAPPING_DRIFT:{session_id}")
        camera = result.get("review_camera", {})
        if camera.get("screen_left_right_matches_anatomical_left_right") is not True:
            raise RuntimeError(f"REVIEW_CAMERA_DRIFT:{session_id}")
        review = result.get("review", {})
        if review.get("decoded_frames") != review.get("frames"):
            raise RuntimeError(f"VIDEO_DECODE_INCOMPLETE:{session_id}")
        _, motion_bytes = verify_ref(result["motion"], CURRENT)
        _, video_bytes = verify_ref(result["video"], CURRENT)
        total_bytes += motion_bytes + video_bytes
        maximum_step = max(maximum_step, float(result["maximum_display_joint_step_rad"]))
        for index, value in enumerate(result["posture_projected_left_right"]):
            projected[index] += int(value)
        for index, value in enumerate(result["wrist_valid_left_right"]):
            valid[index] += int(value)
        for index, value in enumerate(result["raw_tracking_pass_left_right"]):
            tracking_pass[index] += int(value)
        session_results.append(artifact_ref(result_path))

    generated = datetime.now(timezone.utc).isoformat()
    archive_rows = []
    for name in HISTORICAL:
        directory = ATTEMPT / "corrections" / name
        if not directory.is_dir():
            raise RuntimeError(f"HISTORICAL_DIRECTORY_MISSING:{directory}")
        marker = {
            "schema_version": "CHAOYANG_SUPERSEDED_0916_ROBOT100_V1",
            "status": "SUPERSEDED_NOT_CURRENT",
            "baseline_scope": "0916_ROBOT100_ONLY",
            "historical_path": str(directory.resolve()),
            "successor_current_pointer": str(CURRENT_POINTER.resolve()),
            "successor_baseline_id": "0916_ROBOT100_ANATOMICAL_IDENTITY_REAR_VIEW_V8_1",
            "reason": "Excluded from default execution to prevent cross-baseline or stale side-mapping use.",
            "physical_move": False,
            "physical_move_reason": "Preserve immutable predecessor paths referenced by V8.1 receipts.",
            "generated_at": generated,
        }
        atomic_json(directory / "SUPERSEDED_BY_V8_1.json", marker)
        archive_rows.append(marker)

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive_index = {
        "schema_version": "CHAOYANG_0916_ROBOT100_LOGICAL_ARCHIVE_V1",
        "status": "ARCHIVED_EXCLUDED_FROM_CURRENT",
        "scope": "0916 Robot100 only; excludes 0902 Exact78 and all other AI baselines",
        "current_pointer": str(CURRENT_POINTER.resolve()),
        "entries": archive_rows,
        "generated_at": generated,
    }
    atomic_json(ARCHIVE / "INDEX.json", archive_index)

    pointer = {
        "schema_version": "CHAOYANG_CURRENT_0916_ROBOT100_BASELINE_V1",
        "status": "CURRENT_ENGINEERING_BASELINE_WITH_KNOWN_TRAJECTORY_LIMITATIONS",
        "baseline_selected": True,
        "algorithm_quality_adopted": False,
        "baseline_id": "0916_ROBOT100_ANATOMICAL_IDENTITY_REAR_VIEW_V8_1",
        "scope": {
            "included": ["0916 Robot100 offline visualization", "100 fixed 0916 sessions"],
            "excluded": ["0902 Exact78", "inpainting", "other AI tasks", "physical safety certification"],
        },
        "code_commit": git_commit(),
        "entrypoint": "python -m chaoyang.ops.run_robot100_anatomical_identity_v8",
        "result_root": str(CURRENT.resolve()),
        "batch_result": artifact_ref(CURRENT / "RESULT.json"),
        "session_count": len(session_ids),
        "sessions": session_ids,
        "session_results": session_results,
        "mapping": {
            "human_to_physical": [0, 1],
            "review_view": "WEARER_ALIGNED_REAR_VIEW",
            "screen_left_right_matches_anatomical_left_right": True,
        },
        "verification": {
            "motion_and_video_artifacts_rehashed": 200,
            "verified_bytes": total_bytes,
            "all_videos_fully_decoded_by_session_receipts": True,
            "maximum_display_joint_step_rad": maximum_step,
            "posture_projected_frames_left_right": projected,
            "wrist_valid_frames_left_right": valid,
            "raw_tracking_pass_frames_left_right": tracking_pass,
        },
        "human_review": {
            "status": "ACCEPTED_AS_CURRENT_BASELINE_WITH_LIMITATIONS",
            "evidence": "User reviewed play_cards_0916_009 V8.1 and reported it looked acceptable while noting jumps and trajectory issues.",
        },
        "known_limitations": [
            "Visible joint jumps remain in some sequences.",
            "Some action trajectories do not yet match human motion closely.",
            "This selection prevents left/right baseline confusion; it does not certify physical motion quality.",
        ],
        "archive_index": artifact_ref(ARCHIVE / "INDEX.json"),
        "generated_at": generated,
    }
    atomic_json(CURRENT_POINTER, pointer)
    print(json.dumps({"status": pointer["status"], "sessions": 100}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
