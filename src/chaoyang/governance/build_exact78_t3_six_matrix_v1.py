"""Read-only Exact78 T3 six-session snapshot; never consume unsealed partials.

This module is a CPU report builder, not a ``chaoyang run`` algorithm operation.
It prints JSON only. A later bounded T3 packet may seal a chosen stdout snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
FREEZE_REL = Path("_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001/INPUT_FREEZE.json")
FRAME_MAP_REL = Path("_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001/FRAME_TIME_MAP.json")
STATE_REL = Path("docs/governance/LONG_HORIZON_TASK_STATE.json")
FREEZE_SHA = "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88"
FRAME_MAP_SHA = "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a"
T2_PHASE = "EXACT78_RAW_TO_ROBOT_T2_TWO_SOURCE"
ACTIVE = frozenset({"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"})
TERMINAL = frozenset({"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE", "FAILED_QUALITY_FINAL"})
STAGES = ("clean", "depth", "object6d", "occlusion", "composite")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def _verified(root: Path, ref: Any, *, required_parent: Path | None = None) -> tuple[Path, dict[str, Any]]:
    if not isinstance(ref, dict) or not all(key in ref for key in ("path", "bytes", "sha256")):
        raise ValueError("ARTIFACT_REF_REQUIRED")
    path = Path(ref["path"])
    path = path.resolve(strict=True)
    root = root.resolve(strict=True)
    if root not in path.parents or (required_parent is not None and
                                     required_parent.resolve() not in path.parents):
        raise ValueError("ARTIFACT_OUTSIDE_EXPECTED_ATTEMPT")
    if _ref(path) != ref:
        raise ValueError("ARTIFACT_SHA_DRIFT")
    return path, _json(path)


def _empty_stage_status(value: str = "NOT_EVALUATED") -> dict[str, str]:
    return {name: value for name in STAGES}


def _raw_row(session_id: str, selected: dict[str, Any], mapped: dict[str, Any]) -> dict[str, Any]:
    if mapped.get("role") != selected.get("role") or mapped.get("decoded_frames", 0) <= 0:
        raise ValueError("T0_SELECTION_FRAME_MAP_DRIFT")
    video = mapped["source_video"]
    if video not in selected["files"].values():
        raise ValueError("T0_SOURCE_VIDEO_IDENTITY_DRIFT")
    rows = mapped.get("frame_rows")
    if not isinstance(rows, list) or len(rows) != mapped["decoded_frames"]:
        raise ValueError("T0_FRAME_ROWS_INCOMPLETE")
    physical_eye = rows[0].get("sbs_physical_eye")
    return {
        "session_id": session_id,
        "selection_role": selected["role"],
        "split_claim": selected.get("split_claim"),
        "source_group": selected["source_group"],
        "raw_source": video,
        "source_path": selected["source_path"],
        "physical_eye": physical_eye,
        "frames": mapped["decoded_frames"],
        "nominal_fps": mapped.get("nominal_fps_from_capture_metadata"),
        "time_mode": mapped["time_mode"],
        "capture_to_decode_alignment": mapped["capture_to_decode_alignment"],
        "head_pose_clock_domain": mapped["head_pose_clock_domain"],
        "execution": {"status": "NOT_EVALUATED"},
        "hawor": {"status": "NOT_EVALUATED"},
        "robot": {"status": "NOT_EVALUATED"},
        "stage_qualification": _empty_stage_status(),
        "visual_qualified": None,
        "product_qualified": None,
        "training_eligible": None,
        "user_review": "NOT_EVALUATED",
        "evidence": {},
    }


def _fill_terminal(root: Path, row: dict[str, Any], task: dict[str, Any]) -> None:
    task_id = task["task_id"]
    attempt = root / "_run/current" / task_id / "attempts/attempt_0001"
    receipt_ref = task.get("terminal_receipt")
    result_ref = task.get("result")
    if not receipt_ref or not result_ref:
        raise ValueError("TERMINAL_REFS_MISSING")
    receipt_path, receipt = _verified(root, receipt_ref, required_parent=root / "tasks/receipts")
    if receipt_path != root / "tasks/receipts" / receipt_path.name:
        raise ValueError("TERMINAL_RECEIPT_LOCATION_DRIFT")
    result_path, result = _verified(root, result_ref, required_parent=attempt)
    if result_path != attempt / "RESULT.json":
        raise ValueError("TERMINAL_RESULT_LOCATION_DRIFT")
    if (receipt.get("result") != result_ref or receipt.get("task_id") != task_id
            or receipt.get("status") != task["status"] or result.get("task_id") != task_id
            or result.get("status") != task["status"] or result.get("source_session") != row["session_id"]):
        raise ValueError("TERMINAL_IDENTITY_DRIFT")
    row["evidence"].update(terminal_receipt=receipt_ref, terminal_result=result_ref)
    row["execution"] = {"status": result["status"], "complete": bool(result.get("execution_complete")),
                        "algorithm_attempts": result.get("algorithm_attempts"),
                        "failure_code": result.get("failure_code")}
    row["visual_qualified"] = result.get("visual_qualified")
    row["product_qualified"] = result.get("product_qualified")
    row["training_eligible"] = result.get("training_eligible")
    row["user_review"] = result.get("user_review", "NOT_EVALUATED")
    if result["status"] != "PASSED":
        return  # A failed terminal result does not authorize reading partial children.
    if result.get("execution_complete") is not True or result.get("source_frames_expected") != row["frames"]:
        raise ValueError("PASSED_RESULT_INCOMPLETE")
    candidate_path, candidate = _verified(root, result.get("child_result"), required_parent=attempt)
    if candidate_path != attempt / "candidate/RESULT.json":
        raise ValueError("CANDIDATE_RESULT_LOCATION_DRIFT")
    if (candidate.get("session_id") != row["session_id"] or candidate.get("execution_complete") is not True
            or candidate.get("status") != "CAMERA_DIAGNOSTIC_STAGES_COMPLETE"
            or candidate.get("stage_status", {}).get("hawor") != "SEALED_CAMERA_PREDICTIONS"
            or candidate.get("stage_status", {}).get("robot") != "SEALED_CAMERA_DIAGNOSTIC"
            or result.get("stage_status") != candidate.get("stage_status")):
        raise ValueError("CANDIDATE_RESULT_NOT_SEALED")
    stages = candidate.get("stages", {})
    hawor_seal_path, hawor_seal = _verified(root, stages.get("hawor_seal"), required_parent=attempt)
    robot_seal_path, robot_seal = _verified(root, stages.get("robot_seal"), required_parent=attempt)
    hawor_path, hawor = _verified(root, stages.get("hawor"), required_parent=attempt)
    robot_path, robot = _verified(root, stages.get("robot"), required_parent=attempt)
    if (hawor_seal_path != attempt / "candidate/HAWOR_STAGE_SEAL.json"
            or robot_seal_path != attempt / "candidate/ROBOT_STAGE_SEAL.json"
            or hawor_path != attempt / "candidate/hawor/RESULT.json"
            or robot_path != attempt / "candidate/robot/RESULT.json"
            or hawor_seal.get("result") != stages["hawor"]
            or robot_seal.get("result") != stages["robot"]
            or robot_seal.get("input_hawor_seal") != stages["hawor_seal"]
            or any(value.get("session_id") != row["session_id"] for value in
                   (hawor_seal, robot_seal, hawor, robot))):
        raise ValueError("STAGE_SEAL_CHAIN_DRIFT")
    if (hawor.get("source_frames_decoded") != row["frames"] or
            robot.get("frame_count") != row["frames"] or
            hawor.get("model_input_frames", -1) > row["frames"] or
            any(value > row["frames"] or value < 0 for value in
                hawor.get("model_prediction_frames", {}).values())):
        raise ValueError("STAGE_FRAME_COUNT_DRIFT")
    row["evidence"].update(candidate_result=result["child_result"],
                           hawor_result=stages["hawor"], hawor_seal=stages["hawor_seal"],
                           robot_result=stages["robot"], robot_seal=stages["robot_seal"])
    row["hawor"] = {
        "status": hawor["status"], "model_input_frames": hawor.get("model_input_frames"),
        "model_prediction_frames": hawor.get("model_prediction_frames"),
        "model_infill_frames": hawor.get("model_infill_frames"),
        "deterministic_fill_frames": hawor.get("deterministic_fill_frames"),
        "input_audit": hawor.get("input_audit"),
        "frame_provenance": hawor.get("frame_provenance"),
        "prediction_kind": "RGB_MODEL_PREDICTION_NOT_DIRECT_3D_OBSERVATION",
    }
    stage_path, numeric = _verified(root, robot.get("stage_result"), required_parent=attempt)
    if stage_path != attempt / "candidate/robot/v1_stage_with_ffmpeg44_encoder/RESULT.json":
        raise ValueError("ROBOT_NUMERIC_RESULT_LOCATION_DRIFT")
    if (numeric.get("session_id") != row["session_id"] or
            numeric.get("frame_count") != row["frames"] or
            numeric.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"):
        raise ValueError("ROBOT_NUMERIC_RESULT_IDENTITY_DRIFT")
    row["evidence"]["robot_numeric_result"] = robot["stage_result"]
    videos = robot.get("verified_videos", {})
    expected_videos = {"ego_robot", "external_full_robot", "raw_motion_source"}
    video_claim_complete = (set(videos) == expected_videos and
                            all(value.get("decoded_frames") == row["frames"] and
                                isinstance(value.get("video"), dict) and
                                value["video"].get("sha256") for value in videos.values()) and
                            result.get("robot_video_complete") is True and
                            result.get("videos") == videos)
    row["robot"] = {
        "status": robot["status"],
        "numeric_gate_status": "CAMERA_DIAGNOSTIC_COUNTS_ONLY",
        "arm_status_counts": numeric.get("counts", {}).get("arm_status", {}),
        "hand_status_counts": numeric.get("counts", {}).get("hand_status", {}),
        "display_drawable_rows": numeric.get("counts", {}).get("display_drawable_rows"),
        "final_visible_frames": numeric.get("counts", {}).get("final_visible_frames"),
        "video_complete_sealed_claim": video_claim_complete,
        "videos": videos,
    }
    statuses = candidate.get("stage_status", {})
    row["stage_qualification"] = {name: statuses.get(name, "NOT_EVALUATED") for name in STAGES}
    # Occlusion has no separate sealed result in v3. Do not infer it from Robot video.
    row["stage_qualification"]["occlusion"] = "NOT_EVALUATED"


def build_matrix(root: Path = REPO_ROOT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    freeze_path, frame_path, state_path = (root / FREEZE_REL, root / FRAME_MAP_REL, root / STATE_REL)
    freeze_ref, frame_ref, state_ref = (_ref(freeze_path), _ref(frame_path), _ref(state_path))
    if freeze_ref["sha256"] != FREEZE_SHA or frame_ref["sha256"] != FRAME_MAP_SHA:
        raise ValueError("T0_FROZEN_INPUT_SHA_DRIFT")
    freeze, frame_map, state = _json(freeze_path), _json(frame_path), _json(state_path)
    selected = freeze["selected"]
    if len(selected) != 6 or set(selected) != set(frame_map["sessions"]):
        raise ValueError("T0_SIX_SESSION_SET_DRIFT")
    output = {
        "schema_version": "exact78-t3-six-session-cpu-snapshot-v1",
        "status": "READ_ONLY_SNAPSHOT_NOT_T3_SEAL",
        "inputs": {"t0_freeze": freeze_ref, "t0_frame_time_map": frame_ref,
                   "task_state_for_terminal_eligibility_only": state_ref,
                   "generator_code": _ref(Path(__file__).resolve())},
        "selection_rule": "unique_exact78_raw_to_robot_t2_two_source_task_per_t0_session; no latest-try selection",
        "claim_limit": "Six frozen captures only; sealed camera diagnostic is not visual/product/training qualification",
        "sessions": {},
    }
    for session_id, item in sorted(selected.items()):
        row = _raw_row(session_id, item, frame_map["sessions"][session_id])
        matches = [task for task in state.get("tasks", [])
                   if task.get("session") == session_id and task.get("phase") == T2_PHASE]
        if len(matches) > 1:
            row["execution"] = {"status": "AMBIGUOUS_TASK_SELECTION"}
        elif matches:
            task = matches[0]
            status = task.get("status")
            row["task_id"] = task.get("task_id")
            if status in ACTIVE:
                row["execution"] = {"status": "IN_PROGRESS", "task_state": status}
            elif status in TERMINAL:
                try:
                    _fill_terminal(root, row, task)
                except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                    # Partial or malformed terminal chains never contribute stage metrics.
                    row["execution"] = {"status": "BLOCKED_TERMINAL_PROOF", "reason": str(exc)}
                    row["hawor"] = {"status": "NOT_EVALUATED"}
                    row["robot"] = {"status": "NOT_EVALUATED"}
                    row["stage_qualification"] = _empty_stage_status()
                    row["visual_qualified"] = None
                    row["product_qualified"] = None
                    row["training_eligible"] = None
                    row["user_review"] = "NOT_EVALUATED"
                    row["evidence"] = {}
            else:
                row["execution"] = {"status": "NOT_EVALUATED", "task_state": status}
        output["sessions"][session_id] = row
    if _ref(state_path) != state_ref:
        raise ValueError("TASK_STATE_CHANGED_DURING_SNAPSHOT")
    counts: dict[str, int] = {}
    for row in output["sessions"].values():
        status = row["execution"]["status"]
        counts[status] = counts.get(status, 0) + 1
    output["execution_counts"] = counts
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.repo_root), sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
