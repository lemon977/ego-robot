#!/usr/bin/env python3
"""Bind frozen Sensor and Local/HuRo evidence into S2 without rerunning models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes"


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def decode_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path)); count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


def write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sensor_rows = []
    for sid, folder, frames in (("097", "run_097_wave0", 165),
                                ("098", "run_098_wave1", 179),
                                ("101", "run_101_wave1", 122)):
        root = R2 / "lane3_sensor" / folder
        result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
        video = root / "SENSOR_REVIEW.mp4"
        decoded = decode_count(video)
        if result.get("execution") != "EXECUTED" or result.get("structure") != "PASS":
            raise RuntimeError(f"SENSOR_FROZEN_STRUCTURE_NOT_PASS:{sid}")
        if decoded != frames:
            raise RuntimeError(f"SENSOR_VIDEO_DECODE_MISMATCH:{sid}:{decoded}!={frames}")
        sensor_rows.append({
            "session_id": f"play_cards_0916_{sid}", "expected_frames": frames,
            "decoded_frames": decoded, "result": ref(root / "RESULT.json"), "video": ref(video),
            "execution": result["execution"], "structure": result["structure"],
            "quality": result["quality"], "adoption": result["adoption"],
            "motion_source": result.get("motion_source"),
        })
    sensor = {
        "schema_version": "HUMAN_TO_ROBOT_S2_SENSOR_REUSE_AUDIT_V1", "task_id": TASK,
        "created_at": created, "execution": "EXECUTED_REUSED_FROZEN_EVIDENCE",
        "structure": "PASS", "quality": "INCONCLUSIVE_MIXED_KINEMATIC_ONLY",
        "adoption": "CANDIDATE_ONLY", "sessions": sensor_rows,
        "new_model_invocations": 0,
        "claim_limit": "Structural/common-backend and review decode evidence only; no external wrist truth or visual acceptance.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
    }
    write_once(OUT / "sensor/reuse_audit_v1/RESULT.json", sensor)

    numeric_path = R2 / "lane4_compare/wave0_numeric_recheck/RESULT.json"
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if numeric.get("structure") != "PASS" or not numeric.get("same_target_frame_denominator_verified"):
        raise RuntimeError("COMPARE_FROZEN_DENOMINATOR_NOT_VERIFIED")
    compare_rows = []
    for short, frames in (("007", 378), ("031", 149)):
        root = R2 / f"lane4_compare/same_rejected_background_{short}_wave10"
        result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
        video = root / "LOCAL_R0_VS_HURO_SAME_REJECTED_BACKGROUND.mp4"
        decoded = decode_count(video)
        if decoded != frames or result.get("structure") != "PASS":
            raise RuntimeError(f"COMPARE_VIDEO_OR_STRUCTURE_MISMATCH:{short}:{decoded}")
        compare_rows.append({
            "session_id": result["session_id"], "decoded_frames": decoded,
            "result": ref(root / "RESULT.json"), "video": ref(video),
            "common_valid_side_frames": result["common_valid_side_frames"],
            "background_authority": "REJECTED_SCENE_DIAGNOSTIC_ONLY",
        })
    compare = {
        "schema_version": "HUMAN_TO_ROBOT_S2_LOCAL_HURO_REUSE_AUDIT_V1", "task_id": TASK,
        "created_at": created, "execution": "EXECUTED_REUSED_FROZEN_EVIDENCE",
        "structure": "PASS", "quality": "INCONCLUSIVE", "adoption": "NOT_ADOPTED",
        "numeric": ref(numeric_path), "same_target_frame_denominator_verified": True,
        "sessions": compare_rows, "winner": None, "new_solver_invocations": 0,
        "claim_limit": "Frozen same-target numeric evidence and rejected-background visuals; no winner and no final Clean comparison.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
    }
    write_once(OUT / "compare/reuse_audit_v1/RESULT.json", compare)
    print(json.dumps({"status": "PASS", "sensor_sessions": 3, "compare_sessions": 2}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
