#!/usr/bin/env python3
"""P1 three-session repair validation with native FPS and Arm v3."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_exact78_general_p1_action_canary_v1 as base
from chaoyang.ops import run_exact78_hawor_raw_v2 as raw_v2


def _manifest_path() -> Path:
    return Path(sys.argv[sys.argv.index("--manifest") + 1]).resolve(strict=True)


def _storyboard(video: Path, frame_ids: list[int], labels: dict[int, str], output: Path) -> dict:
    capture = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"STORYBOARD_DECODE_FAILED:{video}:{frame_id}")
            image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (319, 30), (0, 0, 0), -1)
            cv2.putText(
                image, f"f{frame_id} {labels.get(frame_id, 'sample')}", (7, 21),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            tiles.append(image)
    finally:
        capture.release()
    while len(tiles) < 12:
        tiles.append(np.zeros((240, 320, 3), dtype=np.uint8))
    canvas = np.vstack([np.hstack(tiles[row:row + 4]) for row in range(0, 12, 4)])
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"STORYBOARD_WRITE_FAILED:{output}")
    return artifact_ref(output) | {"frame_ids": frame_ids}


def enrich_problem_evidence(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_id = manifest["task_id"]
    root = base.REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    result_path = root / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_by_id = {item["session_id"]: item for item in manifest["sessions"]}
    problem_rows = []
    for session in result["sessions"]:
        session_id = session["session_id"]
        source = manifest_by_id[session_id]["source_video"]
        if session["status"] != "TERMINAL":
            problem_rows.append({
                "session_id": session_id,
                "owner": "INTEGRATION",
                "failure": session["failure"],
                "frames": list(range(min(12, int(manifest_by_id[session_id]["frame_count"])))),
                "source_video": source,
                "error_video": session["error_video"],
                "visual_review": "ERROR_VIDEO_ONLY",
            })
            continue
        arm_path = Path(session["arm"]["path"])
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        clusters: dict[str, list[int]] = {}
        labels: dict[int, str] = {}
        for item in arm["rows"]:
            status = str(item.get("static_pose_status", item["status"]))
            if status == "PASS" or status == "NEUTRAL_NO_MOTION":
                continue
            frame = int(item["frame"])
            clusters.setdefault(status, []).append(frame)
            labels.setdefault(frame, status)
        frame_count = int(arm["frame_count"])
        chosen = [0, frame_count // 2, frame_count - 1]
        for frames in clusters.values():
            chosen.extend(frames[:3])
        chosen = list(dict.fromkeys(frame for frame in chosen if 0 <= frame < frame_count))
        for frame in np.rint(np.linspace(0, frame_count - 1, 12)).astype(int).tolist():
            if frame not in chosen:
                chosen.append(frame)
            if len(chosen) == 12:
                break
        chosen = chosen[:12]
        video = Path(session["diagnostic_outputs"]["raw_robot_overlay"]["path"])
        storyboard_path = video.parent / f"{session_id}_ACTION_STORYBOARD_12.jpg"
        storyboard = _storyboard(video, chosen, labels, storyboard_path)
        session["diagnostic_outputs"]["storyboard_12"] = storyboard
        problem_rows.append({
            "session_id": session_id,
            "owner": "BASE_IK",
            "failure_clusters": {key: len(value) for key, value in clusters.items()},
            "representative_frames": {key: value[:12] for key, value in clusters.items()},
            "arrays": arm["outputs"]["states"],
            "arm_result": session["arm"],
            "hawor_result": session["hawor"],
            "source_video": source,
            "diagnostic_video": session["diagnostic_outputs"]["raw_robot_overlay"],
            "storyboard_12": storyboard,
            "visual_review": "PENDING_HUMAN_FULL_VIDEO",
        })
    problem_path = root / "PROBLEM_TABLE.json"
    atomic_json(problem_path, {
        "schema_version": "exact78-p1-per-frame-problem-table-v2",
        "rows": problem_rows,
    })
    result["sessions"] = result["sessions"]
    result["outputs"]["problem_table"] = artifact_ref(problem_path)
    result["diagnostic_storyboards"] = {
        item["session_id"]: item["diagnostic_outputs"]["storyboard_12"]
        for item in result["sessions"] if item["status"] == "TERMINAL"
    }
    atomic_json(result_path, result)


def main() -> int:
    original_cli = base.cli
    original_parameter_audit = base.parameter_audit

    def retry_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        replacements = {
            "run_exact78_hawor_raw_v1": "run_exact78_hawor_raw_v2",
            "run_exact78_hawor_recovery_v2": "run_exact78_hawor_recovery_v6",
            "run_exact78_session_base_arm_v1": "run_exact78_session_base_arm_v3",
        }
        return original_cli(replacements.get(operation, operation), *args, hawor=hawor)

    base.cli = retry_cli
    def parameter_audit(canonical_npz, canonical_result, fresh_runner,
                        cards_raw_result, destination):
        del fresh_runner
        return original_parameter_audit(
            canonical_npz, canonical_result, Path(raw_v2.__file__),
            cards_raw_result, destination,
        )

    base.parameter_audit = parameter_audit
    manifest_path = _manifest_path()
    code = base.main()
    if code == 0:
        enrich_problem_evidence(manifest_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
