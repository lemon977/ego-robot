#!/usr/bin/env python3
"""Fail-closed CPU audit of the frozen SAM3.1 native Poker B queue.

Old B is an internal regression reference, not pixel or instance Gold.
This auditor can prepare visual review; it cannot grant Mask authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref


def _closure(reference: dict) -> None:
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError("artifact closure failed: " + "; ".join(errors))


def _decode_video(path: Path) -> None:
    run = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                          "-f", "null", "-"], capture_output=True, text=True, timeout=180)
    if run.returncode:
        raise RuntimeError(f"full video decode failed: {path}: {run.stderr[-500:]}")


def _make_contact_sheet(video_path: Path, selected: list[int], output: Path) -> None:
    wanted = set(selected)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"review video open failed: {video_path}")
    thumbnails = []
    frame_id = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in wanted:
            thumb = cv2.resize(frame, (960, 240), interpolation=cv2.INTER_AREA)
            cv2.rectangle(thumb, (0, 0), (215, 33), (0, 0, 0), -1)
            cv2.putText(thumb, f"frame {frame_id}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            thumbnails.append(thumb)
            wanted.remove(frame_id)
        frame_id += 1
    cap.release()
    if wanted:
        raise RuntimeError(f"contact sheet missing requested frames: {sorted(wanted)}")
    if len(thumbnails) % 2:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    rows = [cv2.hconcat(thumbnails[i:i+2]) for i in range(0, len(thumbnails), 2)]
    if not cv2.imwrite(str(output), cv2.vconcat(rows)):
        raise RuntimeError(f"contact sheet encode failed: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    decision_path = queue / "REGRESSION_DECISION_CONTRACT.json"
    start_path = queue / "QUEUE_START.json"
    queue_result_path = queue / "QUEUE_RESULT.json"
    if not queue_result_path.is_file():
        raise RuntimeError("queue has no terminal result; do not audit a running attempt")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    start = json.loads(start_path.read_text(encoding="utf-8"))
    queue_result = json.loads(queue_result_path.read_text(encoding="utf-8"))
    if artifact_ref(start_path)["sha256"] != decision["queue_start_sha256"]:
        raise RuntimeError("decision contract does not bind frozen queue start")
    if start.get("session_order") != [entry["session_id"] for entry in decision["sessions"]]:
        raise RuntimeError("decision contract session order changed")

    sessions = {entry["session_id"]: entry for entry in decision["sessions"]}
    audited = []
    for outcome in queue_result["outcomes"]:
        session = outcome["session_id"]
        if session not in sessions:
            raise RuntimeError(f"unfrozen session: {session}")
        if "worker_result" not in outcome:
            audited.append({"session_id": session, "status": outcome["launcher_status"],
                            "reason": "NO_WORKER_RESULT", "launcher_receipt": outcome["launcher_receipt"]})
            continue
        _closure(outcome["worker_result"])
        worker = json.loads(Path(outcome["worker_result"]["path"]).read_text(encoding="utf-8"))
        for reference in worker["inputs"].values():
            _closure(reference)
        for reference in worker["outputs"].values():
            _closure(reference)
        _closure(worker["code"])
        count = sessions[session]["frame_count"]
        if worker.get("session_id") != session or worker.get("frames") != count or len(worker.get("rows", [])) != count:
            raise RuntimeError(f"worker frame identity mismatch: {session}")
        comparisons = worker["comparisons"]
        if worker["frozen_B_observed_frames"] != len(comparisons):
            raise RuntimeError(f"B observed comparison count mismatch: {session}")
        missed = [item["frame_id"] for item in comparisons if not item["native_present"]]
        iou_rows = [item for item in comparisons if item["iou_to_frozen_B_internal_only"] is not None]
        worst = [item["frame_id"] for item in sorted(iou_rows, key=lambda item: item["iou_to_frozen_B_internal_only"])[:10]]
        areas = [row["area_pixels"] for row in worker["rows"]]
        jumps = sorted(((max(a, b) / min(a, b), i)
                        for i, (a, b) in enumerate(zip(areas[:-1], areas[1:]), start=1)
                        if a and b), reverse=True)[:5]
        selected = sorted(set(sessions[session]["fixed_uniform_review_frames"] + missed + worst +
                              [i for _, i in jumps]))
        video_ref = worker["outputs"]["review_video"]
        _decode_video(Path(video_ref["path"]))
        sheet = output / f"{session}_frozen_regression_review.png"
        _make_contact_sheet(Path(video_ref["path"]), selected, sheet)
        audited.append({
            "session_id": session, "frame_count": count,
            "status": "AUTO_GATE_PASS_VISUAL_REVIEW_REQUIRED" if not missed else "AUTO_GATE_FAIL_MISSED_FROZEN_B",
            "old_B_observed_frames": len(comparisons), "native_missed_old_B_observed_frames": missed,
            "internal_iou_median": worker["internal_iou_median"],
            "internal_iou_p05": worker["internal_iou_p05"],
            "fixed_uniform_frames": sessions[session]["fixed_uniform_review_frames"],
            "worst_internal_iou_frames": worst,
            "top_area_jumps_ratio_and_frame": jumps,
            "reviewed_frame_ids": selected,
            "contact_sheet": artifact_ref(sheet), "full_review_video": video_ref,
            "worker_result": outcome["worker_result"],
        })

    status = "HOLD_VISUAL_IDENTITY_REVIEW" if len(audited) == len(sessions) and all(
        row["status"] == "AUTO_GATE_PASS_VISUAL_REVIEW_REQUIRED" for row in audited) else (
        "BLOCKED_RESOURCE" if queue_result["status"] == "BLOCKED_RESOURCE"
        else "FAILED_OR_INCOMPLETE_REGRESSION")
    atomic_json(output / "RESULT.json", {
        "schema_version": "chaoyang-rc1-sam31-native-b-regression-cpu-audit-v1",
        "created_at": now_iso(), "status": status, "sessions": audited,
        "queue_result": artifact_ref(queue_result_path),
        "queue_start": artifact_ref(start_path),
        "decision_contract": artifact_ref(decision_path),
        "authority_promoted": False, "training_eligible": False,
        "claim_limit": "Automated presence/internal-IoU and full-video checks only. Physical card identity, face, hand contamination and pixel accuracy require independent review; old B is not Gold.",
    })
    print(json.dumps({"status": status, "audited_sessions": len(audited),
                      "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
