#!/usr/bin/env python3
"""Append one verified full Poker008 C-grade research video to the frozen 60 rows."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
import subprocess
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json


SESSION = "play_cards_0901_008"
FRAMES = 578


def full_decode(path: Path) -> None:
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json", str(path),
    ], text=True, timeout=90))["streams"][0]
    if int(probe["nb_read_frames"]) != FRAMES or probe["width"] != 1920 or probe["height"] != 480:
        raise ValueError("full video frame/size closure failed")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                    "-f", "null", "-"], check=True, timeout=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--render-result", type=Path, required=True)
    parser.add_argument("--arm-result", type=Path, required=True)
    parser.add_argument("--collision-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("immutable matrix revision required")
    matrix = json.loads(args.base.read_text(encoding="utf-8"))
    rows = matrix["rows"]
    if len(rows) != 60 or len({(row["task"], row["session_id"]) for row in rows}) != 60:
        raise ValueError("frozen 60-row identity/denominator mismatch")
    row_matches = [row for row in rows if row["task"] == "poker" and row["session_id"] == SESSION]
    if len(row_matches) != 1 or row_matches[0]["delivery_full_video"] is not None:
        raise ValueError("Poker008 is missing or already has a primary full video")
    render = json.loads(args.render_result.read_text(encoding="utf-8"))
    arm = json.loads(args.arm_result.read_text(encoding="utf-8"))
    collision = json.loads(args.collision_result.read_text(encoding="utf-8"))
    if (render.get("session"), render.get("frame_count"), render.get("status")) != (
        SESSION, FRAMES, "HOLD_ARM_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
    ) or render.get("authority") or render.get("numeric", {}).get("arm_pass"):
        raise ValueError("render is not the expected C-marked Poker008 review")
    if (arm.get("session_id"), arm.get("frame_count"), arm.get("status")) != (
        SESSION, FRAMES, "FAILED_QUALITY_C"
    ):
        raise ValueError("arm source is not the frozen quality-C full result")
    if (collision.get("session_id"), collision.get("status"),
            collision.get("illegal_contact_count"), len(collision.get("selected_frames", []))) != (
        SESSION, "PASS_DEVELOPMENT_COLLISION_AUDIT", 0, FRAMES
    ):
        raise ValueError("full Robot digital collision audit closure failed")
    if render.get("contract", {}).get("arm", "").find("arm-only mesh") < 0:
        raise ValueError("render omitted arm-only limitation")
    video = render["outputs"]["video"]
    video_ref = artifact_ref(Path(video["path"]))
    if (video_ref["sha256"], video_ref["bytes"], video["decoded_frames"]) != (
        video["sha256"], video["bytes"], FRAMES
    ):
        raise ValueError("video bytes/SHA/frames mismatch")
    full_decode(Path(video_ref["path"]))
    row = row_matches[0]
    row.update({
        "delivery_class": "C_WATERMARKED_FULL_DIAGNOSTIC_VIDEO",
        "delivery_full_video": video_ref,
        "verified_video": video_ref,
        "verified_frame_count": FRAMES,
        "research_video_terminal_status": "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO",
        "full_video_decode": "PASS_FFMPEG_XERROR",
        "robot30_hard_geometry_pass": False,
        "causal_training_eligible": False,
        "training_eligible": False,
        "mesh_gated_arm_only_research": {
            "arm_result": artifact_ref(args.arm_result),
            "render_result": artifact_ref(args.render_result),
            "collision_result": artifact_ref(args.collision_result),
            "target_pass_rows": arm["metrics"]["target_pass_rows"],
            "target_total_rows": arm["metrics"]["target_total_rows"],
            "full_robot_collision_audited": True,
            "full_robot_digital_collision_pass": True,
            "claim_limit": "Full-length quality-C offline review only; no full Robot geometry, causal, training or physical authority.",
        },
        "claim_limit": "Primary research video is complete and C-watermarked; formal Robot30 terminal and training status remain unchanged.",
    })
    counts = {task: sum(bool(item["delivery_full_video"]) for item in rows if item["task"] == task)
              for task in ("chips", "poker")}
    if counts != {"chips": 30, "poker": 21}:
        raise ValueError(f"unexpected full-video counts: {counts}")
    matrix.update({
        "schema_version": "rc1-robot30-research-delivery-matrix-rev6",
        "supersedes": artifact_ref(args.base),
        "generator": artifact_ref(Path(__file__)),
        "normalized_watchable_counts": counts,
        "claim_limit": "Research full-video count only: Chips30/30 and Poker21/30; Poker008 is quality C, not a geometry or training success. Formal RC1 entries unchanged.",
    })
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0006.json", matrix)
    with (args.output_root / "ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0006.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        fields = ["task", "rank", "session_id", "full_video_exists", "complete_decode",
                  "hard_geometry_pass", "causal_training_eligible", "visual_grade", "video_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            candidate = item.get("delivery_full_video") or {}
            writer.writerow({
                "task": item["task"], "rank": item["rank"], "session_id": item["session_id"],
                "full_video_exists": bool(candidate),
                "complete_decode": item.get("full_video_decode", bool(candidate)),
                "hard_geometry_pass": item.get("robot30_hard_geometry_pass", False),
                "causal_training_eligible": item.get("causal_training_eligible", False),
                "visual_grade": item.get("delivery_class", ""),
                "video_path": candidate.get("path", ""),
            })
    print(json.dumps({"counts": counts, "output": str(args.output_root)}))


if __name__ == "__main__":
    main()
