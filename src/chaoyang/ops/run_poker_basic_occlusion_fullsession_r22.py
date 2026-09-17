#!/usr/bin/env python3
"""Export a v77 Poker full z-buffer and run Object6D-gated basic Occlusion."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from chaoyang.ops.build_visible_surface_occlusion_canary_v71 import title  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
EXPORTER = ROOT / "src/chaoyang/ops/export_robot_unified_zbuffer_canary_v71.py"
COMPOSITOR = ROOT / "src/chaoyang/ops/run_basic_occlusion_fullsession_r22.py"


def digest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def checked_ref(ref: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(ref["path"]).resolve(strict=True)
    actual = digest(path)
    for field in ("bytes", "sha256"):
        if field in ref and ref[field] != actual[field]:
            raise ValueError(f"reference mismatch for {path}: {field}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def correct_poker_review(source: Path, target: Path, expected_frames: int) -> None:
    """Replace the Chips-specific legacy first-panel title in the payload video."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open payload review: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create corrected Poker review: {target}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            panel = frame[: height // 2, : width // 2]
            frame[: height // 2, : width // 2] = title(panel, f"帧{count} Raw + 单张可见扑克牌")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    if count != expected_frames:
        raise RuntimeError(f"corrected Poker review frame mismatch: {count} != {expected_frames}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-terminal", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument("--object6d-result", type=Path, required=True)
    parser.add_argument("--object-mask-result", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--clean-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    robot_path = args.robot_terminal.resolve(strict=True)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    if robot.get("session") != args.session_id or robot.get("task") != "poker":
        raise ValueError("v77 Robot terminal session/task mismatch")
    if robot.get("terminal_status") != "PASSED" or robot.get("hard_geometry_pass") is not True:
        raise ValueError("v77 Robot hard geometry is not passed")
    review_path, review = checked_ref(robot["evidence"]["review_result"])
    if review.get("session") != args.session_id or int(review["frame_count"]) <= 0:
        raise ValueError("Robot review identity/frame closure failed")
    lineage = review["lineage"]
    hawor = Path(lineage["hawor_npz"]["path"]).resolve(strict=True)
    arm = Path(lineage["arm_states"]["path"]).resolve(strict=True)
    hand = Path(lineage["hand_states"]["path"]).resolve(strict=True)
    for ref, path in ((lineage["hawor_npz"], hawor), (lineage["arm_states"], arm), (lineage["hand_states"], hand)):
        if digest(path)["sha256"] != ref["sha256"]:
            raise ValueError(f"Robot lineage SHA mismatch: {path}")

    object6d_path = args.object6d_result.resolve(strict=True)
    object6d = json.loads(object6d_path.read_text(encoding="utf-8"))
    if object6d.get("session_id") != args.session_id or object6d.get("task") != "poker":
        raise ValueError("Object6D identity mismatch")
    if object6d.get("unobserved_pose_policy") != "KEEP_INVALID" or object6d.get("downstream_authorized") is not True:
        raise ValueError("Poker Object6D must be direct observed-only KEEP_INVALID")
    if int(object6d.get("physical_instance_count", -1)) != 1 or object6d.get("multi_instance_union_used") is not False:
        raise ValueError("Poker requires exactly one physical card identity without union")

    object_mask_result_path = args.object_mask_result.resolve(strict=True)
    object_mask = json.loads(object_mask_result_path.read_text(encoding="utf-8"))
    if object_mask.get("session") != args.session_id or object_mask.get("downstream_authorized") is not True:
        raise ValueError("task-object Mask is not authorized for this session")
    if int(object_mask.get("instance_count", -1)) != 1 or sum(object_mask.get("observed_counts", {}).values()) <= 0:
        raise ValueError("no direct visible Poker appearance evidence")
    mask_manifest = Path(object_mask["artifacts"]["manifest"]["path"]).resolve(strict=True)

    zbuffer_dir = output / "robot_unified_zbuffer_fullsession"
    run([
        sys.executable, str(EXPORTER), "--session-id", args.session_id,
        "--hawor", str(hawor), "--arm-states", str(arm), "--hand-states", str(hand),
        "--frame-count", str(review["frame_count"]), "--output-dir", str(zbuffer_dir),
    ], output / "ZBUFFER_EXPORT.log")
    zbuffer_result = zbuffer_dir / "RESULT.json"
    zbuffer = json.loads(zbuffer_result.read_text(encoding="utf-8"))
    if zbuffer.get("frame_ids") != list(range(int(review["frame_count"]))):
        raise ValueError("full-session Robot z-buffer is not contiguous")

    payload = output / "payload"
    run([
        sys.executable, str(COMPOSITOR), "--session-id", args.session_id,
        "--robot-zbuffer-result", str(zbuffer_result), "--depth-result", str(args.depth_result),
        "--object-mask-manifest", str(mask_manifest), "--raw-video", str(args.raw_video),
        "--clean-result", str(args.clean_result), "--output-dir", str(payload),
    ], output / "COMPOSITOR.log")
    payload_result_path = payload / "RESULT.json"
    payload_result = json.loads(payload_result_path.read_text(encoding="utf-8"))
    if payload_result.get("terminal_status") != "PASSED":
        raise RuntimeError("basic Occlusion payload did not pass")
    payload_metrics_path = payload / "METRICS.json"
    payload_metrics = json.loads(payload_metrics_path.read_text(encoding="utf-8"))
    payload_review = Path(payload_result["review_video"]["path"]).resolve(strict=True)
    review_video = output / f"{args.session_id}_单实例牌面_Robot基础几何遮挡_全片.mp4"
    correct_poker_review(payload_review, review_video, int(review["frame_count"]))

    metrics = {
        "schema_version": "R22_POKER_BASIC_OCCLUSION_METRICS_V1",
        "session_id": args.session_id,
        "frame_count": int(review["frame_count"]),
        "object6d_gate": "PASSED_ONE_CARD_DIRECT_OBSERVED_ONLY_KEEP_INVALID",
        "object_appearance": "CURRENT_RAW_VISIBLE_PIXELS_ONLY",
        "hidden_object_appearance": "UNAVAILABLE",
        "contact_state": "UNKNOWN",
        "contact_narrow_band_policy": "UNKNOWN_NO_ATTACHMENT_INFERENCE",
        "known_decision_coverage": payload_metrics["known_decision_coverage"],
        "unknown_pixel_ratio": payload_metrics["unknown_pixel_ratio"],
        "protected_retention": payload_metrics["protected_retention"],
        "M_composite_outside_write_violations": payload_metrics["M_composite_outside_write_violations"],
        "composite_pixel_source_counts": payload_metrics["composite_pixel_source_counts"],
        "accuracy_reported": False,
        "training_eligible": False,
    }
    write_json(output / "METRICS.json", metrics)
    (output / "DECISION.md").write_text(
        "# Poker基础几何Occlusion决定\n\n"
        "同会话Clean、v77 Robot硬几何、FoundationStereo Depth、observed-only Object6D和"
        "Raw可见牌面像素已闭合，因此执行基础几何z-buffer。没有同会话CONTACT-10，接触窄带"
        "保持UNKNOWN；隐藏牌面没有合法外观来源，不使用Clean桌面或Attachment补全。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "R22_POKER_BASIC_OCCLUSION_NEXT_V1",
        "status": "PASSED",
        "next_task_id": "CONTACT10_SAME_SESSION_THEN_CONTACT_AWARE_REFINEMENT",
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "R22_POKER_BASIC_OCCLUSION_RECEIPT_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "robot_terminal": digest(robot_path), "robot_review": digest(review_path),
        "depth_result": digest(args.depth_result), "object6d_result": digest(object6d_path),
        "object_mask_result": digest(object_mask_result_path), "clean_result": digest(args.clean_result),
        "raw_video": digest(args.raw_video), "zbuffer_result": digest(zbuffer_result),
        "payload_result": digest(payload_result_path), "producer": digest(Path(__file__)),
        "exporter": digest(EXPORTER), "compositor": digest(COMPOSITOR),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "authority_promoted": False,
    })
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "R22_POKER_BASIC_OCCLUSION_ARTIFACTS_V1",
        "artifacts": [
            digest(zbuffer_result), digest(payload_result_path), digest(payload_metrics_path), digest(payload_review),
            digest(review_video),
            digest(output / "METRICS.json"), digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
        ],
    })
    result = {
        "schema_version": "R22_POKER_BASIC_OCCLUSION_RESULT_V1",
        "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_POKER_BASIC_GEOMETRIC_OCCLUSION",
        "session_id": args.session_id,
        "frame_count": int(review["frame_count"]),
        "review_video": digest(review_video),
        "contact_aware_refinement": "BLOCKED_PREREQ_SAME_SESSION_CONTACT10",
        "authority_promoted": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "claim_limit": "Full-session Poker visible-surface diagnostic; no hidden-card completion, Contact, accuracy, training, control or physical authority.",
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
