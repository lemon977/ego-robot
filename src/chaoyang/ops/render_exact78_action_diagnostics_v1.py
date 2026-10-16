#!/usr/bin/env python3
"""Render P1 raw-overlay and independent whole-Robot action diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared
from chaoyang.pipeline import exact78_robot_raster_v1 as raster
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


WIDTH, HEIGHT = 640, 480
FLANGE = REPO_ROOT / "assets/robot/V2相机固定法兰PRO版本.STL"


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def encoder(path: Path, fps: float) -> subprocess.Popen:
    return subprocess.Popen([
        "/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{fps:.9f}", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
    ], stdin=subprocess.PIPE, cwd=REPO_ROOT)


def close(worker: subprocess.Popen) -> None:
    assert worker.stdin is not None
    worker.stdin.close()
    if worker.wait(timeout=180):
        raise RuntimeError("VIDEO_ENCODE_FAILED")


def decode(path: Path, frames: int) -> bool:
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, value = capture.read()
            if not ok:
                break
            if value.shape[:2] != (HEIGHT, WIDTH):
                return False
            count += 1
    finally:
        capture.release()
    return count == frames


def banner(image: np.ndarray, first: str, second: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (WIDTH, 64), (0, 0, 0), -1)
    cv2.putText(result, first, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .53,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, second, (8, 53), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (0, 230, 255), 1, cv2.LINE_AA)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--hand", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    output.mkdir(parents=True)
    source = args.source_video.resolve(strict=True)
    hawor = arrays(args.hawor)
    arm = arrays(args.arm)
    hand = arrays(args.hand)
    frame_count = len(hawor["c2w"])
    if arm["q_arm"].shape != (frame_count, 2, 7) or hand["q_hand"].shape != (frame_count, 2, 22):
        raise RuntimeError("STATE_SHAPE_DRIFT")
    if not np.isfinite(arm["q_arm"]).all():
        raise RuntimeError("ARM_VISUAL_STATE_NONFINITE")
    q_hand = np.asarray(hand["q_hand"], dtype=np.float64)
    q_hand = np.nan_to_num(q_hand, nan=0.0)
    world_base = np.asarray(arm["T_world_base"], dtype=np.float64)
    fps = float(np.asarray(hawor["fps"]).reshape(-1)[0])
    provenance = np.asarray(hawor["provenance"])
    arm_status = np.asarray(arm["per_frame_status"])
    overlay_path = output / f"{args.session_id}_RAW_ROBOT_OVERLAY.mp4"
    external_path = output / f"{args.session_id}_INDEPENDENT_WHOLE_ROBOT.mp4"
    overlay_writer = encoder(overlay_path, fps)
    external_writer = encoder(external_path, fps)
    capture = cv2.VideoCapture(str(source))
    assets = load_pinned_robot_assets(REPO_ROOT)
    flange = raster.naturalv2_local_triangles(FLANGE)
    cache: dict = {}
    global_camera = shared.same_direction_global_camera(np.linalg.inv(hawor["c2w"][0]) @ world_base)
    global_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    rows = []
    try:
        for frame in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"SOURCE_VIDEO_ENDED:{frame}")
            raw = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            intrinsic = np.asarray(hawor["intrinsics"][frame], dtype=np.float64).copy()
            intrinsic[0] *= WIDTH / 1280.0
            intrinsic[1] *= HEIGHT / 960.0
            camera = np.linalg.inv(hawor["c2w"][frame]) @ world_base
            robot, labels, buffers = raster.render_robot(
                raster, assets, arm["q_arm"][frame], q_hand[frame], arm["T_tool_hand_root"],
                camera, intrinsic, cache, flange, complete_robot=False,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            alpha = labels >= 0
            overlay = raw.copy()
            overlay[alpha] = np.clip(.18 * overlay[alpha].astype(np.float32)
                                     + .82 * robot[alpha].astype(np.float32), 0, 255).astype(np.uint8)
            overlay = banner(
                overlay, f"{args.session_id} f={frame:04d} RAW + ROBOT",
                f"L {provenance[0,frame]}/{arm_status[0,frame]} | R {provenance[1,frame]}/{arm_status[1,frame]}")
            ext_rgb, ext_labels, ext_buffers = raster.render_robot(
                raster, assets, arm["q_arm"][frame], q_hand[frame], arm["T_tool_hand_root"],
                global_camera, global_k, cache, flange, complete_robot=True,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            ext_alpha = ext_labels >= 0
            external = np.full_like(ext_rgb, 242)
            external[ext_alpha] = ext_rgb[ext_alpha]
            external = banner(external, f"{args.session_id} f={frame:04d} WHOLE ROBOT",
                              "INDEPENDENT VIEW | VISUAL ONLY | NOT CONTROL")
            assert overlay_writer.stdin is not None and external_writer.stdin is not None
            overlay_writer.stdin.write(overlay.tobytes())
            external_writer.stdin.write(external.tobytes())
            rows.append({"frame": frame, "first_person_robot_pixels": int(alpha.sum()),
                         "external_robot_pixels": int(ext_alpha.sum()),
                         "left_provenance": str(provenance[0, frame]),
                         "right_provenance": str(provenance[1, frame]),
                         "left_arm_status": str(arm_status[0, frame]),
                         "right_arm_status": str(arm_status[1, frame])})
    finally:
        capture.release()
        close(overlay_writer)
        close(external_writer)
    atomic_json(output / "FRAME_MANIFEST.json", {"session_id": args.session_id, "rows": rows})
    hard = {"frame_count": len(rows) == frame_count,
            "raw_overlay_full_decode": decode(overlay_path, frame_count),
            "external_full_decode": decode(external_path, frame_count),
            "external_robot_present_all_frames": all(row["external_robot_pixels"] > 0 for row in rows)}
    result = {"schema_version": "exact78-action-diagnostics-v1",
              "status": "COMPLETE_PENDING_VISUAL_REVIEW" if all(hard.values()) else "FAILED_STRUCTURAL",
              "session_id": args.session_id, "hard_gates": hard,
              "inputs": {"source_video": artifact_ref(source), "hawor": artifact_ref(args.hawor),
                         "arm": artifact_ref(args.arm), "hand": artifact_ref(args.hand),
                         "flange_visual_proxy": artifact_ref(FLANGE)},
              "metrics": {"frame_count": len(rows),
                          "first_person_robot_present_frames": sum(row["first_person_robot_pixels"] > 0 for row in rows),
                          "external_robot_present_frames": sum(row["external_robot_pixels"] > 0 for row in rows),
                          "wall_seconds": time.monotonic() - started},
              "outputs": {"raw_robot_overlay": artifact_ref(overlay_path),
                          "independent_whole_robot": artifact_ref(external_path),
                          "frame_manifest": artifact_ref(output / "FRAME_MANIFEST.json")},
              "visual_qualified": False, "human_review": "PENDING", "training_eligible": False,
              "claim_limit": "Raw-overlay and independent-view motion diagnostics only; no Clean, occlusion, contact, control or physical claim."}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "session": args.session_id,
                      "metrics": result["metrics"]}, ensure_ascii=False))
    return 0 if all(hard.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
