"""Re-render saved 0916 Sensor arrays with honest, fixed metric plot scales."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_convergence_sensor_review import (
    CASES, _draw_clipped_skeleton, _load,
)
from chaoyang.pipeline.v5_sensor import project

TASK = "human_to_robot_sensor_display_correction_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/sensor/review_metric_v2"
WIDTH, HEIGHT = 640, 480
BOX = (0, 0, WIDTH, HEIGHT)
COLORS = ((255, 120, 20), (30, 70, 255))


def metric_world_xy(point: np.ndarray, center: np.ndarray, pixels_per_m: float,
                    box: tuple[int, int, int, int] = BOX) -> tuple[int, int]:
    x0, y0, w, h = box
    return (int(round(x0 + w / 2 + (float(point[0]) - float(center[0])) * pixels_per_m)),
            int(round(y0 + h / 2 - (float(point[2]) - float(center[1])) * pixels_per_m)))


def fixed_world_geometry(motion: dict) -> tuple[np.ndarray, float, float]:
    head = motion["T_world_head"][:, :3, 3][motion["head_valid"]]
    wrist = motion["T_world_wrist"][:, :, :3, 3][motion["wrist_world_valid"]]
    points = np.concatenate([head, wrist], axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise RuntimeError("WORLD_POINTS_NOT_VALID")
    xz = points[:, [0, 2]]
    low, high = np.min(xz, axis=0), np.max(xz, axis=0)
    center = (low + high) / 2
    span_m = max(float(np.max(high - low)) * 1.25, .5)
    pixels_per_m = .88 * min(WIDTH, HEIGHT) / span_m
    return center, span_m, pixels_per_m


def fixed_local_scale(motion: dict, robot: dict) -> float:
    human = motion["manus_local_21_m"]
    human_valid = motion["manus_local_joint_valid"] & np.isfinite(human).all(axis=-1)
    fk = robot["fk21_root_relative"]
    robot_valid = robot["valid"][:, :, None] & np.isfinite(fk).all(axis=-1)
    values = []
    if human_valid.any():
        values.append(float(np.max(np.abs(human[..., :2][human_valid]))))
    if robot_valid.any():
        values.append(float(np.max(np.abs(fk[..., :2][robot_valid]))))
    return max(*values, .12) * 1.1


def draw_world(panel: np.ndarray, motion: dict, frame: int,
               center: np.ndarray, pixels_per_m: float) -> None:
    panel[:] = (32, 32, 32)
    series = [motion["T_world_head"], motion["T_world_wrist"][:, 0],
              motion["T_world_wrist"][:, 1]]
    valid_series = [motion["head_valid"], motion["wrist_world_valid"][:, 0],
                    motion["wrist_world_valid"][:, 1]]
    for transforms, valid, name, color in zip(
            series, valid_series, ("HEAD", "L WRIST", "R WRIST"),
            ((190, 190, 190), *COLORS), strict=True):
        for i in range(1, frame + 1):
            if valid[i - 1] and valid[i]:
                a = metric_world_xy(transforms[i - 1, :3, 3], center, pixels_per_m)
                b = metric_world_xy(transforms[i, :3, 3], center, pixels_per_m)
                cv2.line(panel, a, b, color, 1, cv2.LINE_AA)
        if not valid[frame]:
            continue
        transform = transforms[frame]
        origin = metric_world_xy(transform[:3, 3], center, pixels_per_m)
        cv2.circle(panel, origin, 5, color, -1, cv2.LINE_AA)
        for axis, axis_color in enumerate(((80, 80, 255), (80, 255, 80), (255, 160, 60))):
            endpoint = transform[:3, 3] + .08 * transform[:3, axis]
            cv2.arrowedLine(panel, origin, metric_world_xy(endpoint, center, pixels_per_m),
                            axis_color, 2, cv2.LINE_AA, tipLength=.25)
        cv2.putText(panel, name, (origin[0] + 6, origin[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
    bar = max(1, int(round(.1 * pixels_per_m)))
    cv2.line(panel, (25, 445), (25 + bar, 445), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(panel, "10 cm", (25, 466), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "Tracked world X/Z, fixed origin & equal metre scale", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, "DEV tracking, not calibrated Robot world", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (160, 160, 255), 1, cv2.LINE_AA)


def draw_local(panel: np.ndarray, points: np.ndarray, valid: np.ndarray,
               scale_m: float, title: str) -> None:
    panel[:] = (32, 32, 32)
    pixels_per_m = 220.0 / scale_m
    for side, color in enumerate(COLORS):
        uv = np.empty((21, 2), dtype=np.float64)
        uv[:, 0] = 320 + points[side, :, 0] * pixels_per_m
        uv[:, 1] = 260 - points[side, :, 1] * pixels_per_m
        finite = valid[side] & np.isfinite(uv).all(axis=1)
        _draw_clipped_skeleton(panel, uv, finite, color)
    bar = max(1, int(round(.1 * pixels_per_m)))
    cv2.line(panel, (25, 445), (25 + bar, 445), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(panel, "10 cm", (25, 466), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .47,
                (240, 240, 240), 1, cv2.LINE_AA)


def run_case(session: str, root: Path) -> dict:
    prior = load_json(root / "RESULT.json")
    motion_path, robot_path = root / "HAND_MOTION_V1.npz", root / "KAI22_COMMON_BACKEND_V1.npz"
    motion, robot = _load(motion_path), _load(robot_path)
    n = int(prior["frames"])
    if (len(motion["frame_id"]) != n or robot["fk21_root_relative"].shape[:2] != (n, 2)
            or not np.array_equal(motion["frame_id"], np.arange(n))):
        raise RuntimeError(f"SENSOR_FRAME_MAP_MISMATCH:{session}")
    center, span_m, world_ppm = fixed_world_geometry(motion)
    local_scale_m = fixed_local_scale(motion, robot)
    uv, positive = project(motion["joints21_camera"], motion["camera_K"])
    valid = positive & motion["joint_valid"]
    source = Path(prior["source_video"]["path"])
    folder = DEST / session
    folder.mkdir(parents=True)
    output = folder / "SENSOR_METRIC_REVIEW_V2.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_NOT_OPEN")
    cap = cv2.VideoCapture(str(source))
    crossing_total = line_total = 0
    samples = {0, n // 2, n - 1}
    try:
        for frame in range(n):
            okay, stereo = cap.read()
            if not okay or stereo.shape[:2] != (1536, 4096):
                raise RuntimeError(f"SOURCE_DECODE:{session}:{frame}")
            original = cv2.resize(stereo[:, 2048:], (960, 540), interpolation=cv2.INTER_AREA)
            overlay = original.copy()
            uv_scaled = uv[frame] * np.asarray([960 / 1280, 540 / 960])
            for side, color in enumerate(COLORS):
                lines, crossing = _draw_clipped_skeleton(overlay, uv_scaled[side], valid[frame, side], color)
                line_total += lines
                crossing_total += crossing
            canvas = np.zeros((1080, 1920, 3), np.uint8)
            canvas[40:580, :960] = original
            canvas[40:580, 960:] = overlay
            draw_world(canvas[600:1080, :640], motion, frame, center, world_ppm)
            draw_local(canvas[600:1080, 640:1280], motion["manus_local_21_m"][frame],
                       motion["manus_local_joint_valid"][frame], local_scale_m,
                       "MANUS wrist-local 21 | fixed metres")
            draw_local(canvas[600:1080, 1280:1920], robot["fk21_root_relative"][frame],
                       robot["valid"][frame, :, None].repeat(21, axis=1), local_scale_m,
                       "Kai22 saved FK | same fixed metres")
            cv2.putText(canvas, "physical-left RGB", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 2)
            cv2.putText(canvas, "MANUS projection (valid clipped segments)", (975, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 245, 245), 2)
            time_s = (int(motion["timestamp_ns"][frame]) - int(motion["timestamp_ns"][0])) / 1e9
            cv2.putText(canvas, f"{session} frame {frame}/{n-1} capture {time_s:.3f}s | reused q/FK | OFFLINE_VISUAL",
                        (650, 596), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
            writer.write(canvas)
            if frame in samples:
                if not cv2.imwrite(str(folder / f"SAMPLE_{frame:06d}.png"), canvas):
                    raise RuntimeError(f"SAMPLE_WRITE_FAILED:{session}:{frame}")
        if cap.read()[0]:
            raise RuntimeError(f"EXTRA_SOURCE_FRAMES:{session}")
    finally:
        cap.release()
        writer.release()
    check = cv2.VideoCapture(str(output))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != n:
        raise RuntimeError(f"REVIEW_DECODE:{session}:{decoded}/{n}")
    samples_ref = [artifact_ref(folder / f"SAMPLE_{frame:06d}.png") for frame in sorted(samples)]
    result = {
        "session_id": session, "frames": n, "execution": "REUSED_ARRAYS_NEW_REVIEW",
        "structure": "PASS", "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW", "adoption": "NOT_ADOPTED",
        "source_motion": artifact_ref(motion_path), "source_robot_fk": artifact_ref(robot_path),
        "source_video": prior["source_video"], "video": {**artifact_ref(output), "decoded_frames": decoded},
        "samples": samples_ref, "sample_frame_ids": sorted(samples),
        "world_center_xz_m": [float(v) for v in center], "world_span_m": span_m,
        "world_pixels_per_m_x": world_ppm, "world_pixels_per_m_z": world_ppm,
        "local_scale_m": local_scale_m, "local_pixels_per_m_manus": 220.0 / local_scale_m,
        "local_pixels_per_m_robot": 220.0 / local_scale_m,
        "segments_drawn": line_total, "crossing_segments_drawn": crossing_total,
        "claim_limit": "Corrected display scales and clipped valid bones from saved arrays; no new solve or measured wrist alignment.",
    }
    (folder / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    cv2.setNumThreads(2)
    DEST.mkdir(parents=True)
    rows = [run_case(session, root) for session, root in CASES.items()]
    if [row["frames"] for row in rows] != [165, 179, 122]:
        raise RuntimeError("FROZEN_SENSOR_DENOMINATOR_CHANGED")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_CORRECTION_V1",
        "task_id": TASK, "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW",
        "adoption": "NOT_ADOPTED", "sessions": rows, "new_solver_invocations": 0,
        "claim_limit": "Three re-rendered Sensor reviews use corrected metric display scales only; no sensor alignment, robot quality, training or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXECUTED_DISPLAY_CORRECTION", "result": str(DEST / "RESULT.json"),
                      "sessions": len(rows), "frames": sum(row["frames"] for row in rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
