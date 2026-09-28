"""Reuse frozen 0916 Controller/MANUS evidence as independent HandMotion.

This operation performs no fitting. It validates the sealed V3 nominal (C1)
and fixed-installation (C3) artifacts, emits the common HandMotion interface,
and renders full-session sensor reviews. C3 is a visual development estimate.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from chaoyang.pipeline.v5_sensor import SESSION_IDS, hand_motion_arrays, project, validate_pair


CANONICAL = Path("/mnt/workspace/code/chaoyang")
V3_SOURCE = CANONICAL / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai1"
V5_SENSOR = CANONICAL / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/sensor"


def _reference(path: Path) -> dict:
    source = path.resolve(strict=True)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed while hashing: {source}")
    return {"path": str(source), "bytes": after.st_size, "sha256": digest.hexdigest()}


def _check_reference(item: dict) -> Path:
    path = Path(item["path"]).resolve(strict=True)
    if _reference(path) != item:
        raise ValueError(f"sealed source reference mismatch: {path}")
    return path


def _load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_rows(source_root: Path) -> list[dict]:
    rows: list[dict] = []
    for session_id in SESSION_IDS:
        candidates = []
        for name in ("candidate1_nominal_run1", "candidate3_installation_run1"):
            result_path = source_root / name / session_id / "RESULT.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            motion = _check_reference(result["motion"])
            video = _check_reference(result["source_video"])
            if session_id not in video.parts:
                raise ValueError(f"source video belongs to another session: {video}")
            candidates.append({"motion": motion, "video": video, "result": _reference(result_path), "motion_ref": result["motion"]})
        if candidates[0]["video"] != candidates[1]["video"]:
            raise ValueError(f"C1/C3 source video drift: {session_id}")
        nominal = _load_motion(candidates[0]["motion"])
        fitted = _load_motion(candidates[1]["motion"])
        audit = validate_pair(nominal, fitted, session_id)
        rows.append({"session_id": session_id, "nominal": nominal, "fitted": fitted,
                     "video": candidates[1]["video"], "sources": candidates, "audit": audit})
    return rows


def _world_xy(point: np.ndarray, low: np.ndarray, span: float, width: int, height: int) -> tuple[int, int]:
    xy = (point[[0, 2]] - low) / span
    return int(30 + xy[0] * (width - 60)), int(height - 20 - xy[1] * (height - 50))


def _world_panel(motion: dict, frame: int, bounds: tuple[np.ndarray, float]) -> np.ndarray:
    import cv2

    width, height = 960, 235
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    low, span = bounds
    head = motion["T_world_head"][:, :3, 3]
    wrists = motion["T_world_wrist"][:, :, :3, 3]
    colors = ((150, 150, 150), (255, 120, 20), (30, 70, 255))
    for series, color in ((head, colors[0]), (wrists[:, 0], colors[1]), (wrists[:, 1], colors[2])):
        pts = np.array([_world_xy(p, low, span, width, height) for p in series[: frame + 1]], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[-1]), 5, color, -1, cv2.LINE_AA)
    cv2.putText(panel, "Tracked world X/Z: head + left/right wrist (diagnostic)", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def _pinch_panel(motion: dict, frame: int, distances: np.ndarray) -> np.ndarray:
    import cv2

    width, height = 960, 235
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "Thumb-tip / index-tip distance (MANUS local, visual aid)", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(panel, (42, 195), (width - 20, 195), (130, 130, 130), 1)
    cv2.line(panel, (42, 40), (42, 195), (130, 130, 130), 1)
    for side, color in enumerate(((255, 120, 20), (30, 70, 255))):
        pts = []
        for i, value in enumerate(distances[:, side]):
            if np.isfinite(value):
                x = int(42 + i * (width - 62) / max(1, len(distances) - 1))
                y = int(195 - np.clip(value, 0, .25) * 155 / .25)
                pts.append((x, y))
        if len(pts) > 1:
            cv2.polylines(panel, [np.asarray(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        value = distances[frame, side]
        if np.isfinite(value):
            cv2.putText(panel, f"{'L' if side == 0 else 'R'} {value * 1000:.0f} mm", (520 + side * 185, 57), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2, cv2.LINE_AA)
    cursor = int(42 + frame * (width - 62) / max(1, len(distances) - 1))
    cv2.line(panel, (cursor, 40), (cursor, 195), (210, 210, 210), 1)
    cv2.putText(panel, "0", (17, 196), cv2.FONT_HERSHEY_SIMPLEX, .42, (170, 170, 170), 1)
    cv2.putText(panel, "250 mm", (2, 47), cv2.FONT_HERSHEY_SIMPLEX, .42, (170, 170, 170), 1)
    return panel


def _render_review(video: Path, motion: dict, output: Path) -> dict:
    import av
    import cv2

    cv2.setNumThreads(2)
    n = len(motion["frame_id"])
    times = np.rint((motion["timestamp_ns"] - motion["timestamp_ns"][0]) / 1000).astype(np.int64)
    if len(np.unique(times)) != n:
        raise ValueError("microsecond presentation times collide")
    xyz = motion["joints_camera_25_m"]
    uv, positive = project(xyz, motion["camera_K"])
    valid = positive & motion["joint_camera_valid_25"]
    controller_uv, controller_positive = project(motion["T_camera_controller"][..., :3, 3], motion["camera_K"])
    head = motion["T_world_head"][:, :3, 3]
    wrists = motion["T_world_wrist"][:, :, :3, 3]
    all_world = np.concatenate((head, wrists.reshape(-1, 3)), axis=0)[:, [0, 2]]
    low = np.min(all_world, axis=0)
    high = np.max(all_world, axis=0)
    span = max(float(np.max(high - low)) * 1.15, .3)
    low = (low + high) / 2 - span / 2
    distances = np.linalg.norm(motion["manus_local_25_m"][:, :, 4] - motion["manus_local_25_m"][:, :, 9], axis=-1)
    distances[~motion["manus_hand_valid"]] = np.nan
    parent = motion["manus_parent_ids"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open source video: {video}")
    container = av.open(str(output), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
    stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
    stream.codec_context.thread_count = 2
    stream.options = {"crf": "20", "preset": "fast"}
    try:
        for frame in range(n):
            okay, stereo = cap.read()
            if not okay or stereo.shape[:2] != (1536, 4096):
                raise ValueError(f"source decode geometry/count mismatch at {frame}")
            source = cv2.resize(stereo[:, 2048:], (1280, 960), interpolation=cv2.INTER_AREA)
            overlay = source.copy()
            for side, color in enumerate(((255, 120, 20), (30, 70, 255))):
                inside = valid[frame, side] & (uv[frame, side, :, 0] >= 0) & (uv[frame, side, :, 0] < 1280) & (uv[frame, side, :, 1] >= 0) & (uv[frame, side, :, 1] < 960)
                for node in range(1, 25):
                    ancestor = int(parent[node])
                    if inside[ancestor] and inside[node]:
                        cv2.line(overlay, tuple(np.rint(uv[frame, side, ancestor]).astype(int)), tuple(np.rint(uv[frame, side, node]).astype(int)), color, 2, cv2.LINE_AA)
                for node in np.flatnonzero(inside):
                    cv2.circle(overlay, tuple(np.rint(uv[frame, side, node]).astype(int)), 4, color, -1, cv2.LINE_AA)
                if controller_positive[frame, side] and np.all((controller_uv[frame, side] >= 0) & (controller_uv[frame, side] < [1280, 960])):
                    cv2.drawMarker(overlay, tuple(np.rint(controller_uv[frame, side]).astype(int)), color, cv2.MARKER_CROSS, 18, 2)
            panel = np.zeros((1080, 1920, 3), dtype=np.uint8)
            panel[45:765, :960] = cv2.resize(source, (960, 720), interpolation=cv2.INTER_AREA)
            panel[45:765, 960:] = cv2.resize(overlay, (960, 720), interpolation=cv2.INTER_AREA)
            panel[790:1025, :960] = _world_panel(motion, frame, (low, span))
            panel[790:1025, 960:] = _pinch_panel(motion, frame, distances)
            cv2.putText(panel, "Original selected eye", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (245, 245, 245), 2)
            cv2.putText(panel, "Controller + MANUS25 C3 projection", (985, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (245, 245, 245), 2)
            cv2.putText(panel, f"{str(motion['session_id'])}  frame {frame + 1}/{n}  capture {times[frame] / 1e6:.3f}s  L=blue R=red", (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, .6, (220, 220, 220), 1)
            encoded = av.VideoFrame.from_ndarray(panel, format="bgr24")
            encoded.pts, encoded.time_base = int(times[frame]), Fraction(1, 1_000_000)
            for packet in stream.encode(encoded):
                container.mux(packet)
        if cap.read()[0]:
            raise ValueError("unexpected extra source frames")
        for packet in stream.encode():
            container.mux(packet)
    finally:
        cap.release()
        container.close()
    decoded = []
    with av.open(str(output)) as check:
        for frame in check.decode(video=0):
            if (frame.width, frame.height) != (1920, 1080):
                raise ValueError("review decode size mismatch")
            decoded.append(float(frame.pts * frame.time_base))
    if len(decoded) != n:
        raise ValueError(f"review decoded {len(decoded)} of {n} frames")
    error = float(np.max(np.abs(np.asarray(decoded) - times / 1e6)))
    if error > 2e-6:
        raise ValueError(f"review capture PTS drift: {error}")
    return {"video": _reference(output), "decoded_frames": n, "capture_pts_max_error_seconds": error,
            "visual_quality_pass": False, "review_status": "PENDING_HUMAN_VISUAL_REVIEW"}


def run(source_root: Path, output_root: Path, dry_run: bool = False) -> dict:
    if source_root.resolve(strict=True) != V3_SOURCE.resolve(strict=True):
        raise ValueError("source must be the exact sealed V3 AI1 lane")
    output = output_root.absolute()
    lane = V5_SENSOR.resolve(strict=True)
    if output.parent.resolve(strict=True) != lane or output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh direct child of the V5 sensor lane")
    rows = _source_rows(source_root)
    manifest = {"schema_version": "chaoyang-v5-sensor-input-v1", "motion_source": "controller_manus",
                "sessions": [{"session_id": row["session_id"], "audit": row["audit"],
                              "C1_nominal": row["sources"][0]["motion_ref"],
                              "C3_installation": row["sources"][1]["motion_ref"],
                              "source_video": _reference(row["video"])} for row in rows]}
    if dry_run:
        return {"status": "DRY_RUN_VALID", "input_manifest": manifest}
    output.mkdir(mode=0o755)
    _write_json(output / "INPUT_MANIFEST.json", manifest)
    results = []
    for row in rows:
        folder = output / row["session_id"]
        folder.mkdir()
        motion = hand_motion_arrays(row["nominal"], row["fitted"])
        with (folder / "HAND_MOTION_V1.npz").open("xb") as stream:
            np.savez_compressed(stream, **motion)
        review = _render_review(row["video"], row["fitted"], folder / "SENSOR_REVIEW.mp4")
        result = {"session_id": row["session_id"], "frames": row["audit"]["frames"],
                  "hand_motion": _reference(folder / "HAND_MOTION_V1.npz"),
                  "source_motion_nominal": row["sources"][0]["motion_ref"],
                  "source_motion_C3": row["sources"][1]["motion_ref"],
                  "source_video": _reference(row["video"]), "audit": row["audit"], **review}
        _write_json(folder / "RESULT.json", result)
        results.append(result)
    final = {"status": "EXECUTED_PENDING_VISUAL_REVIEW", "schema_version": "chaoyang-v5-sensor-result-v1",
             "motion_source": "controller_manus", "sessions": results,
             "total_frames": sum(item["frames"] for item in results),
             "M1_historical": "UNAVAILABLE_IN_THIS_BOUND_SOURCE_NO_OFFSET_APPLIED",
             "training_eligible": False, "control_ground_truth": False, "physical_calibration_verified": False,
             "quality_pass": False}
    _write_json(output / "RESULT.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    value = run(args.source_root, args.output_root, args.dry_run)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
