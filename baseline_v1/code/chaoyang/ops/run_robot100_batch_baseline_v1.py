"""Produce a frozen 100-session Controller+MANUS to Robot visual baseline.

The operation is intentionally a development visualisation.  It preserves the
acquisition-aligned HDF5 time axis, retargets wrist-local MANUS geometry to the
pinned KaiHand assets, drives the pinned Tianji model through one fixed virtual
placement, and renders the real adapter mesh.  It does not create calibration,
contact, control, training, or physical-deployment authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.full_robot_review_v2 import (
    HUMAN_TO_PHYSICAL,
    relative_wrist,
    solve_full_chain,
    validate_motion,
)
from chaoyang.pipeline.huro_hand_frame_v2 import solve_aligned_sequence
from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
from chaoyang.pipeline.pico_manus_motion_v2 import MANUS_NAMES, MAP_25_TO_21
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.v5_product import ProductRobotRenderer


TASK_ID = "robot100_batch_baseline_20260928"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
ALLOWED_DATA_ROOT = Path("/path/to/private_data/egodata/datasets/ego")
SELECTION_SCHEMA = "CHAOYANG_ROBOT100_FROZEN_SELECTION_V1"
RESULT_SCHEMA = "CHAOYANG_ROBOT100_SESSION_RESULT_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_ref(path: Path, expected: dict[str, Any]) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"SYMLINK_INPUT:{path}")
    value = path.resolve(strict=True)
    if not value.is_relative_to(ALLOWED_DATA_ROOT.resolve(strict=True)):
        raise RuntimeError(f"INPUT_SCOPE_ESCAPE:{value}")
    before = value.stat()
    if before.st_size != int(expected["bytes"]):
        raise RuntimeError(f"INPUT_SIZE_DRIFT:{value}")
    actual_sha = _sha256(value)
    after = value.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise RuntimeError(f"UNSTABLE_INPUT:{value}")
    if actual_sha != expected["sha256"]:
        raise RuntimeError(f"INPUT_SHA_DRIFT:{value}")
    return value


def _assert_route() -> dict[str, Any]:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    rows = [
        row for row in index.get("task_packets", [])
        if row.get("task_id") == TASK_ID and row.get("execution_allowed") is True
    ]
    if len(rows) != 1:
        raise RuntimeError("TASK_NOT_EXACTLY_ONCE_ROUTABLE")
    packet_path = REPO_ROOT / rows[0]["packet_path"]
    packet = load_json(packet_path)
    if artifact_ref(packet_path)["sha256"] != rows[0]["packet_sha256"]:
        raise RuntimeError("TASK_PACKET_SHA_DRIFT")
    if packet.get("task_id") != TASK_ID:
        raise RuntimeError("TASK_PACKET_ID_DRIFT")
    return packet


def _pose_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("POSE_LAYOUT_DRIFT")
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    result[:, :3, :3] = Rotation.from_quat(values[:, 3:]).as_matrix()
    result[:, :3, 3] = values[:, :3]
    return result


def _load_motion(row: dict[str, Any], max_hand_evaluations: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hdf5_path = _stable_ref(Path(row["hdf5"]["path"]), row["hdf5"])
    with h5py.File(hdf5_path, "r") as source:
        attrs = dict(source.attrs)
        if attrs.get("schema") != "egodex_v1":
            raise RuntimeError("HDF5_SCHEMA_DRIFT")
        if attrs.get("pose_layout") != "pos(xyz) + quat(x,y,z,w)":
            raise RuntimeError("POSE_CONVENTION_DRIFT")
        if attrs.get("hand_frame") != "wrist-local (relative to MANUS wrist root)":
            raise RuntimeError("MANUS_FRAME_DRIFT")
        if tuple(json.loads(attrs["joint_names"])) != tuple(MANUS_NAMES):
            raise RuntimeError("MANUS_NAME_ORDER_DRIFT")
        source_video_frame_idx = np.asarray(source["video_frame_idx"], dtype=np.int64)
        timestamp_ns = np.asarray(source["timestamp_ns"], dtype=np.int64)
        segment_id = np.asarray(source["segment_id"], dtype=np.int64)
        local25 = np.stack(
            [np.asarray(source["left_hand_joints"]), np.asarray(source["right_hand_joints"])],
            axis=1,
        ).astype(np.float64)
        hand_valid = np.stack(
            [np.asarray(source["left_hand_valid"]), np.asarray(source["right_hand_valid"])],
            axis=1,
        ).astype(bool)
        wrist = np.stack(
            [np.asarray(source["left_wrist_pose"]), np.asarray(source["right_wrist_pose"])],
            axis=1,
        ).astype(np.float64)
        # egodex_v1 exports only complete rows: Controller is a required route
        # and wrist_pose is its same-row calibrated derivative.  There is no
        # separately stored wrist-valid dataset; finite pose plus video_valid
        # is therefore the exact contract, not an invented observation flag.
        video_valid = np.asarray(source["video_valid"], dtype=bool)
        wrist_valid = np.stack(
            [np.isfinite(wrist[:, 0]).all(axis=1), np.isfinite(wrist[:, 1]).all(axis=1)],
            axis=1,
        ) & video_valid[:, None]
        fps = float(attrs["fps"])
    n = len(source_video_frame_idx)
    if n < 2 or not all(value.shape[0] == n for value in (timestamp_ns, segment_id, local25, hand_valid, wrist, wrist_valid)):
        raise RuntimeError("HDF5_TIME_AXIS_DRIFT")
    if np.any(np.diff(source_video_frame_idx) < 0) or np.any(np.diff(timestamp_ns) <= 0):
        raise RuntimeError("NONMONOTONIC_SOURCE_TIME")
    # The cleaned review MP4 is already one frame per exported HDF5 row.  Keep
    # its logical row index separate from the raw VST frame mapping, where a
    # repeated nearest-neighbour source frame is explicitly legal.
    frame_id = np.arange(n, dtype=np.int64)
    if not np.isfinite(local25[hand_valid]).all() or not np.isfinite(wrist[wrist_valid]).all():
        raise RuntimeError("VALID_SENSOR_NONFINITE")

    local21 = np.zeros((n, 2, 21, 3), dtype=np.float64)
    local21[:, :, 1:] = local25[:, :, np.asarray(MAP_25_TO_21[1:], dtype=np.int64)]
    assets = load_pinned_robot_assets(REPO_ROOT)
    hands = tuple(
        load_hand_model(model.path, side)
        for model, side in zip((assets.left_hand, assets.right_hand), ("left", "right"), strict=True)
    )
    fit = solve_aligned_sequence(
        hands,
        local21.transpose(1, 0, 2, 3),
        hand_valid.T,
        frame_id,
        max_evaluations=max_hand_evaluations,
    )

    anatomical_world = np.full((n, 2, 4, 4), np.nan, dtype=np.float64)
    for side in range(2):
        anatomical_world[:, side] = _pose_matrix(wrist[:, side])
    # The existing full-chain baseline maps anatomical L/R to physical robot R/L.
    physical_world = anatomical_world[:, HUMAN_TO_PHYSICAL]
    physical_wrist_valid = wrist_valid[:, HUMAN_TO_PHYSICAL]
    relative, origins = relative_wrist(physical_world, physical_wrist_valid)
    physical_finger_valid = np.asarray(fit["valid"], dtype=bool)[:, HUMAN_TO_PHYSICAL]
    q22 = np.asarray(fit["q22"], dtype=np.float64)[:, HUMAN_TO_PHYSICAL]
    arrays = {
        "q22": q22,
        "finger_valid": physical_finger_valid,
        "wrist_valid": physical_wrist_valid,
        "relative_wrist_T": relative,
        "timestamp_ns": timestamp_ns,
        "frame_id": frame_id,
        "segment_id": segment_id,
        "source_video_frame_idx": source_video_frame_idx,
        "human_to_physical": HUMAN_TO_PHYSICAL.copy(),
    }
    validate_motion(arrays)
    metrics = {
        "frames": n,
        "fps": fps,
        "wrist_valid_left_right": wrist_valid.sum(axis=0).astype(int).tolist(),
        "hand_valid_left_right": hand_valid.sum(axis=0).astype(int).tolist(),
        "finger_solver_valid_physical_left_right": physical_finger_valid.sum(axis=0).astype(int).tolist(),
        "wrist_origin_world": origins.tolist(),
        "hand_solver_max_evaluations": max_hand_evaluations,
    }
    return arrays, metrics


def _mounts() -> tuple[np.ndarray, dict[str, Any]]:
    path = REPO_ROOT / (
        "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    )
    value = load_json(path)
    if value.get("classification") != "REAL_CAD_GEOMETRY_WITH_UNMEASURED_VIRTUAL_INSTALLATION":
        raise RuntimeError("MOUNT_CLASSIFICATION_DRIFT")
    mounts = np.asarray(
        [
            value["transforms"]["left_flange_to_hand_root"],
            value["transforms"]["right_flange_to_hand_root"],
        ],
        dtype=np.float64,
    )
    return mounts, artifact_ref(path)


def _letterbox_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    result = np.full((height, width, 3), 244, dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    x, y = (width - rw) // 2, (height - rh) // 2
    result[y:y + rh, x:x + rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
    return result


def _put_chinese(image: np.ndarray, lines: list[tuple[str, tuple[int, int], int]]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    font_paths = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    font_path = next((path for path in font_paths if path.is_file()), None)
    if font_path is None:
        raise RuntimeError("CHINESE_FONT_MISSING")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    fonts: dict[int, Any] = {}
    for text, position, size in lines:
        fonts.setdefault(size, ImageFont.truetype(str(font_path), size=size))
        draw.text(position, text, font=fonts[size], fill=(24, 24, 24))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _render(
    row: dict[str, Any],
    solved: dict[str, np.ndarray],
    destination: Path,
    *,
    camera_yaw_degrees: float = 40.0,
    view_description: str = "固定外部视角",
) -> dict[str, Any]:
    if not np.isfinite(camera_yaw_degrees):
        raise ValueError("CAMERA_YAW_NONFINITE")
    video = _stable_ref(Path(row["video"]["path"]), row["video"])
    motion = dict(solved)
    motion["q_hand22"] = solved["q22"]
    motion["T_cam_base"] = np.eye(4, dtype=np.float64)
    renderer = ProductRobotRenderer(
        REPO_ROOT,
        motion,
        {"width": 640, "height": 480, "K": [[580.0, 0.0, 320.0], [0.0, 580.0, 240.0], [0.0, 0.0, 1.0]]},
        include_adapter=True,
    )
    from chaoyang.ops.render_tianji_kai_mount_proxy_audit import body_bounds
    lower, upper = body_bounds(renderer.client, (renderer.robot, *renderer.hands, *renderer.adapters))
    center = ((lower + upper) * 0.5).astype(float)
    distance = max(2.1, float(np.linalg.norm(upper - lower)) * 1.08)
    renderer.view = renderer.b.computeViewMatrixFromYawPitchRoll(
        center.tolist(), distance, float(camera_yaw_degrees), -22.0, 0.0, 2
    )
    renderer.projection = renderer.b.computeProjectionMatrixFOV(50.0, 640.0 / 480.0, 0.02, 20.0)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        renderer.close()
        raise RuntimeError("SOURCE_VIDEO_UNREADABLE")
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x720", "-r", str(row["fps"]),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = 0
    visible_pixels = 0
    try:
        for index in range(len(solved["frame_id"])):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"SOURCE_VIDEO_TRUNCATED:{index}")
            source_panel = _letterbox_bgr(raw, 640, 480)
            layers = renderer.frame_layers(index)
            robot_panel = cv2.cvtColor(layers.rgb, cv2.COLOR_RGB2BGR)
            robot_panel[~layers.alpha] = 244
            visible_pixels += int(layers.alpha.sum())
            canvas = np.full((720, 1280, 3), 244, dtype=np.uint8)
            canvas[64:544, :640] = source_panel
            canvas[64:544, 640:] = robot_panel
            valid = solved["wrist_valid"][index].astype(int).tolist()
            passed = solved["tolerance_pass"][index].astype(int).tolist()
            lines = [
                (f"{row['session_id']}｜标准 Robot 开发基线", (18, 12), 28),
                ("左：采集视频　右：Tianji＋KaiHand＋真实连接件", (18, 552), 24),
                (f"源帧 {int(solved['frame_id'][index])}｜腕有效(物理左/右) {valid}｜虚拟跟随门 {passed}", (18, 590), 21),
                (f"固定虚拟安装与{view_description}；非物理标定、非接触真值、不可用于机器人控制或训练", (18, 628), 20),
                ("灰色部件表示该侧输入无效；颜色仅用于左右手语义区分", (18, 662), 20),
            ]
            canvas = _put_chinese(canvas, lines)
            assert writer.stdin is not None
            writer.stdin.write(canvas.tobytes())
            frames += 1
        assert writer.stdin is not None
        writer.stdin.close()
        error = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        if writer.wait() != 0:
            raise RuntimeError("FFMPEG_FAILED:" + error[-1600:])
    finally:
        capture.release()
        renderer.close()
        if writer.poll() is None:
            writer.kill()
            writer.wait()
    if visible_pixels <= 0:
        raise RuntimeError("NO_VISIBLE_ROBOT_PIXELS")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", str(destination)],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    decoded = int(stream["nb_read_frames"])
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2", "-i", str(destination), "-map", "0:v:0", "-f", "null", "-"], check=True)
    if decoded != frames:
        raise RuntimeError("OUTPUT_FRAME_COUNT_DRIFT")
    return {"frames": frames, "decoded_frames": decoded, "visible_robot_pixels": visible_pixels, "probe": stream}


def _session(row: dict[str, Any], output: Path, max_hand_evaluations: int, max_arm_evaluations: int) -> dict[str, Any]:
    folder = output / row["session_id"]
    final = folder / "RESULT.json"
    if final.is_file():
        value = load_json(final)
        if value.get("schema_version") != RESULT_SCHEMA or value.get("session_id") != row["session_id"]:
            raise RuntimeError("EXISTING_RESULT_IDENTITY_DRIFT")
        return value
    if folder.exists():
        raise RuntimeError(f"PARTIAL_SESSION_OUTPUT:{folder}")
    folder.mkdir(parents=True)
    started = time.monotonic()
    arrays, sensor_metrics = _load_motion(row, max_hand_evaluations)
    assets = load_pinned_robot_assets(REPO_ROOT)
    flange_mounts, mount_ref = _mounts()
    solved = solve_full_chain(arrays, assets, flange_mounts, max_nfev=max_arm_evaluations, continuity_weight=0.5)
    solved["source_video_frame_idx"] = arrays["source_video_frame_idx"].copy()
    motion_path = folder / "ROBOT_MOTION.npz"
    np.savez_compressed(motion_path, **solved)
    video_path = folder / f"{row['session_id']}_标准Robot基线.mp4"
    review = _render(row, solved, video_path)
    valid = solved["wrist_valid"]
    passed = solved["tolerance_pass"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "task_id": TASK_ID,
        "session_id": row["session_id"],
        "task": row["task"],
        "status": "COMPLETE_DEVELOPMENT_VISUAL_BASELINE",
        "selection_row": row,
        "sensor_metrics": sensor_metrics,
        "motion": artifact_ref(motion_path),
        "video": artifact_ref(video_path),
        "review": review,
        "wrist_valid_physical_left_right": valid.sum(axis=0).astype(int).tolist(),
        "arm_tolerance_pass_physical_left_right": passed.sum(axis=0).astype(int).tolist(),
        "arm_failure_physical_left_right": (valid & ~passed).sum(axis=0).astype(int).tolist(),
        "mount_contract": mount_ref,
        "elapsed_seconds": time.monotonic() - started,
        "human_review": "PENDING",
        "algorithm_quality_adopted": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "claim_limit": "Synchronized Controller+MANUS driven standardised Robot development visualisation only; fixed virtual placement, no physical calibration/contact/control/training authority.",
    }
    atomic_json(final, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-label")
    parser.add_argument("--max-hand-evaluations", type=int, default=24)
    parser.add_argument("--max-arm-evaluations", type=int, default=60)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _assert_route()
    selection_path = args.selection.resolve(strict=True)
    selection = load_json(selection_path)
    if selection.get("schema_version") != SELECTION_SCHEMA or selection.get("count") != 100:
        raise RuntimeError("SELECTION_SCHEMA_OR_COUNT")
    output = args.output_root.resolve()
    allowed = (ATTEMPT / "sessions").resolve()
    if output != allowed:
        raise RuntimeError("OUTPUT_ROOT_SCOPE")
    output.mkdir(parents=True, exist_ok=True)
    if not (0 <= args.worker_index < args.worker_count <= 4):
        raise ValueError("WORKER_PARTITION_INVALID")
    rows = selection["rows"][args.worker_index::args.worker_count]
    if args.limit is not None:
        rows = rows[:args.limit]
    completed = []
    failures = []
    for row in rows:
        try:
            result = _session(row, output, args.max_hand_evaluations, args.max_arm_evaluations)
            completed.append({"session_id": row["session_id"], "result": result["video"]})
            print(json.dumps({"session": row["session_id"], "status": "COMPLETE", "completed": len(completed)}, ensure_ascii=False), flush=True)
        except Exception as error:
            failures.append({"session_id": row["session_id"], "error_type": type(error).__name__, "error": str(error)})
            print(json.dumps({"session": row["session_id"], "status": "FAILED", "error": str(error)}, ensure_ascii=False), flush=True)
    worker_result = {
        "schema_version": "CHAOYANG_ROBOT100_WORKER_RESULT_V1",
        "task_id": TASK_ID,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "selection": artifact_ref(selection_path),
        "assigned": len(rows),
        "completed": completed,
        "failures": failures,
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
    }
    label = args.worker_label or f"worker_{args.worker_index:02d}"
    if not label.replace("_", "").isalnum():
        raise ValueError("WORKER_LABEL_INVALID")
    worker_path = ATTEMPT / "workers" / f"{label}.json"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_path.exists():
        existing = load_json(worker_path)
        if existing != worker_result:
            raise RuntimeError("WORKER_RESULT_EXISTS_WITH_DIFFERENT_CONTENT")
    else:
        atomic_json(worker_path, worker_result)
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
