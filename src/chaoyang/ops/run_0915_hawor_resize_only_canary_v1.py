#!/usr/bin/env python3
"""Run one governed HaWoR canary on the confirmed 0915 resize-only left eye."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_hawor_resize_only_canary_v1"
PHASE = "0915_HAWOR_PHYSICAL_LEFT_RESIZE_ONLY_CANARY"
SESSION_ID = "play_cards_0915_001"
SOURCE = Path(
    "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/"
    "cleaned/playing_cards/play_cards_0915_001"
)
TARGET_SIZE = (1280, 960)
FRAME_INDICES = (0, 25, 50, 75, 100, 149)
CHAINS = (
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def scaled_intrinsics(camera: dict[str, Any]) -> np.ndarray:
    left = camera["left"]
    if left.get("sourceIndex") != 1:
        raise RuntimeError("physical-left sourceIndex drift")
    sx = TARGET_SIZE[0] / int(camera["width"])
    sy = TARGET_SIZE[1] / int(camera["height"])
    intrinsics = left["intrinsics"]
    return np.asarray([
        [float(intrinsics["fx"]) * sx, 0.0, float(intrinsics["cx"]) * sx],
        [0.0, float(intrinsics["fy"]) * sy, float(intrinsics["cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def physical_left_c2w(selected_c2w: np.ndarray,
                      camera: dict[str, Any]) -> np.ndarray:
    if camera.get("extrinsic_convention") != "head_to_camera_4x4_row_major":
        raise RuntimeError("unexpected camera extrinsic convention")
    left = np.asarray(camera["extrinsics"]["left"], np.float64)
    right = np.asarray(camera["extrinsics"]["right"], np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise RuntimeError("camera extrinsics must be 4x4")
    return selected_c2w @ right @ np.linalg.inv(left)


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load(packet_path)
    if packet != build_packet(TASK_ID) or len(packet.get("weights", [])) != 1:
        raise RuntimeError("task packet differs from the one-weight frozen spec")
    state = load(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    task = next((row for row in state.get("tasks", [])
                 if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"
    }:
        raise RuntimeError("task is not executable")
    index = load(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize execution")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    confirmation = load(ROOT / "tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json")
    if confirmation.get("authorized_next_scope") != "ONE_SESSION_HAWOR_CANARY_ONLY":
        raise RuntimeError("user confirmation scope drift")
    return packet


def heartbeat() -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", "RUNNING", "--phase", PHASE,
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def full_decode(path: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode or int(stream["nb_read_frames"]) != expected_frames:
        raise RuntimeError("video full-decode gate failed")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]), "fps": stream["r_frame_rate"],
        "status": "PASS_FULL_DECODE",
    }


def open_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ], stdin=subprocess.PIPE)


def prepare_input(output: Path) -> tuple[Path, dict[str, Any]]:
    stereo_files = sorted((SOURCE / "source_stereo").glob("CameraRecord_*_stereo.mp4"))
    frame_jsons = sorted((SOURCE / "preprocess/all_data").glob("*/training_data.json"))
    camera_path = SOURCE / "camera_params.json"
    if len(stereo_files) != 1 or len(frame_jsons) != 150:
        raise RuntimeError("fixed canary source identity drift")
    camera = load(camera_path)
    k = scaled_intrinsics(camera)
    source_width, source_height = int(camera["width"]), int(camera["height"])
    video = output / "input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
    video.parent.mkdir(parents=True)
    capture = cv2.VideoCapture(str(stereo_files[0]))
    encoder = open_encoder(video, *TARGET_SIZE, 30.0)
    assert encoder.stdin is not None
    frames = 0
    try:
        while True:
            ok, sbs = capture.read()
            if not ok:
                break
            if sbs.shape[:2] != (source_height, source_width * 2):
                raise RuntimeError("SBS geometry drift")
            left = sbs[:, source_width:source_width * 2]
            resized = cv2.resize(left, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            encoder.stdin.write(resized.tobytes())
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frames != 150:
        raise RuntimeError("resize-only video publication failed")
    decode = full_decode(video, 150)

    adapter = output / "input/adapter_session"
    frame_root = adapter / "preprocess/all_data"
    frame_root.mkdir(parents=True)
    os.symlink(os.path.relpath(video, adapter),
               adapter / "CameraRecord_play_cards_0910_001.mp4")
    for frame_index, path in enumerate(frame_jsons):
        payload = load(path)
        selected_c2w = np.asarray(payload["metadata"]["c2w"], np.float64)
        frame_dir = frame_root / f"{frame_index:05d}"
        frame_dir.mkdir()
        atomic_json(frame_dir / "training_data.json", {
            "metadata": {
                "c2w": physical_left_c2w(selected_c2w, camera).tolist(),
                "k": k.tolist(), "fps": 30.0, "frame_index": frame_index,
                "camera_eye": "physical_left", "camera_source_index": 1,
                "rectified": False,
                "image_domain": "PASSTHROUGH_SOURCEINDEX1_RESIZE_ONLY_1280X960",
                "distortion_policy": "SOURCE_PIXELS_PRESERVED_EQUIDIS62_NOT_APPLIED",
            }
        })
    evidence = {
        "schema_version": "0915-hawor-resize-only-input-domain-v1",
        "status": "PASS", "session_id": SESSION_ID, "frame_count": frames,
        "physical_eye": "left", "source_index": 1,
        "image_domain": "PASSTHROUGH_SOURCEINDEX1_RESIZE_ONLY_1280X960",
        "rectified": False, "remap_applied": False,
        "scaled_factory_intrinsics_unverified_externally": k.tolist(),
        "source": {"stereo": ref(stereo_files[0]), "camera": ref(camera_path)},
        "video": ref(video), "decode": decode,
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "controller_pose": "NOT_CONSUMED",
        "trackingData_hand": "NOT_CONSUMED",
        "user_confirmation": ref(
            ROOT / "tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json"
        ),
        "claim_limit": (
            "User-confirmed monocular appearance only. Scaled factory K is a development "
            "model input, not proof that the encoded pixels are a physical pinhole domain."
        ),
    }
    atomic_json(output / "INPUT_DOMAIN.json", evidence)
    return adapter, evidence


def draw_review(video: Path, npz_path: Path, visual: Path) -> dict[str, Any]:
    visual.mkdir(parents=True)
    with np.load(npz_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    review = visual / "0915_HAWOR_RESIZE_ONLY_REVIEW.mp4"
    contact = visual / "0915_HAWOR_RESIZE_ONLY_CONTACT_SHEET.jpg"
    capture = cv2.VideoCapture(str(video))
    encoder = open_encoder(review, *TARGET_SIZE, 30.0)
    assert encoder.stdin is not None
    selected: list[np.ndarray] = []
    frame_index = 0
    colors = ((255, 140, 20), (20, 40, 255))
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            for side in (0, 1):
                if not observed[side, frame_index]:
                    continue
                uv = np.rint(joints[side, frame_index]).astype(np.int32)
                for chain in CHAINS:
                    cv2.polylines(image, [uv[np.asarray(chain)]], False,
                                  colors[side], 3, cv2.LINE_AA)
                cv2.circle(image, tuple(uv[0]), 7, colors[side], -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (1279, 66), (0, 0, 0), -1)
            cv2.putText(
                image, f"{SESSION_ID} | A resize-only HaWoR | frame {frame_index:03d}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            label = (
                f"left={'OBS' if observed[0, frame_index] else 'MISS'}  "
                f"right={'OBS' if observed[1, frame_index] else 'MISS'} | BLUE=L RED=R"
            )
            cv2.putText(image, label, (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 2, cv2.LINE_AA)
            encoder.stdin.write(image.tobytes())
            if frame_index in FRAME_INDICES:
                selected.append(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA))
            frame_index += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frame_index != 150 or len(selected) != 6:
        raise RuntimeError("HaWoR visual publication failed")
    sheet = np.vstack([np.hstack(selected[:3]), np.hstack(selected[3:])])
    if not cv2.imwrite(str(contact), sheet):
        raise RuntimeError("contact sheet write failed")
    return {
        "review": ref(review), "review_decode": full_decode(review, 150),
        "contact_sheet": ref(contact),
    }


def write_terminal(output: Path, receipt: Path, status: str,
                   packet: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = {
        "schema_version": "0915-hawor-resize-only-canary-result-v1",
        "task_id": TASK_ID, "status": status, "weights": packet["weights"],
        "session_id": SESSION_ID,
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY",
        "source_mutated": False, "sam31_started": False,
        "depth_started": False, "batch_started": False,
        "claim_limit": packet["claim_limit"], **extra,
    }
    atomic_json(output / "RESULT.json", result)
    terminal = {**result, "result": ref(output / "RESULT.json")}
    atomic_json(receipt, terminal)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-hawor-resize-only-canary-run-receipt-v1",
        "task_id": TASK_ID, "status": status,
        "result": ref(output / "RESULT.json"), "terminal_receipt": ref(receipt),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)
    heartbeat()
    adapter, input_evidence = prepare_input(output)
    heartbeat()
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_output = output / "hawor"
    worker_command = [
        str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
        str(ROOT / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"),
        "--session-root", str(adapter), "--output-root", str(worker_output),
    ]
    lease_command = [
        sys.executable, str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", TASK_ID, "--attempt-id", output.name,
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds), "--wall-seconds", "3600",
        "--receipt", str(gpu_receipt), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-hawor-resize-only-command-v1",
        "task_id": TASK_ID, "weights": packet["weights"],
        "input_domain": ref(output / "INPUT_DOMAIN.json"),
        "lease_command": lease_command, "worker_command": worker_command,
    })
    started = time.time()
    with (output / "GPU_WRAPPER.log").open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            time.sleep(30)
            heartbeat()
    gpu = load(gpu_receipt) if gpu_receipt.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(
            output, receipt, status, packet,
            first_blocker=gpu.get("reason", "HAWOR_RUNTIME_FAILED"),
            input_domain=ref(output / "INPUT_DOMAIN.json"),
            gpu_command_receipt=ref(gpu_receipt) if gpu_receipt.is_file() else None,
            wall_seconds=time.time() - started,
        )
        return 3 if status == "BLOCKED_RESOURCE" else 2

    from chaoyang.ops.run_0915_hawor_persistent_worker_v1 import quality

    npz_path = worker_output / "HAWOR_RAW_MANO21.npz"
    metrics = quality(npz_path)
    atomic_json(output / "HAWOR_METRICS.json", metrics)
    visuals = draw_review(Path(input_evidence["video"]["path"]), npz_path, visual)
    session_admission = (
        "PASS_DEVELOPMENT_HAWOR" if metrics["numeric_mask_gate_pass"]
        else "FAILED_QUALITY_C"
    )
    write_terminal(
        output, receipt, "PASSED", packet,
        session_admission=session_admission,
        input_domain=ref(output / "INPUT_DOMAIN.json"),
        hawor_npz=ref(npz_path), hawor_metrics=ref(output / "HAWOR_METRICS.json"),
        gpu_command_receipt=ref(gpu_receipt), visuals=visuals,
        observed_frames=metrics["observed_frames"],
        numeric_mask_gate_pass=metrics["numeric_mask_gate_pass"],
        wall_seconds=time.time() - started,
        next_authority="USER_REVIEW_REQUIRED_NO_AUTOMATIC_SUCCESSOR",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
