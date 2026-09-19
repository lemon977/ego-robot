#!/usr/bin/env python3
"""Run persistent HaWoR on the frozen four-session 0915 W0 cohort."""

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

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet
from chaoyang.ops.run_0915_hawor_resize_only_canary_v1 import (
    CHAINS,
    TARGET_SIZE,
    full_decode,
    open_encoder,
    physical_left_c2w,
    scaled_intrinsics,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_hawor_wave0_v1"
PHASE = "ROBOT15H_HAWOR_WAVE0_RESIZE_ONLY"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001"
OUTPUT = ROOT / "_run/current/0915_robot15h_hawor_wave0_v1/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_HAWOR_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_HAWOR_WAVE0_V1_RESULT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or not isinstance(packet.get("weights"), list) or len(packet["weights"]) != 1:
        raise RuntimeError("current HaWoR W0 packet differs from frozen one-weight spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("HaWoR W0 is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("current packet index does not authorize HaWoR W0")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("current HaWoR W0 packet SHA differs from route")
    return packet, packet_path


def heartbeat(status: str = "RUNNING") -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            TASK_ID,
            "--pid",
            str(os.getpid()),
            "--status",
            status,
            "--phase",
            PHASE,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def prepare_session(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    source_root = Path(str(row["source_stereo"]["path"])).parent.parent
    source_video = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
    if sha256(source_video) != row["source_stereo"]["sha256"]:
        raise RuntimeError(f"frozen W0 source stereo SHA drift: {row['session_id']}")
    camera_path = source_root / "camera_params.json"
    camera = load_json(camera_path)
    if camera.get("left", {}).get("sourceIndex") != 1 or camera.get("right", {}).get("sourceIndex") != 0:
        raise RuntimeError(f"physical stereo identity drift: {row['session_id']}")
    intrinsics = scaled_intrinsics(camera)
    source_width = int(camera["width"])
    source_height = int(camera["height"])
    frame_jsons = sorted((source_root / "preprocess/all_data").glob("*/training_data.json"))
    expected_frames = int(row["frame_count"])
    if len(frame_jsons) != expected_frames:
        raise RuntimeError(f"processed frame axis drift: {row['session_id']}")

    destination.mkdir(parents=True)
    video = destination / "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
    capture = cv2.VideoCapture(str(source_video))
    encoder = open_encoder(video, *TARGET_SIZE, 30.0)
    assert encoder.stdin is not None
    count = 0
    try:
        while True:
            ok, sbs = capture.read()
            if not ok:
                break
            if sbs.shape[:2] != (source_height, source_width * 2):
                raise RuntimeError(f"SBS geometry drift: {row['session_id']}")
            physical_left = sbs[:, source_width: source_width * 2]
            resized = cv2.resize(physical_left, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            encoder.stdin.write(resized.tobytes())
            count += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or count != expected_frames:
        raise RuntimeError(f"resize-only publication failed: {row['session_id']}")
    decode = full_decode(video, expected_frames)

    adapter = destination / "adapter_session"
    frames_root = adapter / "preprocess/all_data"
    frames_root.mkdir(parents=True)
    os.symlink(os.path.relpath(video, adapter), adapter / "CameraRecord_play_cards_0910_001.mp4")
    for frame_index, path in enumerate(frame_jsons):
        payload = load_json(path)
        selected_c2w = np.asarray(payload["metadata"]["c2w"], np.float64)
        frame_dir = frames_root / f"{frame_index:05d}"
        frame_dir.mkdir()
        atomic_json(frame_dir / "training_data.json", {
            "metadata": {
                "c2w": physical_left_c2w(selected_c2w, camera).tolist(),
                "k": intrinsics.tolist(),
                "fps": 30.0,
                "frame_index": frame_index,
                "camera_eye": "physical_left",
                "camera_source_index": 1,
                "rectified": False,
                "image_domain": "PASSTHROUGH_SOURCEINDEX1_RESIZE_ONLY_1280X960",
                "distortion_policy": "SOURCE_PIXELS_PRESERVED_EQUIDIS62_NOT_APPLIED",
            }
        })
    return {
        "task": row["task"],
        "session_id": row["session_id"],
        "source_group": row["source_group"],
        "frame_count": expected_frames,
        "adapter_session": str(adapter.resolve()),
        "prepared_video": ref(video),
        "source_stereo": row["source_stereo"],
        "camera_params": ref(camera_path),
        "decode": decode,
        "input_domain": {
            "physical_left_source_index": 1,
            "physical_right_source_index": 0,
            "operation": "CROP_THEN_RESIZE_ONLY",
            "output_size": [1280, 960],
            "lens_undistortion": False,
            "remap_applied": False,
            "pico26_hand_consumed": False,
            "trackingData_hand_consumed": False,
        },
    }


def render_review(video: Path, npz_path: Path, destination: Path, session_id: str, expected_frames: int) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    if joints.shape[:3] != (2, expected_frames, 21) or observed.shape != (2, expected_frames):
        raise RuntimeError(f"HaWoR frame axis drift: {session_id}")
    capture = cv2.VideoCapture(str(video))
    encoder = open_encoder(destination, *TARGET_SIZE, 30.0)
    assert encoder.stdin is not None
    colors = ((255, 140, 20), (20, 40, 255))
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            for side in (0, 1):
                if not observed[side, frame]:
                    continue
                uv = np.rint(joints[side, frame]).astype(np.int32)
                for chain in CHAINS:
                    cv2.polylines(image, [uv[np.asarray(chain)]], False, colors[side], 3, cv2.LINE_AA)
                cv2.circle(image, tuple(uv[0]), 7, colors[side], -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (1279, 66), (0, 0, 0), -1)
            cv2.putText(image, f"{session_id} | resize-only HaWoR W0 | frame {frame:04d}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"left={'OBS' if observed[0, frame] else 'MISS'} right={'OBS' if observed[1, frame] else 'MISS'} | NON_CONTROL",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA,
            )
            encoder.stdin.write(image.tobytes())
            frame += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frame != expected_frames:
        raise RuntimeError(f"review render failed: {session_id}")
    return {"video": ref(destination), "decode": full_decode(destination, expected_frames)}


def write_terminal(output: Path, receipt: Path, packet: dict[str, Any], status: str, **extra: Any) -> None:
    result = {
        "schema_version": "0915-robot15h-hawor-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": status,
        "weights": packet["weights"],
        "source_mutated": False,
        "processed_mutated": False,
        "pico26_hand_consumed": False,
        "trackingData_hand_consumed": False,
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": packet["claim_limit"],
        **extra,
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-hawor-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed HaWoR W0 output namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("executor epoch and fencing token are invalid")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)

    batch_path = INVENTORY / "BATCH_MANIFEST.json"
    source_groups_path = INVENTORY / "SOURCE_GROUP_MANIFEST.json"
    batch = load_json(batch_path)
    w0_rows = [row for row in batch.get("sessions", []) if row.get("wave") == "W0"]
    if len(w0_rows) != 4 or any(row.get("split") != "development" for row in w0_rows):
        raise RuntimeError("inventory did not freeze four development W0 sessions")
    signature_payload = {
        "schema_version": "0915-robot15h-hawor-wave0-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": packet["weights"],
        "task_packet": ref(packet_path),
        "inputs": [ref(batch_path), ref(source_groups_path)],
        "code": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
            ref(ROOT / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"),
        ],
        "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
        "forbidden_inputs": ["legacy_prepared_mono", "equiDis62_remap", "PICO26_hand", "trackingData_hand"],
    }
    signature_sha = canonical_sha(signature_payload)
    atomic_json(output / "RUN_SIGNATURE.json", {**signature_payload, "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-hawor-wave0-writer-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": packet["weights"],
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output),
        "run_signature_sha256": signature_sha,
        "task_packet": ref(packet_path),
    })
    heartbeat("CLAIMED")

    prepared_rows = []
    for row in w0_rows:
        prepared_rows.append(prepare_session(row, output / "prepared" / str(row["session_id"])))
        heartbeat()
    prepared_manifest = {
        "schema_version": "0915-hawor-resize-only-wave0-prepared-manifest-v1",
        "status": "PASS",
        "window_run_id": WINDOW_RUN_ID,
        "session_count": len(prepared_rows),
        "frame_count": sum(int(row["frame_count"]) for row in prepared_rows),
        "results": prepared_rows,
        "source_mutated": False,
        "processed_mutated": False,
    }
    atomic_json(output / "PREPARED_MANIFEST.json", prepared_manifest)

    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_output = output / "hawor"
    worker_command = [
        str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
        str(ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
        "--prepared-manifest", str(output / "PREPARED_MANIFEST.json"),
        "--output-root", str(worker_output),
    ]
    lease_command = [
        sys.executable,
        str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", TASK_ID,
        "--attempt-id", output.name,
        "--priority", "CANARY",
        "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", "9000",
        "--receipt", str(gpu_receipt),
        "--claim-limit", packet["claim_limit"],
        "--",
        *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-robot15h-hawor-wave0-command-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "weights": packet["weights"],
        "worker_command": worker_command,
        "lease_command": lease_command,
    })
    started = time.time()
    with (output / "GPU_WRAPPER.log").open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            heartbeat()
            state_path = worker_output / "STATE.json"
            state = load_json(state_path) if state_path.is_file() else {}
            atomic_json(output / "PROGRESS.json", {
                "schema_version": "0915-robot15h-hawor-wave0-progress-v1",
                "task_id": TASK_ID,
                "window_run_id": WINDOW_RUN_ID,
                "elapsed_seconds": time.time() - started,
                "gpu_wrapper_pid": process.pid,
                "worker": {key: state.get(key) for key in (
                    "state", "completed", "session_count", "passed", "quality_c",
                    "failed_runtime", "model_load_count", "detector_load_count",
                )},
            })
    gpu = load_json(gpu_receipt) if gpu_receipt.is_file() else {}
    worker_batch_path = worker_output / "BATCH_RESULT.json"
    worker_batch = load_json(worker_batch_path) if worker_batch_path.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED" or worker_batch.get("failed_runtime") != 0:
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        atomic_json(output / "REVIEW_MANIFEST.json", {
            "schema_version": "0915-robot15h-hawor-wave0-review-manifest-v1",
            "status": "NOT_RENDERED_UPSTREAM_RUNTIME_TERMINAL",
            "reviews": [],
        })
        atomic_json(output / "METRICS.json", {
            "schema_version": "0915-robot15h-hawor-wave0-metrics-v1",
            "status": status,
            "gpu_wrapper_returncode": process.returncode,
            "gpu_status": gpu.get("status"),
            "worker": worker_batch,
        })
        write_terminal(
            output, receipt, packet, status,
            first_blocker=gpu.get("reason", "HAWOR_W0_RUNTIME_FAILURE"),
            gpu_command_receipt=ref(gpu_receipt) if gpu_receipt.is_file() else None,
            worker_batch=ref(worker_batch_path) if worker_batch_path.is_file() else None,
            wall_seconds=time.time() - started,
        )
        return 3 if status == "BLOCKED_RESOURCE" else 2

    visual.mkdir(parents=True)
    prepared_by_id = {str(row["session_id"]): row for row in prepared_rows}
    reviews = []
    for item in worker_batch["results"]:
        session_id = str(item["session_id"])
        session_npz = Path(str(item["npz"]["path"]))
        prepared = prepared_by_id[session_id]
        destination = visual / f"{session_id}_HAWOR_REVIEW.mp4"
        review = render_review(
            Path(str(prepared["prepared_video"]["path"])),
            session_npz,
            destination,
            session_id,
            int(prepared["frame_count"]),
        )
        reviews.append({"session_id": session_id, "status": item["status"], **review})
    atomic_json(output / "REVIEW_MANIFEST.json", {
        "schema_version": "0915-robot15h-hawor-wave0-review-manifest-v1",
        "status": "COMPLETE",
        "reviews": reviews,
        "user_visual_acceptance": "PENDING",
    })
    readme = "# 0915 Robot15h W0 HaWoR 审阅\n\n"
    readme += "本目录只展示四个冻结 development 会话的物理左目 `sourceIndex=1 + crop + resize-only` HaWoR 结果。\n"
    readme += "没有应用 `equiDis62` 或其他 lens remap；蓝色为左手，红色为右手，缺失帧保持 MISS。\n\n"
    for item in reviews:
        name = Path(str(item["video"]["path"])).name
        readme += f"- [{item['session_id']}]({name})：`{item['status']}`\n"
    readme += "\n自动数值门与人工视觉验收分开；这些结果均为 NON_CONTROL / NON_DEPLOYABLE。\n"
    (visual / "README_ZH.md").write_text(readme, encoding="utf-8")
    metrics = {
        "schema_version": "0915-robot15h-hawor-wave0-metrics-v1",
        "status": "COMPLETED_ALL_TERMINAL",
        "session_count": worker_batch["session_count"],
        "frame_count": worker_batch["frame_count"],
        "pass_development_hawor": worker_batch["passed"],
        "rejected_quality": worker_batch["quality_c"],
        "failed_runtime": worker_batch["failed_runtime"],
        "model_load_count": worker_batch["model_load_count"],
        "detector_load_count": worker_batch["detector_load_count"],
        "tracker_reset_count": worker_batch["tracker_reset_count"],
        "wall_seconds": time.time() - started,
    }
    atomic_json(output / "METRICS.json", metrics)
    write_terminal(
        output, receipt, packet, "PASSED",
        batch_terminal="COMPLETED_ALL_TERMINAL",
        counts={
            "attempted": worker_batch["session_count"],
            "pass_development_hawor": worker_batch["passed"],
            "rejected_quality": worker_batch["quality_c"],
            "failed_runtime": worker_batch["failed_runtime"],
            "frames": worker_batch["frame_count"],
        },
        model_load_count=worker_batch["model_load_count"],
        detector_load_count=worker_batch["detector_load_count"],
        gpu_command_receipt=ref(gpu_receipt),
        worker_batch=ref(worker_batch_path),
        review_manifest=ref(output / "REVIEW_MANIFEST.json"),
        visual_readme=ref(visual / "README_ZH.md"),
        user_visual_acceptance="PENDING",
        wall_seconds=time.time() - started,
    )
    heartbeat("RUNNING")
    print(json.dumps({"status": "PASSED", **metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
