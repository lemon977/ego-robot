#!/usr/bin/env python3
"""Build a model-free A/B of the held 0915 VST image domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_vst_image_domain_ab_v1"
PHASE = "0915_VST_IMAGE_DOMAIN_SINGLE_SESSION_AB"
SESSION = Path(
    "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/"
    "cleaned/playing_cards/play_cards_0915_001"
)
PREPARED = (
    ROOT / "_run/current/0915_input_prepare_cad_v2/attempts/attempt_0001/"
    "prepared_physical_left/sessions/playing_cards/play_cards_0915_001"
)
FRAME_INDICES = (0, 25, 50, 75, 100, 149)
TARGET_SIZE = (1280, 960)


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
    return json.loads(path.read_text(encoding="utf-8"))


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


def validate_route() -> None:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load(packet_path)
    state = load(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    index = load(ROOT / "tasks/current/INDEX.json")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("VST image-domain task must have weights=ABSENT")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    task = next((row for row in state.get("tasks", [])
                 if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in {"PENDING", "CLAIMED", "RUNNING"}:
        raise RuntimeError("task state is not executable")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize execution")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")


def heartbeat() -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", "RUNNING", "--phase", PHASE,
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("image metric inputs differ in shape")
    delta = reference.astype(np.float32) - candidate.astype(np.float32)
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    psnr = 99.0 if rmse == 0 else float(20.0 * math.log10(255.0 / rmse))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_centered = ref_gray - float(ref_gray.mean())
    cand_centered = cand_gray - float(cand_gray.mean())
    denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered))
    correlation = float(np.sum(ref_centered * cand_centered) / denom) if denom else 0.0
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr,
            "luma_correlation": correlation}


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "p05": float(np.percentile([row[key] for row in rows], 5)),
            "p50": float(np.percentile([row[key] for row in rows], 50)),
            "p95": float(np.percentile([row[key] for row in rows], 95)),
        }
        for key in ("mae", "rmse", "psnr_db", "luma_correlation")
    }


def panel(image: np.ndarray, title: str, frame_index: int,
          size: tuple[int, int] = (480, 360)) -> np.ndarray:
    body = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    result = np.zeros((size[1] + 42, size[0], 3), np.uint8)
    result[42:] = body
    cv2.putText(result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"frame {frame_index:03d}", (size[0] - 115, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1,
                cv2.LINE_AA)
    return result


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("no review frames")
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("review frame geometry changed")
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg review encoder failed")
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise RuntimeError(decoded.stderr[-2000:])
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    return json.loads(probe.stdout)["streams"][0]


def load_stereo_module() -> Any:
    script_root = ROOT / "vendor/FoundationStereo/scripts"
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import pico_stereo_depth  # type: ignore
    return pico_stereo_depth


def map_metrics(map_x: np.ndarray, map_y: np.ndarray,
                source_width: int, source_height: int) -> tuple[dict[str, Any], np.ndarray]:
    width, height = TARGET_SIZE
    scale_x = source_width / width
    scale_y = source_height / height
    grid_x, grid_y = np.meshgrid(
        (np.arange(width, dtype=np.float32) + 0.5) * scale_x - 0.5,
        (np.arange(height, dtype=np.float32) + 0.5) * scale_y - 0.5,
    )
    dx = (map_x - grid_x) / scale_x
    dy = (map_y - grid_y) / scale_y
    displacement = np.hypot(dx, dy)
    valid = ((map_x >= 0) & (map_x < source_width - 1)
             & (map_y >= 0) & (map_y < source_height - 1))
    values = displacement[valid]
    stats = {
        "units": "output_pixels_1280x960",
        "valid_map_fraction": float(valid.mean()),
        "border_fill_fraction": float(1.0 - valid.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "center": float(displacement[height // 2, width // 2]),
    }
    return stats, displacement


def warp_visual(displacement: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                source_width: int, source_height: int,
                stats: dict[str, Any]) -> np.ndarray:
    normalized = np.clip(displacement / max(float(stats["p99"]), 1.0), 0, 1)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    scale_x = source_width / TARGET_SIZE[0]
    scale_y = source_height / TARGET_SIZE[1]
    for y in range(40, TARGET_SIZE[1], 80):
        for x in range(40, TARGET_SIZE[0], 80):
            target = (
                int(np.clip(map_x[y, x] / scale_x, 0, TARGET_SIZE[0] - 1)),
                int(np.clip(map_y[y, x] / scale_y, 0, TARGET_SIZE[1] - 1)),
            )
            cv2.arrowedLine(heat, (x, y), target, (255, 255, 255), 1,
                            cv2.LINE_AA, tipLength=0.2)
    cv2.rectangle(heat, (0, 0), (1279, 76), (0, 0, 0), -1)
    text = (
        f"Current remap displacement vs resize-only: p50={stats['p50']:.1f}px  "
        f"p95={stats['p95']:.1f}px  p99={stats['p99']:.1f}px  "
        f"border={100*stats['border_fill_fraction']:.2f}%"
    )
    cv2.putText(heat, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heat, "Arrows: resize-only output pixel -> source sample used by current remap",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (220, 220, 220), 1, cv2.LINE_AA)
    return heat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    heartbeat()
    started = time.time()

    stereo_path = next((SESSION / "source_stereo").glob("CameraRecord_*_stereo.mp4"))
    legacy_path = SESSION / "CameraRecord_play_cards_0915_001.mp4"
    camera_path = SESSION / "camera_params.json"
    manifest_path = SESSION / "clip_manifest.json"
    remapped_path = PREPARED / "leftmono/LEFT_MONO_RECTIFIED.mp4"
    for path in (stereo_path, legacy_path, camera_path, manifest_path, remapped_path):
        if not path.is_file():
            raise RuntimeError(f"missing input: {path}")

    stereo_module = load_stereo_module()
    eyes, eye_width, eye_height, _baseline = stereo_module.load_camera_params(camera_path)
    physical_left_index, physical_right_index = stereo_module.load_source_indices(camera_path)
    if (physical_left_index, physical_right_index) != (1, 0):
        raise RuntimeError("fixed session physical-eye routing changed")
    virtual_k = stereo_module.virtual_intrinsics(*TARGET_SIZE, 90.0)
    map_x, map_y = stereo_module.make_map(
        eyes[0], np.eye(3), *TARGET_SIZE, virtual_k
    )
    remap_stats, displacement = map_metrics(
        map_x, map_y, eye_width, eye_height
    )

    captures = [cv2.VideoCapture(str(path)) for path in
                (stereo_path, legacy_path, remapped_path)]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("one or more videos failed to open")
    metrics = {
        "legacy_vs_source0_resize": [],
        "legacy_vs_source1_resize": [],
        "stored_remap_vs_source1_resize": [],
        "stored_remap_vs_computed_remap": [],
    }
    video_frames: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    frame_count = 0
    try:
        while True:
            decoded = [capture.read() for capture in captures]
            if not decoded[0][0]:
                break
            if not decoded[1][0] or not decoded[2][0]:
                raise RuntimeError("video frame count mismatch")
            stereo, legacy, stored_remap = (row[1] for row in decoded)
            if stereo.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"unexpected SBS shape: {stereo.shape}")
            source0 = stereo[:, :eye_width]
            source1 = stereo[:, eye_width:]
            source0_resized = cv2.resize(source0, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            source1_resized = cv2.resize(source1, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            if legacy.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                legacy = cv2.resize(legacy, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            if stored_remap.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                raise RuntimeError("stored remap geometry drift")
            computed_remap = cv2.remap(
                source1, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            metrics["legacy_vs_source0_resize"].append(
                image_metrics(legacy, source0_resized))
            metrics["legacy_vs_source1_resize"].append(
                image_metrics(legacy, source1_resized))
            metrics["stored_remap_vs_source1_resize"].append(
                image_metrics(stored_remap, source1_resized))
            metrics["stored_remap_vs_computed_remap"].append(
                image_metrics(stored_remap, computed_remap))
            difference = cv2.convertScaleAbs(
                cv2.absdiff(source1_resized, stored_remap), alpha=2.5
            )
            panels = [
                panel(source1_resized, "A: physical left sourceIndex=1 | resize only", frame_count),
                panel(stored_remap, "B: current equiDis62 -> pinhole | HELD", frame_count),
                panel(difference, "|A-B| x2.5 | geometry + codec", frame_count),
                panel(legacy, "Legacy processed mono | sourceIndex=0 ref", frame_count),
            ]
            row = np.hstack(panels)
            video_frames.append(row)
            if frame_count in FRAME_INDICES:
                selected_rows.append(cv2.resize(row, (1280, 268), interpolation=cv2.INTER_AREA))
            frame_count += 1
    finally:
        for capture in captures:
            capture.release()
    if frame_count != 150 or len(selected_rows) != len(FRAME_INDICES):
        raise RuntimeError(f"expected 150 frames and six review rows, got {frame_count}/{len(selected_rows)}")

    aggregated = {name: aggregate(rows) for name, rows in metrics.items()}
    source0_psnr = aggregated["legacy_vs_source0_resize"]["psnr_db"]["p50"]
    source1_psnr = aggregated["legacy_vs_source1_resize"]["psnr_db"]["p50"]
    legacy_match = 0 if source0_psnr > source1_psnr else 1
    metrics_payload = {
        "schema_version": "0915-vst-image-domain-metrics-v1",
        "session_id": "play_cards_0915_001",
        "frame_count": frame_count,
        "source_geometry": {"sbs": [eye_width * 2, eye_height],
                            "per_eye": [eye_width, eye_height]},
        "target_geometry": list(TARGET_SIZE),
        "physical_eye_source_indices": {"left": physical_left_index,
                                        "right": physical_right_index},
        "legacy_processed_mono_best_sbs_half": legacy_match,
        "legacy_processed_mono_manifest_source_index": load(manifest_path)["selection"]["source_index"],
        "current_remap_displacement": remap_stats,
        "full_video_image_metrics": aggregated,
        "claim_limit": "Pixel-domain and transform diagnostics only; no calibration or model accuracy claim.",
    }
    metrics_path = output / "IMAGE_DOMAIN_METRICS.json"
    atomic_json(metrics_path, metrics_payload)

    contact_path = visual / "0915_VST_IMAGE_DOMAIN_AB_CONTACT_SHEET.jpg"
    if not cv2.imwrite(str(contact_path), np.vstack(selected_rows),
                       [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError("contact sheet write failed")
    warp_path = visual / "0915_VST_IMAGE_DOMAIN_WARP_FIELD.png"
    if not cv2.imwrite(str(warp_path), warp_visual(
        displacement, map_x, map_y, eye_width, eye_height, remap_stats
    )):
        raise RuntimeError("warp visual write failed")
    video_path = visual / "0915_VST_IMAGE_DOMAIN_AB_REVIEW.mp4"
    video_probe = write_video(video_path, video_frames, 30.0)

    conclusion = {
        "current_remap_baseline_admissible": False,
        "candidate_mono_domain": "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY",
        "candidate_authority": "BOUNDED_VISUAL_RESEARCH_PENDING_USER_CONFIRMATION",
        "legacy_mono_role": "DIFFERENT_PHYSICAL_EYE_REFERENCE_ONLY",
        "stereo_domain_decision": "NOT_EVALUATED",
        "reason": (
            "The held path performs a material nonlinear pixel remap. The legacy processed "
            "mono independently matches sourceIndex=0, while physical left is sourceIndex=1. "
            "No VST export-domain specification authorizes applying factory equiDis62 to the "
            "encoded SBS pixels."
        ),
    }
    result = {
        "schema_version": "0915-vst-image-domain-ab-result-v1",
        "task_id": TASK_ID,
        "status": "BLOCKED_EXTERNAL",
        "first_blocker": "USER_CONFIRM_PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_DOMAIN",
        "session_id": "play_cards_0915_001",
        "weights": "ABSENT",
        "gpu_used": False,
        "frame_count": frame_count,
        "inputs": {
            "source_stereo": ref(stereo_path),
            "legacy_processed_mono": ref(legacy_path),
            "camera_params": ref(camera_path),
            "clip_manifest": ref(manifest_path),
            "held_remapped_mono": ref(remapped_path),
        },
        "metrics": ref(metrics_path),
        "visuals": {
            "contact_sheet": ref(contact_path),
            "warp_field": ref(warp_path),
            "review_video": {**ref(video_path), "decode": video_probe},
        },
        "conclusion": conclusion,
        "source_mutated": False,
        "pico26_consumed": False,
        "controller_consumed": False,
        "model_inference_run": False,
        "wall_seconds": time.time() - started,
        "claim_limit": "Single-session image-domain A/B only; user confirmation is required before any new model task.",
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    receipt_payload = {
        "schema_version": "0915-vst-image-domain-ab-receipt-v1",
        "task_id": TASK_ID,
        "status": "BLOCKED_EXTERNAL",
        "first_blocker": result["first_blocker"],
        "result": ref(result_path),
        "metrics": ref(metrics_path),
        "visuals": result["visuals"],
        "conclusion": conclusion,
        "source_mutated": False,
        "claim_limit": result["claim_limit"],
    }
    atomic_json(receipt, receipt_payload)
    print(json.dumps(receipt_payload, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
