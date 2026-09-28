#!/usr/bin/env python3
"""Consume the S1 attachment tracks in bounded ProPainter Clean canaries.

Only the two frozen difficult windows are processed.  The propagated device
mask is unioned with the current semantic hand/forearm evidence, while direct
visible task-object interiors remain protected.  The output is diagnostic and
cannot become full-session Clean authority.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref
from chaoyang.ops.run_v5_scene import _mask_files, load as load_json, merge_roles
from chaoyang.pipeline.v5_scene import (
    FrameRoles,
    build_object_protected_repair_window,
    composite_clean,
)


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
OUTPUT = ATTEMPT / "lanes/scene_evidence/attachment_clean_canary_v1"
TRACK_ROOT = ATTEMPT / "lanes/scene_evidence/attachment_track_canary_v1"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene"
R2_SCENE = REPO / (
    "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/"
    "attempt_0001/lanes/lane1_scene"
)
MODEL_SIZE = (960, 720)
SPECS = {
    "play_cards_0915_031": {
        "short": "031", "start": 66, "stop": 82, "diag": V5 / "DIAG_031.json",
        "baseline": R2_SCENE / "clean_candidate_031_wave5/clean",
    },
    "get_potato_chips_0915_007": {
        "short": "007", "start": 181, "stop": 197, "diag": V5 / "DIAG_007.json",
        "baseline": R2_SCENE / "clean_candidate_007_wave4/clean",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def image(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"IMAGE_WRITE:{path}")


def decode_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path)); count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def prepare(session_id: str, spec: dict) -> tuple[Path, dict]:
    root = OUTPUT / session_id
    diag = load_json(spec["diag"])
    domain = load_json(Path(diag["domain_manifest"]))
    manifests = [load_json(Path(path)) for path in diag["mask_manifests"]]
    start, stop = int(spec["start"]), int(spec["stop"])
    if domain["session_id"] != session_id or not (0 <= start < stop <= int(domain["frame_count"])):
        raise RuntimeError("SESSION_OR_WINDOW_DRIFT")
    shape = (int(domain["height"]), int(domain["width"]))
    roles: list[FrameRoles] = []
    for frame_id in range(start, stop):
        base = merge_roles(_mask_files(manifests, frame_id, shape), shape)
        attachment_path = TRACK_ROOT / session_id / "candidate_masks" / f"{frame_id:06d}.png"
        attachment = cv2.imread(str(attachment_path), cv2.IMREAD_UNCHANGED)
        if attachment is None or attachment.shape != shape:
            raise RuntimeError(f"ATTACHMENT_MASK:{frame_id}")
        roles.append(FrameRoles(base.human, base.device | (attachment > 0), base.object_visible))

    for name in ("frames", "model_masks", "write", "protect", "unknown", "clean"):
        (root / name).mkdir(parents=True, exist_ok=False)
    rows = []
    device_covered = 0
    device_total = 0
    for local_id, frame_id in enumerate(range(start, stop)):
        left, right = max(0, local_id - 2), min(len(roles), local_id + 3)
        masks = build_object_protected_repair_window(
            roles[left:right], local_id - left, frame_id, support_margin=4,
        )
        raw_path = Path(domain["frames"][frame_id]["rgb"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != shape:
            raise RuntimeError(f"RAW_FRAME:{frame_id}")
        model_context = cv2.dilate(
            np.asarray(masks["context_exclude"], np.uint8), np.ones((3, 3), np.uint8),
        )
        small = cv2.resize(model_context, MODEL_SIZE, interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        restored = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.any(np.asarray(masks["context_exclude"], bool) & ~restored):
            raise RuntimeError(f"MODEL_CONTEXT_LOSS:{frame_id}")
        attachment = cv2.imread(
            str(TRACK_ROOT / session_id / "candidate_masks" / f"{frame_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        ) > 0
        device_total += int(attachment.sum())
        device_covered += int((attachment & np.asarray(masks["write"], bool)).sum())
        image(root / "frames" / f"{local_id:06d}.png", cv2.resize(raw, MODEL_SIZE, interpolation=cv2.INTER_AREA))
        image(root / "model_masks" / f"{local_id:06d}.png", small * 255)
        for name in ("write", "protect", "unknown"):
            image(root / name / f"{local_id:06d}.png", np.asarray(masks[name], np.uint8) * 255)
        rows.append({
            "local_frame_id": local_id, "source_frame_id": frame_id,
            "raw": str(raw_path.resolve()), "stats": masks["stats"],
        })
    if device_total <= 0 or device_covered != device_total:
        raise RuntimeError(f"DEVICE_NOT_FULLY_CONSUMED:{device_covered}:{device_total}")
    manifest = {
        "schema_version": "S1_ATTACHMENT_CLEAN_CANARY_PREP_V1",
        "task_id": TASK, "session_id": session_id, "created_at": now(),
        "source_frame_range_half_open": [start, stop], "frame_count": stop - start,
        "image_domain": domain["image_domain"], "model_size_wh": list(MODEL_SIZE),
        "roles": "semantic human/forearm + local attachment track; direct visible object interior protected",
        "attachment_pixels": device_total, "attachment_pixels_in_write": device_covered,
        "attachment_consumption_fraction": device_covered / device_total,
        "M_context_exclude_consumed_by_model": True,
        "full_session_authority": False, "rows": rows,
        "inputs": {
            "domain": artifact_ref(Path(diag["domain_manifest"])),
            "role_manifests": [artifact_ref(Path(path)) for path in diag["mask_manifests"]],
            "attachment_track": artifact_ref(TRACK_ROOT / session_id / "RESULT.json"),
        },
    }
    write_once(root / "PREP_MANIFEST.json", manifest)
    return root, manifest


def run_model(root: Path, manifest: dict) -> dict:
    vendor = REPO / "vendor/ProPainter"
    upstream = root / "upstream"
    command = [
        sys.executable, "-B", str(vendor / "inference_propainter.py"),
        "--video", str(root / "frames"), "--mask", str(root / "model_masks"),
        "--output", str(upstream), "--width", "960", "--height", "720",
        "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
        "--subvideo_length", "40", "--raft_iter", "20", "--save_fps", "30",
        "--save_frames", "--fp16",
    ]
    invocation = {
        "command": command, "started_at": now(),
        "weights": [artifact_ref(vendor / "weights" / name) for name in (
            "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth",
        )],
        "vendor_license": artifact_ref(vendor / "LICENSE"),
        "gpu_lease": artifact_ref(REPO / "_run/current/GPU_LEASE.json"),
    }
    write_once(root / "INVOCATION.json", invocation)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    log_path = root / "PROPAINTER.log"
    with log_path.open("xb") as log:
        completed = subprocess.run(command, cwd=vendor, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT:{completed.returncode}:{log_path}")
    generated = sorted((upstream / "frames/frames").glob("*.png"))
    if len(generated) != manifest["frame_count"]:
        raise RuntimeError(f"MODEL_FRAME_COUNT:{len(generated)}")
    return {"returncode": completed.returncode, "generated": generated, "log": log_path}


def finalize(session_id: str, spec: dict, root: Path, manifest: dict, runtime: dict) -> dict:
    baseline_root = Path(spec["baseline"])
    changed_attachment = attachment_total = 0
    rows = []
    panels = []
    for row, generated_path in zip(manifest["rows"], runtime["generated"], strict=True):
        local_id, frame_id = row["local_frame_id"], row["source_frame_id"]
        raw = cv2.imread(row["raw"], cv2.IMREAD_COLOR)
        generated = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        baseline = cv2.imread(str(baseline_root / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
        write = cv2.imread(str(root / "write" / f"{local_id:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        protect = cv2.imread(str(root / "protect" / f"{local_id:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        unknown = cv2.imread(str(root / "unknown" / f"{local_id:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        attachment = cv2.imread(
            str(TRACK_ROOT / session_id / "candidate_masks" / f"{frame_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        ) > 0
        if raw is None or generated is None or baseline is None:
            raise RuntimeError(f"FINALIZE_DECODE:{frame_id}")
        generated = cv2.resize(generated, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        clean = composite_clean(raw, generated, write, protect)
        image(root / "clean" / f"{local_id:06d}.png", clean)
        changed = np.any(clean != raw, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError(f"WRITE_OR_PROTECT_VIOLATION:{frame_id}")
        attachment_total += int(attachment.sum())
        changed_attachment += int((attachment & changed).sum())
        rows.append({
            "local_frame_id": local_id, "source_frame_id": frame_id,
            "changed_px": int(changed.sum()),
            "attachment_px": int(attachment.sum()),
            "attachment_changed_px": int((attachment & changed).sum()),
            "outside_write_changed_px": int((changed & ~write).sum()),
            "protected_changed_px": int((changed & protect).sum()),
        })
        raw_s = cv2.resize(raw, (480, 360), interpolation=cv2.INTER_AREA)
        baseline_s = cv2.resize(baseline, (480, 360), interpolation=cv2.INTER_AREA)
        clean_s = cv2.resize(clean, (480, 360), interpolation=cv2.INTER_AREA)
        overlay = raw.copy(); overlay[write] = (0, 0, 255); overlay[protect] = (0, 255, 0); overlay[unknown] = (0, 255, 255)
        overlay_s = cv2.resize(overlay, (480, 360), interpolation=cv2.INTER_AREA)
        panel = np.concatenate((raw_s, overlay_s, baseline_s, clean_s), axis=1)
        cv2.rectangle(panel, (0, 0), (1920, 50), (0,0,0), -1)
        for x, text in ((8,"RAW"),(488,"WRITE/PROTECT"),(968,"OLD CLEAN"),(1448,"S1 LOCAL CLEAN")):
            cv2.putText(panel, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"source frame {frame_id} | local evidence canary only", (8, 350), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
        panels.append(panel)
    video = root / "ATTACHMENT_CLEAN_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (1920, 360))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER")
    for panel in panels:
        writer.write(panel)
    writer.release()
    decoded = decode_count(video)
    if decoded != manifest["frame_count"]:
        raise RuntimeError(f"VIDEO_DECODE:{decoded}")
    result = {
        "schema_version": "S1_ATTACHMENT_CLEAN_CANARY_RESULT_V1",
        "task_id": TASK, "session_id": session_id, "created_at": now(),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW", "adoption": "CANDIDATE_ONLY",
        "frame_count": manifest["frame_count"],
        "attachment_consumption_fraction": manifest["attachment_consumption_fraction"],
        "attachment_pixel_change_fraction": changed_attachment / attachment_total,
        "outside_write_changed_px": sum(row["outside_write_changed_px"] for row in rows),
        "protected_changed_px": sum(row["protected_changed_px"] for row in rows),
        "rows": rows, "prep": artifact_ref(root / "PREP_MANIFEST.json"),
        "invocation": artifact_ref(root / "INVOCATION.json"),
        "log": artifact_ref(runtime["log"]),
        "review_video": {**artifact_ref(video), "decoded_frames": decoded},
        "full_session_clean_authority": False, "geometry_input_allowed": False,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    write_once(root / "RESULT.json", result)
    return result


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    results = {}
    for session_id, spec in SPECS.items():
        root, manifest = prepare(session_id, spec)
        runtime = run_model(root, manifest)
        results[session_id] = finalize(session_id, spec, root, manifest, runtime)
    summary = {
        "schema_version": "S1_ATTACHMENT_CLEAN_CANARY_SUMMARY_V1",
        "task_id": TASK, "created_at": now(),
        "status": "EXECUTED_PENDING_INDEPENDENT_VISUAL_REVIEW",
        "sessions": {
            session: {"result": artifact_ref(OUTPUT / session / "RESULT.json"),
                      "review_video": value["review_video"]}
            for session, value in results.items()
        },
        "full_session_clean_authority": False,
    }
    write_once(OUTPUT / "RESULT.json", summary)
    print(json.dumps({session: {"quality": value["quality"], "attachment_pixel_change_fraction": value["attachment_pixel_change_fraction"]} for session, value in results.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
