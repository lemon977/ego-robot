#!/usr/bin/env python3
"""Render the terminal, fail-closed 0915 left-mono pipeline review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def draw_hawor(image: np.ndarray, points: np.ndarray, observed: np.ndarray) -> np.ndarray:
    output = image.copy()
    for side, color in ((0, (255, 190, 30)), (1, (20, 50, 255))):
        if not observed[side]:
            continue
        uv = np.rint(points[side]).astype(np.int32)
        for chain in CHAINS:
            cv2.polylines(output, [uv[np.asarray(chain)]], False, color, 3, cv2.LINE_AA)
    return output


def tint(image: np.ndarray, role: np.ndarray, objects: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32)
    for label, color in ((1, (255, 180, 20)), (2, (20, 50, 255))):
        mask = role == label
        output[mask] = .4 * output[mask] + .6 * np.asarray(color, np.float32)
    for label, color in ((1, (60, 230, 255)), (2, (50, 255, 80)), (3, (220, 70, 255))):
        mask = objects == label
        output[mask] = .25 * output[mask] + .75 * np.asarray(color, np.float32)
    return output.astype(np.uint8)


def depth_panel(depth_root: Path, frame: int) -> np.ndarray:
    with np.load(depth_root / "frames" / f"{frame:06d}.npz", allow_pickle=False) as payload:
        depth = np.asarray(payload["depth_m"], np.float32)
        valid = np.asarray(payload["valid"], bool)
    scalar = np.zeros(depth.shape, np.uint8)
    scalar[valid] = np.rint(255 * (1 - np.clip((depth[valid] - .1) / 2.9, 0, 1))).astype(np.uint8)
    output = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return cv2.resize(output, (640, 480), interpolation=cv2.INTER_NEAREST)


def status_panel(frame: int, stages: dict[str, Any]) -> np.ndarray:
    output = np.full((480, 640, 3), 25, np.uint8)
    cv2.putText(output, f"0915 left-mono terminal | f{frame:03d}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .70, (245, 245, 245), 2, cv2.LINE_AA)
    y = 75
    for name in ("Raw", "HaWoR", "Role Mask", "Object Mask", "Depth", "Object6D", "Clean", "Contact", "Robot", "HumanEgo"):
        status = str(stages[name]["status"])
        good = status.startswith("PASS")
        color = (70, 215, 90) if good else (40, 170, 255)
        cv2.putText(output, name, (22, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(output, status[:48], (170, y), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        y += 36
    cv2.putText(output, "No downstream artifact is fabricated across a failed gate.", (20, 455), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    video = args.video.resolve(strict=True)
    hawor_path = args.hawor_npz.resolve(strict=True)
    mask_root = args.mask_root.resolve(strict=True)
    depth_root = args.depth_root.resolve(strict=True)
    output = args.output_root.resolve()
    mask = json.loads((mask_root / "RESULT.json").read_text())
    depth = json.loads((depth_root / "RESULT.json").read_text())
    with np.load(hawor_path, allow_pickle=False) as payload:
        joints = np.asarray(payload["joints_2d"], np.float32)
        observed = np.asarray(payload["observed"], bool)
        fps = float(payload["fps"])
    object_counts = mask["task_object_masks"]["present_frames_by_id"]
    stages = {
        "Raw": {"status": "PASS_379_PHYSICAL_LEFT"},
        "HaWoR": {"status": "PASS_DEVELOPMENT", "right_observed": int(observed[1].sum()), "left_observed": int(observed[0].sum())},
        "Role Mask": {"status": "PASS_DEVELOPMENT_379_BILATERAL", "tracker": "ABSENT_NOT_FABRICATED"},
        "Object Mask": {"status": "FAILED_QUALITY_C_ID_PROPAGATION", "present_frames_by_id": object_counts, "seed_instances_correct": True},
        "Depth": {"status": depth["status"], "rectification_quality": depth["rectification_quality"], "depth_quality": depth["depth_quality"]},
        "Object6D": {"status": "BLOCKED_PREREQ_OBJECT_MASK"},
        "Clean": {"status": "BLOCKED_PREREQ_OBJECT_PROTECTION_MASK"},
        "Contact": {"status": "BLOCKED_PREREQ_OBJECT6D"},
        "Robot": {"status": "BLOCKED_PREREQ_CONTACT_AND_EXTERNAL_RIG"},
        "HumanEgo": {"status": "BLOCKED_EXTERNAL_REAL_ACTION_AND_ROBOT_RGB"},
    }
    stage_path = output / "PIPELINE_STAGE_MATRIX.json"
    stage_path.write_text(json.dumps(stages, ensure_ascii=False, indent=2) + "\n")
    review = output / "PIPELINE_OVERVIEW.mp4"
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 960))
    capture = cv2.VideoCapture(str(video))
    sample_indices = set(np.linspace(0, 378, 12, dtype=int).tolist())
    samples = []
    started = time.monotonic()
    try:
        for frame in range(379):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decode ended at {frame}")
            role = cv2.imread(str(mask_root / "role" / f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
            objects = cv2.imread(str(mask_root / "object" / f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
            hawor = cv2.resize(draw_hawor(raw, joints[:, frame], observed[:, frame]), (640, 480))
            masks = cv2.resize(tint(raw, role, objects), (640, 480))
            metric = depth_panel(depth_root, frame)
            terminal = status_panel(frame, stages)
            overview = np.vstack((np.hstack((hawor, masks)), np.hstack((metric, terminal))))
            for text, origin in (("HaWoR / physical left mono", (12, 27)), ("SAM3.1: human pass, chips C", (652, 27)), ("FoundationStereo / stereo only", (12, 507)), ("Fail-closed stage terminal", (652, 507))):
                cv2.putText(overview, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(overview)
            if frame in sample_indices:
                samples.append(cv2.resize(overview, (640, 480), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    sheet = output / "PIPELINE_OVERVIEW.png"
    canvas = np.zeros((3 * 480, 4 * 640, 3), np.uint8)
    for index, image in enumerate(samples):
        row, column = divmod(index, 4)
        canvas[row * 480:(row + 1) * 480, column * 640:(column + 1) * 640] = image
    cv2.imwrite(str(sheet), canvas)
    result = {
        "schema_version": "0915-leftmono-terminal-review-v1",
        "status": "TERMINAL_FAILED_QUALITY_C_OBJECT_MASK",
        "session_id": "get_potato_chips_0915_001",
        "stages": stages,
        "inputs": {"video": ref(video), "hawor": ref(hawor_path), "mask": ref(mask_root / "RESULT.json"), "depth": ref(depth_root / "RESULT.json")},
        "artifacts": {"review": ref(review), "overview": ref(sheet), "stage_matrix": ref(stage_path)},
        "wall_seconds": time.monotonic() - started,
        "claim_limit": "Terminal fail-closed development review. Object Mask quality C blocks Object6D/Clean/Contact/Robot/HumanEgo; no blocked-stage artifact is fabricated.",
    }
    (output / "TERMINAL_REVIEW_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "review": result["artifacts"]["review"], "overview": result["artifacts"]["overview"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
