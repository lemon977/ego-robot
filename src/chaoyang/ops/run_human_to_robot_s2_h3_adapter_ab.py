#!/usr/bin/env python3
"""S2 H3 fixed-window A/B: historical renderer versus real adapter consumption."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.pipeline.v5_product import (
    ProductRobotRenderer, ROBOT_COMPONENT_ADAPTER_LEFT,
    ROBOT_COMPONENT_ADAPTER_RIGHT, composite,
)

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
S1 = REPO / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001"
CASES = {
    "031": ("play_cards_0915_031", 66, "recovered_031_wave0", "clean_candidate_031_wave5"),
    "007": ("get_potato_chips_0915_007", 181, "recovered_007_wave0", "clean_candidate_007_wave4"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def label(image: np.ndarray, first: str, second: str) -> np.ndarray:
    value = image.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 56), (18, 18, 18), -1)
    cv2.putText(value, first, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(value, second, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (0, 215, 255), 1, cv2.LINE_AA)
    return value


def run(short: str) -> Path:
    session, start, motion_name, scene_name = CASES[short]
    out = ATTEMPT / f"lanes/motion_product/h3_adapter_ab/{session}/attempt_0002"
    if out.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{out}")
    out.mkdir(parents=True)
    motion_path = R2 / f"lanes/lane2_motion/{motion_name}/ROBOT_R0_V1.npz"
    motion = load_npz(motion_path)
    prep_path = R2 / f"lanes/lane1_scene/{scene_name}/SCENE_PREP_MANIFEST.json"
    prep = load_json(prep_path)
    domain_path = Path(prep["source_domain"]["path"])
    domain = load_json(domain_path)
    clean_root = S1 / f"lanes/scene_evidence/attachment_clean_canary_v1/{session}"
    clean_result_path = clean_root / "RESULT.json"
    clean_result = load_json(clean_result_path)
    rows = clean_result["rows"]
    if [row["source_frame_id"] for row in rows] != list(range(start, start + 16)):
        raise ValueError("FROZEN_WINDOW_DRIFT")
    width, height = int(domain["width"]), int(domain["height"])
    panel_size = (640, 480)
    video = out / "ADAPTER_AB_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
                             (panel_size[0] * 3, panel_size[1]))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_OPEN_FAILED")
    old = ProductRobotRenderer(REPO, motion, domain, include_adapter=False)
    new = ProductRobotRenderer(REPO, motion, domain, include_adapter=True)
    adapter_pixels = []
    transforms = []
    try:
        for local, row in enumerate(rows):
            frame = int(row["source_frame_id"])
            clean = cv2.imread(str(clean_root / "clean" / f"{local:06d}.png"), cv2.IMREAD_COLOR)
            if clean is None or clean.shape[:2] != (height, width):
                raise ValueError(f"CLEAN_DECODE:{frame}")
            a = old.frame_layers(frame)
            b = new.frame_layers(frame)
            old_product = composite(clean, a.rgb, a.alpha)
            new_product = composite(clean, b.rgb, b.alpha)
            adapter = np.isin(b.component_id, [ROBOT_COMPONENT_ADAPTER_LEFT,
                                                ROBOT_COMPONENT_ADAPTER_RIGHT])
            evidence = clean.copy()
            evidence[adapter] = (0, 0, 255)
            adapter_pixels.append(int(adapter.sum()))
            transforms.append({
                "source_frame_id": frame, "adapter_px": int(adapter.sum()),
                "T_world_flange": b.T_world_flange.tolist(),
                "T_world_adapter": b.T_world_adapter.tolist(),
                "T_world_hand": b.T_world_hand.tolist(),
            })
            panels = [
                label(cv2.resize(old_product, panel_size), f"A OLD PRODUCT | frame {frame}",
                      "same q/camera/mount/background | no adapter"),
                label(cv2.resize(new_product, panel_size), f"B REAL ADAPTER | frame {frame}",
                      "same q/camera/mount/background | occlusion still UNKNOWN"),
                label(cv2.resize(evidence, panel_size), f"ADAPTER ID EVIDENCE | frame {frame}",
                      f"adapter pixels={int(adapter.sum())} | collision UNVERIFIED"),
            ]
            writer.write(np.concatenate(panels, axis=1))
    finally:
        old.close(); new.close(); writer.release()
    cap = cv2.VideoCapture(str(video)); decoded = 0
    while cap.read()[0]: decoded += 1
    cap.release()
    if decoded != 16:
        raise ValueError(f"VIDEO_DECODE:{decoded}/16")
    np.savez_compressed(
        out / "RENDER_TRANSFORMS.npz",
        frame_id=np.arange(start, start + 16, dtype=np.int32),
        adapter_pixels=np.asarray(adapter_pixels, dtype=np.int64),
        T_world_flange=np.asarray([row["T_world_flange"] for row in transforms]),
        T_world_adapter=np.asarray([row["T_world_adapter"] for row in transforms]),
        T_world_hand=np.asarray([row["T_world_hand"] for row in transforms]),
    )
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_ADAPTER_AB_V1", "task_id": TASK,
        "session_id": session, "created_at": now(), "execution": "EXECUTED",
        "structure": "PASS", "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW",
        "adoption": "CANDIDATE_ONLY", "frame_count": 16, "decoded_frames": decoded,
        "frozen_variables": ["q", "camera", "mount", "background", "frame_set"],
        "only_change": "REAL_ADAPTER_VISUAL_RGB_DEPTH_ID_CONSUMPTION",
        "adapter_pixel_frames": int(sum(value > 0 for value in adapter_pixels)),
        "adapter_pixel_p50": float(np.median(adapter_pixels)),
        "adapter_collision_status": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
        "occlusion_status": "UNKNOWN_NOT_PART_OF_AB_TEST",
        "motion": sha(motion_path), "domain": sha(domain_path),
        "clean": sha(clean_result_path), "video": sha(video),
        "transforms": sha(out / "RENDER_TRANSFORMS.npz"),
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (out / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(out),
                      "adapter_pixel_frames": result["adapter_pixel_frames"]}, ensure_ascii=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=tuple(CASES), required=True)
    args = parser.parse_args()
    run(args.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
