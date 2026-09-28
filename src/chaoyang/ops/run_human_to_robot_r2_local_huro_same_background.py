#!/usr/bin/env python3
"""Render Local and HuRo on one rejected Clean background for diagnosis.

The output is not the final same-background comparison because the Scene
candidate failed quality and registered real-session occlusion is unavailable.
It nevertheless closes the renderer-consumption question with identical
background, camera domain, assets and frame denominator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, REPO, TASK, ref, write_json
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite


CASES = {
    "031": {
        "session": "play_cards_0915_031", "count": 149, "scene_wave": "wave5",
        "local": ATTEMPT / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz",
    },
    "007": {
        "session": "get_potato_chips_0915_007", "count": 378, "scene_wave": "wave4",
        "local": ATTEMPT / "lanes/lane2_motion/recovered_007_wave0/ROBOT_R0_V1.npz",
    },
}
HURO_ROOT = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/full_0001"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def run(short: str) -> Path:
    case = CASES[short]
    output = ATTEMPT / f"lanes/lane4_compare/same_rejected_background_{short}_wave10"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return result_path
    if output.exists():
        raise FileExistsError(f"PARTIAL_COMPARE:{output}")
    scene_root = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{case['scene_wave']}"
    scene_path, prep_path = scene_root / "RESULT.json", scene_root / "SCENE_PREP_MANIFEST.json"
    scene, prep = load(scene_path), load(prep_path)
    domain_path = Path(prep["source_domain"]["path"])
    domain = load(domain_path)
    local_path = Path(case["local"])
    huro_path = HURO_ROOT / case["session"] / "HURO_CORE_V1.npz"
    local, huro = load_npz(local_path), load_npz(huro_path)
    count = int(case["count"])
    if scene.get("quality") != "REJECTED_QUALITY" or len(scene.get("rows", [])) != count:
        raise ValueError("COMPARE_SCENE_NOT_REJECTED_STRUCTURAL_CANDIDATE")
    for label, value in (("local", local), ("huro", huro)):
        if len(value["frame_id"]) != count or not np.array_equal(value["frame_id"], np.arange(count)):
            raise ValueError(f"COMPARE_FRAME_ID:{label}")
    local_valid = np.asarray(local["wrist_valid"] & local["finger_valid"], dtype=np.bool_)
    huro_valid = np.asarray(huro["wrist_valid"] & huro["finger_valid"], dtype=np.bool_)
    common = local_valid & huro_valid
    output.mkdir(parents=True)
    width, height = int(domain["width"]), int(domain["height"])
    panel_w, panel_h = 640, 480
    video = output / "LOCAL_R0_VS_HURO_SAME_REJECTED_BACKGROUND.mp4"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{panel_w * 2}x{panel_h}", "-r", "30",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderers = [ProductRobotRenderer(REPO, local, domain), ProductRobotRenderer(REPO, huro, domain)]
    changed_pixels = []
    try:
        for index, row in enumerate(scene["rows"]):
            clean = cv2.imread(row["clean"]["path"], cv2.IMREAD_COLOR)
            if clean is None or clean.shape[:2] != (height, width):
                raise ValueError(f"COMPARE_CLEAN_DECODE:{index}")
            panels = []
            masks = []
            for method, renderer in zip(("LOCAL R0", "HURO"), renderers, strict=True):
                robot_rgb, robot_mask = renderer.frame(index)
                image = composite(clean, robot_rgb, robot_mask)
                image = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 0), (panel_w, 58), (18, 18, 18), -1)
                cv2.putText(image, f"{method} | frame {index:04d}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "DIAGNOSTIC: CLEAN REJECTED / OCCLUSION UNKNOWN", (10, 49),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 210, 255), 1, cv2.LINE_AA)
                panels.append(image); masks.append(robot_mask)
            changed_pixels.append(int(np.count_nonzero(masks[0] ^ masks[1])))
            canvas = np.concatenate(panels, axis=1)
            try:
                writer.stdin.write(canvas.tobytes())
            except BrokenPipeError as error:
                stderr = writer.stderr.read().decode("utf-8", errors="replace")
                code = writer.wait()
                raise RuntimeError(f"COMPARE_FFMPEG_PIPE:{code}:{stderr[-4000:]}") from error
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace")
        code = writer.wait()
        if code:
            raise RuntimeError(f"COMPARE_FFMPEG:{code}:{stderr[-4000:]}")
    finally:
        for renderer in renderers:
            renderer.close()
        if writer.poll() is None:
            writer.terminate(); writer.wait()
    capture = cv2.VideoCapture(str(video)); decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != count:
        raise ValueError(f"COMPARE_DECODE:{decoded}:{count}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_LOCAL_HURO_SAME_REJECTED_BACKGROUND_V1",
        "task_id": TASK,
        "session_id": case["session"],
        "created_at": now(),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE_REJECTED_BACKGROUND",
        "adoption": "NOT_ADOPTED",
        "frame_count": count,
        "decoded_frames": decoded,
        "common_valid_side_frames": common.sum(axis=0).tolist(),
        "robot_mask_difference_p50_pixels": float(np.median(changed_pixels)),
        "same_background": True,
        "same_camera_domain": True,
        "same_robot_assets": True,
        "occlusion": "UNKNOWN",
        "scene": ref(scene_path),
        "domain": ref(domain_path),
        "local": ref(local_path),
        "huro": ref(huro_path),
        "video": ref(video),
        "winner": None,
        "claim_limit": "Diagnostic same rejected background only; not final Clean comparison and not evidence of method superiority.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=tuple(CASES), required=True)
    args = parser.parse_args()
    print(run(args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
