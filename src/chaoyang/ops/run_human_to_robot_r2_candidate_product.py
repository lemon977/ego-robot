#!/usr/bin/env python3
"""Render per-session R2 product candidates from rejected-but-structural Clean.

This is intentionally separate from the strict product entry.  It proves the
downstream consumer and per-session join while preserving Scene quality as
REJECTED and adoption as CANDIDATE_ONLY.  It also permits a genuine one-hand
window: an invalid side is absent from rendering rather than filled or held.
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
from chaoyang.pipeline.r2_dependency_signature import build_product_binding
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite


CASES = {
    "031": {
        "session": "play_cards_0915_031", "frames": 149, "scene_wave": "wave5",
        "motion": "recovered_031_wave0",
    },
    "007": {
        "session": "get_potato_chips_0915_007", "frames": 378, "scene_wave": "wave4",
        "motion": "recovered_007_wave0",
    },
    "103": {
        "session": "get_potato_chips_0902_103", "frames": 284, "scene_wave": "wave4",
        "motion": "recovered_0902_103_wave2",
    },
    "042": {
        "session": "play_cards_0902_042", "frames": 171, "scene_wave": "wave4",
        "motion": "recovered_0902_042_wave2",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def product_binding(*, scene_path: Path, robot_path: Path, domain_path: Path) -> dict:
    return build_product_binding(
        files={
            "scene_clean": scene_path,
            "motion_r0": robot_path,
            "camera_domain": domain_path,
            "robot_asset": REPO / "assets/robot/ROBOT_ASSET_PIN.json",
            "renderer_code": REPO / "src/chaoyang/pipeline/v5_product.py",
            "product_entry_code": Path(__file__).resolve(),
        },
        render_config={
            "renderer": "PYBULLET_TINY_RENDERER",
            "invalid_side_policy": "NOT_RENDERED_NO_FILL_NO_HOLD",
            "overlay": "ROBOT_OVER_REJECTED_CLEAN",
            "fps": 30,
        },
        occlusion_status="UNKNOWN_NO_REGISTERED_SCENE_DEPTH",
    )


def run(short: str) -> Path:
    case = CASES[short]
    output = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
    result_path = output / "RESULT.json"
    scene_root = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{case['scene_wave']}"
    scene_path = scene_root / "RESULT.json"
    prep_path = scene_root / "SCENE_PREP_MANIFEST.json"
    robot_path = ATTEMPT / f"lanes/lane2_motion/{case['motion']}/ROBOT_R0_V1.npz"
    if result_path.is_file():
        binding_path = output / "CACHE_BINDING_V2.json"
        if not binding_path.is_file():
            raise ValueError(f"PRODUCT_RESUME_BINDING_REQUIRED:{binding_path}")
        prep = load(prep_path)
        domain_path = Path(prep["source_domain"]["path"])
        expected = load(binding_path)
        current = product_binding(scene_path=scene_path, robot_path=robot_path, domain_path=domain_path)
        if expected.get("signature_sha256") != current["signature_sha256"]:
            raise ValueError("PRODUCT_RESUME_SIGNATURE_MISMATCH")
        result = load(result_path)
        video_ref = result.get("video", {})
        video_path = Path(video_ref.get("path", ""))
        if ref(result_path) != expected.get("product_result") or ref(video_path) != expected.get("product_video"):
            raise ValueError("PRODUCT_RESUME_OUTPUT_DRIFT")
        return result_path
    if output.exists():
        raise FileExistsError(f"PARTIAL_PRODUCT:{output}")
    scene, prep = load(scene_path), load(prep_path)
    motion = load_npz(robot_path)
    domain_path = Path(prep["source_domain"]["path"])
    domain = load(domain_path)
    count = case["frames"]
    if scene.get("session_id") != case["session"] or prep.get("session_id") != case["session"]:
        raise ValueError("PRODUCT_SESSION_MISMATCH")
    if scene.get("execution") != "EXECUTED" or scene.get("structure") != "PASS":
        raise ValueError("SCENE_CANDIDATE_STRUCTURE")
    if scene.get("quality") != "REJECTED_QUALITY" or scene.get("adoption") != "CANDIDATE_ONLY":
        raise ValueError("STRICT_PRODUCT_ENTRY_REQUIRED_FOR_ADOPTED_SCENE")
    if len(scene.get("rows", [])) != count or len(motion["frame_id"]) != count:
        raise ValueError("PRODUCT_FRAME_COUNT")
    if not np.array_equal(motion["frame_id"], np.arange(count)):
        raise ValueError("PRODUCT_FRAME_ID")
    output.mkdir(parents=True)
    video = output / "robot_candidate.mp4"
    width, height = int(domain["width"]), int(domain["height"])
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "30",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderer = ProductRobotRenderer(REPO, motion, domain)
    robot_pixels = []
    valid_side_frames = np.asarray(motion["wrist_valid"] & motion["finger_valid"], bool).sum(axis=0)
    try:
        for index, row in enumerate(scene["rows"]):
            clean = cv2.imread(row["clean"]["path"], cv2.IMREAD_COLOR)
            if clean is None or clean.shape[:2] != (height, width):
                raise ValueError(f"PRODUCT_CLEAN_DECODE:{index}")
            robot_rgb, robot_mask = renderer.frame(index)
            mixed = composite(clean, robot_rgb, robot_mask)
            cv2.rectangle(mixed, (0, 0), (width, 42), (20, 20, 20), -1)
            cv2.putText(
                mixed,
                f"{case['session']} frame {index:04d} | CANDIDATE_ONLY | CLEAN REJECTED | UNKNOWN OCCLUSION",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2, cv2.LINE_AA,
            )
            try:
                writer.stdin.write(mixed.tobytes())
            except BrokenPipeError as error:
                stderr = writer.stderr.read().decode("utf-8", errors="replace")
                code = writer.wait()
                raise RuntimeError(f"PRODUCT_FFMPEG_PIPE:{code}:{stderr[-4000:]}") from error
            robot_pixels.append(int(robot_mask.sum()))
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace")
        code = writer.wait()
        if code:
            raise RuntimeError(f"PRODUCT_FFMPEG:{code}:{stderr[-4000:]}")
    finally:
        renderer.close()
        if writer.poll() is None:
            writer.terminate()
            writer.wait()
    capture = cv2.VideoCapture(str(video)); decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != count:
        raise ValueError(f"PRODUCT_DECODE:{decoded}:{count}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PRODUCT_CANDIDATE_V1",
        "task_id": TASK,
        "session_id": case["session"],
        "created_at": now(),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY",
        "frame_count": count,
        "decoded_frames": decoded,
        "valid_side_frames_anatomical_left_right": valid_side_frames.tolist(),
        "invalid_side_policy": "NOT_RENDERED_NO_FILL_NO_HOLD",
        "robot_pixel_p50": float(np.median(robot_pixels)),
        "robot_zero_pixel_frames": int(sum(value == 0 for value in robot_pixels)),
        "occlusion_policy": "ROBOT_OVER_REJECTED_CLEAN_UNKNOWN_DEPTH",
        "scene": ref(scene_path),
        "motion": ref(robot_path),
        "domain": ref(domain_path),
        "video": ref(video),
        "claim_limit": "Offline integration candidate only; rejected Clean and unknown occlusion forbid adoption, geometry, contact or training claims.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    write_json(result_path, result)
    binding = product_binding(scene_path=scene_path, robot_path=robot_path, domain_path=domain_path)
    binding.update({
        "task_id": TASK, "session_id": case["session"], "created_at": now(),
        "product_result": ref(result_path), "product_video": ref(video),
        "resume_policy": "REUSE_ONLY_WHEN_SIGNATURE_AND_OUTPUT_SHA_MATCH",
    })
    write_json(output / "CACHE_BINDING_V2.json", binding)
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
