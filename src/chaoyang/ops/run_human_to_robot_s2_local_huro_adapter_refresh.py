#!/usr/bin/env python3
"""Re-render frozen Local/HuRo arrays with S2's real adapter renderer.

No solver is invoked.  The rejected Clean is used only as a labelled common
diagnostic background; this run cannot select a winner or upgrade adoption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
HURO = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/full_0001"
CASES = {
    "007": {
        "session": "get_potato_chips_0915_007",
        "binding": ROOT / "bindings/get_potato_chips_0915_007/attempt_0002/BINDING.json",
        "output": ROOT / "lanes/compare/adapter_refresh_007/attempt_0001",
    },
    "031": {
        "session": "play_cards_0915_031",
        "binding": ROOT / "bindings/play_cards_0915_031/BINDING.json",
        "output": ROOT / "lanes/compare/adapter_refresh_031/attempt_0001",
    },
}


def ref(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def run(short: str) -> Path:
    case = CASES[short]
    output = case["output"]
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    binding = json.loads(case["binding"].read_text(encoding="utf-8"))
    if binding.get("execution_round") != "S2":
        raise RuntimeError("S2_BINDING_REQUIRED")
    if binding.get("renderer_interface") != "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1":
        raise RuntimeError("S2_RENDERER_INTERFACE_REQUIRED")
    scene_path = Path(binding["scene_clean_manifest"]["path"])
    local_path = Path(binding["robot_r0"]["path"])
    prep_path = Path(binding["scene_prep_manifest"]["path"])
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    domain_path = Path(prep["source_domain"]["path"])
    huro_path = HURO / case["session"] / "HURO_CORE_V1.npz"
    for path in (scene_path, local_path, domain_path, huro_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    local, huro = load_npz(local_path), load_npz(huro_path)
    count = int(domain["frame_count"])
    if scene.get("quality") != "REJECTED_QUALITY" or len(scene.get("rows", [])) != count:
        raise RuntimeError("REJECTED_STRUCTURAL_CLEAN_REQUIRED")
    for label, arrays in (("local", local), ("huro", huro)):
        if not np.array_equal(arrays["frame_id"], np.arange(count)):
            raise RuntimeError(f"FRAME_DENOMINATOR_MISMATCH:{label}")
        if not np.array_equal(arrays["human_to_physical"], local["human_to_physical"]):
            raise RuntimeError(f"SIDE_MAPPING_MISMATCH:{label}")
        if not np.allclose(arrays["T_flange_hand"], local["T_flange_hand"], atol=1e-10, rtol=0.0):
            raise RuntimeError(f"MOUNT_MISMATCH:{label}")
        if not np.allclose(arrays["T_cam_base"], local["T_cam_base"], atol=1e-10, rtol=0.0):
            raise RuntimeError(f"CAMERA_PLACEMENT_MISMATCH:{label}")
    local_valid = np.asarray(local["wrist_valid"] & local["finger_valid"], dtype=bool)
    huro_valid = np.asarray(huro["wrist_valid"] & huro["finger_valid"], dtype=bool)
    common = local_valid & huro_valid

    output.mkdir(parents=True)
    width, height = int(domain["width"]), int(domain["height"])
    panel_width, panel_height = 640, 480
    video = output / "LOCAL_R0_VS_HURO_S2_ADAPTER_REVIEW.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{panel_width * 2}x{panel_height}",
        "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
        "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderers = [
        ProductRobotRenderer(REPO_ROOT, local, domain, include_adapter=True),
        ProductRobotRenderer(REPO_ROOT, huro, domain, include_adapter=True),
    ]
    adapter_pixels = [[], []]
    robot_mask_difference = []
    try:
        for frame, row in enumerate(scene["rows"]):
            clean = cv2.imread(str(row["clean"]["path"]), cv2.IMREAD_COLOR)
            if clean is None or clean.shape[:2] != (height, width):
                raise RuntimeError(f"CLEAN_DECODE_OR_DOMAIN:{frame}")
            panels, masks = [], []
            for method_index, (method, renderer) in enumerate(
                    zip(("LOCAL R0", "HURO"), renderers, strict=True)):
                layers = renderer.frame_layers(frame)
                image = composite(clean, layers.rgb, layers.alpha)
                adapter_count = int(np.isin(layers.component_id, [2, 3]).sum())
                adapter_pixels[method_index].append(adapter_count)
                masks.append(layers.alpha)
                image = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 0), (panel_width, 66), (18, 18, 18), -1)
                cv2.putText(image, f"{method} | frame {frame:04d}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"real adapter px={adapter_count}; collision UNVERIFIED", (10, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "rejected Clean; occlusion not compared; no winner", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (180, 180, 255), 1, cv2.LINE_AA)
                panels.append(image)
            robot_mask_difference.append(int(np.count_nonzero(masks[0] ^ masks[1])))
            writer.stdin.write(np.concatenate(panels, axis=1).tobytes())
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", "replace")
        code = writer.wait()
        if code:
            raise RuntimeError(f"FFMPEG_FAILED:{code}:{stderr[-2000:]}")
    finally:
        for renderer in renderers:
            renderer.close()
        if writer.poll() is None:
            writer.terminate()
            writer.wait()

    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != count:
        raise RuntimeError(f"VIDEO_DECODE_MISMATCH:{decoded}/{count}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_LOCAL_HURO_ADAPTER_REFRESH_V1",
        "task_id": TASK,
        "session_id": case["session"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED_RENDER_ONLY",
        "structure": "PASS",
        "quality": "INCONCLUSIVE_REJECTED_BACKGROUND",
        "adoption": "NOT_ADOPTED",
        "frame_count": count,
        "decoded_frames": decoded,
        "common_valid_side_frames": common.sum(axis=0).astype(int).tolist(),
        "adapter_visible_frames": {
            "local": int(np.count_nonzero(adapter_pixels[0])),
            "huro": int(np.count_nonzero(adapter_pixels[1])),
        },
        "adapter_pixels_p50": {
            "local": float(np.median(adapter_pixels[0])),
            "huro": float(np.median(adapter_pixels[1])),
        },
        "robot_mask_difference_p50_pixels": float(np.median(robot_mask_difference)),
        "same_background": True,
        "same_camera_domain": True,
        "same_mount": True,
        "same_robot_assets": True,
        "real_adapter_consumed_both_methods": True,
        "adapter_collision_scope": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
        "occlusion_scope": "NOT_COMPARED_REJECTED_BACKGROUND_DIAGNOSTIC",
        "new_solver_invocations": 0,
        "winner": None,
        "inputs": {
            "binding": ref(case["binding"]),
            "scene_prep": ref(prep_path),
            "scene": ref(scene_path),
            "domain": ref(domain_path),
            "local": ref(local_path),
            "huro": ref(huro_path),
        },
        "video": ref(video),
        "claim_limit": (
            "Render-only same rejected background and real-adapter comparison. It does not compare "
            "occlusion, establish a winner, or grant Clean, collision, control, or deployment authority."
        ),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": ref(result_path)}, ensure_ascii=False))
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=tuple(CASES), required=True)
    args = parser.parse_args()
    run(args.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
