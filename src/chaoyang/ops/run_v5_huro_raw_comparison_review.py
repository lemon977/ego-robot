#!/usr/bin/env python3
"""同原片、同资产/相机/渲染器的 R0 对 HuRo 全片诊断（非产品）。

用法：chaoyang run run_v5_huro_raw_comparison_review \
  --huro-result <本会话HuRo RESULT.json> --output <新的V5 huro lane子目录>
此入口绝不读取 Clean，也不生成 robot.mp4 产品文件；画面永久标记 RAW_DIAGNOSTIC。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite

ROUTE = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/ROUTE_MANIFEST.json"
LANE = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro"


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(item: dict) -> Path:
    path = Path(item["path"])
    if ref(path) != item:
        raise ValueError(f"SHA_BINDING_MISMATCH:{path}")
    return path


def load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def annotate(frame: np.ndarray, heading: str, subline: str) -> np.ndarray:
    small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    panel = np.full((550, 640, 3), 22, np.uint8)
    panel[70:] = small
    cv2.putText(panel, heading, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, subline, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (85, 180, 255), 1, cv2.LINE_AA)
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huro-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.parent != LANE.resolve(strict=True) or output.exists():
        raise ValueError("OUTPUT_NOT_NEW_CHILD_OF_V5_HURO_LANE")
    result_path = args.huro_result.resolve(strict=True)
    if LANE.resolve(strict=True) not in result_path.parents:
        raise ValueError("HURO_RESULT_OUTSIDE_V5_LANE")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    sid = result["session_id"]
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    candidates = [row for row in route["sessions"] if row["session_id"] == sid]
    if len(candidates) != 1 or sid not in ("get_potato_chips_0915_007", "play_cards_0915_031"):
        raise ValueError("HURO_ROUTE_SESSION_MISMATCH")
    domain_path = checked(candidates[0]["domain_manifest"])
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    r0_path, huro_path = checked(result["R0"]), checked(result["core_motion"])
    r0, huro = load_motion(r0_path), load_motion(huro_path)
    count = candidates[0]["frame_count"]
    if len(domain["frames"]) != count or len(r0["frame_id"]) != count or len(huro["frame_id"]) != count:
        raise ValueError("FULL_SESSION_FRAME_COUNT_MISMATCH")
    for field in ("frame_id", "timestamp_ns", "wrist_valid", "finger_valid", "human_to_physical"):
        if not np.array_equal(r0[field], huro[field]):
            raise ValueError(f"HURO_FAIR_INPUT_MISMATCH:{field}")
    for field in ("T_cam_base", "T_flange_hand"):
        if not np.allclose(r0[field], huro[field], atol=0, rtol=0):
            raise ValueError(f"HURO_FAIR_GEOMETRY_MISMATCH:{field}")
    if args.dry_run:
        print(json.dumps({"status": "READY_DIAGNOSTIC", "session_id": sid,
                          "frames": count, "valid_side_frames": r0["wrist_valid"].sum(axis=0).tolist()}))
        return 0
    output.mkdir()
    video = output / "R0_VS_HURO_RAW_DIAGNOSTIC.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x550", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", str(video)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    render_r0 = ProductRobotRenderer(REPO_ROOT, r0, domain)
    render_huro = ProductRobotRenderer(REPO_ROOT, huro, domain)
    try:
        for i, item in enumerate(domain["frames"]):
            raw = cv2.imread(item["rgb"], cv2.IMREAD_COLOR)
            if raw is None or raw.shape[:2] != (domain["height"], domain["width"]):
                raise ValueError(f"RAW_FRAME_DECODE_OR_DOMAIN:{i}")
            r_rgb, r_mask = render_r0.frame(i)
            h_rgb, h_mask = render_huro.frame(i)
            panels = [
                annotate(raw, f"{sid} | frame {i}", "ORIGINAL RGB / same source frame"),
                annotate(composite(raw, r_rgb, r_mask), "LOCAL R0 / legacy snapshot",
                         f"valid L/R={r0['wrist_valid'][i].astype(int).tolist()} | RAW DIAGNOSTIC"),
                annotate(composite(raw, h_rgb, h_mask), "HuRo core wrist objective",
                         "FAILED_QUALITY_LIMITS | RAW DIAGNOSTIC"),
            ]
            writer.stdin.write(np.concatenate(panels, axis=1).tobytes())
            if i in (0, count // 2, count - 1):
                cv2.imwrite(str(output / f"preview_{i:06d}.png"), np.concatenate(panels, axis=1))
        writer.stdin.close()
        error = writer.stderr.read().decode("utf-8", "replace")
        if writer.wait() != 0:
            raise RuntimeError("FFMPEG_FAILED:" + error[-2000:])
    finally:
        render_r0.close()
        render_huro.close()
        if writer.poll() is None:
            writer.terminate()
            writer.wait()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != count:
        raise ValueError(f"FULL_VIDEO_DECODE_MISMATCH:{decoded}/{count}")
    receipt = {
        "schema_version": "chaoyang-v5-huro-raw-review-v1", "status": "EXECUTED_DIAGNOSTIC_ONLY",
        "session_id": sid, "decoded_frames": decoded, "expected_frames": count,
        "video": ref(video), "huro_result": ref(result_path), "domain_manifest": ref(domain_path),
        "r0": ref(r0_path), "huro": ref(huro_path), "same_scene_and_renderer": True,
        "clean_consumed": False, "product_video": False, "quality_pass": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "video": receipt["video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
