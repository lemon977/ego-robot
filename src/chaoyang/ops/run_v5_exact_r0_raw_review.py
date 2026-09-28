#!/usr/bin/env python3
"""生成 0902 同域 R0 全片 Raw 诊断，不是去人产品。

用法：chaoyang run run_v5_exact_r0_raw_review \
  --motion-result <V5 recovered_0902_103_v1 或 recovered_0902_042_v1/RESULT.json> \
  --output <V5 motion lane 下的新目录> [--dry-run]

仅消费冻结 0902 exact-domain RGB 与 V5 R0；输出始终带 RAW_DIAGNOSTIC 水印，
不得改名为 robot.mp4、Clean 后产品或质量通过证据。
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

ROOT = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
LANE = ROOT / "lanes/motion"


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(item: dict) -> Path:
    path = Path(item["path"])
    if file_ref(path) != item:
        raise ValueError(f"SHA_BINDING_MISMATCH:{path}")
    return path


def panel(frame: np.ndarray, title: str, sub: str) -> np.ndarray:
    resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    result = np.full((550, 640, 3), 22, np.uint8)
    result[70:] = resized
    cv2.putText(result, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65,
                (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(result, sub, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .41,
                (80, 175, 255), 1, cv2.LINE_AA)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result_path = args.motion_result.resolve(strict=True)
    if LANE.resolve(strict=True) not in result_path.parents:
        raise ValueError("MOTION_RESULT_OUTSIDE_V5_LANE")
    output = args.output.resolve()
    if output.parent != LANE.resolve(strict=True) or output.exists():
        raise ValueError("OUTPUT_NOT_NEW_MOTION_LANE_CHILD")
    motion_result = json.loads(result_path.read_text(encoding="utf-8"))
    sid = motion_result["session_id"]
    if sid not in ("get_potato_chips_0902_103", "play_cards_0902_042"):
        raise ValueError("EXACT0902_ONLY")
    route = json.loads((ROOT / "ROUTE_MANIFEST.json").read_text(encoding="utf-8"))
    matches = [row for row in route["sessions"] if row["session_id"] == sid]
    if len(matches) != 1 or not matches[0]["image_domain"].startswith("EXACT0902_SOURCEINDEX0"):
        raise ValueError("ROUTE_IMAGE_DOMAIN_MISMATCH")
    domain_path = checked(matches[0]["domain_manifest"])
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    r0_path = checked(motion_result["outputs"]["robot_r0"])
    with np.load(r0_path, allow_pickle=False) as archive:
        r0 = {key: np.asarray(archive[key]) for key in archive.files}
    count = matches[0]["frame_count"]
    if len(domain["frames"]) != count or len(r0["frame_id"]) != count:
        raise ValueError("FULL_SESSION_FRAME_COUNT_MISMATCH")
    if args.dry_run:
        print(json.dumps({"status": "READY_DIAGNOSTIC", "session_id": sid,
                          "frames": count, "numeric_quality_pass": motion_result["numeric_quality_pass"]}))
        return 0
    output.mkdir()
    video = output / "R0_RAW_DIAGNOSTIC.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x550", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(video)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderer = ProductRobotRenderer(REPO_ROOT, r0, domain)
    try:
        for i, item in enumerate(domain["frames"]):
            raw = cv2.imread(item["rgb"], cv2.IMREAD_COLOR)
            if raw is None or raw.shape[:2] != (domain["height"], domain["width"]):
                raise ValueError(f"RAW_FRAME_DECODE_OR_DOMAIN:{i}")
            robot_rgb, robot_mask = renderer.frame(i)
            frame = np.concatenate([
                panel(raw, f"{sid} | frame {i}", "EXACT0902 sourceIndex0 / original"),
                panel(composite(raw, robot_rgb, robot_mask), "V5 R0 / RAW DIAGNOSTIC",
                      f"quality=C | valid L/R={r0['wrist_valid'][i].astype(int).tolist()} | NOT PRODUCT"),
            ], axis=1)
            writer.stdin.write(frame.tobytes())
            if i in (0, count // 2, count - 1):
                cv2.imwrite(str(output / f"preview_{i:06d}.png"), frame)
        writer.stdin.close()
        error = writer.stderr.read().decode("utf-8", "replace")
        if writer.wait() != 0:
            raise RuntimeError("FFMPEG_FAILED:" + error[-2000:])
    finally:
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
        raise ValueError(f"FULL_VIDEO_DECODE_MISMATCH:{decoded}/{count}")
    receipt = {
        "schema_version": "chaoyang-v5-exact-r0-raw-review-v1",
        "status": "EXECUTED_DIAGNOSTIC_ONLY", "session_id": sid,
        "decoded_frames": decoded, "expected_frames": count,
        "video": file_ref(video), "motion_result": file_ref(result_path),
        "domain_manifest": file_ref(domain_path), "r0": file_ref(r0_path),
        "clean_consumed": False, "product_video": False,
        "numeric_quality_pass": False, "training_eligible": False,
        "control_ground_truth": False,
    }
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "video": receipt["video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
