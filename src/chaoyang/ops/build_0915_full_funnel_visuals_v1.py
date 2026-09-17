#!/usr/bin/env python3
"""Build shallow, reviewable 0915 full-funnel visuals from terminal evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.pipeline.full_funnel_ledger_v1 import STAGE_ORDER, validate


ROOT = Path(__file__).resolve().parents[3]
POST = ROOT / "_run/current/0915_post_geometry_robot_v1/attempts/attempt_0001"
PREPARED = ROOT / "_run/current/0915_input_prepare_cad_v1/attempts/attempt_0001/prepared_physical_left"
HAWOR = ROOT / "_run/current/0915_hawor_full_v1/attempts/attempt_0001/hawor"
MASK = ROOT / "_run/current/0915_sam31_mask_full_v1/attempts/attempt_0001/sam31"
DEPTH = ROOT / "_run/current/0915_foundationstereo_full_v1/attempts/attempt_0001/depth"
DEFAULT_OUTPUT = ROOT / "docs/current/visuals/0915_FULL_FUNNEL_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(relative_to.resolve())),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def draw_funnel(ledger: dict[str, Any]) -> np.ndarray:
    canvas = np.full((980, 1500, 3), 247, np.uint8)
    cv2.putText(canvas, "0915 PHYSICAL-LEFT FULL FUNNEL (N=220)", (55, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.35, (25, 25, 25), 3, cv2.LINE_AA)
    colors = {"PASS": (80, 170, 70), "REJECTED_QUALITY": (40, 150, 235),
              "BLOCKED_UPSTREAM": (170, 120, 70), "BLOCKED_EXTERNAL": (170, 90, 170),
              "FAILED_RUNTIME": (40, 40, 220), "NOT_RUN": (150, 150, 150)}
    for index, stage in enumerate(STAGE_ORDER):
        y = 120 + index * 88
        counts = ledger["summary"][stage]
        cv2.putText(canvas, stage, (45, y + 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (25, 25, 25), 2, cv2.LINE_AA)
        x = 270
        for status in colors:
            count = int(counts.get(status, 0))
            width = int(round(1080 * count / 220))
            if width:
                cv2.rectangle(canvas, (x, y), (x + width, y + 48), colors[status], -1)
                if width >= 55:
                    cv2.putText(canvas, str(count), (x + 8, y + 33),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255),
                                2, cv2.LINE_AA)
            x += width
    y = 920
    x = 55
    for status, color in colors.items():
        cv2.rectangle(canvas, (x, y), (x + 26, y + 26), color, -1)
        cv2.putText(canvas, status, (x + 34, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (35, 35, 35), 1, cv2.LINE_AA)
        x += 225
    return canvas


def draw_blockers(ledger: dict[str, Any]) -> np.ndarray:
    blockers = Counter(row["first_blocker"] or "NONE_ALL_PASS" for row in ledger["sessions"])
    rows = sorted(blockers.items(), key=lambda item: (-item[1], item[0]))
    canvas = np.full((760, 1280, 3), 248, np.uint8)
    cv2.putText(canvas, "FIRST BLOCKER WALL", (45, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (25, 25, 25), 3, cv2.LINE_AA)
    maximum = max(blockers.values(), default=1)
    for index, (name, count) in enumerate(rows):
        y = 115 + index * 64
        cv2.putText(canvas, name, (45, y + 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.63, (30, 30, 30), 2, cv2.LINE_AA)
        width = int(790 * count / maximum)
        cv2.rectangle(canvas, (360, y), (360 + width, y + 38), (60, 105, 210), -1)
        cv2.putText(canvas, str(count), (370 + width, y + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def select_representatives(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in ("playing_cards", "potato_chips"):
        rows = [
            row for row in ledger["sessions"]
            if row["task"] == task
            and all(row["stages"][stage]["status"] == "PASS"
                    for stage in ("HaWoR", "Mask", "Depth", "Clean"))
        ]
        rows.sort(key=lambda row: (
            -sum(row["stages"][stage]["status"] == "PASS" for stage in STAGE_ORDER),
            row["session_id"],
        ))
        selected.extend(rows[:2])
    return selected


def _mask_union(root: Path, manifest: dict[str, Any], frame: int,
                roles: set[str]) -> np.ndarray:
    union = np.zeros((960, 1280), bool)
    for instance in manifest["instances"]:
        if instance["role"] not in roles:
            continue
        path = root / instance["mask_directory"] / f"{frame:05d}.png"
        if path.is_file():
            value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if value is None or value.shape != union.shape:
                raise RuntimeError(f"mask decode drift: {path}")
            union |= value > 0
    return union


def _load_contact_counts(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[int(row["frame"])] = int(row.get("supported_count", 0))
    return result


def build_clip(row: dict[str, Any], target: Path) -> dict[str, Any]:
    task, session = row["task"], row["session_id"]
    prepared_result = load(PREPARED / "sessions" / task / session / "RESULT.json")
    video = PREPARED / "sessions" / task / session / prepared_result["output"]["video_relative"]
    hawor_path = HAWOR / "sessions" / task / session / "HAWOR_RAW_MANO21.npz"
    mask_root = MASK / "sessions" / task / session
    manifest = load(mask_root / "ROLE_MANIFEST.json")
    post_root = POST / "sessions" / task / session
    with np.load(hawor_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    with np.load(post_root / "CLEAN_INVALID_MASKS.npz", allow_pickle=False) as archive:
        packed = np.asarray(archive["packed_invalid"], np.uint8)
    contacts = _load_contact_counts(post_root / "CONTACT_HYPOTHESES.jsonl")
    robot = load(post_root / "ROBOT_VISUAL.json")
    frame_count = int(prepared_result["frame_count"])
    length = min(150, frame_count)
    start = max(0, frame_count // 2 - length // 2)
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                             (1280, 720))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("review clip reader/writer failed")
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
              (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    try:
        for frame in range(start, start + length):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"clip decode ended at {frame}")
            human = _mask_union(mask_root, manifest, frame, set(manifest["roles"]) - {"task_object"})
            objects = _mask_union(mask_root, manifest, frame, {"task_object"})
            overlay = rgb.copy()
            overlay[human] = (0.35 * overlay[human] + 0.65 * np.asarray((70, 70, 240))).astype(np.uint8)
            overlay[objects] = (0.35 * overlay[objects] + 0.65 * np.asarray((70, 220, 70))).astype(np.uint8)
            for side, color in ((0, (255, 220, 40)), (1, (255, 80, 220))):
                if not observed[side, frame]:
                    continue
                for chain in chains:
                    points = np.rint(joints[side, frame, np.asarray(chain)]).astype(np.int32)
                    cv2.polylines(overlay, [points], False, color, 3, cv2.LINE_AA)
            invalid = np.unpackbits(packed[frame])[:960 * 1280].reshape(960, 1280).astype(bool)
            clean = rgb.copy()
            clean[invalid] = 0
            depth_path = DEPTH / "sessions" / task / session / "frames" / f"{frame:06d}.npz"
            with np.load(depth_path, allow_pickle=False) as depth_archive:
                depth = np.asarray(depth_archive["depth_m"], np.float32)
                valid = np.asarray(depth_archive["valid"], bool)
            normalized = np.zeros(depth.shape, np.uint8)
            if valid.any():
                low, high = np.nanpercentile(depth[valid], (5, 95))
                normalized[valid] = np.clip((depth[valid] - low) / max(high - low, 1e-6) * 255, 0, 255)
            depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            depth_color[~valid] = 0
            panels = [overlay, clean, cv2.resize(depth_color, (1280, 960)), rgb]
            panels = [cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA) for panel in panels]
            canvas = np.concatenate((np.concatenate(panels[:2], axis=1),
                                     np.concatenate(panels[2:], axis=1)), axis=0)
            cv2.rectangle(canvas, (0, 0), (1280, 48), (0, 0, 0), -1)
            cv2.putText(canvas,
                        f"{session} f{frame:05d} | SAM3.1 | tactile-contact={contacts.get(frame, 0)} | Robot={robot['status']}",
                        (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        capture.release()
        writer.release()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
        "-of", "json", str(target),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise RuntimeError(f"review clip full decode failed: {decoded.stderr[-1000:]}")
    return {
        **ref(target, target.parent), "session_id": session, "task": task,
        "source_start_frame": start, "decoded_frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]), "height": int(stream["height"]),
        "full_decode": "PASS_FFMPEG_XERROR",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=POST / "FULL_FUNNEL_LEDGER.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger_path = args.ledger.resolve(strict=True)
    ledger = load(ledger_path)
    validate(ledger)
    if any(row["stages"][stage]["status"] == "NOT_RUN"
           for row in ledger["sessions"] for stage in STAGE_ORDER):
        raise RuntimeError("terminal 220-session ledger required")
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh shallow output required: {output}")
    output.mkdir(parents=True)
    funnel = output / "FUNNEL_220.png"
    blockers = output / "FIRST_BLOCKER_WALL.png"
    if not cv2.imwrite(str(funnel), draw_funnel(ledger)):
        raise RuntimeError("failed to write funnel")
    if not cv2.imwrite(str(blockers), draw_blockers(ledger)):
        raise RuntimeError("failed to write blocker wall")
    clips = []
    for row in select_representatives(ledger):
        target = output / f"{row['session_id']}_REVIEW.mp4"
        clips.append(build_clip(row, target))
    summary = {
        "schema_version": "0915-full-funnel-shallow-visuals-v1",
        "fixed_denominator": 220, "ledger_sha256": sha256(ledger_path),
        "summary": ledger["summary"], "representative_clips": clips,
        "files": [ref(funnel, output), ref(blockers, output)],
        "mask_model_policy": "SAM3.1_ONLY_USER_LOCKED",
        "claim_limit": "Shallow review copies only; numeric authority remains in the terminal ledger and stage receipts.",
    }
    atomic_json(output / "MANIFEST.json", summary)
    lines = [
        "# 0915 裸手全链浅层可视化 V1", "",
        "固定分母为 220；Mask 仅使用已确定的 SAM3.1。", "",
        "- `FUNNEL_220.png`：全阶段终态漏斗。",
        "- `FIRST_BLOCKER_WALL.png`：每个会话的首阻塞阶段。",
        "- `*_REVIEW.mp4`：物理左目 RGB、HaWoR/SAM3.1、Clean、Depth、触觉接触计数与 Robot Visual 状态的代表性短片。",
        "- `MANIFEST.json`：视频完整解码、字节数与 SHA 收据。", "",
        "这些图像和视频只用于审阅；不构成毫米精度、接触真值、控制真值或真机部署授权。", "",
    ]
    (output / "README_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    # Rebind the final README after it exists without recursively hashing the manifest.
    summary["files"].append(ref(output / "README_ZH.md", output))
    atomic_json(output / "MANIFEST.json", summary)
    print(json.dumps({"status": "PASS", "output": str(output),
                      "clips": len(clips)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
