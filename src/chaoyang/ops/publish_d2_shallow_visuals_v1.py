#!/usr/bin/env python3
"""Atomically publish the two D2 full-review MP4 files to shallow visuals.

The deep D2 result must already be complete.  This publisher copies real MP4
files (never symlinks) into a same-filesystem sibling staging directory, writes
an index whose refs already name the final directory, and exposes the complete
directory with one ``os.replace``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

try:
    from chaoyang.ops.run_d2_fresh_propainter_offline_v1 import (
        ContractError, decode_count, file_ref, projected_ref,
        publish_directory_atomically, read_json, verify_ref, write_json,
    )
except ImportError:
    from run_d2_fresh_propainter_offline_v1 import (
        ContractError, decode_count, file_ref, projected_ref,
        publish_directory_atomically, read_json, verify_ref, write_json,
    )


EXPECTED = {
    "play_cards_0915_044": 166,
    "get_potato_chips_0915_097": 394,
}
TASK_ID = "0915_robot_quality_recovery_v21_d2_fresh_propainter_offline_v1"
ATTEMPT_ID = "attempt_0001"
CONFIG_PATH = Path(
    "/mnt/workspace/code/chaoyang/contracts/robot_recovery/"
    "D2_FRESH_PROPAINTER_OFFLINE_V1.json"
)


def validate_gpu_receipt(receipt_path: Path, deep_result_path: Path) -> dict[str, Any]:
    if receipt_path.is_symlink():
        raise ContractError("GPU lease receipt must not be a symlink")
    receipt_path = receipt_path.resolve(strict=True)
    deep_result_path = deep_result_path.resolve(strict=True)
    before = file_ref(receipt_path)
    value = read_json(receipt_path)
    if value.get("schema_version") != "v71-gpu-command-receipt-v1":
        raise ContractError("GPU receipt schema is not V7.1 command receipt")
    if value.get("status") != "PASSED":
        raise ContractError("GPU lease receipt status is not PASSED")
    if value.get("task_id") != TASK_ID or value.get("attempt_id") != ATTEMPT_ID:
        raise ContractError("GPU lease receipt task/attempt mismatch")
    expected_command = [
        "python", "-m", "chaoyang.cli", "run",
        "run_d2_fresh_propainter_offline_v1",
        "--config", str(CONFIG_PATH),
        "--output-root", str(deep_result_path.parent),
    ]
    if value.get("command") != expected_command:
        raise ContractError("GPU lease receipt command/deep-result path mismatch")
    after = file_ref(receipt_path)
    if before != after:
        raise ContractError("GPU lease receipt changed during SHA verification")
    return after


def _build(
    deep_result_path: Path, gpu_receipt_path: Path, staging: Path, final: Path
) -> dict[str, Any]:
    deep = read_json(deep_result_path)
    if deep.get("status") != "COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED":
        raise ContractError("deep D2 result is not a completed offline candidate")
    if deep.get("training_eligible") is not False or deep.get("clean_terminal") is not False:
        raise ContractError("deep D2 result exceeded offline authority")
    gpu_receipt = validate_gpu_receipt(gpu_receipt_path, deep_result_path)
    session_results: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(deep.get("sessions", [])):
        path = verify_ref(value, f"deep session result {index}")
        session = read_json(path)
        session_results[session["session_id"]] = session
    if set(session_results) != set(EXPECTED):
        raise ContractError("deep D2 session set mismatch")

    videos = []
    for sid, expected_frames in EXPECTED.items():
        session = session_results[sid]
        if int(session.get("frame_count", -1)) != expected_frames:
            raise ContractError(f"{sid}: deep frame count mismatch")
        source = verify_ref(session["review_video"], f"{sid} review video")
        if source.is_symlink():
            raise ContractError(f"{sid}: deep review video must not be a symlink")
        target = staging / f"{sid}_D2_FRESH_PROPAINTER_OFFLINE_FULL_REVIEW.mp4"
        shutil.copy2(source, target)
        if target.is_symlink() or file_ref(target)["sha256"] != session["review_video"]["sha256"]:
            raise ContractError(f"{sid}: shallow copy is not byte-exact")
        decoded = decode_count(target)
        if decoded != expected_frames:
            raise ContractError(f"{sid}: shallow MP4 decoded {decoded} != {expected_frames}")
        videos.append({
            "session_id": sid,
            "frame_count": expected_frames,
            "fps": 30,
            "video": projected_ref(target, staging, final),
            "label": "OFFLINE_VISUAL / NOT_FOR_TRAINING / UNKNOWN retained",
        })

    index = {
        "schema_version": "0915-robot-recovery-v21-d2-shallow-visual-index-v1",
        "status": "COMMITTED_OFFLINE_VISUALS",
        "deep_result": file_ref(deep_result_path),
        "gpu_lease_receipt": gpu_receipt,
        "videos": videos,
        "training_eligible": False,
        "clean_terminal": False,
        "control_ground_truth": False,
        "claim_limit": "Shallow copies of D2 offline review videos only; not Clean truth or training evidence.",
    }
    write_json(staging / "INDEX.json", index)
    (staging / "README_ZH.md").write_text(
        "# D2 fresh-fill 浅层视频\n\n"
        "本目录只包含两条实体 MP4 和机器可读索引。"
        "视频是 `OFFLINE_VISUAL / NOT_FOR_TRAINING`，合成写域仍为 `UNKNOWN`。\n\n"
        "- Poker044：166 帧，零写域 Raw 直通对照。\n"
        "- Chips097：394 帧，Raw / D1 domains / D2 candidate / change+UNKNOWN 对照。\n",
        encoding="utf-8",
    )
    return index


def publish(
    deep_result_path: Path, gpu_receipt_path: Path, shallow_root: Path
) -> dict[str, Any]:
    deep_result_path = deep_result_path.resolve(strict=True)
    gpu_receipt_path = gpu_receipt_path.resolve(strict=True)
    return publish_directory_atomically(
        shallow_root,
        lambda staging, final: _build(
            deep_result_path, gpu_receipt_path, staging, final
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-result", type=Path, required=True)
    parser.add_argument("--gpu-receipt", type=Path, required=True)
    parser.add_argument("--shallow-root", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.deep_result, args.gpu_receipt, args.shallow_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
