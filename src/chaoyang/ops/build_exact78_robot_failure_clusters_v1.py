#!/usr/bin/env python3
"""Summarize current exact78 Robot quality-C terminals without changing gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.terminal_root.resolve().glob("*/*/RESULT.json"))
    rows = []
    combinations: Counter[str] = Counter()
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "FAILED_QUALITY_C":
            continue
        arm = result["numeric"]["arm"]
        hand = result["numeric"]["hand"]
        reason = " + ".join(result["reason_codes"])
        combinations[reason] += 1
        rows.append(
            {
                "task": result["task"],
                "session": result["session"],
                "reason": reason,
                "arm_failed_rows": arm.get("failed_rows", 0),
                "arm_unknown_rows": arm.get("unknown_rows", 0),
                "arm_position_mm_max": arm.get("position_mm_max"),
                "arm_rotation_deg_max": arm.get("rotation_deg_max"),
                "hand_failed_rows": hand.get("failed_rows", 0),
                "hand_bone_error_deg_max": hand.get("bone_error_deg_max"),
                "hand_tip_direction_error_deg_max": hand.get("tip_direction_error_deg_max"),
                "result": ref(path),
            }
        )
    hand_failures = [row for row in rows if row["hand_failed_rows"] > 0]
    arm_failures = [row for row in rows if row["arm_failed_rows"] > 0]
    result = {
        "schema_version": "exact78-robot-failure-clusters-v1",
        "status": "PASS_DEVELOPMENT_FAILURE_DIAGNOSIS",
        "counts": {
            "quality_c_terminals": len(rows),
            "arm_failure_sessions": len(arm_failures),
            "hand_failure_sessions": len(hand_failures),
            "hand_failures_with_tip_over_15_deg": sum(
                row["hand_tip_direction_error_deg_max"] > 15 for row in hand_failures
            ),
            "hand_failures_with_bone_over_60_deg": sum(
                row["hand_bone_error_deg_max"] > 60 for row in hand_failures
            ),
        },
        "reason_combinations": dict(combinations),
        "rows": rows,
        "bounded_successors": {
            "arm": "Optimize one fixed per-session base placement against sampled full-session worst-case reachability and joint-margin objectives; retain the same residual gates and two old A/B regressions.",
            "hand": "Add a distal-link/tip-direction refinement objective and temporal warm start per finger; retain the 15 degree tip and 60 degree bone gates and verify old A/B regressions.",
        },
        "claim_limit": "Digital Robot quality-C diagnosis only; no authority, physical accuracy or permission to loosen gates.",
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "ROBOT_FAILURE_CLUSTERS.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# exact78 Robot 质量 C 失败簇",
        "",
        f"本快照包含 {len(rows)} 条质量 C：Arm 失败 {len(arm_failures)} 条，Hand 失败 {len(hand_failures)} 条。",
        "",
        f"Hand 失败中，tip direction 超过 15° 为 {result['counts']['hand_failures_with_tip_over_15_deg']}/{len(hand_failures)}，bone 超过 60° 为 {result['counts']['hand_failures_with_bone_over_60_deg']}/{len(hand_failures)}。因此当前 Hand 的共同瓶颈是末端方向，不是 MANO 骨段门。",
        "",
        "Arm 失败表现为全片部分帧位置/旋转残差和少量 unknown，优先优化固定 base placement 的全片最坏帧与关节余量，不按单条会话手工平移，也不放宽门槛。",
        "",
        "## 有界 successor",
        "",
        "- Arm：在固定世界坐标链下，用均匀帧+困难帧联合选择单一 base placement，并冻结旧 A/B 回归。",
        "- Hand：增加逐指 distal/tip-direction refinement 与时序 warm start；继续使用 tip≤15°、bone≤60° 原门。",
        "",
        "本报告仅是数字 Robot 失败诊断，不是 Robot authority 或物理精度结论。",
    ]
    atomic_text(output / "REPORT_ZH.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
