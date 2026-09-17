#!/usr/bin/env python3
"""Publish an immutable compact Lane C/D summary without touching governance."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def digest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intersection-result", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument("--poker-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)
    intersection = load(args.intersection_result)
    depth = load(args.depth_result)
    poker = load(args.poker_result)
    if intersection.get("terminal_status") != "PASSED" or poker.get("terminal_status") != "PASSED":
        raise ValueError("intersection and Poker Occlusion must be passed development terminals")
    if depth.get("terminal_status") != "BLOCKED_PREREQ" or depth.get("foundationstereo_executed") is not False:
        raise ValueError("Depth must remain fail-closed")
    visuals = {
        "schema_version": "R22_LANE_CD_VISUAL_REVIEW_INDEX_V1",
        "poker_basic_occlusion_fullsession": poker["review_video"],
        "selected_eye_source_index_audit": depth["visualization"],
        "claim_limit": "Development visuals only; no accuracy, training, control or physical authority.",
    }
    write(output / "VISUAL_REVIEW_INDEX.json", visuals)
    metrics = {
        "schema_version": "R22_LANE_CD_SUMMARY_METRICS_V1",
        "intersection": intersection["counts"],
        "depth_terminal": depth["terminal_status"],
        "depth_foundationstereo_executed": False,
        "poker_session": poker["session_id"],
        "poker_frames": poker["frame_count"],
        "poker_contact_refinement": poker["contact_aware_refinement"],
    }
    write(output / "METRICS.json", metrics)
    (output / "DECISION.md").write_text(
        "# Lane C/D V2封账\n\n"
        "58条Wave0与v77已处理21条完成同会话证据交集；Poker227已生成196帧基础几何"
        "Occlusion全片。对象外观仅使用当前Raw直接可见牌面，隐藏区域和Contact窄带保持UNKNOWN。\n\n"
        "play_cards_0910_001的selected-eye/source-index映射一致，但rectification P90未通过，"
        "因此Depth保持BLOCKED_PREREQ，未运行FoundationStereo。\n",
        encoding="utf-8",
    )
    write(output / "NEXT_ACTION.json", {
        "schema_version": "R22_LANE_CD_SUMMARY_NEXT_V1",
        "status": "PASSED_WITH_LOCAL_BLOCKER",
        "next_task_id": "CONTACT10_POKER227_OR_RECTIFICATION_EVIDENCE_SUCCESSOR",
        "automatic_expansion": False,
    })
    write(output / "RUN_RECEIPT.json", {
        "schema_version": "R22_LANE_CD_SUMMARY_RECEIPT_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "intersection_result": digest(args.intersection_result),
        "depth_result": digest(args.depth_result), "poker_result": digest(args.poker_result),
        "producer": digest(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authority_promoted": False,
    })
    write(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "R22_LANE_CD_SUMMARY_ARTIFACTS_V1",
        "artifacts": [
            digest(output / "VISUAL_REVIEW_INDEX.json"), digest(output / "METRICS.json"),
            digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
        ],
    })
    result = {
        "schema_version": "R22_LANE_CD_SUMMARY_RESULT_V1",
        "terminal_status": "PASSED",
        "status": "PASS_BOUNDED_LANE_CD_WITH_DEPTH_BLOCKED_PREREQ",
        "components": {
            "intersection": digest(args.intersection_result),
            "depth": digest(args.depth_result), "poker_basic_occlusion": digest(args.poker_result),
        },
        "visual_review_index": digest(output / "VISUAL_REVIEW_INDEX.json"),
        "authority_promoted": False,
        "claim_limit": "Bounded task closure only; local Depth blocker remains and no authority is promoted.",
    }
    write(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
