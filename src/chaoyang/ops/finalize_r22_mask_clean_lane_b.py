#!/usr/bin/env python3
"""Publish an immutable, non-authoritative R2.2 Lane-B terminal summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1"
CPU = RUN / "attempts/attempt_0001_cpu_prereq"
CHIPS = RUN / "gpu_mask_canaries/chips010_role_sam31_r22/sessions/get_potato_chips_0901_010/attempts/attempt_0002"
POKER1 = RUN / "gpu_mask_canaries/poker015_object_sam31_r22/attempts/attempt_0001"
POKER2 = RUN / "gpu_mask_canaries/poker015_object_sam31_r22/attempts/attempt_0002"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def gpu_process_audit() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        command_path = Path(f"/proc/{pid}/cmdline")
        process_command = None
        if command_path.exists():
            process_command = command_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            ).strip()
        rows.append(
            {
                "pid": pid,
                "used_memory_mib": int(fields[1]),
                "command": process_command,
                "ownership": (
                    "EXTERNAL_EGOSTEER_SERVICE"
                    if process_command and "egosteer_touch" in process_command
                    else "UNKNOWN_DO_NOT_SIGNAL"
                ),
                "signal_sent": False,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    cpu = load(CPU / "RESULT.json")
    cpu_metrics = load(CPU / "METRICS.json")
    clean_reports = {row["label"]: row for row in cpu_metrics["reports"]}
    poker_clean = clean_reports["Poker245"]["metrics"]
    chips_clean = clean_reports["Chips039"]["metrics"]
    chips = load(CHIPS / "RESULT.json")
    chips_inner = load(CHIPS / "role_mask_output/RESULT.json")
    poker1 = load(POKER1 / "RESULT.json")
    poker2 = load(POKER2 / "RESULT.json")
    if "prompt sweep found no frame-0 card candidate" not in str(poker2.get("error")):
        raise RuntimeError("Poker attempt_0002 did not reach the frozen quality stop")

    metrics = {
        "schema_version": "r22-mask-clean-lane-b-metrics-v1",
        "clean_cpu_preflight": {
            "status": cpu["status"],
            "frame_count": poker_clean["frame_count"] + chips_clean["frame_count"],
            "source_map_verified_frames": poker_clean["frame_count"] + chips_clean["frame_count"],
            "semantic_donor_eligibility_frames": poker_clean["frame_count"] + chips_clean["frame_count"],
            "poker_causal_atlas_direct_observed_frames": poker_clean["atlas"]["direct_observed_candidate_frames"],
            "poker_pose_verified_canonical_atlas": poker_clean["atlas"]["verified_object_atlas"],
        },
        "chips010_role_mask": {
            "status": chips["status"],
            "formal_output_fps": 30.0,
            "source_fps_is_diagnostic_only": True,
            "human_pass_fractions": chips_inner["metrics"]["human_pass_fractions"],
            "tracker_expected_visible_coverage_fractions": chips_inner["metrics"]["tracker_expected_visible_coverage_fractions"],
        },
        "poker015_object_mask": {
            "status": "FAILED_QUALITY_C",
            "attempt_0001_status": poker1["status"],
            "attempt_0001_error": poker1["error"],
            "attempt_0002_recorded_status": poker2["status"],
            "attempt_0002_terminal_classification": "FAILED_QUALITY_C",
            "attempt_0002_error": poker2["error"],
            "classification_reason": "Real SAM3.1 prompt sweep completed but no frame-0 playing-card candidate intersected the frozen diagnostic guide.",
            "formal_output_fps_if_published": 30.0,
        },
        "gpu_process_audit_after_terminal": gpu_process_audit(),
    }
    artifacts = {
        "cpu_result": ref(CPU / "RESULT.json"),
        "candidate_evaluation_contract": ref(CPU / "MASK_CANDIDATE_EVALUATION_REFERENCE.json"),
        "chips_attempt_result": ref(CHIPS / "RESULT.json"),
        "chips_inner_result": ref(CHIPS / "role_mask_output/RESULT.json"),
        "chips_review_video": ref(CHIPS / "role_mask_output/get_potato_chips_0901_010_HAND_TRACKER_MASK_中文逐帧复核.mp4"),
        "poker_runtime_attempt": ref(POKER1 / "RESULT.json"),
        "poker_quality_attempt": ref(POKER2 / "RESULT.json"),
    }
    result = {
        "schema_version": "r22-mask-clean-lane-b-result-v1",
        "task_id": "r22_mask_clean_lane_b",
        "status": "TERMINAL_WITH_QUALITY_C_AND_BLOCKED_PREREQ",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "mask_canaries": {
            "chips010_role_sam31": "FAILED_QUALITY_C",
            "poker015_object_sam31": "FAILED_QUALITY_C",
        },
        "clean_successor": "BLOCKED_PREREQ",
        "fresh_propainter_executed": False,
        "authority_promoted": False,
        "current_mask_baseline_changed": False,
        "current_mask_baseline": "SAM3.1",
        "r7_0_modified": False,
        "metrics": metrics,
        "artifacts": artifacts,
        "claim_limit": "Development canaries and CPU readiness only. No Mask/Clean authority, Gold accuracy, or training eligibility.",
    }
    write_json(output / "RESULT.json", result)
    write_json(output / "METRICS.json", metrics)
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "r22-mask-clean-lane-b-run-receipt-v1",
        "task_id": result["task_id"], "status": result["status"],
        "created_at": result["created_at"], "host": result["host"],
        "worker_wrote_current_governance": False,
    })
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "r22-mask-clean-lane-b-next-v1",
        "status": result["status"],
        "next": "Keep SAM3.1 current. Freeze both failed canaries; do not run A/B regressions. Resolve support-surface semantic evidence and causal Poker atlas before a fresh Clean successor.",
        "stop_condition": "Both minimum Mask canaries reached quality-C terminal states; Clean prerequisites are explicitly blocked.",
    })
    write_text(output / "DECISION.md", (
        "# R2.2 Lane B 终态\n\n"
        "- Chips010 Role Mask：真实 SAM3.1 全片推理后 `FAILED_QUALITY_C`。\n"
        "- Poker015 Object Mask：真实 SAM3.1 文本提示扫描未找到与冻结诊断参考相交的牌实例，按质量 C 封账。\n"
        "- 正式输出合同保持 30 FPS；历史容器的 25 FPS 声明只记录为诊断。\n"
        "- Clean 三项 CPU 前置已执行，但缺 support-surface 语义、fresh lossless RGB/source-map 闭包和 pose 验证的 Poker causal atlas，因此 `BLOCKED_PREREQ`。\n"
        "- 未运行或伪造 fresh ProPainter；未改变 SAM3.1 current baseline 或任何 authority。\n"
    ))
    pinned = {name: ref(output / name) for name in (
        "RESULT.json", "METRICS.json", "RUN_RECEIPT.json", "NEXT_ACTION.json", "DECISION.md"
    )}
    pinned.update(artifacts)
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "r22-mask-clean-lane-b-artifact-manifest-v1",
        "task_id": result["task_id"], "status": result["status"],
        "artifacts": pinned, "authority_promoted": False,
    })
    print(json.dumps({"status": result["status"], "result": ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
