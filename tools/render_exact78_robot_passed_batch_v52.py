#!/usr/bin/env python3
"""Render full-session reviews for complete bounded arm+hand numeric rows.

Strict PASS rows become human-review candidates.  Exhausted arm/hand quality-C
rows use the renderer's explicit HOLD flags and produce unmistakably watermarked
failure evidence for terminal publication.  No HOLD row is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
RENDER_TOOL = PROJECT / "tools/render_robot_motion_transfer_fullsession_v2.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_ref(value: dict[str, Any], label: str) -> Path:
    if set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: exact ref required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = PROJECT / path
    actual = ref(path)
    if actual["bytes"] != value["bytes"] or actual["sha256"] != value["sha256"]:
        raise ValueError(f"{label}: ref mismatch")
    return Path(actual["path"])


def verify_artifact_ref(value: dict[str, Any], label: str) -> Path:
    """Verify the immutable file triple while allowing audited media metadata."""
    required = {"path", "bytes", "sha256"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError(f"{label}: artifact ref required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = PROJECT / path
    actual = ref(path)
    if actual["bytes"] != value["bytes"] or actual["sha256"] != value["sha256"]:
        raise ValueError(f"{label}: ref mismatch")
    return Path(actual["path"])


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def unique_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or not session:
            raise ValueError(f"{label}: every row requires a non-empty session")
        if session in output:
            raise ValueError(f"{label}: duplicate session {session}")
        output[session] = row
    return output


def arm_payload(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    status = row["status"]
    if status == "PASS_ARM_METHOD1_CARRIED_NO_ROUND2":
        return (
            row["method1_result"],
            row["method1_states"],
            f"fixed placement={row['selected_backoff_m']:.4f} m",
            True,
        )
    if status == "PASS_ARM_ROUND2_BIDIRECTIONAL":
        return (
            row["result"],
            row["states"],
            f"fixed placement={row['method1_best_hold_backoff_m']:.4f} m; arm method2 bidirectional",
            True,
        )
    if status == "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C":
        return (
            row["result"],
            row["states"],
            f"fixed placement={row['method1_best_hold_backoff_m']:.4f} m; arm method2 HOLD",
            False,
        )
    raise ValueError(f"unexpected final arm status {status}")


def hand_payload(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    status = row["status"]
    if status == "PASS_HAND_METHOD1_CARRIED_NO_ROUND2":
        return row["method1_result"], row["method1_states"], True
    if status == "PASS_HAND_ROUND2_BIDIRECTIONAL":
        return row["result"], row["states"], True
    if status == "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C":
        return row["result"], row["states"], False
    raise ValueError(f"unexpected final hand status {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--arm-final-result", type=Path, required=True)
    parser.add_argument("--hand-final-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")

    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    expected_tool = preflight.get("programs", {}).get(RENDER_TOOL.name)
    if expected_tool is None or ref(RENDER_TOOL) != expected_tool:
        raise SystemExit("render tool differs from preflight closure")

    temporal_result_path = args.temporal_root / "RESULT.json"
    temporal = load(temporal_result_path)
    if temporal.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise SystemExit("passed temporal root required")
    temporal_by_session = unique_index(temporal.get("sessions", []), "temporal")

    arm = load(args.arm_final_result)
    if arm.get("status") != "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED":
        raise SystemExit("complete final arm aggregate required")
    arm_by_session = unique_index(arm.get("sessions", []), "arm")

    hand = load(args.hand_final_result)
    if hand.get("status") != "PASS_BOUNDED_TWO_METHOD_HAND_RESULTS_PUBLISHED":
        raise SystemExit("complete final hand aggregate required")
    hand_by_session = unique_index(hand.get("sessions", []), "hand")
    hand_round1_path = verify_ref(hand["hand_round1_result"], "hand_round1_result")
    hand_round1 = load(hand_round1_path)
    if hand_round1.get("arm_final_result") != ref(args.arm_final_result):
        raise SystemExit("hand/arm final closure mismatch")

    expected_sessions = set(unique_index(preflight.get("sessions", []), "preflight"))
    if set(temporal_by_session) != expected_sessions:
        raise SystemExit("preflight/temporal session set mismatch")
    if set(arm_by_session) != expected_sessions or set(hand_by_session) != expected_sessions:
        raise SystemExit("preflight/arm/hand session set mismatch")

    args.output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    counts = {
        "strict_pass_rendered": 0,
        "quality_hold_rendered": 0,
        "arm_quality_c": 0,
        "hand_quality_c": 0,
        "runtime_failed_retryable": 0,
    }
    for preflight_row in preflight.get("sessions", []):
        session = preflight_row["session"]
        task = preflight_row["task"]
        arm_row = arm_by_session[session]
        hand_row = hand_by_session[session]
        if arm_row.get("task") != task or hand_row.get("task") != task:
            raise SystemExit(f"{session}: task identity mismatch")
        arm_data = arm_payload(arm_row)
        hand_data = hand_payload(hand_row)
        arm_result_ref, arm_states_ref, placement_label, arm_numeric_pass = arm_data
        hand_result_ref, hand_states_ref, hand_numeric_pass = hand_data
        if hand_row.get("arm_numeric_pass") is not arm_numeric_pass:
            raise SystemExit(f"{session}: hand aggregate did not inherit arm numeric status")
        if not arm_numeric_pass:
            counts["arm_quality_c"] += 1
        if not hand_numeric_pass:
            counts["hand_quality_c"] += 1
        arm_result_path = verify_ref(arm_result_ref, f"{session}:arm_result")
        arm_states_path = verify_ref(arm_states_ref, f"{session}:arm_states")
        hand_result_path = verify_ref(hand_result_ref, f"{session}:hand_result")
        hand_states_path = verify_ref(hand_states_ref, f"{session}:hand_states")
        temporal_session_result_path = verify_ref(
            temporal_by_session[session]["result"], f"{session}:temporal_result"
        )
        temporal_session_result = load(temporal_session_result_path)
        hawor_path = verify_ref(temporal_session_result["outputs"]["npz"], f"{session}:temporal_npz")
        output_dir = args.output_root / task / session / "fullsession_review"
        command = [
            sys.executable,
            str(RENDER_TOOL),
            "--task",
            task,
            "--session",
            session,
            "--hawor",
            str(hawor_path),
            "--hawor-result",
            str(temporal_session_result_path),
            "--arm-states",
            str(arm_states_path),
            "--arm-result",
            str(arm_result_path),
            "--hand-states",
            str(hand_states_path),
            "--hand-result",
            str(hand_result_path),
            "--fixed-placement-label",
            placement_label,
            "--output-dir",
            str(output_dir),
        ]
        if not arm_numeric_pass:
            command.append("--allow-arm-hold-review")
        if not hand_numeric_pass:
            command.append("--allow-hand-hold-review")
        session_root = args.output_root / task / session
        session_root.mkdir(parents=True)
        log_path = session_root / "render_execution.log"
        started = now()
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=PROJECT, text=True, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            os.fsync(log.fileno())
        item: dict[str, Any] = {
            "task": task,
            "session": session,
            "started_at": started,
            "finished_at": now(),
            "returncode": process.returncode,
            "command": command,
            "execution_log": ref(log_path),
            "placement_label": placement_label,
        }
        result_path = output_dir / "RESULT.json"
        if process.returncode == 0 and result_path.is_file():
            result = load(result_path)
            expected_status = (
                "PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
                if arm_numeric_pass and hand_numeric_pass
                else (
                    "HOLD_ARM_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
                    if not arm_numeric_pass
                    else "HOLD_HAND_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
                )
            )
            if result.get("status") != expected_status:
                raise SystemExit(f"{session}: renderer status does not match numeric gates")
            if result.get("task") != task or result.get("session") != session:
                raise SystemExit(f"{session}: render identity mismatch")
            video_meta = result["outputs"]["video"]
            manifest_meta = result["outputs"]["frame_manifest"]
            video_path = verify_artifact_ref(video_meta, f"{session}:review_video")
            manifest_path = verify_artifact_ref(manifest_meta, f"{session}:frame_manifest")
            frame_count = int(result["frame_count"])
            if int(video_meta.get("decoded_frames", -1)) != frame_count:
                raise SystemExit(f"{session}: decoded review frame count mismatch")
            if int(manifest_meta.get("rows", -1)) != frame_count:
                raise SystemExit(f"{session}: frame manifest row count mismatch")
            item.update(
                status=(
                    "PASS_RENDER_READY_FOR_HUMAN_REVIEW"
                    if arm_numeric_pass and hand_numeric_pass
                    else "HOLD_RENDER_READY_FOR_QUALITY_TERMINAL"
                ),
                rendered=True,
                result=ref(result_path),
                video=ref(video_path),
                frame_manifest=ref(manifest_path),
                frame_count=frame_count,
            )
            if arm_numeric_pass and hand_numeric_pass:
                counts["strict_pass_rendered"] += 1
            else:
                counts["quality_hold_rendered"] += 1
        else:
            item.update(
                status="FAILED_RUNTIME_RETRYABLE",
                rendered=False,
                error="renderer failed without a valid strict numeric PASS review",
            )
            counts["runtime_failed_retryable"] += 1
        rows.append(item)

    status = (
        "PASS_NUMERIC_ROBOT_REVIEWS_PUBLISHED"
        if counts["runtime_failed_retryable"] == 0
        else "FAILED_RUNTIME_RETRYABLE"
    )
    aggregate = {
        "schema_version": "exact78-robot-passed-batch-render-v52-v1",
        "created_at": now(),
        "status": status,
        "preflight": ref(args.preflight),
        "temporal_root_result": ref(temporal_result_path),
        "arm_final_result": ref(args.arm_final_result),
        "hand_final_result": ref(args.hand_final_result),
        "render_tool": ref(RENDER_TOOL),
        "counts": counts,
        "sessions": rows,
        "authority": False,
        "action_sidecar_published": False,
        "human_visual_approval": "PENDING" if counts["strict_pass_rendered"] else "NOT_APPLICABLE",
        "claim_limit": (
            "Strict numeric arm+hand PASS videos are human-review candidates. Exhausted HOLD videos are only "
            "watermarked quality-C evidence and cannot be promoted. No Robot/action/contact/training/"
            "deployment/physical-accuracy authority."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(json.dumps({"status": status, "counts": counts, "result": ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0 if counts["runtime_failed_retryable"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
