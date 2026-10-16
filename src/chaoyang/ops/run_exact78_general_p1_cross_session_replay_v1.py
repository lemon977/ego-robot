#!/usr/bin/env python3
"""Replay the current P1 Arm/diagnostic chain on three frozen sessions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from chaoyang.governance.common import artifact_ref, atomic_json, load_json
from chaoyang.ops import run_exact78_general_p1_action_canary_v1 as base
from chaoyang.ops.run_exact78_general_p1_action_canary_v8 import _storyboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest, _packet, root = base.authorize(args.manifest)
    if len(manifest.get("sessions", [])) != 3:
        raise RuntimeError("THREE_SESSION_REPLAY_REQUIRED")

    results = []
    problems = []
    for spec in manifest["sessions"]:
        session_id = spec["session_id"]
        session = root / "sessions" / session_id
        session.mkdir(parents=True)
        source = base.checked(spec["source_video"])
        hawor = base.checked(spec["hawor"])
        hawor_result_path = base.checked(spec["hawor_result"])
        hand = base.checked(spec["hand_states"])
        try:
            arm_root = session / "arm"
            base.execute(
                base.cli(
                    "run_exact78_session_base_arm_v8",
                    "--session-id", session_id,
                    "--hawor", hawor,
                    "--hawor-result", hawor_result_path,
                    "--output-root", arm_root,
                ),
                session / "arm.log",
            )
            diagnostics = session / "diagnostics"
            base.execute(
                base.cli(
                    "render_exact78_action_diagnostics_v2",
                    "--session-id", session_id,
                    "--source-video", source,
                    "--hawor", hawor,
                    "--arm", arm_root / "ARM_SESSION_BASE_STATES.npz",
                    "--hand", hand,
                    "--output-root", diagnostics,
                ),
                session / "diagnostics.log",
                allowed=(0, 2),
            )
            arm = load_json(arm_root / "RESULT.json")
            motion = load_json(hawor_result_path)["metrics"]
            diagnostic = load_json(diagnostics / "RESULT.json")
            frame_count = int(diagnostic["metrics"]["frame_count"])
            frame_ids = sorted(set(
                [0, frame_count // 2, frame_count - 1]
                + [int(round(index * (frame_count - 1) / 11)) for index in range(12)]
            ))[:12]
            storyboard = _storyboard(
                Path(diagnostic["outputs"]["raw_robot_overlay"]["path"]),
                frame_ids,
                {},
                diagnostics / f"{session_id}_P1_CROSS_SESSION_STORYBOARD_12.jpg",
            )
            metrics = arm["metrics"]
            item = {
                "session_id": session_id,
                "task": spec["task"],
                "fps": spec["fps"],
                "status": "TERMINAL",
                "motion": artifact_ref(hawor_result_path),
                "arm": artifact_ref(arm_root / "RESULT.json"),
                "diagnostics": artifact_ref(diagnostics / "RESULT.json"),
                "outputs": diagnostic["outputs"] | {"storyboard_12": storyboard},
                "motion_metrics": motion,
                "reachability": {
                    "observed_rows": metrics["metric_observed_rows"],
                    "static_ik_passed_rows": metrics["metric_passed_rows"],
                    "static_ik_pass_fraction": metrics["metric_pass_fraction"],
                    "smoothed_visual_passed_rows": metrics["metric_observed_visual_passed_rows"],
                    "smoothed_visual_pass_fraction": (
                        metrics["metric_observed_visual_passed_rows"] / metrics["metric_observed_rows"]
                        if metrics["metric_observed_rows"] else None
                    ),
                },
                "temporal": {
                    "velocity_rad_per_frame_max": metrics["velocity_rad_per_frame_max_contiguous"],
                    "acceleration_rad_per_frame2_max": metrics["acceleration_rad_per_frame2_max_contiguous"],
                    "velocity_gate": 0.12,
                    "acceleration_gate": 0.06,
                },
                "trajectory_failure_clusters": metrics.get("trajectory_failure_clusters", {}),
                "palmar_normal_camera_y": metrics.get("palmar_normal_camera_y"),
                "human_visual_review": "PENDING",
            }
            results.append(item)
            problems.append({
                "session_id": session_id,
                "owner": "VISUAL_REVIEW",
                "static_reach_fraction": item["reachability"]["static_ik_pass_fraction"],
                "smoothed_follow_fraction": item["reachability"]["smoothed_visual_pass_fraction"],
                "trajectory_failure_clusters": item["trajectory_failure_clusters"],
                "storyboard_sha256": storyboard["sha256"],
            })
        except Exception as error:
            failure = f"{type(error).__name__}:{error}"
            error_output = base.error_video(
                source,
                session / f"{session_id}_ERROR_DIAGNOSTIC.mp4",
                session_id,
                failure,
            )
            results.append({
                "session_id": session_id,
                "task": spec["task"],
                "fps": spec["fps"],
                "status": "FAILED_RUNTIME",
                "failure": failure,
                "error_video": error_output,
            })
            problems.append({"session_id": session_id, "owner": "INTEGRATION", "failure": failure})
        atomic_json(root / "PROGRESS.json", {"sessions": results})

    terminal = [row for row in results if row["status"] == "TERMINAL"]
    visible = sum(row["motion_metrics"]["role_mask_visible_side_frames"] for row in terminal)
    missing = sum(row["motion_metrics"]["visible_without_visual_action_side_frames"] for row in terminal)
    observed = sum(row["reachability"]["observed_rows"] for row in terminal)
    static_passed = sum(row["reachability"]["static_ik_passed_rows"] for row in terminal)
    smooth_passed = sum(row["reachability"]["smoothed_visual_passed_rows"] for row in terminal)
    provenance = Counter()
    for row in terminal:
        for side_counts in row["motion_metrics"]["provenance_counts"].values():
            provenance.update(side_counts)
    atomic_json(root / "PROBLEM_TABLE.json", {"rows": problems})
    result = {
        "schema_version": "exact78-general-p1-cross-session-replay-result-v1",
        "status": "EXECUTION_COMPLETE" if len(results) == 3 else "INCOMPLETE",
        "execution_complete": len(results) == 3,
        "visual_qualified": False,
        "quality_passed": False,
        "training_eligible": False,
        "canary_count": 3,
        "terminal_action_results": len(terminal),
        "four_metrics": {
            "motion_completeness": {
                "visible_without_action": missing,
                "role_mask_visible": visible,
                "fraction": missing / visible if visible else None,
                "target_max": 0.05,
                "provenance_counts": dict(provenance),
            },
            "cross_session_reachability": {
                "contract": "independent per-frame static IK on OBSERVED rows",
                "metric_observed_passed": static_passed,
                "metric_observed_rows": observed,
                "fraction": static_passed / observed if observed else None,
                "target_min": 0.90,
            },
            "smoothed_visual_follow": {
                "contract": "bounded visual trajectory against original per-frame targets; diagnostic, not base reachability",
                "metric_observed_passed": smooth_passed,
                "metric_observed_rows": observed,
                "fraction": smooth_passed / observed if observed else None,
            },
            "terminal_coverage": {
                "terminal": len(results),
                "expected": 3,
                "action_complete": len(terminal),
                "diagnostic_video_or_error_video": len(results),
            },
            "visual_quality_distribution": "PENDING_HUMAN_FULL_VIDEO_REVIEW_NOT_A_VISUAL_PASS",
        },
        "sessions": results,
        "outputs": {
            "progress": artifact_ref(root / "PROGRESS.json"),
            "problem_table": artifact_ref(root / "PROBLEM_TABLE.json"),
        },
        "metrics": {"wall_seconds": time.monotonic() - started},
        "deletion_authorized": False,
        "claim_limit": "Three-session CPU replay of frozen same-session HaWoR and Hand states. It validates the current base/Arm/diagnostic integration, not fresh HaWoR, Clean, occlusion, training, control or deployment.",
    }
    atomic_json(root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "four_metrics": result["four_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
