#!/usr/bin/env python3
"""Fast P1 correction from SHA-pinned terminal same-session intermediates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, load_json
from chaoyang.ops import run_exact78_general_p1_action_canary_v1 as base
from chaoyang.ops.run_exact78_general_p1_action_canary_v8 import _storyboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest, packet, root = base.authorize(args.manifest)
    if len(manifest["sessions"]) != 1:
        raise RuntimeError("ONE_SESSION_FAST_CORRECTION_REQUIRED")
    spec = manifest["sessions"][0]
    inputs = spec["fast_correction_inputs"]
    source = base.checked(spec["source_video"])
    hawor = base.checked(inputs["hawor"])
    hawor_result = base.checked(inputs["hawor_result"])
    arm_states = base.checked(inputs["arm_states"])
    arm_result = base.checked(inputs["arm_result"])
    hand_states = base.checked(inputs["hand_states"])
    session = root / "sessions" / spec["session_id"]
    session.mkdir(parents=True)
    arm_output = session / "arm_causal"
    base.execute(base.cli(
        "run_exact78_arm_causal_visual_v2",
        "--session-id", spec["session_id"],
        "--source-states", arm_states,
        "--source-result", arm_result,
        "--hawor", hawor,
        "--output-root", arm_output,
    ), session / "arm_causal.log")
    diagnostics = session / "diagnostics"
    base.execute(base.cli(
        "render_exact78_action_diagnostics_v2",
        "--session-id", spec["session_id"],
        "--source-video", source,
        "--hawor", hawor,
        "--arm", arm_output / "ARM_CAUSAL_VISUAL_STATES.npz",
        "--hand", hand_states,
        "--output-root", diagnostics,
    ), session / "diagnostics.log", allowed=(0, 2))
    arm = load_json(arm_output / "RESULT.json")
    diagnostic = load_json(diagnostics / "RESULT.json")
    frame_count = int(diagnostic["metrics"]["frame_count"])
    frame_ids = np.rint(np.linspace(0, frame_count - 1, 12)).astype(int).tolist()
    storyboard = _storyboard(
        Path(diagnostic["outputs"]["raw_robot_overlay"]["path"]), frame_ids, {},
        diagnostics / f"{spec['session_id']}_FAST_CORRECTION_STORYBOARD_12.jpg",
    )
    result = {
        "schema_version": "exact78-general-p1-fast-correction-result-v1",
        "status": "EXECUTION_COMPLETE",
        "execution_complete": True,
        "visual_qualified": False,
        "quality_passed": False,
        "training_eligible": False,
        "canary_count": 1,
        "terminal_action_results": 1,
        "four_metrics": {
            "motion_completeness": load_json(hawor_result)["metrics"],
            "cross_session_reachability": {
                "metric_observed_passed": arm["metrics"]["metric_observed_visual_passed_rows"],
                "metric_observed_rows": arm["metrics"]["metric_observed_rows"],
                "fraction": (arm["metrics"]["metric_observed_visual_passed_rows"]
                             / arm["metrics"]["metric_observed_rows"]),
                "target_min": 0.90,
            },
            "terminal_coverage": {"terminal": 1, "expected": 1, "action_complete": 1,
                                  "diagnostic_video_or_error_video": 1},
            "visual_quality_distribution": "P1_FAST_CORRECTION_PENDING_HUMAN_REVIEW",
        },
        "sessions": [{
            "session_id": spec["session_id"],
            "task": spec["task"],
            "status": "TERMINAL",
            "arm": artifact_ref(arm_output / "RESULT.json"),
            "diagnostics": artifact_ref(diagnostics / "RESULT.json"),
            "diagnostic_outputs": diagnostic["outputs"] | {"storyboard_12": storyboard},
        }],
        "metrics": {
            "postprocess_wall_seconds": arm["metrics"]["wall_seconds"],
            "diagnostics_wall_seconds": diagnostic["metrics"]["wall_seconds"],
            "total_wall_seconds": time.monotonic() - started,
            "arm": arm["metrics"],
        },
        "outputs": {
            "arm_result": artifact_ref(arm_output / "RESULT.json"),
            "diagnostics_result": artifact_ref(diagnostics / "RESULT.json"),
            "storyboard_12": storyboard,
        },
        "deletion_authorized": False,
        "next_action": "HUMAN_REVIEW_THEN_THROUGHPUT_REDESIGN",
        "claim_limit": "One-session fast causal postprocess validation only; not general-baseline completion, visual adoption, training, deletion or Robot100 authority.",
    }
    atomic_json(root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
