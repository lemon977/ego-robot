#!/usr/bin/env python3
"""Run the CPU-only Exact78 E0/E1 current-input audit in its assigned lane.

This maintained operation deliberately treats missing cohort, paired bundles,
or encoded-domain calibration authority as an honest ``BLOCKED_INPUTS``
terminal.  Such evidence blockers are not process/runtime failures.  The
operation never runs FoundationStereo and never writes source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.human_ego.tools import (
    build_exact78_development_pair_ledgers_v31 as e0_builder,
)
from chaoyang.ops import build_exact78_stereo_preflight_input_v31 as e1_builder


TASK_ID = "three_stream_stable_baseline_v31"
LANE = "exact78"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
LANE_ROOT = ATTEMPT_ROOT / "lanes/exact78"
OUTPUT_ROOT = LANE_ROOT / "exact78_development_pair_audit_v31"
DATA_ROOT = Path("/mnt/data/egodata/datasets/ego")
E1_SESSION_ROOT = (
    DATA_ROOT
    / "chips_cards_tracker_0902/potato_chips/get_potato_chips_0902_023"
)
RESULT_SCHEMA = "chaoyang-exact78-development-pair-audit-lane-result-v31"


class Exact78LaneAuditError(RuntimeError):
    """Raised for writer/state/runtime contract violations."""


def validate_lane_state(lane_root: Path) -> dict[str, Any]:
    root = lane_root.resolve(strict=True)
    state_path = root / "STATE.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise Exact78LaneAuditError("exact78 lane requires a regular STATE.json")
    state = load_json(state_path)
    if (
        state.get("schema_version") != "chaoyang-three-stream-lane-state-v1"
        or state.get("parent_task_id") != TASK_ID
        or state.get("lane") != LANE
        or state.get("status") != "READY_CPU_PREFLIGHT"
        or state.get("current_action") != "CPU_PREFLIGHT_NOT_STARTED"
        or Path(str(state.get("writer_root", ""))).resolve(strict=False) != root
    ):
        raise Exact78LaneAuditError("exact78 lane STATE identity/readiness mismatch")
    return state


def _stage(
    *,
    name: str,
    output_root: Path,
    operation: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    try:
        value = operation()
        result_path = output_root / "RESULT.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise Exact78LaneAuditError(f"{name} did not publish a regular RESULT.json")
        if load_json(result_path) != value:
            raise Exact78LaneAuditError(f"{name} returned/published RESULT mismatch")
        return (
            {
                "status": str(value.get("status", "FAILED_RUNTIME")),
                "execution_allowed": value.get("execution_allowed") is True,
                "result": artifact_ref(result_path),
            },
            None,
        )
    except Exception as error:  # noqa: BLE001 - aggregate the independent E0/E1 stages.
        return (
            {
                "status": "FAILED_RUNTIME",
                "execution_allowed": False,
                "result": None,
            },
            f"{name}:RUNTIME:{type(error).__name__}:{error}",
        )


def run(
    *,
    lane_root: Path,
    data_root: Path,
    e1_session_root: Path,
    cohort_manifest: Path | None,
    candidate_index: Path | None,
    encoded_calibration: Path | None,
) -> dict[str, Any]:
    """Run E0 then E1, preserving quality blockers as a terminal result."""

    validate_lane_state(lane_root)
    root = lane_root.resolve(strict=True)
    output = root / OUTPUT_ROOT.name
    if output.exists() or output.is_symlink():
        raise Exact78LaneAuditError(f"fresh lane output required: {output}")
    output.mkdir()

    stages: dict[str, Any] = {}
    runtime_errors: list[str] = []
    e0_output = output / "e0_current_pairs"
    stages["E0_CURRENT_DEVELOPMENT_PAIRS"], runtime_error = _stage(
        name="E0_CURRENT_DEVELOPMENT_PAIRS",
        output_root=e0_output,
        operation=lambda: e0_builder.build(
            data_root=data_root,
            cohort_manifest=cohort_manifest,
            candidate_index=candidate_index,
            output_root=e0_output,
        ),
    )
    if runtime_error is not None:
        runtime_errors.append(runtime_error)

    e1_output = output / "e1_stereo_preflight_input"
    stages["E1_STEREO_PREFLIGHT_INPUT"], runtime_error = _stage(
        name="E1_STEREO_PREFLIGHT_INPUT",
        output_root=e1_output,
        operation=lambda: e1_builder.build(
            e1_session_root,
            e1_output,
            encoded_calibration=encoded_calibration,
        ),
    )
    if runtime_error is not None:
        runtime_errors.append(runtime_error)

    blockers: list[str] = []
    for stage_name, stage in stages.items():
        result_ref = stage.get("result")
        if not isinstance(result_ref, dict):
            continue
        stage_result = load_json(Path(str(result_ref["path"])))
        for blocker in stage_result.get("blockers", []):
            blockers.append(f"{stage_name}:{blocker}")
    blockers = sorted(set(blockers))
    all_allowed = all(stage["execution_allowed"] for stage in stages.values())
    returned_runtime_failure = any(
        stage["status"].startswith(("FAILED_RUNTIME", "FAIL_RUNTIME"))
        for stage in stages.values()
    )
    if runtime_errors or returned_runtime_failure:
        status = "FAILED_RUNTIME"
    elif all_allowed:
        status = "PASS_EXACT78_E0_E1_INPUT_ADMISSION"
    else:
        status = "BLOCKED_INPUTS"

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "lane": LANE,
        "parent_task_id": TASK_ID,
        "stages": stages,
        "blockers": blockers,
        "runtime_errors": runtime_errors,
        "execution_allowed": all_allowed and not runtime_errors,
        "foundationstereo_inference_performed": False,
        "model_inference_performed": False,
        "gpu_used": False,
        "source_mutated": False,
        "archive_promoted_to_current": False,
        "claims": {
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
        },
        "claim_limit": (
            "CPU-only current-input admission. A blocked result is a valid terminal, "
            "not a runtime failure. No FoundationStereo inference, training, control "
            "ground truth, or physical deployment authority is produced."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    return result


def terminal_exit_code(result: dict[str, Any]) -> int:
    """Expected evidence blockers exit successfully; runtime failures do not."""

    return 1 if result.get("status") == "FAILED_RUNTIME" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--e1-session-root", type=Path, default=E1_SESSION_ROOT)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--encoded-calibration", type=Path)
    args = parser.parse_args()
    result = run(
        lane_root=LANE_ROOT,
        data_root=args.data_root,
        e1_session_root=args.e1_session_root,
        cohort_manifest=args.cohort_manifest,
        candidate_index=args.candidate_index,
        encoded_calibration=args.encoded_calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return terminal_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
