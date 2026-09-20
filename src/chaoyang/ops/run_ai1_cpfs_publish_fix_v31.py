#!/usr/bin/env python3
"""Run the sole CPU-only AI1 CPFS publication successor."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import build_wiyh_wrist_dual_input_v1 as input_builder
from chaoyang.ops import run_wiyh_ai1_current_lane_v31 as lane_runner


TASK_ID = "ai1_cpfs_publish_fix_v31"
SCHEMA_VERSION = "AI1_CPFS_PUBLISH_FIX_V31_RESULT"
OUTPUT_ROOT_RELATIVE = f"_run/current/{TASK_ID}/attempts/attempt_0001"
PREDECESSOR_RELATIVE = (
    "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/"
    "lanes/ai1/current_only_run_0001/RESULT.json"
)
DEFAULT_PROCESSED_ROOT = Path(
    "/mnt/data/egodata/datasets/ego/processed/"
    "chips_cards_handle_highview_0916/cleaned/playing_cards"
)
DEFAULT_EXPERIMENT_RELATIVE = "_run/current/worktrees/wiyh-hand-depth-v1/_run/current/experiments"


class SuccessorRunError(RuntimeError):
    """Raised when the immutable successor cannot be run safely."""


def _assert_routed(root: Path) -> None:
    index = load_json(root / "tasks/current/INDEX.json")
    routes = index.get("task_packets", [])
    if len(routes) != 1 or routes[0].get("task_id") != TASK_ID:
        raise SuccessorRunError("AI1 CPFS successor is not the sole current route")
    if routes[0].get("execution_allowed") is not True:
        raise SuccessorRunError("AI1 CPFS successor is not execution_allowed")


def _future_ref(source: Path, future: Path) -> dict[str, Any]:
    reference = artifact_ref(source)
    reference["path"] = str(future.absolute())
    return reference


def _terminal_status(machine_status: str) -> str:
    if machine_status.startswith(("PASS", "PASSED")):
        return "PASSED"
    if machine_status.startswith("BLOCKED"):
        return "BLOCKED_PREREQ"
    if machine_status.startswith(("FAILED", "FAIL_", "REJECT")):
        return "FAILED_RUNTIME"
    raise SuccessorRunError(f"unsupported AI1 lane terminal: {machine_status}")


def run_successor(
    *,
    root: Path = REPO_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    experiment_root: Path | None = None,
    runner: Callable[..., dict[str, Any]] = lane_runner.run,
    require_route: bool = True,
) -> dict[str, Any]:
    """Invoke the fixed lane once and preserve the sealed failed predecessor."""

    root = root.resolve(strict=True)
    if require_route:
        _assert_routed(root)
    predecessor = (root / PREDECESSOR_RELATIVE).resolve(strict=True)
    predecessor_value = load_json(predecessor)
    if (
        predecessor_value.get("schema_version") != "chaoyang-wiyh-ai1-lane-result-v31"
        or predecessor_value.get("status") != "FAILED_RUNTIME"
    ):
        raise SuccessorRunError("sealed predecessor schema or FAILED_RUNTIME status mismatch")
    predecessor_ref = artifact_ref(predecessor)
    output_root = root / OUTPUT_ROOT_RELATIVE
    if output_root.exists() or output_root.is_symlink():
        raise SuccessorRunError(f"fresh successor attempt required: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    resolved_experiments = (
        experiment_root.resolve(strict=True)
        if experiment_root is not None
        else (root / DEFAULT_EXPERIMENT_RELATIVE).resolve(strict=True)
    )
    try:
        lane_stage = staging / "AI1_CURRENT_ONLY_LANE"
        lane_value = runner(
            processed_root=processed_root.resolve(strict=True),
            experiment_root=resolved_experiments,
            output_root=lane_stage,
        )
        lane_result_path = lane_stage / "RESULT.json"
        if load_json(lane_result_path) != lane_value:
            raise SuccessorRunError("AI1 lane return value and RESULT.json differ")
        if lane_value.get("lane") != "ai1" or not isinstance(lane_value.get("status"), str):
            raise SuccessorRunError("AI1 lane result identity mismatch")
        status = _terminal_status(lane_value["status"])
        if artifact_ref(predecessor) != predecessor_ref:
            raise SuccessorRunError("sealed predecessor changed during successor execution")
        result = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": status,
            "machine_status": lane_value["status"],
            "predecessor_failed_result": predecessor_ref,
            "predecessor_preserved": True,
            "lane_result": _future_ref(
                lane_result_path,
                output_root / "AI1_CURRENT_ONLY_LANE" / "RESULT.json",
            ),
            "runner_invocations": 1,
            "single_writer": True,
            "writer_fence": "RENAME_NOREPLACE_OR_FIXED_FLOCK",
            "current_only": True,
            "archive_consumed": False,
            "weights": "ABSENT",
            "model_calls": 0,
            "gpu_calls": 0,
            "source_mutated": False,
            "old_artifact_mutated": False,
            "pipeline_complete": False,
            "numeric_quality_pass": False,
            "visual_review_status": "NOT_REVIEWED",
            "training_complete": False,
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
        }
        atomic_json(staging / "RESULT.json", result)
        input_builder._publish_directory_no_clobber(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_json(output_root / "RESULT.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    result = run_successor(
        root=args.root,
        processed_root=args.processed_root,
        experiment_root=args.experiment_root,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
