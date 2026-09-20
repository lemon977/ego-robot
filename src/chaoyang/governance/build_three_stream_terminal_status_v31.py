#!/usr/bin/env python3
"""Build the final V3.1 shallow status from immutable terminal receipts.

The initial three-stream task and its bounded runtime successors remain
immutable.  This projection reports their combined state without rewriting a
predecessor receipt or treating a runtime fix as algorithmic success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import (
    REPO_ROOT,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    validate_artifact_ref,
)


PARENT_RECEIPT = REPO_ROOT / "tasks/receipts/THREE_STREAM_STABLE_BASELINE_V31_RESULT.json"
AI1_RECEIPT = REPO_ROOT / "tasks/receipts/AI1_CPFS_PUBLISH_FIX_V31_RESULT.json"
AI2_RECEIPT = REPO_ROOT / "tasks/receipts/AI2_CURRENT_ASSET_PATH_FIX_V31_RESULT.json"
AI1_REBIND_RECEIPT = REPO_ROOT / "tasks/receipts/AI1_ARTIFACT_REF_REBIND_V31_RESULT.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/current/STATUS.json"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _verified_result(receipt: Mapping[str, Any], *, label: str) -> tuple[Mapping[str, Any], Path]:
    reference = _mapping(receipt.get("result"), label=f"{label}.result")
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError(f"invalid {label} result reference: {'; '.join(errors)}")
    path = Path(str(reference["path"]))
    return _mapping(load_json(path), label=f"{label} result"), path


def _verified_lane_result(
    parent_receipt: Mapping[str, Any], lane: str
) -> tuple[Mapping[str, Any], Path]:
    lane_results = _mapping(parent_receipt.get("lane_results"), label="parent.lane_results")
    reference = _mapping(lane_results.get(lane), label=f"parent.lane_results.{lane}")
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError(f"invalid {lane} lane result reference: {'; '.join(errors)}")
    path = Path(str(reference["path"]))
    return _mapping(load_json(path), label=f"{lane} lane result"), path


def build_terminal_status(
    *,
    parent_receipt: Mapping[str, Any],
    parent_receipt_path: Path,
    exact_result: Mapping[str, Any],
    exact_result_path: Path,
    ai1_receipt: Mapping[str, Any],
    ai1_receipt_path: Path,
    ai1_result: Mapping[str, Any],
    ai1_result_path: Path,
    ai2_receipt: Mapping[str, Any],
    ai2_receipt_path: Path,
    ai2_result: Mapping[str, Any],
    ai2_result_path: Path,
    ai1_rebind_receipt: Mapping[str, Any] | None,
    ai1_rebind_receipt_path: Path | None,
    ai1_rebind_result: Mapping[str, Any] | None,
    ai1_rebind_result_path: Path | None,
    generated_at: str,
) -> dict[str, Any]:
    exact_blockers = list(exact_result.get("blockers", []))
    ai2_counts = dict(_mapping(ai2_result.get("counts", {}), label="ai2.counts"))
    ai1_corrections = [
        {
            "kind": "CPFS_ATOMIC_PUBLISH_FIX",
            "status": ai1_receipt.get("status"),
            "receipt": artifact_ref(ai1_receipt_path),
            "result": artifact_ref(ai1_result_path),
        }
    ]
    if (
        ai1_rebind_receipt is not None
        and ai1_rebind_receipt_path is not None
        and ai1_rebind_result is not None
        and ai1_rebind_result_path is not None
    ):
        ai1_corrections.append(
            {
                "kind": "IMMUTABLE_ARTIFACT_REFERENCE_REBIND",
                "status": ai1_rebind_receipt.get("status"),
                "receipt": artifact_ref(ai1_rebind_receipt_path),
                "result": artifact_ref(ai1_rebind_result_path),
            }
        )

    return {
        "schema_version": "chaoyang-three-stream-shallow-status-v2",
        "generated_at": generated_at,
        "source": {
            "parent_receipt": artifact_ref(parent_receipt_path),
            "runtime_successors": {
                "ai1": artifact_ref(ai1_receipt_path),
                "ai2": artifact_ref(ai2_receipt_path),
                "ai1_artifact_ref_rebind": (
                    artifact_ref(ai1_rebind_receipt_path)
                    if ai1_rebind_receipt_path is not None
                    else None
                ),
            },
        },
        "parent": {
            "task_id": parent_receipt.get("task_id"),
            "status": parent_receipt.get("status"),
            "pipeline_complete": bool(parent_receipt.get("pipeline_complete", False)),
            "training_complete": bool(parent_receipt.get("training_complete", False)),
            "training_eligible": bool(parent_receipt.get("training_eligible", False)),
        },
        "lanes": {
            "exact78": {
                "status": exact_result.get("status"),
                "blockers": exact_blockers,
                "e0_execution_allowed": bool(
                    _mapping(exact_result.get("stages", {}), label="exact.stages")
                    .get("E0_CURRENT_DEVELOPMENT_PAIRS", {})
                    .get("execution_allowed", False)
                ),
                "gpu_used": bool(exact_result.get("gpu_used", False)),
                "model_inference_performed": bool(
                    exact_result.get("model_inference_performed", False)
                ),
                "training_started": False,
                "result": artifact_ref(exact_result_path),
                "next_step": (
                    "Provide or explicitly authorize a current non-archive frozen-156 cohort and "
                    "legal Raw/Robotized pair manifest; provide encoded-domain K/P/baseline authority "
                    "before metric Stereo consumption."
                ),
            },
            "ai1": {
                "status": ai1_result.get("machine_status", ai1_result.get("status")),
                "wrapper_status": ai1_result.get("status"),
                "blockers": [
                    "play_cards_0916_102:BLOCKED_MISSING_PINNED_OBSERVATION",
                    "play_cards_0916_103:BLOCKED_MISSING_PINNED_OBSERVATION",
                    "M2:BLOCKED_ORIENTATION_EVIDENCE",
                ],
                "evaluated_frames": 466,
                "model_calls": int(ai1_result.get("model_calls", 0)),
                "gpu_calls": int(ai1_result.get("gpu_calls", 0)),
                "m1_authority": "DEVELOPMENT_CANDIDATE_NOT_ADOPTED",
                "visible_surface_authority": "REGION_ONLY_NO_3D_SURFACE_POINT",
                "result": artifact_ref(ai1_result_path),
                "runtime_corrections": ai1_corrections,
                "next_step": (
                    "Pin independent anatomical-wrist observations for frozen adoption sessions "
                    "102/103; do not fill missing observations or promote M1 from internal residuals."
                ),
            },
            "ai2": {
                "status": ai2_result.get("machine_status", ai2_result.get("status")),
                "wrapper_status": ai2_result.get("status"),
                "counts": ai2_counts,
                "all_current_paths_bound": bool(ai2_result.get("all_existing_inputs_sha_bound", False)),
                "all_sessions_terminal": bool(ai2_result.get("all_sessions_terminal", False)),
                "model_calls": int(ai2_result.get("model_calls", 0)),
                "gpu_calls": int(ai2_result.get("gpu_calls", 0)),
                "result": artifact_ref(ai2_result_path),
                "runtime_corrections": [
                    {
                        "kind": "POKER_ASSET_DIRECTORY_MAPPING",
                        "status": ai2_receipt.get("status"),
                        "receipt": artifact_ref(ai2_receipt_path),
                    }
                ],
                "next_step": (
                    "Materialize an independent part-observability/reprojection evidence set, "
                    "suffix pairs, and an R0 local-quality candidate without using the evaluated "
                    "HaWoR output as its own observability judge."
                ),
            },
        },
        "final_claims": {
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
            "EXTERNAL_METRIC_AUTHORITY": False,
        },
        "resource_use": {
            "gpu_used": False,
            "model_inference_performed": False,
            "source_data_mutated": False,
        },
        "claim_limit": (
            "Generated terminal navigation projection. Runtime fixes remove execution defects only; "
            "they do not convert blocked evidence, internal residuals, or path binding into algorithmic "
            "quality, training eligibility, control truth, external metric authority, or deployment approval."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parent_receipt = _mapping(load_json(PARENT_RECEIPT), label="parent receipt")
    exact_result, exact_result_path = _verified_lane_result(parent_receipt, "exact78")
    ai1_receipt = _mapping(load_json(AI1_RECEIPT), label="AI1 receipt")
    ai1_result, ai1_result_path = _verified_result(ai1_receipt, label="AI1")
    ai2_receipt = _mapping(load_json(AI2_RECEIPT), label="AI2 receipt")
    ai2_result, ai2_result_path = _verified_result(ai2_receipt, label="AI2")

    ai1_rebind_receipt = None
    ai1_rebind_result = None
    ai1_rebind_result_path = None
    if AI1_REBIND_RECEIPT.is_file():
        ai1_rebind_receipt = _mapping(
            load_json(AI1_REBIND_RECEIPT), label="AI1 rebind receipt"
        )
        ai1_rebind_result, ai1_rebind_result_path = _verified_result(
            ai1_rebind_receipt, label="AI1 rebind"
        )

    value = build_terminal_status(
        parent_receipt=parent_receipt,
        parent_receipt_path=PARENT_RECEIPT,
        exact_result=exact_result,
        exact_result_path=exact_result_path,
        ai1_receipt=ai1_receipt,
        ai1_receipt_path=AI1_RECEIPT,
        ai1_result=ai1_result,
        ai1_result_path=ai1_result_path,
        ai2_receipt=ai2_receipt,
        ai2_receipt_path=AI2_RECEIPT,
        ai2_result=ai2_result,
        ai2_result_path=ai2_result_path,
        ai1_rebind_receipt=ai1_rebind_receipt,
        ai1_rebind_receipt_path=(AI1_REBIND_RECEIPT if ai1_rebind_receipt else None),
        ai1_rebind_result=ai1_rebind_result,
        ai1_rebind_result_path=ai1_rebind_result_path,
        generated_at=now_iso(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, value)
    print(json.dumps({"status": "PASSED", "output": artifact_ref(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
