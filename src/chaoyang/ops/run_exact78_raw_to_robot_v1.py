"""Governed raw-input preflight for the Exact78 visual Robot candidate.

This V1 freezes a source/recipe boundary only. A later, separately signed
candidate must execute HaWoR and Robot; V1 never reports a completed video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    ColdStartInputError, PROJECT_ROOT, file_ref, stage_signature, validate_output_path,
)
from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


def candidate_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Prepare explicit stage boundaries without importing an algorithm model."""
    sources = request["sources"]
    return {
        "schema_version": "exact78-raw-to-robot-candidate-plan-v1",
        "submission_input_sha256": request["submission_input_sha256"],
        "sessions": [
            {"session_id": source["session_id"], "source_kind": source["kind"],
             "input_files": source["files"],
             "source_time_mode": source["time_mode"],
             "capture_to_decode_alignment": source["capture_to_decode_alignment"],
             "metric_calibration": source["metric_calibration"],
             "stages": {"raw_input": "SHA_VERIFIED_PENDING_RUNTIME_OPEN_AUDIT",
                        "hawor": "NOT_EVALUATED", "robot": "NOT_EVALUATED",
                        "clean": "NOT_EVALUATED", "depth": "NOT_EVALUATED",
                        "object6d": "NOT_EVALUATED", "composite": "NOT_EVALUATED"}}
            for source in sources
        ],
        "model_weights": "NOT_CONSUMED_BY_PREFLIGHT",
        "control_ground_truth": False, "training_eligible": False,
        "execution_complete": False, "visual_qualified": False, "product_qualified": False,
        "claim_limit": "Raw input and stage plan only; no model, Robot video, geometry or quality result.",
    }


def _active_packet() -> tuple[dict[str, Any], Path]:
    index = json.loads((PROJECT_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1:
        raise ColdStartInputError("CANDIDATE_PACKET_REQUIRED")
    packet_path = PROJECT_ROOT / active[0]["packet_path"]
    if file_ref(packet_path)["sha256"] != active[0]["packet_sha256"]:
        raise ColdStartInputError("ACTIVE_PACKET_SHA_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("execution_mode") != "EXACT78_CANDIDATE":
        raise ColdStartInputError("EXACT78_CANDIDATE_PACKET_REQUIRED")
    attempts = [(Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw).resolve()
                for raw in packet.get("write_set", [])
                if Path(raw).name.startswith("attempt_")]
    if (len(attempts) != 1
            or attempts[0].parent != PROJECT_ROOT / "_run/current" / packet["task_id"] / "attempts"):
        raise ColdStartInputError("CANDIDATE_ATTEMPT_WRITE_SET")
    return packet, attempts[0]


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    authorize_exact78_operation("run_exact78_raw_to_robot_v1", sys.argv[1:])
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    packet, attempt = _active_packet()
    output = validate_output_path(args.output_root, attempt_root=attempt)
    if output.exists():
        raise ColdStartInputError("FRESH_OUTPUT_REQUIRED")
    request = build_request(session=args.session, manifest=args.manifest)
    if packet.get("inputs", {}).get("submission_input_sha256") != request["submission_input_sha256"]:
        raise ColdStartInputError("SUBMISSION_INPUT_SHA_DRIFT")
    plan = candidate_plan(request)
    code = [file_ref(Path(__file__))]
    plan["preflight_signature"] = stage_signature(
        inputs=[item for source in request["sources"] for item in source["files"].values()],
        code=code, config={"physical_left_source_index": 1, "model_calls": 0},
        weights="NOT_CONSUMED", calibration="SOURCE_ONLY_UNVERIFIED",
        schema_version=plan["schema_version"],
    )
    output.mkdir(parents=True)
    _write_new(output / "INPUT_REQUEST.json", request)
    _write_new(output / "STAGE_PLAN.json", plan)
    _write_new(output / "RESULT.json", {
        "schema_version": "exact78-raw-to-robot-preflight-result-v1",
        "status": "PREPARED_NO_MODEL", "execution_complete": False,
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "input_request": file_ref(output / "INPUT_REQUEST.json"),
        "stage_plan": file_ref(output / "STAGE_PLAN.json"),
        "next_action": "T1_B_REGISTER_FRESH_HAWOR_ROBOT_CANDIDATE",
        "claim_limit": plan["claim_limit"],
    })
    print(json.dumps({"status": "PREPARED_NO_MODEL", "result": file_ref(output / "RESULT.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
