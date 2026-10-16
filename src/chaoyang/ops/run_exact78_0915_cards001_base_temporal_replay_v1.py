"""Unregistered, fenced 0915 Cards001 CPU base temporal replay operation.

The CLI exposes a read-only preflight. A replay requires the exact active
candidate command, a single fenced writer, and an independently published
Task Packet. This module does not register or publish any route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_0915_cards001_base_temporal_replay_v1"
PHASE = "EXACT78_0915_CARDS001_BASE_TEMPORAL_REPLAY_CPU_CANDIDATE"
TASK_ID = "exact78_0915_cards001_base_temporal_replay_v1_20261001"
OUTPUT_NAME = "candidate/base_temporal_cpu"


def check_only() -> dict[str, Any]:
    from chaoyang.pipeline.exact78_0915_cards001_base_temporal_replay_v1 import check_only as verify

    value = verify()
    return {"status": "INPUTS_READY_ROUTE_AND_TASK_PACKET_NOT_REGISTERED",
            "operation": OPERATION, "phase": PHASE,
            "task_id_draft": TASK_ID, "preflight": value,
            "model_calls": 0, "render_calls": 0, "training_steps": 0}


def _active_packet() -> tuple[dict[str, Any], Path]:
    from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref

    root = PROJECT_ROOT.resolve(strict=True)
    index = json.loads((root / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    rows = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(rows) != 1 or rows[0].get("task_id") != TASK_ID:
        raise RuntimeError("BASE_TEMPORAL_SINGLE_MATCHING_PACKET_REQUIRED")
    path = root / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    if (rows[0].get("packet_path") != str(path.relative_to(root))
            or rows[0].get("packet_sha256") != file_ref(path)["sha256"]):
        raise RuntimeError("BASE_TEMPORAL_PACKET_REF_DRIFT")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != TASK_ID or packet.get("phase") != PHASE
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("BASE_TEMPORAL_PACKET_IDENTITY_DRIFT")
    attempt = root / "_run/current" / TASK_ID / "attempts/attempt_0001"
    if str(attempt.relative_to(root)) not in packet.get("write_set", []):
        raise RuntimeError("BASE_TEMPORAL_WRITE_SET_DRIFT")
    return packet, attempt


def run_candidate(output_root: Path) -> dict[str, Any]:
    authorize_exact78_operation(OPERATION, ["--output-root", str(output_root)])
    from chaoyang.pipeline import exact78_0915_cards001_base_temporal_replay_v1 as replay

    packet, attempt = _active_packet()
    if output_root != attempt / OUTPUT_NAME or output_root.exists():
        raise RuntimeError("BASE_TEMPORAL_FRESH_OUTPUT_REQUIRED")
    preflight = replay.check_only()
    inputs = packet.get("inputs") or {}
    for key in ("input_refs", "raw_source_ref", "robot_asset_pin", "mount_proxy"):
        if inputs.get(key) != preflight[key]:
            raise RuntimeError("BASE_TEMPORAL_PACKET_INPUT_DRIFT:" + key)
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
    if file_ref(Path(replay.__file__)) not in packet.get("runtime_code", []):
        raise RuntimeError("BASE_TEMPORAL_EXECUTOR_CODE_NOT_PINNED")
    config = packet.get("config") or {}
    expected = {"session_id": replay.SESSION, "frame_count": replay.FRAMES,
                "top_k": 8, "spatial_candidate_count": 32,
                "gpu_count": 0, "model_calls": 0, "render_calls": 0,
                "training_steps": 0, "old_robot_q_input": False,
                "numeric_gates_changed": False, "ik_changed": False,
                "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("BASE_TEMPORAL_PACKET_CONFIG_DRIFT")
    budgets = packet.get("budgets") or {}
    if (budgets.get("session_attempts_max") != 1 or budgets.get("gpu_count") != 0
            or not isinstance(budgets.get("wall_seconds_max"), int)
            or not 0 < budgets["wall_seconds_max"] <= 3600):
        raise RuntimeError("BASE_TEMPORAL_BUDGET_DRIFT")
    result = replay.run_replay(output_root)
    if (result["status"] != "COMPLETE_CPU_NUMERIC_CANDIDATE_NOT_VISUAL_QUALIFICATION"
            or len(result["candidates"]) != 8):
        raise RuntimeError("BASE_TEMPORAL_INCOMPLETE_RESULT")
    return result


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if "--check-only" not in given:
        authorize_exact78_operation(OPERATION, given)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(given)
    if args.check_only:
        if args.output_root is not None:
            raise RuntimeError("BASE_TEMPORAL_CHECK_ONLY_MUST_NOT_SET_OUTPUT")
        value = check_only()
    else:
        if args.output_root is None:
            raise RuntimeError("BASE_TEMPORAL_OUTPUT_REQUIRED")
        value = run_candidate(args.output_root)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
