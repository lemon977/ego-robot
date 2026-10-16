"""Gated Cards001 sealed-HaWoR two-seed CPU diagnostic candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_newraw_two_seed_cpu_candidate_v1"
PHASE = "EXACT78_NEWRAW_TWO_SEED_CPU_CANDIDATE"
EXPECTED_HAWOR_SHA = "f04b819f06ec5745aeb585516415f4513b9a68cd1f05e7499dc8caa4cb80b1f5"
EXPECTED_SESSION = "play_cards_0902_001"


def _handoff(path: Path) -> dict[str, Any]:
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
    from chaoyang.governance.exact78_newraw_robot_handoff_candidate_v1 import verify_handoff
    from chaoyang.governance.execute_exact78_newraw_robot_cpu_candidate_v1 import _cpu_probe

    path = path.resolve(strict=True)
    if file_ref(path)["sha256"] != EXPECTED_HAWOR_SHA:
        raise RuntimeError("TWO_SEED_HAWOR_V4_SHA_DRIFT")
    _, _, admission = _cpu_probe(path)
    handoff = verify_handoff(outer_result=path, admission=admission)
    if (handoff["session_id"] != EXPECTED_SESSION or handoff["frame_count"] != 201
            or handoff["source_kind"] != "tracker"
            or handoff["hawor_outer_result"] != file_ref(path)):
        raise RuntimeError("TWO_SEED_HANDOFF_IDENTITY_DRIFT")
    return handoff


def _active_packet() -> tuple[dict[str, Any], Path]:
    from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref

    root = PROJECT_ROOT.resolve(strict=True)
    index = json.loads((root / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    rows = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(rows) != 1:
        raise RuntimeError("TWO_SEED_SINGLE_ACTIVE_PACKET_REQUIRED")
    indexed = rows[0]
    task_id = indexed.get("task_id")
    path = root / "tasks/current" / str(task_id) / "TASK_PACKET.json"
    if (indexed.get("packet_path") != str(path.relative_to(root))
            or indexed.get("packet_sha256") != file_ref(path)["sha256"]):
        raise RuntimeError("TWO_SEED_PACKET_REF_DRIFT")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != task_id or packet.get("phase") != PHASE
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("TWO_SEED_PACKET_IDENTITY_DRIFT")
    attempt = root / "_run/current" / str(task_id) / "attempts/attempt_0001"
    if str(attempt.relative_to(root)) not in packet.get("write_set", []):
        raise RuntimeError("TWO_SEED_ATTEMPT_WRITE_SET_DRIFT")
    return packet, attempt


def check_only(path: Path) -> dict[str, Any]:
    """Read-only input check; deliberately does not authorize registration."""
    handoff = _handoff(path)
    return {"status": "INPUTS_READY_ROUTE_AND_TASK_PACKET_NOT_REGISTERED",
            "operation": OPERATION, "phase": PHASE,
            "hawor_outer_result": handoff["hawor_outer_result"],
            "hawor_npz": handoff["hawor_npz"],
            "frame_provenance": handoff["hawor_frame_provenance"],
            "input_audit": handoff["hawor_input_audit"],
            "raw_video": handoff["raw_video"],
            "frame_count": handoff["frame_count"], "source_kind": handoff["source_kind"],
            "session_id": handoff["session_id"],
            "expected_output_schema": "exact78-newraw-two-seed-cpu-result-v1",
            "model_calls": 0, "render_calls": 0,
            "old_robot_result_producer_input": False}


def run_candidate(path: Path, output_root: Path) -> dict[str, Any]:
    authorize_exact78_operation(OPERATION, ["--hawor-result", str(path),
                                            "--output-root", str(output_root)])
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
    from chaoyang.pipeline.exact78_newraw_two_seed_cpu_candidate_v1 import run_cpu_candidate

    packet, attempt = _active_packet()
    expected_output = attempt / "candidate/two_seed_cpu"
    if output_root != expected_output or output_root.exists():
        raise RuntimeError("TWO_SEED_FRESH_ATTEMPT_OUTPUT_REQUIRED")
    handoff = _handoff(path)
    inputs = packet.get("inputs") or {}
    for key in ("hawor_outer_result", "hawor_npz", "hawor_frame_provenance",
                "hawor_input_audit", "raw_video"):
        if inputs.get(key) != handoff[key]:
            raise RuntimeError("TWO_SEED_PACKET_INPUT_DRIFT:" + key)
    if packet.get("config", {}).get("frame_count") != 201:
        raise RuntimeError("TWO_SEED_PACKET_FRAME_COUNT_DRIFT")
    for key in ("robot_asset_pin", "mount_proxy"):
        ref = inputs.get(key)
        if not isinstance(ref, dict) or file_ref(Path(ref["path"])) != ref:
            raise RuntimeError("TWO_SEED_ASSET_REF_DRIFT:" + key)
    result = run_cpu_candidate(
        hawor_npz=Path(handoff["hawor_npz"]["path"]),
        frame_provenance=Path(handoff["hawor_frame_provenance"]["path"]),
        input_audit=Path(handoff["hawor_input_audit"]["path"]),
        raw_video=Path(handoff["raw_video"]["path"]), output_root=output_root)
    if (result["status"] != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result["frame_count"] != 201):
        raise RuntimeError("TWO_SEED_STAGE_INCOMPLETE")
    return result


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if "--check-only" not in given:
        authorize_exact78_operation(OPERATION, given)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor-result", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(given)
    if args.check_only:
        if args.output_root is not None:
            raise RuntimeError("TWO_SEED_CHECK_ONLY_MUST_NOT_SET_OUTPUT")
        value = check_only(args.hawor_result)
    else:
        if args.output_root is None:
            raise RuntimeError("TWO_SEED_OUTPUT_REQUIRED")
        value = run_candidate(args.hawor_result, args.output_root)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
