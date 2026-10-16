"""Unregistered, gated CPU Robot consumer for one sealed new-raw HaWoR stage.

This operation cannot be invoked until a separate algorithm-contract route and
single-writer Task Packet are published together. It never consumes old Robot
motion, fitted placement, Mask or Clean as action inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_newraw_robot_cpu_candidate_v1"


def _active_packet() -> tuple[dict[str, Any], Path]:
    from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref

    root = PROJECT_ROOT.resolve(strict=True)
    index = json.loads((root / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", [])
              if row.get("execution_allowed") is True]
    if len(active) != 1:
        raise RuntimeError("EXACT_ROBOT_CANDIDATE_PACKET_REQUIRED")
    row = active[0]
    task_id = row.get("task_id")
    path = root / "tasks/current" / str(task_id) / "TASK_PACKET.json"
    if (row.get("packet_path") != str(path.relative_to(root))
            or file_ref(path)["sha256"] != row.get("packet_sha256")):
        raise RuntimeError("ROBOT_PACKET_SHA_OR_PATH_DRIFT")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != task_id
            or packet.get("phase") != "EXACT78_NEWRAW_ROBOT_CPU_CANDIDATE"
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("ROBOT_PACKET_IDENTITY_DRIFT")
    attempt = root / "_run/current" / str(task_id) / "attempts/attempt_0001"
    if str(attempt.relative_to(root)) not in packet.get("write_set", []):
        raise RuntimeError("ROBOT_ATTEMPT_WRITE_SET_DRIFT")
    return packet, attempt


def run_candidate(*, hawor_result: Path, output_root: Path) -> dict[str, Any]:
    """Verify terminal lineage before importing or calling the Robot solver."""
    # Keep the live packet/command/writer fence even for a direct Python call.
    authorize_exact78_operation(
        OPERATION, ["--hawor-result", str(hawor_result),
                    "--output-root", str(output_root)])
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
    from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import (
        build_spec_from_sealed,
    )
    from chaoyang.governance.exact78_newraw_robot_handoff_candidate_v1 import (
        verify_handoff,
    )

    packet, attempt = _active_packet()
    expected_output = attempt / "candidate/robot"
    if output_root != expected_output or output_root.exists():
        raise RuntimeError("FRESH_ROBOT_OUTPUT_OUTSIDE_ATTEMPT_REQUIRED")
    inputs = packet.get("inputs") or {}
    if (hawor_result != Path(inputs.get("hawor_outer_result", {}).get("path", ""))
            or file_ref(hawor_result) != inputs.get("hawor_outer_result")):
        raise RuntimeError("HAWOR_OUTER_PACKET_REF_DRIFT")
    admission = build_spec_from_sealed(
        packet=packet,
        descriptor_ref=inputs["upstream_admission_descriptor"],
        frame_map_ref=inputs["upstream_admission_frame_time_map"],
        seal_ref=inputs["upstream_admission_seal"],
        producer_result_ref=inputs["upstream_admission_result"],
        producer_receipt_ref=inputs["upstream_admission_receipt"],
    )
    handoff = verify_handoff(outer_result=hawor_result, admission=admission)
    immutable = {
        "hawor_outer_result", "hawor_receipt", "hawor_packet", "hawor_stage_seal",
        "hawor_native_open_audit", "hawor_inner_result", "hawor_npz",
        "hawor_frame_provenance", "hawor_input_audit", "raw_video",
    }
    if (any(inputs.get(name) != handoff.get(name) for name in immutable)
            or inputs.get("source_files") != handoff["source_files"]
            or packet.get("config", {}).get("frame_count") != handoff["frame_count"]
            or packet.get("config", {}).get("source_kind") != handoff["source_kind"]
            or packet.get("config", {}).get("time_mode") != handoff["time_mode"]):
        raise RuntimeError("ROBOT_PACKET_HANDOFF_DRIFT")
    assets = inputs.get("robot_assets")
    if not isinstance(assets, dict) or not assets:
        raise RuntimeError("ROBOT_ASSET_PIN_ABSENT")
    for name, ref in assets.items():
        if file_ref(Path(ref["path"])) != ref:
            raise RuntimeError(f"ROBOT_ASSET_SHA_DRIFT:{name}")

    # Stage import happens only after the CLI pre-import gate and every sealed
    # same-session/ref/timeline check above. No new model or threshold is used.
    from chaoyang.pipeline.exact78_fresh_robot_stage_v2 import run_robot_stage_ffmpeg44

    result = run_robot_stage_ffmpeg44(
        hawor_npz=Path(handoff["hawor_npz"]["path"]),
        frame_provenance=Path(handoff["hawor_frame_provenance"]["path"]),
        input_audit=Path(handoff["hawor_input_audit"]["path"]),
        raw_video=Path(handoff["raw_video"]["path"]),
        source_kind=handoff["source_kind"], session_id=handoff["session_id"],
        output_root=output_root,
    )
    stage_path = output_root / "v1_stage_with_ffmpeg44_encoder/RESULT.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    rows_ref = (stage.get("outputs") or {}).get("frame_states")
    if (result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("frame_count") != handoff["frame_count"]
            or result.get("session_id") != handoff["session_id"]
            or not isinstance(rows_ref, dict)
            or file_ref(Path(rows_ref["path"])) != rows_ref):
        raise RuntimeError("ROBOT_STAGE_OR_FRAME_ROWS_DRIFT")
    return result


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    authorize_exact78_operation(OPERATION, args_in)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(args_in)
    result = run_candidate(hawor_result=args.hawor_result, output_root=args.output_root)
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref

    print(json.dumps({"status": result["status"],
                      "result": file_ref(args.output_root / "RESULT.json")},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
