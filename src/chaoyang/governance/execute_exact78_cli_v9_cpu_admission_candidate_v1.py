"""Check-only v9 bridge; future execution delegates to the published CPU lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.cli_v9 import cpu_admission_packet_v1 as bridge
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as cpu
from chaoyang.governance.common import REPO_ROOT, load_json


def check_only(session: Path) -> dict[str, Any]:
    prepared = bridge.prepare(session)
    packet = bridge.packet_value(prepared)
    state, index = load_json(cpu.TASK_STATE_PATH), load_json(cpu.INDEX)
    active = bool(state.get("next_task")
                  or any(row.get("status") in cpu.ACTIVE for row in state.get("tasks", []))
                  or any(row.get("execution_allowed") is True for row in index.get("task_packets", [])))
    used = any(row.get("task_id") == packet["task_id"] for row in state.get("tasks", []))
    attempt, packet_path, receipt = cpu._paths(packet["task_id"])
    if any(path.exists() for path in (attempt, packet_path, receipt)):
        used = True
    status = ("WAIT_ACTIVE_WRITER" if active else
              "EXISTING_TASK_ID_REQUIRES_SEALED_ATTACHMENT" if used else
              "READY_NOT_REGISTERED")
    descriptor = prepared["cpu"]["descriptor"]
    return {"status": status, "route": "PUBLISHED_CPU_CANDIDATE_ROUTE",
            "task_id_preview": packet["task_id"],
            "run_signature_preview": packet["run_signature"],
            "source_kind": descriptor["source_kind"],
            "source_session": descriptor["session_id"],
            "source_files": descriptor["source_files"],
            "v7_source_only_sha256": prepared["v7"]["submission_input_sha256"],
            "descriptor_sha256": prepared["cpu"]["descriptor_sha256"],
            "decoded_frames": descriptor["decoded_frames"],
            "time_mode": descriptor["time_mode"],
            "physical_eye": descriptor["physical_eye"],
            "packet_schema": packet["schema_version"],
            "operation": cpu.OPERATION,
            "registration_started": False, "model_calls": 0,
            "gpu_count": 0, "visual_qualified": False,
            "training_eligible": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(check_only(args.session), ensure_ascii=False, sort_keys=True))
        return 0
    decision = check_only(args.session)
    if decision["status"] != "READY_NOT_REGISTERED":
        raise RuntimeError("V9_CPU_ADMISSION_NOT_READY:" + decision["status"])
    prepared = bridge.prepare(args.session)
    packet = bridge.packet_value(prepared)
    if (packet["task_id"] != decision["task_id_preview"]
            or packet["run_signature"] != decision["run_signature_preview"]):
        raise RuntimeError("V9_RECHECK_SIGNATURE_DRIFT")
    cpu.register(packet)
    result = cpu.run_registered_child(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
