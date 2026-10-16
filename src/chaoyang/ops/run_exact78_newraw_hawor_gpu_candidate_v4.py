"""Fenced py310-compatible HaWoR worker for a sealed new-raw CPU admission.

The worker intentionally does not import governance.common: that module needs
jsonschema, which is absent from the frozen HaWoR environment. The system
Python executor owns registration, native-open audit and stage sealing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation
from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref


OPERATION = "run_exact78_newraw_hawor_gpu_candidate_v4"


def _packet_for_worker() -> dict:
    index = json.loads((PROJECT_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [item for item in index.get("task_packets", [])
              if item.get("execution_allowed") is True]
    if len(active) != 1:
        raise RuntimeError("SINGLE_ACTIVE_PACKET_REQUIRED")
    task_id = active[0].get("task_id")
    if not isinstance(task_id, str) or not task_id.startswith("exact78_newraw_hawor_") or not task_id.endswith("_v4"):
        raise RuntimeError("V4_TASK_ID_REQUIRED")
    packet_path = PROJECT_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    if (PROJECT_ROOT / active[0]["packet_path"]).resolve() != packet_path:
        raise RuntimeError("ACTIVE_PACKET_PATH_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("task_id") != task_id or file_ref(packet_path)["sha256"] != active[0].get("packet_sha256"):
        raise RuntimeError("ACTIVE_PACKET_SHA_OR_ID_DRIFT")
    return packet


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    authorize_exact78_operation(OPERATION, arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(arguments)
    packet = _packet_for_worker()
    producer = packet["inputs"]["upstream_admission_result"]
    if args.producer_result.resolve(strict=True) != Path(producer["path"]).resolve(strict=True) or file_ref(args.producer_result) != producer:
        raise RuntimeError("PRODUCER_RESULT_COMMAND_BINDING_DRIFT")
    attempt = PROJECT_ROOT / "_run/current" / packet["task_id"] / "attempts/attempt_0001"
    if args.output_root.resolve() != attempt / "candidate/hawor":
        raise RuntimeError("V4_OUTPUT_ATTEMPT_DRIFT")

    # The frozen stage and model are imported only after both the operation
    # gate and exact source/attempt binding have succeeded.
    from chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1 import run_stage

    result = run_stage(packet, args.output_root)
    print(json.dumps({"status": result["status"],
                      "result": str(args.output_root / "RESULT.json"),
                      "native_open_audit_pending": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


