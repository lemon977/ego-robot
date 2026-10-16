"""Fenced CPU admission worker for one pre-registered Exact78 raw session.

This candidate is intentionally absent from the current algorithm contract.
It cannot be run until a separate finite producer packet and contract route are
published. It never self-registers and never invokes HaWoR or Robot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


OPERATION = "run_exact78_newraw_cpu_admission_candidate_v1"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Direct ``python -m`` is not an escape hatch: this same gate must pass
    # before importing even the CPU producer implementation.
    from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation

    authorize_exact78_operation(OPERATION, arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(arguments)

    from chaoyang.governance.common import REPO_ROOT, atomic_json, load_json
    from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer

    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    allowed = [row for row in index.get("task_packets", [])
               if row.get("execution_allowed") is True]
    if len(allowed) != 1:
        raise RuntimeError("CPU_ADMISSION_ACTIVE_PACKET_REQUIRED")
    task_id = allowed[0]["task_id"]
    attempt, packet_path, _ = producer._paths(task_id)
    packet = load_json(packet_path)
    session = args.session.resolve(strict=True)
    if (args.output_root.resolve(strict=True) != attempt.resolve(strict=True)
            or packet.get("task_id") != task_id
            or packet.get("inputs", {}).get("source_root") != str(session)
            or packet.get("writer", {}).get("pid") == os.getpid()):
        raise RuntimeError("CPU_ADMISSION_WORKER_PACKET_OR_PATH_DRIFT")
    prepared = producer.preflight(session)
    if (prepared["descriptor_sha256"] != packet["inputs"].get("descriptor_sha256")
            or prepared["descriptor"]["source_files"] != packet["inputs"].get("source_files")
            or prepared["descriptor"]["submission_input_sha256"]
               != packet["inputs"].get("submission_input_sha256")):
        raise RuntimeError("CPU_ADMISSION_WORKER_SOURCE_DRIFT")
    refs = producer.produce_admission(prepared, packet, cli_worker=True)
    stage_path = attempt / "ADMISSION_STAGE.json"
    if stage_path.exists():
        raise RuntimeError("IMMUTABLE_ADMISSION_STAGE_EXISTS")
    atomic_json(stage_path, {
        "schema_version": "exact78-first-submit-cpu-admission-stage-v1",
        "task_id": task_id, "source_session": prepared["descriptor"]["session_id"],
        "descriptor_sha256": prepared["descriptor_sha256"],
        "status": "SEALED_CPU_ADMISSION_STAGE", "admission": refs,
        "model_calls": 0, "gpu_count": 0,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
