"""Thin v7 identity bridge to the existing governed CPU admission Packet.

This module decodes and validates the raw session during read-only preparation.
It adds the accepted v7 source identity to the existing dynamic CPU Packet;
the published v1 operation and worker remain the execution route.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from typing import Any

from chaoyang.cli_v7.source_neutral_plan_v1 import plan_one
from chaoyang.cli_v8.source_neutral_stage_packet_plan_v1 import _accepted_code
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as cpu
from chaoyang.governance.common import REPO_ROOT, artifact_ref, canonical_bytes, validate_artifact_ref
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError
from chaoyang.governance.register_single_task_packet import _validate_packet


WALL_SECONDS_MAX = 1800
BRIDGE_RUNTIME = (
    "src/chaoyang/cli_v9/cpu_admission_packet_v1.py",
    "src/chaoyang/governance/execute_exact78_cli_v9_cpu_admission_candidate_v1.py",
    "src/chaoyang/cli_v7/source_neutral_plan_v1.py",
    "src/chaoyang/cli_v8/source_neutral_stage_packet_plan_v1.py",
)


def prepare(session: Path) -> dict[str, Any]:
    """Read-only v7/source-to-decoded-descriptor cross-check."""
    accepted = _accepted_code()
    source = plan_one(session)
    decoded = cpu.preflight(session)
    descriptor = decoded["descriptor"]
    v7 = source["source"]
    if (descriptor["source_kind"] != v7["source_kind"]
            or descriptor["session_root"] != v7["session_root"]
            or descriptor["session_id"] != v7["session_id"]
            or descriptor["source_files"] != v7["source_files"]
            or descriptor["physical_eye"] != v7["physical_eye"]
            or descriptor["time_mode"] != v7["time_mode"]
            or descriptor["capture_to_decode_alignment"] != v7["capture_to_decode_alignment"]
            or descriptor["decoded_frames"] != v7["frame_count_declared"]
            or not descriptor["frame_rows"]):
        raise ColdStartInputError("V9_V7_SOURCE_TO_DECODED_DESCRIPTOR_DRIFT")
    for ref in descriptor["source_files"].values():
        if errors := validate_artifact_ref(ref):
            raise ColdStartInputError("V9_SOURCE_SHA_DRIFT_AFTER_DECODE:" + ";".join(errors))
    return {"v7": source, "cpu": decoded, "code_acceptance": accepted}


def packet_value(prepared: dict[str, Any]) -> dict[str, Any]:
    """Add v7 identity to v1's real dynamic Packet without changing its route."""
    packet = copy.deepcopy(cpu.packet_value(prepared["cpu"]))
    descriptor = prepared["cpu"]["descriptor"]
    if (packet["inputs"]["source_files"] != descriptor["source_files"]
            or packet["inputs"]["descriptor_sha256"] != prepared["cpu"]["descriptor_sha256"]):
        raise RuntimeError("V9_CPU_PACKET_DESCRIPTOR_DRIFT")
    runtime = list(packet["runtime_code"])
    already = {ref["path"] for ref in runtime}
    for relative in BRIDGE_RUNTIME:
        ref = artifact_ref(REPO_ROOT / relative)
        if ref["path"] not in already:
            runtime.append(ref)
            already.add(ref["path"])
    accepted = prepared["code_acceptance"]
    inputs = {**packet["inputs"],
              "v7_source_only_sha256": prepared["v7"]["submission_input_sha256"],
              "v7_code_acceptance_receipt": accepted["receipt"],
              "v7_code_acceptance_result": accepted["result"]}
    config = {**packet["config"], "wall_seconds_max": WALL_SECONDS_MAX,
              "cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "training_steps": 0, "source_kind": descriptor["source_kind"],
              "time_mode": descriptor["time_mode"],
              "world_pose_authority": False, "metric_authority": False}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": packet["commands"][0],
        "schema": "exact78-first-submit-cpu-producer-v1",
    })).hexdigest()
    packet.update(inputs=inputs, runtime_code=runtime, config=config,
                  run_signature=signature,
                  deadline_at=(datetime.now().astimezone() +
                               timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
                  budgets={"workers_max": 1, "cpu_threads_total": 2,
                           "gpu_count": 0, "model_calls": 0, "training_steps": 0,
                           "wall_seconds_max": WALL_SECONDS_MAX,
                           "session_attempts_max": 1},
                  claim_limit="Accepted-v7 source and decoded CPU frame admission only; no HaWoR, Robot, Scene, visual, control or training qualification")
    if errors := _validate_packet(packet):
        raise RuntimeError("V9_PACKET_INVALID:" + ";".join(errors))
    readiness = cpu._assert_cli_admission_ready(packet)
    if readiness != "PUBLISHED_CPU_CANDIDATE_ROUTE":
        raise RuntimeError("V9_EXISTING_CPU_ROUTE_NOT_PUBLISHED")
    return packet


__all__ = ["prepare", "packet_value"]
