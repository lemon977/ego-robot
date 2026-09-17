#!/usr/bin/env python3
"""Freeze and CAS-register the bounded R2.2 integration task graph."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import (  # noqa: E402
    ALGORITHM_CONTRACT_PATH,
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
    sha256_bytes,
    validate_artifact_ref,
)


RUN_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h"
ATTEMPT_ROOT = RUN_ROOT / "g0/attempts/attempt_0001"
PACKET_ROOT = RUN_ROOT / "task_packets"
HARD_SOFT_INDEX = (
    ROOT
    / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_v76_r3/ROBOT_HARD_SOFT_CANDIDATE_INDEX.json"
)
GPU_LEASE = ROOT / "_run/current/GPU_LEASE.json"
PLAN = ROOT / "docs/governance/CHAOYANG_R2_2_4H_INTEGRATION_PLAN_ZH.md"


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "robot_v77_adopt12_r22",
        "phase": "ROBOT_V77_ADOPT_12",
        "objective": "Publish twelve single-session terminals from closed hard/soft evidence only.",
        "prerequisites": ["g0_r22=PASSED", "v76=FAILED_RUNTIME_FINAL", "hard_soft_index=SHA_PINNED"],
        "expected_resource": "CPU",
    },
    *[
        {
            "task_id": f"robot_v77_batch_{index:03d}_r22",
            "phase": f"ROBOT_V77_BATCH_{index:03d}",
            "objective": f"Close immutable v77 execution batch {index:03d} with unchanged v76 algorithm signature.",
            "prerequisites": ["g0_r22=PASSED", "v77_adopt12_terminal"],
            "expected_resource": "CPU",
        }
        for index in range(1, 6)
    ],
    {
        "task_id": "robot_v77_terminal_index_r22",
        "phase": "ROBOT_V77_TERMINAL_INDEX",
        "objective": "Aggregate v77 microbatch terminals without consuming partial output.",
        "prerequisites": ["all_started_v77_batches_terminal"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "robot_yield_20_r22",
        "phase": "ROBOT_YIELD_20",
        "objective": "Report Robot conversion rates on processed, frozen-25 and upstream-101 denominators.",
        "prerequisites": ["v77_terminal_index_available"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "robot_v78_target_reach_canary_r22",
        "phase": "ROBOT_V78_TARGET_REACH_CANARY",
        "objective": "Run separate causal target/reach canary without changing v77 hard gates.",
        "prerequisites": ["v77_minimum_terminal", "fixed_canary_and_regressions"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "mask_independent_canary_r22",
        "phase": "MASK_INDEPENDENT_EVALUATION",
        "objective": "Run one Poker and one Chips SAM3.1 canary against a frozen independent evaluation reference.",
        "prerequisites": ["evaluation_reference_frozen", "gpu_lease_available"],
        "expected_resource": "GPU_CANARY",
    },
    {
        "task_id": "clean_prereq_r22",
        "phase": "CLEAN_CAUSAL_PREREQUISITES",
        "objective": "Close semantic donor rejection, lossless source map and causal Poker atlas for Poker245 and Chips039.",
        "prerequisites": ["mask_canary_terminal"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "depth_rectification_r22",
        "phase": "DEPTH_RECTIFICATION_DIAGNOSTIC",
        "objective": "Publish reproducible same-session rectification diagnostics and gate FoundationStereo.",
        "prerequisites": ["same_session_sensor_inputs"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "contact10_real_canary_r22",
        "phase": "CONTACT10_REAL_CANARY",
        "objective": "Run at least one real-session per-finger HYPOTHESIS_ONLY contact canary.",
        "prerequisites": ["direct_object6d_or_independent_tracking", "evidence_dag_acyclic"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "occlusion_base_r22",
        "phase": "BASE_GEOMETRIC_OCCLUSION",
        "objective": "Run base z-buffer occlusion independently of Contact and publish provenance/UNKNOWN.",
        "prerequisites": ["same_session_robot_clean_depth_object_intersection"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "doc_r22",
        "phase": "DOCUMENT_AGGREGATOR",
        "objective": "CAS-publish receipt-bound task state, current docs and visual review index.",
        "prerequisites": ["worker_receipts_only"],
        "expected_resource": "CPU",
    },
    {
        "task_id": "cleanup_current_only_r72_dryrun",
        "phase": "CLEANUP_DRY_RUN_ONLY",
        "objective": "Delete ordinary regenerated caches and produce a protected old-asset dry-run manifest; do not delete runs or history.",
        "prerequisites": ["protection_snapshot_frozen", "active_read_write_sets_known"],
        "expected_resource": "LOW_PRIORITY_CPU_IO",
    },
]


def _write_once_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, encoded)


def _freeze_file(source: Path, bundle_root: Path) -> dict[str, Any]:
    """Copy mutable control-plane bytes into the immutable G0 bundle."""
    reference = artifact_ref(source)
    destination = bundle_root / reference["sha256"] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if artifact_ref(destination)["sha256"] != reference["sha256"]:
            raise RuntimeError(f"content-addressed freeze conflict: {destination}")
    else:
        shutil.copyfile(source, destination)
    frozen = artifact_ref(destination)
    if frozen["bytes"] != reference["bytes"] or frozen["sha256"] != reference["sha256"]:
        raise RuntimeError(f"content-addressed freeze verification failed: {source}")
    return {"source": reference, "frozen": frozen}


def _run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return (result.stdout + "\n" + result.stderr).strip()
    return result.stdout.strip()


def _validate_current(receipt: dict[str, Any]) -> None:
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))


def _selection(index: dict[str, Any]) -> dict[str, Any]:
    hard = sorted(
        [row for row in index.get("rows", []) if row.get("hard_geometry_pass") is True],
        key=lambda row: str(row.get("session")),
    )
    chips = [row["session"] for row in hard if row.get("task") == "chips"]
    poker = [row["session"] for row in hard if row.get("task") == "poker"]
    return {
        "schema_version": "chaoyang-r22-integration-selection-v1",
        "fixed_clean_sessions": ["play_cards_0903_245", "get_potato_chips_0902_039"],
        "fixed_contact_sessions": ["play_cards_0903_245", "play_cards_0903_243", chips[0] if chips else None],
        "robot_occlusion_chips": chips[0] if chips else None,
        "robot_occlusion_poker": poker[0] if poker else None,
        "v78_failure_canary": "play_cards_0903_189",
        "v78_chips_regression": chips[0] if chips else None,
        "v78_poker_regression": poker[0] if poker else None,
        "selection_policy": "Lexicographically first current hard-pass session per task; missing task remains BLOCKED_PREREQ.",
        "authority_promoted": False,
    }


def _packet(task: dict[str, Any], snapshot_path: Path, selection_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": task["task_id"],
        "plan_revision": "R2.2",
        "phase": task["phase"],
        "objective": task["objective"],
        "read_set": [str(snapshot_path), str(selection_path), str(PLAN)],
        "write_set": [str(RUN_ROOT / "lanes" / task["task_id"] / "attempts")],
        "prerequisites": task["prerequisites"],
        "budgets": {"max_runtime_attempts": 2, "gpu_wait_s": 1800, "four_hour_parent_cutoff": True},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE", "BLOCKED_REFERENCE_PROOF", "CANCELLED"],
        "required_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"],
        "claim_limit": "R2.2 bounded development task; no automatic stage, control, physical or external-truth authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if ATTEMPT_ROOT.exists():
        raise RuntimeError(f"fresh G0 attempt required: {ATTEMPT_ROOT}")

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before R2.2 freeze")
    _validate_current(receipt)
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    predecessor_path = ROOT / str(pointer["index_path"])
    if validate_artifact_ref(artifact_ref(predecessor_path)):
        raise RuntimeError("current Task Packet index is not closed")
    hard_soft = load_json(HARD_SOFT_INDEX)

    ATTEMPT_ROOT.mkdir(parents=True)
    git_status_path = ATTEMPT_ROOT / "GIT_STATUS_SNAPSHOT.txt"
    process_path = ATTEMPT_ROOT / "PROCESS_AND_GPU_SNAPSHOT.txt"
    atomic_write(git_status_path, (_run(["git", "status", "--short"]) + "\n").encode())
    process_text = "\n\n".join(
        [
            _run(["ps", "-eo", "pid,lstart,stat,cmd"]),
            _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        ]
    )
    atomic_write(process_path, (process_text + "\n").encode())

    selection_path = ATTEMPT_ROOT / "INTEGRATION_SELECTION.json"
    _write_once_json(selection_path, _selection(hard_soft))
    snapshot_path = ATTEMPT_ROOT / "RUN_START_SNAPSHOT.json"
    frozen_root = ATTEMPT_ROOT / "input_snapshot"
    receipt_freeze = _freeze_file(RECEIPT_PATH, frozen_root)
    packet_index_freeze = _freeze_file(predecessor_path, frozen_root)
    algorithm_contract_freeze = _freeze_file(ALGORITHM_CONTRACT_PATH, frozen_root)
    hard_soft_freeze = _freeze_file(HARD_SOFT_INDEX, frozen_root)
    snapshot = {
        "schema_version": "chaoyang-r22-run-start-snapshot-v1",
        "created_at": now_iso(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "current_status_receipt": receipt_freeze,
        "generation_id": receipt["generation_id"],
        "governance_revision": receipt["governance_revision"],
        "task_packet_index": packet_index_freeze,
        "algorithm_contract": algorithm_contract_freeze,
        "robot_hard_soft_candidate_index": hard_soft_freeze,
        "integration_selection": artifact_ref(selection_path),
        "git_status": artifact_ref(git_status_path),
        "process_and_gpu_snapshot": artifact_ref(process_path),
        "gpu_lease": artifact_ref(GPU_LEASE) if GPU_LEASE.is_file() else None,
        "worker_input_policy": "PINNED_AT_G0_NO_LATEST_AUTO_SWITCH",
        "external_protected": ["/mnt/data/egodata", "/nas/chenxianchi"],
    }
    _write_once_json(snapshot_path, snapshot)

    packet_entries: list[dict[str, Any]] = []
    predecessor = load_json(predecessor_path)
    existing = list(predecessor.get("task_packets", []))
    existing_ids = {str(row.get("task_id")) for row in existing}
    for task in TASKS:
        packet_path = PACKET_ROOT / task["task_id"] / "TASK_PACKET.json"
        _write_once_json(packet_path, _packet(task, snapshot_path, selection_path))
        if task["task_id"] not in existing_ids:
            packet_entries.append(
                {
                    "task_id": task["task_id"],
                    "packet_path": str(packet_path.relative_to(ROOT)),
                    "packet_sha256": artifact_ref(packet_path)["sha256"],
                }
            )

    created = now_iso()
    state_by_id = {str(row.get("task_id")): row for row in state.get("tasks", [])}
    v76 = state_by_id.get("robot_geometry_expansion_v76")
    if not v76 or v76.get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("R2.2 requires v76 FAILED_RUNTIME_FINAL")
    for task in TASKS:
        if task["task_id"] in state_by_id:
            continue
        state["tasks"].append(
            {
                "task_id": task["task_id"],
                "phase": task["phase"],
                "attempt": 0,
                "status": "PENDING",
                "updated_at": created,
                "heartbeat_at": None,
                "session": None,
                "pid": None,
                "proc_start_ticks": None,
                "gpu_id": None,
                "task_packet": artifact_ref(PACKET_ROOT / task["task_id"] / "TASK_PACKET.json"),
            }
        )
    state["next_task"] = {
        "task_id": "robot_v77_adopt12_r22",
        "session": None,
        "prerequisites": TASKS[0]["prerequisites"],
        "expected_resource": "CPU publisher/reviewer; unchanged Robot algorithm",
        "stop_condition": "Publish twelve single-session terminals or a bounded failure/block receipt.",
    }

    g0_result_path = ATTEMPT_ROOT / "RESULT.json"
    g0_result = {
        "schema_version": "chaoyang-r22-g0-result-v1",
        "task_id": "g0_r22_integration",
        "status": "PASSED",
        "created_at": created,
        "run_start_snapshot": artifact_ref(snapshot_path),
        "integration_selection": artifact_ref(selection_path),
        "registered_task_count": len(TASKS),
        "v76_terminal_verified": True,
        "next_task_replaced_by_cas": "robot_v77_adopt12_r22",
        "authority_promoted": False,
    }
    _write_once_json(g0_result_path, g0_result)
    _write_once_json(ATTEMPT_ROOT / "METRICS.json", {"status": "PASSED", "registered_tasks": len(TASKS), "hard_soft_sessions": hard_soft.get("counts", {}).get("sessions"), "hard_soft_pass": hard_soft.get("counts", {}).get("hard_geometry_pass")})
    _write_once_json(ATTEMPT_ROOT / "NEXT_ACTION.json", {"task_id": "robot_v77_adopt12_r22", "action": "RUN_PINNED_V77_ADOPT"})
    atomic_write(ATTEMPT_ROOT / "DECISION.md", b"# G0 R2.2\n\nPinned R2.2 inputs and replaced the terminal v76 next-task route through one CAS publication.\n")
    g0_manifest_path = ATTEMPT_ROOT / "ARTIFACT_MANIFEST.json"
    _write_once_json(
        g0_manifest_path,
        {
            "schema_version": "chaoyang-r22-g0-artifact-manifest-v1",
            "artifacts": [artifact_ref(path) for path in (snapshot_path, selection_path, g0_result_path, ATTEMPT_ROOT / "METRICS.json", ATTEMPT_ROOT / "NEXT_ACTION.json", ATTEMPT_ROOT / "DECISION.md")],
        },
    )
    _write_once_json(
        ATTEMPT_ROOT / "RUN_RECEIPT.json",
        {
            "schema_version": "chaoyang-r22-g0-run-receipt-v1",
            "task_id": "g0_r22_integration",
            "status": "PASSED",
            "result": artifact_ref(g0_result_path),
            "manifest": artifact_ref(g0_manifest_path),
            "producer_code_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
    )
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": "g0_r22_integration",
                "session": None,
                "attempt": 1,
                "status": "PASSED",
                "created_at": created,
                "message": "R2.2 start snapshot frozen; terminal v76 next-task route replaced; bounded lanes registered.",
                "result": artifact_ref(g0_result_path),
            }
        ]
    )[-100:]

    successor_index_path = ATTEMPT_ROOT / "TASK_PACKET_INDEX.json"
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "R7_3_R2_2",
        "plan_revision": "chaoyang-v7.1-r3+r2.2",
        "execution_revision": "R2.2",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "task_packets": existing + packet_entries,
        "claim_limit": "R2.2 bounded execution routing; registrations do not prove task execution or authority.",
    }
    published = publish_bundle(
        authority,
        state,
        event_type="R22_G0_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=successor_index_path,
        task_packet_index_value=successor_index,
    )
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "snapshot": str(snapshot_path), "next_task": state["next_task"]["task_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
