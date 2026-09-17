from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Fail-closed G0 core audit and immutable result publication for V7.1."""

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    RECEIPT_PATH,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    process_identity,
    sha256_file,
)
from chaoyang.governance.v52_contracts import atomic_write_new, validate_task_packet


ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/governance/EXACT78_V7_1_EXECUTION_PLAN_ZH.md"
PLAN_REVISION_PATH = ROOT / "docs/governance/PLAN_REVISION.json"
PACKET_INDEX_PATH = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/TASK_PACKET_INDEX.json"
DEFAULT_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/g0_core/RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    errors: list[str] = []
    receipt = load_json(RECEIPT_PATH)
    state = load_json(TASK_STATE_PATH)
    plan_revision = load_json(PLAN_REVISION_PATH)
    packet_index = load_json(PACKET_INDEX_PATH)

    plan_ref = artifact_ref(PLAN_PATH)
    approved = plan_revision.get("approved_source", {})
    if plan_revision.get("plan_revision") != "chaoyang-v7.1":
        errors.append("plan revision is not chaoyang-v7.1")
    if plan_ref["bytes"] != approved.get("bytes") or plan_ref["sha256"] != approved.get("sha256"):
        errors.append("canonical plan bytes/SHA mismatch")
    if len(PLAN_PATH.read_text(encoding="utf-8").splitlines()) != approved.get("lines"):
        errors.append("canonical plan line count mismatch")

    indexed = packet_index.get("task_packets", [])
    if len(indexed) != 22:
        errors.append(f"expected 22 V7.1 packets, found {len(indexed)}")
    for row in indexed:
        path = ROOT / row["packet_path"]
        if not path.is_file() or sha256_file(path) != row["packet_sha256"]:
            errors.append(f"packet ref mismatch: {path}")
            continue
        packet_errors = validate_task_packet(load_json(path))
        if packet_errors:
            errors.append(f"invalid packet {path}: {packet_errors}")

    active_dead = []
    for task in state.get("tasks", []):
        if task.get("status") not in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
            continue
        pid = task.get("pid")
        ticks = task.get("proc_start_ticks")
        if pid is None:
            continue
        identity = process_identity(int(pid))
        if not identity["alive"] or (ticks is not None and identity["start_ticks"] != ticks):
            active_dead.append(task["task_id"])
    if active_dead:
        errors.append(f"dead active tasks remain: {active_dead}")

    robot = next((t for t in state.get("tasks", []) if t.get("task_id") == "exact78_v52_lane_c_contact_robot"), None)
    if not robot or robot.get("status") != "FAILED_RUNTIME_FINAL" or robot.get("pid") is not None:
        errors.append("legacy dead Robot task is not closed with cleared runtime identity")

    legacy_lease_path = ROOT / "_run/current/GPU_LEASE.json"
    legacy_lease = load_json(legacy_lease_path) if legacy_lease_path.is_file() else {"status": "ABSENT"}
    if legacy_lease.get("status") == "ACQUIRED":
        errors.append("legacy GPU lease remains acquired")

    result = {
        "schema_version": "chaoyang-g0-core-result-v71",
        "status": "PASSED" if not errors else "FAILED_RUNTIME_FINAL",
        "task_id": "g0_core_governance_v71",
        "plan_revision": "chaoyang-v7.1",
        "checks": {
            "current_receipt_revision": receipt["governance_revision"],
            "canonical_plan": plan_ref,
            "packet_index": artifact_ref(PACKET_INDEX_PATH),
            "packet_count": len(indexed),
            "active_dead_tasks": active_dead,
            "legacy_robot_terminal": robot.get("status") if robot else "MISSING",
            "legacy_gpu_lease_status": legacy_lease.get("status"),
            "v71_gpu_lease_contract": artifact_ref(ROOT / "docs/governance/schemas/gpu_lease_v71.schema.json"),
        },
        "errors": errors,
        "claim_limit": "Core governance closure only; no algorithm, contact, Robot, training or physical authority.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(args.output, payload)
    print(args.output)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
