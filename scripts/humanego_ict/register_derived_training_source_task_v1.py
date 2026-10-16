"""Register one bounded CPU engineering successor using the existing publisher.

Run the publisher and validators with the same real /proc view, outside the
virtualized process view. No model, training, SSH, SDK or hardware invocation. Keep stdin open; terminate
with ``finish <RESULT.json>`` after CPU verification. New attempts only.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import secrets
import select
import sys
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "pick_place_humanego_ict_derived_training_source_cpu_v1_20261010"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1"
EVIDENCE = REPO_ROOT / "_run/handoffs/revo2_ict_readonly_info_20261010/codex_20261010T104651Z_964a1dcb/audits/audit_20261010T111954Z_a0cce169/revision_v2"
WALL_SECONDS = 7200


def publish(state: dict, index: dict, event: str) -> None:
    index["supersedes_index"] = artifact_ref(INDEX)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), copy.deepcopy(load_json(INDEX))
    if (state.get("next_task") or any(r.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for r in state["tasks"])
            or any(r.get("execution_allowed") for r in index["task_packets"])):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_ATTEMPT_REQUIRED")
    receipt = load_json(RECEIPT_PATH)
    if errors := validate_artifact_ref(receipt["files"]["algorithm_contract"]):
        raise RuntimeError(str(errors))
    contract = load_json(Path(receipt["files"]["algorithm_contract"]["path"]))
    protected = contract["special_status_contracts"]["pick_place_humanego_ict_revo2_candidate_v1"]["code_closure"]
    derived_protected = contract["special_status_contracts"]["pick_place_humanego_ict_derived_revo2_v1"]["code_closure"]
    for ref in [*protected, *derived_protected]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError(str(errors))
    inputs = {
        "protected_real_control_code": protected,
        "protected_derived_code": derived_protected,
        "baseline_pointer": artifact_ref(REPO_ROOT / "docs/current/DUAL_BRANCH_BASELINE_V1.json"),
        "evidence": [artifact_ref(EVIDENCE / name) for name in
                     ("REPORT_ZH.md", "HAND_CHANNELS.csv", "EVIDENCE_INDEX.json", "MISSING_ITEMS.md")],
        "plan": artifact_ref(REPO_ROOT / PLAN / "TRAINING_SOURCE_EXECUTION_ZH.md"),
    }
    stat = Path("/proc/self/stat").read_text()
    writer = {"pid": int(stat.split(" ", 1)[0]),
              "proc_start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Wire separately named derived ICT trainer and tensor-only inference export; audit actual source mapping; CPU only",
        "phase": "HUMANEGO_ICT_DERIVED_TRAINING_SOURCE_CPU_ENGINEERING",
        "plan_revision": "PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1",
        "execution_revision": TASK.upper(), "execution_mode": "CPU_DEVELOPMENT_ONLY",
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
                     "docs/governance/ALGORITHM_CONTRACT.json", "tasks/current/INDEX.json",
                     str((EVIDENCE / "REPORT_ZH.md").relative_to(REPO_ROOT)),
                     "src/chaoyang/humanego_ict/data.py", "src/chaoyang/humanego_ict/model.py",
                     "src/chaoyang/governance/current_r3_contracts.py"],
        "read_set_expansions": ["SHA-bound v2 evidence and explicitly referenced SDK/retarget/IK/loader snapshots",
                                "existing humanego_ict source/tests and HumanEgo dependencies",
                                "project governance registration/publishing implementation and baseline pointer",
                                "new engineering source/tests/plan; read-only representative NAS metadata and source group/calibration records; no model, GPU or hardware"],
        "write_set": ["src/chaoyang/humanego_ict/source_adapter.py",
                      "src/chaoyang/humanego_ict/derived_training_cli.py",
                      "src/chaoyang/humanego_ict/derived_training_gate.py",
                      "src/chaoyang/humanego_ict/derived_training_contract.py",
                      "src/chaoyang/humanego_ict/checkpoint_export.py",
                      "src/chaoyang/ops/train_pick_place_humanego_ict_derived_v1.py",
                      "scripts/humanego_ict/train_derived_pick_place.sh",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/humanego_ict/test_source_adapter.py",
                      "tests/humanego_ict/test_derived_training.py",
                      "tests/humanego_ict/test_checkpoint_export.py",
                      "scripts/humanego_ict/register_derived_training_source_task_v1.py", PLAN,
                      str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "V2_STATIC_EVIDENCE_SHA_MATCH"],
        "required_outputs": ["derived_trainer_candidate", "tensor_checkpoint_export", "SOURCE_READINESS.json", "CPU_TEST_RESULT.json",
                             "PROTECTED_INPUTS_CHECK.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": {"wall_seconds_max": WALL_SECONDS, "gpu_count": 0, "training_steps": 0, "model_calls": 0},
        "stop_conditions": ["protected SHA drift", "writer conflict", "deadline", "unrecoverable write capacity"],
        "claim_limit": "CPU engineering with synthetic tests only; derived labels are not real controls; causal RGB and physical calibration gates remain unchanged; no model or deployment qualification",
        "inputs": inputs, "writer": writer, "writer_file": str(ROOT / "WRITER.json"),
        "runtime_code": [artifact_ref(Path(__file__))],
        "run_signature": hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=WALL_SECONDS)).isoformat(),
        "commands": ["CPU editing and bounded synthetic pytest verification only; no production algorithm invocation"],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(ROOT / "FROZEN_INPUTS.json", inputs)
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
                           "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
                           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
                           "gpu_id": None, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": TASK, "expected_resource": "CPU_ENGINEERING_ONLY"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_allowed": True,
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    publish(state, index, "HUMANEGO_ICT_DERIVED_BRIDGE_CPU_REGISTERED")
    print(json.dumps({"status": "REGISTERED", "task_id": TASK, "writer": writer}), flush=True)
    deadline, last = time.monotonic() + WALL_SECONDS, time.monotonic()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 5)
        if ready:
            command = sys.stdin.readline().strip()
            if command.startswith("finish "):
                path = Path(command[7:]).resolve(strict=True)
                result = load_json(path)
                if not path.is_relative_to(ROOT) or result.get("task_id") != TASK:
                    raise RuntimeError("RESULT_BINDING_MISMATCH")
                for ref in [inputs["baseline_pointer"], *protected, *derived_protected]:
                    if errors := validate_artifact_ref(ref):
                        raise RuntimeError("PROTECTED_SHA_DRIFT:" + str(errors))
                terminal = result.get("task_terminal_status", "PASSED")
                if terminal not in {"PASSED", "BLOCKED_PREREQ", "FAILED_RUNTIME_FINAL"}:
                    raise RuntimeError("INVALID_TERMINAL_STATUS")
                state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
                row = next(r for r in state["tasks"] if r["task_id"] == TASK)
                if row.get("pid") != writer["pid"] or row.get("run_signature") != packet["run_signature"]:
                    raise RuntimeError("WRITER_FENCE_DRIFT")
                row.update(status=terminal, pid=None, proc_start_ticks=None,
                           result=artifact_ref(path), updated_at=now_iso(), heartbeat_at=now_iso())
                state["next_task"] = None
                item = next(r for r in index["task_packets"] if r["task_id"] == TASK)
                item.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
                task_receipt = REPO_ROOT / "tasks/receipts" / (TASK.upper() + ".json")
                atomic_json(task_receipt, {"task_id": TASK, "status": terminal,
                                           "result": artifact_ref(path), "claim_limit": packet["claim_limit"]})
                row["receipt"] = artifact_ref(task_receipt)
                publish(state, index, "HUMANEGO_ICT_DERIVED_BRIDGE_CPU_SEALED")
                print(json.dumps({"status": terminal, "receipt": str(task_receipt)}), flush=True)
                return 0
        if time.monotonic() - last >= 20:
            state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
            row = next(r for r in state["tasks"] if r["task_id"] == TASK)
            if row["status"] != "RUNNING" or row["pid"] != writer["pid"]:
                raise RuntimeError("WRITER_FENCE_DRIFT")
            row.update(updated_at=now_iso(), heartbeat_at=now_iso())
            publish(state, index, "HUMANEGO_ICT_DERIVED_BRIDGE_CPU_HEARTBEAT")
            last = time.monotonic()
    raise RuntimeError("CPU_ENGINEERING_DEADLINE_REACHED")


if __name__ == "__main__":
    raise SystemExit(main())
