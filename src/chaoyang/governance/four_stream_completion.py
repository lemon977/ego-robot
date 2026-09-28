"""Bounded registration and generated navigation for the completion campaign."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import secrets
from datetime import datetime, timedelta
from chaoyang.governance.common import (
    REPO_ROOT, RECEIPT_PATH, AUTHORITY_PATH, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "four_stream_completion_20260928"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
ACTIVE = {"PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
LANES = ("scene", "sensor", "motion", "huro")
PLAN = REPO_ROOT / "docs/current/COMPLETION_20260928_ZH.md"
BASE = REPO_ROOT / "_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/RESULT.json"

def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])

def build_status(state, generated_at):
    rows = [r for r in state["tasks"] if r["task_id"] == TASK or r.get("parent_task_id") == TASK]
    parent = next(r for r in rows if r["task_id"] == TASK)
    return {
        "schema_version": "FOUR_STREAM_COMPLETION_STATUS_V1",
        "generated_at": generated_at, "governance_revision": state["governance_revision"],
        "parent_task_id": TASK, "latest_task": TASK, "parent_status": parent["status"],
        "status": "ACTIVE_COMPLETION" if any(r["status"] in ACTIVE for r in rows) else "NO_ACTIVE_COMPLETION",
        "active_tasks": [r["task_id"] for r in rows if r["status"] in ACTIVE],
        "next_task": state.get("next_task"),
        "lanes": [{"task_id": r["task_id"], "status": r["status"]} for r in rows if r["task_id"] != TASK],
        "prior_result": artifact_ref(BASE),
        "counts_scope": "PREDECESSOR_ONLY_NOT_NEW_EXECUTION",
        "counts": load_json(BASE).get("counts", {}),
        "progress_checkpoint": parent.get("progress_checkpoint"),
        "new_quality": "NOT_EVALUATED", "review": "NOT_REVIEWED", "adoption": "NOT_ADOPTED",
        "control_ground_truth": False, "physical_deployable": False,
        "training_eligible": False, "external_metric_authority": False,
        "claim_limit": "Active repair campaign. Historical counts are not new algorithm results.",
    }

def publish_navigation(root, state, generated_at):
    if not any(r["task_id"] == TASK for r in state["tasks"]):
        return False
    latest = next((e for e in reversed(state.get("recent_events", [])) if e.get("task_id")), {})
    if latest.get("task_id") != TASK and not str(latest.get("task_id", "")).startswith(TASK + "_"):
        return False
    atomic_json(root / "docs/current/STATUS.json", build_status(state, generated_at))
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--writer-pid", type=int, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if receipt["governance_revision"] != args.expected_revision:
        raise RuntimeError("CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    index_path = REPO_ROOT / "tasks/current/INDEX.json"
    index = load_json(index_path)
    created = now_iso()
    if args.refresh:
        writer = load_json(ATTEMPT / "WRITER.json")
        if writer["pid"] != args.writer_pid or writer["proc_start_ticks"] != ticks(args.writer_pid):
            raise RuntimeError("WRITER_IDENTITY_MISMATCH")
        parent = next(r for r in state["tasks"] if r["task_id"] == TASK)
        if parent["status"] not in ACTIVE:
            raise RuntimeError("PARENT_NOT_ACTIVE")
        parent.update(status="PENDING", heartbeat_at=None, updated_at=created,
                      phase_detail="IMPLEMENTING_ISOLATED_PATCHES_NO_ALGORITHM_RUNNING")
        if args.checkpoint is not None:
            checkpoint = args.checkpoint.resolve(strict=True)
            if not checkpoint.is_relative_to(ATTEMPT.resolve()) or load_json(checkpoint).get("task_id") != TASK:
                raise RuntimeError("CHECKPOINT_SCOPE_MISMATCH")
            parent["progress_checkpoint"] = artifact_ref(checkpoint)
        # Only the parent is directly routable; child packets isolate work.
        for entry in index["task_packets"]:
            if entry["task_id"].startswith(TASK + "_"):
                entry.update(execution_allowed=False,
                             execution_class="CATALOG_REFERENCE", delegation_scope_only=True)
        publication = publish_bundle(load_json(AUTHORITY_PATH), state,
            event_type="FOUR_STREAM_COMPLETION_CODE_REFRESH", expected_revision=args.expected_revision,
            generator_path=Path(__file__), task_packet_index_path=index_path,
            task_packet_index_value=index)
        print(json.dumps({"revision": publication["governance_revision"], "status": "REFRESHED"}))
        return 0
    if any(r["status"] in ACTIVE for r in state["tasks"]) or index["task_packets"]:
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if any(r["task_id"] == TASK for r in state["tasks"]) or ATTEMPT.exists():
        raise RuntimeError("TASK_ALREADY_EXISTS")
    writer = {"pid": args.writer_pid, "proc_start_ticks": ticks(args.writer_pid),
              "executor_epoch": 31, "fencing_token": secrets.token_hex(24),
              "scope": TASK, "created_at": created}
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    deadline = (datetime.now().astimezone() + timedelta(hours=24)).isoformat(timespec="seconds")
    entries = []
    for lane in ("publisher", *LANES):
        task_id = TASK if lane == "publisher" else TASK + "_" + lane
        worktree = REPO_ROOT if lane == "publisher" else REPO_ROOT / "_run/current/worktrees" / task_id
        output = ATTEMPT if lane == "publisher" else ATTEMPT / "lanes" / lane
        output.mkdir(parents=True, exist_ok=True)
        packet = {
            "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
            "objective": f"Completion campaign {lane}: bounded root-cause repair, no training.",
            "phase": "COMPLETION_PHASE_AB", "plan_revision": TASK,
            "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                         str(BASE.relative_to(REPO_ROOT)), "docs/governance/ALGORITHM_CONTRACT.json"],
            "write_set": [str(output.relative_to(REPO_ROOT)), str(worktree.relative_to(REPO_ROOT))],
            "prerequisites": ["governance_PASS", "single_publisher", "isolated_write_set", "registered_operation"],
            "required_outputs": ["STATE.json", "HANDOFF_ZH.md", "evidence_bound_test_results"],
            "budgets": {"cpu_threads": 2, "gpu_owners_total": 1, "candidates_per_root_cause": 3,
                        "training_steps": 0, "batch_wall_seconds_max": 86400},
            "stop_conditions": ["24h_batch_checkpoint_not_project_completion",
                                "three_candidates_require_root_cause_review",
                                "writer_identity_or_scope_conflict"],
            "claim_limit": "Offline development only; no physical accuracy or training authorization.",
            "plan": artifact_ref(PLAN), "predecessor": artifact_ref(BASE),
            "worktree": str(worktree), "output_root": str(output), "writer": writer,
        }
        errors = _validate_packet(packet)
        if errors:
            raise RuntimeError(str(errors))
        pp = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
        pp.parent.mkdir(parents=True, exist_ok=False)
        atomic_json(pp, packet)
        ref = artifact_ref(pp)
        entries.append({"task_id": task_id, "packet_path": str(pp.relative_to(REPO_ROOT)),
                        "packet_sha256": ref["sha256"], "execution_allowed": lane == "publisher",
                        "execution_class": "CURRENT_LEDGER_ROUTABLE" if lane == "publisher" else "CATALOG_REFERENCE"})
        row = {"task_id": task_id, "phase": "COMPLETION_PHASE_AB", "attempt": 1,
               "status": "PENDING",
               "updated_at": created, "heartbeat_at": None,
               "pid": args.writer_pid if lane == "publisher" else None,
               "proc_start_ticks": writer["proc_start_ticks"] if lane == "publisher" else None,
               "task_packet": ref, "deadline_at": deadline, "parent_task_id": TASK if lane != "publisher" else None}
        state["tasks"].append(row)
        if lane != "publisher":
            atomic_json(output / "STATE.json", {
                "task_id": task_id, "execution": "NOT_STARTED", "quality": "NOT_EVALUATED",
                "adoption": "NOT_ADOPTED", "review": "NOT_REVIEWED",
                "next_action": "CPU_ROOT_CAUSE_DIAGNOSTIC_THEN_REGISTER_CANDIDATE",
                "dependencies": [], "updated_at": created})
    state["next_task"] = {"task_id": TASK, "session": "all_fixed_sessions",
                          "stop_condition": "Finite phase AB checkpoint, not project completion"}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "status": "RUNNING", "created_at": created,
        "message": "User authorized completion campaign; independent lanes and predecessor preserved."}])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": TASK, "plan_revision": TASK, "execution_revision": TASK,
                 "status": "PASS", "supersedes_index": artifact_ref(index_path), "task_packets": entries}
    publication = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_COMPLETION_REGISTERED", expected_revision=args.expected_revision,
        generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=successor)
    atomic_json(ATTEMPT / "REGISTRATION.json", {
        "task_id": TASK, "revision": publication["governance_revision"],
        "writer": writer, "predecessor": artifact_ref(BASE), "deadline": deadline,
        "project_complete": False})
    print(json.dumps({"status": "REGISTERED", "revision": publication["governance_revision"], "task_id": TASK}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
