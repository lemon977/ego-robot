"""Terminalize only the HuRo child of the active completion campaign.

This publisher records an already established quality rejection and user scope
decision.  It does not execute HuRo, change the active parent, or infer quality
for any other lane.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.four_stream_completion import ACTIVE, ATTEMPT, TASK, ticks


HURO_TASK = TASK + "_huro"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LANE = ATTEMPT / "lanes/huro"
LANE_STATE = LANE / "STATE.json"
TERMINAL_RECEIPT = LANE / "TERMINAL_NOT_ADOPTED.json"
CANDIDATE = LANE / "H01_007_181_196_PROJECTED_C3_R0_ARM_CPU_V1/WINDOW_RESULT.json"
OBJECTIVE_AUDIT = LANE / "H01_OBJECTIVE_CONFLICT_AUDIT_V1.json"


def _one(rows: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("task_id") == task_id]
    if len(selected) != 1:
        raise RuntimeError("TASK_ROW_IDENTITY_MISMATCH:" + task_id)
    return selected[0]


def validate_terminal_evidence(progress: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen HuRo rejection without executing a solver."""
    if progress.get("schema_version") != "FOUR_STREAM_COMPLETION_PROGRESS_V1":
        raise RuntimeError("HURO_PROGRESS_SCHEMA_MISMATCH")
    scope_change = progress.get("scope_change")
    if not isinstance(scope_change, Mapping) or scope_change.get("decision") != "STOP_HURO_AFTER_ROOT_CAUSE_REVIEW":
        raise RuntimeError("HURO_STOP_AUTHORITY_MISSING")
    huro = progress.get("huro")
    if not isinstance(huro, Mapping):
        raise RuntimeError("HURO_PROGRESS_SECTION_MISSING")
    if huro.get("quality") != "TERMINAL_RESEARCH_CONCLUSION_NOT_ADOPTED":
        raise RuntimeError("HURO_PROGRESS_QUALITY_MISMATCH")
    if huro.get("automatic_candidates_exhausted") is not True:
        raise RuntimeError("HURO_CANDIDATES_NOT_EXHAUSTED")
    if huro.get("next_review") is None or not str(huro["next_review"]).startswith("NONE_UNLESS_USER_REOPENS_HURO_SCOPE"):
        raise RuntimeError("HURO_SCOPE_NOT_CLOSED")

    candidate = load_json(CANDIDATE)
    if (
        candidate.get("task_id") != HURO_TASK
        or candidate.get("execution") != "REAL_OFFICIAL_CORE_ADAPTED_WINDOW_EXECUTED"
        or candidate.get("pose_gate_pass") is not False
        or candidate.get("pose_pass_side_frames") != 0
        or candidate.get("rotation_valid_side_frames") != 32
        or candidate.get("adoption") != "NOT_ADOPTED"
        or candidate.get("full_session_executed") is not False
    ):
        raise RuntimeError("HURO_CANDIDATE_SEMANTICS_MISMATCH")

    audit = load_json(OBJECTIVE_AUDIT)
    conclusion = audit.get("conclusion")
    if (
        audit.get("schema_version") != "H01_OBJECTIVE_CONFLICT_AUDIT_V1"
        or audit.get("solver_called") is not False
        or audit.get("new_candidate_created") is not False
        or not isinstance(conclusion, Mapping)
        or conclusion.get("rotation_penalty_increased") is not True
        or conclusion.get("local_and_wrist_rotation_gradients_opposed_at_final") is not True
        or conclusion.get("hard_wrist_constraint_present") is not False
    ):
        raise RuntimeError("HURO_OBJECTIVE_AUDIT_MISMATCH")

    progress_candidate = huro.get("candidate3_result")
    progress_audit = huro.get("rotation_audit")
    actual_candidate = artifact_ref(CANDIDATE)
    actual_audit = artifact_ref(OBJECTIVE_AUDIT)
    for claimed, actual in ((progress_candidate, actual_candidate), (progress_audit, actual_audit)):
        if not isinstance(claimed, Mapping):
            raise RuntimeError("HURO_PROGRESS_EVIDENCE_MISSING")
        claimed_path = Path(str(claimed.get("path", "")))
        if not claimed_path.is_absolute():
            claimed_path = REPO_ROOT / claimed_path
        if claimed_path.resolve(strict=True) != Path(actual["path"]):
            raise RuntimeError("HURO_PROGRESS_EVIDENCE_PATH_MISMATCH")
        if claimed.get("sha256") != actual["sha256"]:
            raise RuntimeError("HURO_PROGRESS_EVIDENCE_SHA_MISMATCH")
    return {"candidate": actual_candidate, "objective_audit": actual_audit}


def terminalize_huro_row(
    state: dict[str, Any], result_ref: Mapping[str, Any], created_at: str
) -> dict[str, Any]:
    """Return a state copy where only the HuRo child row is terminalized."""
    updated = deepcopy(state)
    parent = _one(updated["tasks"], TASK)
    child = _one(updated["tasks"], HURO_TASK)
    if parent.get("status") not in ACTIVE:
        raise RuntimeError("PARENT_NOT_ACTIVE")
    if child.get("parent_task_id") != TASK or child.get("status") != "PENDING":
        raise RuntimeError("HURO_CHILD_NOT_PENDING")
    if child.get("pid") is not None or child.get("proc_start_ticks") is not None:
        raise RuntimeError("HURO_CHILD_HAS_WRITER")

    before = {row["task_id"]: deepcopy(row) for row in updated["tasks"] if row["task_id"] != HURO_TASK}
    child.update(
        status="REJECTED_QUALITY",
        phase="HURO_TERMINAL_RESEARCH_NOT_ADOPTED",
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        updated_at=created_at,
        result=dict(result_ref),
        quality="REJECTED_WRIST_POSE_OBJECTIVE_CONFLICT",
        adoption="NOT_ADOPTED",
        last_attempt_terminal="REJECTED_QUALITY",
        last_attempt_reason="THREE_CANDIDATES_EXHAUSTED_OBJECTIVE_CONFLICT_USER_CLOSED_SCOPE",
    )
    after = {row["task_id"]: row for row in updated["tasks"] if row["task_id"] != HURO_TASK}
    if before != after:
        raise RuntimeError("NON_HURO_TASK_ROW_CHANGED")
    if updated.get("next_task") != state.get("next_task"):
        raise RuntimeError("NEXT_TASK_CHANGED")
    return updated


def _write_terminal_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if TERMINAL_RECEIPT.exists():
        existing = load_json(TERMINAL_RECEIPT)
        stable_keys = set(value) - {"created_at", "governance_revision_before_publish"}
        if any(existing.get(key) != value.get(key) for key in stable_keys):
            raise RuntimeError("HURO_TERMINAL_RECEIPT_CONFLICT")
        return artifact_ref(TERMINAL_RECEIPT)
    atomic_json(TERMINAL_RECEIPT, value)
    return artifact_ref(TERMINAL_RECEIPT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--writer-pid", type=int, required=True)
    args = parser.parse_args()

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS_MISMATCH")
    writer = load_json(ATTEMPT / "WRITER.json")
    if writer.get("pid") != args.writer_pid or writer.get("proc_start_ticks") != ticks(args.writer_pid):
        raise RuntimeError("WRITER_IDENTITY_MISMATCH")

    state = load_json(TASK_STATE_PATH)
    parent = _one(state["tasks"], TASK)
    checkpoint_ref = parent.get("progress_checkpoint")
    if not isinstance(checkpoint_ref, Mapping) or validate_artifact_ref(checkpoint_ref):
        raise RuntimeError("BOUND_PROGRESS_CHECKPOINT_DRIFT")
    progress_path = Path(str(checkpoint_ref["path"])).resolve(strict=True)
    if progress_path != ATTEMPT / "checkpoints/PROGRESS_0006.json":
        raise RuntimeError("EXPECTED_PROGRESS_0006_NOT_BOUND")
    progress = load_json(progress_path)
    evidence = validate_terminal_evidence(progress)

    index = load_json(INDEX)
    entries = {entry["task_id"]: entry for entry in index.get("task_packets", [])}
    if entries.get(TASK, {}).get("execution_allowed") is not True:
        raise RuntimeError("PARENT_NOT_ROUTABLE")
    expected_children = {TASK + suffix for suffix in ("_scene", "_sensor", "_motion", "_huro")}
    if not expected_children.issubset(entries):
        raise RuntimeError("CHILD_PACKET_SET_INCOMPLETE")
    for child_id in expected_children:
        entry = entries[child_id]
        if entry.get("execution_allowed") is not False or entry.get("execution_class") != "CATALOG_REFERENCE":
            raise RuntimeError("CHILD_PACKET_UNEXPECTEDLY_ROUTABLE")

    lane_state = load_json(LANE_STATE)
    if lane_state.get("task_id") != HURO_TASK or lane_state.get("adoption") != "NOT_ADOPTED":
        raise RuntimeError("HURO_LANE_STATE_MISMATCH")
    created = now_iso()
    terminal = {
        "schema_version": "FOUR_STREAM_COMPLETION_HURO_TERMINAL_V1",
        "task_id": HURO_TASK,
        "status": "REJECTED_QUALITY",
        "execution": "REAL_OFFICIAL_CORE_ADAPTED_WINDOW_EXECUTED",
        "quality": "REJECTED_WRIST_POSE_OBJECTIVE_CONFLICT",
        "adoption": "NOT_ADOPTED",
        "review": "ROOT_CAUSE_REVIEW_COMPLETE",
        "further_compute": "NOT_AUTHORIZED_UNLESS_USER_REOPENS_SCOPE",
        "candidate_count": 3,
        "full_session_executed": False,
        "evidence": evidence,
        "scope_authority": artifact_ref(progress_path),
        "training_executed": False,
        "gpu_used_by_terminalizer": False,
        "raw_or_processed_modified": False,
        "physical_accuracy_proven": False,
        "claim_limit": "HuRo research rejection only; no quality or physical claim for the three product lanes.",
        "governance_revision_before_publish": args.expected_revision,
        "created_at": created,
    }
    terminal_ref = _write_terminal_receipt(terminal)
    updated = terminalize_huro_row(state, terminal_ref, created)
    updated["recent_events"] = (updated.get("recent_events", []) + [{
        "task_id": HURO_TASK,
        "attempt": 1,
        "status": "REJECTED_QUALITY",
        "created_at": created,
        "message": "HuRo objective conflict reviewed; current method is terminal and not adopted. Parent and product lanes remain active.",
        "result": terminal_ref,
    }])[-100:]

    terminal_state = dict(lane_state)
    terminal_state.update(
        status="TERMINAL_REJECTED_QUALITY",
        execution="REAL_OFFICIAL_CORE_ADAPTED_WINDOW_EXECUTED",
        quality="REJECTED_WRIST_POSE_OBJECTIVE_CONFLICT",
        adoption="NOT_ADOPTED",
        review="ROOT_CAUSE_REVIEW_COMPLETE",
        next_action="NONE_UNLESS_USER_REOPENS_HURO_SCOPE",
        result=terminal_ref,
        updated_at=created,
    )
    atomic_json(LANE_STATE, terminal_state)

    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        updated,
        event_type="FOUR_STREAM_COMPLETION_HURO_TERMINAL_NOT_ADOPTED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    print(json.dumps({
        "status": "REJECTED_QUALITY",
        "task_id": HURO_TASK,
        "parent_remains_active": True,
        "revision": published["governance_revision"],
        "result": terminal_ref,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
