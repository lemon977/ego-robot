#!/usr/bin/env python3
"""Publish the user-confirmed 0901/0902 visual baseline after a fenced edit window.

Run as ``python -B -m chaoyang.governance.publish_exact78_visual_baseline_v1``.
This is a CPU governance publication. It never imports the Exact78 algorithm.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_visual_baseline_v1_publish_20261008"
BASELINE_ID = "EXACT78_VISUAL_BASELINE_V1"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"
POINTER = REPO_ROOT / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"
HANDOFF = REPO_ROOT / "docs/current/EXACT78_VISUAL_BASELINE_V1_ZH.md"
V13_PACKET = REPO_ROOT / "tasks/current/exact78_takeover_b2_full_v13/TASK_PACKET.json"
PROOFS = {
    "v13_result": ("_run/current/exact78_takeover_b2_full_v13/attempts/attempt_0001/RESULT.json",
                   "cbb2d4eab493f9fb1132f11f3488c3ef3ce6125cd0fe2b1b1c6194a6be273dd1", 2438),
    "fresh_replay_result": (
        "_run/current/exact78_takeover_v13_cards042_full_replay_v1_20261008/attempts/attempt_0001/RESULT.json",
        "ab9a91557ec2777db2ae7f7ae2b12e4ab77bdbbb3405271eae9a92618c14a725", 5480),
    "comparison": ("review/v13_cards042_replay_20261008/COMPARISON.json",
                   "5ce47c269b7eafbee8c33c15bba828f6224931f953936205ac96a491b860f0cc", 69416),
    "docs_audit": ("docs/plans/EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008/lanes/docs/AUDIT.json",
                   "ce54ec4442302f2259e9824b925d08e3734fd908657c08fd60b4986aca4dcf72", None),
    "code_audit": ("docs/plans/EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008/lanes/code/REVERSE_REFERENCE_AUDIT.json",
                   "a25385bdd9b09fb05d3bb876929130aa51f5738fcf9e9e505285deb7fa2c5c8b", None),
    "media_proof": ("docs/plans/EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008/lanes/media/PROOF.json",
                    "9585fe46efbf2b91dc41d6893e630a4b4f02457107c49a416f44564df3303fa2", None),
}
PROTECTED = (
    "docs/current/ROBOT100_BATCH_ZH.md",
    "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    "src/chaoyang/ops/run_robot100_world_first_v10.py",
    "src/chaoyang/ops/run_robot100_constraint_review_v16_retry1.py",
)
GOVERNANCE_WRAPPERS = (
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/governance/current_r3_contracts.py",
)
ALGORITHM_PREFIXES = (
    "/src/chaoyang/exact78_baseline/",
    "/src/chaoyang/pipeline/",
)
ALGORITHM_OPERATION = "/src/chaoyang/ops/run_exact78_takeover_baseline_v1.py"
WAIT_SECONDS = 20 * 60


def _validate_current() -> None:
    run = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"),
                            "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=False,
    )
    if run.returncode:
        raise RuntimeError("GOVERNANCE_INVALID:" + run.stdout[-1800:] + run.stderr[-1800:])


def _fixed_ref(relative: str, sha: str, size: int | None) -> dict:
    actual = artifact_ref(REPO_ROOT / relative)
    if actual["sha256"] != sha or (size is not None and actual["bytes"] != size):
        raise RuntimeError(f"PROOF_DRIFT:{relative}")
    return actual


def _proof_refs() -> dict[str, dict]:
    refs = {name: _fixed_ref(path, sha, size) for name, (path, sha, size) in PROOFS.items()}
    for name in ("v13_result", "fresh_replay_result"):
        if load_json(Path(refs[name]["path"])).get("status") != "PASSED":
            raise RuntimeError(f"RESULT_NOT_PASSED:{name}")
    reproduction = load_json(Path(refs["comparison"]["path"])).get("reproduction", {})
    expected = ("all_motion_arrays_elementwise_exact", "all_mask_arrays_elementwise_exact",
                "all_three_mp4_sha_equal", "all_three_mp4_decoded_pixel_exact")
    if not all(reproduction.get(key) is True for key in expected) or reproduction.get("decoded_frames_each") != 171:
        raise RuntimeError("FRESH_REPLAY_NOT_EXACT")
    return refs


def _algorithm_refs() -> list[dict]:
    packet = load_json(V13_PACKET)
    rows = [ref for ref in packet["runtime_code"]
            if any(prefix in str(ref.get("path")) for prefix in ALGORITHM_PREFIXES)
            or str(ref.get("path", "")).endswith(ALGORITHM_OPERATION)]
    if not rows or not any(str(ref["path"]).endswith(ALGORITHM_OPERATION) for ref in rows):
        raise RuntimeError("V13_ALGORITHM_CLOSURE_ABSENT")
    for ref in rows:
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError(f"V13_ALGORITHM_SHA_DRIFT:{ref['path']}")
    return rows


def _refs(paths: tuple[str, ...]) -> dict[str, dict]:
    return {path: artifact_ref(REPO_ROOT / path) for path in paths}


def _index_body() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _write_set() -> list[str]:
    return [
        str(ATTEMPT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
        str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
        "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/EXACT78.md",
        "docs/current/README_ZH.md", "docs/current/STATUS.json",
        "docs/current/V5_SCENE.md", "docs/current/visuals/README_ZH.md",
        "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
        "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md",
        "docs/current/EXACT78_TAKEOVER_BASELINE_20261007_ZH.md",
        "docs/current/EXACT78_VISUAL_BASELINE_V1.json",
        "docs/current/EXACT78_VISUAL_BASELINE_V1_ZH.md",
        "README.md", "docs/README.md", "review/README.md",
        "review/v13_cards042_replay_20261008/README.md",
        "docs/plans/EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/governance/current_r3_contracts.py",
        "src/chaoyang/governance/exact78_visual_selection_v1.py",
        "tests/unit/test_exact78_visual_selection_v1.py",
    ]


def _register(proofs: dict[str, dict], algorithm: list[dict], protected: dict[str, dict],
              wrappers: dict[str, dict]) -> dict:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(task.get("task_id") == TASK for task in state.get("tasks", [])):
        raise RuntimeError("TASK_ID_ALREADY_USED")
    if RECEIPT.exists() or PACKET.exists() or ATTEMPT.exists():
        raise RuntimeError("FRESH_PUBLICATION_REQUIRED")
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"proofs": proofs, "v13_packet": artifact_ref(V13_PACKET),
              "algorithm_code": algorithm, "protected_before": protected,
              "governance_wrappers_before": wrappers,
              "publisher_code": artifact_ref(Path(__file__))}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs,
                                                "decision": "PUBLISH_EXACT78_VISUAL_BASELINE_V1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Publish user-confirmed 0901/0902 Exact78 V1 batch visual baseline without algorithm changes",
        "phase": "EXACT78_VISUAL_BASELINE_V1_PUBLICATION",
        "plan_revision": "EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008",
        "execution_revision": "EXACT78_VISUAL_BASELINE_V1_PUBLISH",
        "session_id": None,
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=WAIT_SECONDS + 180)).isoformat(),
        "read_set": ["AGENTS.md", str(V13_PACKET.relative_to(REPO_ROOT)),
                     *(str(Path(ref["path"]).relative_to(REPO_ROOT)) for ref in proofs.values())],
        "write_set": _write_set(),
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V13_ALGORITHM_SHA_PINNED",
                          "FRESH_REPLAY_EXACT", "0916_PROTECTED", "NO_MODEL_RUN"],
        "required_outputs": ["RESULT.json", "EXACT78_VISUAL_BASELINE_V1_receipt",
                             "EXACT78_VISUAL_BASELINE_V1_pointer", "governance_PASS"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": WAIT_SECONDS + 180,
                    "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "active_writer_conflict", "protected_branch_drift",
                            "timeout", "one_attempt_terminal"],
        "claim_limit": ("User-confirmed 0901/0902 batch visual baseline, not 234-session full-film human review; "
                        "no training, control, physical-occlusion or deployment authority."),
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": ["python -B -m chaoyang.governance.publish_exact78_visual_baseline_v1"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "0901_0902_visual_baseline_publication", "updated_at": now, "heartbeat_at": now,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "SOURCE_REPRODUCED", "adoption": "PENDING_PUBLICATION",
        "review": "USER_CONFIRMED_BATCH_VISUAL_BASELINE_FULL_FILM_REVIEW_NOT_CLAIMED",
    })
    state["next_task"] = {"task_id": TASK, "session": "0901_0902_visual_baseline_publication",
                          "prerequisites": packet["prerequisites"], "expected_resource": "CPU_ONE_THREAD",
                          "stop_condition": "one terminal publication"}
    index["task_packets"].append({"task_id": TASK,
                                  "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE",
                                  "execution_allowed": True})
    index.update(packet_revision="EXACT78_VISUAL_BASELINE_V1_RUNNING", status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state,
                   event_type="EXACT78_VISUAL_BASELINE_V1_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return packet


def _verify_pointer(proofs: dict[str, dict]) -> dict:
    pointer = load_json(POINTER)
    if pointer.get("baseline_id") != BASELINE_ID:
        raise RuntimeError("POINTER_BASELINE_ID")
    policy = pointer.get("execution_policy")
    expected_operations = {"run_exact78_takeover_audit_v1", "run_exact78_takeover_baseline_v1",
                           "run_exact78_takeover_cleanup_v1"}
    if (not isinstance(policy, dict) or policy.get("enabled") is not True
            or policy.get("baseline_id") != BASELINE_ID
            or policy.get("scope") != "chips_cards_tracker_0901_0902"
            or not isinstance(policy.get("allowed_operations"), list)
            or not all(isinstance(name, str) for name in policy["allowed_operations"])
            or set(policy["allowed_operations"]) != expected_operations
            or len(policy["allowed_operations"]) != len(expected_operations)):
        raise RuntimeError("POINTER_EXECUTION_POLICY")
    for name in ("v13_result", "fresh_replay_result", "comparison"):
        if pointer.get(name) != proofs[name]:
            raise RuntimeError(f"POINTER_PROOF_DRIFT:{name}")
    if not HANDOFF.is_file():
        raise RuntimeError("POINTER_HANDOFF_ABSENT")
    return pointer


def _terminal_index() -> dict:
    index = _index_body()
    rows = [row for row in index["task_packets"] if row.get("task_id") == TASK]
    if len(rows) != 1 or rows[0].get("execution_allowed") is not True:
        raise RuntimeError("ACTIVE_PACKET_DRIFT")
    rows[0]["execution_allowed"] = False
    index.update(packet_revision="EXACT78_VISUAL_BASELINE_V1_TERMINAL",
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    return index


def _finish(packet: dict, protected_before: dict[str, dict]) -> None:
    proofs = _proof_refs()
    algorithm = _algorithm_refs()
    if algorithm != packet["inputs"]["algorithm_code"]:
        raise RuntimeError("V13_ALGORITHM_CLOSURE_DRIFT")
    protected_after = _refs(PROTECTED)
    if protected_after != protected_before:
        raise RuntimeError("0916_PROTECTED_FILE_DRIFT")
    pointer = _verify_pointer(proofs)
    wrappers_after = _refs(GOVERNANCE_WRAPPERS)
    receipt = {
        "schema_version": "exact78-visual-baseline-v1-publication-receipt",
        "task_id": TASK, "status": "PASSED", "recorded_at": now_iso(),
        "decision": "ADOPT_USER_CONFIRMED_0901_0902_BATCH_VISUAL_BASELINE_V1",
        "baseline_id": BASELINE_ID, "baseline_pointer": artifact_ref(POINTER),
        "handoff": artifact_ref(HANDOFF), "proofs": proofs,
        "algorithm_code": algorithm, "governance_wrappers_before": packet["inputs"]["governance_wrappers_before"],
        "governance_wrappers_after": wrappers_after,
        "protected_before": protected_before, "protected_after": protected_after,
        "scope": "chips_cards_tracker_0901_0902",
        "review": "USER_CONFIRMED_BATCH_VISUAL_BASELINE; 234_FULL_FILMS_NOT_HUMAN_REVIEWED",
        "training_eligible": False, "control_ground_truth": False,
        "physical_occlusion_authority": False, "deletion_authorized": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(RECEIPT, receipt)
    result = {
        "schema_version": "exact78-visual-baseline-v1-publication-result",
        "task_id": TASK, "status": "PASSED", "baseline_id": BASELINE_ID,
        "baseline_pointer": artifact_ref(POINTER), "publication_receipt": artifact_ref(RECEIPT),
        "v13_result": proofs["v13_result"], "fresh_replay_result": proofs["fresh_replay_result"],
        "comparison": proofs["comparison"], "visual_batch_adopted": True,
        "human_full_film_review": "NOT_CLAIMED", "claim_limit": packet["claim_limit"],
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    retired_scopes = {
        "exact78/current_visual_parent_and_general_route",
        "exact78/play_cards_0902_042/current_visual_baseline_v2",
        "exact78/play_cards_0902_042/current_visual_adjustment_parent_v1",
        "exact78/play_cards_0902_042/current_visual_stage_version_lock_v1",
    }
    for claim in authority.get("claims", []):
        if claim.get("scope") in retired_scopes:
            claim["status"] = "WITHDRAWN"
            claim["claim_limit"] = "Historical selection superseded by EXACT78_VISUAL_BASELINE_V1; preserve evidence only, never current execution authority."
    authority["claims"].append({
        "claim": ("0901/0902 Exact78 V1 是用户确认、171帧独立复现匹配的当前批量视觉基线；"
                  "234条全片人工审阅未宣称完成。"),
        "claim_limit": packet["claim_limit"],
        "scope": "exact78/chips_cards_tracker_0901_0902/current_visual_baseline_v1",
        "status": "DEVELOPMENT_EVIDENCE",
        "evidence": [artifact_ref(RECEIPT), proofs["v13_result"], proofs["fresh_replay_result"],
                     proofs["comparison"], artifact_ref(POINTER)],
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK]
    if len(rows) != 1 or rows[0].get("status") != "RUNNING":
        raise RuntimeError("PUBLISH_TASK_STATE_DRIFT")
    rows[0].update(status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(),
                   heartbeat_at=now_iso(), result=artifact_ref(ATTEMPT / "RESULT.json"),
                   terminal_receipt=artifact_ref(RECEIPT), quality="REPRODUCIBLE_VISUAL_BATCH",
                   adoption="CURRENT_VISUAL_BASELINE_V1",
                   review="USER_CONFIRMED_BATCH_VISUAL_FULL_FILM_NOT_CLAIMED")
    state["next_task"] = None
    # Registration imported the old generator; activate only the explicitly
    # edited selection/navigation generator for this final CAS publication.
    from chaoyang.governance import current_r3_contracts
    importlib.reload(current_r3_contracts)
    published = publish_bundle(authority, state, event_type="EXACT78_VISUAL_BASELINE_V1_PUBLISHED",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=_terminal_index())
    _validate_current()
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"],
                      "baseline_id": pointer["baseline_id"],
                      "result": artifact_ref(ATTEMPT / "RESULT.json"),
                      "receipt": artifact_ref(RECEIPT)}, ensure_ascii=False), flush=True)


def _abort_registered(reason: str) -> None:
    """Terminalize a registered edit window that did not publish a receipt."""

    if RECEIPT.exists() or (ATTEMPT / "RESULT.json").exists():
        return  # A partial final publication requires manual receipt-bound recovery.
    atomic_json(ATTEMPT / "RESULT.json", {
        "schema_version": "exact78-visual-baseline-v1-publication-result",
        "task_id": TASK, "status": "FAILED_RUNTIME_FINAL", "failure_code": reason,
        "files_deleted": False, "visual_batch_adopted": False,
        "next_action": "Publish a new finite task id after resolving this failure.",
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK]
    if len(rows) != 1 or rows[0].get("status") != "RUNNING":
        raise RuntimeError("ABORT_TASK_STATE_DRIFT")
    rows[0].update(status="FAILED_RUNTIME_FINAL", pid=None, proc_start_ticks=None,
                   updated_at=now_iso(), heartbeat_at=now_iso(),
                   result=artifact_ref(ATTEMPT / "RESULT.json"), quality="NOT_PUBLISHED",
                   adoption="NOT_ADOPTED_BY_THIS_TASK", review="NOT_APPLICABLE")
    state["next_task"] = None
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state,
                   event_type="EXACT78_VISUAL_BASELINE_V1_PUBLICATION_FAILED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=_terminal_index())


def main() -> int:
    _validate_current()
    proofs = _proof_refs()
    algorithm = _algorithm_refs()
    protected = _refs(PROTECTED)
    wrappers = _refs(GOVERNANCE_WRAPPERS)
    packet = _register(proofs, algorithm, protected, wrappers)
    print(json.dumps({"status": "READY", "task_id": TASK, "packet": artifact_ref(PACKET),
                      "deadline_seconds": WAIT_SECONDS}, ensure_ascii=False), flush=True)
    try:
        readable, _, _ = select.select([sys.stdin], [], [], WAIT_SECONDS)
        if not readable or sys.stdin.readline().strip() != "PUBLISH":
            raise RuntimeError("PUBLICATION_WINDOW_EXPIRED_OR_NOT_AUTHORIZED")
        _finish(packet, protected)
    except Exception as exc:
        _abort_registered(str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
