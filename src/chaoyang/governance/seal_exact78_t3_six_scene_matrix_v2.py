"""Finite CPU-only sealer for the fixed six-session Action+Scene T3 snapshot.

``--check-only`` is read-only. ``--seal`` is a separate, explicitly authorized
publisher action; this module does not register itself on import or check-only.
PASSED denotes report integrity, never visual/product/training qualification.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Iterator

from chaoyang.governance.build_exact78_t3_six_scene_matrix_v2 import build_matrix
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "exact78_raw_to_robot_t3_six_scene_matrix_v2_20261001"
REVISION = "EXACT78_RAW_TO_ROBOT_T3_SIX_SCENE_MATRIX_V2"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T3_SIX_SCENE_MATRIX_V2.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
FREEZE = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001/INPUT_FREEZE.json"
COMMAND = "PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.governance.seal_exact78_t3_six_scene_matrix_v2 --seal"
RUNTIME_CODE = (
    "src/chaoyang/governance/seal_exact78_t3_six_scene_matrix_v2.py",
    "src/chaoyang/governance/build_exact78_t3_six_scene_matrix_v2.py",
    "src/chaoyang/governance/build_exact78_t3_six_scene_matrix_v1.py",
    "src/chaoyang/governance/build_exact78_t3_six_matrix_v1.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/v52_contracts.py",
    "src/chaoyang/governance/v71_contracts.py",
    "src/chaoyang/governance/current_r3_contracts.py",
)
WRITE_SET = (
    str(ATTEMPT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
    str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
)
EXPECTED_FRAMES = {
    "get_potato_chips_0901_013": 870,
    "get_potato_chips_0902_103": 284,
    "play_cards_0901_001": 645,
    "play_cards_0902_042": 171,
    "play_cards_0915_001": 150,
    "play_cards_0915_002": 131,
}


class T3SceneNotReady(RuntimeError):
    """The fixed scene-joined report cannot yet be sealed."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _refs(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if (isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str)
                and isinstance(value.get("bytes"), int)):
            yield {key: value[key] for key in ("path", "bytes", "sha256")}
            return
        for child in value.values():
            yield from _refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _refs(child)


def _unique_refs(value: Any) -> list[dict[str, Any]]:
    # The pre-registration task-state snapshot is pinned separately; the CAS
    # registration itself necessarily advances that ledger before finish().
    refs = {(ref["path"], ref["sha256"]): ref for ref in _refs(value)
            if ref["path"] != str(TASK_STATE_PATH)}
    return [refs[key] for key in sorted(refs)]


def _is_ref(value: Any) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("path"), str)
            and isinstance(value.get("bytes"), int) and value["bytes"] >= 0
            and isinstance(value.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None)


def candidate_report() -> dict[str, Any]:
    matrix = build_matrix()
    if artifact_ref(FREEZE) != matrix["input_evidence"]["t0_freeze"]:
        raise T3SceneNotReady("T0_FREEZE_SHA_CHANGED_DURING_JOIN")
    freeze = load_json(FREEZE)
    selected = freeze.get("selected", {})
    if (set(selected) != set(EXPECTED_FRAMES)
            or sum(row.get("role") == "development" for row in selected.values()) != 4
            or sum(row.get("role") == "holdout" for row in selected.values()) != 2
            or freeze.get("source_group_limits", {}).get("0902") != "SPLIT_NOT_ESTABLISHED"):
        raise T3SceneNotReady("T0_SIX_COHORT_OR_SPLIT_DRIFT")
    report = copy.deepcopy(matrix)
    report["schema_version"] = "exact78-t3-six-scene-joined-sealed-report-v2"
    report["status"] = "PRE_REGISTER_SNAPSHOT_NOT_YET_SEALED"
    report["cohort"] = {"development": 4, "pre_frozen_cross_group_holdout": 2,
                        "total": 6, "frames": 2251,
                        "0902_independent_holdout": "SPLIT_NOT_ESTABLISHED",
                        "blind_test_claim": False}
    report["execution_counts"] = {
        "action_passed": sum(row["execution"]["action_status"] == "PASSED"
                             for row in report["sessions"].values()),
        "scene_passed_execution_only": sum(
            row["execution"]["scene_frontend_status"] == "PASSED_EXECUTION_ONLY"
            for row in report["sessions"].values()),
        "visual_qualified": 0, "product_qualified": 0, "training_eligible": 0,
    }
    return report


def readiness(report: dict[str, Any], *, now: datetime | None = None) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    stamp = now or datetime.now().astimezone()
    if stamp >= CAMPAIGN_END:
        blockers.append("CAMPAIGN_HARD_CUTOFF_PASSED")
    rows = report.get("sessions", {})
    if set(rows) != set(EXPECTED_FRAMES):
        blockers.append("SIX_PRE_FROZEN_SESSION_SET_DRIFT")
    if sum(row.get("raw", {}).get("frames", 0) for row in rows.values()) != 2251:
        blockers.append("TOTAL_FRAME_COUNT_NOT_2251")
    if report.get("execution_counts", {}).get("visual_qualified") != 0 or any(
            report.get("execution_counts", {}).get(key) != 0
            for key in ("product_qualified", "training_eligible")):
        blockers.append("QUALITY_COUNT_PROMOTION")
    for sid, frames in EXPECTED_FRAMES.items():
        row = rows.get(sid)
        if not isinstance(row, dict):
            continue
        mapping = row.get("cross_lane_frame_mapping", {})
        execution = row.get("execution", {})
        expected_source = sid.rsplit("_", 2)[-2]
        expected_mapping = ("SAME_RAW_VST_BYTES_INDEX_ONLY_TIMEBASE_UNVERIFIED"
                            if expected_source == "0915" else
                            "NOT_PROVEN_MONO_STEREO_FRAME_MAPPING")
        if (row.get("source") != expected_source
                or row.get("raw", {}).get("frames") != frames
                or execution.get("action_status") != "PASSED"
                or execution.get("scene_frontend_status") != "PASSED_EXECUTION_ONLY"
                or execution.get("scene_mask_sealed_frames") != frames
                or execution.get("scene_clean_sealed_frames") != frames
                or mapping.get("status") != expected_mapping
                or mapping.get("scene_raw_frame_rows") != frames
                or row.get("coverage", {}).get("mask_frames") != frames
                or row.get("coverage", {}).get("clean_frames") != frames):
            blockers.append(sid + ":SOURCE_EXECUTION_OR_FRAME_MAPPING_DRIFT")
        if expected_source == "0915" and (mapping.get("time_mode") != "TIMEBASE_UNVERIFIED"
                                          or mapping.get("same_source_file_sha") is not True
                                          or mapping.get("scene_physical_source_index") != 1):
            blockers.append(sid + ":VST_TIMEBASE_PROMOTION")
        if expected_source != "0915" and (mapping.get("same_source_file_sha") is not False
                                          or mapping.get("scene_physical_source_index") != 0):
            blockers.append(sid + ":TRACKER_MONO_STEREO_PROMOTION")
        if (row.get("visual_qualification") != "NOT_QUALIFIED"
                or row.get("full_product_qualification") != "NOT_QUALIFIED"
                or row.get("training_label_qualification") != "NOT_QUALIFIED"):
            blockers.append(sid + ":QUALITY_PROMOTION")
        evidence = row.get("evidence", {})
        if not all(_is_ref(evidence.get(key)) for key in (
                "action_rgb_source", "scene_raw_source", "scene_camera_params",
                "scene_terminal_result", "scene_terminal_receipt", "scene_mask_result",
                "scene_clean_result", "scene_map_result", "scene_raw_result")):
            blockers.append(sid + ":SOURCE_OR_OUTPUT_SHA_REF_MISSING")
        robot_videos = row.get("robot", {}).get("videos", {})
        clean_videos = evidence.get("scene_clean_internal_videos", {})
        if (len(robot_videos) != 3
                or any(video.get("decoded_frames") != frames or not _is_ref(video.get("video"))
                       for video in robot_videos.values())
                or len(clean_videos) != 2
                or any(video.get("decoded_frames") != frames or
                       not _is_ref(video.get("artifact")) for video in clean_videos.values())):
            blockers.append(sid + ":VIDEO_SHA_OR_FRAME_COUNT_MISSING")
    return not blockers, blockers


def packet_value(report: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    stamp = now or datetime.now().astimezone()
    ready, blockers = readiness(report, now=stamp)
    if not ready:
        raise T3SceneNotReady("T3_SCENE_SIX_NOT_READY:" + ";".join(blockers))
    state_ref = report["input_evidence"]["state_for_terminal_eligibility_only"]
    if artifact_ref(TASK_STATE_PATH) != state_ref:
        raise T3SceneNotReady("PRE_REGISTER_TASK_STATE_SHA_DRIFT")
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    runtime = [artifact_ref(REPO_ROOT / path) for path in RUNTIME_CODE]
    inputs = {"plan": artifact_ref(PLAN), "t0_freeze": artifact_ref(FREEZE),
              "t0_frame_time_map": report["input_evidence"]["t0_frame_time_map"],
              "pre_register_task_state": state_ref,
              "pre_register_matrix_sha256": _sha(report),
              "terminal_source_output_refs": _unique_refs(report)}
    config = {"mode": "CPU_ONLY_FIXED_SIX_SCENE_JOIN", "gpu_count": 0,
              "session_ids": sorted(EXPECTED_FRAMES), "total_frames": 2251,
              "visual_product_training_qualified": 0,
              "claim_limit": "REPORT_INTEGRITY_ONLY"}
    signature = _sha({"inputs": inputs, "runtime_code": runtime,
                      "config": config, "command": COMMAND,
                      "schema": "exact78-t3-six-scene-joined-seal-run-v2"})
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Seal fixed six-session Action+Scene SHA and quality-gate matrix",
        "phase": "EXACT78_RAW_TO_ROBOT_T3_SIX_SCENE_MATRIX",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": REVISION, "registered_at": stamp.isoformat(),
        "deadline_at": min(stamp + timedelta(seconds=3600), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str(FREEZE.relative_to(REPO_ROOT)),
                     str(Path(inputs["t0_frame_time_map"]["path"]).relative_to(REPO_ROOT)),
                     str(TASK_STATE_PATH.relative_to(REPO_ROOT)),
                     "docs/governance/DOC_AUTHORITY_MAP.json",
                     "src/chaoyang/governance/build_exact78_t3_six_scene_matrix_v2.py",
                     "src/chaoyang/governance/seal_exact78_t3_six_scene_matrix_v2.py"],
        "expanded_read_set": [ref["path"] for ref in inputs["terminal_source_output_refs"]],
        "write_set": list(WRITE_SET),
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "T0_SIX_FROZEN",
                          "ACTION_AND_SCENE_TERMINAL_SHA_CHAINS", "CPU_ONLY",
                          "NO_PARTIAL_READS", "NO_MODEL", "NO_DELETION"],
        "required_outputs": ["MATRIX.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 3600,
                    "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Integrity only: six/2251, four mono-stereo unproven, two VST timebase unverified; no quality or geometry PASS",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise T3SceneNotReady("INVALID_TASK_PACKET:" + ";".join(errors))
    return packet


def publisher_blockers(state: dict[str, Any], index: dict[str, Any]) -> list[str]:
    blockers = []
    if state.get("next_task"):
        blockers.append("NEXT_TASK_ACTIVE:" + str(state["next_task"].get("task_id")))
    if any(row.get("status") in {"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"}
           for row in state.get("tasks", [])):
        blockers.append("ACTIVE_TASK_ROW")
    if any(row.get("execution_allowed") for row in index.get("task_packets", [])):
        blockers.append("EXECUTION_ALLOWED_PACKET")
    return blockers


def dry_run(*, now: datetime | None = None) -> dict[str, Any]:
    report = candidate_report()
    stamp = now or datetime.now().astimezone()
    ready, blockers = readiness(report, now=stamp)
    if not ready:
        return {"status": "DRY_RUN_H24_CUTOFF" if "CAMPAIGN_HARD_CUTOFF_PASSED" in blockers
                else "DRY_RUN_NOT_READY", "blockers": blockers,
                "registered": False, "execution_counts": report["execution_counts"]}
    packet = packet_value(report, now=stamp)
    publisher = publisher_blockers(load_json(TASK_STATE_PATH), load_json(INDEX))
    return {"status": "DRY_RUN_EVIDENCE_READY_PUBLISHER_BUSY" if publisher else
            "DRY_RUN_READY", "publisher_blockers": publisher, "registered": False,
            "task_id": TASK_ID, "run_signature": packet["run_signature"],
            "matrix_sha256": packet["inputs"]["pre_register_matrix_sha256"],
            "source_output_ref_count": len(packet["inputs"]["terminal_source_output_refs"]),
            "cohort": report["cohort"], "execution_counts": report["execution_counts"],
            "write_set": packet["write_set"], "budgets": packet["budgets"]}


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _validate_frozen_refs(packet: dict[str, Any]) -> None:
    for ref in (*packet["runtime_code"], *packet["inputs"]["terminal_source_output_refs"],
                packet["inputs"]["plan"], packet["inputs"]["t0_freeze"],
                packet["inputs"]["t0_frame_time_map"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("T3_SCENE_FROZEN_SHA_DRIFT:" + ";".join(errors))


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer):
        raise RuntimeError("T3_SCENE_WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    rows = [row for row in state["tasks"] if row["task_id"] == TASK_ID]
    allowed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(rows) != 1 or rows[0]["status"] != "RUNNING"
            or rows[0]["run_signature"] != packet["run_signature"]
            or rows[0]["pid"] != writer["pid"]
            or rows[0]["proc_start_ticks"] != writer["proc_start_ticks"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(allowed) != 1 or allowed[0]["task_id"] != TASK_ID
            or allowed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("T3_SCENE_ACTIVE_TASK_OR_PACKET_DRIFT")
    _validate_frozen_refs(packet)


def register(report: dict[str, Any], packet: dict[str, Any]) -> None:
    """One governance CAS. Never reached by ``--check-only``."""
    if datetime.now().astimezone() >= CAMPAIGN_END:
        raise RuntimeError("T3_SCENE_CAMPAIGN_HARD_CUTOFF_PASSED")
    if any(path.exists() for path in (ATTEMPT, PACKET, RECEIPT)):
        raise RuntimeError("T3_SCENE_FRESH_ATTEMPT_REQUIRED")
    if (packet["writer"]["pid"] != os.getpid()
            or packet["writer"]["proc_start_ticks"] !=
            process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("T3_SCENE_WRITER_IDENTITY_DRIFT")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if (artifact_ref(TASK_STATE_PATH) != packet["inputs"]["pre_register_task_state"]
            or publisher_blockers(state, index)):
        raise RuntimeError("T3_SCENE_GOVERNANCE_CAS_CONFLICT")
    if _sha(report) != packet["inputs"]["pre_register_matrix_sha256"]:
        raise RuntimeError("T3_SCENE_MATRIX_SHA_DRIFT")
    _validate_frozen_refs(packet)
    if datetime.now().astimezone() >= CAMPAIGN_END:
        raise RuntimeError("T3_SCENE_CAMPAIGN_HARD_CUTOFF_PASSED")
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    atomic_json(ATTEMPT / "MATRIX.json", report)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "T0_SIX_FROZEN_SESSIONS",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(PACKET),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": TASK_ID, "session": "T0_SIX_FROZEN_SESSIONS",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "ONE_CPU_NO_GPU",
                          "stop_condition": "one immutable six-session matrix seal or deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("T3_SCENE_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=REVISION + "_RUNNING", plan_revision=packet["plan_revision"],
                 execution_revision=REVISION, status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=REVISION + "_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)


def result_value(report: dict[str, Any], packet: dict[str, Any],
                 matrix_ref: dict[str, Any], failure: str | None = None) -> dict[str, Any]:
    ready, blockers = readiness(report)
    if not ready:
        failure = failure or "T3_SCENE_TERMINAL_CHAIN_DRIFT:" + ";".join(blockers)
    status = "PASSED" if failure is None else "FAILED_RUNTIME_FINAL"
    source_output = {}
    for sid, row in sorted(report["sessions"].items()):
        evidence = row.get("evidence", {})
        source_output[sid] = {
            "action_rgb_source": evidence.get("action_rgb_source"),
            "scene_raw_source": evidence.get("scene_raw_source"),
            "scene_camera_params": evidence.get("scene_camera_params"),
            "mask_result": evidence.get("scene_mask_result"),
            "clean_result": evidence.get("scene_clean_result"),
            "clean_internal_videos": {name: value.get("artifact") for name, value in
                                      evidence.get("scene_clean_internal_videos", {}).items()},
            "robot_videos": {name: value.get("video") for name, value in
                             row.get("robot", {}).get("videos", {}).items()},
        }
    return {
        "schema_version": "exact78-t3-six-scene-joined-seal-result-v2",
        "task_id": TASK_ID, "status": status,
        "report_integrity": "SEALED_ACTION_SCENE_SHA_CHAINS" if status == "PASSED" else "FAILED",
        "failure_code": failure, "matrix": matrix_ref,
        "cohort": report["cohort"], "execution_counts": report["execution_counts"],
        "source_output_sha": source_output,
        "mono_stereo_mapping": "NOT_PROVEN_FOR_FOUR_TRACKER_SESSIONS",
        "vst_timebase": "TIMEBASE_UNVERIFIED_FOR_TWO_0915_SESSIONS",
        "geometry_qualified": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "ai_review": "NOT_EVALUATED", "user_review": "PER_SESSION_PENDING",
        "general_runnable_operation": None, "claim_limit": packet["claim_limit"],
    }


def finish(packet: dict[str, Any], *, failure: str | None = None) -> dict[str, Any]:
    _fence(packet)
    matrix_path = ATTEMPT / "MATRIX.json"
    report = load_json(matrix_path)
    if _sha(report) != packet["inputs"]["pre_register_matrix_sha256"]:
        failure = failure or "T3_SCENE_MATRIX_SHA_DRIFT"
    if (ATTEMPT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("T3_SCENE_TERMINAL_RESULT_ALREADY_EXISTS")
    result = result_value(report, packet, artifact_ref(matrix_path), failure)
    atomic_json(ATTEMPT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-t3-six-scene-joined-receipt-v2",
                          "task_id": TASK_ID, "status": result["status"],
                          "result": artifact_ref(ATTEMPT / "RESULT.json"),
                          "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=result["status"], pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="REPORT_INTEGRITY_ONLY" if result["status"] == "PASSED" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("T3_SCENE_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=REVISION + "_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=REVISION,
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=REVISION + "_" + result["status"],
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)
    return {"status": result["status"],
            "result": artifact_ref(ATTEMPT / "RESULT.json"),
            "governance_revision": published["governance_revision"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        result = dry_run()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 2 if result["status"] == "DRY_RUN_H24_CUTOFF" else 0
    report = candidate_report()
    packet = packet_value(report)
    register(report, packet)
    result = finish(packet)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
