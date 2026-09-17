"""Publish a receipt-stable R3 contract refresh without changing worker state."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)


ATTEMPTS_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/MIGRATE-00/attempts"
PIPELINE_PACKET_INDEX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/TASK_PACKET_INDEX.json"
PIPELINE_TERMINALS = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0004_packet_hardening/TASK_TERMINALS.json"
DEPTH10_RECEIPT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_10/attempts/attempt_0003_real_play_cards_0910_001/RUN_RECEIPT.json"
DEPTH20_RECEIPT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_20/attempts/attempt_0003_real_input_preflight/RUN_RECEIPT.json"
LOCAL_DEPTH_BLOCKER = {
    "name": "Sensor-line DEPTH-10/20 play_cards_0910_001 prerequisites",
    "status": "BLOCKED_PREREQ",
    "scope": "New sensor-line play_cards_0910_001 fresh Stereo QA and bounded wrist correction only; exact78 Depth authority is unchanged.",
    "resolution": "Fix the selected-eye/SBS source-index mapping, pass same-session rectification P90 <=5 px, run fresh FoundationStereo plus LR/registration QA, then rerun DEPTH-20; Controller/MANUS branches remain independent.",
}
V76_REGISTRATION_BLOCKER = {
    "name": "Robot v76 registration preflight",
    "status": "BLOCKED_PREREQ",
    "scope": "Exact78 Robot successor scheduling only; v75 remains immutable FAILED_RUNTIME_FINAL.",
    "resolution": "Publish complete immutable v76 task/state/index candidates, then register them through the governance aggregator; preparation directories alone are not executable routing authority.",
}

R3_PACKET_MIGRATIONS = {
    "DEPTH-00": [],
    "DEPTH-10": ["sensor_h3_stereo_v1"],
    "DEPTH-20": ["sensor_h3_stereo_v1"],
    "MASK-ROLE": ["successor_role_mask_v71"],
    "MASK-OBJECT": ["successor_object_identity_v71"],
    "ATLAS-10": ["exact78_clean_r70_v71"],
    "DONOR-10K": ["exact78_clean_r70_v71"],
    "CLEAN-DOMAINS-R3": ["exact78_clean_r70_v71"],
    "CONTACT-10": ["contact_evidence_dag_v1"],
    "OCCLUSION-SILVER-R3": ["occlusion_silver_v1"],
}


def _stable_current() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    first = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    second = load_json(RECEIPT_PATH)
    if first.get("generation_id") != second.get("generation_id"):
        raise RuntimeError("current receipt changed during read")
    for reference in first.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt reference conflict: " + "; ".join(errors))
    return first, authority, state


def _successor_packet_index(receipt: dict[str, Any]) -> dict[str, Any]:
    current_reference = receipt.get("files", {}).get("task_packet_index_current")
    if not isinstance(current_reference, dict):
        raise RuntimeError("current receipt does not bind a task packet index")
    errors = validate_artifact_ref(current_reference)
    if errors:
        raise RuntimeError("current packet index reference conflict: " + "; ".join(errors))
    predecessor_path = Path(current_reference["path"])
    predecessor = load_json(predecessor_path)
    predecessor_entries = copy.deepcopy(predecessor.get("task_packets", []))
    existing_ids = [str(item.get("task_id")) for item in predecessor_entries]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("current packet index has duplicate task ids")

    source = load_json(PIPELINE_PACKET_INDEX)
    terminals = load_json(PIPELINE_TERMINALS)
    terminal_by_id = {str(item["task_id"]): item for item in terminals.get("rows", [])}
    added: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    for source_entry in source.get("packets", []):
        task_id = str(source_entry["task_id"])
        errors = validate_artifact_ref(source_entry)
        if errors:
            raise RuntimeError(f"R3 packet reference conflict for {task_id}: " + "; ".join(errors))
        path = Path(source_entry["path"])
        try:
            relative = str(path.resolve().relative_to(REPO_ROOT))
        except ValueError as error:
            raise RuntimeError(f"R3 packet lies outside repository: {path}") from error
        terminal = terminal_by_id.get(task_id)
        if terminal is None:
            raise RuntimeError(f"R3 packet lacks a bounded terminal: {task_id}")
        for reference in (terminal.get("result"), terminal.get("receipt")):
            if not isinstance(reference, dict):
                raise RuntimeError(f"R3 terminal reference missing for {task_id}")
            errors = validate_artifact_ref(reference)
            if errors:
                raise RuntimeError(f"R3 terminal reference conflict for {task_id}: " + "; ".join(errors))
        entry = {
            "task_id": task_id,
            "packet_path": relative,
            "packet_sha256": source_entry["sha256"],
            "routing_revision": "R7_3",
            "routing_status": "CURRENT_R3_CONTRACT",
            "development_terminal_status": terminal["status"],
            "development_result": terminal["result"],
            "development_receipt": terminal["receipt"],
            "authority_promoted": False,
            "refines_legacy_task_ids": R3_PACKET_MIGRATIONS[task_id],
        }
        if task_id in existing_ids:
            old = next(item for item in predecessor_entries if str(item.get("task_id")) == task_id)
            if old.get("packet_sha256") != entry["packet_sha256"]:
                raise RuntimeError(f"current task id already has a different packet: {task_id}")
        else:
            added.append(entry)
        migrations.append({
            "current_task_id": task_id,
            "legacy_task_ids_retained": R3_PACKET_MIGRATIONS[task_id],
            "migration_type": "R3_REFINEMENT_WITH_HISTORY_RETAINED",
            "authority_promoted": False,
        })
    entries = predecessor_entries + added
    if entries[: len(predecessor_entries)] != predecessor_entries:
        raise RuntimeError("predecessor task packet entries were not preserved")
    statuses = [
        item["development_terminal_status"] for item in entries
        if str(item.get("task_id")) in R3_PACKET_MIGRATIONS
    ]
    return {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "R7_3",
        "plan_revision": "chaoyang-v7.1",
        "execution_revision": "R3",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "source_r3_packet_index": artifact_ref(PIPELINE_PACKET_INDEX),
        "source_r3_terminal_index": artifact_ref(PIPELINE_TERMINALS),
        "legacy_entries_retained": True,
        "task_packets": entries,
        "routing_migrations": migrations,
        "development_terminal_summary": {
            "passed_development": statuses.count("PASSED_DEVELOPMENT"),
            "blocked_prereq": statuses.count("BLOCKED_PREREQ"),
            "authority_promoted": False,
        },
        "claim_limit": "R3 packets are current execution contracts with development terminals only; retained legacy packets remain historical routes and no stage authority is promoted.",
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_json(path, value)


def _write_text_once(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_write(path, encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cas-retries", type=int, default=20)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.25)
    parser.add_argument("--attempt-name", default="attempt_0003_incremental_contract_refresh")
    args = parser.parse_args()
    if args.max_cas_retries < 1:
        raise RuntimeError("max CAS retries must be positive")
    attempt_root = ATTEMPTS_ROOT / args.attempt_name
    successor_packet_index = attempt_root / "TASK_PACKET_INDEX.json"
    for reference_path in (DEPTH10_RECEIPT, DEPTH20_RECEIPT):
        errors = validate_artifact_ref(artifact_ref(reference_path))
        if errors:
            raise RuntimeError("development receipt is not closed: " + "; ".join(errors))

    last_error: Exception | None = None
    before_v75: dict[str, Any] | None = None
    published: dict[str, Any] | None = None
    for _ in range(args.max_cas_retries):
        try:
            receipt, authority_current, state_current = _stable_current()
            authority = copy.deepcopy(authority_current)
            state = copy.deepcopy(state_current)
            before_v75 = copy.deepcopy(next(
                item for item in state.get("tasks", [])
                if item.get("task_id") == "robot_geometry_expansion_v75"
            ))
            if not any(item.get("name") == LOCAL_DEPTH_BLOCKER["name"] for item in state.get("blockers", [])):
                state.setdefault("blockers", []).append(copy.deepcopy(LOCAL_DEPTH_BLOCKER))
            if before_v75.get("status") in {"PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE", "BLOCKED_EXTERNAL", "BLOCKED_REFERENCE_PROOF", "UNKNOWN_VERIFICATION_REQUIRED", "CANCELLED"}:
                if not any(item.get("task_id") == "robot_geometry_expansion_v76" for item in state.get("tasks", [])):
                    state["next_task"] = None
                    if not any(item.get("name") == V76_REGISTRATION_BLOCKER["name"] for item in state.get("blockers", [])):
                        state.setdefault("blockers", []).append(copy.deepcopy(V76_REGISTRATION_BLOCKER))
            successor_index = _successor_packet_index(receipt)
            state["recent_events"] = (
                state.get("recent_events", [])
                + [{
                    "task_id": "r3_governance_incremental_refresh",
                    "session": "depth_visualaux_robot_cleanup_contracts",
                    "attempt": 1,
                    "status": "PASSED",
                    "created_at": now_iso(),
                    "message": "Refreshed current code/test/document contracts; no model task or authority was promoted.",
                    "depth10_receipt": artifact_ref(DEPTH10_RECEIPT),
                    "depth20_receipt": artifact_ref(DEPTH20_RECEIPT),
                }]
            )[-100:]
            published = publish_bundle(
                authority,
                state,
                event_type="R3_INCREMENTAL_CONTRACT_REFRESH",
                expected_revision=int(receipt["governance_revision"]),
                generator_path=Path(__file__),
                task_packet_index_path=successor_packet_index,
                task_packet_index_value=successor_index,
            )
            after_v75 = next(
                item for item in load_json(TASK_STATE_PATH).get("tasks", [])
                if item.get("task_id") == "robot_geometry_expansion_v75"
            )
            if after_v75 != before_v75:
                raise RuntimeError("v75 task row changed across a successful publication")
            break
        except RuntimeError as error:
            last_error = error
            if "CAS revision mismatch" not in str(error) and "changed during read" not in str(error):
                raise
            time.sleep(args.retry_delay_seconds)
    if published is None or before_v75 is None:
        raise RuntimeError(f"CAS retries exhausted: {last_error}")

    created = now_iso()
    result_path = attempt_root / "RESULT.json"
    metrics_path = attempt_root / "METRICS.json"
    decision_path = attempt_root / "DECISION.md"
    next_action_path = attempt_root / "NEXT_ACTION.json"
    task_packet_path = attempt_root / "TASK_PACKET.json"
    manifest_path = attempt_root / "ARTIFACT_MANIFEST.json"
    run_receipt_path = attempt_root / "RUN_RECEIPT.json"
    _write_once(result_path, {
        "schema_version": "r3-incremental-contract-refresh-result-v1",
        "task_id": "r3_governance_incremental_refresh",
        "status": "PASSED",
        "generated_at": created,
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "v75_state_preserved": before_v75,
        "depth10_receipt": artifact_ref(DEPTH10_RECEIPT),
        "depth20_receipt": artifact_ref(DEPTH20_RECEIPT),
        "current_task_packet_index": artifact_ref(successor_packet_index),
        "authority_promoted": False,
        "model_process_started": False,
        "claim_limit": "Governance contract refresh only; development watchers and Depth preflights are not authority.",
    })
    _write_once(metrics_path, {"status": "PASSED", "authority_promotions": 0, "model_processes_started": 0, "v75_state_mutations": 0})
    _write_once(task_packet_path, {
        "task_id": "r3_governance_incremental_refresh",
        "objective": "CAS-refresh current document, algorithm and regression contracts while preserving the latest v75 worker state.",
        "read_set": [str(RECEIPT_PATH), str(DEPTH10_RECEIPT), str(DEPTH20_RECEIPT)],
        "write_set": [str(RECEIPT_PATH.parent), str(attempt_root)],
        "stop_condition": "PASSED or bounded CAS/reference failure",
        "claim_limit": "Governance only.",
    })
    _write_text_once(decision_path, "# 决定\n\n刷新 R3 文档、算法和回归闭包；保留 v75 最新终态，Depth 与两个 watcher 均不晋升 authority。\n")
    _write_once(next_action_path, {"action": "FOLLOW_CURRENT_TASK_STATE", "claim_limit": "Do not infer v76 registration from a preparation directory."})
    _write_once(manifest_path, {
        "schema_version": "r3-incremental-contract-refresh-artifact-manifest-v1",
        "artifacts": [artifact_ref(path) for path in (result_path, metrics_path, task_packet_path, decision_path, next_action_path, successor_packet_index)],
    })
    _write_once(run_receipt_path, {
        "schema_version": "r3-incremental-contract-refresh-run-receipt-v1",
        "task_id": "r3_governance_incremental_refresh",
        "status": "PASSED",
        "generated_at": created,
        "result": artifact_ref(result_path),
        "artifact_manifest": artifact_ref(manifest_path),
        "metrics": artifact_ref(metrics_path),
        "authority_promoted": False,
    })
    print(json.dumps({
        "status": "PASSED",
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "v75_status": before_v75.get("status"),
        "authority_promoted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
