#!/usr/bin/env python3
"""Bootstrap and supervise the frozen 0915 Robot recovery V2.1 campaign.

This parent operation owns only governance heartbeat and immutable campaign
metadata.  Algorithm work packages write below ``packages/<id>/`` and are
admitted by explicit result receipts; they never mutate the 0915 source or
processed dataset roots.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from chaoyang.governance.common import (
    REPO_ROOT,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    process_identity,
    sha256_file,
)


TASK_ID = "0915_robot_quality_recovery_15h_v2"
PHASE = "ROBOT_QUALITY_RECOVERY_0915_15H_V21"
RUN_ROOT = REPO_ROOT / "_run/current/0915_robot_quality_recovery_15h_v2"
CLOCK = RUN_ROOT / "WINDOW_CLOCK.json"
OUTPUT = RUN_ROOT / "attempts/attempt_0001"
OLD_INVENTORY = REPO_ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001"
OLD_SOURCE = OLD_INVENTORY / "SOURCE_GROUP_MANIFEST.json"
OLD_BATCH = OLD_INVENTORY / "BATCH_MANIFEST.json"
OLD_FINAL_AUDIT = REPO_ROOT / "_run/current/0915_robot15h_window_release_audit_v1/attempts/attempt_0001/FINAL_AUDIT.json"
DOC_ROOT = REPO_ROOT / "docs/chaoyang_robot_recovery_15h_v2"

COHORTS = {
    "W0": (
        "play_cards_0915_031",
        "play_cards_0915_119",
        "get_potato_chips_0915_007",
        "get_potato_chips_0915_042",
    ),
    "W1_DIAG": ("play_cards_0915_044", "get_potato_chips_0915_097"),
    "W1_ADOPTION": ("play_cards_0915_106", "get_potato_chips_0915_029"),
    "H9_FINAL_HOLDOUT": (
        "play_cards_0915_054",
        "play_cards_0915_003",
        "get_potato_chips_0915_068",
        "get_potato_chips_0915_056",
    ),
    "EXTRA_FINAL": (
        "play_cards_0915_030",
        "play_cards_0915_085",
        "get_potato_chips_0915_017",
        "get_potato_chips_0915_095",
    ),
}
PACKAGE_IDS = ("P0", "A0", "B0", "C0", "A1", "B1", "C1", "A2", "F0")


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def heartbeat(status: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            TASK_ID,
            "--pid",
            str(os.getpid()),
            "--status",
            status,
            "--phase",
            PHASE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet.get("task_id") != TASK_ID or packet.get("plan_revision") != "0915_ROBOT_RECOVERY_15H_V2_1":
        raise RuntimeError("unexpected V2.1 parent task packet")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    rows = [row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID]
    if len(rows) != 1 or rows[0].get("execution_allowed") is not True:
        raise RuntimeError("V2.1 parent is not uniquely routable")
    if rows[0].get("packet_sha256") != sha256_file(packet_path):
        raise RuntimeError("V2.1 route packet SHA mismatch")
    state = load_json(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("V2.1 parent is not current next_task")
    return packet, packet_path


def live_session_ref(row: dict[str, Any]) -> dict[str, Any]:
    stereo = Path(str(row["source_stereo"]))
    if not stereo.is_file():
        raise RuntimeError(f"missing frozen source stereo: {row['session_id']}")
    if row.get("timeline_anomaly") is not False:
        raise RuntimeError(f"timeline anomaly in frozen cohort: {row['session_id']}")
    return {
        "session_id": row["session_id"],
        "task": row["task"],
        "source_group": row["source_group"],
        "frame_count": row["frame_count"],
        "capture_time": row["independence_evidence"]["camera_captured_at_utc"],
        "vst_sha256": row["independence_evidence"]["vst_sha256"],
        "vst_qpc_sha256": row["independence_evidence"]["vst_qpc_sha256"],
        "qpc_first": row["independence_evidence"]["vst_qpc_first"],
        "qpc_last": row["independence_evidence"]["vst_qpc_last"],
        "source_snapshot_sha256": row["independence_evidence"]["source_snapshot_sha256"],
        "source_stereo": artifact_ref(stereo),
        "processed_root": row["processed_root"],
        "raw_root": row["raw_root"],
    }


def freeze_campaign(packet: dict[str, Any], packet_path: Path) -> None:
    clock = load_json(CLOCK)
    if clock.get("plan_version") != "v2.1" or int(clock.get("duration_seconds", -1)) != 54000:
        raise RuntimeError("invalid V2.1 window clock")
    source = load_json(OLD_SOURCE)
    if len(source.get("sessions", [])) != 220:
        raise RuntimeError("old source inventory is not the frozen 220-session denominator")
    rows = {str(row["session_id"]): row for row in source["sessions"]}
    expected = {sid for values in COHORTS.values() for sid in values}
    if set(rows).isdisjoint(expected) or expected - set(rows):
        raise RuntimeError("frozen V2.1 cohort is absent from source inventory")
    frozen = {sid: live_session_ref(rows[sid]) for sid in sorted(expected)}
    source_groups = [row["source_group"] for row in frozen.values()]
    vst_shas = [row["vst_sha256"] for row in frozen.values()]
    qpc_shas = [row["vst_qpc_sha256"] for row in frozen.values()]
    captures = [row["capture_time"] for row in frozen.values()]
    if not all(len(values) == len(set(values)) == 16 for values in (source_groups, vst_shas, qpc_shas, captures)):
        raise RuntimeError("V2.1 cohort independence metadata is not unique")
    ordered_qpc = sorted((int(row["qpc_first"]), int(row["qpc_last"]), sid) for sid, row in frozen.items())
    if any(previous[1] >= current[0] for previous, current in zip(ordered_qpc, ordered_qpc[1:])):
        raise RuntimeError("V2.1 cohort QPC intervals overlap")

    old_source_ref = artifact_ref(OLD_SOURCE)
    correction = {
        "schema_version": "0915-robot-recovery-source-group-provenance-correction-v2",
        "window_run_id": clock["run_id"],
        "status": "PASSED_METADATA_VERIFIED_NO_USER_ATTESTATION",
        "base_manifest": old_source_ref,
        "base_manifest_preserved": True,
        "authority_precedence": "THIS_SIDECAR_SUPERSEDES_ONLY_THE_USER_ATTESTATION_CLAIM_FOR_V2_1",
        "superseded_fields": ["status", "independence", "independence_basis.USER_CONFIRMATION_SEPARATELY_RECORDED", "claim_limit"],
        "source_group_independence": "METADATA_VERIFIED_NO_USER_ATTESTATION",
        "user_attestation_present": False,
        "capture_id_scope": "TACTILE_ONLY_NOT_CAMERA_RECORDING_UUID",
        "allowed_evidence": [
            "UNIQUE_VST_SHA256",
            "UNIQUE_VST_QPC_SHA256",
            "NON_OVERLAPPING_VST_QPC_INTERVAL",
            "UNIQUE_CAMERA_CAPTURE_TIMESTAMP",
            "NO_CROSS_SEGMENT_MERGE_EVIDENCE",
        ],
        "removed_evidence": ["USER_CONFIRMATION_SEPARATELY_RECORDED"],
        "global_closure": {"inventory_sessions": 220, "v2_1_frozen_sessions": 16},
        "sessions": frozen,
        "source_mutated": False,
        "claim_limit": "Metadata corroboration only; no user attestation, camera recording UUID, algorithm generalization or physical authority.",
    }
    write_once(OUTPUT / "SOURCE_GROUP_PROVENANCE_CORRECTION_V2.json", correction)

    access_rows = []
    for cohort, ids in COHORTS.items():
        for sid in ids:
            if cohort in {"W0", "W1_DIAG"}:
                access_state = "OPEN_AT_T0"
                open_after = clock["started_at"]
            elif cohort == "W1_ADOPTION":
                access_state = "SEALED_UNTIL_CANDIDATE_FREEZE"
                open_after = "CANDIDATE_FREEZE_V1"
            elif cohort == "H9_FINAL_HOLDOUT":
                access_state = "SEALED_UNTIL_H9_ADOPTION_GATE"
                open_after = clock["algorithm_freeze_at"]
            else:
                access_state = "SEALED_SELECTOR_CONTRACT_UNRESOLVED"
                open_after = "NEVER_IN_V2_1_UNLESS_REGISTERED_SELECTOR_CONTRACT_CLOSES"
            access_rows.append({
                **frozen[sid],
                "cohort_role": cohort,
                "access_state": access_state,
                "open_after": open_after,
                "replacement_allowed": False,
            })
    ledger = {
        "schema_version": "0915-robot-recovery-cohort-access-ledger-v1",
        "window_run_id": clock["run_id"],
        "status": "FROZEN_FAIL_CLOSED",
        "base_batch_manifest": artifact_ref(OLD_BATCH),
        "provenance_correction": artifact_ref(OUTPUT / "SOURCE_GROUP_PROVENANCE_CORRECTION_V2.json"),
        "selector_contract": {
            "salt": "chaoyang_robot_recovery_15h_v2_extra_final_v1",
            "status": "UNRESOLVED_SPECIFICATION_AMBIGUITY",
            "missing_fields": [
                "candidate_pool_query",
                "half_boundary_algorithm",
                "rank_preimage_schema",
                "field_encoding_and_separator",
                "rank_order_and_tie_break",
                "full_candidate_rank_digest",
            ],
            "designated_extra_final": list(COHORTS["EXTRA_FINAL"]),
            "designated_rows_metadata_valid": True,
            "selection_reproducible": False,
            "access_effect": "EXTRA_FINAL_REMAINS_SEALED_FAIL_CLOSED",
        },
        "sessions": access_rows,
        "access_events": [{
            "timestamp": now_iso(),
            "event": "T0_LEDGER_FROZEN",
            "opened": list(COHORTS["W0"] + COHORTS["W1_DIAG"]),
            "sealed": list(COHORTS["W1_ADOPTION"] + COHORTS["H9_FINAL_HOLDOUT"] + COHORTS["EXTRA_FINAL"]),
        }],
        "invariants": {
            "source_mutation_allowed": False,
            "0916_consumption_allowed": False,
            "extra_replacement_allowed": False,
            "algorithm_result_based_selection_allowed": False,
        },
    }
    write_once(OUTPUT / "COHORT_ACCESS_LEDGER_V1.json", ledger)

    technical_paths = [
        DOC_ROOT / "00_START_HERE_ZH.md",
        DOC_ROOT / "01_EXECUTION_PLAN_15H_ZH.md",
        DOC_ROOT / "02_ALGORITHM_AUDIT_LIVE_ZH.md",
        DOC_ROOT / "03_TASK_MANIFEST.template.json",
        DOC_ROOT / "04_ACCEPTANCE_TESTS_ZH.md",
        DOC_ROOT / "05_TECHNICAL_SOURCES_ZH.md",
        DOC_ROOT / "SHA256SUMS.txt",
        OLD_FINAL_AUDIT,
        Path(__file__),
    ]
    snapshot = {
        "schema_version": "0915-robot-recovery-technical-contract-snapshot-index-v1",
        "window_run_id": clock["run_id"],
        "status": "FROZEN",
        "task_packet": artifact_ref(packet_path),
        "clock": artifact_ref(CLOCK),
        "documents_and_code": [artifact_ref(path) for path in technical_paths],
        "weights": "ABSENT_PARENT_COORDINATOR_CHILDREN_MUST_BIND_OWN_WEIGHT_SHA",
        "source_mutated": False,
    }
    write_once(OUTPUT / "TECHNICAL_CONTRACT_SNAPSHOT_INDEX_V1.json", snapshot)

    package_state = {
        "schema_version": "0915-robot-recovery-v21-work-package-ledger-v1",
        "window_run_id": clock["run_id"],
        "updated_at": now_iso(),
        "packages": [
            {
                "id": package_id,
                "status": "READY" if package_id in {"P0", "A0", "B0", "C0"} else "BLOCKED_PREREQ",
                "output_root": str(OUTPUT / "packages" / package_id),
            }
            for package_id in PACKAGE_IDS
        ],
        "claim_limit": "Coordinator readiness only; no algorithm or quality result.",
    }
    write_once(OUTPUT / "WORK_PACKAGE_LEDGER_T0.json", package_state)
    write_once(OUTPUT / "BOOTSTRAP_RESULT.json", {
        "schema_version": "0915-robot-recovery-v21-bootstrap-result-v1",
        "task_id": TASK_ID,
        "window_run_id": clock["run_id"],
        "status": "PASSED",
        "cohort_sessions": 16,
        "opened_sessions": 6,
        "sealed_sessions": 10,
        "extra_selector_status": "UNRESOLVED_FAIL_CLOSED",
        "source_mutated": False,
        "processed_mutated": False,
        "0916_consumed": False,
        "artifacts": {
            "provenance": artifact_ref(OUTPUT / "SOURCE_GROUP_PROVENANCE_CORRECTION_V2.json"),
            "access_ledger": artifact_ref(OUTPUT / "COHORT_ACCESS_LEDGER_V1.json"),
            "technical_snapshot": artifact_ref(OUTPUT / "TECHNICAL_CONTRACT_SNAPSHOT_INDEX_V1.json"),
            "work_packages": artifact_ref(OUTPUT / "WORK_PACKAGE_LEDGER_T0.json"),
        },
        "claim_limit": packet["claim_limit"],
    })


def monitor() -> None:
    clock = load_json(CLOCK)
    deadline = datetime.fromisoformat(str(clock["deadline_at"])).timestamp()
    while time.time() < deadline:
        package_terminals = {}
        for package_id in PACKAGE_IDS:
            result = OUTPUT / "packages" / package_id / "RESULT.json"
            package_terminals[package_id] = artifact_ref(result) if result.is_file() else None
        atomic_json(OUTPUT / "COORDINATOR_STATE.json", {
            "schema_version": "0915-robot-recovery-v21-coordinator-state-v1",
            "task_id": TASK_ID,
            "window_run_id": clock["run_id"],
            "status": "RUNNING",
            "updated_at": now_iso(),
            "pid": os.getpid(),
            "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
            "package_terminals": package_terminals,
            "deadline_at": clock["deadline_at"],
        })
        heartbeat("RUNNING")
        stop = OUTPUT / "STOP_REQUESTED.json"
        if stop.is_file():
            break
        time.sleep(45)
    atomic_json(OUTPUT / "COORDINATOR_EXIT.json", {
        "schema_version": "0915-robot-recovery-v21-coordinator-exit-v1",
        "task_id": TASK_ID,
        "window_run_id": clock["run_id"],
        "status": "MONITOR_EXITED_NOT_CAMPAIGN_FINALIZATION",
        "at": now_iso(),
        "deadline_reached": time.time() >= deadline,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()
    if args.output_root.resolve() != OUTPUT.resolve():
        raise RuntimeError("fixed V2.1 attempt namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("invalid executor epoch or fencing token")
    packet, packet_path = validate_route()
    if args.executor_epoch != int(packet.get("executor_epoch", -1)):
        raise RuntimeError("executor epoch differs from immutable parent packet")
    if not OUTPUT.exists():
        OUTPUT.mkdir(parents=True)
        write_once(OUTPUT / "CLAIM.json", {
            "schema_version": "0915-robot-recovery-v21-parent-claim-v1",
            "task_id": TASK_ID,
            "window_run_id": load_json(CLOCK)["run_id"],
            "status": "CLAIMED",
            "pid": os.getpid(),
            "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
            "executor_epoch": args.executor_epoch,
            "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode("utf-8")).hexdigest(),
            "unique_write_root": str(OUTPUT),
            "task_packet": artifact_ref(packet_path),
        })
        signature_payload = {
            "schema_version": "0915-robot-recovery-v21-parent-run-signature-v1",
            "task_id": TASK_ID,
            "window_run_id": load_json(CLOCK)["run_id"],
            "executor_epoch": args.executor_epoch,
            "weights": "ABSENT",
            "gpu_used": False,
            "source_mutation_allowed": False,
            "processed_mutation_allowed": False,
            "0916_consumption_allowed": False,
            "task_packet": artifact_ref(packet_path),
            "clock": artifact_ref(CLOCK),
            "inputs": [artifact_ref(OLD_SOURCE), artifact_ref(OLD_BATCH), artifact_ref(OLD_FINAL_AUDIT)],
            "code": [artifact_ref(Path(__file__))],
        }
        write_once(OUTPUT / "RUN_SIGNATURE.json", {
            **signature_payload,
            "run_signature_sha256": canonical_sha(signature_payload),
        })
        heartbeat("CLAIMED")
        freeze_campaign(packet, packet_path)
    elif not (OUTPUT / "BOOTSTRAP_RESULT.json").is_file():
        raise RuntimeError("existing V2.1 attempt is incomplete and cannot be silently resumed")
    heartbeat("RUNNING")
    print(json.dumps({
        "status": "RUNNING",
        "task_id": TASK_ID,
        "window_run_id": load_json(CLOCK)["run_id"],
        "bootstrap": artifact_ref(OUTPUT / "BOOTSTRAP_RESULT.json"),
        "monitoring": not args.bootstrap_only,
    }, ensure_ascii=False))
    if not args.bootstrap_only:
        monitor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
