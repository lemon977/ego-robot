#!/usr/bin/env python3
"""Build the fail-closed F0 closure for the finite 0915 recovery campaign.

This publisher is deliberately separate from the long-running coordinator.  It
may only run after the recorded wall-clock deadline and after the coordinator
has emitted ``COORDINATOR_EXIT.json``.  It never turns diagnostic completion
into algorithm success: the frozen W0/W1, Contact, Mask and Robot evidence is
summarised exactly as recorded and the parent closes ``REJECTED_QUALITY`` when
no recovery was adopted.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TASK_ID = "0915_robot_quality_recovery_15h_v2"
RUN_ID = "0915-robot-quality-recovery-v21-20260919T234412+0800"
REQUIRED_PARENT_INPUTS = (
    "SOURCE_GROUP_PROVENANCE_CORRECTION_V2.json",
    "COHORT_ACCESS_LEDGER_V1.json",
    "W0_FAILURE_MATRIX_V2.json",
    "CONSUMER_ADMISSION_V1.json",
    "WINDOW_VALIDITY_V1.json",
    "CONTACT_OBSERVABILITY_AUDIT_V1.json",
    "LOCAL_STEREO_METRIC_DEV_V1.json",
    "CANDIDATE_FREEZE_V1.json",
    "TECHNICAL_CONTRACT_SNAPSHOT_INDEX_V1.json",
    "CLAIM.json",
    "RUN_SIGNATURE.json",
    "WORK_PACKAGE_LEDGER_T0.json",
    "P0_RESOLUTION_V1.json",
    "R2_CONTRACT_SNAPSHOT_V1.json",
)
REQUIRED_VERIFICATION_CHECKS = {
    "pytest_full",
    "video_decode",
    "markdown_links",
    "governance",
    "reference_closure",
    "source_read_only",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def validate_ref(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("artifact reference is not an object")
    path = Path(str(value.get("path", "")))
    if not path.is_file():
        raise RuntimeError(f"referenced artifact missing: {path}")
    if path.stat().st_size != value.get("bytes") or sha256(path) != value.get("sha256"):
        raise RuntimeError(f"referenced artifact drifted: {path}")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def require_identity(value: dict[str, Any], path: Path) -> None:
    task_id = value.get("task_id")
    if task_id not in (None, TASK_ID):
        raise RuntimeError(f"wrong task identity in {path}: {task_id}")
    run_id = value.get("window_run_id")
    if run_id not in (None, RUN_ID):
        raise RuntimeError(f"wrong run identity in {path}: {run_id}")


def require_inputs(root: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_PARENT_INPUTS:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"required parent evidence missing: {path}")
        value = load_json(path)
        require_identity(value, path)
        evidence[name] = value
    package_paths = {
        "P0": root / "packages/P0/RESULT.json",
        "A0": root / "packages/A0/RESULT.json",
        "A1": root / "packages/A1/RESULT.json",
        "A2": root / "packages/A2/RESULT.json",
        "A7": root / "packages/A7_STOP_NO_AUTHORITY_AUDIT/RESULT.json",
        "B0": root / "packages/B0/RESULT.json",
        "B1": root / "packages/B1/RESULT.json",
        "B1R": root / "packages/B1R/RESULT.json",
        "B2": root / "packages/B2_DEPTH_TO_OBJECT_POKER044/RESULT.json",
        "B3": root / "packages/B3_FINAL_TERMINAL_AUDIT/RESULT.json",
        "C0": root / "packages/C0/RESULT.json",
        "C1": root / "packages/C1/RESULT.json",
        "D2": root / "packages/D2_FRESH_FILL_OFFLINE_V1/RESULT.json",
    }
    for name, path in package_paths.items():
        if not path.is_file():
            raise RuntimeError(f"required package terminal missing: {name}: {path}")
        value = load_json(path)
        require_identity(value, path)
        evidence[f"PACKAGE_{name}"] = value
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--coordinator-exit", required=True, type=Path)
    parser.add_argument("--verification-report", required=True, type=Path)
    parser.add_argument("--terminal-receipt", required=True, type=Path)
    parser.add_argument("--now", help="ISO timestamp; tests only")
    args = parser.parse_args()

    root = args.campaign_root.resolve(strict=True)
    coordinator_exit = args.coordinator_exit.resolve(strict=True)
    verification_report = args.verification_report.resolve(strict=True)
    state_path = root / "COORDINATOR_STATE.json"
    state = load_json(state_path)
    if state.get("task_id") != TASK_ID or state.get("window_run_id") != RUN_ID:
        raise RuntimeError("coordinator identity mismatch")
    exit_value = load_json(coordinator_exit)
    require_identity(exit_value, coordinator_exit)
    deadline = parse_time(str(state["deadline_at"]))
    now = parse_time(args.now) if args.now else datetime.now(deadline.tzinfo)
    if now < deadline:
        raise RuntimeError("F0 cannot be published before the immutable deadline")
    if (
        exit_value.get("status") != "MONITOR_EXITED_NOT_CAMPAIGN_FINALIZATION"
        or exit_value.get("deadline_reached") is not True
    ):
        raise RuntimeError("coordinator did not prove a deadline terminal")

    verification = load_json(verification_report)
    if verification.get("status") != "PASS":
        raise RuntimeError("verification report is not PASS")
    checks = verification.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("verification report has no check map")
    missing_checks = sorted(REQUIRED_VERIFICATION_CHECKS - checks.keys())
    failed_checks = sorted(name for name in REQUIRED_VERIFICATION_CHECKS if checks.get(name) != "PASS")
    if missing_checks or failed_checks:
        raise RuntimeError(
            f"verification report is incomplete: missing={missing_checks}, failed={failed_checks}"
        )
    for artifact in verification.get("evidence", []):
        validate_ref(artifact)

    evidence = require_inputs(root)
    a0 = evidence["W0_FAILURE_MATRIX_V2.json"]
    a1 = evidence["PACKAGE_A1"]
    a2 = evidence["PACKAGE_A2"]
    contact = evidence["CONTACT_OBSERVABILITY_AUDIT_V1.json"]
    metric = evidence["LOCAL_STEREO_METRIC_DEV_V1.json"]
    b3 = evidence["PACKAGE_B3"]
    d2 = evidence["PACKAGE_D2"]

    if a0.get("strict_session_passed") != 0 or a0.get("r0_quality_admitted") != 0:
        raise RuntimeError("W0 evidence no longer supports the recorded rejection")
    if a1.get("counts", {}).get("session_strict_pass") != 0:
        raise RuntimeError("W1-DIAG unexpectedly contains a strict pass")
    if a2.get("counts", {}).get("session_strict_pass") != 0 or a2.get("r0_quality_upgraded") is not False:
        raise RuntimeError("W1-ADOPTION evidence no longer supports rejection")
    if contact.get("within_fixed_5mm_rows") != 0 or metric.get("r1_e_development_allowed") is not False:
        raise RuntimeError("Contact evidence no longer supports R1-E closure")
    if b3.get("decision") != "STOP_NO_THIRD_ATTEMPT":
        raise RuntimeError("B3 terminal decision is not frozen")

    created_at = now.isoformat()
    adoption = {
        "schema_version": "0915-robot-recovery-v21-adoption-decision-v1",
        "task_id": TASK_ID,
        "window_run_id": RUN_ID,
        "created_at": created_at,
        "decision": "REJECT",
        "candidate_id": evidence["CANDIDATE_FREEZE_V1.json"].get("candidate_id"),
        "w1_adoption_sessions": 2,
        "w1_adoption_strict_pass": 0,
        "r0_quality_upgraded": False,
        "h9_opened": False,
        "extra_final_opened": False,
        "reason_codes": [
            "W1_ADOPTION_STRICT_PASS_ZERO_OF_TWO",
            "R0_QUALITY_UPGRADE_ZERO",
            "NO_AUTHORIZED_BOUNDED_SUCCESSOR",
        ],
        "claim_limit": "Independent adoption rows reject the frozen candidate; no later tuning, holdout opening, Robot quality, training, control or deployment authority.",
    }
    write_new(root / "ADOPTION_DECISION_V1.json", adoption)

    audit = {
        "schema_version": "0915-robot-recovery-v21-algorithm-audit-final-v2",
        "task_id": TASK_ID,
        "window_run_id": RUN_ID,
        "created_at": created_at,
        "classification": "REJECTED_NO_RECOVERY",
        "human_robot": {
            "w0_strict": "0/4",
            "w0_r0_quality": "0/4",
            "w1_diag_strict": "0/2",
            "w1_adoption_strict": "0/2",
            "adoption": "REJECT",
            "h9": "NOT_OPENED",
            "extra_final": "SEALED_NOT_OPENED",
        },
        "contact": {
            "pair_rows": contact["pair_rows"],
            "direct_visible_finger_surface_rows": contact["direct_visible_finger_surface_rows"],
            "unique_visible_surface_sample_keys": contact["unique_visible_surface_sample_keys"],
            "metric_pair_rows": contact["metric_pair_rows"],
            "inside_finite_patch_rows": contact["inside_finite_patch_rows"],
            "within_fixed_5mm_rows": contact["within_fixed_5mm_rows"],
            "pixel_registration_bound_rows": contact["pixel_registration_bound_rows"],
            "local_stereo_metric_dev": metric["local_stereo_metric_dev"],
            "r1_e_allowed": metric["r1_e_development_allowed"],
        },
        "mask": {
            "terminal": b3["status"],
            "decision": b3["decision"],
            "attempt_count": len(b3.get("attempts", [])),
            "authority_promoted": b3.get("authority_promoted"),
        },
        "clean_visual": {
            "status": d2["status"],
            "model_sessions": d2.get("model", {}).get("executed_for_sessions", []),
            "clean_terminal": d2["clean_terminal"],
            "training_eligible": d2["training_eligible"],
        },
        "authority": {
            "training_eligible": False,
            "contact_ground_truth": False,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "external_metric_authority": False,
        },
        "adoption_decision": ref(root / "ADOPTION_DECISION_V1.json"),
        "claim_limit": "Final evidence audit: useful diagnostics were produced, but no cross-session HaWoR/R0 recovery, strict Contact, R1-E, training, control or deployment authority was achieved.",
    }
    write_new(root / "ALGORITHM_AUDIT_LIVE_V2.json", audit)

    final_audit = {
        "schema_version": "0915-robot-recovery-v21-final-audit-v1",
        "task_id": TASK_ID,
        "window_run_id": RUN_ID,
        "created_at": created_at,
        "status": "PASS",
        "campaign_classification": "REJECTED_NO_RECOVERY",
        "all_started_packages_terminal": True,
        "candidate_adoption": "REJECT",
        "h9_opened": False,
        "extra_final_opened": False,
        "required_authority_boundaries_preserved": True,
        "source_mutation_claim": "NO_MUTATION_OBSERVED_WITHIN_RECORDED_ACCESS_LEDGER",
        "publisher_code": ref(Path(__file__)),
        "verification": ref(verification_report),
        "coordinator_exit": ref(coordinator_exit),
        "algorithm_audit": ref(root / "ALGORITHM_AUDIT_LIVE_V2.json"),
        "adoption_decision": ref(root / "ADOPTION_DECISION_V1.json"),
        "claim_limit": "Audit completeness PASS does not mean algorithm recovery PASS; the algorithm outcome is REJECTED_NO_RECOVERY.",
    }
    write_new(root / "FINAL_AUDIT.json", final_audit)

    result = {
        "schema_version": "0915-robot-recovery-v21-result-v1",
        "task_id": TASK_ID,
        "window_run_id": RUN_ID,
        "created_at": created_at,
        "status": "REJECTED_QUALITY",
        "campaign_classification": "REJECTED_NO_RECOVERY",
        "first_blocker": "NO_CROSS_SESSION_HAWOR_OR_R0_QUALITY_RECOVERY",
        "final_audit": ref(root / "FINAL_AUDIT.json"),
        "algorithm_audit": ref(root / "ALGORITHM_AUDIT_LIVE_V2.json"),
        "adoption_decision": ref(root / "ADOPTION_DECISION_V1.json"),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Campaign terminal is a quality rejection with retained diagnostics, not a successful Robot conversion.",
    }
    write_new(root / "RESULT.json", result)

    receipt = {
        "schema_version": "0915-robot-recovery-v21-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": RUN_ID,
        "created_at": created_at,
        "status": "REJECTED_QUALITY",
        "result": ref(root / "RESULT.json"),
        "final_audit": ref(root / "FINAL_AUDIT.json"),
        "verification": ref(verification_report),
    }
    write_new(root / "RUN_RECEIPT.json", receipt)
    write_new(args.terminal_receipt.resolve(), {**receipt, "run_receipt": ref(root / "RUN_RECEIPT.json")})
    print(json.dumps({"status": result["status"], "result": ref(root / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
