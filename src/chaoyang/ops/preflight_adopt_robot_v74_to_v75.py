#!/usr/bin/env python3
"""Read-only SHA closure check for phase-complete v7.4 Robot batches."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    RobotTargetContractError,
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    status_specific_artifacts,
    verify_ref,
)


def unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or not session or session in output:
            raise RobotTargetContractError(f"{label}: non-empty unique session required")
        output[session] = row
    return output


def check_batch(batch: Path) -> list[dict[str, Any]]:
    paths = {
        "preflight": batch / "preflight/RESULT.json",
        "temporal": batch / "hawor_temporal/RESULT.json",
        "arm": batch / "arm_round2/RESULT.json",
        "hand": batch / "hand_round2/RESULT.json",
        "render": batch / "render/RESULT.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise RobotTargetContractError(f"{batch.name}: missing {label} result")
    values = {label: load_json(path) for label, path in paths.items()}
    expected = unique(values["preflight"].get("sessions", []), f"{batch.name}.preflight")
    indexed = {
        label: unique(values[label].get("sessions", []), f"{batch.name}.{label}")
        for label in ("temporal", "arm", "hand", "render")
    }
    if any(set(rows) != set(expected) for rows in indexed.values()):
        raise RobotTargetContractError(f"{batch.name}: phase session sets differ")
    output = []
    for session, source in expected.items():
        arm_result, arm_states = status_specific_artifacts(indexed["arm"][session], kind="arm", session=session)
        hand_result, hand_states = status_specific_artifacts(indexed["hand"][session], kind="hand", session=session)
        refs = {
            "preflight_result": artifact_ref(paths["preflight"]),
            "temporal_result": indexed["temporal"][session].get("result"),
            "arm_result": arm_result,
            "arm_states": arm_states,
            "hand_result": hand_result,
            "hand_states": hand_states,
            "review_result": indexed["render"][session].get("result"),
            "review_video": indexed["render"][session].get("video"),
        }
        verified = {name: artifact_ref(verify_ref(ref, f"{session}.{name}")) for name, ref in refs.items()}
        signature_payload = {
            "task": source.get("task"),
            "session": session,
            "source_artifact_revision": "R7_ROBOT_4",
            "target_artifact_revision": "R7_ROBOT_5",
            "artifacts": verified,
        }
        phase_closure_signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output.append(
            {
                **signature_payload,
                "status": "ADOPTABLE_PHASE_COMPLETE",
                "phase_closure_signature": phase_closure_signature,
                "authority": False,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v74-root", type=Path, required=True)
    parser.add_argument("--governance-status-min", type=Path, required=True)
    parser.add_argument("--supersedes-attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.v74_root.resolve(strict=True)
    selection = load_json(root / "SELECTION.json")
    frozen_sessions = set(selection.get("sessions", []))
    rows = []
    failures = []
    for batch in sorted(path for path in root.glob("batch_*" ) if path.is_dir()):
        try:
            rows.extend(check_batch(batch))
        except (OSError, KeyError, ValueError, RobotTargetContractError) as error:
            failures.append({"batch": batch.name, "error": str(error)})
    status_min_path = args.governance_status_min.resolve(strict=True)
    load_json(status_min_path)
    validation = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        validation_payload = json.loads(validation.stdout)
    except json.JSONDecodeError:
        validation_payload = {
            "status": "VALIDATOR_OUTPUT_INVALID",
            "stdout_tail": validation.stdout[-2000:],
            "stderr_tail": validation.stderr[-2000:],
        }
    governance_fresh = validation.returncode == 0 and validation_payload.get("status") == "PASS"
    unique_sessions = {row["session"] for row in rows}
    bundle_signature = hashlib.sha256(
        json.dumps(
            [{"session": row["session"], "signature": row["phase_closure_signature"]} for row in rows],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "robot-v74-to-v75-adopt-preflight-v1",
        "artifact_revision": "R7_ROBOT_5_PREFLIGHT",
        "created_at": now_iso(),
        "task_id": "RECOVER-V74",
        "terminal_status": "BLOCKED_PREREQ" if not governance_fresh else "PASSED",
        "robot_tier": "NONE",
        "adopt_preflight_status": "PASS_EXACT_SHA_ADOPTABLE" if rows and not failures else "FAILED_REFERENCE_PROOF",
        "governance_gate": "PASS_FRESH" if governance_fresh else "BLOCKED_STATUS_CONFLICT_OR_STALE",
        "counts": {
            "frozen_selection": len(frozen_sessions),
            "phase_complete_adoptable": len(unique_sessions),
            "not_yet_phase_complete": len(frozen_sessions - unique_sessions),
            "batch_reference_failures": len(failures),
        },
        "adopt_bundle_signature": bundle_signature,
        "selection": artifact_ref(root / "SELECTION.json"),
        "governance_status_min": artifact_ref(status_min_path),
        "governance_validation": validation_payload,
        "rows": rows,
        "failures": failures,
        "code_closure": [
            artifact_ref(Path(__file__)),
            artifact_ref(ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py"),
            artifact_ref(ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py"),
            artifact_ref(ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py"),
        ],
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Read-only SHA adopt eligibility only; no v7.4 artifact is modified and no Robot authority is granted.",
    }
    if args.supersedes_attempt:
        payload["supersedes_attempt"] = artifact_ref(args.supersedes_attempt)
    atomic_new_json(args.output.resolve(), payload)
    return 0 if payload["terminal_status"] == "PASSED" and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
