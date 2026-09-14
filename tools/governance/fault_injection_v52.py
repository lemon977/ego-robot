from __future__ import annotations

"""Run bounded fault injection for exact78 V5.2 governance contracts."""

import json
import os
import tempfile
import argparse
from pathlib import Path

from tools.governance.v52_contracts import (
    artifact_ref,
    build_run_signature,
    claim_executor,
    commit_final,
    process_identity,
    validate_task_packet,
)
from tools.governance.common import atomic_json


def expect_failure(name: str, function) -> dict:
    try:
        function()
    except Exception as error:  # the precise exception is part of the receipt
        return {"name": name, "status": "PASS", "observed_failure": f"{type(error).__name__}: {error}"}
    return {"name": name, "status": "FAIL", "observed_failure": None}


def packet(task_id: str = "fixture") -> dict:
    return {
        "schema_version": "exact78-task-packet-v1", "task_id": task_id,
        "objective": "fault fixture", "read_set": [], "write_set": [],
        "prerequisites": [], "gates": [],
        "budgets": {"cpu_seconds": 1, "gpu_seconds": 0, "wall_seconds": 1},
        "attempt_max": 1, "stop_condition": "fixture", "output_contract": ["FINAL.json"],
        "claim_limit": "fixture only", "executor_epoch": 1,
        "fencing": {"pid_startticks_required": True, "immutable_final": True},
        "initial_search_result_limit": 20, "initial_log_line_limit": 80,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="exact78_v52_faults_") as name:
        root = Path(name)
        packet_path = root / "task" / "TASK_PACKET.json"
        packet_path.parent.mkdir()
        packet_path.write_text(json.dumps(packet()), encoding="utf-8")
        errors = validate_task_packet(packet())
        checks.append({"name": "valid_packet", "status": "PASS" if not errors else "FAIL", "errors": errors})
        oversized = packet()
        oversized["read_set"] = [str(i) for i in range(9)]
        checks.append({
            "name": "bounded_initial_read_set",
            "status": "PASS" if "initial read_set exceeds 8 files" in validate_task_packet(oversized) else "FAIL",
        })

        components = {}
        for component in ("input_manifest", "code_closure", "config", "weights", "schema"):
            path = root / f"{component}.json"
            path.write_text(json.dumps({"component": component}), encoding="utf-8")
            components[component] = artifact_ref(path)
        components["calibration_or_absent"] = "ABSENT"
        signature_a = build_run_signature(components)
        signature_b = build_run_signature(components)
        checks.append({
            "name": "canonical_signature_deterministic",
            "status": "PASS" if signature_a == signature_b else "FAIL",
        })

        claim = claim_executor(packet_path, "fault-injection", os.getpid(), signature_a["run_signature_sha256"])
        checks.append(expect_failure(
            "duplicate_executor_rejected",
            lambda: claim_executor(packet_path, "second-executor", os.getpid(), signature_a["run_signature_sha256"]),
        ))
        candidate = root / "candidate.json"
        candidate.write_text(json.dumps({"status": "PASSED", "fencing_token": claim["fencing_token"]}), encoding="utf-8")
        final = root / "FINAL.json"
        commit_final(candidate, final, packet_path.parent / "EXECUTOR_CLAIM.json")
        different = root / "different.json"
        different.write_text(json.dumps({"status": "FAILED_QUALITY_C", "fencing_token": claim["fencing_token"]}), encoding="utf-8")
        checks.append(expect_failure(
            "different_final_overwrite_rejected",
            lambda: commit_final(different, final, packet_path.parent / "EXECUTOR_CLAIM.json"),
        ))

        stale_claim = dict(claim)
        stale_claim["proc_start_ticks"] = int(process_identity(os.getpid())["start_ticks"]) + 1
        stale_claim_path = root / "STALE_CLAIM.json"
        stale_claim_path.write_text(json.dumps(stale_claim), encoding="utf-8")
        stale_final = root / "STALE_FINAL.json"
        checks.append(expect_failure(
            "pid_startticks_fence_rejected",
            lambda: commit_final(candidate, stale_final, stale_claim_path),
        ))

        missing_component = dict(components)
        del missing_component["weights"]
        checks.append(expect_failure(
            "incomplete_signature_rejected",
            lambda: build_run_signature(missing_component),
        ))

    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    result = {
        "schema_version": "exact78-v52-fault-injection-result-v1",
        "status": status,
        "checks": checks,
        "claim_limit": "Governance contract fault injection only; no pipeline authority.",
    }
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
