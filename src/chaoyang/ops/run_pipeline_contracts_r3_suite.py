#!/usr/bin/env python3
"""运行 R3 合同 CPU 验收并发布一个不可变开发 attempt。

本工具只写 ``--attempt-root``，拒绝覆盖。它不会更新 current governance，且在
治理冲突时只允许发布 ``DEVELOPMENT_EVIDENCE``。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import re
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
MODES = ("depth-00", "depth-10", "depth-20", "mask-role", "mask-object", "atlas", "donor", "clean", "contact-10", "occlusion-silver")
SCHEMAS = (
    "depth_source_audit_r3.schema.json", "depth_quality_evidence_r3.schema.json", "wrist_fusion_r3.schema.json",
    "mask_role_object_r3.schema.json", "clean_write_domain_r3.schema.json", "causal_donor_atlas_r3.schema.json",
    "contact_hypothesis_r3.schema.json", "occlusion_silver_r3.schema.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt_root.resolve()
    if attempt.exists():
        raise SystemExit(f"refusing to overwrite immutable attempt: {attempt}")
    attempt.mkdir(parents=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    governance = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=PROJECT, capture_output=True, text=True, check=False,
    )
    (attempt / "GOVERNANCE_VALIDATION.log").write_text(governance.stdout + governance.stderr)
    try:
        governance_record = json.loads(governance.stdout[governance.stdout.find("{"):])
    except (ValueError, json.JSONDecodeError):
        governance_record = {"status": "UNKNOWN"}

    dryrun_dir = attempt / "dry_runs"; dryrun_dir.mkdir()
    dryrun_refs = []
    for mode in MODES:
        output = dryrun_dir / f"{mode}.json"
        completed = subprocess.run(
            [sys.executable, str(PROJECT / "src/chaoyang/ops/validate_pipeline_contracts_r3.py"), "--mode", mode, "--dry-run", "--output", str(output)],
            cwd=PROJECT, capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            (attempt / f"{mode}.stderr.log").write_text(completed.stderr)
            raise SystemExit(f"dry-run failed: {mode}")
        dryrun_refs.append(ref(output))

    pytest_log = attempt / "PYTEST.log"
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_pipeline_contracts_r3.py"],
        cwd=PROJECT, capture_output=True, text=True, check=False,
    )
    pytest_log.write_text(pytest_run.stdout + pytest_run.stderr)
    if pytest_run.returncode != 0:
        raise SystemExit("R3 contract pytest failed")

    match = re.search(r"(\d+) passed", pytest_run.stdout)
    pytest_passed = int(match.group(1)) if match else None
    governance_passed = governance.returncode == 0 and governance_record.get("status") == "PASS"
    metrics = {
        "schema_version": "PIPELINE_CONTRACTS_R3_METRICS", "schema_count": len(SCHEMAS),
        "task_packet_count": 10, "dry_run_modes_passed": len(dryrun_refs), "pytest_passed": pytest_passed,
        "depth_confidence_present": False, "external_metric_accuracy": "UNKNOWN",
        "formal_h4": {"execution_status": "BLOCKED_RESOURCE", "qa_status": "NOT_EVALUATED", "policy_status": "POLICY_DEFERRED", "pixel_mask_authority": False},
    }
    write_json(attempt / "METRICS.json", metrics)
    write_json(attempt / "NEXT_ACTION.json", {
        "schema_version": "PIPELINE_CONTRACTS_R3_NEXT_ACTION",
        "status": "READY_FOR_PACKET_DISPATCH" if governance_passed else "BLOCKED_PREREQ",
        "next_action": (
            "Dispatch individual immutable Task Packets."
            if governance_passed
            else "Recover G0_CORE_GOVERNANCE, then dispatch individual immutable Task Packets."
        ),
        "blocker": None if governance_passed else governance_record.get("status", "UNKNOWN"),
    })
    (attempt / "DECISION.md").write_text(
        "# R3 合同开发验收决定\n\n"
        "合同、Schema、fixture、dry-run 和任务包已通过开发验收。"
        + ("当前治理已通过，可按独立任务包调度；" if governance_passed else "当前治理未通过，不启动模型推理；")
        + "本开发验收本身不晋升 authority；formal H4 状态保持 BLOCKED_RESOURCE / "
        "NOT_EVALUATED / POLICY_DEFERRED。\n"
    )
    artifact_paths = [
        attempt / "GOVERNANCE_VALIDATION.log", attempt / "PYTEST.log", attempt / "METRICS.json",
        attempt / "NEXT_ACTION.json", attempt / "DECISION.md",
        PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/TASK_PACKET_INDEX.json",
        PROJECT / "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py", PROJECT / "src/chaoyang/pipeline/contact_occlusion_contracts_r3.py",
        PROJECT / "src/chaoyang/ops/validate_pipeline_contracts_r3.py", PROJECT / "src/chaoyang/ops/materialize_pipeline_contract_task_packets_r3.py",
        PROJECT / "tests/test_pipeline_contracts_r3.py",
        *(PROJECT / "contracts" / name for name in SCHEMAS),
        *(Path(item["path"]) for item in dryrun_refs),
    ]
    manifest = {"schema_version": "PIPELINE_CONTRACTS_R3_ARTIFACT_MANIFEST", "artifacts": [ref(path) for path in artifact_paths]}
    write_json(attempt / "ARTIFACT_MANIFEST.json", manifest)
    result = {
        "schema_version": "PIPELINE_CONTRACTS_R3_RESULT", "task_id": "pipeline_contracts_r3",
        "status": "PASSED_DEVELOPMENT", "generated_at": generated_at, "authority_promoted": False,
        "claim_status": "DEVELOPMENT_EVIDENCE", "governance_at_run": governance_record.get("status", "UNKNOWN"),
        "formal_h4": metrics["formal_h4"], "depth_confidence_present": False, "external_metric_accuracy": "UNKNOWN",
        "artifact_manifest": ref(attempt / "ARTIFACT_MANIFEST.json"),
        "claim_limit": "CPU contracts and synthetic fixtures only; no model quality, Gold accuracy, physical precision, or current authority."
    }
    write_json(attempt / "RESULT.json", result)
    receipt = {
        "schema_version": "PIPELINE_CONTRACTS_R3_RUN_RECEIPT", "task_id": "pipeline_contracts_r3", "attempt_id": attempt.name,
        "generated_at": generated_at, "terminal_status": "PASSED_DEVELOPMENT", "authority_promoted": False,
        "result": ref(attempt / "RESULT.json"), "artifact_manifest": ref(attempt / "ARTIFACT_MANIFEST.json"),
        "metrics": ref(attempt / "METRICS.json"), "pytest": ref(pytest_log),
    }
    write_json(attempt / "RUN_RECEIPT.json", receipt)
    print(json.dumps({"status": "PASSED_DEVELOPMENT", "attempt": str(attempt), "result": ref(attempt / "RESULT.json"), "receipt": ref(attempt / "RUN_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
