#!/usr/bin/env python3
"""Publish post-terminal validation without changing terminal quality claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_baseline_v1_convergence_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
VALIDATION = ROOT / "FINAL_VALIDATION.json"
PLAN = REPO_ROOT / "docs/current/PLAN.md"
VIS = REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE/INDEX_ZH.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    if VALIDATION.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{VALIDATION}")
    result = load_json(ROOT / "RESULT.json")
    if result.get("task_terminal_status") != "REJECTED_QUALITY":
        raise RuntimeError("TERMINAL_RESULT_NOT_REJECTED_QUALITY")
    validation = {
        "schema_version": "HUMAN_TO_ROBOT_CONVERGENCE_FINAL_VALIDATION_V1", "task_id": TASK,
        "created_at": now_iso(), "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
        "targeted_regression": {"passed": 33, "failed": 0,
            "scope": ["convergence evidence", "R2 Scene contract", "S2 checkpoints", "S2 terminalizer"]},
        "collision_and_current_state_regression": {"passed": 14, "failed": 0,
            "environment": {"TMPDIR": "PROJECT_LOCAL_CPFS", "CUDA_VISIBLE_DEVICES": "", "CHAOYANG_REPO_ROOT": str(REPO_ROOT)}},
        "full_suite_observation": {"passed": 1513, "skipped": 1, "failed": 14,
            "cpfs_hardlink_identity_drift_failures": 13,
            "environment_fixture_failure_resolved_by_targeted_rerun": 1,
            "status": "NOT_EVALUATED_ENV_FOR_HARDLINK_TRANSACTION_GROUP",
            "claim_limit": "CPFS link-count/identity semantics remain incompatible with the transaction tests; they are not counted as pass and do not invalidate unrelated algorithm evidence."},
        "governance_validation_before_post_validation_publish": "PASS_REVISION_13966",
        "delivery": artifact_ref(ROOT / "delivery/DELIVERY_MANIFEST.json"),
        "terminal_result": artifact_ref(ROOT / "RESULT.json"),
        "quality_unchanged": {"products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4"},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    atomic_json(VALIDATION, validation)
    PLAN.write_text(PLAN.read_text(encoding="utf-8") +
                    "\n验证：变更相关33项通过；碰撞/当前状态14项在项目内受控环境通过。全量回归中13项hardlink事务测试受CPFS语义限制，保持NOT_EVALUATED_ENV；不计通过。\n",
                    encoding="utf-8")
    VIS.write_text(VIS.read_text(encoding="utf-8") +
                   f"\n## 验证\n\n[最终验证收据]({VALIDATION})：33项相关回归通过；14项碰撞/当前状态回归通过；CPFS hardlink事务组保持环境未评估。\n",
                   encoding="utf-8")
    state = load_json(TASK_STATE_PATH)
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": TASK, "attempt": 1,
        "status": "REJECTED_QUALITY", "created_at": now_iso(),
        "message": "Post-terminal validation published; quality/adoption unchanged.",
        "validation": artifact_ref(VALIDATION)}])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_CONVERGENCE_POST_TERMINAL_VALIDATION",
        expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
                      "revision": published["governance_revision"], "validation": artifact_ref(VALIDATION)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
