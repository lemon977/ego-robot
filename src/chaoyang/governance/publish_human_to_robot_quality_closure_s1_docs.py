#!/usr/bin/env python3
"""Rebind governed current-document hashes after S1 terminal publication."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.build_human_to_robot_r2_terminal_status import SYNC_VALIDATION

TASK = "human_to_robot_quality_closure_s1_20260923"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--regression-junit", type=Path)
    parser.add_argument("--regression-log", type=Path)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None:
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    row = next(item for item in state["tasks"] if item.get("task_id") == TASK)
    if row.get("status") != "REJECTED_QUALITY":
        raise RuntimeError("S1_NOT_TERMINAL_REJECTED_QUALITY")
    required = {
        "docs/current/PLAN.md": "Quality Closure S1 已终态",
        "docs/current/README_ZH.md": "S1 最终结果",
        "docs/current/AI_WORK_ENTRY_ZH.md": "Quality Closure S1 最终交接",
        "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md": "S1 质量闭环终态",
    }
    for relative, marker in required.items():
        if marker not in (REPO_ROOT / relative).read_text(encoding="utf-8"):
            raise RuntimeError(f"CURRENT_DOC_MARKER_MISSING:{relative}")
    if bool(args.regression_junit) != bool(args.regression_log):
        raise RuntimeError("JUNIT_AND_LOG_REQUIRED_TOGETHER")
    if args.regression_junit:
        if SYNC_VALIDATION.exists():
            raise RuntimeError("IMMUTABLE_SYNC_REGRESSION_ALREADY_EXISTS")
        tree = ET.parse(args.regression_junit)
        cases = list(tree.iter("testcase"))
        failures = sum(case.find("failure") is not None for case in cases)
        errors = sum(case.find("error") is not None for case in cases)
        skipped = sum(case.find("skipped") is not None for case in cases)
        if not cases or failures or errors:
            raise RuntimeError("REGRESSION_NOT_PASS")
        atomic_json(SYNC_VALIDATION, {
            "schema_version": "HUMAN_TO_ROBOT_STATUS_SYNC_REGRESSION_V1",
            "algorithm_task_id": TASK, "created_at": now_iso(),
            "scope": "status publisher fix and full CPU regression; no algorithm inference",
            "junit": artifact_ref(args.regression_junit.resolve()),
            "log": artifact_ref(args.regression_log.resolve()),
            "pytest": {"passed": len(cases)-skipped, "skipped": skipped,
                       "failures": failures, "errors": errors, "total": len(cases)},
            "producer": artifact_ref(Path(__file__)),
        })
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_TERMINAL_DOC_REBIND",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    status = load_json(REPO_ROOT / "docs/current/STATUS.json")
    if status.get("latest_task") != TASK:
        raise RuntimeError("SHALLOW_STATUS_REGRESSED_TO_OLD_TASK")
    if status.get("governance_revision") != published["governance_revision"]:
        raise RuntimeError("SHALLOW_STATUS_REVISION_MISMATCH")
    print(json.dumps({"status": "PASS", "governance_revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
