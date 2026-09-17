#!/usr/bin/env python3
"""Close R2 metric-contact evaluation when no V7.1 R1 pass exists."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/ROBOT_TERMINAL_MATRIX.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/metric_contact_robot"


def main() -> None:
    data = MATRIX.read_bytes()
    matrix = json.loads(data)
    passed_r1 = matrix["counts"]["METRIC_CONTACT_ROBOT"] + matrix["counts"]["POSE_ONLY_VISUAL_ROBOT"]
    if passed_r1:
        raise RuntimeError("R1 passes exist; a real R2 evaluation is required")
    value = {
        "schema_version": "METRIC_CONTACT_ROBOT_V71_BLOCKED_RECEIPT",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "task_id": "metric_contact_robot_v1",
        "status": "BLOCKED_PREREQ",
        "blocker": "NO_V71_ROBOT_GEOMETRY_PASS",
        "robot_geometry_matrix": {"path": str(MATRIX), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        "metric_contact_robot_count": 0,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "resume_condition": "At least one R1 session passes V7.1 unified z-buffer, self-collision, joint, chirality, reachability and 30-second total-frame gates.",
        "claim_limit": "No metric-contact Robot result, physical contact accuracy, control truth, or deployment authority.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (OUT / "RESULT.json").write_text(content, encoding="utf-8")
    print(json.dumps({"status": value["status"], "metric_contact_robot_count": 0}))


if __name__ == "__main__":
    main()
