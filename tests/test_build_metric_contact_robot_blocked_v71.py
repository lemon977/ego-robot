import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metric_contact_does_not_bypass_robot_geometry() -> None:
    result = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/metric_contact_robot/RESULT.json").read_text())
    assert result["status"] == "BLOCKED_PREREQ"
    assert result["metric_contact_robot_count"] == 0
    assert result["authority"] is False
    assert result["control_ground_truth"] is False
