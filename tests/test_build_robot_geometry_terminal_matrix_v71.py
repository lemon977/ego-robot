import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robot_matrix_is_156_unique_and_does_not_promote_legacy_candidates() -> None:
    value = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/ROBOT_GEOMETRY_TERMINAL_MATRIX.json").read_text())
    assert len(value["rows"]) == 156
    assert len({r["session_id"] for r in value["rows"]}) == 156
    assert value["authority"] is False
    assert value["legacy_candidate_count"] == 8
    assert value["counts"]["FAILED_QUALITY_C"] == 16
    assert value["counts"]["BLOCKED_PREREQ"] == 140
    assert value["counts"]["METRIC_CONTACT_ROBOT"] == 0
    assert value["counts"]["POSE_ONLY_VISUAL_ROBOT"] == 0
    assert all(not row["control_ground_truth"] for row in value["rows"])
    assert all(not row["physical_deployment_authorized"] for row in value["rows"])
