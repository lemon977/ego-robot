import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sensor_terminal_matrix_joins_h0_h4_without_false_gpu_authority() -> None:
    value = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/SENSOR_PIPELINE_TERMINAL_MATRIX.json").read_text())
    assert len(value["rows"]) == 332
    assert len({row["session_id"] for row in value["rows"]}) == 332
    assert value["counts"]["h1_passed"] == 331
    assert value["counts"]["h2_passed"] == 331
    assert value["counts"]["h3_cpu_preflight_passed"] == 331
    assert value["counts"]["h4_cpu_preflight_passed"] == 331
    assert all(row["object6d"] == "BLOCKED_PREREQ" for row in value["rows"])
    assert all(row["robot_geometry"] == "BLOCKED_PREREQ" for row in value["rows"])
