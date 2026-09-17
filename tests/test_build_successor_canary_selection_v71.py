import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_successor_selection_has_one_failure_and_two_regressions_per_subcluster() -> None:
    value = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/SUCCESSOR_CANARY_SELECTION_V71.json").read_text())
    assert value["counts"] == {"groups": 5, "failure_canaries": 5, "regression_slots": 10}
    assert {(g["stage"], g["task"]) for g in value["groups"]} == {
        ("hawor", "chips"), ("hawor", "poker"),
        ("role_mask", "chips"), ("role_mask", "poker"),
        ("object_identity", "poker"),
    }
    assert all(len(g["regression_sessions"]) == 2 for g in value["groups"])
    assert all(g["canary_session"] not in g["regression_sessions"] for g in value["groups"])
    assert all(g["round_budget"]["max_rounds"] == 2 for g in value["groups"])
