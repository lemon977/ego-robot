from __future__ import annotations

from chaoyang.governance.heartbeat_task import heartbeat_guard_reason


def _state(status: str, *, next_task: str | None = "task_a") -> dict:
    return {
        "tasks": [{"task_id": "task_a", "status": status}],
        "next_task": None if next_task is None else {"task_id": next_task},
    }


def test_live_current_task_can_heartbeat() -> None:
    for status in ("PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"):
        assert heartbeat_guard_reason(_state(status), "task_a") is None


def test_terminal_task_cannot_be_resurrected_after_finalization() -> None:
    state = _state("PASSED", next_task=None)
    assert heartbeat_guard_reason(state, "task_a") == "TASK_NOT_LIVE"
    assert state["tasks"][0]["status"] == "PASSED"


def test_live_row_cannot_heartbeat_after_route_moves_to_successor() -> None:
    assert heartbeat_guard_reason(
        _state("RUNNING", next_task="task_b"), "task_a",
    ) == "TASK_NOT_CURRENT"


def test_unknown_and_duplicate_task_rows_fail_closed() -> None:
    assert heartbeat_guard_reason({"tasks": [], "next_task": None}, "task_a") == "UNKNOWN_TASK"
    duplicate = _state("RUNNING")
    duplicate["tasks"].append({"task_id": "task_a", "status": "RUNNING"})
    assert heartbeat_guard_reason(duplicate, "task_a") == "DUPLICATE_TASK_ROWS"
