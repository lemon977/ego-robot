from chaoyang.ops.audit_robot_hand_round2_value_v71 import compare


def row(session, failed, tip, bone, start, finish):
    return {
        "session": session,
        "task": "chips",
        "started_at": start,
        "finished_at": finish,
        "metrics": {
            "failed_rows": failed,
            "tip_direction_error_deg_max": tip,
            "bone_error_deg_max": bone,
        },
    }


def test_identical_round_summary_is_not_improvement():
    first = row("s", 10, 18.0, 40.0, "2026-09-15T00:00:00+08:00", "2026-09-15T00:03:00+08:00")
    second = row("s", 10, 18.0, 40.0, "2026-09-15T00:03:00+08:00", "2026-09-15T00:03:40+08:00")
    result = compare(first, second)
    assert result["summary_improved"] is False
    assert result["round2_seconds"] == 40


def test_failed_row_reduction_counts_as_improvement():
    first = row("s", 10, 18.0, 40.0, "2026-09-15T00:00:00+08:00", "2026-09-15T00:03:00+08:00")
    second = row("s", 9, 18.0, 40.0, "2026-09-15T00:03:00+08:00", "2026-09-15T00:03:40+08:00")
    assert compare(first, second)["summary_improved"] is True
