from chaoyang.ops.build_visual_aux_robot_expansion_plan_v71 import build


def _fixtures():
    rows = []
    matrix = []
    position = 0
    for task in ("CHIPS", "POKER"):
        for split, metric_count, visual_count in (
            ("train", 20 if task == "CHIPS" else 10, 0 if task == "CHIPS" else 12),
            ("validation", 4 if task == "CHIPS" else 1, 0 if task == "CHIPS" else 3),
        ):
            for tier, count in (("METRIC_GEOMETRY_READY", metric_count), ("CALIBRATION_MISSING", visual_count)):
                for index in range(count):
                    position += 1
                    session = f"{task.lower()}_{split}_{tier}_{index:02d}"
                    rows.append({
                        "task": task, "split": split, "session_id": session,
                        "first_blocker": tier, "frame_count": 100 + index,
                    })
                    matrix.append({
                        "session_id": session,
                        "clean_state": "PASSED_GRADE_B" if tier == "METRIC_GEOMETRY_READY" else "NOT_ELIGIBLE_WAVE0",
                        "robot_current_state": "READY_FOR_ROBOT_CURRENT_DRAFT",
                    })
    while len(rows) < 156:
        position += 1
        session = f"blocked_{position:03d}"
        rows.append({"task": "CHIPS", "split": "test", "session_id": session, "first_blocker": "HAWOR_C", "frame_count": 50})
        matrix.append({"session_id": session, "clean_state": "NOT_ELIGIBLE_WAVE0", "robot_current_state": "BLOCKED_PREREQ_CURRENT_DRAFT"})
    return {"rows": rows}, {"rows": matrix}, {"rows": []}


def test_prefers_metric_and_adds_split_preserving_visual_tier_reserve():
    ledger, matrix, eligibility = _fixtures()
    result = build(ledger, matrix, eligibility)
    assert result["status"] == "PASS_DETERMINISTIC_SELECTION"
    assert result["summaries"]["chips"]["train"]["selected_metric_sessions"] == 16
    assert result["summaries"]["poker"]["train"]["selected_metric_sessions"] == 10
    assert result["summaries"]["poker"]["train"]["selected_visual_tier_sessions"] == 10
    assert result["summaries"]["poker"]["validation"]["selected_visual_tier_sessions"] == 3
    assert result["summaries"]["poker"]["train"]["visual_tier_reserve_sessions"] == 4
    assert result["summaries"]["poker"]["validation"]["visual_tier_reserve_sessions"] == 1
    assert result["counts"] == {"sessions": 43, "metric": 30, "visual_tier": 13, "ready_robot_now": 30, "needs_clean": 13}


def test_existing_ready_reduces_need_and_attempted_zero_window_is_not_reused():
    ledger, matrix, _ = _fixtures()
    first = next(row for row in ledger["rows"] if row["task"] == "CHIPS" and row["split"] == "train")
    second = next(row for row in ledger["rows"] if row["task"] == "CHIPS" and row["split"] == "train" and row is not first)
    eligibility = {"rows": [
        {"task": "chips", "split": "train", "session": first["session_id"], "status": "READY"},
        {"task": "chips", "split": "train", "session": second["session_id"], "status": "BLOCKED_PREREQ_ZERO_H50_WINDOWS"},
    ]}
    result = build(ledger, matrix, eligibility)
    chips = [row for row in result["selection"] if row["task"] == "chips" and row["split"] == "train"]
    assert len(chips) == 15
    assert first["session_id"] not in {row["session_id"] for row in chips}
    assert second["session_id"] not in {row["session_id"] for row in chips}
    assert result["summaries"]["chips"]["train"]["existing_ready_sessions"] == 1


def test_reports_deficit_without_moving_split():
    ledger, matrix, eligibility = _fixtures()
    ledger["rows"] = [row for row in ledger["rows"] if not (row["task"] == "POKER" and row["split"] == "validation" and row["first_blocker"] == "CALIBRATION_MISSING")]
    removed = 3
    for i in range(removed):
        session = f"replacement_{i}"
        ledger["rows"].append({"task": "POKER", "split": "test", "session_id": session, "first_blocker": "HAWOR_C", "frame_count": 1})
        matrix["rows"] = [row for row in matrix["rows"] if row["session_id"] != f"poker_validation_CALIBRATION_MISSING_{i:02d}"]
        matrix["rows"].append({"session_id": session, "clean_state": "NOT_ELIGIBLE_WAVE0", "robot_current_state": "BLOCKED_PREREQ_CURRENT_DRAFT"})
    result = build(ledger, matrix, eligibility)
    assert result["status"] == "BLOCKED_DATA_VOLUME"
    assert result["summaries"]["poker"]["validation"]["remaining_session_deficit"] == 2
    assert result["constraints"]["split_movement_allowed"] is False
