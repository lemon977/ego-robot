import pytest

from chaoyang.ops.prepare_robot_arm_only_research_render_v1 import build_compatibility


def test_quality_c_rows_become_explicit_failed_frame_watermarks():
    source = {
        "status": "FAILED_QUALITY_C",
        "session_id": "play_cards_0901_008",
        "frame_count": 2,
        "metrics": {"target_pass_rows": 3, "target_total_rows": 4},
        "pose_rows": [
            {"frame": 0, "target_pose_gate": True},
            {"frame": 0, "target_pose_gate": False},
            {"frame": 1, "target_pose_gate": True},
            {"frame": 1, "target_pose_gate": True},
        ],
    }
    result = build_compatibility(source, {"sha256": "frozen"})
    assert result["status"] == "HOLD_NUMERIC_CANARY"
    assert result["research_source_status"] == "FAILED_QUALITY_C"
    assert result["failed_target_frame_ids"] == [0]
    assert result["training_eligible"] is False
    assert result["authority"] is False


@pytest.mark.parametrize("status,rows", [("PASSED", 4), ("FAILED_QUALITY_C", 3)])
def test_refuse_non_c_or_incomplete_rows(status, rows):
    source = {
        "status": status,
        "session_id": "play_cards_0901_008",
        "frame_count": 2,
        "metrics": {},
        "pose_rows": [{"frame": 0, "target_pose_gate": False}] * rows,
    }
    with pytest.raises(ValueError):
        build_compatibility(source, {})
