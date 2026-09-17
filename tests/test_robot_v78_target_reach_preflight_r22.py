from chaoyang.ops.finalize_robot_v78_target_reach_preflight_r22 import evaluate


def test_missing_frozen_poker_and_framewise_fields_block_preflight() -> None:
    selection = {
        "v78_failure_canary": "play_cards_0903_189",
        "v78_chips_regression": "get_potato_chips_0902_025",
        "v78_poker_regression": None,
    }
    terminal = {
        "rows": [
            {"session": "play_cards_0903_189", "task": "poker", "terminal_status": "FAILED_QUALITY_C"},
            {"session": "get_potato_chips_0902_025", "task": "chips", "terminal_status": "PASSED"},
        ]
    }
    robot_yield = {
        "missing_profile_fields": [
            "action_amplitude",
            "wrist_object_approach",
            "ik_frame_valid_rate",
        ]
    }
    selected, blockers = evaluate(selection, terminal, robot_yield)
    assert selected["poker_hard_pass_regression"] is None
    assert {row["code"] for row in blockers} == {
        "FROZEN_SELECTION_MISSING",
        "FRAMEWISE_REACH_FIELD_MISSING",
    }


def test_ready_selection_has_no_blocker() -> None:
    selection = {
        "v78_failure_canary": "poker_c",
        "v78_chips_regression": "chips_pass",
        "v78_poker_regression": "poker_pass",
    }
    terminal = {
        "rows": [
            {"session": "poker_c", "task": "poker", "terminal_status": "FAILED_QUALITY_C"},
            {"session": "chips_pass", "task": "chips", "terminal_status": "PASSED"},
            {"session": "poker_pass", "task": "poker", "terminal_status": "PASSED"},
        ]
    }
    _, blockers = evaluate(selection, terminal, {"missing_profile_fields": []})
    assert blockers == []
