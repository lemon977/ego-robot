from chaoyang.ops.build_visual_aux_capacity_preflight_v71 import summarize


def test_capacity_keeps_sources_separate_and_closes_both_pairs():
    matrix = {
        "rows": [
            {"session_id": f"c{i}", "task": "chips", "split": "train"}
            for i in range(15)
        ] + [
            {"session_id": f"cv{i}", "task": "chips", "split": "validation"}
            for i in range(3)
        ] + [
            {"session_id": f"p{i}", "task": "poker", "split": "train"}
            for i in range(10)
        ] + [
            {"session_id": "pv0", "task": "poker", "split": "validation"}
        ]
    }
    robot = {"sessions": [row["session_id"] for row in matrix["rows"]]}
    visual = {"sessions": [
        *({"session_id": f"pt{i}", "task": "poker", "split": "train"} for i in range(10)),
        *({"session_id": f"ptv{i}", "task": "poker", "split": "validation"} for i in range(3)),
    ]}
    legacy = {"rows": [
        {"session": "legacy_c", "task": "chips", "split": "train", "status": "READY", "eligible_h50_window_count": 300}
    ]}

    result = summarize(matrix, robot, visual, legacy)

    assert result["chips"]["train"]["unique_potential_sessions"] == 16
    assert result["poker"]["train"]["unique_potential_sessions"] == 20
    assert result["chips"]["capacity_status"].startswith("PASS_")
    assert result["poker"]["capacity_status"].startswith("PASS_")
    assert result["chips"]["train"]["actual_training_ready"] is False


def test_capacity_does_not_count_heldout_as_validation():
    matrix = {"rows": [
        {"session_id": f"c{i}", "task": "chips", "split": "train"} for i in range(16)
    ] + [
        {"session_id": f"cv{i}", "task": "chips", "split": "validation"} for i in range(3)
    ] + [
        {"session_id": f"p{i}", "task": "poker", "split": "train"} for i in range(16)
    ] + [
        {"session_id": f"ph{i}", "task": "poker", "split": "heldout"} for i in range(3)
    ]}
    robot = {"sessions": [row["session_id"] for row in matrix["rows"]]}
    result = summarize(matrix, robot, {"sessions": []}, {"rows": []})

    assert result["poker"]["validation"]["unique_potential_sessions"] == 0
    assert result["poker"]["capacity_status"] == "BLOCKED_DATA_VOLUME_CAPACITY"
