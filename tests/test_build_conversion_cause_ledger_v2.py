from chaoyang.ops.build_conversion_cause_ledger_v2 import build


def test_first_blocker_partition_is_exact_and_mutually_exclusive() -> None:
    value = build()
    assert len(value["rows"]) == 156
    assert len({row["session_id"] for row in value["rows"]}) == 156
    assert value["counts"]["first_blocker"] == {
        "HAWOR_C": 12,
        "ROLE_MASK_C": 20,
        "OBJECT_MASK_C": 23,
        "CALIBRATION_MISSING": 43,
        "METRIC_GEOMETRY_READY": 58,
    }


def test_each_first_blocker_has_one_downstream_route_and_class() -> None:
    value = build()
    for row in value["rows"]:
        assert row["successor_route"]
        assert row["blocker_class"] in {"ALGORITHM_QUALITY", "INFRASTRUCTURE_FAILURE", "MISSING_EVIDENCE", "NONE"}
        if row["first_blocker"] == "CALIBRATION_MISSING":
            assert row["downstream_tier"] == "TIER_V_VISUAL"
        if row["first_blocker"] == "METRIC_GEOMETRY_READY":
            assert row["downstream_tier"] == "TIER_M_METRIC"


def test_poker042_is_infrastructure_not_object_mask_failure() -> None:
    value = build()
    row = next(item for item in value["rows"] if item["session_id"] == "play_cards_0901_042")
    assert row["first_blocker"] == "ROLE_MASK_C"
    assert row["blocker_class"] == "INFRASTRUCTURE_FAILURE"
    assert "FRAME_FPS_MISMATCH" in row["reason"]
