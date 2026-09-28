import numpy as np

from chaoyang.ops.run_human_to_robot_007_spatial_donor_check import PAIRS, hull_mask, spatial_support


def test_frozen_pairs_are_from_prior_search():
    assert PAIRS[192] == (115, 135, 170, 175, 281, 282, 336, 363)
    assert PAIRS[196] == (48, 51, 197, 377)


def test_target_and_source_hulls_both_required():
    matrix = np.eye(3, dtype=np.float64)
    source = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float32)
    target = np.array([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=np.float32)
    table = np.ones((9, 9), dtype=np.uint8) * 255
    eligible = np.ones((9, 9), dtype=bool)
    support, target_hull, source_hull = spatial_support(matrix, source, target, table, eligible)
    assert source_hull[2, 2] and not target_hull[2, 2] and not support[2, 2]
    assert target_hull[6, 6] and not source_hull[6, 6] and not support[6, 6]
    assert support[4, 4]


def test_no_three_point_area_is_not_spatial_support():
    collinear = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    assert not hull_mask(collinear, (6, 6)).any()


def test_spatial_status_takes_priority_over_older_sensor(monkeypatch):
    from chaoyang.governance import build_human_to_robot_r2_terminal_status as status

    written = []
    monkeypatch.setattr(status, "build_spatial_donor_status", lambda state, time: {"latest_task": "spatial"})
    monkeypatch.setattr(status, "atomic_json", lambda path, value: written.append((path, value)))
    ledger = {"tasks": [{"task_id": status.SENSOR_DISPLAY_TASK, "status": "PASSED"},
                        {"task_id": status.SPATIAL_DONOR_TASK, "status": "PENDING"}]}
    assert status.publish_navigation_if_human_to_robot_r2(status.REPO_ROOT, ledger, "now")
    assert written == [(status.OUTPUT, {"latest_task": "spatial"})]
