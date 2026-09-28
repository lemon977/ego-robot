from __future__ import annotations

import numpy as np

from chaoyang.ops.run_human_to_robot_product_first_cable import classify_support


def test_cable_reason_distinguishes_empty_roles_from_missing_complaint_coverage() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    cable = empty.copy()
    cable[4, 4] = True
    assert classify_support(empty, empty, cable, empty, empty)["reason"] == "EMPTY_HUMAN_DEVICE_ROLES"

    human = empty.copy()
    human[1, 1] = True
    assert classify_support(human, empty, cable, empty, empty)["reason"] == "ROLES_PRESENT_NO_COMPLAINT_COVERAGE"


def test_cable_reason_requires_actual_write_and_prepared_model_support() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    cable = empty.copy()
    cable[4, 4] = True
    assert classify_support(cable, empty, cable, empty, empty)["reason"] == "ROLE_COVERS_COMPLAINT_BUT_WRITE_OR_MODEL_MISSING"
    result = classify_support(cable, empty, cable, cable, cable)
    assert result["reason"] == "ROLE_AND_PREPARED_MODEL_SUPPORT_PRESENT"
    assert result["prepared_model_on_complaint_pixels"] == 1
