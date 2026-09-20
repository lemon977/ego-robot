from __future__ import annotations

import numpy as np

from chaoyang.ops.render_huro_hand_only_common_review_v1 import _local_field_names


def test_local_fields_support_current_and_legacy_results() -> None:
    current = {
        "comparison": {
            "local_q22_field": "q22_frozen_postclip",
            "local_valid_field": "q22_valid_physical",
        }
    }
    arrays = {
        "q22_frozen_postclip": np.zeros((2, 2, 22)),
        "q22_valid_physical": np.ones((2, 2), dtype=bool),
    }
    assert _local_field_names(current, arrays) == (
        "q22_frozen_postclip",
        "q22_valid_physical",
    )
    legacy = {"comparison": {}}
    legacy_arrays = {
        "q22": np.zeros((2, 2, 22)),
        "q22_computed": np.ones((2, 2), dtype=bool),
    }
    assert _local_field_names(legacy, legacy_arrays) == ("q22", "q22_computed")
