from __future__ import annotations

import inspect

from chaoyang.governance import current_r3_contracts


def test_0915_campaign_is_sam31_only() -> None:
    source = inspect.getsource(current_r3_contracts.build_algorithm_contract)
    assert '"current": "SAM3.1"' in source
    assert '"challengers": []' in source
    assert '"selection_policy": "SAM3.1_ONLY_USER_LOCKED"' in source
    assert '"mask": "SAM3.1_ONLY"' in source


def test_0915_campaign_does_not_register_challenger_operations() -> None:
    source = inspect.getsource(current_r3_contracts.build_algorithm_contract)
    campaign = source.split('"processed_0915_full_funnel_v1"', 1)[1].split(
        '"handle_data_cleaning_v4_0916"', 1
    )[0]
    assert "sam2" not in campaign.lower()
    assert "cutie" not in campaign.lower()

